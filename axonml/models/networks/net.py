from contextlib import nullcontext
from typing import Dict

from tqdm.auto import tqdm

import torch

from ..core import Population, make_intra
from ..parametric import to_param
from ..callbacks import CallbackList
from .delaydelivery import NetCon
from axonml.helpers import BACKEND, FULLGRAPH, DYNAMIC, JIT, COMPILE_MODE


def to_flat_idx_torch(arr, idx):
    """
    Converts any valid PyTorch index into a 1D tensor of flat indices.

    Args:
        arr (torch.Tensor): The original tensor, used for its shape and device.
        idx: The index to be converted. Can be a slice, int, tuple,
             boolean tensor, or integer tensor.

    Returns:
        torch.LongTensor: A 1D tensor containing the flat indices that
                          correspond to the elements selected by `arr[idx]`.
    """
    if not isinstance(arr, torch.Tensor):
        raise TypeError("Input 'arr' must be a torch.Tensor.")

    # 1. Create a grid of flat indices with the same shape as the input array.
    #    e.g., for a (2, 3) tensor, this becomes [[0, 1, 2], [3, 4, 5]]
    indices_grid = torch.arange(arr.numel(), device=arr.device).view(arr.shape)

    # 2. Apply the user's index to this grid. PyTorch's indexing logic
    #    will select the corresponding flat indices for us.
    selected_indices = indices_grid[idx]

    # 3. Flatten the result to get a 1D tensor of flat indices.
    return selected_indices.flatten()


def step_pop(integrator, model, dt, ve=None, intra=None):
    integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def step(populations, synapses, dt, ve: Dict[str, torch.Tensor | None]={}, intra: Dict[str, torch.Tensor | None]={}):
    for s in synapses.values():
        s.advance()
    for n, pop in populations.items():
        step_pop(pop.integrator, pop, dt, ve.get(n, None), intra.get(n, None))


def get_local_index(population, mech, index):
    indices = torch.full_like(population.v, -1, dtype=torch.long, device=population.device()).flatten()
    mech_key_flat = to_flat_idx_torch(population.v, mech.key)
    indices.index_copy_(0, mech_key_flat, torch.arange(mech_key_flat.numel(), device=population.device(), dtype=torch.long))
    return indices.index_select(0, index)


def prepare_indices_one_one(source, target, synapse):
    source_model = source.model
    target_model = target.model
    syn = synapse

    index_arr = torch.arange(
        target_model.v.numel(), device=target_model.device(), dtype=target_model.dtype()
    ).view_as(target_model.v)

    indices_in_synapse = syn.get(index_arr).flatten()
    post_idx = to_flat_idx_torch(target_model.v, target.index)

    if not torch.all(torch.isin(post_idx, indices_in_synapse)):
        raise ValueError(f"Target population '{target.name}' does not have the synapse '{synapse}' at all target locations.")
    
    pre_idx = to_flat_idx_torch(source_model.v, source.index)
    post_idx = get_local_index(target_model, syn, post_idx)

    return pre_idx, post_idx


def prepare_indices_one_one_flat(source_model, source_index, target_model, target_index, synapse):
    """
    Prepares indices for a one-to-one connection between source and target populations.
    This function assumes that the synapse exists at all target locations.
    """
    syn = synapse

    index_arr = torch.arange(
        target_model.v.numel(), device=target_model.device(), dtype=target_model.dtype()
    ).view_as(target_model.v)

    indices_in_synapse = syn.get(index_arr).flatten()

    pre_idx = source_index
    post_idx = target_index

    if not torch.all(torch.isin(post_idx, indices_in_synapse)):
        raise ValueError(f"Target population '{target_model.name}' does not have the synapse '{synapse}' at all target locations.")

    post_idx = get_local_index(target_model, syn, post_idx)

    return pre_idx, post_idx


def check_weight_shape(weight, pre_idx):
    """
    Checks the shape of the weight tensor against the pre-synaptic indices.
    If the weight is a scalar, it returns the number of pre-synaptic indices.
    If the weight is a tensor, it checks if its shape matches the number of pre-synaptic indices.
    """
    if isinstance(weight, float):
        return len(pre_idx)
    if isinstance(weight, torch.nn.Module):
        return len(pre_idx)
    if isinstance(weight, torch.Tensor):
        if weight.ndim == 0:
            return len(pre_idx)
        if weight.ndim == 1:
            if len(weight) == 1:
                return len(pre_idx)
            if weight.shape[0] == len(pre_idx):
                return 1
            raise ValueError(f"Weight tensor shape {weight.shape} does not match pre-synaptic indices shape {pre_idx.shape}.")
    if hasattr(weight, "__len__"):
        if len(weight) != len(pre_idx):
            raise ValueError(f"Weight tensor shape {weight.shape} does not match pre-synaptic indices shape {pre_idx.shape}.")
        return 1
    raise TypeError(f"Unsupported type for weight: {type(weight)}.")


def expand(value, n):
    if isinstance(value, torch.nn.Module):
        return value.sample(n)
    return torch.tensor(value).repeat(n)


def make_weight(weights, n):

    class ParameterOrDistributionWrapper(torch.nn.Module):
        """A simple wrapper for parameters or distributions that can be sampled."""
        def __init__(self, param):
            super().__init__()
            # nn.Parameter() is idempotent, so it's safe to call on an existing parameter.
            self.param = param
            
        def sample(self, n):
            if isinstance(self.param, torch.Tensor):
                return self.param.repeat(n)
            else:
                return self.param.sample(n)

    class WeightExpander(torch.nn.Module):
        def __init__(self, weights, n):
            super(WeightExpander, self).__init__()
            self.weights = torch.nn.ModuleList([
                ParameterOrDistributionWrapper(w) for w in weights
            ])
            self.n = n
            self.register_buffer("w", torch.empty(0))

        def forward(self):
            return self.w
        
        def init(self, reinit=True):
            if reinit or not self.w.numel():
                self.w = torch.cat([w.sample(n) for w, n in zip(self.weights, self.n)])

    return WeightExpander(weights, n)


class Network(torch.nn.Module):
    """
    Base class for networks in AxonML.
    """
    def __init__(self, populations: Dict[str, Population]):
        super(Network, self).__init__()
        self.populations = populations
        for name, pop in populations.items():
            pop.build()
            pop.name = name
            setattr(self, name, pop)

        self.synapse_spec = {}
        self.synapses = torch.nn.ModuleDict()
        self.dt = None
        self.built = False
        self.t_ind = 0

        self.backend = BACKEND.value
        self.fullgraph = bool(FULLGRAPH)
        self.dynamic = bool(DYNAMIC)
        self.jit = bool(JIT)
        self.compile_mode = COMPILE_MODE.value 

        torch._dynamo.reset()

        if self.jit:
            self._step = torch.compile(
                step, 
                backend   = self.backend,
                fullgraph = self.fullgraph,
                dynamic   = self.dynamic, 
                mode      = self.compile_mode,
            )
        else:
            self._step = torch.compile(
                step,
                backend   = 'eager',
            )
        
        self.eval()

    def train(self, mode=True):
        """
        Set the network to training mode.
        """
        for pop in self.populations.values():
            pop.train(mode)
        self.training = mode
        return self

    def train_(self, mode=True):
        self.train(mode)

    def eval(self):
        super(Network, self).eval()
        for pop in self.populations.values():
            pop.eval()
        self.training = False
        return self
    
    def eval_(self):
        self.eval()

    def device(self):
        """
        Returns the device on which the network is located.
        """
        return next(iter(self.populations.values())).device()

    def dtype(self):
        """
        Returns the data type of the network's populations.
        """
        return next(iter(self.populations.values())).dtype()

    def clear_synapses(self):
        """
        Clears all synapse specifications in the network.
        """
        self.synapse_spec = {}
        self.synapses.clear()

    def _connect(self, source_pop, source_idx, target_pop, target_idx, synapse: str,
                 threshold=0.0, weight=1.0, delay=0.0):
        """
        Internal method to connect two populations with a synapse.
        """
        n_threshold = check_weight_shape(threshold, source_idx)
        n_weight = check_weight_shape(weight, source_idx)
        n_delay = check_weight_shape(delay, source_idx)


        # Add the connection to the synapse specification
        self.synapse_spec.setdefault((source_pop.name, target_pop.name, synapse), []).append(
            (source_idx, target_idx, threshold, n_threshold, to_param(weight), n_weight, delay, n_delay)
        )

    def connect_one_to_one(
        self, source, target, synapse: str, threshold=0.0, weight=1.0, delay=0.0
    ):
        if isinstance(source, Population):
            source = source[:]  # Ensure source is a slice if it's a Population
        if isinstance(target, Population):
            target = target[:]  # Ensure target is a slice if it's a Population
        # every target compartment receives input from exactly one source compartment
        # 1. validate that the synapse exists at all the target locations
        pre_idx, post_idx = prepare_indices_one_one(source, target, synapse)
        self._connect(
            source.model, pre_idx, target.model, post_idx, synapse, threshold, weight, delay
        )

    def connect_dense(
        self, source, target, synapse: str, threshold=0.0, weight=1.0, delay=0.0
    ):
        if isinstance(source, Population):
            source = source[:]  # Ensure source is a slice if it's a Population
        if isinstance(target, Population):
            target = target[:]  # Ensure target is a slice if it's a Population
        
        # every target compartment receives input from every source compartment
        source_model = source.model
        target_model = target.model
        pre_idx  = to_flat_idx_torch(source_model.v, source.index)
        post_idx = to_flat_idx_torch(target_model.v, target.index)

        # 1. Get the original number of elements
        num_pre  = pre_idx.numel()
        num_post = post_idx.numel()

        # 2. Expand the first tensor to repeat its elements
        # Shape becomes [3, 1] -> [3, 4] -> [12]
        pre_idx = pre_idx.unsqueeze(1).expand(num_pre, num_post).flatten()

        # 3. Expand the second tensor to repeat the whole sequence
        # Shape becomes [1, 4] -> [3, 4] -> [12]
        post_idx = post_idx.unsqueeze(0).expand(num_pre, num_post).flatten()

        pre_idx, post_idx = prepare_indices_one_one_flat(
            source_model, pre_idx, target_model, post_idx, synapse
        )

        # now connect
        self._connect(
            source_model, pre_idx, target_model, post_idx, synapse, threshold, weight, delay
        )

    def connect_sparse(
        self, source, target, synapse: str, prob: float, threshold=0.0, weight=1.0, delay=0.0
    ):
        if isinstance(source, Population):
            source = source[:]  # Ensure source is a slice if it's a Population
        if isinstance(target, Population):
            target = target[:]  # Ensure target is a slice if it's a Population

        # every target compartment receives input from every source compartment
        source_model = source.model
        target_model = target.model
        pre_idx  = to_flat_idx_torch(source_model.v, source.index)
        post_idx = to_flat_idx_torch(target_model.v, target.index)

        # 1. Get the original number of elements
        num_pre  = pre_idx.numel()
        num_post = post_idx.numel()

        # 2. Expand the first tensor to repeat its elements
        # Shape becomes [3, 1] -> [3, 4] -> [12]
        pre_idx = pre_idx.unsqueeze(1).expand(num_pre, num_post).flatten()

        # 3. Expand the second tensor to repeat the whole sequence
        # Shape becomes [1, 4] -> [3, 4] -> [12]
        post_idx = post_idx.unsqueeze(0).expand(num_pre, num_post).flatten()

        # randomly select connections based on the probability
        mask = torch.rand(pre_idx.numel(), device=source_model.device()) < prob
        pre_idx = pre_idx[mask]
        post_idx = post_idx[mask]

        pre_idx, post_idx = prepare_indices_one_one_flat(
            source_model, pre_idx, target_model, post_idx, synapse
        )

        # now connect
        self._connect(
            source_model, pre_idx, target_model, post_idx, 
            synapse, threshold, weight, delay
        )

    def build_synapses(self, dt):
        for (pre_name, post_name, synapse), specs in self.synapse_spec.items():
            pre = self.populations[pre_name]
            post = self.populations[post_name]
            pre_idx = torch.cat([s[0] for s in specs])
            post_idx = torch.cat([s[1] for s in specs])
            thresholds = torch.cat([expand(s[2], s[3]) for s in specs])
            weights = make_weight([s[4] for s in specs], [s[5] for s in specs])
            delay = torch.cat([expand(s[6], s[7]) for s in specs])

            syn = NetCon(
                pre=pre,
                pre_idx=pre_idx,
                thresholds=thresholds,
                post=post,
                post_idx=post_idx,
                post_syn=synapse,
                weight=weights,
                delay=delay,
                dt=dt
            ).to(device=self.device(), dtype=self.dtype())
            self.synapses[f"{pre_name}_{post_name}_{synapse.name}"] = syn

    def build(self, dt):
        """
        Build the network by initializing populations and synapses.
        This method should be called before running the network.
        """
        if not self.built or self.dt != dt:
            torch._dynamo.reset()
            self.dt = dt
            self.build_synapses(dt)
            self.built = True
        return self

    def initialize(self, dt: float, reinit_weights: bool = True):
        """
        Initialize the network. This method should be overridden by subclasses.
        """
        self.build(dt)
        dt = torch.tensor(dt, device=self.device(), dtype=self.dtype())
        for pop in self.populations.values():
            pop.initialize()
            pop.integrator.initialize(pop, dt)
            pop.intra = pop.build_intra()
        self.init_synapses(reinit_weights=reinit_weights)
        self.t_ind = 0
        return self
    
    def initialize_(self, dt: float, reinit_weights: bool = True):
        """
        Initialize the network. This method should be overridden by subclasses.
        """
        self.initialize(dt, reinit_weights=reinit_weights)

    def init_synapses(self, reinit_weights: bool = True):
        for syn in self.synapses.values():
            syn.zero()
            syn.detach()
            syn.weight.init(reinit=reinit_weights)

    def run(self, tstop, callbacks=None, progressbar=False):
        dt = torch.tensor(self.dt, device=self.device(), dtype=self.dtype())
        dt_f = self.dt

        ctx = nullcontext() if self.training else torch.no_grad()

        intra = {}
        for n, p in self.populations.items():
            if p is not None:
                intra[n] = p.intra

        with_intra = bool(intra)

        with ctx:
            n_steps = int(tstop / self.dt)

            if with_intra:
                intra = {n: (intra_, *self.populations[n].prep_intra(intra_, n_steps, dt)) for n, intra_ in intra.items()}

            if callbacks is None:
                callbacks = []
            for c in callbacks:
                c.dt = self.dt
            callbacks = CallbackList(callbacks)
            pre_loop_hook(callbacks, self)

            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(total=n_steps, desc=f"{self.t_ind*dt_f:.3f} ms")

            local_ind = 0

            for _ in range(n_steps):

                intra_c = {}

                if with_intra:
                    intra_c = prepare_intra(intra_c, intra, local_ind)

                self._step(self.populations, self.synapses, dt, intra=intra_c)
                post_step_hook(callbacks, self)
                self.t_ind += 1
                local_ind += 1

                if progressbar:
                    progressbar.update(1)
                    if self.t_ind % 100 == 0:
                        progressbar.set_description(f"{self.t_ind*dt_f:.1f} ms")
            
            if progressbar:
                progressbar.close()    

            post_loop_hook(callbacks, self)


def prepare_intra(intra_c, intra, local_ind):
    """
    Prepares the intra-cellular data for the current step.
    """
    for n, (intra_, stims, indices) in intra.items():
        s = [st[local_ind] for st in stims]
        intra_c[n] = make_intra(intra_, s, indices)
    return intra_c


# callback helpers
def pre_loop_hook(c, m):
    c.pre_loop_hook(m)

def post_loop_hook(c, m):
    c.post_loop_hook(m)

def pre_step_hook(c, m):
    c.pre_step_hook(m)

@torch.compile
def post_step_hook(c, m):
    c.post_step_hook(m)

def pre_chunk_hook(c, m, n):
    c.pre_chunk_hook(m, n)

def post_chunk_hook(c, m, n):
    c.post_chunk_hook(m, n)