from typing import Dict, Tuple

import torch

from ..core import Population, Waveform
from ..parametric import to_param
from ..callbacks import CallbackList
from .delaydelivery import VariableDelayDelivery


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


@torch.compile
def step(populations, synapses, dt, ve: Dict[str, torch.Tensor | None]={}, intra: Dict[str, torch.Tensor | None]={}):
    for s in synapses:
        s.advance()
    for n, pop in populations.items():
        step_pop(pop.integrator, pop, dt, ve=ve.get(n, None), intra=intra.get(n, None))


def get_local_index(population, mech, index):
    indices = torch.full_like(population.v, -1, dtype=torch.long, device=population.device()).flatten()
    mech_key_flat = to_flat_idx_torch(population.v, mech.key)
    indices.index_copy_(0, mech_key_flat, torch.arange(mech_key_flat.numel(), device=population.device(), dtype=torch.long))
    return indices.index_select(0, index)


def prepare_indices_one_one(source, target, synapse):
    source_model = source.model
    target_model = target.model
    syn = getattr(getattr(target_model, 'mech'), synapse)

    index_arr = torch.arange(
        target_model.v.numel(), device=target_model.device(), dtype=target_model.dtype()
    ).view(target_model.v.shape)

    indices_in_synapse = syn.get(index_arr).flatten()
    post_idx = to_flat_idx_torch(target_model.v, target.index)

    if not torch.all(torch.isin(post_idx, indices_in_synapse)):
        raise ValueError(f"Target population '{target.name}' does not have the synapse '{synapse}' at all target locations.")
    
    pre_idx = to_flat_idx_torch(source_model.v, source.index)
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
    elif hasattr(weight, "__len__"):
        if len(weight) != len(pre_idx):
            raise ValueError(f"Weight tensor shape {weight.shape} does not match pre-synaptic indices shape {pre_idx.shape}.")
        return 1
    else:
        raise TypeError(f"Unsupported type for weight: {type(weight)}.")


def expand(value, n):
    return torch.tensor(value).repeat(n)


def make_weight(weights, n):
    class WeightExpander(torch.nn.Module):
        def __init__(self, weights, n):
            super(WeightExpander, self).__init__()
            self.weights = torch.nn.ParameterList(weights)
            self.n = n

        def forward(self):
            return torch.cat([w.repeat(n) for w, n in zip(self.weights, self.n)])
    return WeightExpander(weights, n)


class Network(torch.nn.Module):
    """
    Base class for networks in AxonML.
    """
    def __init__(self, populations: Dict[str, Population]):
        super(Network, self).__init__()
        self.populations = torch.nn.ModuleDict(populations)
        for name, pop in populations.items():
            pop.build()
            pop.name = name
            setattr(self, name, pop)

        self.synapse_spec = {}

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
        # 1. validate that the synapse exists at all the target locations
        pre_idx, post_idx = prepare_indices_one_one(source, target, synapse)
        self._connect(source.model, pre_idx, target.model, post_idx, synapse, threshold, weight, delay)

    def connect_dense(
        self, source, target, synapse: str, threshold=0.0, weight=1.0, delay=0.0
    ):
        # every target compartment receives input from every source compartment
        source_model = source.model
        target_model = target.model
        pre_idx = to_flat_idx_torch(source_model.v, source.index)
        post_idx = to_flat_idx_torch(target_model.v, target.index)

        # 1. Get the original number of elements
        num_pre = pre_idx.numel()
        num_post = post_idx.numel()

        # 2. Expand the first tensor to repeat its elements
        # Shape becomes [3, 1] -> [3, 4] -> [12]
        pre_out = pre_idx.unsqueeze(1).expand(num_pre, num_post).flatten()

        # 3. Expand the second tensor to repeat the whole sequence
        # Shape becomes [1, 4] -> [3, 4] -> [12]
        post_out = post_idx.unsqueeze(0).expand(num_pre, num_post).flatten()

        # now connect
        self._connect(source_model, pre_out, target_model, post_out, synapse, threshold, weight, delay)

    def connect_sparse(
        self, source, target, synapse: str, prob: float, threshold=0.0, weight=1.0, delay=0.0
    ):
        # every target compartment receives input from every source compartment
        source_model = source.model
        target_model = target.model
        pre_idx = to_flat_idx_torch(source_model.v, source.index)
        post_idx = to_flat_idx_torch(target_model.v, target.index)

        # 1. Get the original number of elements
        num_pre = pre_idx.numel()
        num_post = post_idx.numel()

        # 2. Expand the first tensor to repeat its elements
        # Shape becomes [3, 1] -> [3, 4] -> [12]
        pre_out = pre_idx.unsqueeze(1).expand(num_pre, num_post).flatten()

        # 3. Expand the second tensor to repeat the whole sequence
        # Shape becomes [1, 4] -> [3, 4] -> [12]
        post_out = post_idx.unsqueeze(0).expand(num_pre, num_post).flatten()

        # randomly select connections based on the probability
        mask = torch.rand(pre_out.numel(), device=source_model.device()) < prob
        pre_out = pre_out[mask]
        post_out = post_out[mask]

        # now connect
        self._connect(source_model, pre_out, target_model, post_out, synapse, threshold, weight, delay)

    def build_synapses(self, dt):
        synapses = []
        for (pre_name, post_name, synapse), specs in self.synapse_spec.items():
            pre = self.populations[pre_name]
            post = self.populations[post_name]
            pre_idx = torch.cat([s[0] for s in specs])
            post_idx = torch.cat([s[1] for s in specs])
            thresholds = torch.cat([expand(s[2], s[3]) for s in specs])
            weights = make_weight([s[4] for s in specs], [s[5] for s in specs])
            delay = torch.cat([expand(s[6], s[7]) for s in specs])

            syn = VariableDelayDelivery(
                pre=pre,
                pre_idx=pre_idx,
                thresholds=thresholds,
                post=post,
                post_idx=post_idx,
                post_syn=getattr(post.mech, synapse),
                weight=weights,
                delay=delay,
                dt=dt
            )
            synapses.append(syn)
            self.add_module(f"{pre_name}_{post_name}_{synapse}", syn)
        self.synapses = torch.nn.ModuleList(synapses)

    def initialize(self, dt):
        """
        Initialize the network. This method should be overridden by subclasses.
        """
        for pop in self.populations.values():
            pop.initialize()
            pop.integrator.initialize(pop, dt)
        self.build_synapses(float(dt))

    def run(self, tstop, dt, callbacks=None):
        dt = torch.tensor(dt, device=self.device(), dtype=self.dtype())
        self.initialize(dt)
        n_steps = int(tstop / dt.item())
        if callbacks is None:
            callbacks = []
        for c in callbacks:
            c.dt = float(dt.item())
        callbacks = CallbackList(callbacks)
        pre_loop_hook(callbacks, self)
        for _ in range(n_steps):
            step(self.populations, self.synapses, dt)
            post_step_hook(callbacks, self)
        post_loop_hook(callbacks, self)


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