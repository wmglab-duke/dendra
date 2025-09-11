import gc
import math
from contextlib import nullcontext
from typing import Dict

import torch
from tqdm.auto import tqdm

from axonml.helpers import BACKEND, COMPILE_MODE, DYNAMIC, FULLGRAPH, JIT

from ..callbacks import CallbackList
from ..core import Population, make_intra
from ..parametric import to_param
from .netcon import NetCon
from .netstim import NetStim


def to_flat_idx_torch(shape, idx, device):
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

    # 1. Create a grid of flat indices with the same shape as the input array.
    #    e.g., for a (2, 3) tensor, this becomes [[0, 1, 2], [3, 4, 5]]
    indices_grid = torch.arange(torch.prod(torch.tensor(shape)), device=device).view(
        shape
    )

    # 2. Apply the user's index to this grid. PyTorch's indexing logic
    #    will select the corresponding flat indices for us.
    selected_indices = indices_grid[idx]

    # 3. Flatten the result to get a 1D tensor of flat indices.
    return selected_indices.flatten()


def step_pop(integrator, model, dt, ve=None, intra=None):
    integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def step(
    populations,
    synapses,
    netstim,
    t,
    dt,
    ve: Dict[str, torch.Tensor | None] = {},
    intra: Dict[str, torch.Tensor | None] = {},
):
    if netstim is not None:
        netstim(t)
    for s in synapses.values():
        s.advance()
    for n, pop in populations.items():
        step_pop(pop.integrator, pop, dt, ve.get(n, None), intra.get(n, None))


def get_local_index(population, mech, index):
    indices = torch.full_like(
        population.v, -1, dtype=torch.long, device=population.device()
    ).flatten()
    mech_key_flat = to_flat_idx_torch(population.shape, mech.key, population.device())
    indices.index_copy_(
        0,
        mech_key_flat,
        torch.arange(
            mech_key_flat.numel(), device=population.device(), dtype=torch.long
        ),
    )
    return indices.index_select(0, index)


def prepare_indices_one_one(source, target, synapse):
    source_model = source.model
    target_model = target.model
    syn = synapse

    index_arr = torch.arange(
        target_model.v.numel(), device=target_model.device(), dtype=target_model.dtype()
    ).view_as(target_model.v)

    indices_in_synapse = syn.get(index_arr).flatten()
    post_idx = to_flat_idx_torch(
        target_model.shape, target.index, target_model.device()
    )

    if not torch.all(torch.isin(post_idx, indices_in_synapse)):
        raise ValueError(
            f"Target population '{target.name}' does not have the synapse '{synapse}' at all target locations."
        )

    pre_idx = to_flat_idx_torch(source_model.shape, source.index, source_model.device())
    post_idx = get_local_index(target_model, syn, post_idx)

    return pre_idx, post_idx


def prepare_indices_one_one_flat(
    source_model, source_index, target_model, target_index, synapse
):
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
        raise ValueError(
            f"Target population '{target_model.name}' does not have the synapse '{synapse}' at all target locations."
        )

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
            raise ValueError(
                f"Weight tensor shape {weight.shape} does not match pre-synaptic indices shape {pre_idx.shape}."
            )
    if hasattr(weight, "__len__"):
        if len(weight) != len(pre_idx):
            raise ValueError(
                f"Weight tensor shape {weight.shape} does not match pre-synaptic indices shape {pre_idx.shape}."
            )
        return 1
    raise TypeError(f"Unsupported type for weight: {type(weight)}.")


def expand(value, n):
    if isinstance(value, torch.nn.Module):
        return value.sample(n)
    return torch.tensor(value).repeat(n)


def batchify_index(old_shape, n: int, i: torch.Tensor) -> torch.Tensor:
    """
    Build i_n so that:
        t_n = t.unsqueeze(0).repeat(n, *([1]*len(old_shape)))   # n copies of t
        # (Assuming t_n is contiguous; if you used expand(), call .contiguous() before .view)
        t_n.view(-1)[i_n] == t_n.reshape(n, -1)[..., i].reshape(-1)

    Args:
        old_shape: shape of t BEFORE adding the batch (tuple/torch.Size)
        n:         batch size
        i:         LongTensor of indices into the last dim of t_n.reshape(n, -1)
                   (any shape; negatives allowed; broadcast across the n rows)

    Returns:
        i_n: LongTensor of shape (n, *i.shape) — flat indices into t_n.view(-1)
    """
    M = math.prod(tuple(old_shape))  # width of the last dim in t_n.reshape(n, -1)
    if i.dtype != torch.long:
        i = i.to(torch.long)
    i = i % M  # normalize negatives

    # row offsets: 0, M, 2M, ..., (n-1)M; broadcast across i's shape
    r = torch.arange(n, device=i.device, dtype=torch.long).view((n,) + (1,) * i.ndim)
    i_n = r * M + i
    return i_n.reshape(-1)


def make_weight(weights, n):
    class ParameterOrDistributionWrapper(torch.nn.Module):
        """A simple wrapper for parameters or distributions that can be sampled."""

        def __init__(self, param):
            super().__init__()
            self.param = param

        def sample(self, n):
            if isinstance(self.param, torch.Tensor):
                return self.param.repeat(n)
            else:
                return self.param.sample(n)

    class WeightExpander(torch.nn.Module):
        def __init__(self, weights, n):
            super(WeightExpander, self).__init__()
            self.weights = torch.nn.ModuleList(
                [ParameterOrDistributionWrapper(w) for w in weights]
            )
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

    def __init__(self, populations: Dict[str, Population], netstim=None):
        if any(pop.is_batched() for pop in populations.values()):
            raise ValueError(
                "Batched populations are not supported. Implement your networks with unbatched populations and then call .batch(batch_size)."
            )
        super(Network, self).__init__()
        self.populations = populations
        for name, pop in populations.items():
            pop.build()
            pop.name = name
            setattr(self, name, pop)

        self.netstim = netstim

        self.synapse_spec = {}
        self.synapses = torch.nn.ModuleDict()
        self.dt = None
        self.built = False

        self.backend = BACKEND.value
        self.fullgraph = bool(FULLGRAPH)
        self.dynamic = bool(DYNAMIC)
        self.jit = bool(JIT)
        self.compile_mode = COMPILE_MODE.value

        torch._dynamo.reset()

        if self.jit:
            self._step = torch.compile(
                step,
                backend=self.backend,
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
                mode=self.compile_mode,
            )
        else:
            self._step = torch.compile(
                step,
                backend="eager",
            )

        self.register_buffer(
            "t", torch.tensor(0.0, device=self.device(), dtype=self.dtype())
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
        self.built = False

    def _connect(
        self,
        source_pop,
        source_idx,
        target_pop,
        target_idx,
        synapse,
        threshold=0.0,
        weight=1.0,
        delay=0.0,
    ):
        """
        Internal method to connect two populations with a synapse.
        """
        n_threshold = check_weight_shape(threshold, source_idx)
        n_weight = check_weight_shape(weight, source_idx)
        n_delay = check_weight_shape(delay, source_idx)

        # Add the connection to the synapse specification
        self.synapse_spec.setdefault(
            (source_pop.name, target_pop.name, synapse), []
        ).append(
            (
                source_idx,
                target_idx,
                threshold,
                n_threshold,
                to_param(weight),
                n_weight,
                delay,
                n_delay,
            )
        )

    def connect_one_to_one(
        self, source, target, synapse, threshold=0.0, weight=1.0, delay=0.0
    ):
        """
        Connect source to target one-to-one.

        Each selected source element connects to exactly one selected target
        element (pairwise). The target locations must already host the given
        synapse mechanism.

        Parameters
        ----------
        source : Population | NetStim | PopulationSlice | NetStimSlice
            Source population (or a slice produced via source[...]).
            If a Population/NetStim is passed, it is converted to source[:].
        target : Population | PopulationSlice
            Target population (or a slice via target[...]). Converted to target[:]
            if a Population is passed.
        synapse : object
            Target-side synapse mechanism attached to the target population. It
            must be present at all target locations selected by `target`.
        threshold : float | torch.Tensor | torch.nn.Module, optional
            Spike threshold(s) for the pre-synaptic units. A scalar applies to
            all connections. A length-N tensor/module output provides one value
            per pre-synaptic unit. Default is 0.0.
        weight : float | torch.Tensor | torch.nn.Module, optional
            Synaptic weight(s). A scalar applies to all connections. A tensor of
            length N (number of pre-synaptic indices) supplies per-connection
            weights. A torch.nn.Module is expected to implement .sample(N).
            Default is 1.0.
        delay : float | torch.Tensor | torch.nn.Module, optional
            Synaptic delay(s) in ms. Same broadcasting rules as `weight`.
            Default is 0.0.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the synapse is not present at all specified target locations, or
            if provided tensors have incompatible shapes with the selected
            indices.

        Notes
        -----
        - The number of selected source elements must match the number of
          selected target elements for a one-to-one mapping.
        - The connection specifications are queued and materialized during
          build()/initialize().

        Examples
        --------
        >>> net.connect_one_to_one(pop_pre, pop_post, pop_post.mech.syn)
        """
        if isinstance(source, Population) or isinstance(source, NetStim):
            source = source[:]  # Ensure source is a slice if it's a Population
        if isinstance(target, Population):
            target = target[:]  # Ensure target is a slice if it's a Population
        # every target compartment receives input from exactly one source compartment
        # 1. validate that the synapse exists at all the target locations
        pre_idx, post_idx = prepare_indices_one_one(source, target, synapse)
        self._connect(
            source.model,
            pre_idx,
            target.model,
            post_idx,
            synapse,
            threshold,
            weight,
            delay,
        )

    def connect_dense(
        self, source, target, synapse, threshold=0.0, weight=1.0, delay=0.0
    ):
        """
        Connect source to target densely (all-to-all between selections).

        Every selected target element receives input from every selected source
        element. Target locations must already host the given synapse mechanism.

        Parameters
        ----------
        source : Population | NetStim | PopulationSlice | NetStimSlice
            Source population (or a slice produced via source[...]). If a
            Population/NetStim is passed, it is converted to source[:].
        target : Population | PopulationSlice
            Target population (or a slice via target[...]). Converted to target[:]
            if a Population is passed.
        synapse : object
            Target-side synapse mechanism attached to the target population.
        threshold : float | torch.Tensor | torch.nn.Module, optional
            Spike threshold(s) for the pre-synaptic units. See connect_one_to_one
            for broadcasting rules. Default is 0.0.
        weight : float | torch.Tensor | torch.nn.Module, optional
            Synaptic weight(s). See connect_one_to_one for broadcasting rules.
            Default is 1.0.
        delay : float | torch.Tensor | torch.nn.Module, optional
            Synaptic delay(s) in ms. See connect_one_to_one for broadcasting
            rules. Default is 0.0.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the synapse is not present at some target locations, or if
            provided tensors have incompatible shapes.

        Notes
        -----
        - Forms a complete bipartite connectivity between the selected pre and
          post indices (all pairs).
        - Connection specs are queued and built during build()/initialize().

        Examples
        --------
        >>> net.connect_dense(pop_pre[:], pop_post[:], pop_post.mech.syn)
        """
        if isinstance(source, Population) or isinstance(source, NetStim):
            source = source[:]  # Ensure source is a slice if it's a Population
        if isinstance(target, Population):
            target = target[:]  # Ensure target is a slice if it's a Population

        # every target compartment receives input from every source compartment
        source_model = source.model
        target_model = target.model
        pre_idx = to_flat_idx_torch(
            source_model.shape, source.index, source_model.device()
        )
        post_idx = to_flat_idx_torch(
            target_model.shape, target.index, target_model.device()
        )

        # 1. Get the original number of elements
        num_pre = pre_idx.numel()
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
            source_model,
            pre_idx,
            target_model,
            post_idx,
            synapse,
            threshold,
            weight,
            delay,
        )

    def connect_sparse(
        self, source, target, synapse, prob: float, threshold=0.0, weight=1.0, delay=0.0
    ):
        """
        Connect source to target sparsely via Bernoulli sampling over all pairs.

        Starting from the dense all-to-all candidate set between `source` and
        `target`, keep each candidate connection independently with probability
        `prob`. Target locations must already host the given synapse mechanism.

        Parameters
        ----------
        source : Population | NetStim | PopulationSlice | NetStimSlice
            Source selection; converted to source[:] if a full Population/NetStim
            is provided.
        target : Population | PopulationSlice
            Target selection; converted to target[:] if a full Population.
        synapse : object
            Target-side synapse mechanism attached to the target population.
        prob : float
            Independent probability (0 ≤ prob ≤ 1) of keeping each candidate
            pre-post pair.
        threshold : float | torch.Tensor | torch.nn.Module, optional
            Spike threshold(s); broadcasting as in connect_one_to_one.
        weight : float | torch.Tensor | torch.nn.Module, optional
            Synaptic weight(s); broadcasting as in connect_one_to_one.
        delay : float | torch.Tensor | torch.nn.Module, optional
            Synaptic delay(s); broadcasting as in connect_one_to_one.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the synapse is not present at some target locations, or if
            provided tensors have incompatible shapes.

        Notes
        -----
        - If no connections are sampled, no specs are added (early return).
        - Randomness comes from torch.rand on the source device.

        Examples
        --------
        >>> net.connect_sparse(pop_pre[:], pop_post[:], pop_post.mech.syn, prob=0.2)
        """
        if isinstance(source, Population) or isinstance(source, NetStim):
            source = source[:]  # Ensure source is a slice if it's a Population
        if isinstance(target, Population):
            target = target[:]  # Ensure target is a slice if it's a Population

        # every target compartment receives input from every source compartment
        source_model = source.model
        target_model = target.model
        pre_idx = to_flat_idx_torch(
            source_model.shape, source.index, source_model.device()
        )
        post_idx = to_flat_idx_torch(
            target_model.shape, target.index, target_model.device()
        )

        # 1. Get the original number of elements
        num_pre = pre_idx.numel()
        num_post = post_idx.numel()

        # 2. Expand the first tensor to repeat its elements
        # Shape becomes [3, 1] -> [3, 4] -> [12]
        pre_idx = pre_idx.unsqueeze(1).expand(num_pre, num_post).flatten()

        # 3. Expand the second tensor to repeat the whole sequence
        # Shape becomes [1, 4] -> [3, 4] -> [12]
        post_idx = post_idx.unsqueeze(0).expand(num_pre, num_post).flatten()

        # randomly select connections based on the probability
        mask = torch.rand(pre_idx.numel(), device=source_model.device()) < prob

        if mask.sum() == 0:
            # If no connections are selected, return early
            return

        pre_idx = pre_idx[mask]
        post_idx = post_idx[mask]

        pre_idx, post_idx = prepare_indices_one_one_flat(
            source_model, pre_idx, target_model, post_idx, synapse
        )

        # now connect
        self._connect(
            source_model,
            pre_idx,
            target_model,
            post_idx,
            synapse,
            threshold,
            weight,
            delay,
        )

    connect_prob = connect_sparse

    def connect_prob_n(
        self, source, target, synapse, n: int, threshold=0.0, weight=1.0, delay=0.0
    ):
        """
        Connect exactly n random pre-post pairs (without replacement).

        From the dense all-to-all candidate set between `source` and `target`,
        sample n unique pairs uniformly without replacement. Target locations
        must already host the given synapse mechanism.

        Parameters
        ----------
        source : Population | NetStim | PopulationSlice | NetStimSlice
            Source selection; converted to source[:] if a full Population/NetStim
            is provided.
        target : Population | PopulationSlice
            Target selection; converted to target[:] if a full Population.
        synapse : object
            Target-side synapse mechanism attached to the target population.
        n : int
            Number of connections to sample. If n <= 0, no connections are added.
            If n exceeds the number of possible pairs, all pairs are selected.
        threshold : float | torch.Tensor | torch.nn.Module, optional
            Spike threshold(s); broadcasting as in connect_one_to_one.
        weight : float | torch.Tensor | torch.nn.Module, optional
            Synaptic weight(s); broadcasting as in connect_one_to_one.
        delay : float | torch.Tensor | torch.nn.Module, optional
            Synaptic delay(s); broadcasting as in connect_one_to_one.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the synapse is not present at some target locations, or if
            provided tensors have incompatible shapes.

        Notes
        -----
        - Sampling uses torch.randperm on the source device.
        - n is effectively clipped by the number of candidate pairs.

        Examples
        --------
        >>> net.connect_prob_n(pop_pre[:], pop_post[:], pop_post.mech.syn, n=1000)
        """
        """
        Connect the source population to the target population with a fixed number of connections.
        """
        if n <= 0:
            return
        if isinstance(source, Population) or isinstance(source, NetStim):
            source = source[:]
        if isinstance(target, Population):
            target = target[:]

        # every target compartment receives input from every source compartment
        source_model = source.model
        target_model = target.model
        pre_idx = to_flat_idx_torch(
            source_model.shape, source.index, source_model.device()
        )
        post_idx = to_flat_idx_torch(
            target_model.shape, target.index, target_model.device()
        )

        # 1. Get the original number of elements
        num_pre = pre_idx.numel()
        num_post = post_idx.numel()

        # 2. Expand the first tensor to repeat its elements
        # Shape becomes [3, 1] -> [3, 4] -> [12]
        pre_idx = pre_idx.unsqueeze(1).expand(num_pre, num_post).flatten()

        # 3. Expand the second tensor to repeat the whole sequence
        # Shape becomes [1, 4] -> [3, 4] -> [12]
        post_idx = post_idx.unsqueeze(0).expand(num_pre, num_post).flatten()

        # randomly select connections based on the probability
        mask = torch.randperm(pre_idx.numel(), device=source_model.device())[:n]

        pre_idx = pre_idx[mask]
        post_idx = post_idx[mask]

        pre_idx, post_idx = prepare_indices_one_one_flat(
            source_model, pre_idx, target_model, post_idx, synapse
        )

        # now connect
        self._connect(
            source_model,
            pre_idx,
            target_model,
            post_idx,
            synapse,
            threshold,
            weight,
            delay,
        )

    connect_sparse_n = connect_prob_n

    def build_synapses(self, dt):
        for (pre_name, post_name, synapse), specs in self.synapse_spec.items():
            pre = getattr(self, pre_name)
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
                dt=dt,
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
        if self.netstim is not None:
            self.netstim.initialize()
        self.t = self.t.detach()
        self.t.zero_()
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

    def run(self, tstop, ve=None, callbacks=None, progressbar=False):
        dt = torch.tensor(self.dt, device=self.device(), dtype=self.dtype())
        dt_f = self.dt

        ctx = nullcontext() if self.training else torch.no_grad()

        intra = {}
        for n, p in self.populations.items():
            if p.intra is not None:
                intra[n] = p.intra

        ve = ve if ve is not None else {}
        ve = {
            n: (
                v.to(device=self.device(), dtype=self.dtype()),
                t.to(device=self.device(), dtype=self.dtype()),
            )
            for n, (v, t) in ve.items()
        }
        ve = {
            n: (v, t.assemble(self.t, self.t + tstop, dt)) for n, (v, t) in ve.items()
        }

        with_intra = bool(intra)
        with_ve = bool(ve)

        tstart = self.t.item()

        with ctx:
            n_steps = int(tstop / self.dt)

            if with_intra:
                intra = {
                    n: (intra_, *self.populations[n].prep_intra(intra_, n_steps, dt))
                    for n, intra_ in intra.items()
                }

            if callbacks is None:
                callbacks = []
            for c in callbacks:
                c.dt = self.dt
            callbacks = CallbackList(callbacks)
            pre_loop_hook(callbacks, self)

            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(total=n_steps, desc=f"{tstart:.1f} ms")

            local_ind = 0

            for _ in range(n_steps):
                intra_c = {}
                ve_c = {}

                if with_intra:
                    intra_c = prepare_intra(intra_c, intra, local_ind)

                if with_ve:
                    ve_c = prepare_ve(ve, local_ind)

                self._step(
                    self.populations,
                    self.synapses,
                    self.netstim,
                    self.t,
                    dt,
                    ve=ve_c,
                    intra=intra_c,
                )
                self.t = self.t + dt
                post_step_hook(callbacks, self)
                local_ind += 1

                if progressbar:
                    progressbar.update(1)
                    if local_ind % 100 == 0:
                        progressbar.set_description(
                            f"{tstart + local_ind * dt_f:.1f} ms"
                        )

            if progressbar:
                progressbar.close()

            post_loop_hook(callbacks, self)

    def batch(self, n, include_netstim=True):
        _synapse_spec = self.synapse_spec.copy()
        self.clear_synapses()
        _old_shapes = {}
        for name, p in self.populations.items():
            _old_shapes[name] = p.shape
            device = p.device()
            p.batch(n)
            p.build(force_rebuild=True)
            p.to(device)
        if include_netstim and self.netstim is not None:
            _old_shapes["netstim"] = self.netstim.shape
            self.netstim.batch(n)
        for k, v in _synapse_spec.items():
            source_name, target_name, synapse = k
            source_pop = getattr(self, source_name)
            target_pop = getattr(self, target_name)
            synapse = getattr(target_pop.mech, synapse.name)
            for data in v:
                source_idx, target_idx = data[0], data[1]
                threshold, weight, delay = data[2], data[4], data[6]
                if source_name == "netstim" and not include_netstim:
                    new_source_idx = source_idx.repeat(n)
                else:
                    new_source_idx = batchify_index(
                        _old_shapes[source_name], n, source_idx
                    )
                new_target_idx = batchify_index(_old_shapes[target_name], n, target_idx)
                self._connect(
                    source_pop,
                    new_source_idx,
                    target_pop,
                    new_target_idx,
                    synapse,
                    threshold,
                    weight,
                    delay,
                )
        self.built = False
        gc.collect()
        return self

    def batch_(self, n, include_netstim=True):
        self.batch(n, include_netstim=include_netstim)


def prepare_intra(intra_c, intra, local_ind):
    """
    Prepares the intra-cellular data for the current step.
    """
    for n, (intra_, stims, indices) in intra.items():
        s = [st[local_ind] for st in stims]
        intra_c[n] = make_intra(intra_, s, indices)
    return intra_c


@torch.compile
def prepare_ve(ve, local_ind: int):
    """
    Prepares the voltage and time data for the current step.
    """
    return {n: v * t[local_ind] for n, (v, t) in ve.items()}


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
