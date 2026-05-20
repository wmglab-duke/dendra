import gc
import math
from contextlib import nullcontext
from typing import Dict, Literal, Optional

import torch
from tqdm.auto import tqdm

from dendra.helpers import BACKEND, COMPILE_MODE, DYNAMIC, FULLGRAPH, JIT

from ..callbacks import CallbackList
from ..core import Population, _match_state_dict, make_intra
from ..multi import concat_models, indices
from ..parametric import is_parametric, to_param
from ..rng import RNGMixin
from .netcon import NetCon
from .netstim import NetStim

ConnRule = Literal[
    "one_to_one",
    "all_to_all",
    "pairwise_bernoulli",
    "pairwise_poisson",
    "fixed_total_number",
    "fixed_indegree",
    "fixed_outdegree",
]


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


compiled_step_pop = torch.compile(step_pop)


@torch.compile
def advance_populations(populations, dt, extra, intra):
    for n, pop in populations.items():
        step_pop(pop.integrator, pop, dt, extra.get(n, None), intra.get(n, None))


def step(
    populations,
    synapses,
    netstim,
    t,
    dt,
    extra: Dict[str, torch.Tensor | None] = {},
    intra: Dict[str, torch.Tensor | None] = {},
    compiled_advance_population=True,
):
    if netstim is not None:
        netstim(t, bptt=netstim.training)
    for s in synapses.values():
        s.advance()

    if compiled_advance_population:
        advance_populations(populations, dt, extra, intra)
    else:
        for n, pop in populations.items():
            if pop.jit_in_network:
                compiled_step_pop(
                    pop.integrator, pop, dt, extra.get(n, None), intra.get(n, None)
                )
            else:
                step_pop(
                    pop.integrator, pop, dt, extra.get(n, None), intra.get(n, None)
                )


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


def _last_celsius(populations):
    pops = [pop for pop in populations.values()]
    return pops[-1].celsius if pops else None


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
    if isinstance(weight, torch.nn.Module):
        return len(pre_idx)
    raise TypeError(f"Unsupported type for weight: {type(weight)}.")


def expand(value, n):
    if isinstance(value, torch.nn.Module):
        return value.sample(n)
    return torch.tensor(value).repeat(n)


def _require(spec: dict, *names):
    for name in names:
        if name in spec:
            return spec[name]
    raise ValueError(
        f"Missing required connection parameter: one of {', '.join(names)}."
    )


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
            if is_parametric(self.param):
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
            return self

    return WeightExpander(weights, n)


def dilate(
    event_deliveries: torch.Tensor,
    dt_old: float,
    dt_new: float,
    *,
    mode: Literal["nearest", "floor", "ceil"] = "nearest",
    horizon_ms: Optional[float] = None,
    n_limit: Optional[int] = None,
) -> torch.Tensor:
    """
    Re-bin a (T, E) event buffer from dt_old to dt_new by mapping each old
    time bin t=k*dt_old to a new row index on the dt_new grid and summing
    collisions.

    Parameters
    ----------
    event_deliveries : (T, E) torch.Tensor
        Rows are time bins; columns are event channels. Any numeric dtype works.
        If dtype == bool, counts will be summed as integers.
    dt_old : float
        Original bin width (ms).
    dt_new : float
        Target bin width (ms), must be > 0.
    mode : {"nearest","floor","ceil"}, default "nearest"
        How to choose the new bin for each original time:
          - "nearest": half-up rounding (i.e., round(x) via floor(x+0.5) for x >= 0)
          - "floor":   lower bin
          - "ceil":    upper bin
    horizon : int, optional
        If provided, forces the output to cover [0, horizon) on the new grid.
        If omitted, the output length is just enough to include the last mapped bin.

    Returns
    -------
    (T_new, E) torch.Tensor
        Re-binned tensor on the dt_new grid, with rows summed where multiple
        old bins map to the same new bin.
    """
    if event_deliveries.ndim != 2:
        raise ValueError("event_deliveries must be 2D (T, E).")
    if not (dt_old > 0 and dt_new > 0):
        raise ValueError("dt_old and dt_new must be > 0.")
    if dt_old == dt_new:
        return event_deliveries

    T, E = event_deliveries.shape
    device = event_deliveries.device

    # old times: t_k = k * dt_old ; compute fractional new indices t_k / dt_new
    idx_f = torch.arange(T, device=device, dtype=torch.float64) * (dt_old / dt_new)

    if mode == "nearest":
        # half-up rounding for nonnegative times (avoids banker's rounding)
        idx = torch.floor(idx_f + 0.5).to(torch.long)
    elif mode == "floor":
        idx = torch.floor(idx_f).to(torch.long)
    elif mode == "ceil":
        idx = torch.ceil(idx_f).to(torch.long)
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    # Determine output length
    if T == 0:
        T_new = 0
    else:
        idx_max = int(idx.max().item())
        if horizon_ms is not None:
            forced_len = int(math.ceil(horizon_ms / dt_new))
            T_new = max(idx_max + 1, forced_len)
        else:
            T_new = idx_max + 1

    if n_limit is not None:
        T_new = n_limit

    idx = idx.clamp(0, n_limit - 1) if n_limit is not None else idx

    out_dtype = event_deliveries.dtype
    # For boolean inputs, sum counts as integers (can .bool() after if you want OR semantics)
    if out_dtype == torch.bool:
        src = event_deliveries.to(torch.int64)
        out = torch.zeros((T_new, E), device=device, dtype=torch.int64)
    else:
        src = event_deliveries
        out = torch.zeros((T_new, E), device=device, dtype=out_dtype)

    # Sum rows that map to the same new bin. O(T*E) with efficient fused add.
    if T_new > 0:
        out.index_add_(0, idx, src)

    return out


class Network(RNGMixin):
    r"""
    Container for interacting populations, synaptic connections, and optional stimuli.

    A ``Network`` bundles:

    - **Populations**: named collections of axons/neurons (``Population`` instances).
    - **NetCons**: directed connections materialized as :class:`NetCon` objects that
      deliver events from pre- to post-synaptic compartments (including optional
      weights, delays, thresholds, and alternate pre variables such as conductance).
    - **NetStim**: an optional :class:`NetStim` source that injects spikes into the
      network (treated as a special "population" named ``netstim``).

    Connections are specified incrementally (e.g., via :meth:`connect_one_to_one`,
    :meth:`connect_dense`, :meth:`connect_prob`, :meth:`connect_prob_n`), then materialized when
    :meth:`build`/:meth:`initialize` is called. Running the network steps each
    population forward in time while advancing all synapses and optional NetStim.

    .. note::
        Device placement can be heterogeneous across populations and NetStim. NetCons
        are built on each post-synaptic population's device. Device changes are
        detected at :meth:`build` time and trigger rebuilds (as do ``dt`` changes or
        ``force_rebuild=True``). Call :meth:`build` (and typically
        :meth:`initialize`) after manual device moves to realign connectivity buffers.

    Parameters
    ----------
    populations : dict[str, Population]
        Mapping from population name to :class:`Population` instance. Populations
        are built, registered as attributes, and used as sources/targets for
        connectivity.
    netstim : NetStim, optional
        Optional spike generator attached under the name ``netstim``.
    seed : int, optional
        Seed for network-level RNG used in stochastic wiring utilities.
    """

    def __init__(self, populations: Dict[str, Population], netstim=None, seed=None):
        if any(pop.is_batched() for pop in populations.values()):
            raise ValueError(
                "Batched populations are not supported. Implement your networks with unbatched populations and then call .batch(batch_size)."
            )
        if netstim is not None and not isinstance(netstim, NetStim):
            raise TypeError("netstim must be an instance of NetStim or None.")
        super(Network, self).__init__(seed=seed)
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

        self.is_batched = False

        torch._dynamo.reset()

        with torch.set_grad_enabled(True):
            if self.jit:
                self._step_train = torch.compile(
                    step,
                    backend=self.backend,
                    fullgraph=self.fullgraph,
                    dynamic=self.dynamic,
                    mode=self.compile_mode,
                )
            else:
                self._step_train = step

        with torch.set_grad_enabled(False):
            if self.jit:
                self._step_eval = torch.compile(
                    step,
                    backend=self.backend,
                    fullgraph=self.fullgraph,
                    dynamic=self.dynamic,
                    mode=self.compile_mode,
                )
            else:
                self._step_eval = step

        # Network clock lives on CPU by default; move when needed.
        self.register_buffer(
            "t", torch.tensor(0.0, device=torch.device("cpu"), dtype=torch.float32)
        )

        if all(p.jit_in_network for p in populations.values()):
            self.use_compiled_advance_populations = True
        else:
            self.use_compiled_advance_populations = False

        # Track device signature to trigger rebuilds if placements change.
        self._device_sig = self._device_signature()

        self._state_cache = {}
        self._syn_cache = {}

        self.eval()

    def train(self, mode=True):
        """
        Switch populations and synapses into training mode.

        Parameters
        ----------
        mode : bool, optional
            If True, enable training mode (grad-enabled stepping). If False,
            disable grads for faster inference. Default is True.

        Returns
        -------
        Network
            Self, for chaining.
        """
        for pop in self.populations.values():
            pop.train(mode)
        for syn in self.synapses.values():
            syn.train(mode)
        self.training = mode
        self._step = self._step_train
        return self

    def train_(self, mode=True):
        """In-place variant of :meth:`train` that returns ``None``."""
        self.train(mode)

    def eval(self):
        """
        Switch populations and synapses into evaluation mode.

        Disables gradients for stepping and sets the compiled stepping function
        to the eval variant.

        Returns
        -------
        Network
            Self, for chaining.
        """
        super(Network, self).eval()
        for pop in self.populations.values():
            pop.eval()
        for syn in self.synapses.values():
            syn.eval()
        self.training = False
        self._step = self._step_eval
        return self

    def eval_(self):
        """In-place variant of :meth:`eval` that returns ``None``."""
        self.eval()

    def devices(self):
        """
        Return a mapping of population name -> device to support heterogeneous placement.
        """
        return {name: pop.device() for name, pop in self.populations.items()}

    def dtypes(self):
        """
        Return a mapping of population name -> dtype to support heterogeneous placement.
        """
        return {name: pop.dtype() for name, pop in self.populations.items()}

    def _device_signature(self):
        sig = [(name, str(pop.device())) for name, pop in self.populations.items()]
        if self.netstim is not None:
            sig.append(("netstim", str(self.netstim.device())))
        return tuple(sorted(sig))

    def device(self):
        """
        Legacy aggregate device (first population). For multi-device setups, prefer devices().
        """
        return next(iter(self.populations.values())).device()

    def dtype(self):
        """
        Legacy aggregate dtype (first population). For multi-device setups, prefer dtypes().
        """
        return next(iter(self.populations.values())).dtype()

    def clear_synapses(self):
        """
        Remove all queued synapse specs and built NetCon modules.

        Use when re-wiring the network before calling :meth:`build` again.
        """
        self.synapse_spec = {}
        self.synapses.clear()
        self.built = False

    def _normalize_endpoint(self, source, target):
        """Normalize full populations/devices to slice-like endpoints."""
        if isinstance(source, Population) or isinstance(source, NetStim):
            source = source[:]
        if isinstance(target, Population):
            target = target[:]
        return source, target, source.model, target.model

    def _flat_selection(self, selection):
        """Return selected population-flat indices for a Population/NetStim slice."""
        model = selection.model
        return to_flat_idx_torch(
            model.shape,
            selection.index,
            model.device(),
        ).to(torch.long)

    def _to_synapse_local_post_idx(self, target_model, post_flat, synapse):
        """
        Convert target population-flat indices to target synapse-local indices.

        Connectivity rules operate in population-flat coordinates so that
        autapse/multapse checks are well-defined. NetCon construction expects
        target indices local to the target-side synapse mechanism, so conversion
        happens only after the final edge set has been generated.
        """
        post_flat = post_flat.to(device=target_model.device(), dtype=torch.long)
        local = get_local_index(target_model, synapse, post_flat)

        if torch.any(local < 0):
            bad = post_flat[local < 0][:10].detach().cpu().tolist()
            raise ValueError(
                f"Target population '{target_model.name}' does not have "
                f"the synapse '{synapse}' at target locations including {bad}."
            )

        return local

    def _all_to_all_edges(self, pre_pool, post_pool):
        """Return all directed edges from pre_pool to post_pool."""
        if pre_pool.numel() == 0 or post_pool.numel() == 0:
            return pre_pool[:0], post_pool[:0]

        n_pre = pre_pool.numel()
        n_post = post_pool.numel()
        pre = pre_pool.repeat_interleave(n_post)
        post = post_pool.repeat(n_pre)
        return pre, post

    def _drop_autapses(self, pre_idx, post_idx, *, same_population: bool):
        """Remove self-connections in population-flat coordinates."""
        if not same_population or pre_idx.numel() == 0:
            return pre_idx, post_idx
        mask = pre_idx != post_idx
        return pre_idx[mask], post_idx[mask]

    def _drop_multapses(self, pre_idx, post_idx, *, num_targets_total: int):
        """
        Remove duplicate directed source-target pairs within one connection call.

        This deliberately mirrors NEST's per-call interpretation: it does not
        scan previously queued calls for duplicates.
        """
        if pre_idx.numel() <= 1:
            return pre_idx, post_idx

        device = pre_idx.device
        post_on_pre_device = post_idx.to(device=device, dtype=torch.long)
        key = pre_idx.to(torch.long) * int(num_targets_total) + post_on_pre_device

        unique_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
        positions = torch.arange(key.numel(), device=device, dtype=torch.long)
        first = torch.full(
            (unique_key.numel(),),
            key.numel(),
            device=device,
            dtype=torch.long,
        )
        first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)

        keep = first.sort().values
        return pre_idx[keep], post_on_pre_device[keep]

    def _draw_from_pool(self, pool, k: int, *, replace: bool, device):
        """Sample k entries from pool, with or without replacement."""
        if k < 0:
            raise ValueError("Degree/count must be non-negative.")
        if k == 0:
            return pool[:0]
        if pool.numel() == 0:
            raise ValueError("Cannot draw from an empty candidate pool.")

        if replace:
            idx = torch.randint(
                pool.numel(),
                (k,),
                device=device,
                generator=self._rng(device),
            )
            return pool[idx]

        if k > pool.numel():
            raise ValueError(
                f"Requested {k} unique connections, but only "
                f"{pool.numel()} candidates are available. "
                "Use allow_multapses=True or reduce the requested degree/count."
            )

        idx = torch.randperm(
            pool.numel(),
            device=device,
            generator=self._rng(device),
        )[:k]
        return pool[idx]

    def _edges_for_rule(
        self,
        *,
        rule: ConnRule,
        spec: dict,
        source_model,
        target_model,
        pre_pool,
        post_pool,
        allow_autapses: bool,
        allow_multapses: bool,
    ):
        """
        Generate edges in population-flat coordinates for one connection call.
        """
        device = source_model.device()
        pre_pool = pre_pool.to(device=device, dtype=torch.long)
        post_pool = post_pool.to(device=device, dtype=torch.long)
        same_population = source_model is target_model
        rule = str(rule).replace("-", "_")

        if rule == "one_to_one":
            if pre_pool.numel() != post_pool.numel():
                raise ValueError(
                    "one_to_one requires source and target selections to have "
                    "the same number of elements."
                )
            pre_idx, post_idx = pre_pool, post_pool

            if not allow_autapses:
                pre_idx, post_idx = self._drop_autapses(
                    pre_idx,
                    post_idx,
                    same_population=same_population,
                )
            if not allow_multapses:
                pre_idx, post_idx = self._drop_multapses(
                    pre_idx,
                    post_idx,
                    num_targets_total=target_model.v.numel(),
                )
            return pre_idx, post_idx

        if rule in ("all_to_all", "dense"):
            pre_idx, post_idx = self._all_to_all_edges(pre_pool, post_pool)

            if not allow_autapses:
                pre_idx, post_idx = self._drop_autapses(
                    pre_idx,
                    post_idx,
                    same_population=same_population,
                )
            if not allow_multapses:
                pre_idx, post_idx = self._drop_multapses(
                    pre_idx,
                    post_idx,
                    num_targets_total=target_model.v.numel(),
                )
            return pre_idx, post_idx

        if rule in ("pairwise_bernoulli", "bernoulli"):
            p = float(_require(spec, "p", "prob", "probability"))
            if not 0.0 <= p <= 1.0:
                raise ValueError("pairwise_bernoulli requires 0 <= p <= 1.")

            pre_idx, post_idx = self._all_to_all_edges(pre_pool, post_pool)
            if not allow_autapses:
                pre_idx, post_idx = self._drop_autapses(
                    pre_idx,
                    post_idx,
                    same_population=same_population,
                )
            if pre_idx.numel() == 0 or p == 0.0:
                return pre_idx[:0], post_idx[:0]
            if p < 1.0:
                mask = (
                    torch.rand(
                        pre_idx.numel(),
                        device=device,
                        generator=self._rng(device),
                    )
                    < p
                )
                pre_idx, post_idx = pre_idx[mask], post_idx[mask]

            # Pairwise Bernoulli visits every pair at most once. This branch is
            # only needed when user-supplied index selections contain repeats.
            if not allow_multapses:
                pre_idx, post_idx = self._drop_multapses(
                    pre_idx,
                    post_idx,
                    num_targets_total=target_model.v.numel(),
                )
            return pre_idx, post_idx

        if rule == "pairwise_poisson":
            lam = float(
                _require(
                    spec,
                    "pairwise_avg_num_conns",
                    "lambda",
                    "lam",
                    "mean",
                )
            )
            if lam < 0.0:
                raise ValueError("pairwise_poisson requires a non-negative mean.")
            if not allow_multapses:
                raise ValueError(
                    "pairwise_poisson can create multiple connections per "
                    "source-target pair, so allow_multapses=False is invalid."
                )

            pre_idx, post_idx = self._all_to_all_edges(pre_pool, post_pool)
            if not allow_autapses:
                pre_idx, post_idx = self._drop_autapses(
                    pre_idx,
                    post_idx,
                    same_population=same_population,
                )
            if pre_idx.numel() == 0 or lam == 0.0:
                return pre_idx[:0], post_idx[:0]

            rate = torch.full(
                (pre_idx.numel(),),
                lam,
                device=device,
                dtype=torch.float32,
            )
            counts = torch.poisson(rate, generator=self._rng(device)).to(torch.long)
            if counts.sum() == 0:
                return pre_idx[:0], post_idx[:0]

            edge_idx = torch.repeat_interleave(
                torch.arange(pre_idx.numel(), device=device),
                counts,
            )
            return pre_idx[edge_idx], post_idx[edge_idx]

        if rule in ("fixed_total_number", "fixed_total"):
            n = int(_require(spec, "N", "n"))
            if n <= 0:
                return pre_pool[:0], post_pool[:0]

            candidates_pre, candidates_post = self._all_to_all_edges(
                pre_pool, post_pool
            )
            if not allow_autapses:
                candidates_pre, candidates_post = self._drop_autapses(
                    candidates_pre,
                    candidates_post,
                    same_population=same_population,
                )

            m = candidates_pre.numel()
            if m == 0:
                raise ValueError("No candidate edges are available.")

            if allow_multapses:
                chosen = torch.randint(
                    m,
                    (n,),
                    device=device,
                    generator=self._rng(device),
                )
            else:
                if n > m:
                    raise ValueError(
                        f"Requested {n} unique connections, but only {m} "
                        "candidate edges are available after autapse filtering."
                    )
                chosen = torch.randperm(
                    m,
                    device=device,
                    generator=self._rng(device),
                )[:n]

            return candidates_pre[chosen], candidates_post[chosen]

        if rule == "fixed_indegree":
            indegree = int(_require(spec, "indegree", "in_degree", "K", "N"))
            if indegree < 0:
                raise ValueError("indegree must be non-negative.")
            if indegree == 0 or post_pool.numel() == 0:
                return pre_pool[:0], post_pool[:0]

            pre_chunks = []
            post_chunks = []
            for target in post_pool:
                pool = pre_pool
                if not allow_autapses and same_population:
                    pool = pool[pool != target]

                chosen_pre = self._draw_from_pool(
                    pool,
                    indegree,
                    replace=allow_multapses,
                    device=device,
                )
                pre_chunks.append(chosen_pre)
                post_chunks.append(target.expand(chosen_pre.numel()))

            pre_idx = torch.cat(pre_chunks) if pre_chunks else pre_pool[:0]
            post_idx = torch.cat(post_chunks) if post_chunks else post_pool[:0]

            if not allow_multapses:
                pre_idx, post_idx = self._drop_multapses(
                    pre_idx,
                    post_idx,
                    num_targets_total=target_model.v.numel(),
                )
            return pre_idx, post_idx

        if rule == "fixed_outdegree":
            outdegree = int(_require(spec, "outdegree", "out_degree", "K", "N"))
            if outdegree < 0:
                raise ValueError("outdegree must be non-negative.")
            if outdegree == 0 or pre_pool.numel() == 0:
                return pre_pool[:0], post_pool[:0]

            pre_chunks = []
            post_chunks = []
            for source in pre_pool:
                pool = post_pool
                if not allow_autapses and same_population:
                    pool = pool[pool != source]

                chosen_post = self._draw_from_pool(
                    pool,
                    outdegree,
                    replace=allow_multapses,
                    device=device,
                )
                pre_chunks.append(source.expand(chosen_post.numel()))
                post_chunks.append(chosen_post)

            pre_idx = torch.cat(pre_chunks) if pre_chunks else pre_pool[:0]
            post_idx = torch.cat(post_chunks) if post_chunks else post_pool[:0]

            if not allow_multapses:
                pre_idx, post_idx = self._drop_multapses(
                    pre_idx,
                    post_idx,
                    num_targets_total=target_model.v.numel(),
                )
            return pre_idx, post_idx

        raise ValueError(f"Unsupported connection rule: {rule!r}")

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
        pre_var=None,
    ):
        """
        Append a finalized connection spec.

        `source_idx` is source population-flat. `target_idx` is target
        synapse-local. Higher-level connection methods are responsible for
        generating edges and applying autapse/multapse policy before calling
        this method. The `allow_autapses` parameter remains accepted for older
        internal callers but is intentionally not used here.
        """
        if source_idx.numel() == 0:
            return

        n_threshold = check_weight_shape(threshold, source_idx)
        n_weight = check_weight_shape(weight, source_idx)
        n_delay = check_weight_shape(delay, source_idx)

        self.synapse_spec.setdefault(
            (source_pop.name, target_pop.name, synapse, pre_var), []
        ).append(
            (
                source_idx,
                target_idx,
                threshold,
                n_threshold,
                to_param(weight, positive=True),
                n_weight,
                to_param(delay, positive=True),
                n_delay,
            )
        )

    def connect(
        self,
        source,
        target,
        synapse,
        conn_spec=None,
        *,
        threshold=0.0,
        weight=1.0,
        delay=0.0,
        pre_var=None,
        auto_expand=False,
        allow_autapses: Optional[bool] = None,
        allow_multapses: Optional[bool] = None,
    ):
        """
        Connect source to target using a NEST-style connectivity specification.

        Examples
        --------
        >>> net.connect(pre, post, post.mech.syn, "all_to_all")
        >>> net.connect(pre, post, post.mech.syn,
        ...             {"rule": "pairwise_bernoulli", "p": 0.1,
        ...              "allow_autapses": False, "allow_multapses": False})
        >>> net.connect(pre, post, post.mech.syn,
        ...             {"rule": "fixed_indegree", "indegree": 20})

        Supported rules are ``one_to_one``, ``all_to_all``,
        ``pairwise_bernoulli``, ``pairwise_poisson``,
        ``fixed_total_number``, ``fixed_indegree``, and ``fixed_outdegree``.
        For this NEST-style entry point, ``allow_autapses`` and
        ``allow_multapses`` default to ``True`` unless specified in
        ``conn_spec`` or as keyword arguments.
        """
        source, target, source_model, target_model = self._normalize_endpoint(
            source,
            target,
        )

        if conn_spec is None:
            spec = {"rule": "all_to_all"}
        elif isinstance(conn_spec, str):
            spec = {"rule": conn_spec}
        else:
            spec = dict(conn_spec)

        rule = spec.pop("rule", "all_to_all")

        if allow_autapses is None:
            allow_autapses = bool(spec.pop("allow_autapses", True))
        else:
            spec.pop("allow_autapses", None)

        if allow_multapses is None:
            allow_multapses = bool(spec.pop("allow_multapses", True))
        else:
            spec.pop("allow_multapses", None)

        if threshold is None:
            threshold = torch.nan

        pre_pool = self._flat_selection(source)
        post_pool = self._flat_selection(target)

        # Preserve the previous contract: all explicitly selected target
        # locations must host the requested synapse, even if a stochastic rule
        # would later sample only a subset of them.
        self._to_synapse_local_post_idx(target_model, post_pool, synapse)

        pre_idx, post_flat = self._edges_for_rule(
            rule=rule,
            spec=spec,
            source_model=source_model,
            target_model=target_model,
            pre_pool=pre_pool,
            post_pool=post_pool,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
        )

        if pre_idx.numel() == 0:
            return

        post_idx = self._to_synapse_local_post_idx(
            target_model,
            post_flat.to(target_model.device()),
            synapse,
        )

        if auto_expand:
            n_connections = pre_idx.numel()
            threshold = expand(threshold, n_connections)
            weight = expand(weight, n_connections)
            delay = expand(delay, n_connections)

        self._connect(
            source_model,
            pre_idx,
            target_model,
            post_idx,
            synapse,
            threshold=threshold,
            weight=weight,
            delay=delay,
            pre_var=pre_var,
        )

    def connect_one_to_one(
        self,
        source,
        target,
        synapse,
        threshold=0.0,
        weight=1.0,
        delay=0.0,
        pre_var=None,
        allow_autapses=False,
        allow_multapses=False,
    ):
        """
        Connect source to target one-to-one.

        This is a backward-compatible wrapper around ``connect(...,
        conn_spec={"rule": "one_to_one"})``. Unlike the general NEST-style
        ``connect`` entry point, the legacy wrapper keeps autapses disabled by
        default.
        """
        return self.connect(
            source,
            target,
            synapse,
            conn_spec={"rule": "one_to_one"},
            threshold=threshold,
            weight=weight,
            delay=delay,
            pre_var=pre_var,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
        )

    def connect_dense(
        self,
        source,
        target,
        synapse,
        threshold=0.0,
        weight=1.0,
        delay=0.0,
        pre_var=None,
        auto_expand=False,
        allow_autapses=False,
        allow_multapses=False,
    ):
        """
        Connect source to target densely, i.e. all-to-all between selections.

        Backward-compatible wrapper around ``connect(...,
        conn_spec={"rule": "all_to_all"})``.
        """
        return self.connect(
            source,
            target,
            synapse,
            conn_spec={"rule": "all_to_all"},
            threshold=threshold,
            weight=weight,
            delay=delay,
            pre_var=pre_var,
            auto_expand=auto_expand,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
        )

    def connect_prob(
        self,
        source,
        target,
        synapse,
        prob: float,
        threshold=0.0,
        weight=1.0,
        delay=0.0,
        pre_var=None,
        auto_expand=False,
        allow_autapses=False,
        allow_multapses=False,
        strategy="bernoulli",
    ):
        """
        Connect source to target probabilistically.

        ``strategy='bernoulli'`` maps to NEST ``pairwise_bernoulli`` and uses
        ``prob`` as the pairwise connection probability. ``strategy='poisson'``
        or ``'pairwise_poisson'`` maps to NEST ``pairwise_poisson`` and uses
        ``prob`` as ``pairwise_avg_num_conns``; for that rule,
        ``allow_multapses`` must be ``True``.
        """
        strategy = str(strategy).replace("-", "_")
        if strategy in ("bernoulli", "pairwise_bernoulli"):
            conn_spec = {"rule": "pairwise_bernoulli", "p": prob}
        elif strategy in ("poisson", "pairwise_poisson"):
            conn_spec = {
                "rule": "pairwise_poisson",
                "pairwise_avg_num_conns": prob,
            }
        else:
            raise ValueError(
                "strategy must be 'bernoulli', 'pairwise_bernoulli', "
                "'poisson', or 'pairwise_poisson'."
            )

        return self.connect(
            source,
            target,
            synapse,
            conn_spec=conn_spec,
            threshold=threshold,
            weight=weight,
            delay=delay,
            pre_var=pre_var,
            auto_expand=auto_expand,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
        )

    def connect_prob_n(
        self,
        source,
        target,
        synapse,
        n: int,
        threshold=0.0,
        weight=1.0,
        delay=0.0,
        pre_var=None,
        auto_expand=False,
        allow_autapses=False,
        allow_multapses=False,
    ):
        """
        Connect exactly ``n`` random pre-post pairs.

        This maps to NEST ``fixed_total_number``. With
        ``allow_multapses=True``, pairs are drawn with replacement. With
        ``allow_multapses=False`` the sampled source-target pairs are unique.
        """
        return self.connect(
            source,
            target,
            synapse,
            conn_spec={"rule": "fixed_total_number", "N": n},
            threshold=threshold,
            weight=weight,
            delay=delay,
            pre_var=pre_var,
            auto_expand=auto_expand,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
        )

    def build_synapses(self, dt, max_delay_ms=None):
        """
        Materialize queued connection specs into :class:`NetCon` modules.

        Parameters
        ----------
        dt : float
            Simulation timestep (ms) for delivery buffers.
        max_delay_ms : float, optional
            Optional ceiling on allowable synaptic delay; passed to NetCon.
        """
        for (pre_name, post_name, synapse, pre_var), specs in self.synapse_spec.items():
            pre_var = (
                pre_var
                if pre_var is not None
                else ("v" if pre_name != "netstim" else "spike")
            )
            pre = getattr(self, pre_name)
            post = self.populations[post_name]
            post_device = post.device()
            post_dtype = post.dtype()
            pre_idx = torch.cat([s[0] for s in specs])
            post_idx = torch.cat([s[1] for s in specs])
            thresholds = torch.cat([expand(s[2], s[3]) for s in specs])
            weights = make_weight([s[4] for s in specs], [s[5] for s in specs])
            delay = make_weight([s[6] for s in specs], [s[7] for s in specs])

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
                pre_var=pre_var,
                max_delay=max_delay_ms,
            ).to(device=post_device, dtype=post_dtype)

            syn.setreference("t", lambda: self.t)

            if self.training:
                syn.train()
            else:
                syn.eval()

            self.synapses[
                f"{pre_name}:{pre_var.replace('.', '_')}->{post_name}:{synapse.name}"
            ] = syn

    def build(self, dt, max_delay_ms=None, force_rebuild=False):
        """
        Build synaptic modules for the current wiring spec.

        Parameters
        ----------
        dt : float
            Simulation timestep (ms) used for NetCon buffers.
        max_delay_ms : float, optional
            Maximum delay allowed for NetCons. Default is None (no cap).
        force_rebuild : bool, optional
            If True, rebuild even if dt has not changed. Default is False.

        Returns
        -------
        Network
            Self, for chaining.
        """
        current_sig = self._device_signature()
        devices_changed = current_sig != getattr(self, "_device_sig", None)

        if not self.built or self.dt != dt or force_rebuild or devices_changed:
            torch._dynamo.reset()
            self.dt = dt
            self.build_synapses(dt, max_delay_ms=max_delay_ms)
            self.built = True
            self._device_sig = self._device_signature()
        return self

    def initialize(
        self,
        dt: float,
        reinit_weights: bool = True,
        reinit_delays: bool = True,
        t=0.0,
        max_delay_ms=None,
        force_rebuild: bool = False,
    ):
        """
        Initialize populations, synapses, and optional NetStim for simulation.

        Parameters
        ----------
        dt : float
            Simulation timestep (ms).
        reinit_weights : bool, optional
            Re-sample or reset synaptic weights. Default is True.
        reinit_delays : bool, optional
            Re-sample or reset synaptic delays. Default is True.
        t : float, optional
            Starting simulation time (ms). Default is 0.0.
        max_delay_ms : float, optional
            Maximum allowed synaptic delay. Default is None.
        force_rebuild : bool, optional
            If True, rebuild NetCons even if already built. Default is False.

        Returns
        -------
        Network
            Self, for chaining.
        """
        self.build(dt, max_delay_ms=max_delay_ms, force_rebuild=force_rebuild)
        self.t = self.t.detach()
        self.t.fill_(t)
        for pop in self.populations.values():
            pop.t = pop.t.detach()
            pop.t.fill_(t)
        if self._state_cache:
            self.initialize_pops_from_state_cache()
            self.initialize_synapses_from_state_cache()
            clear_deliveries = False
        else:
            clear_deliveries = True

        dt_f = float(dt)
        for pop in self.populations.values():
            if not self._state_cache:
                pop.initialize()
            dt_pop = torch.tensor(dt_f, device=pop.device(), dtype=pop.dtype())
            pop.integrator._initialize(pop, dt_pop)
            pop.intra = pop.build_intra()
        self.init_synapses(
            reinit_weights=reinit_weights,
            reinit_delays=reinit_delays,
            clear_deliveries=clear_deliveries,
        )
        if self.netstim is not None:
            self.netstim.initialize()
            self.netstim.detach()
        return self

    def initialize_pops_from_state_cache(self):
        for name, pop in self.populations.items():
            pop.load_state_dict(self._state_cache[name])
            pop.detach()

    def initialize_synapses_from_state_cache(self):
        for name, syn in self.synapses.items():
            old_dt, has_spiked, is_spiking, delivery_buffer = self._syn_cache[name]
            n_limit = syn.delivery_buffer.shape[0]
            delivery_buffer = dilate(
                delivery_buffer, float(old_dt), float(self.dt), n_limit=n_limit
            )
            if not syn.skip_thresholding:
                syn.has_spiked = (has_spiked).detach()
                syn.is_spiking = (is_spiking).detach()
            syn.delivery_buffer = delivery_buffer.clone().detach()

    def initialize_(self, *args, **kwargs):
        """In-place variant of :meth:`initialize` that returns ``None``."""
        self.initialize(*args, **kwargs)

    def init_synapses(
        self,
        reinit_weights: bool = True,
        reinit_delays: bool = True,
        clear_deliveries: bool = True,
    ):
        """
        Initialize all built synapses (NetCons).

        Parameters
        ----------
        reinit_weights : bool, optional
            If True, re-sample/reset weight parameters. Default is True.
        reinit_delays : bool, optional
            If True, re-sample/reset delay parameters. Default is True.
        clear_deliveries : bool, optional
            If True, zero the delivery buffers. Default is True.
        """
        for syn in self.synapses.values():
            syn.initialize(
                reinit_weights=reinit_weights,
                reinit_delays=reinit_delays,
                clear_deliveries=clear_deliveries,
            )

    def run(self, tstop, extra=None, callbacks=None, progressbar=False):
        """
        Advance the network for a fixed duration.

        Parameters
        ----------
        tstop : float
            Total simulation time (ms) to advance from current ``self.t``.
        extra : dict[str, tuple[torch.Tensor, object]], optional
            Optional mapping of population name to extracellular stimulus tuple
            ``(v, t)`` where ``t`` is assembled against the current time; values
            are moved to the network device/dtype.
        callbacks : list[Callback], optional
            Callbacks invoked each step (wrapped in :class:`CallbackList`).
        progressbar : bool or tqdm.tqdm, optional
            If truthy, show a progress bar (auto-created if True). Default False.

        Returns
        -------
        None
        """
        dt_f = float(self.dt)
        dt_t = torch.tensor(self.dt, device=self.t.device, dtype=self.t.dtype)

        ctx = nullcontext() if self.training else torch.no_grad()

        intra = {}
        for n, p in self.populations.items():
            if p.intra is not None:
                intra[n] = p.intra

        extra = extra if extra is not None else {}
        extra_prepped = {}
        for n, (v, t) in extra.items():
            pop = self.populations[n]
            dev, dtp = pop.device(), pop.dtype()
            v_dev = v.to(device=dev, dtype=dtp)
            t_dev = t.to(device=dev, dtype=dtp)
            t0 = self.t.to(device=dev, dtype=dtp)
            dt_pop = torch.tensor(self.dt, device=dev, dtype=dtp)
            t1 = t0 + torch.tensor(tstop, device=dev, dtype=dtp)
            extra_prepped[n] = (v_dev, t_dev.assemble(t0, t1, dt_pop))
        extra = extra_prepped

        with_intra = bool(intra)
        with_extra = bool(extra)
        tstart = self.t.item()

        with ctx:
            n_steps = int(tstop / self.dt)

            if with_intra:
                intra = {
                    n: (intra_, *self.populations[n].prep_intra(intra_, n_steps, dt_f))
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
                extra_c = {}

                if with_intra:
                    intra_c = prepare_intra(intra_c, intra, local_ind)

                if with_extra:
                    extra_c = prepare_extra(extra, local_ind)

                self._step(
                    self.populations,
                    self.synapses,
                    self.netstim,
                    self.t,
                    dt_f,
                    extra=extra_c,
                    intra=intra_c,
                    compiled_advance_population=self.use_compiled_advance_populations,
                )
                self.t = self.t + dt_t
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
        """
        Create a batched version of the network by tiling populations n times.

        Parameters
        ----------
        n : int
            Batch size (number of replicas).
        include_netstim : bool, optional
            If False, do not batch NetStim even if present. Default True.

        Returns
        -------
        Network
            Self, with populations and NetCons batched.
        """
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
            source_name, target_name, synapse, pre_var = k
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
                    pre_var=pre_var,
                )
        self.built = False
        self.is_batched = True
        gc.collect()
        return self

    def batch_(self, n, include_netstim=True):
        """In-place variant of :meth:`batch` that returns ``None``."""
        self.batch(n, include_netstim=include_netstim)

    def concat(self, **kwargs):
        """
        Concatenates populations to solve with fewer kernel launches.
        All connections are preserved.

        Parameters
        ----------
        **kwargs : dict
            Populations to concatenate, where keys are the new population names
            and values are lists of population names to concatenate. If no
            populations are specified, all populations will be concatenated into
            'all_populations'.

        Returns
        -------
        Network
            A new network with the concatenated populations.

        Examples
        --------
        >>> net = Network(...)
        >>> print(list(net.populations.keys()))
        ['exc1', 'exc2', 'inh']
        >>> net0 = net.concat()
        >>> print(list(net0.populations.keys()))
        ['all_populations']
        >>> net1 = net.concat(all=['exc1', 'exc2', 'inh'])
        >>> print(list(net1.populations.keys()))
        ['all']
        >>> net2 = net.concat(exc=['exc1', 'exc2'])
        >>> print(list(net2.populations.keys()))
        ['exc', 'inh']
        """
        if not kwargs:
            return self._concat("all_populations")
        net = self
        for name, pops_to_concatenate in kwargs.items():
            net = net._concat(name, pops_to_concatenate)
        return net

    def _concat(self, name, pops_to_concatenate=None, threads=16):
        if pops_to_concatenate is None:
            pops_to_concatenate = []
        already_used = [
            n for n in self.populations.keys() if n not in pops_to_concatenate
        ]
        if name in already_used:
            raise ValueError(f"Population '{name}' is already in use.")
        if self.is_batched:
            raise ValueError("Cannot concatenate populations in a batched network.")
        if not pops_to_concatenate:
            pops_to_concatenate = list(self.populations.keys())

        concat_pops = {n: self.populations[n] for n in pops_to_concatenate}
        p_type = type(self.populations[pops_to_concatenate[0]])
        assert all(type(self.populations[n]) is p_type for n in pops_to_concatenate), (
            "All populations must be of the same type."
        )
        celsius = _last_celsius(concat_pops).item()
        concatenated = concat_models(concat_pops, threads=threads, celsius=celsius)
        new_populations = {
            n: p for n, p in self.populations.items() if n not in pops_to_concatenate
        }
        new_populations[name] = concatenated
        new_net = Network(new_populations, netstim=self.netstim)

        all_indices = indices(concat_pops)
        all_indices = {n: i.flatten() for n, i in zip(pops_to_concatenate, all_indices)}

        # now reapply connections
        for k, v in self.synapse_spec.items():
            source_name, target_name, synapse, pre_var = k
            if (source_pop := new_net.populations.get(name)) is None:
                source_pop = getattr(new_net, source_name)
            if (target_pop := new_net.populations.get(name)) is None:
                target_pop = getattr(new_net, target_name)
            synapse = getattr(target_pop.mech, synapse.name)

            for data in v:
                source_idx, target_idx = data[0], data[1]
                threshold, weight, delay = data[2], data[4], data[6]

                if source_name in pops_to_concatenate:
                    source_idx = all_indices[source_name][source_idx]
                if target_name in pops_to_concatenate:
                    target_idx = all_indices[target_name][target_idx]

                new_net._connect(
                    source_pop,
                    source_idx,
                    target_pop,
                    target_idx,
                    synapse,
                    threshold,
                    weight,
                    delay,
                    pre_var=pre_var,
                )

        return new_net

    def steady_state(self, tstop=1000, dt=0.025, progressbar=False):
        """
        Run the network until it reaches a steady state.

        Parameters
        ----------
        tstop : float, optional
            Total time to run the network in ms. Default is 1000 ms.
        dt : float, optional
            Time step in ms. Default is 0.025 ms.

        Returns
        -------
        Network
            The network after running to steady state.
        """
        self.clear_state_cache()
        was_training = self.training
        with torch.no_grad():
            if self.netstim is not None:
                self.netstim._prep_start_for_steady_state(tstop)
            self.eval()
            self.initialize(dt, t=-tstop, force_rebuild=True)
            self.run(tstop, progressbar=progressbar)
            if self.netstim is not None:
                self.netstim._reset_start_times()
            self.cache_state()
        if was_training:
            self.train()
        return self

    def cache_state(self):
        self._state_cache.clear()
        self._syn_cache.clear()
        for name, pop in self.populations.items():
            self._state_cache[name] = pop.state_dict()
        for name, syn in self.synapses.items():
            self._syn_cache[name] = (
                syn.dt,
                syn.has_spiked.clone(),
                syn.is_spiking.clone(),
                torch.roll(
                    syn.delivery_buffer, -syn.current_time_step.item(), dims=0
                ).clone(),
            )

    def clear_state_cache(self):
        self._state_cache.clear()
        self._syn_cache.clear()

    def load_state_cache(self, state_cache=None, syn_cache=None):
        if state_cache is not None:
            self._state_cache = state_cache
        if syn_cache is not None:
            self._syn_cache = syn_cache

    def set_synaptic_diff_config(self, **kwargs):
        for syn in self.synapses.values():
            syn.set_diff_config(**kwargs)

    # load utilities
    def load(self, state_dict):
        """
        Load model weights from a state dictionary.

        This method supports loading weights from:
        1. A file path as a string
        2. An actual state dictionary object

        The loaded weights are matched to the model's current state dict structure
        and only compatible weights are loaded.

        Parameters
        ----------
        state_dict : str or dict
            Can be one of:
            - A file path to a saved model state
            - A state dictionary object

        Returns
        -------
        self
            The model instance with loaded weights
        """
        if isinstance(state_dict, str):
            state_dict = torch.load(
                state_dict, map_location=self.device(), weights_only=True
            )
        matched, _ = _match_state_dict(self.state_dict(), state_dict)
        self.load_state_dict(matched, strict=False)
        return self

    # utilities
    def concatenated_weights(self):
        """
        Returns the concatenated weights of all synapses.
        """
        return torch.cat([syn.w.flatten() for syn in self.synapses.values()])

    def weights(self):
        """
        Returns the weights of all synapses as a dictionary.
        """
        return {name: syn.w for name, syn in self.synapses.items()}

    def weight_modules(self):
        """
        Returns the weight modules of all synapses as a dictionary.
        """
        return {name: syn.weight for name, syn in self.synapses.items()}

    # population utilities
    def delete_injections(self):
        """
        Deletes all current injections from all populations in the network.
        """
        for pop in self.populations.values():
            pop.delete_injections()

    # checkpoint utilities

    def populations_state_dict_for_checkpoint(self):
        """
        Returns a state dict of all populations suitable for checkpointing.
        """
        return {
            name: pop.state_dict_for_checkpoint()
            for name, pop in self.populations.items()
        }

    def netcons_state_dict_for_checkpoint(self):
        """
        Returns a state dict of all NetCons suitable for checkpointing.
        """
        return {
            name: syn.state_dict_for_checkpoint() for name, syn in self.synapses.items()
        }

    def netstim_state_dict_for_checkpoint(self):
        """
        Returns a state dict of the NetStim suitable for checkpointing.
        """
        if self.netstim is None:
            return None
        return self.netstim.state_dict_for_checkpoint()

    def state_dict_for_checkpoint(self):
        """
        Returns a state dict of the entire network suitable for checkpointing.
        """
        return {
            "populations": self.populations_state_dict_for_checkpoint(),
            "netcons": self.netcons_state_dict_for_checkpoint(),
            "netstim": self.netstim_state_dict_for_checkpoint(),
            "t": self.t,
        }

    def restore_dict_from_checkpoint(self, state_dict):
        """
        Restores the network state from a checkpoint state dict.
        """
        for name, pop_state in state_dict["populations"].items():
            self.populations[name].restore_dict_from_checkpoint(pop_state)
        for name, syn_state in state_dict["netcons"].items():
            self.synapses[name].restore_dict_from_checkpoint(syn_state)
        if state_dict["netstim"] is not None and self.netstim is not None:
            self.netstim.restore_dict_from_checkpoint(state_dict["netstim"])
        self.t = state_dict["t"]

    # -- checkpointed run --

    def longrun_checkpointed(
        self,
        tstop: float,
        chunklength: int,
        extra=None,
        callbacks=None,
        progressbar=False,
        *,
        safe_checkpoint: bool = False,
        restore_state_after_backward: bool = True,
        return_final_state: bool = False,
    ):
        """Run the network for a long horizon using activation checkpointing.

        This method mirrors :meth:`Population.longrun_checkpointed` but is
        specialized for :class:`Network`:

        * ``dt`` is **not** an argument here. The network time step is fixed by
          :meth:`initialize` / :meth:`build` (required for constructing
          :class:`NetCon` delay buffers).
        * Simulation state is checkpointed at chunk boundaries via
          :meth:`state_dict_for_checkpoint` / :meth:`restore_dict_from_checkpoint`.

        Parameters
        ----------
        tstop:
            Total simulated time (ms).
        chunklength:
            Number of time steps per checkpoint chunk.
        extra:
            Optional extracellular specification (same as :meth:`run`).
        callbacks:
            Optional list of callbacks.
        progressbar:
            If ``True``, displays a tqdm progress bar over chunks.
        safe_checkpoint:
            If ``True``, clone checkpoint input tensors at each chunk boundary
            to guard against inadvertent in-place mutation.
        restore_state_after_backward:
            If ``True``, restores the forward final state after backward
            completes (useful because checkpointing replays forward during
            backward and mutates module state).
        return_final_state:
            If ``True``, return ``(loss, final_state_dict)``.

        Notes
        -----
        **Callback contract (restricted)**

        To remain replay-safe under activation checkpointing, callbacks should
        be pure functions of the model state. Hooks may optionally return a
        scalar tensor loss contribution. Any non-``None`` returns from:

        * ``post_step_hook(model)``
        * ``post_chunk_hook(model, t_chunk)``
        * ``post_loop_hook(model)``

        are summed and returned as the total loss.
        """

        if self.dt is None:
            raise RuntimeError(
                "Network.dt is None. Call net.initialize(dt=...) before longrun_checkpointed()."
            )
        if chunklength <= 0:
            raise ValueError("chunklength must be a positive integer")

        # Normalize callbacks.
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        for c in callbacks:
            c.dt = self.dt

        dt_f = float(self.dt)
        dt_t = torch.tensor(self.dt, device=self.t.device, dtype=self.t.dtype)

        # Build a global time grid (used only for chunking + callback timing).
        t_grid = torch.arange(
            self.t.double(),
            (self.t.double() + float(tstop)),
            dt_f,
            dtype=torch.double,
            device=self.t.device,
        ).to(dtype=self.t.dtype)

        if t_grid.numel() == 0:
            # Nothing to do.
            if return_final_state:
                return None, self.state_dict_for_checkpoint()
            return None

        n_chunks = int(math.ceil(len(t_grid) / chunklength))
        t_chunks = torch.tensor_split(t_grid, n_chunks)

        # Preprocess extra: move spatial fields to the target population devices.
        extra = extra if extra is not None else {}
        extra_prepped = {}
        if len(extra) > 0:
            for name, (v, tt) in extra.items():
                if name not in self.populations:
                    raise KeyError(f"extra specified for unknown population '{name}'")
                pop = self.populations[name]
                dev, dtp = pop.device(), pop.dtype()
                v_dev = v.to(device=dev, dtype=dtp)
                tt_dev = tt.to(device=dev, dtype=dtp)
                dt_pop = torch.tensor(self.dt, device=dev, dtype=dtp)
                extra_prepped[name] = (v_dev, tt_dev, dt_pop, dev, dtp)

        # Identify intra sources (prepared per chunk to bound memory).
        intra_sources = {
            name: pop.intra
            for name, pop in self.populations.items()
            if pop.intra is not None
        }

        def _as_loss_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x
            # Default device/dtype: first population.
            return torch.as_tensor(x, device=self.device(), dtype=self.dtype())

        def _add_loss(acc, x):
            x_t = _as_loss_tensor(x)
            if x_t is None:
                return acc, False
            if acc is None:
                return x_t, True
            return acc + x_t, True

        def _clone_checkpoint_state_dict(sd):
            """Clone all tensors in a nested checkpoint state dict.

            This is used when ``safe_checkpoint=True`` to ensure that tensors
            passed as checkpoint inputs are not mutated in-place during the
            chunk forward.
            """

            memo = {}

            def _clone_any(v):
                if isinstance(v, torch.Tensor):
                    k = id(v)
                    if k in memo:
                        return memo[k]
                    out = v.clone()
                    memo[k] = out
                    return out
                if isinstance(v, dict):
                    return {kk: _clone_any(vv) for kk, vv in v.items()}
                if isinstance(v, list):
                    return [_clone_any(vv) for vv in v]
                if isinstance(v, tuple):
                    return tuple(_clone_any(vv) for vv in v)
                return v

            return _clone_any(sd)

        def _copy_state_containers(sd):
            """Deep-copy container structure while preserving tensor identity."""

            def _copy_any(v):
                if isinstance(v, dict):
                    return {kk: _copy_any(vv) for kk, vv in v.items()}
                if isinstance(v, list):
                    return [_copy_any(vv) for vv in v]
                if isinstance(v, tuple):
                    return tuple(_copy_any(vv) for vv in v)
                return v

            return _copy_any(sd)

        total_loss = None
        saw_any_loss = False

        with torch.nn.utils.parametrize.cached():
            with torch.set_grad_enabled(self.training):
                pre_loop_hook(callbacks, self)

                # Snapshot boundary state after pre-loop hooks.
                state = self.state_dict_for_checkpoint()

                pbar = (
                    tqdm(total=n_chunks, desc=f"{t_grid[0].item():.1f} ms")
                    if progressbar
                    else None
                )

                # Local alias for speed
                _checkpoint = torch.utils.checkpoint.checkpoint

                for chunk_idx, t_chunk in enumerate(t_chunks):
                    if pbar is not None:
                        pbar.set_description(f"{t_chunk[0].item():.1f} ms")

                    def _run_chunk(state_in, t_chunk_local=t_chunk):
                        # Optional safety: clone checkpoint inputs to avoid
                        # in-place mutation of checkpoint input tensors.
                        state_local = state_in
                        if safe_checkpoint:
                            state_local = _clone_checkpoint_state_dict(state_in)

                        self.restore_dict_from_checkpoint(state_local)
                        pre_chunk_hook(callbacks, self, t_chunk_local)

                        # Prepare per-chunk intra specs (bounded to chunklength).
                        intra_chunk = {}
                        if len(intra_sources) > 0:
                            for name, intra_obj in intra_sources.items():
                                stims, indices = self.populations[name].prep_intra(
                                    intra_obj, len(t_chunk_local), dt_f
                                )
                                intra_chunk[name] = (intra_obj, stims, indices)

                        # Prepare per-chunk extracellular time series (bounded).
                        extra_ts = {}
                        if len(extra_prepped) > 0:
                            for name, (
                                v_dev,
                                tt_dev,
                                dt_pop,
                                dev,
                                dtp,
                            ) in extra_prepped.items():
                                t0 = self.t.to(device=dev, dtype=dtp)
                                t1 = t0 + (dt_pop * int(len(t_chunk_local)))
                                extra_ts[name] = (
                                    v_dev,
                                    tt_dev.assemble(t0, t1, dt_pop),
                                )

                        chunk_loss = None
                        saw_loss_local = False

                        for i_t in range(len(t_chunk_local)):
                            # Intra for this step.
                            intra_c = {}
                            if len(intra_chunk) > 0:
                                for name, (
                                    intra_obj,
                                    stims,
                                    indices,
                                ) in intra_chunk.items():
                                    s = [st[i_t] for st in stims]
                                    intra_c[name] = make_intra(intra_obj, s, indices)
                            # Extra for this step.
                            extra_c = {}
                            if len(extra_ts) > 0:
                                for name, (v_dev, ts) in extra_ts.items():
                                    extra_c[name] = v_dev * ts[i_t]

                            # Advance network dynamics.
                            self._step(
                                self.populations,
                                self.synapses,
                                self.netstim,
                                self.t,
                                dt_f,
                                extra=extra_c,
                                intra=intra_c,
                                compiled_advance_population=self.use_compiled_advance_populations,
                            )
                            self.t = self.t + dt_t

                            # Replay-safe loss aggregation via callback returns.
                            for cb in callbacks:
                                hook = getattr(cb, "post_step_hook", None)
                                if hook is not None:
                                    chunk_loss, saw = _add_loss(chunk_loss, hook(self))
                                    saw_loss_local = saw_loss_local or saw

                        for cb in callbacks:
                            hook = getattr(cb, "post_chunk_hook", None)
                            if hook is not None:
                                chunk_loss, saw = _add_loss(
                                    chunk_loss, hook(self, t_chunk_local)
                                )
                                saw_loss_local = saw_loss_local or saw

                        state_out = self.state_dict_for_checkpoint()

                        if chunk_loss is None:
                            chunk_loss = torch.zeros(
                                (), device=self.device(), dtype=self.dtype()
                            )
                        saw_loss_flag = torch.tensor(
                            1 if saw_loss_local else 0,
                            device=chunk_loss.device,
                            dtype=torch.int32,
                        )
                        return state_out, chunk_loss, saw_loss_flag

                    state, chunk_loss, saw_loss_flag = _checkpoint(
                        _run_chunk,
                        state,
                        use_reentrant=False,
                        determinism_check="none",
                    )

                    total_loss, _ = _add_loss(total_loss, chunk_loss)
                    saw_any_loss = saw_any_loss or bool(int(saw_loss_flag.item()))

                    if pbar is not None:
                        pbar.update(1)

                for cb in callbacks:
                    hook = getattr(cb, "post_loop_hook", None)
                    if hook is not None:
                        total_loss, saw = _add_loss(total_loss, hook(self))
                        saw_any_loss = saw_any_loss or saw

                if pbar is not None:
                    pbar.close()

        if not saw_any_loss:
            if return_final_state:
                return None, self.state_dict_for_checkpoint()
            return None

        final_state = None
        if restore_state_after_backward or return_final_state:
            final_state = self.state_dict_for_checkpoint()

        if (
            restore_state_after_backward
            and isinstance(total_loss, torch.Tensor)
            and total_loss.requires_grad
        ):
            final_state_for_hook = _copy_state_containers(final_state)

            def _queue_restore(grad, fs=final_state_for_hook):
                torch.autograd.Variable._execution_engine.queue_callback(
                    lambda: self.restore_dict_from_checkpoint(fs)
                )
                return grad

            total_loss.register_hook(_queue_restore)

        if return_final_state:
            return total_loss, final_state
        return total_loss


def prepare_intra(intra_c, intra, local_ind):
    """
    Prepares the intra-cellular data for the current step.
    """
    for n, (intra_, stims, indexes) in intra.items():
        s = [st[local_ind] for st in stims]
        intra_c[n] = make_intra(intra_, s, indexes)
    return intra_c


@torch.compile
def prepare_extra(extra, local_ind: int):
    """
    Prepares the voltage and time data for the current step.
    """
    return {n: v * t[local_ind] for n, (v, t) in extra.items()}


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
