import copy
import gc
import math
import os
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Dict, Literal, Optional

import torch
from tqdm.auto import tqdm

from dendra.helpers import (
    BACKEND,
    COMPILE_MODE,
    DYNAMIC,
    FULLGRAPH,
    JIT,
    JIT_NETWORK_OPS,
    JIT_NETWORK_SOLVES,
    compile_options_key,
    current_compile_options,
    jit_enabled_for_scope,
)

from ..callbacks import CallbackList
from ..core import (
    Population,
    _duration_step_budget,
    _load_compatible_state_dict_transactionally,
    _validate_time_scalar,
    make_intra,
)
from ..multi import concat_models, indices
from ..parametric import is_parametric, to_param
from ..rng import RNGMixin
from ..slice import SynapseSlots
from .netcon import ContinuousCon, NetCon
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


def to_flat_idx_mech(shape, mech, device):
    # 1. Create a grid of flat indices with the same shape as the input array.
    #    e.g., for a (2, 3) tensor, this becomes [[0, 1, 2], [3, 4, 5]]
    indices_grid = torch.arange(torch.prod(torch.tensor(shape)), device=device).view(
        shape
    )

    # 2. Apply the user's index to this grid. PyTorch's indexing logic
    #    will select the corresponding flat indices for us.
    selected_indices = mech.get(indices_grid)

    # 3. Flatten the result to get a 1D tensor of flat indices.
    return selected_indices.flatten()


def step_pop(integrator, model, dt, ve=None, intra=None):
    integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


compiled_step_pop = step_pop


def _named_items(collection):
    """Return an iterable of ``(name, component)`` pairs.

    ``Network`` caches device-ordered runtime schedules as tuples of named
    pairs, while the public ``step`` helper historically accepted dict-like
    containers.  This adapter keeps both call styles supported.
    """
    return collection.items() if hasattr(collection, "items") else collection


def _clone_checkpoint_state(value, memo=None):
    """Clone a nested runtime checkpoint without severing autograd history."""
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        value_id = id(value)
        if value_id not in memo:
            memo[value_id] = value.clone()
        return memo[value_id]
    if isinstance(value, dict):
        return {key: _clone_checkpoint_state(item, memo) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_state(item, memo) for item in value)
    if isinstance(value, list):
        return [_clone_checkpoint_state(item, memo) for item in value]
    return copy.deepcopy(value)


def _component_device(component):
    device = getattr(component, "device", None)
    if callable(device):
        try:
            return torch.device(device())
        except Exception:
            return torch.device("cpu")
    if device is not None:
        try:
            return torch.device(device)
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def _accelerator_first_step_items(collection):
    """Stable device-aware ordering for per-step component launches.

    CUDA and other accelerator operations are normally enqueued asynchronously
    from the host.  Launching accelerator work before CPU work lets CPU-bound
    model components run while accelerator kernels are already in flight.
    The sort is stable, so relative order is preserved within each device class.
    """
    items = tuple(_named_items(collection))

    def priority(item):
        dev = _component_device(item[1])
        return 1 if dev.type in ("cpu", "meta") else 0

    return tuple(sorted(items, key=priority))


def advance_populations(populations, dt, extra, intra):
    for n, pop in _named_items(populations):
        step_pop(pop.integrator, pop, dt, extra.get(n, None), intra.get(n, None))


def _advance_network_component(component, *, compile_network_ops: bool = False):
    """Advance a network-side component.

    Components may expose an explicitly compile-safe ``advance_compiled`` method.
    We do not torch.compile arbitrary ``advance`` methods here because NetCon,
    ContinuousCon, and NetStim contain Python/module state mutations that should
    remain outside the default population-solve JIT path.
    """
    if compile_network_ops and hasattr(component, "advance_compiled"):
        return component.advance_compiled()
    return component.advance()


def step(
    populations,
    synapses,
    continuous_synapses,
    continuous_targets,
    netstim,
    t,
    dt,
    extra: Dict[str, torch.Tensor | None] = {},
    intra: Dict[str, torch.Tensor | None] = {},
    compile_network_ops=False,
):
    if netstim is not None:
        # NetStim has explicit heap/schedule side effects; keep it eager unless
        # NetStim grows a compile-safe method of its own.
        netstim(t, bptt=netstim.training, dt=dt)
    for _, target in _named_items(continuous_targets):
        target.reset_continuous_inputs()
    for _, c in _named_items(continuous_synapses):
        _advance_network_component(c, compile_network_ops=compile_network_ops)
    for _, s in _named_items(synapses):
        _advance_network_component(s, compile_network_ops=compile_network_ops)

    # Population/integrator compilation is handled inside each integrator with
    # scope="network_population". The stateful Python wrapper remains eager.
    advance_populations(populations, dt, extra, intra)


def get_local_index(population, mech, index):
    indices = torch.full_like(
        population.v, -1, dtype=torch.long, device=population.device()
    ).flatten()
    mech_key_flat = to_flat_idx_mech(population.shape, mech, population.device())
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


def _value_shape_description(value):
    """Human-readable shape/type summary for connection arguments."""
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device})"
        )
    if isinstance(value, torch.nn.Module):
        return f"{type(value).__name__} module"
    if isinstance(value, (float, int, bool)):
        return f"scalar {value!r}"
    if hasattr(value, "__len__"):
        try:
            return f"{type(value).__name__}(len={len(value)})"
        except TypeError:
            pass
    return type(value).__name__


def _connection_shape_context_lines(context, *, actual_edges=None, value_len=None):
    """Format connection context for shape-mismatch errors."""
    if not context:
        return []

    lines = []
    kind = context.get("kind", "event")
    rule = context.get("rule", "<unknown>")
    source_name = context.get("source_name", "<unknown>")
    target_name = context.get("target_name", "<unknown>")
    synapse_name = context.get("synapse_name", "<unknown>")
    pre_var = context.get("pre_var", None)
    input_name = context.get("input", None)

    endpoint = (
        f"{kind} connection {source_name} -> {target_name}:{synapse_name} "
        f"using rule={rule!r}"
    )
    if pre_var is not None:
        endpoint += f", pre_var={pre_var!r}"
    if input_name is not None:
        endpoint += f", input={input_name!r}"
    lines.append(endpoint + ".")

    selected_pre = context.get("selected_pre")
    selected_post = context.get("selected_post")
    candidate_edges = context.get("candidate_edges")
    final_edges = context.get("final_edges", actual_edges)
    allow_autapses = context.get("allow_autapses")
    allow_multapses = context.get("allow_multapses")

    summary = []
    if selected_pre is not None:
        summary.append(f"selected source elements={selected_pre}")
    if selected_post is not None:
        summary.append(f"selected target elements={selected_post}")
    if candidate_edges is not None:
        summary.append(f"candidate edges before filtering={candidate_edges}")
    if final_edges is not None:
        summary.append(f"edges after rule/filtering={final_edges}")
    if allow_autapses is not None:
        summary.append(f"allow_autapses={allow_autapses}")
    if allow_multapses is not None:
        summary.append(f"allow_multapses={allow_multapses}")
    if summary:
        lines.append("Connection summary: " + ", ".join(str(x) for x in summary) + ".")

    if (
        candidate_edges is not None
        and final_edges is not None
        and candidate_edges != final_edges
    ):
        removed = int(candidate_edges) - int(final_edges)
        if removed > 0:
            reasons = []
            if allow_autapses is False and context.get("same_population"):
                reasons.append("autapse filtering")
            if allow_multapses is False:
                reasons.append("duplicate/multapse filtering")
            reason_txt = " or ".join(reasons) if reasons else "connection filtering"
            lines.append(
                f"{removed} candidate edge(s) were removed by {reason_txt} "
                "before per-connection threshold/weight/delay shapes were checked."
            )

    if (
        value_len is not None
        and candidate_edges is not None
        and final_edges is not None
        and int(value_len) == int(candidate_edges)
        and int(value_len) != int(final_edges)
    ):
        lines.append(
            "The provided value length matches the unfiltered candidate edge "
            "count, not the final connection count. If values were built from "
            "the original pre/post arrays, either pass allow_multapses=True / "
            "allow_autapses=True as appropriate, or apply the same filtering to "
            "pre_idx, post_idx, and all per-edge values before calling connect."
        )
    else:
        lines.append(
            "Per-connection values must be scalar/0-D, length 1, length equal "
            "to the final number of connections, or a repeat-compatible length "
            "that evenly divides the final connection count."
        )

    if context.get("auto_expand"):
        lines.append(
            "auto_expand=True expands values after connectivity is sampled; it "
            "is safest for scalar values or distribution/parameter modules, not "
            "for already-expanded per-candidate tensors."
        )

    return lines


def _raise_connection_shape_error(
    name, value, pre_idx, *, context=None, value_len=None
):
    actual_edges = int(pre_idx.numel()) if torch.is_tensor(pre_idx) else len(pre_idx)
    base = (
        f"Invalid {name} shape for connection: {_value_shape_description(value)} "
        f"cannot be aligned to {actual_edges} final connection(s)."
    )
    lines = [base]
    lines.extend(
        _connection_shape_context_lines(
            context, actual_edges=actual_edges, value_len=value_len
        )
    )
    raise ValueError("\n".join(lines))


def check_weight_shape(weight, pre_idx, *, name="weight", context=None):
    """
    Validate a scalar or 1-D per-connection value against finalized pre indices.

    Returns the repeat count used by ``make_weight``/``expand``. ``context`` is
    optional diagnostic metadata supplied by Network.connect so shape errors can
    explain connectivity-rule filtering, autapse/multapse removal, and the
    source/target/synapse involved.
    """
    n_edges = int(pre_idx.numel()) if torch.is_tensor(pre_idx) else len(pre_idx)

    if isinstance(weight, (float, int, bool)):
        return n_edges

    if isinstance(weight, torch.Tensor):
        if weight.ndim == 0:
            return n_edges
        if weight.ndim != 1:
            raise ValueError(
                f"Invalid {name} shape for connection: expected a scalar/0-D "
                f"or 1-D tensor, got Tensor(shape={tuple(weight.shape)}, "
                f"dtype={weight.dtype}, device={weight.device})."
            )
        value_len = int(weight.shape[0])
        if value_len == 1:
            return n_edges
        if value_len == n_edges:
            return 1
        if value_len > 0 and n_edges % value_len == 0:
            return n_edges // value_len
        _raise_connection_shape_error(
            name, weight, pre_idx, context=context, value_len=value_len
        )

    if hasattr(weight, "__len__"):
        value_len = len(weight)
        if value_len == 1:
            return n_edges
        if value_len == n_edges:
            return 1
        if value_len > 0 and n_edges % value_len == 0:
            return n_edges // value_len
        _raise_connection_shape_error(
            name, weight, pre_idx, context=context, value_len=value_len
        )

    if isinstance(weight, torch.nn.Module):
        return n_edges

    raise TypeError(
        f"Unsupported type for {name}: {type(weight)}. Expected a scalar, "
        "1-D tensor/list, torch.nn.Module, parameter, or distribution-like module."
    )


def _evaluate(value):
    if isinstance(value, torch.nn.Module):
        return value()
    return value


def expand(value, n, *, device=None, dtype=None):
    if isinstance(value, torch.nn.Module):
        out = value.sample(n)
        if torch.is_tensor(out) and (device is not None or dtype is not None):
            out = out.to(
                device=device if device is not None else out.device,
                dtype=(
                    dtype
                    if dtype is not None and torch.is_floating_point(out)
                    else out.dtype
                ),
            )
        return out
    return torch.as_tensor(value, device=device, dtype=dtype).repeat(n)


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


def make_weight(weights, n, *, device=None, dtype=None):
    class ParameterOrDistributionWrapper(torch.nn.Module):
        """A simple wrapper for parameters or distributions that can be sampled."""

        def __init__(self, param):
            super().__init__()
            if isinstance(param, torch.nn.Module) and (
                device is not None or dtype is not None
            ):
                try:
                    param = param.to(device=device, dtype=dtype)
                except TypeError:
                    param = param.to(device=device)
            self.param = param

        def sample(self, n):
            if is_parametric(self.param):
                out = self.param.repeat(n)
            else:
                out = self.param.sample(n)
            if torch.is_tensor(out) and (device is not None or dtype is not None):
                out = out.to(
                    device=device if device is not None else out.device,
                    dtype=(
                        dtype
                        if dtype is not None and torch.is_floating_point(out)
                        else out.dtype
                    ),
                )
            return out

    class WeightExpander(torch.nn.Module):
        def __init__(self, weights, n):
            super(WeightExpander, self).__init__()
            self.weights = torch.nn.ModuleList(
                [ParameterOrDistributionWrapper(w) for w in weights]
            )
            self.n = n
            self.register_buffer(
                "w",
                torch.empty(
                    0,
                    device=device if device is not None else None,
                    dtype=dtype if dtype is not None else torch.float32,
                ),
            )

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

    .. note::
        Event delivery follows Dendra's sampled, fixed-step ordering. At the start
        of a timestep, NetStim and NetCon state is advanced using the currently
        visible source state; populations are then integrated to the next sample.
        A voltage threshold crossing created by that integration is consequently
        observed by its NetCon on the following network step. Ordinary hard-event
        connection delays are quantized to integer timesteps (differentiable delay
        modes may interpolate adjacent bins). This phase convention is part of
        Dendra's network semantics and need not match another simulator's within-step
        event queue, even when the membrane and synaptic response kinetics agree.

    Parameters
    ----------
    populations : dict[str, Population]
        Mapping from population name to :class:`Population` instance. Populations
        are built, registered as attributes, and used as sources/targets for
        connectivity.
    netstim : NetStim, optional
        Optional spike generator attached under the name ``netstim``. A source
        may also be registered later with :meth:`attach_netstim`.
    seed : int, optional
        Seed for network-level RNG used in stochastic wiring utilities.
    track_netcon_events : bool, optional
        If True, event-based NetCons allocate the historical per-connection
        ``event_queue`` used for delivery introspection. If False (default),
        NetCons skip that large debug queue during inference and keep only the
        lightweight current-step ``events`` buffer.
    netcon_delay_backend : {"dense", "sparse_calendar", "bitpacked_history"}, optional
        Delay-line backend for event-based NetCons. ``"dense"`` is the default
        differentiable/dense implementation. ``"sparse_calendar"`` is
        an inference-only backend that stores pending nonzero deliveries in
        sparse calendar buckets while preserving the existing ``net_receive``
        dense-payload API. Can be beneficial for memory efficiency and speed on CPU
        when networks have long delays and are not densely spiking. Should be avoided on
        GPU. ``"bitpacked_history"`` is an inference-only backend
        that stores packed source-spike history and reconstructs the dense
        delivery payload from the static edge list. Suitable for networks where
        threshold is uniform and / or determined solely by the pre-synaptic population,
        so that the same source spike history can be shared across synapses. Can be very
        beneficial for memory efficiency and speed on GPU for large networks with long
        delays. The first run with 'bitpacked_history' will be slow due to CUDA kernel
        compilation, but subsequent runs will be much faster.
    netcon_train_backend : {"dense", "source_history", "auto"}, optional
        Differentiable training backend for event NetCons. ``"dense"`` keeps the
        fully general per-synapse delay buffer. ``"source_history"`` uses compact
        source-level histories when thresholds/events are source-level; with
        ``diff_spiking=False`` this uses packed hard source spikes and with
        ``diff_spiking=True`` this uses floating source gates. ``"auto"`` uses
        source history when exact and otherwise falls back to dense. Default is
        ``"auto"`` so common source-level SNN projections get the compact
        differentiable backend without opting in explicitly.
    """

    def __init__(
        self,
        populations: Dict[str, Population],
        netstim=None,
        seed=None,
        *,
        track_netcon_events: bool = False,
        netcon_delay_backend: Literal[
            "dense", "sparse_calendar", "bitpacked_history"
        ] = "dense",
        netcon_train_backend: Literal["dense", "source_history", "auto"] = "auto",
    ):
        if any(pop.is_batched() for pop in populations.values()):
            raise ValueError(
                "Batched populations are not supported. Implement your networks with unbatched populations and then call .batch(batch_size)."
            )
        if netstim is not None and not isinstance(netstim, NetStim):
            raise TypeError("netstim must be an instance of NetStim or None.")
        if "netstim" in populations:
            raise ValueError(
                "'netstim' is reserved for the optional NetStim source and "
                "cannot be used as a biological population name."
            )
        super(Network, self).__init__(seed=seed)
        self.populations = populations
        for name, pop in populations.items():
            pop.build()
            pop.name = name
            setattr(self, name, pop)

        # Register an empty slot first.  A NetStim supplied to the constructor
        # is attached through the same public path used for post-construction
        # attachment below.
        self.netstim = None
        self.track_netcon_events = bool(track_netcon_events)
        if netcon_delay_backend not in (
            "dense",
            "sparse_calendar",
            "bitpacked_history",
        ):
            raise ValueError(
                "netcon_delay_backend must be one of 'dense', 'sparse_calendar', or 'bitpacked_history'."
            )
        self.netcon_delay_backend = netcon_delay_backend
        if netcon_train_backend not in ("dense", "source_history", "auto"):
            raise ValueError(
                "netcon_train_backend must be one of 'dense', 'source_history', or 'auto'."
            )
        self.netcon_train_backend = netcon_train_backend

        self.synapse_spec = {}
        self.synapses = torch.nn.ModuleDict()
        self.continuous_synapse_spec = {}
        self.continuous_synapses = torch.nn.ModuleDict()
        # Plain dict of unique target ContinuousSynapse mechanisms that need
        # their per-step analog input buffers reset.  These mechanisms are
        # already registered on their owning populations, so this dict does not
        # duplicate module ownership/state_dict entries.
        self.continuous_targets = {}
        self.dt = None
        self.built = False
        self._mode_requires_initialize = False

        self.backend = BACKEND.value
        self.fullgraph = bool(FULLGRAPH)
        self.dynamic = bool(DYNAMIC)
        self.jit = bool(JIT)
        self.jit_network_solves = bool(JIT_NETWORK_SOLVES)
        self.jit_network_ops = bool(JIT_NETWORK_OPS)
        # Legacy spelling retained as an attribute alias.
        self.jit_in_network = self.jit_network_solves
        self.compile_mode = COMPILE_MODE.value
        self.compile_options = current_compile_options()
        self.compile_options_key = compile_options_key(self.compile_options)

        self.is_batched = False

        torch._dynamo.reset()

        # Keep the stateful network/population wrapper eager. Population JIT is
        # now handled by each integrator, which compiles only its tensor kernel
        # and leaves model state commits outside Dynamo.
        self._step_train = step
        self._step_eval = step

        # Keep network time as an origin plus an integer step count.  Repeated
        # floating-point addition can drift below an analytically exact event
        # boundary (for example, 40 float32 additions of 0.025 are less than
        # 1.0), delaying NetStim/NetCon events by a complete timestep.
        self.register_buffer(
            "t", torch.tensor(0.0, device=torch.device("cpu"), dtype=torch.float64)
        )
        self.register_buffer(
            "_clock_origin",
            torch.tensor(0.0, device=torch.device("cpu"), dtype=torch.float64),
        )
        self.register_buffer(
            "_clock_step",
            torch.tensor(0, device=torch.device("cpu"), dtype=torch.long),
        )
        # Duration-based entrypoints execute only complete fixed timesteps.
        # Keep requested-but-not-yet-simulated physical time separate from the
        # public, whole-step simulation clock so partitioning a run across
        # calls cannot silently lose a fractional timestep.
        self.register_buffer(
            "_duration_remainder",
            torch.tensor(0.0, device=torch.device("cpu"), dtype=torch.float64),
        )

        self.compile_network_ops = jit_enabled_for_scope("network_ops", self)

        self._population_step_items = _accelerator_first_step_items(self.populations)
        self._synapse_step_items = ()
        self._continuous_synapse_step_items = ()
        self._continuous_target_step_items = ()

        if netstim is not None:
            self.attach_netstim(netstim)

        # Track device signature to trigger rebuilds if placements change.
        self._device_sig = self._device_signature()

        self._state_cache = {}
        self._syn_cache = {}

        self.eval()

    def _refresh_compile_config_from_ctx(self):
        """Refresh network compile policy from the active dendra.ctx."""
        self.backend = BACKEND.value
        self.fullgraph = bool(FULLGRAPH)
        self.dynamic = bool(DYNAMIC)
        self.jit = bool(JIT)
        self.jit_network_solves = bool(JIT_NETWORK_SOLVES)
        self.jit_network_ops = bool(JIT_NETWORK_OPS)
        self.jit_in_network = self.jit_network_solves
        self.compile_mode = COMPILE_MODE.value
        self.compile_options = current_compile_options()
        self.compile_options_key = compile_options_key(self.compile_options)
        self.compile_network_ops = jit_enabled_for_scope("network_ops", self)

        for pop in self.populations.values():
            refresh = getattr(pop, "_refresh_compile_config_from_ctx", None)
            if refresh is not None:
                refresh()
            if getattr(pop, "integrator", None) is not None:
                pop.integrator.configure_jit(pop, scope="network_population")
        return self

    def clear_jit_cache(self):
        """Drop lazily compiled functions/caches from this network.

        Network itself keeps only eager top-level stepping functions, but its
        populations own compiled integrator kernels and compiled ``make_intra``
        helpers after ``with dn.ctx(JIT=1): ...``.  Those compiled callables are
        deliberately lazy and can be recreated after load, so they should not be
        serialized with checkpoints or arbitrary pickle payloads.
        """
        self._step_train = step
        self._step_eval = step
        self._step = step

        for pop in self.populations.values():
            clear = getattr(pop, "clear_jit_cache", None)
            if clear is not None:
                clear()
            elif getattr(pop, "integrator", None) is not None and hasattr(
                pop.integrator, "_compiled_kernels"
            ):
                pop.integrator._compiled_kernels.clear()

        if self.netstim is not None:
            clear = getattr(self.netstim, "clear_jit_cache", None)
            if clear is not None:
                clear()

        for syn in self.synapses.values():
            clear = getattr(syn, "clear_jit_cache", None)
            if clear is not None:
                clear()
        for syn in self.continuous_synapses.values():
            clear = getattr(syn, "clear_jit_cache", None)
            if clear is not None:
                clear()
        return self

    def pickleable(
        self,
        *,
        inplace: bool = False,
        clone: bool = False,
        reset_global_compiler: bool = False,
    ):
        """Return a pickle-friendly network handle.

        By default this method is cheap and non-mutating:

        .. code-block:: python

            payload = {"model": net.pickleable(), "loss": loss}
            pickle.dump(payload, f)

        The live network keeps its compiled population/integrator kernels.
        Pickling calls :meth:`__getstate__`, which strips process-local compiler
        objects only from the serialized state.  Therefore saving a checkpoint
        does not force the current model to recompile before continuing.

        Parameters
        ----------
        inplace : bool, default False
            If True, clear the live network's local JIT caches before returning
            it.  The next JIT-enabled run may need to recompile.
        clone : bool, default False
            If True, return a sanitized deep copy.  This avoids mutating the
            live network but duplicates tensor storage.
        reset_global_compiler : bool, default False
            Also clear global Torch compiler caches.  Leave this False when you
            want to save and continue running with already-compiled kernels.
        """
        if inplace and clone:
            raise ValueError(
                "pickleable(...): choose at most one of inplace=True or clone=True."
            )
        if clone:
            import copy as _copy

            obj = _copy.deepcopy(self)
            obj.clear_jit_cache()
        elif inplace:
            obj = self.clear_jit_cache()
        else:
            obj = self
        if reset_global_compiler:
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
                torch.compiler.reset()
            elif hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
                torch._dynamo.reset()
        return obj

    def __getstate__(self):
        """Serialize without process-local network compile wrappers.

        This is a defensive hook: direct ``pickle.dump(net, f)`` should work
        after JIT-enabled runs without requiring callers to remember
        ``net.pickleable()``.  Population/integrator compiled kernels are
        stripped by their own ``__getstate__`` hooks.
        """
        state = self.__dict__.copy()
        state["_step_train"] = step
        state["_step_eval"] = step
        state["_step"] = step
        return state

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

        Notes
        -----
        Dense-to-dense NetCon mode switches preserve queued deliveries and take
        effect immediately. A switch that crosses a compact inference or
        source-history runtime layout requires :meth:`initialize` before the
        next execution call; stepping fails closed until then.
        """
        # Validate every event connection before changing any child module.
        # Strict compact-training policy can reject a transition (for example,
        # after scheduled events were added); preflight keeps that rejection
        # atomic across the complete Network.
        for syn in self.synapses.values():
            syn._validate_train_mode_transition(mode)
        for pop in self.populations.values():
            pop.train(mode)
        for syn in self.synapses.values():
            syn.train(mode)
        for syn in self.continuous_synapses.values():
            syn.train(mode)
        if self.netstim is not None:
            self.netstim.train(mode)
        self.training = mode
        self._step = self._step_train if mode else self._step_eval
        if any(
            getattr(synapse, "_mode_requires_initialize", False)
            for synapse in self.synapses.values()
        ):
            self._mode_requires_initialize = True
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

        Notes
        -----
        Dense-to-dense switches preserve pending traffic. Compact runtime
        layouts must be reinitialized before execution, as described by
        :meth:`train`.
        """
        super(Network, self).eval()
        for pop in self.populations.values():
            pop.eval()
        for syn in self.synapses.values():
            syn.eval()
        for syn in self.continuous_synapses.values():
            syn.eval()
        if self.netstim is not None:
            self.netstim.eval()
        self.training = False
        self._step = self._step_eval
        return self

    def eval_(self):
        """In-place variant of :meth:`eval` that returns ``None``."""
        self.eval()

    def _require_mode_runtime_ready(self):
        """Reject execution after a mode switch needing storage conversion."""
        pending = [
            name
            for name, synapse in self.synapses.items()
            if getattr(synapse, "_mode_requires_initialize", False)
        ]
        if self._mode_requires_initialize or pending:
            raise RuntimeError(
                "Network train/eval mode changed across incompatible NetCon "
                "runtime layouts. Call network.initialize(dt) before execution; "
                f"pending NetCons: {pending}."
            )

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

    def _reset_runtime_clock(self, t):
        """Anchor the exact network clock at ``t`` with zero elapsed steps."""
        value = torch.as_tensor(t, device=self.t.device, dtype=self.t.dtype).reshape(())
        with torch.no_grad():
            self._clock_origin.copy_(value)
            self._clock_step.zero_()
            self.t.copy_(value)
            self._sync_population_clocks()

    def _runtime_clock_value(self):
        """Return ``origin + integer_step * dt`` without accumulated drift."""
        if self.dt is None:
            return self.t.detach().clone()
        return self._clock_origin + self._clock_step.to(self.t.dtype) * float(self.dt)

    def _sync_runtime_clock(self):
        with torch.no_grad():
            self.t.copy_(self._runtime_clock_value())
            self._sync_population_clocks()

    def _advance_runtime_clock(self):
        with torch.no_grad():
            self._clock_step.add_(1)
            self.t.copy_(self._runtime_clock_value())
            self._sync_population_clocks()

    def _reanchor_runtime_clock_from_time(self):
        """Treat public ``t`` as authoritative after external state loading."""
        with torch.no_grad():
            if self.dt is None:
                self._clock_origin.copy_(self.t)
                self._clock_step.zero_()
            else:
                metadata_t = self._runtime_clock_value()
                if not torch.equal(metadata_t, self.t):
                    # Missing/stale clock metadata (including callers editing
                    # only public ``t``) must be re-anchored. A coherent loaded
                    # origin/step pair is preserved exactly; subtracting the
                    # elapsed duration can otherwise introduce a one-ulp drift.
                    elapsed = self._clock_step.to(self.t.dtype) * float(self.dt)
                    self._clock_origin.copy_(self.t - elapsed)
            self._sync_runtime_clock()

    def _sync_population_clocks(self):
        """Snap population clocks to the authoritative step-derived time."""
        for population in self.populations.values():
            value = self.t.to(device=population.t.device, dtype=population.t.dtype)
            population.t.copy_(value)

    def _duration_budget(self, duration: float, dt: float) -> tuple[int, float]:
        """Resolve a duration call against retained unsimulated time."""
        pending = float(self._duration_remainder.detach().cpu().item())
        return _duration_step_budget(duration, dt, pending)

    def _set_duration_remainder(self, value: float) -> None:
        """Commit retained physical time without mutating checkpoint aliases."""
        self._duration_remainder = torch.tensor(
            value,
            device=self._duration_remainder.device,
            dtype=torch.float64,
        )

    def _clear_duration_remainder(self) -> None:
        """Start a fresh duration budget for a new simulation episode."""
        self._set_duration_remainder(0.0)

    def attach_netstim(self, netstim: NetStim, *, replace: bool = False):
        """Attach a :class:`NetStim` after network construction.

        The constructor's ``netstim=`` argument remains supported, but callers
        may now build a purely cellular network first and add an artificial
        source later::

            net = dn.Network({"cell": cell})
            net.build(dt)
            net.attach_netstim(dn.NetStim(N=8))
            net.connect_one_to_one(net.netstim[:], target, synapse)
            net.initialize(dt)

        Attaching a source invalidates materialized NetCons.  Likewise, every
        subsequent ``connect*`` call invalidates them.  Call :meth:`build` or,
        normally, :meth:`initialize` before the next :meth:`run`.  Attaching to
        an already initialized/running network is supported structurally, but
        reinitialization is required because rebuilding replaces all NetCon
        runtime queues.

        Parameters
        ----------
        netstim : NetStim
            Artificial spike source to register as ``network.netstim``.
        replace : bool, default False
            Replace an existing NetStim.  When existing NetStim-sourced
            connection specifications are present, replacement is accepted only
            if the new source has the same shape, because those specifications
            store source-flat indices.

        Returns
        -------
        Network
            Self, for chaining.
        """
        if not isinstance(netstim, NetStim):
            raise TypeError("netstim must be an instance of NetStim.")
        if "netstim" in self.populations:
            raise ValueError(
                "'netstim' is reserved for the optional NetStim source and "
                "cannot be used as a biological population name."
            )

        current = self.netstim
        if current is netstim:
            return self
        if current is not None and not replace:
            raise RuntimeError(
                "This Network already has a NetStim. Pass replace=True to "
                "replace it explicitly."
            )

        has_netstim_specs = any(
            pre_name == "netstim" for pre_name, *_ in self.synapse_spec.keys()
        ) or any(
            pre_name == "netstim"
            for pre_name, *_ in self.continuous_synapse_spec.keys()
        )
        if current is not None and has_netstim_specs:
            old_shape = tuple(current.shape)
            new_shape = tuple(netstim.shape)
            if old_shape != new_shape:
                raise ValueError(
                    "Cannot replace a NetStim with a different shape while "
                    "NetStim-sourced connection specifications exist: "
                    f"old shape={old_shape}, new shape={new_shape}. Clear and "
                    "recreate those connections first."
                )

        netstim.name = "netstim"
        if self.dt is not None and hasattr(netstim, "set_dt"):
            netstim.set_dt(float(self.dt))
        netstim.train(bool(self.training))
        self.netstim = netstim

        # Existing NetCon modules were built without this source (or against a
        # replaced source) and must not be used again until rebuilt.
        self.built = False
        return self

    def clear_synapses(self):
        """
        Remove all queued synapse specs and built NetCon modules.

        Use when re-wiring the network before calling :meth:`build` again.
        """
        self.synapse_spec = {}
        self.synapses.clear()
        self.continuous_synapse_spec = {}
        self.continuous_synapses.clear()
        self.continuous_targets = {}
        self.built = False

    def _normalize_endpoint(self, source, target):
        """Normalize full populations/devices to endpoint objects."""
        if isinstance(source, Population) or isinstance(source, NetStim):
            source = source[:]
        if isinstance(target, Population):
            target = target[:]
        return source, target, source.model, target.model

    def _estimate_candidate_edges(
        self,
        *,
        rule,
        spec,
        pre_pool,
        post_pool,
        source_model,
        target_model,
        allow_autapses: bool,
    ):
        """Best-effort count of candidate edges before filtering.

        This is diagnostics-only. Some stochastic rules cannot know the exact
        pre-filter count a user had in mind after random sampling, so this
        method returns ``None`` when a useful exact estimate is unavailable.
        """
        rule = str(rule).replace("-", "_")
        n_pre = int(pre_pool.numel())
        n_post = int(post_pool.numel())
        if rule == "one_to_one":
            return n_pre if n_pre == n_post else None

        if rule in ("all_to_all", "dense"):
            # Candidate count before autapse/multapse filtering. Keep this O(1);
            # exact self-pair counts for arbitrary selections are not worth an
            # additional large tensor operation on every connect call.
            return n_pre * n_post

        if rule in ("fixed_total_number", "fixed_total"):
            try:
                return int(_require(spec, "N", "n"))
            except Exception:
                return None

        if rule == "fixed_indegree":
            try:
                return int(_require(spec, "indegree", "in_degree", "K", "N")) * n_post
            except Exception:
                return None

        if rule == "fixed_outdegree":
            try:
                return int(_require(spec, "outdegree", "out_degree", "K", "N")) * n_pre
            except Exception:
                return None

        # pairwise_bernoulli and pairwise_poisson are stochastic; the final count
        # is the most meaningful diagnostic value.
        return None

    def _connection_shape_context(
        self,
        *,
        kind: str,
        rule,
        spec,
        source_model,
        target_model,
        synapse,
        pre_pool,
        post_pool,
        pre_idx,
        allow_autapses: bool,
        allow_multapses: bool,
        auto_expand: bool,
        pre_var=None,
        input=None,
    ):
        """Build diagnostic context for threshold/weight/delay shape errors."""
        candidate_edges = self._estimate_candidate_edges(
            rule=rule,
            spec=spec,
            pre_pool=pre_pool,
            post_pool=post_pool,
            source_model=source_model,
            target_model=target_model,
            allow_autapses=allow_autapses,
        )
        return {
            "kind": kind,
            "rule": str(rule).replace("-", "_"),
            "source_name": getattr(source_model, "name", "<unnamed>"),
            "target_name": getattr(target_model, "name", "<unnamed>"),
            "synapse_name": getattr(synapse, "name", repr(synapse)),
            "pre_var": pre_var,
            "input": input,
            "selected_pre": int(pre_pool.numel()),
            "selected_post": int(post_pool.numel()),
            "same_population": bool(source_model is target_model),
            "candidate_edges": candidate_edges,
            "final_edges": int(pre_idx.numel()),
            "allow_autapses": bool(allow_autapses),
            "allow_multapses": bool(allow_multapses),
            "auto_expand": bool(auto_expand),
        }

    def _flat_selection(self, selection):
        """Return selected endpoint ids for a Slice or SynapseSlots selection."""
        if isinstance(selection, SynapseSlots):
            return selection.local_index.to(
                device=selection.model.device(), dtype=torch.long
            )
        model = selection.model
        return to_flat_idx_torch(
            model.shape,
            selection.index,
            model.device(),
        ).to(torch.long)

    def _target_id_universe_size(self, target, target_model):
        if isinstance(target, SynapseSlots):
            return int(target.slot_to_flat_index.numel())
        return int(target_model.v.numel())

    def _post_flat_by_target_id(self, target):
        if isinstance(target, SynapseSlots):
            return target.slot_to_flat_index
        return None

    def _validate_target_synapse_endpoint(self, target, synapse):
        if isinstance(target, SynapseSlots):
            if synapse is not None and synapse is not target.synapse:
                raise ValueError(
                    "The target SynapseSlots selection belongs to a different "
                    "mechanism than the supplied synapse argument."
                )
            return target.synapse
        if synapse is None:
            raise TypeError(
                "A postsynaptic synapse mechanism is required unless target is "
                "a SynapseSlots selection."
            )
        return synapse

    def _target_post_idx(self, target_model, post_ids, synapse, target):
        """Convert generated target ids to synapse-local NetCon indices."""
        if isinstance(target, SynapseSlots):
            return post_ids.to(device=target_model.device(), dtype=torch.long)
        return self._to_synapse_local_post_idx(
            target_model,
            post_ids.to(target_model.device()),
            synapse,
        )

    def _to_synapse_local_post_idx(self, target_model, post_flat, synapse):
        """
        Convert target population-flat indices to target synapse-local indices.

        If a mechanism has multiple local slots at one physical compartment, a
        compartment target is ambiguous. Use ``population.slots(...)`` or
        ``slice.slots(...)`` to target a specific local slot.
        """
        post_flat = post_flat.to(device=target_model.device(), dtype=torch.long)

        mech_key_flat = to_flat_idx_mech(
            target_model.shape, synapse, target_model.device()
        ).to(torch.long)
        counts = torch.zeros(
            target_model.v.numel(), device=target_model.device(), dtype=torch.long
        )
        if mech_key_flat.numel() > 0:
            counts.scatter_add_(
                0,
                mech_key_flat,
                torch.ones_like(mech_key_flat, dtype=torch.long),
            )
        selected_counts = counts.index_select(0, post_flat)
        if torch.any(selected_counts > 1):
            bad = post_flat[selected_counts > 1][:10].detach().cpu().tolist()
            raise ValueError(
                f"Target population '{target_model.name}' has multiple local "
                f"slots of synapse '{synapse}' at physical target locations "
                f"including {bad}. Use population.slots(synapse, ...) or "
                "slice.slots(synapse, ...) to target synapse-local slots "
                "explicitly."
            )

        local = get_local_index(target_model, synapse, post_flat)

        if torch.any(local < 0):
            bad = post_flat[local < 0][:10].detach().cpu().tolist()
            raise ValueError(
                f"Target population '{target_model.name}' does not have "
                f"the synapse '{synapse}' at target locations including {bad}."
            )

        return local

    def synapse_slots(self, target, synapse=None, *, slots=None, local_index=None):
        """Return explicit local slots of a postsynaptic mechanism.

        This is a convenience wrapper around ``Population.slots`` and
        ``Slice.slots``. It is useful for banked point-process mechanisms where
        several independent synapse slots may share one physical compartment.
        """
        if local_index is not None:
            if slots is not None:
                raise ValueError("Provide either slots or local_index, not both.")
            slots = local_index

        if isinstance(target, SynapseSlots):
            if synapse is not None and synapse is not target.synapse:
                raise ValueError(
                    "synapse argument does not match the supplied SynapseSlots."
                )
            return target if slots is None else target[slots]

        if synapse is None:
            raise TypeError(
                "synapse_slots(target, synapse, ...) requires a postsynaptic "
                "mechanism unless target is already a SynapseSlots selection."
            )

        if isinstance(target, Population):
            return target.slots(synapse, local_index=slots)
        if hasattr(target, "slots"):
            return target.slots(synapse, local_index=slots)
        raise TypeError(
            "target must be a Population, Slice, or SynapseSlots selection."
        )

    def _mechanism_from_pre_var(self, pre, pre_var):
        """Return the mechanism addressed by a ``mech.<alias>.<var>`` pre_var.

        ``Network`` stores source indices in population-flat coordinates.
        Mechanism variables, however, can be local to the mechanism insertion
        region.  When ``pre_var`` names such a mechanism-local variable, the
        indices passed to ``NetCon`` need to be converted into that local frame.

        Returns ``None`` for ordinary population variables, NetStim variables,
        malformed strings, or pre variables whose mechanism cannot be resolved.
        """
        if not isinstance(pre_var, str):
            return None
        parts = pre_var.split(".")
        if len(parts) < 3 or parts[0] != "mech":
            return None
        handler = getattr(pre, "mech", None)
        if handler is None:
            return None
        alias = parts[1]
        try:
            return getattr(handler, alias)
        except AttributeError:
            mechanisms = getattr(handler, "mechanisms", None)
            if mechanisms is None:
                return None
            return mechanisms.get(alias, None)

    def _pre_idx_for_pre_var(self, pre, pre_idx, pre_var):
        """Convert population-flat pre indices to the frame used by ``pre_var``.

        For population-wide variables such as ``v`` or mechanisms inserted
        everywhere, ``pre_idx`` is already correct.  For variables owned by a
        mechanism inserted on a slice, e.g. ``mech.ctx_fs.syn_spikes``,
        ``get_pre_var(pre).view(-1)`` is mechanism-local, so the population-flat
        indices must be mapped to local indices using the mechanism key.
        """
        mech = self._mechanism_from_pre_var(pre, pre_var)
        if mech is None or getattr(mech, "key", None) is None:
            return pre_idx

        pre_flat = pre_idx.to(device=pre.device(), dtype=torch.long)
        local = get_local_index(pre, mech, pre_flat)
        if torch.any(local < 0):
            bad = pre_flat[local < 0][:10].detach().cpu().tolist()
            raise ValueError(
                f"Pre-synaptic variable '{pre_var}' is local to mechanism "
                f"'{getattr(mech, 'name', '<unknown>')}', but source locations "
                f"including {bad} are outside that mechanism's insertion region."
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
        target_id_universe_size: int | None = None,
        post_flat_by_target_id: torch.Tensor | None = None,
    ):
        """
        Generate edges for one connection call.

        ``pre_pool`` always uses source population-flat coordinates. ``post_pool``
        normally uses target population-flat coordinates. For banked point-process
        targeting, callers may instead pass target synapse-local slot ids and set
        ``target_id_universe_size`` plus ``post_flat_by_target_id``. Autapse
        filtering then compares presynaptic flat indices with the physical
        compartment reached by each local slot, while multapse filtering treats
        local slots as distinct targets.
        """
        device = source_model.device()
        pre_pool = pre_pool.to(device=device, dtype=torch.long)
        post_pool = post_pool.to(device=device, dtype=torch.long)
        same_population = source_model is target_model
        target_id_universe_size = int(
            target_model.v.numel()
            if target_id_universe_size is None
            else target_id_universe_size
        )
        if post_flat_by_target_id is not None:
            post_flat_by_target_id = post_flat_by_target_id.to(
                device=device, dtype=torch.long
            )
        rule = str(rule).replace("-", "_")

        def _post_flat_for_ids(ids):
            ids = ids.to(device=device, dtype=torch.long)
            if post_flat_by_target_id is None:
                return ids
            return post_flat_by_target_id.index_select(0, ids)

        def _filter_autapses(pre_idx, post_idx):
            if allow_autapses or not same_population or pre_idx.numel() == 0:
                return pre_idx, post_idx
            post_flat = _post_flat_for_ids(post_idx)
            keep = pre_idx.to(device=device, dtype=torch.long) != post_flat
            return pre_idx[keep], post_idx[keep]

        def _filter_multapses(pre_idx, post_idx):
            if allow_multapses or pre_idx.numel() <= 1:
                return pre_idx, post_idx
            return self._drop_multapses(
                pre_idx,
                post_idx,
                num_targets_total=target_id_universe_size,
            )

        if rule == "one_to_one":
            if pre_pool.numel() != post_pool.numel():
                raise ValueError(
                    "one_to_one requires source and target selections to have "
                    "the same number of elements."
                )
            pre_idx, post_idx = pre_pool, post_pool
            pre_idx, post_idx = _filter_autapses(pre_idx, post_idx)
            pre_idx, post_idx = _filter_multapses(pre_idx, post_idx)
            return pre_idx, post_idx

        if rule in ("all_to_all", "dense"):
            pre_idx, post_idx = self._all_to_all_edges(pre_pool, post_pool)
            pre_idx, post_idx = _filter_autapses(pre_idx, post_idx)
            pre_idx, post_idx = _filter_multapses(pre_idx, post_idx)
            return pre_idx, post_idx

        if rule in ("pairwise_bernoulli", "bernoulli"):
            p = float(_require(spec, "p", "prob", "probability"))
            if not 0.0 <= p <= 1.0:
                raise ValueError("pairwise_bernoulli requires 0 <= p <= 1.")

            pre_idx, post_idx = self._all_to_all_edges(pre_pool, post_pool)
            pre_idx, post_idx = _filter_autapses(pre_idx, post_idx)
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

            pre_idx, post_idx = _filter_multapses(pre_idx, post_idx)
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
            pre_idx, post_idx = _filter_autapses(pre_idx, post_idx)
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
            if n < 0:
                raise ValueError("fixed_total_number requires a non-negative count.")
            if n == 0:
                return pre_pool[:0], post_pool[:0]

            candidates_pre, candidates_post = self._all_to_all_edges(
                pre_pool, post_pool
            )
            candidates_pre, candidates_post = _filter_autapses(
                candidates_pre, candidates_post
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
                    target_flat = _post_flat_for_ids(target.reshape(1))[0]
                    pool = pool[pool != target_flat]

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
            pre_idx, post_idx = _filter_multapses(pre_idx, post_idx)
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
                    post_flat = _post_flat_for_ids(pool)
                    pool = pool[post_flat != source]

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
            pre_idx, post_idx = _filter_multapses(pre_idx, post_idx)
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
        shape_context=None,
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

        if threshold is None:
            threshold = torch.nan

        n_threshold = check_weight_shape(
            threshold, source_idx, name="threshold", context=shape_context
        )
        n_weight = check_weight_shape(
            weight, source_idx, name="weight", context=shape_context
        )
        n_delay = check_weight_shape(
            delay, source_idx, name="delay", context=shape_context
        )

        self.synapse_spec.setdefault(
            (source_pop.name, target_pop.name, synapse, pre_var), []
        ).append(
            (
                source_idx,
                target_idx,
                threshold,
                n_threshold,
                to_param(
                    weight,
                    positive=True,
                    device=target_pop.device(),
                    dtype=target_pop.dtype(),
                ),
                n_weight,
                to_param(
                    delay,
                    positive=True,
                    device=target_pop.device(),
                    dtype=target_pop.dtype(),
                ),
                n_delay,
            )
        )
        # A materialized NetCon set no longer represents the wiring spec after
        # any new connection is appended.
        self.built = False

    def _connect_continuous(
        self,
        source_pop,
        source_idx,
        target_pop,
        target_idx,
        synapse,
        *,
        weight=1.0,
        delay=0.0,
        pre_var=None,
        input=None,
        reduce="sum",
        transform=None,
        shape_context=None,
    ):
        """Append a finalized continuous-connection spec.

        ``source_idx`` is in source population-flat or pre-var-local coordinates
        after build-time conversion. ``target_idx`` is target synapse-local.
        """
        if source_idx.numel() == 0:
            return

        n_weight = check_weight_shape(
            weight, source_idx, name="weight", context=shape_context
        )
        n_delay = check_weight_shape(
            delay, source_idx, name="delay", context=shape_context
        )

        self.continuous_synapse_spec.setdefault(
            (
                source_pop.name,
                target_pop.name,
                synapse,
                pre_var,
                input,
                reduce,
                transform,
            ),
            [],
        ).append(
            (
                source_idx,
                target_idx,
                to_param(
                    weight,
                    positive=True,
                    device=target_pop.device(),
                    dtype=target_pop.dtype(),
                ),
                n_weight,
                to_param(
                    delay,
                    positive=True,
                    device=target_pop.device(),
                    dtype=target_pop.dtype(),
                ),
                n_delay,
            )
        )
        self.built = False

    def connect_continuous(
        self,
        source,
        target,
        synapse=None,
        conn_spec=None,
        *,
        pre_var="v",
        input=None,
        weight=1.0,
        delay=0.0,
        reduce="sum",
        transform=None,
        auto_expand=False,
        allow_autapses: Optional[bool] = None,
        allow_multapses: Optional[bool] = None,
    ):
        """Connect a continuous presynaptic variable to a ContinuousSynapse.

        Slot targets are supported in the same way as for event-based
        connections: a ``SynapseSlots`` target is already in synapse-local
        coordinates and avoids ambiguous colocated slots.
        """
        source, target, source_model, target_model = self._normalize_endpoint(
            source,
            target,
        )
        synapse = self._validate_target_synapse_endpoint(target, synapse)

        if not hasattr(synapse, "continuous_receive"):
            raise TypeError(
                "connect_continuous requires a target mechanism that implements "
                "continuous_receive; did you subclass ContinuousSynapse?"
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

        pre_pool = self._flat_selection(source)
        post_pool = self._flat_selection(target)

        if not isinstance(target, SynapseSlots):
            self._to_synapse_local_post_idx(target_model, post_pool, synapse)

        pre_idx, post_ids = self._edges_for_rule(
            rule=rule,
            spec=spec,
            source_model=source_model,
            target_model=target_model,
            pre_pool=pre_pool,
            post_pool=post_pool,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
            target_id_universe_size=self._target_id_universe_size(target, target_model),
            post_flat_by_target_id=self._post_flat_by_target_id(target),
        )

        if pre_idx.numel() == 0:
            return

        shape_context = self._connection_shape_context(
            kind="continuous",
            rule=rule,
            spec=spec,
            source_model=source_model,
            target_model=target_model,
            synapse=synapse,
            pre_pool=pre_pool,
            post_pool=post_pool,
            pre_idx=pre_idx,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
            auto_expand=auto_expand,
            pre_var=pre_var,
            input=input,
        )

        post_idx = self._target_post_idx(target_model, post_ids, synapse, target)

        if auto_expand:
            n_connections = pre_idx.numel()
            weight = expand(weight, n_connections)
            delay = expand(delay, n_connections)

        self._connect_continuous(
            source_model,
            pre_idx,
            target_model,
            post_idx,
            synapse,
            weight=weight,
            delay=delay,
            pre_var=pre_var,
            input=input,
            reduce=reduce,
            transform=transform,
            shape_context=shape_context,
        )

    def connect_continuous_one_to_one(
        self,
        source,
        target,
        synapse=None,
        *,
        pre_var="v",
        input=None,
        weight=1.0,
        delay=0.0,
        reduce="sum",
        transform=None,
        allow_autapses=False,
        allow_multapses=False,
    ):
        """One-to-one wrapper around :meth:`connect_continuous`."""
        return self.connect_continuous(
            source,
            target,
            synapse,
            conn_spec={"rule": "one_to_one"},
            pre_var=pre_var,
            input=input,
            weight=weight,
            delay=delay,
            reduce=reduce,
            transform=transform,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
        )

    def connect(
        self,
        source,
        target,
        synapse=None,
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

        ``target`` may be a physical population/slice, or a ``SynapseSlots``
        object returned by ``population.slots(...)``, ``slice.slots(...)``, or
        ``network.synapse_slots(...)``. Slot targets remove the ambiguity that
        arises when a banked point-process mechanism has several local slots on
        one physical compartment.
        """
        if isinstance(target, SynapseSlots):
            return self.connect_to_slots(
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

        source, target, source_model, target_model = self._normalize_endpoint(
            source,
            target,
        )
        synapse = self._validate_target_synapse_endpoint(target, synapse)

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

        # Preserve the previous contract for physical targets: all explicitly
        # selected target locations must host the requested synapse.  Ambiguous
        # banked targets now raise and instruct users to select local slots.
        self._to_synapse_local_post_idx(target_model, post_pool, synapse)

        pre_idx, post_ids = self._edges_for_rule(
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

        shape_context = self._connection_shape_context(
            kind="event",
            rule=rule,
            spec=spec,
            source_model=source_model,
            target_model=target_model,
            synapse=synapse,
            pre_pool=pre_pool,
            post_pool=post_pool,
            pre_idx=pre_idx,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
            auto_expand=auto_expand,
            pre_var=pre_var,
        )

        post_idx = self._target_post_idx(target_model, post_ids, synapse, target)

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
            shape_context=shape_context,
        )

    def connect_to_slots(
        self,
        source,
        target,
        synapse=None,
        *,
        slots=None,
        local_index=None,
        conn_spec=None,
        threshold=0.0,
        weight=1.0,
        delay=0.0,
        pre_var=None,
        auto_expand=False,
        allow_autapses: Optional[bool] = None,
        allow_multapses: Optional[bool] = None,
    ):
        """Connect sources to explicit synapse-local target slots.

        ``target`` may be a ``SynapseSlots`` object, or a population/slice plus
        ``synapse`` and ``slots``/``local_index``.  This is the public API for
        banked point processes with multiple colocated local slots.
        """
        if local_index is not None:
            if slots is not None:
                raise ValueError("Provide either slots or local_index, not both.")
            slots = local_index

        if isinstance(source, Population) or isinstance(source, NetStim):
            source = source[:]
        source_model = source.model

        target_slots = self.synapse_slots(target, synapse, slots=slots)
        target_model = target_slots.model
        synapse = target_slots.synapse

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
        post_pool = target_slots.local_index.to(
            device=source_model.device(), dtype=torch.long
        )

        pre_idx, post_idx = self._edges_for_rule(
            rule=rule,
            spec=spec,
            source_model=source_model,
            target_model=target_model,
            pre_pool=pre_pool,
            post_pool=post_pool,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
            target_id_universe_size=int(target_slots.slot_to_flat_index.numel()),
            post_flat_by_target_id=target_slots.slot_to_flat_index,
        )

        if pre_idx.numel() == 0:
            return

        shape_context = self._connection_shape_context(
            kind="event",
            rule=rule,
            spec=spec,
            source_model=source_model,
            target_model=target_model,
            synapse=synapse,
            pre_pool=pre_pool,
            post_pool=post_pool,
            pre_idx=pre_idx,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
            auto_expand=auto_expand,
            pre_var=pre_var,
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
            post_idx.to(device=target_model.device(), dtype=torch.long),
            synapse,
            threshold=threshold,
            weight=weight,
            delay=delay,
            pre_var=pre_var,
            shape_context=shape_context,
        )

    def connect_one_to_one_slots(
        self,
        source,
        target,
        synapse=None,
        *,
        slots=None,
        local_index=None,
        threshold=0.0,
        weight=1.0,
        delay=0.0,
        pre_var=None,
        allow_autapses=False,
        allow_multapses=False,
    ):
        """Connect source elements one-to-one with explicit target slots."""
        return self.connect_to_slots(
            source,
            target,
            synapse,
            slots=slots,
            local_index=local_index,
            conn_spec={"rule": "one_to_one"},
            threshold=threshold,
            weight=weight,
            delay=delay,
            pre_var=pre_var,
            allow_autapses=allow_autapses,
            allow_multapses=allow_multapses,
        )

    def connect_one_to_one(
        self,
        source,
        target,
        synapse=None,
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
        synapse=None,
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
            if pre_name == "netstim" and pre is None:
                raise RuntimeError(
                    "The wiring specification contains NetStim-sourced "
                    "connections, but no NetStim is attached. Call "
                    "network.attach_netstim(netstim) before build()."
                )
            post = self.populations[post_name]
            pre_device = pre.device()
            pre_dtype = pre.dtype()
            post_device = post.device()
            post_dtype = post.dtype()
            pre_idx = torch.cat([s[0] for s in specs])
            post_idx = torch.cat([s[1] for s in specs])
            thresholds = torch.cat(
                [expand(s[2], s[3], device=pre_device, dtype=pre_dtype) for s in specs]
            )
            weights = make_weight(
                [s[4] for s in specs],
                [s[5] for s in specs],
                device=post_device,
                dtype=post_dtype,
            )
            delay = make_weight(
                [s[6] for s in specs],
                [s[7] for s in specs],
                device=post_device,
                dtype=post_dtype,
            )
            pre_idx_for_var = self._pre_idx_for_pre_var(pre, pre_idx, pre_var)

            syn = NetCon(
                pre=pre,
                pre_idx=pre_idx_for_var,
                thresholds=thresholds,
                post=post,
                post_idx=post_idx,
                post_syn=synapse,
                weight=weights,
                delay=delay,
                dt=dt,
                pre_var=pre_var,
                max_delay=max_delay_ms,
                track_events=self.track_netcon_events,
                delay_backend=self.netcon_delay_backend,
                train_delay_backend=self.netcon_train_backend,
                device=post_device,
                dtype=post_dtype,
                pre_device=pre_device,
                pre_dtype=pre_dtype,
            )

            syn.setreference("t", lambda: self.t)

            if self.training:
                syn.train()
            else:
                syn.eval()

            self.synapses[
                f"{pre_name}:{pre_var.replace('.', '_')}->{post_name}:{synapse.name}"
            ] = syn

    def build_continuous_synapses(self, dt, max_delay_ms=None):
        """Materialize queued continuous connection specs into ContinuousCon modules.

        Connection specs are coalesced before construction, mirroring the
        event-based NetCon builder: all user connect_continuous calls with the
        same ``(pre_name, post_name, synapse, pre_var, input, reduce,
        transform)`` key are concatenated into one ContinuousCon.  Separately,
        Network owns a unique target-synapse reset list so each ContinuousSynapse
        input buffer is reset exactly once per timestep before all analog
        deliveries.
        """
        self.continuous_targets = {}
        for group_i, (
            (
                pre_name,
                post_name,
                synapse,
                pre_var,
                input_name,
                reduce,
                transform,
            ),
            specs,
        ) in enumerate(self.continuous_synapse_spec.items()):
            pre_var = pre_var if pre_var is not None else "v"
            pre = getattr(self, pre_name)
            if pre_name == "netstim" and pre is None:
                raise RuntimeError(
                    "The wiring specification contains NetStim-sourced "
                    "connections, but no NetStim is attached. Call "
                    "network.attach_netstim(netstim) before build()."
                )
            post = self.populations[post_name]
            pre_device = pre.device()
            pre_dtype = pre.dtype()
            post_device = post.device()
            post_dtype = post.dtype()
            pre_idx = torch.cat([s[0] for s in specs])
            post_idx = torch.cat([s[1] for s in specs])
            weights = make_weight(
                [s[2] for s in specs],
                [s[3] for s in specs],
                device=post_device,
                dtype=post_dtype,
            )
            delay = make_weight(
                [s[4] for s in specs],
                [s[5] for s in specs],
                device=post_device,
                dtype=post_dtype,
            )
            pre_idx_for_var = self._pre_idx_for_pre_var(pre, pre_idx, pre_var)

            target_name = f"{post_name}:{synapse.name}"
            if target_name not in self.continuous_targets:
                if not hasattr(synapse, "reset_continuous_inputs"):
                    raise TypeError(
                        "connect_continuous target must implement "
                        "reset_continuous_inputs()."
                    )
                self.continuous_targets[target_name] = synapse

            con = ContinuousCon(
                pre=pre,
                pre_idx=pre_idx_for_var,
                post=post,
                post_idx=post_idx,
                post_syn=synapse,
                weight=weights,
                delay=delay,
                dt=dt,
                pre_var=pre_var,
                input=input_name,
                reduce=reduce,
                max_delay=max_delay_ms,
                transform=transform,
                reset_inputs=False,
                device=post_device,
                dtype=post_dtype,
                pre_device=pre_device,
                pre_dtype=pre_dtype,
            )

            con.setreference("t", lambda: self.t)

            if self.training:
                con.train()
            else:
                con.eval()

            iname = input_name if input_name is not None else "input"
            # group_i makes the ModuleDict key collision-proof when distinct
            # transforms or other non-rendered key fields are used.
            cname = (
                f"{group_i}:{pre_name}:{pre_var.replace('.', '_')}~>{post_name}:"
                f"{synapse.name}:{iname}:{reduce}"
            )
            self.continuous_synapses[cname] = con

    def _refresh_step_schedule(self):
        """Cache accelerator-first launch order for timestep execution."""
        self._population_step_items = _accelerator_first_step_items(self.populations)
        self._synapse_step_items = _accelerator_first_step_items(self.synapses)
        self._continuous_synapse_step_items = _accelerator_first_step_items(
            self.continuous_synapses
        )
        self._continuous_target_step_items = _accelerator_first_step_items(
            self.continuous_targets
        )

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
        dt_f = _validate_time_scalar(dt, name="dt", positive=True)
        self._refresh_compile_config_from_ctx()
        current_sig = self._device_signature()
        devices_changed = current_sig != getattr(self, "_device_sig", None)
        dt_changed = self.dt is not None and self.dt != dt_f

        if not self.built or self.dt != dt_f or force_rebuild or devices_changed:
            torch._dynamo.reset()
            self.dt = dt_f
            if dt_changed:
                # A step count anchored to the previous dt cannot be reused with
                # a new timestep. Preserve the current absolute time and re-anchor.
                self._reset_runtime_clock(float(self.t))
            self.synapses.clear()
            self.continuous_synapses.clear()
            self.continuous_targets = {}
            self.build_synapses(dt_f, max_delay_ms=max_delay_ms)
            self.build_continuous_synapses(dt_f, max_delay_ms=max_delay_ms)
            self.built = True
            self._device_sig = self._device_signature()
        self._refresh_step_schedule()
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
            Re-sample or refresh synaptic weights. Default is True. When
            initializing from a steady-state cache after an optimizer update,
            leave this True so expanded ``WeightExpander.w`` buffers are
            regenerated from the current weight parameters before cached pending
            deliveries are rebuilt.
        reinit_delays : bool, optional
            Re-sample or refresh synaptic delays. Default is True. When
            initializing from a steady-state cache after delay parameters have
            changed, leave this True so expanded delay values and integer delay
            metadata are regenerated before cache restore.
        t : float, optional
            Starting simulation time (ms). Finite negative values are allowed;
            booleans and non-finite values are rejected. Default is 0.0.
        max_delay_ms : float, optional
            Maximum allowed synaptic delay. Default is None.
        force_rebuild : bool, optional
            If True, rebuild NetCons even if already built. Default is False.

        Returns
        -------
        Network
            Self, for chaining.
        """
        dt_f = _validate_time_scalar(dt, name="dt", positive=True)
        t_f = _validate_time_scalar(t, name="t", positive=None)
        self._refresh_compile_config_from_ctx()

        has_state_cache = bool(self._state_cache)
        if has_state_cache:
            self.initialize_pops_from_state_cache(dt=dt_f)

        for pop in self.populations.values():
            if not has_state_cache:
                pop.initialize()
            dt_pop = torch.tensor(dt_f, device=pop.device(), dtype=pop.dtype())
            pop.integrator._initialize(
                pop,
                dt_pop,
                force=pop.force_integrator_reinit(),
                compile_scope="network_population",
            )
            pop.intra = pop.build_intra()

        # NetCons derive their absolute scheduled-event step from ``self.t``
        # during initialization, so synchronize all clocks before building or
        # resetting connection state.
        self._reset_runtime_clock(t_f)
        self._clear_duration_remainder()
        self.build(dt_f, max_delay_ms=max_delay_ms, force_rebuild=force_rebuild)
        # Always clear runtime delivery state during synapse initialization.
        # If a state cache is present, the backend-specific restore below will
        # rebuild the pending traffic from cached presynaptic history using the
        # freshly expanded current weights/delays.
        self.init_synapses(
            reinit_weights=reinit_weights,
            reinit_delays=reinit_delays,
            clear_deliveries=True,
        )
        if has_state_cache:
            self.initialize_synapses_from_state_cache(
                reinit_weights=reinit_weights,
                reinit_delays=reinit_delays,
            )
        if self.netstim is not None:
            if hasattr(self.netstim, "set_dt"):
                self.netstim.set_dt(dt_f)
            self.netstim.initialize()
            self.netstim.detach()
        self._mode_requires_initialize = False
        return self

    def initialize_pops_from_state_cache(self, *, dt=None):
        for name, pop in self.populations.items():
            # A cache can be loaded into a fresh, structurally equivalent
            # network. Shape timestep-dependent integrator buffers before the
            # strict state-dict load so scalar construction defaults (for
            # example ``cmdt``) match initialized cached tensors.
            pop.build()
            if dt is not None:
                dt_pop = torch.tensor(float(dt), device=pop.device(), dtype=pop.dtype())
                pop.integrator._initialize(
                    pop,
                    dt_pop,
                    force=pop.force_integrator_reinit(),
                    compile_scope="network_population",
                )
            pop.load_state_dict(self._state_cache[name])
            pop.detach()
            pop.initialized = True
            pop.initializing_from_state_cache = True

    def initialize_synapses_from_state_cache(
        self,
        *,
        reinit_weights: bool = True,
        reinit_delays: bool = True,
    ):
        """Restore event and continuous synapse runtime caches.

        Connection classes own their backend-specific cache format.  Network only
        routes the cached payloads to the matching module names.  This method is
        intentionally called after :meth:`init_synapses`: expanded
        ``WeightExpander.w`` buffers and delay metadata must already reflect the
        current parameters before cached presynaptic history is converted back
        into pending deliveries.  A legacy fallback is retained for older caches
        whose event NetCon entries were stored as
        ``(old_dt, has_spiked, is_spiking, delivery_buffer)`` tuples.

        Keep ``reinit_weights=True`` and ``reinit_delays=True`` when restoring a
        steady-state cache after optimizer updates.  Set either flag False only
        when the corresponding expanded parameter values should be reused.
        """
        if not self._syn_cache:
            return

        if "event" in self._syn_cache or "continuous" in self._syn_cache:
            event_cache = self._syn_cache.get("event", {})
            continuous_cache = self._syn_cache.get("continuous", {})
        else:
            # Backward compatibility with the previous flat event-NetCon cache.
            event_cache = self._syn_cache
            continuous_cache = {}

        for name, syn in self.synapses.items():
            if name not in event_cache:
                continue
            cache = event_cache[name]
            if hasattr(syn, "initialize_from_state_cache"):
                try:
                    syn.initialize_from_state_cache(
                        cache,
                        dt=self.dt,
                        rebuild_delays=False,
                    )
                except TypeError:
                    # Backward-compatible call for custom connection modules
                    # that implemented the cache API before ``rebuild_delays``.
                    syn.initialize_from_state_cache(cache, dt=self.dt)
            else:
                old_dt, has_spiked, is_spiking, delivery_buffer = cache
                n_limit = syn.delivery_buffer.shape[0]
                delivery_buffer = dilate(
                    delivery_buffer, float(old_dt), float(self.dt), n_limit=n_limit
                )
                if not syn.skip_thresholding:
                    syn.has_spiked = (has_spiked).detach()
                    syn.is_spiking = (is_spiking).detach()
                syn.delivery_buffer = delivery_buffer.clone().detach()

        for name, syn in self.continuous_synapses.items():
            if name not in continuous_cache:
                continue
            cache = continuous_cache[name]
            if not hasattr(syn, "initialize_from_state_cache"):
                raise RuntimeError(
                    f"Continuous synapse {name!r} does not implement initialize_from_state_cache()."
                )
            try:
                syn.initialize_from_state_cache(
                    cache,
                    dt=self.dt,
                    rebuild_delays=False,
                )
            except TypeError:
                syn.initialize_from_state_cache(cache, dt=self.dt)

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
        for syn in self.continuous_synapses.values():
            syn.initialize(
                reinit_weights=reinit_weights,
                reinit_delays=reinit_delays,
                clear_deliveries=clear_deliveries,
            )

    def step(self, *, extra=None, callbacks=None, loop_hooks: bool = False):
        """Advance this network by exactly one timestep.

        The timestep is the network timestep established by ``initialize(dt)``
        or ``build(dt)``. This method preserves the same event-delivery phase
        ordering as :meth:`run`, but avoids constructing a one-step simulation
        loop.

        Parameters
        ----------
        extra : dict, optional
            Optional mapping from population name to extracellular stimulus
            specification ``(v, t)``, with the same semantics as :meth:`run`.
            The stimulus is evaluated at the network's current time for a
            single timestep.
        callbacks : sequence of Callback, optional
            Callbacks to execute around this single step. By default ``step``
            calls ``pre_step_hook`` and ``post_step_hook`` only.
        loop_hooks : bool, optional
            If ``True``, also call ``pre_loop_hook`` before the step and
            ``post_loop_hook`` after the step.

        Returns
        -------
        Network
            ``self``, for chaining.
        """
        if self.dt is None:
            raise RuntimeError(
                "Network has no simulation timestep. Call initialize(dt) or "
                "build(dt) before step()."
            )
        if not self.built:
            raise RuntimeError(
                "Network wiring has changed since the last build. Call "
                "initialize(dt) (recommended) or build(dt) before step()."
            )
        self._require_mode_runtime_ready()

        dt_f = _validate_time_scalar(self.dt, name="dt", positive=True)
        self._refresh_compile_config_from_ctx()
        self._sync_runtime_clock()

        ctx = nullcontext() if self.training else torch.no_grad()

        intra = {}
        for n, p in self.populations.items():
            # Mirror Population.step/run: new injections set ``p.intra = None``.
            # Rebuild lazily so one-step network stepping respects intracellular
            # currents added after initialization.
            if p.intra is None and getattr(p, "injections", None):
                p.intra = p.build_intra()
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
            t1 = t0 + dt_pop
            extra_prepped[n] = (v_dev, t_dev.assemble(t0, t1, dt_pop))
        extra = extra_prepped

        with_intra = bool(intra)
        with_extra = bool(extra)

        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        for c in callbacks:
            c.dt = self.dt

        with ctx:
            if with_intra:
                intra = {
                    n: (intra_, *self.populations[n].prep_intra(intra_, 1, dt_f))
                    for n, intra_ in intra.items()
                }

            if loop_hooks:
                pre_loop_hook(callbacks, self)
            pre_step_hook(callbacks, self)

            intra_c = prepare_intra({}, intra, 0) if with_intra else {}
            extra_c = prepare_extra(extra, 0) if with_extra else {}

            self._step(
                self._population_step_items,
                self._synapse_step_items,
                self._continuous_synapse_step_items,
                self._continuous_target_step_items,
                self.netstim,
                self.t,
                dt_f,
                extra=extra_c,
                intra=intra_c,
                compile_network_ops=self.compile_network_ops,
            )
            self._advance_runtime_clock()

            post_step_hook(callbacks, self)
            if loop_hooks:
                post_loop_hook(callbacks, self)

        return self

    def run(self, tstop, extra=None, callbacks=None, progressbar=False):
        """
        Advance the network for a fixed duration.

        Parameters
        ----------
        tstop : float
            Requested simulation duration in milliseconds. Only complete
            timesteps are executed. Any fractional remainder is retained and
            combined with the next duration-based call, so partitioned calls
            advance the same number of steps as their combined duration.
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

        Notes
        -----
        The call that completes a retained fractional timestep supplies that
        step's ``extra`` input and callbacks; inputs from earlier no-step calls
        are not buffered.
        """
        if self.dt is None:
            raise RuntimeError(
                "Network has no simulation timestep. Call initialize(dt) or "
                "build(dt) before run()."
            )
        if not self.built:
            raise RuntimeError(
                "Network wiring has changed since the last build. Call "
                "initialize(dt) (recommended) or build(dt) before run()."
            )
        self._require_mode_runtime_ready()
        dt_f = _validate_time_scalar(self.dt, name="dt", positive=True)
        tstop_f = _validate_time_scalar(tstop, name="tstop", positive=False)
        self._refresh_compile_config_from_ctx()
        self._sync_runtime_clock()

        n_steps, duration_remainder = self._duration_budget(tstop_f, dt_f)

        ctx = nullcontext() if self.training else torch.no_grad()

        intra = {}
        for n, p in self.populations.items():
            if p.intra is None and getattr(p, "injections", None):
                p.intra = p.build_intra()
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
            t1 = t0 + n_steps * dt_pop
            extra_prepped[n] = (v_dev, t_dev.assemble(t0, t1, dt_pop))
        extra = extra_prepped

        with_intra = bool(intra)
        with_extra = bool(extra)
        tstart = self.t.item()

        with ctx:
            self._set_duration_remainder(duration_remainder)

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

                pre_step_hook(callbacks, self)
                self._step(
                    self._population_step_items,
                    self._synapse_step_items,
                    self._continuous_synapse_step_items,
                    self._continuous_target_step_items,
                    self.netstim,
                    self.t,
                    dt_f,
                    extra=extra_c,
                    intra=intra_c,
                    compile_network_ops=self.compile_network_ops,
                )
                self._advance_runtime_clock()
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

    def longrun(
        self,
        tstop,
        chunklength,
        extra=None,
        callbacks=None,
        progressbar=False,
    ):
        """Advance the network in bounded-memory chunks without resetting state.

        ``longrun`` has the same timestep and event-delivery semantics as
        :meth:`run`, but prepares intracellular and extracellular stimuli one
        chunk at a time. Population, NetStim, and NetCon state remains live
        across chunk boundaries. Callback loop hooks run once for the complete
        call, while chunk hooks run once around each chunk.

        Parameters
        ----------
        tstop : float
            Requested simulation duration in milliseconds. Only complete
            timesteps are executed; a fractional remainder is retained and
            combined with the next :meth:`run`, :meth:`longrun`, or
            :meth:`longrun_checkpointed` call.
        chunklength : int
            Positive number of timesteps per chunk. The final chunk may be
            shorter.
        extra : dict[str, tuple[torch.Tensor, object]], optional
            Population-specific extracellular stimuli with the same public
            format as :meth:`run`.
        callbacks : sequence of Callback, optional
            Callbacks invoked around the complete loop, each chunk, and each
            timestep.
        progressbar : bool or tqdm.tqdm, optional
            If truthy, display progress over completed timesteps.

        Returns
        -------
        None
        """
        if self.dt is None:
            raise RuntimeError(
                "Network has no simulation timestep. Call initialize(dt) or "
                "build(dt) before longrun()."
            )
        if not self.built:
            raise RuntimeError(
                "Network wiring has changed since the last build. Call "
                "initialize(dt) (recommended) or build(dt) before longrun()."
            )
        self._require_mode_runtime_ready()
        if not isinstance(chunklength, int) or isinstance(chunklength, bool):
            raise ValueError("chunklength must be a positive integer.")
        if chunklength <= 0:
            raise ValueError("chunklength must be a positive integer.")

        dt_f = _validate_time_scalar(self.dt, name="dt", positive=True)
        tstop_f = _validate_time_scalar(tstop, name="tstop", positive=False)
        self._refresh_compile_config_from_ctx()
        self._sync_runtime_clock()

        ctx = nullcontext() if self.training else torch.no_grad()

        intra = {}
        for name, pop in self.populations.items():
            if pop.intra is None and getattr(pop, "injections", None):
                pop.intra = pop.build_intra()
            if pop.intra is not None:
                intra[name] = pop.intra

        extra = extra if extra is not None else {}
        extra_sources = {}
        for name, (spatial, temporal) in extra.items():
            pop = self.populations[name]
            device, dtype = pop.device(), pop.dtype()
            extra_sources[name] = (
                spatial.to(device=device, dtype=dtype),
                temporal.to(device=device, dtype=dtype),
            )

        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        for callback in callbacks:
            callback.dt = self.dt

        n_steps, duration_remainder = self._duration_budget(tstop_f, dt_f)
        initial_time = self.t.double().detach().clone()
        tstart = self.t.item()

        with ctx:
            self._set_duration_remainder(duration_remainder)
            pre_loop_hook(callbacks, self)

            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(total=n_steps, desc=f"{tstart:.1f} ms")

            completed_steps = 0
            for chunk_start in range(0, n_steps, chunklength):
                n_chunk = min(chunklength, n_steps - chunk_start)
                time_chunk = initial_time + dt_f * torch.arange(
                    chunk_start,
                    chunk_start + n_chunk,
                    device=self.t.device,
                    dtype=torch.double,
                )
                time_chunk = time_chunk.to(dtype=self.t.dtype)
                intra_chunk = {
                    name: (
                        source,
                        *self.populations[name].prep_intra(source, n_chunk, dt_f),
                    )
                    for name, source in intra.items()
                }

                extra_chunk = {}
                for name, (spatial, temporal) in extra_sources.items():
                    pop = self.populations[name]
                    device, dtype = pop.device(), pop.dtype()
                    dt_pop = torch.tensor(self.dt, device=device, dtype=dtype)
                    start = self.t.to(device=device, dtype=dtype)
                    stop = start + n_chunk * dt_pop
                    extra_chunk[name] = (
                        spatial,
                        temporal.assemble(start, stop, dt_pop),
                    )

                pre_chunk_hook(callbacks, self, time_chunk)
                for local_ind in range(n_chunk):
                    intra_c = (
                        prepare_intra({}, intra_chunk, local_ind) if intra_chunk else {}
                    )
                    extra_c = (
                        prepare_extra(extra_chunk, local_ind) if extra_chunk else {}
                    )

                    pre_step_hook(callbacks, self)
                    self._step(
                        self._population_step_items,
                        self._synapse_step_items,
                        self._continuous_synapse_step_items,
                        self._continuous_target_step_items,
                        self.netstim,
                        self.t,
                        dt_f,
                        extra=extra_c,
                        intra=intra_c,
                        compile_network_ops=self.compile_network_ops,
                    )
                    self._advance_runtime_clock()
                    post_step_hook(callbacks, self)
                    completed_steps += 1

                post_chunk_hook(callbacks, self, time_chunk)

                if progressbar:
                    progressbar.update(n_chunk)
                    progressbar.set_description(
                        f"{tstart + completed_steps * dt_f:.1f} ms"
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
        _continuous_synapse_spec = self.continuous_synapse_spec.copy()

        # Connection specs store source indices in the source-population flat
        # coordinate frame, but store target indices in the target *synapse*
        # local coordinate frame.  This distinction matters for banked
        # point-processes: several local synapse slots may live on the same
        # physical compartment, and the mechanism's slot axis can be much
        # smaller/larger than the owning population's compartment axis.  Capture
        # both coordinate-frame shapes before populations are rebuilt below.
        _old_population_shapes = {}
        _old_synapse_shapes = {}
        for name, p in self.populations.items():
            _old_population_shapes[name] = tuple(p.shape)
        if self.netstim is not None:
            _old_population_shapes["netstim"] = tuple(self.netstim.shape)

        for source_name, target_name, synapse, pre_var in _synapse_spec.keys():
            _old_synapse_shapes[(target_name, synapse.name)] = tuple(synapse.shape_f)
        for (
            source_name,
            target_name,
            synapse,
            pre_var,
            input_name,
            reduce,
            transform,
        ) in _continuous_synapse_spec.keys():
            _old_synapse_shapes[(target_name, synapse.name)] = tuple(synapse.shape_f)

        self.clear_synapses()

        for name, p in self.populations.items():
            device = p.device()
            p.batch(n)
            p.build(force_rebuild=True)
            p.to(device)
        if include_netstim and self.netstim is not None:
            self.netstim.batch(n)
        for k, v in _synapse_spec.items():
            source_name, target_name, synapse, pre_var = k
            source_pop = getattr(self, source_name)
            target_pop = getattr(self, target_name)
            synapse_name = synapse.name
            target_slot_shape = _old_synapse_shapes[(target_name, synapse_name)]
            synapse = getattr(target_pop.mech, synapse_name)
            for data in v:
                source_idx, target_idx = data[0], data[1]
                threshold, weight, delay = data[2], data[4], data[6]
                if source_name == "netstim" and not include_netstim:
                    new_source_idx = source_idx.repeat(n)
                else:
                    new_source_idx = batchify_index(
                        _old_population_shapes[source_name], n, source_idx
                    )
                new_target_idx = batchify_index(target_slot_shape, n, target_idx)
                self._connect(
                    source_pop,
                    new_source_idx,
                    target_pop,
                    new_target_idx,
                    synapse,
                    threshold,
                    _evaluate(weight),
                    _evaluate(delay),
                    pre_var=pre_var,
                )
        for k, v in _continuous_synapse_spec.items():
            (
                source_name,
                target_name,
                synapse,
                pre_var,
                input_name,
                reduce,
                transform,
            ) = k
            source_pop = getattr(self, source_name)
            target_pop = getattr(self, target_name)
            synapse_name = synapse.name
            target_slot_shape = _old_synapse_shapes[(target_name, synapse_name)]
            synapse = getattr(target_pop.mech, synapse_name)
            for data in v:
                source_idx, target_idx = data[0], data[1]
                weight, delay = data[2], data[4]
                if source_name == "netstim" and not include_netstim:
                    new_source_idx = source_idx.repeat(n)
                else:
                    new_source_idx = batchify_index(
                        _old_population_shapes[source_name], n, source_idx
                    )
                new_target_idx = batchify_index(target_slot_shape, n, target_idx)
                self._connect_continuous(
                    source_pop,
                    new_source_idx,
                    target_pop,
                    new_target_idx,
                    synapse,
                    weight=_evaluate(weight),
                    delay=_evaluate(delay),
                    pre_var=pre_var,
                    input=input_name,
                    reduce=reduce,
                    transform=transform,
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
        new_net = Network(
            new_populations,
            netstim=self.netstim,
            track_netcon_events=self.track_netcon_events,
            netcon_delay_backend=self.netcon_delay_backend,
            netcon_train_backend=self.netcon_train_backend,
        )

        all_indices = indices(concat_pops)
        all_indices = {n: i.flatten() for n, i in zip(pops_to_concatenate, all_indices)}

        def remapped_endpoint(population_name):
            if population_name in pops_to_concatenate:
                return new_net.populations[name]
            if population_name == "netstim":
                if new_net.netstim is None:
                    raise RuntimeError(
                        "Cannot restore a NetStim connection while concatenating "
                        "a network without an attached NetStim."
                    )
                return new_net.netstim
            return new_net.populations[population_name]

        # now reapply connections
        for k, v in self.synapse_spec.items():
            source_name, target_name, synapse, pre_var = k
            source_pop = remapped_endpoint(source_name)
            target_pop = remapped_endpoint(target_name)
            synapse = getattr(target_pop.mech, synapse.name)

            for data in v:
                source_idx, target_idx = data[0], data[1]
                threshold, weight, delay = data[2], data[4], data[6]

                if source_name in pops_to_concatenate:
                    source_idx = all_indices[source_name][source_idx]

                # target_idx is local to the target synapse, not population-flat.
                # Reinserted component synapses preserve their own local ordering
                # inside the concatenated population, so it must not be offset by
                # the owning population's compartment position.

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

        for k, v in self.continuous_synapse_spec.items():
            (
                source_name,
                target_name,
                synapse,
                pre_var,
                input_name,
                reduce,
                transform,
            ) = k
            source_pop = remapped_endpoint(source_name)
            target_pop = remapped_endpoint(target_name)
            synapse = getattr(target_pop.mech, synapse.name)

            for data in v:
                source_idx, target_idx = data[0], data[1]
                weight, delay = data[2], data[4]
                if source_name in pops_to_concatenate:
                    source_idx = all_indices[source_name][source_idx]

                new_net._connect_continuous(
                    source_pop,
                    source_idx,
                    target_pop,
                    target_idx,
                    synapse,
                    weight=weight,
                    delay=delay,
                    pre_var=pre_var,
                    input=input_name,
                    reduce=reduce,
                    transform=transform,
                )

        return new_net

    def _enable_synapse_state_cache_recording(self):
        """Ask connection modules to record parameter-invariant cache histories."""
        for syn in self.synapses.values():
            hook = getattr(syn, "enable_state_cache_recording", None)
            if hook is not None:
                hook()
        for syn in self.continuous_synapses.values():
            hook = getattr(syn, "enable_state_cache_recording", None)
            if hook is not None:
                hook()

    def _disable_synapse_state_cache_recording(self, *, release: bool = False):
        for syn in self.synapses.values():
            hook = getattr(syn, "disable_state_cache_recording", None)
            if hook is not None:
                hook(release=release)
        for syn in self.continuous_synapses.values():
            hook = getattr(syn, "disable_state_cache_recording", None)
            if hook is not None:
                hook(release=release)

    def steady_state(self, tstop=1000, dt=0.025, progressbar=False, max_delay_ms=None):
        """
        Run the network until it reaches a steady state.

        Parameters
        ----------
        tstop : float, optional
            Total time to run the network in ms. Default is 1000 ms.
        dt : float, optional
            Time step in ms. Default is 0.025 ms.
        progressbar : bool, optional
            If True, show a progress bar during the steady-state run.
        max_delay_ms : float, optional
            Optional delay-horizon cap used when building synapse state for the
            steady-state/cache run. Supplying the largest delay expected in
            later simulations lets the cache retain enough history when delays
            grow after optimization updates. Later calls to ``initialize`` may
            keep the default ``reinit_weights=True`` and ``reinit_delays=True``;
            the cached synapse histories are parameter-invariant and are replayed
            through the current expanded weights and delays.

        Returns
        -------
        Network
            The network after running to steady state.
        """
        self.clear_state_cache()
        was_training = self.training
        with torch.no_grad():
            try:
                if self.netstim is not None:
                    self.netstim._prep_start_for_steady_state(tstop)
                self.eval()
                self.initialize(
                    dt,
                    t=-tstop,
                    max_delay_ms=max_delay_ms,
                    force_rebuild=True,
                )
                # Record unweighted event/value histories during the steady-state
                # run.  This lets the later cache restore rebuild pending
                # deliveries with changed dt, weights, and delays.
                self._enable_synapse_state_cache_recording()
                self.run(tstop, progressbar=progressbar)
                self.cache_state()
            finally:
                self._disable_synapse_state_cache_recording(release=True)
                if self.netstim is not None:
                    self.netstim._reset_start_times()
        if was_training:
            self.train()
        return self

    def cache_state(self):
        """Cache population and connection runtime state for steady_state().

        Populations still use their normal state_dict().  Event NetCons and
        ContinuousCons package their own backend-specific delay/runtime state so
        Network does not need to know whether a NetCon is dense, sparse-calendar,
        or bitpacked-history backed.
        """
        self._state_cache.clear()
        self._syn_cache.clear()
        for name, pop in self.populations.items():
            live_state = pop.state_dict()
            cached_state = live_state.__class__(
                (
                    key,
                    value.detach().clone() if torch.is_tensor(value) else value,
                )
                for key, value in live_state.items()
            )
            if hasattr(live_state, "_metadata"):
                cached_state._metadata = live_state._metadata.copy()
            self._state_cache[name] = cached_state

        event_cache = {}
        for name, syn in self.synapses.items():
            if hasattr(syn, "state_cache"):
                event_cache[name] = syn.state_cache()
            else:
                # Legacy dense fallback.
                event_cache[name] = (
                    syn.dt,
                    syn.has_spiked.clone(),
                    syn.is_spiking.clone(),
                    torch.roll(
                        syn.delivery_buffer, -syn.current_time_step.item(), dims=0
                    ).clone(),
                )

        continuous_cache = {}
        for name, syn in self.continuous_synapses.items():
            if not hasattr(syn, "state_cache"):
                raise RuntimeError(
                    f"Continuous synapse {name!r} does not implement state_cache()."
                )
            continuous_cache[name] = syn.state_cache()

        self._syn_cache["event"] = event_cache
        self._syn_cache["continuous"] = continuous_cache

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
        if self.netstim is not None and hasattr(self.netstim, "set_diff_config"):
            # Propagate only the scheduled-time pieces NetStim understands.
            self.netstim.set_diff_config(
                diff_scheduled_times=kwargs.get("diff_scheduled_times", True),
                sched_width=kwargs.get("sched_width", 1.0),
            )

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
        if isinstance(state_dict, (str, os.PathLike)):
            state_dict = torch.load(
                state_dict, map_location=self.device(), weights_only=True
            )
        has_duration_remainder = "_duration_remainder" in state_dict
        _load_compatible_state_dict_transactionally(self, state_dict)
        if not has_duration_remainder:
            self._clear_duration_remainder()
        else:
            self._set_duration_remainder(
                float(self._duration_remainder.detach().cpu().item())
            )
        self._reanchor_runtime_clock_from_time()
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
            "event": {
                name: syn.state_dict_for_checkpoint()
                for name, syn in self.synapses.items()
            },
            "continuous": {
                name: syn.state_dict_for_checkpoint()
                for name, syn in self.continuous_synapses.items()
            },
        }

    def netstim_state_dict_for_checkpoint(self):
        """
        Returns a state dict of the NetStim suitable for checkpointing.
        """
        if self.netstim is None:
            return None
        return self.netstim.state_dict_for_checkpoint()

    def checkpoint_structure(self):
        """Return the immutable runtime structure required for exact replay."""
        self._require_mode_runtime_ready()
        checkpoint_dt = None if self.dt is None else float(self.dt)
        if checkpoint_dt is not None and (
            not math.isfinite(checkpoint_dt) or checkpoint_dt <= 0.0
        ):
            raise ValueError(
                "Network checkpoint timestep must be positive and finite, "
                f"got {checkpoint_dt!r}."
            )
        return {
            "dt": checkpoint_dt,
            "training": bool(self.training),
            "populations": {
                name: {
                    "class": f"{type(pop).__module__}.{type(pop).__qualname__}",
                    "shape": tuple(int(dim) for dim in pop.shape),
                }
                for name, pop in self.populations.items()
            },
            "event": {
                name: syn.checkpoint_topology_signature()
                for name, syn in self.synapses.items()
            },
            "continuous": {
                name: syn.checkpoint_topology_signature()
                for name, syn in self.continuous_synapses.items()
            },
            "has_netstim": self.netstim is not None,
            "netstim_shape": (
                None
                if self.netstim is None
                else tuple(int(dim) for dim in self.netstim.shape)
            ),
        }

    def state_dict_for_checkpoint(self):
        """
        Return the mutable runtime state required to continue an exact replay.

        Weights, delays, and other model parameters remain in ``state_dict()``;
        callers restoring into a fresh object must load or construct matching
        parameters separately. The structure fingerprint rejects mismatched
        population/connection layouts and runtime backend modes before mutation.
        """
        self._require_mode_runtime_ready()
        return {
            "populations": self.populations_state_dict_for_checkpoint(),
            "netcons": self.netcons_state_dict_for_checkpoint(),
            "netstim": self.netstim_state_dict_for_checkpoint(),
            "t": self.t,
            "clock_origin": self._clock_origin,
            "clock_step": self._clock_step,
            "duration_remainder": self._duration_remainder,
            "structure": self.checkpoint_structure(),
        }

    def _restore_dict_from_checkpoint_unchecked(self, state_dict):
        if not isinstance(state_dict, Mapping):
            raise TypeError("Network checkpoint state must be a mapping.")
        missing = [
            name
            for name in ("populations", "netcons", "netstim", "t")
            if name not in state_dict
        ]
        if missing:
            raise KeyError(f"Network checkpoint is missing entries: {missing}.")
        if not isinstance(state_dict["populations"], Mapping):
            raise TypeError("Network checkpoint populations must be a mapping.")
        if not isinstance(state_dict["netcons"], Mapping):
            raise TypeError("Network checkpoint netcons must be a mapping.")

        checkpoint_t = state_dict["t"]
        if not torch.is_tensor(checkpoint_t):
            raise TypeError("Network checkpoint time must be a tensor.")
        if tuple(checkpoint_t.shape) != tuple(self.t.shape):
            raise ValueError(
                f"Network checkpoint time has shape {tuple(checkpoint_t.shape)}, "
                f"expected {tuple(self.t.shape)}."
            )
        restored_t = checkpoint_t.to(device=self.t.device, dtype=self.t.dtype).clone()
        if not bool(torch.isfinite(restored_t).item()):
            raise ValueError("Network checkpoint time must be finite.")

        has_clock_origin = "clock_origin" in state_dict
        has_clock_step = "clock_step" in state_dict
        checkpoint_origin = state_dict.get("clock_origin", checkpoint_t)
        if not torch.is_tensor(checkpoint_origin):
            raise TypeError("Network checkpoint clock origin must be a tensor.")
        if tuple(checkpoint_origin.shape) != tuple(self._clock_origin.shape):
            raise ValueError(
                "Network checkpoint clock origin has shape "
                f"{tuple(checkpoint_origin.shape)}, expected "
                f"{tuple(self._clock_origin.shape)}."
            )
        restored_origin = checkpoint_origin.to(
            device=self._clock_origin.device, dtype=self._clock_origin.dtype
        ).clone()
        if not bool(torch.isfinite(restored_origin).item()):
            raise ValueError("Network checkpoint clock origin must be finite.")

        checkpoint_step = state_dict.get(
            "clock_step", torch.zeros_like(self._clock_step)
        )
        if not torch.is_tensor(checkpoint_step):
            raise TypeError("Network checkpoint clock step must be a tensor.")
        if tuple(checkpoint_step.shape) != tuple(self._clock_step.shape):
            raise ValueError(
                "Network checkpoint clock step has shape "
                f"{tuple(checkpoint_step.shape)}, expected "
                f"{tuple(self._clock_step.shape)}."
            )
        if checkpoint_step.dtype != torch.long:
            raise TypeError("Network checkpoint clock step must have dtype torch.long.")
        restored_step = checkpoint_step.to(device=self._clock_step.device).clone()
        if int(restored_step.item()) < 0:
            raise ValueError("Network checkpoint clock step must be non-negative.")

        checkpoint_remainder = state_dict.get(
            "duration_remainder", torch.zeros_like(self._duration_remainder)
        )
        if not torch.is_tensor(checkpoint_remainder):
            raise TypeError("Network checkpoint duration remainder must be a tensor.")
        if tuple(checkpoint_remainder.shape) != tuple(self._duration_remainder.shape):
            raise ValueError(
                "Network checkpoint duration remainder has shape "
                f"{tuple(checkpoint_remainder.shape)}, expected "
                f"{tuple(self._duration_remainder.shape)}."
            )
        if not checkpoint_remainder.is_floating_point():
            raise TypeError(
                "Network checkpoint duration remainder must have a floating dtype."
            )
        restored_remainder = checkpoint_remainder.to(
            device=self._duration_remainder.device,
            dtype=torch.float64,
        ).clone()
        if not bool(torch.isfinite(restored_remainder).item()):
            raise ValueError("Network checkpoint duration remainder must be finite.")
        if float(restored_remainder.item()) < 0.0:
            raise ValueError(
                "Network checkpoint duration remainder must be non-negative."
            )
        if self.dt is not None:
            elapsed = restored_step.to(restored_t.dtype) * float(self.dt)
            metadata_t = restored_origin + elapsed
            # A checkpoint produced by this runtime contains a coherent
            # origin/step pair. Preserve that original anchor exactly so a
            # restore cannot introduce a one-ulp continuation difference through
            # subtractive re-anchoring. ``t`` remains the historically public
            # authority for legacy checkpoints and for callers that intentionally
            # replace only ``t``: absent or inconsistent metadata is re-anchored.
            if not (
                has_clock_origin
                and has_clock_step
                and torch.equal(metadata_t, restored_t)
            ):
                restored_origin = restored_t - elapsed
            if not bool(torch.isfinite(restored_origin).item()):
                raise ValueError(
                    "Network checkpoint clock metadata implies a non-finite origin."
                )

        checkpoint_structure = state_dict.get("structure", None)
        if checkpoint_structure is not None:
            if not isinstance(checkpoint_structure, Mapping):
                raise TypeError("Network checkpoint structure must be a mapping.")
            expected_structure = self.checkpoint_structure()
            if checkpoint_structure != expected_structure:
                raise ValueError(
                    "Network checkpoint topology or runtime mode does not match "
                    "the receiving network."
                )

        checkpoint_populations = set(state_dict["populations"])
        expected_populations = set(self.populations)
        missing_populations = expected_populations - checkpoint_populations
        unknown_populations = checkpoint_populations - expected_populations
        if missing_populations or unknown_populations:
            raise KeyError(
                "Network checkpoint population names do not match the network: "
                f"missing={sorted(missing_populations)}, "
                f"unknown={sorted(unknown_populations)}."
            )

        netcons = state_dict["netcons"]
        # Backward compatibility: older checkpoints stored only event NetCons as
        # a flat mapping.  New checkpoints separate event and continuous
        # connection states.
        if "event" in netcons or "continuous" in netcons:
            missing_sections = {
                name for name in ("event", "continuous") if name not in netcons
            }
            if missing_sections:
                raise KeyError(
                    "Separated Network checkpoints must contain both event and "
                    f"continuous NetCon sections; missing={sorted(missing_sections)}."
                )
            event_states = netcons.get("event", {})
            continuous_states = netcons.get("continuous", {})
            if not isinstance(event_states, Mapping):
                raise TypeError("Network checkpoint event NetCons must be a mapping.")
            if not isinstance(continuous_states, Mapping):
                raise TypeError(
                    "Network checkpoint continuous NetCons must be a mapping."
                )
        else:
            event_states = netcons
            continuous_states = {}

        checkpoint_event = set(event_states)
        expected_event = set(self.synapses)
        missing_event = expected_event - checkpoint_event
        unknown_event = checkpoint_event - expected_event
        if missing_event or unknown_event:
            raise KeyError(
                "Network checkpoint event NetCon names do not match the network: "
                f"missing={sorted(missing_event)}, unknown={sorted(unknown_event)}."
            )
        checkpoint_continuous = set(continuous_states)
        expected_continuous = set(self.continuous_synapses)
        missing_continuous = expected_continuous - checkpoint_continuous
        unknown_continuous = checkpoint_continuous - expected_continuous
        if missing_continuous or unknown_continuous:
            raise KeyError(
                "Network checkpoint continuous NetCon names do not match the network: "
                f"missing={sorted(missing_continuous)}, "
                f"unknown={sorted(unknown_continuous)}."
            )

        checkpoint_has_netstim = state_dict["netstim"] is not None
        network_has_netstim = self.netstim is not None
        if checkpoint_has_netstim != network_has_netstim:
            raise ValueError(
                "Network checkpoint NetStim presence does not match the network."
            )
        if checkpoint_has_netstim:
            checkpoint_netstim = state_dict["netstim"]
            if not isinstance(checkpoint_netstim, Mapping):
                raise TypeError("Network checkpoint NetStim state must be a mapping.")
            if "shape" in checkpoint_netstim:
                try:
                    checkpoint_netstim_shape = tuple(
                        int(dim) for dim in checkpoint_netstim["shape"]
                    )
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        "Network checkpoint nested NetStim shape must be an "
                        "iterable of integers."
                    ) from error
                expected_netstim_shape = tuple(int(dim) for dim in self.netstim.shape)
                if checkpoint_netstim_shape != expected_netstim_shape:
                    raise ValueError(
                        "Network checkpoint nested NetStim shape does not match "
                        f"the receiving network: checkpoint={checkpoint_netstim_shape}, "
                        f"expected={expected_netstim_shape}."
                    )

        for name, pop_state in state_dict["populations"].items():
            self.populations[name].restore_dict_from_checkpoint(pop_state)
        for name, syn_state in event_states.items():
            self.synapses[name].restore_dict_from_checkpoint(syn_state)
        for name, syn_state in continuous_states.items():
            self.continuous_synapses[name].restore_dict_from_checkpoint(syn_state)
        if checkpoint_has_netstim:
            self.netstim.restore_dict_from_checkpoint(state_dict["netstim"])
        self.t = restored_t
        self._clock_origin = restored_origin
        self._clock_step = restored_step
        self._duration_remainder = restored_remainder
        self._sync_runtime_clock()

    def restore_dict_from_checkpoint(self, state_dict):
        """Restore a whole-network runtime checkpoint atomically."""
        previous = _clone_checkpoint_state(self.state_dict_for_checkpoint())
        try:
            self._restore_dict_from_checkpoint_unchecked(state_dict)
        except Exception as restore_error:
            try:
                self._restore_dict_from_checkpoint_unchecked(previous)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Network checkpoint restore failed and rollback could not "
                    "recover the previous runtime state."
                ) from rollback_error
            raise restore_error
        return self

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
            Requested simulation duration in milliseconds. Only complete
            timesteps are executed; fractional time is retained for the next
            duration-based network call.
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
        if not self.built:
            raise RuntimeError(
                "Network wiring has changed since the last build. Call "
                "initialize(dt) (recommended) or build(dt) before "
                "longrun_checkpointed()."
            )
        self._require_mode_runtime_ready()
        if isinstance(chunklength, bool) or not isinstance(chunklength, int):
            raise TypeError("chunklength must be a positive integer")
        if chunklength <= 0:
            raise ValueError("chunklength must be a positive integer")
        dt_f = _validate_time_scalar(self.dt, name="dt", positive=True)
        tstop_f = _validate_time_scalar(tstop, name="tstop", positive=False)
        self._refresh_compile_config_from_ctx()
        self._sync_runtime_clock()

        # Normalize callbacks.
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        for c in callbacks:
            c.dt = self.dt

        # Use the same retained-duration budget as run()/longrun(). Exact
        # decimal-form arithmetic keeps common decimal multiples composable
        # without promoting a genuinely just-before-grid duration.
        n_steps, duration_remainder = self._duration_budget(tstop_f, dt_f)
        initial_time = self.t.double().detach().clone()
        t_grid = initial_time + dt_f * torch.arange(
            n_steps,
            dtype=torch.double,
            device=self.t.device,
        )
        t_grid = t_grid.to(dtype=self.t.dtype)
        n_chunks = int(math.ceil(n_steps / chunklength)) if n_steps else 0
        t_chunks = torch.tensor_split(t_grid, n_chunks) if n_chunks else ()

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
        intra_sources = {}
        for name, pop in self.populations.items():
            # New injections invalidate ``pop.intra``. Match Network.step/run/
            # longrun by rebuilding the solver-level stimulus lazily.
            if pop.intra is None and getattr(pop, "injections", None):
                pop.intra = pop.build_intra()
            if pop.intra is not None:
                intra_sources[name] = pop.intra

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
                self._set_duration_remainder(duration_remainder)
                pre_loop_hook(callbacks, self)

                # Snapshot boundary state after pre-loop hooks.
                state = self.state_dict_for_checkpoint()

                pbar = (
                    tqdm(total=n_chunks, desc=f"{initial_time.item():.1f} ms")
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
                                self.continuous_synapses,
                                self.continuous_targets,
                                self.netstim,
                                self.t,
                                dt_f,
                                extra=extra_c,
                                intra=intra_c,
                                compile_network_ops=self.compile_network_ops,
                            )
                            self._advance_runtime_clock()

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


def post_step_hook(c, m):
    c.post_step_hook(m)


def pre_chunk_hook(c, m, n):
    c.pre_chunk_hook(m, n)


def post_chunk_hook(c, m, n):
    c.post_chunk_hook(m, n)
