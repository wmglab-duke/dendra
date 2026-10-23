"""Explicit-state projections over NetCon's shared transport kernels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import NamedTuple

import torch

from dendra.models.core import Population
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.networks._connection_initialization import event_initial_state
from dendra.models.networks._event_cache import (
    EventCache,
    clone_event_cache,
    event_cache_metadata,
    normalize_event_cache,
    restore_event_cache,
    validate_event_cache,
)
from dendra.models.networks._event_delivery import (
    connection_event_counts,
    dense_due,
    dense_fixed_delay_transition,
    dense_interpolated_delay_transition,
    gather_source_gates,
)
from dendra.models.networks._pending_calendar import (
    CalendarState,
    pending_calendar_append,
    pending_calendar_from_legacy,
    pending_calendar_pop,
    pending_calendar_to_legacy,
)
from dendra.models.networks._source_history import (
    source_history_gates,
    source_history_project,
    source_history_record,
)
from dendra.models.networks._tensor_schedules import (
    netcon_tensor_schedule,
    resolve_schedule_values,
)
from dendra.models.networks.netcon import NetCon
from dendra.models.networks.netstim import NetStim
from dendra.models.networks.spiking import threshold_transition

from ._schedules import TensorScheduleTensors, _TensorSchedulePlan
from ._types import FunctionalizationError

_INITIALIZATION_METHODS = {
    name: getattr(NetCon, name)
    for name in ("initialize", "zero", "detach", "_clear_sparse_calendar")
}
_CACHE_INITIALIZATION_METHODS = {
    "initialize_from_state_cache": NetCon.initialize_from_state_cache,
}
_SOURCE_HISTORY_METHODS = {
    name: getattr(NetCon, name)
    for name in (
        "_source_history_source_gate_this_step",
        "_source_history_build_delivery_from_history",
        "_source_history_record_current_gate",
        "_source_history_read_float_connection_gates",
        "_source_history_read_packed_connection_gates",
    )
}
_BITPACKED_HISTORY_METHODS = {
    name: getattr(NetCon, name)
    for name in (
        "_bitpack_source_spikes_this_step",
        "_bitpack_pack_source_spikes",
        "_bitpack_build_delivery_from_history",
        "_bitpack_schedule_due_scheduled_events",
    )
}
_PENDING_CALENDAR_METHODS = {
    name: getattr(NetCon, name)
    for name in (
        "_append_pending_calendar",
        "_append_sparse_payloads",
        "_append_sparse_events",
        "_pop_sparse_calendar_to_dense",
        "_pop_sparse_events",
        "_add_sparse_calendar_to_dense",
    )
}
_HISTORY_METADATA = (
    "bitpack_source_pre_idx",
    "bitpack_source_word_idx",
    "bitpack_source_bit_mask",
    "bitpack_conn_word_idx",
    "bitpack_conn_bit_mask",
    "bitpack_source_threshold",
    "source_history_conn_source_pos",
)
_HISTORY_RUNTIMES = {"source_history", "bitpacked_history"}
_CALENDAR_RUNTIMES = {"sparse_calendar", "bitpacked_history"}
_CALENDAR_FIELDS = CalendarState._fields


def _calendar_state(state, prefix):
    return CalendarState(*(state[f"{prefix}{name}"] for name in _CALENDAR_FIELDS))


def _put_calendar(state, prefix, calendar):
    state.update(
        (f"{prefix}{name}", value)
        for name, value in zip(_CALENDAR_FIELDS, calendar, strict=True)
    )


class NetConTensors(NamedTuple):
    """Independent effective parameters, transport carry, and schedule values.

    ``delays`` contains effective milliseconds for differentiable-delay plans;
    fixed-delay plans store ``None``. Weights and delays are parameters rather
    than checkpoint carry.
    """

    weights: torch.Tensor
    state: dict[str, torch.Tensor]
    schedule: TensorScheduleTensors | None = None
    delays: torch.Tensor | None = None


def _snapshot(value, *, parameter=False):
    with torch.inference_mode(False):
        result = value.detach().clone(memory_format=torch.preserve_format)
        if parameter:
            result.requires_grad_(value.requires_grad)
        return result


def _effective_tensor(module, label):
    """Read the initialized internal weight wrapper without executing it."""
    name = getattr(type(module), "_effective_value_buffer", None)
    value = module._buffers.get(name)
    if (
        not torch.is_tensor(value)
        or "forward" in module.__dict__
        or module._forward_hooks
        or module._forward_pre_hooks
    ):
        raise FunctionalizationError(
            f"Functional NetCon requires initialized internal {label} storage "
            "whose forward returns its registered w Tensor."
        )
    return value


def _source_tensor(netcon):
    source = netcon.pre
    if isinstance(source, NetStim):
        return source._buffers["spikes"]
    if not isinstance(source, Population):
        raise FunctionalizationError(
            "Functional NetCon requires a Population or NetStim source."
        )
    value = source
    for name in netcon.pre_var.split("."):
        if isinstance(value, MechanismHandler) and name in value.mechanisms:
            value = value.mechanisms[name]
            continue
        if not isinstance(value, torch.nn.Module):
            raise FunctionalizationError("NetCon pre_var must name a stored Tensor.")
        slots = {**value._modules, **value._parameters, **value._buffers}
        if name not in slots:
            raise FunctionalizationError(
                "Functional NetCon pre_var must name a registered Tensor; "
                "callbacks and properties are not source values."
            )
        value = slots[name]
    if not torch.is_tensor(value):
        raise FunctionalizationError("NetCon pre_var must name a stored Tensor.")
    return value


def _mode(netcon):
    if netcon.train_flags is not None and not isinstance(netcon.train_flags[5], Real):
        raise FunctionalizationError(
            "Functional NetCon requires scalar real tau metadata. Configure "
            "set_diff_config(tau=float(...)) before lowering it."
        )
    return (
        bool(netcon.training),
        netcon.delay_backend,
        netcon.train_delay_backend,
        None if netcon.train_flags is None else tuple(netcon.train_flags),
        bool(netcon.track_events),
        bool(netcon.skip_thresholding),
        bool(netcon.apply_masking),
    )


def _eligible(netcon):
    owner_type = type(netcon)
    canonical_proxy = (
        owner_type.__module__ == "dendra.models.parametric"
        and owner_type.__bases__ == (NetCon,)
        and set(vars(owner_type)) <= {"t", "__module__", "__doc__"}
    )
    if owner_type is not NetCon and not canonical_proxy:
        raise FunctionalizationError("Expected an initialized canonical NetCon.")
    if not getattr(netcon, "_event_delivery_initialized", False):
        raise FunctionalizationError("Initialize the NetCon before lowering it.")
    if netcon._mode_requires_initialize:
        raise FunctionalizationError("NetCon mode changed; initialize it again.")
    runtime = netcon._checkpoint_runtime_kind()
    if runtime not in {
        "dense",
        "source_history",
        "bitpacked_history",
        "sparse_calendar",
    }:
        raise FunctionalizationError(
            "Functional NetCon currently requires dense queues, source-history "
            "training, bitpacked-history inference, or sparse-calendar inference."
        )
    if runtime == "source_history":
        if not netcon.training or netcon._has_scheduled_events():
            raise FunctionalizationError(
                "Functional source history requires intrinsic training traffic."
            )
    if runtime in _CALENDAR_RUNTIMES and netcon.training:
        raise FunctionalizationError(
            "Functional compact pending calendars require inference mode."
        )
    if runtime in _HISTORY_RUNTIMES:
        methods = (
            _SOURCE_HISTORY_METHODS
            if runtime == "source_history"
            else _BITPACKED_HISTORY_METHODS
        )
        for name, expected in methods.items():
            if name in netcon.__dict__ or getattr(owner_type, name) is not expected:
                raise FunctionalizationError(
                    f"Functional compact history requires its canonical {name} adapter."
                )
    if runtime in _CALENDAR_RUNTIMES:
        for name, expected in _PENDING_CALENDAR_METHODS.items():
            if name in netcon.__dict__ or getattr(owner_type, name) is not expected:
                raise FunctionalizationError(
                    f"Functional pending calendar requires its canonical {name} adapter."
                )
    if netcon.training:
        if netcon.train_flags is None:
            raise FunctionalizationError(
                "Configure NetCon training before lowering it."
            )
        if netcon.train_flags[1] and (
            not isinstance(netcon.train_flags[3], Real)
            or netcon.train_flags[3] not in (2, 3)
            or (
                netcon.train_flags[3] == 3
                and not isinstance(netcon.train_flags[4], Real)
            )
        ):
            raise FunctionalizationError(
                "Functional NetCon requires two or three interpolation taps "
                "and scalar real sigma metadata."
            )
        if netcon.track_events and (
            netcon.train_flags[1] or netcon.train_flags[2] or netcon.train_flags[6]
        ):
            raise FunctionalizationError(
                "Tracked NetCon events require hard delay, spiking, and "
                "scheduled-time semantics."
            )
    if netcon._sched_pre:
        raise FunctionalizationError(
            "Functional NetCon does not represent legacy presynaptic schedules."
        )
    allowed = (
        (NetCon.advance_diff_source_history,)
        if runtime == "source_history"
        else (
            (NetCon.advance_non_diff_bitpacked_history,)
            if runtime == "bitpacked_history"
            else (
                (NetCon.advance_non_diff_sparse_calendar,)
                if runtime == "sparse_calendar"
                else (
                    NetCon.advance_non_diff_dense_uniform,
                    NetCon.advance_non_diff_dense_mixed,
                    NetCon.advance_diff,
                )
            )
        )
    )
    if (
        getattr(netcon.advance, "__self__", None) is not netcon
        or getattr(netcon.advance, "__func__", None) not in allowed
    ):
        raise FunctionalizationError(
            "Functional NetCon requires its canonical transport adapter."
        )


class FunctionalNetCon:
    """A frozen routing plan using the same pure transport core as NetCon.

    Supply full source-shaped numerical values and effective per-connection
    weights. Values use ``source_dtype``/``source_device`` (the initialized NetCon's
    presynaptic floating dtype/device), including when a NetStim stores hard
    spikes as booleans. Weights use the target ``dtype``/``device``.
    Training with ``diff_delays=True`` also requires explicit per-connection
    ``delays=`` in milliseconds using that target dtype/device. Interpolation
    uses the initialized two- or three-tap policy and ring capacity. The native
    minimum delay and modulo behavior are preserved as delay values change.
    Finite thresholds detect rising edges using explicit detector carry.
    Dense raw projections and differentiable source-history gates retain signed
    values and gradients for NaN thresholds; hard compact histories use binary
    source events. Training preserves
    the initialized hard or surrogate detector configuration. The surrogate
    temperature ``tau`` is scalar real metadata, rather than an explicit
    differentiable Tensor input.

    Tensor schedule routing and reference indices are frozen in the plan.
    Pass the extracted :class:`~dendra.func.TensorScheduleTensors` as ``schedule=`` on each
    advance. Floating times, dimensionless amplitudes, and bound source vectors
    are explicit independent values. Their hard or triangular timing policy
    matches the initialized native configuration. Evaluation uses the carry's
    absolute ``global_step`` and does not consume events.

    Call :meth:`read_due`, deliver its payload with one Population ``receive``
    call (including zero payloads), then observe the source and call
    :meth:`advance`. This preserves native receive-before-observe ordering,
    including recurrent sources changed by reception. :meth:`step` combines
    these phases when the supplied source values have already been chosen.
    Neither phase generates source activity or advances any Population clock.

    Source-history training stores one gate per unique source in a compact
    history ring. Surrogate gates use floating history; hard spikes use packed
    bits. These plans require current weights/delays in :meth:`read_due`, and
    :meth:`advance` records only source activity. Parameter edits between calls
    therefore affect historical events at delivery, matching native source
    history. Dense and sparse-calendar plans retain weighted payloads from
    enqueue time instead.
    Functional source-history carry exposes history/latch tensors directly;
    extraction and commit translate the native checkpoint's ``backend_state``.
    Source-history training with scheduled traffic follows native policy and
    uses the dense fallback.

    Bitpacked-history inference uses hard source events, fixed initialized delay
    offsets, and the same packed history kernels. Supply current weights to
    :meth:`read_due`; its returned carry publishes the native delivery scratch
    row. NetStim's own schedules are source activity. Per-connection NetCon
    schedules use a shared pending calendar with enqueue-time weights and fixed
    delay offsets. Sparse-calendar inference uses that calendar for intrinsic
    and scheduled traffic, with independent tracked event counts. These
    runtimes retain compact delivery storage. Lower-level explicit weights
    retain their gradient flags; inference runner execution suppresses simulation
    gradients. Pending capacity is fixed at lowering; :meth:`check_overflow`
    validates the explicit status before using a lower-level result.

    Routing, fixed quantized delays, timestep, and tensor schemas are copied at
    lowering. Reinitialization, calling ``set_diff_config``, mode changes, or
    changing schedule topology or bindings invalidates source-bound calls.
    Direct edits to source topology or detector buffers do not alter this frozen
    plan; extraction and commit reject incompatible metadata. Lower again to adopt changed routing,
    fixed delays, or thresholds. Differentiable delay values may change without
    lowering again; changing capacity requires initialization and lowering.
    Explicit Population batch axes are part of the source/target shapes;
    additional ``vmap`` axes stay implicit.
    """

    def __init__(self, netcon, *, pending_capacity=None):
        _eligible(netcon)
        source = _source_tensor(netcon)
        self._source = netcon
        self._source_type = type(netcon)
        self._pre = netcon.pre
        self._post = netcon.post
        self._syn = netcon.syn
        self._generation = netcon._event_delivery_generation
        self._schedule_generation = netcon._schedule_config_generation
        self._mode = _mode(netcon)
        self._advance = netcon.advance.__func__
        self._runtime = netcon._checkpoint_runtime_kind()
        self._pre_var = netcon.pre_var
        self.source_shape = tuple(source.shape)
        self.shape = tuple(netcon.syn.shape_f)
        self.device = torch.device(netcon.device)
        self.dtype = netcon.dtype
        self.source_device = torch.device(netcon.pre_device)
        self.source_dtype = netcon.pre_dtype
        self._depth = int(netcon.max_delay_steps)
        self._width = math.prod(self.shape)
        self._connections = int(netcon._n_conn)
        self._pre_idx = _snapshot(netcon.pre_idx)
        self._post_idx = _snapshot(netcon.post_idx)
        self._delay_steps = _snapshot(netcon.inference_delay_steps)
        self._dt = _snapshot(netcon.dt)
        self._track_events = bool(netcon.track_events)
        # NetStim gates are already generated activity. Its adapter never uses
        # the NetCon threshold, even when a default finite threshold was stored.
        self._skip_thresholding = isinstance(netcon.pre, NetStim) or bool(
            netcon.skip_thresholding
        )
        self._threshold = _snapshot(netcon.threshold)
        self._nan_mask = _snapshot(netcon.thresh_is_nan)
        self._apply_masking = bool(netcon.apply_masking)
        self._diff_spiking = bool(netcon.training and netcon.train_flags[2])
        self._diff_delays = bool(netcon.training and netcon.train_flags[1])
        self._taps = int(netcon.train_flags[3]) if self._diff_delays else 2
        self._sigma = float(netcon.train_flags[4]) if self._taps == 3 else 0.35
        self._tau = float(netcon.train_flags[5]) if netcon.training else 0.1
        self._diff_scheduled_times = bool(netcon.training and netcon.train_flags[6])
        self._sched_width = float(netcon.train_flags[7]) if netcon.training else 1.0
        self._schedule = _TensorSchedulePlan(
            netcon, index_name="sched_con_idx", device=self.device, dtype=self.dtype
        )
        if pending_capacity is not None and (
            isinstance(pending_capacity, bool)
            or not isinstance(pending_capacity, Integral)
            or pending_capacity < 1
        ):
            raise ValueError("pending_capacity must be a positive integer")
        if self._runtime in _CALENDAR_RUNTIMES:
            backend = netcon.state_dict_for_checkpoint()["backend_state"]
            existing = max(
                sum(
                    values.numel()
                    for chunks in backend.get(name, {}).values()
                    for _, values in chunks
                )
                for name in ("sparse_calendar", "sparse_event_calendar")
            )
            intrinsic_only = (
                self._runtime == "bitpacked_history"
                and not self._schedule.events
                and existing == 0
            )
            self._pending_capacity = (
                (0 if intrinsic_only else max(existing, 2 * self._connections, 16))
                if pending_capacity is None
                else int(pending_capacity)
            )
        else:
            self._pending_capacity = None
        self._history_mode = (
            netcon._bitpack_mode if self._runtime in _HISTORY_RUNTIMES else None
        )
        self._history_metadata = (
            {name: _snapshot(getattr(netcon, name)) for name in _HISTORY_METADATA}
            if self._runtime in _HISTORY_RUNTIMES
            else {}
        )
        self._history_name = (
            "source_gate_history" if self._diff_spiking else "spike_history_packed"
        )
        self._cache_metadata = event_cache_metadata(
            netcon, frozen=self._history_metadata
        )
        checkpoint = self._checkpoint_values(netcon)
        self._state_schema = {
            name: (tuple(value.shape), value.dtype, value.device)
            for name, value in checkpoint.items()
        }
        self._validate_compatible(netcon)

    @property
    def runtime(self):
        """Frozen native transport runtime."""
        return self._runtime

    @property
    def pending_capacity(self):
        """Maximum entries per pending calendar, or ``None`` for other runtimes.

        Intrinsic-only packed plans with no imported pending traffic use zero
        entries by default and skip calendar transport. Other defaults use the
        largest of existing checkpoint traffic, twice the connection count, and
        sixteen entries. An explicit positive ``pending_capacity`` overrides
        either default. Increase it when lowering if traffic can accumulate.
        """
        return self._pending_capacity

    def check_overflow(self, state):
        """Raise on pending overflow; transformed calls retain explicit status.

        Pure transitions expose sticky ``pending_overflow`` and, for tracked
        sparse projections, ``event_pending_overflow`` tensors. An overflow
        invalidates that invocation. Restart from an earlier checkpoint with
        a larger ``pending_capacity`` rather than continuing its returned carry.
        Host checks accept stacked or ``vmap`` outputs with leading batch axes
        and reject overflow in any lane, including nested batches.
        """
        if (
            torch.compiler.is_compiling()
            or torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level >= 0
        ):
            return
        if not isinstance(state, Mapping):
            raise ValueError("NetCon overflow status requires a state mapping")
        flags = []
        for name in ("pending_overflow", "event_pending_overflow"):
            if name not in self._state_schema:
                continue
            shape, dtype, device = self._state_schema[name]
            value = state.get(name)
            if (
                not torch.is_tensor(value)
                or value.ndim < len(shape)
                or tuple(value.shape[-len(shape) :]) != shape
                or value.dtype != dtype
                or value.device != device
            ):
                raise ValueError(
                    f"state.{name} must match {device}/{dtype} and end in {shape}"
                )
            flags.append(value)
        if any(bool(value.any()) for value in flags):
            raise RuntimeError(
                "NetCon pending calendar overflowed; increase pending_capacity "
                "when lowering and restart from the previous checkpoint."
            )

    def _checkpoint_values(self, netcon):
        """Normalize native compact checkpoints to a Tensor-only carry packet."""
        checkpoint = dict(netcon.state_dict_for_checkpoint())
        if self._runtime == "dense":
            return checkpoint
        backend = checkpoint.pop("backend_state", None)
        if not isinstance(backend, dict) or backend.get("kind") != self._runtime:
            raise FunctionalizationError(
                "NetCon checkpoint runtime does not match this plan"
            )
        if self._runtime == "source_history" and (
            backend.get("sparse_calendar") or backend.get("sparse_event_calendar")
        ):
            raise FunctionalizationError(
                "Functional compact history cannot lower nonempty side calendars"
            )
        allowed = {
            "kind",
            "sparse_calendar",
            "sparse_event_calendar",
            "source_gate_history",
            "spike_history_packed",
            "bitpack_source_has_spiked",
        }
        if set(backend) - allowed:
            raise FunctionalizationError("Unknown compact-history checkpoint carry")
        for name in (
            "source_gate_history",
            "spike_history_packed",
            "bitpack_source_has_spiked",
        ):
            if name in backend:
                checkpoint[name] = backend[name]
        # Native checkpoints omit empty fields. Keep active zero-source history
        # and latches explicit so transformed kernels retain a stable schema.
        if self._runtime in _HISTORY_RUNTIMES:
            checkpoint.setdefault(
                self._history_name, getattr(netcon, self._history_name)
            )
        if self._history_mode == "threshold":
            checkpoint.setdefault(
                "bitpack_source_has_spiked", netcon.bitpack_source_has_spiked
            )
        if self._runtime in _CALENDAR_RUNTIMES:

            def pending_values(mapping, legacy, dtype):
                # Legacy buckets retain within-slot order, but grouping them
                # would change the cross-slot ordering of a live flat table.
                if hasattr(mapping, "tensor_state"):
                    pending = mapping.tensor_state
                    size = int(pending.size.item())
                    if size > self._pending_capacity:
                        raise ValueError(
                            "pending_capacity is too small for the existing checkpoint"
                        )
                    return CalendarState(
                        *(
                            torch.cat(
                                (
                                    value[:size],
                                    value.new_zeros(self._pending_capacity - size),
                                )
                            )
                            for value in pending[:3]
                        ),
                        pending.size,
                        pending.overflow,
                    )
                return pending_calendar_from_legacy(
                    legacy,
                    capacity=self._pending_capacity,
                    device=self.device,
                    dtype=dtype,
                )

            _put_calendar(
                checkpoint,
                "pending_",
                pending_values(
                    netcon._sparse_calendar,
                    backend.get("sparse_calendar", {}),
                    self.dtype,
                ),
            )
            if self._runtime == "sparse_calendar" and self._track_events:
                _put_calendar(
                    checkpoint,
                    "event_pending_",
                    pending_values(
                        netcon._sparse_event_calendar,
                        backend.get("sparse_event_calendar", {}),
                        torch.int32,
                    ),
                )
        return checkpoint

    def _validate_source(self):
        source = self._source
        if (
            type(source) is not self._source_type
            or not source._event_delivery_initialized
            or source._event_delivery_generation != self._generation
            or source._schedule_config_generation != self._schedule_generation
            or source._mode_requires_initialize
            or _mode(source) != self._mode
            or source.pre is not self._pre
            or source.post is not self._post
            or source.syn is not self._syn
            or source.pre_var != self._pre_var
            or source.device != self.device
            or source.dtype != self.dtype
            or source.pre_device != self.source_device
            or source.pre_dtype != self.source_dtype
            or source.max_delay_steps != self._depth
            or source._n_conn != self._connections
            or source._syn_numel != self._width
            or tuple(source.syn.shape_f) != self.shape
            or getattr(source.advance, "__func__", None) is not self._advance
            or source._sched_pre
            or (
                self._runtime in _HISTORY_RUNTIMES
                and (
                    source._bitpack_mode != self._history_mode
                    or not source._bitpack_can_use
                )
            )
        ):
            raise FunctionalizationError(
                "The source NetCon configuration changed; initialize and lower it again."
            )
        if self._runtime in _HISTORY_RUNTIMES:
            methods = (
                _SOURCE_HISTORY_METHODS
                if self._runtime == "source_history"
                else _BITPACKED_HISTORY_METHODS
            )
            for name, expected in methods.items():
                if (
                    name in source.__dict__
                    or getattr(type(source), name) is not expected
                ):
                    raise FunctionalizationError(
                        f"Compact-history {name} changed; lower again."
                    )
        if self._runtime in _CALENDAR_RUNTIMES:
            for name, expected in _PENDING_CALENDAR_METHODS.items():
                if (
                    name in source.__dict__
                    or getattr(type(source), name) is not expected
                ):
                    raise FunctionalizationError(
                        f"Pending-calendar {name} changed; lower again."
                    )

    @staticmethod
    def _validate_tensor(value, *, shape, dtype, device, label):
        if not torch.is_tensor(value):
            raise TypeError(f"{label} must be a Tensor")
        if (
            tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != device
        ):
            raise ValueError(f"{label} must match {device}/{dtype}/{shape}")

    def _validate_state(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self._state_schema):
            raise ValueError("NetCon state must match its transport checkpoint schema")
        for name, (shape, dtype, device) in self._state_schema.items():
            self._validate_tensor(
                state[name],
                shape=shape,
                dtype=dtype,
                device=device,
                label=f"state.{name}",
            )

    def _validate_pending(self, state):
        """Validate variable live-prefix metadata before any native mutation."""
        for prefix, width in (
            ("pending_", self._width),
            ("event_pending_", self._connections),
        ):
            if f"{prefix}size" not in state:
                continue
            calendar = _calendar_state(state, prefix)
            size = int(calendar.size.item())
            if not 0 <= size <= self._pending_capacity:
                raise ValueError("Pending calendar size is outside its capacity")
            slots, indices = calendar.slots[:size], calendar.indices[:size]
            if bool(((slots < 0) | (slots >= self._depth)).any()):
                raise ValueError("Pending calendar slot is outside its delay ring")
            if bool(((indices < 0) | (indices >= width)).any()):
                raise ValueError("Pending calendar index is outside its topology")

    def _validate_compatible(self, netcon):
        _eligible(netcon)
        source = _source_tensor(netcon)
        if (
            _mode(netcon) != self._mode
            or netcon._checkpoint_runtime_kind() != self._runtime
            or (isinstance(netcon.pre, NetStim) or bool(netcon.skip_thresholding))
            != self._skip_thresholding
            or tuple(source.shape) != self.source_shape
            or tuple(netcon.syn.shape_f) != self.shape
            or netcon.pre_var != self._pre_var
            or netcon.device != self.device
            or netcon.dtype != self.dtype
            or netcon.pre_device != self.source_device
            or netcon.pre_dtype != self.source_dtype
            or int(netcon.max_delay_steps) != self._depth
            or int(netcon._n_conn) != self._connections
            or int(netcon._syn_numel) != self._width
            or not torch.equal(netcon.pre_idx, self._pre_idx)
            or not torch.equal(netcon.post_idx, self._post_idx)
            or (
                not self._diff_delays
                and not torch.equal(netcon.inference_delay_steps, self._delay_steps)
            )
            or not torch.equal(netcon.dt, self._dt)
            or not torch.equal(netcon.threshold, self._threshold)
            or not torch.equal(netcon.thresh_is_nan, self._nan_mask)
            or (
                not self._diff_delays
                and not torch.equal(
                    (_effective_tensor(netcon.delay_ms, "delay") / netcon.dt)
                    .round()
                    .long()
                    .clamp_min(1),
                    self._delay_steps,
                )
            )
        ):
            raise FunctionalizationError(
                "NetCon topology or detector does not match this frozen plan."
            )
        if self._runtime in _HISTORY_RUNTIMES and (
            netcon._bitpack_mode != self._history_mode
            or any(
                not torch.equal(getattr(netcon, name), value)
                for name, value in self._history_metadata.items()
            )
        ):
            raise FunctionalizationError(
                "Compact-history routing or detector does not match this plan"
            )
        self._validate_tensor(
            _effective_tensor(netcon.weight, "weight"),
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="effective weights",
        )
        self._validate_tensor(
            _effective_tensor(netcon.delay_ms, "delay"),
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="effective delays",
        )
        self._schedule.validate_compatible(netcon)
        if bool(torch.any(self._schedule.indices >= self._connections)):
            raise FunctionalizationError(
                "NetCon schedule connection indices are outside its topology."
            )
        self._validate_state(self._checkpoint_values(netcon))
        if self._depth < 1 or bool(torch.any(self._delay_steps < 1)):
            raise FunctionalizationError("NetCon needs a nonempty fixed-delay ring.")
        if bool(torch.any((self._pre_idx < 0) | (self._pre_idx >= source.numel()))):
            raise FunctionalizationError(
                "NetCon source indices are outside its source."
            )
        if bool(torch.any((self._post_idx < 0) | (self._post_idx >= self._width))):
            raise FunctionalizationError(
                "NetCon target indices are outside its target."
            )

    def extract(self, netcon=None) -> NetConTensors:
        """Snapshot effective parameters, checkpoint carry, and schedule values."""
        if torch.compiler.is_compiling():
            raise FunctionalizationError("Extract NetCon tensors before torch.compile.")
        netcon = self._source if netcon is None else netcon
        self._validate_compatible(netcon)
        return NetConTensors(
            _snapshot(_effective_tensor(netcon.weight, "weight"), parameter=True),
            {
                name: _snapshot(value)
                for name, value in self._checkpoint_values(netcon).items()
            },
            self._schedule.extract(netcon),
            (
                _snapshot(_effective_tensor(netcon.delay_ms, "delay"), parameter=True)
                if self._diff_delays
                else None
            ),
        )

    def initial_state(self, state, *, t):
        """Clear delivery/detector carry at finite scalar starting time ``t``.

        Routing and scheduled events are unchanged. The absolute schedule
        counter uses native NetCon rounding; all returned carry is detached,
        matching native initialization without weight or delay resampling.
        """
        self._validate_source()
        self._validate_state(state)
        for name, expected in _INITIALIZATION_METHODS.items():
            if (
                name in self._source.__dict__
                or getattr(type(self._source), name) is not expected
            ):
                raise FunctionalizationError(
                    f"Fresh NetCon initialization requires its canonical {name} adapter."
                )
        return event_initial_state(state, t=t, dt=self._dt)

    def normalize_cache(self, cache):
        """Own a detached steady-state cache at the lowered dt and topology."""
        return self._normalize_cache(cache, copy_normalized=True)

    def _normalize_cache(self, cache, *, copy_normalized):
        self._validate_source()
        for name, expected in _CACHE_INITIALIZATION_METHODS.items():
            if (
                name in self._source.__dict__
                or getattr(type(self._source), name) is not expected
            ):
                raise FunctionalizationError(
                    f"NetCon cache initialization requires its canonical {name} adapter."
                )
        if isinstance(cache, EventCache):
            validate_event_cache(
                cache,
                runtime=self._runtime,
                depth=self._depth,
                delivery_depth=self._state_schema["delivery_buffer"][0][0],
                width=self._width,
                connections=self._connections,
                device=self.device,
                dtype=self.dtype,
                source_device=self.source_device,
                source_dtype=self.source_dtype,
                metadata=self._cache_metadata,
                binary_history=not self._diff_spiking,
            )
            return clone_event_cache(cache) if copy_normalized else cache
        if torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():
            raise FunctionalizationError(
                "Normalize event state caches outside compiled or transformed execution"
            )
        return normalize_event_cache(
            cache,
            backend=self._source.delay_backend,
            runtime=self._runtime,
            depth=self._depth,
            delivery_depth=self._state_schema["delivery_buffer"][0][0],
            width=self._width,
            dt=float(self._dt.detach().cpu().item()),
            device=self.device,
            dtype=self.dtype,
            source_device=self.source_device,
            source_dtype=self.source_dtype,
            track_events=self._track_events,
            metadata=self._cache_metadata,
            binary_history=not self._diff_spiking,
        )

    def restore_cache(self, parameters, state, cache, *, t):
        """Restore detached steady-state history using explicit current parameters."""
        self._validate_source()
        self._validate_state(state)
        weights = parameters["weights"]
        delays = parameters.get("delays")
        self._validate_projection_parameters(weights, delays)
        cache = self._normalize_cache(cache, copy_normalized=False)
        delay_steps = (
            self._delay_steps
            if delays is None
            else torch.round(delays.detach() / self._dt).to(torch.long).clamp_min(1)
        )
        result = restore_event_cache(
            state,
            cache,
            weights,
            delay_steps=delay_steps,
            post_idx=self._post_idx,
            metadata=self._cache_metadata,
            runtime=self._runtime,
            depth=self._depth,
            width=self._width,
            t=t,
            dt=self._dt,
            binary_history=not self._diff_spiking,
        )
        self._validate_state(result)
        self.check_overflow(result)
        return result

    def _validate_projection_parameters(self, weights, delays):
        self._validate_tensor(
            weights,
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="weights",
        )
        if self._diff_delays:
            self._validate_tensor(
                delays,
                shape=(self._connections,),
                dtype=self.dtype,
                device=self.device,
                label="delays",
            )
        elif delays is not None:
            raise ValueError("Fixed-delay NetCon plans do not accept explicit delays")

    def read_due(self, state, *, weights=None, delays=None):
        """Return ``(delivery_state, payload)`` for one separate receive call.

        This publishes due tracked events in the returned state without clearing
        the queue or changing the detector latch, ring slot, or global step.
        Compact-history plans require current ``weights`` because native delivery
        reconstructs traffic from past source gates at delivery time. Training
        with differentiable delays also requires current ``delays``. Bitpacked
        inference publishes its scratch payload in the returned carry, including
        zero deliveries, without changing the input state.
        """
        self._validate_source()
        self._validate_state(state)
        if self._runtime in _HISTORY_RUNTIMES:
            self._validate_projection_parameters(weights, delays)
            metadata = self._history_metadata
            payload = source_history_project(
                state[self._history_name],
                state["current_time_step"],
                weights,
                post_idx=self._post_idx,
                depth=self._depth,
                width=self._width,
                source_pos=(
                    metadata["source_history_conn_source_pos"]
                    if self._diff_spiking
                    else None
                ),
                word_idx=(
                    None if self._diff_spiking else metadata["bitpack_conn_word_idx"]
                ),
                bit_mask=(
                    None if self._diff_spiking else metadata["bitpack_conn_bit_mask"]
                ),
                delay_steps=None if self._diff_delays else self._delay_steps,
                delays=delays,
                dt=self._dt,
                taps=self._taps,
                sigma=self._sigma,
            )
            result = dict(state)
            if self._runtime == "bitpacked_history":
                if self._pending_capacity:
                    calendar, payload = pending_calendar_pop(
                        _calendar_state(state, "pending_"),
                        state["current_time_step"],
                        width=self._width,
                        base=payload,
                    )
                    _put_calendar(result, "pending_", calendar)
                result["delivery_buffer"] = payload.reshape(1, self._width)
                if "events" in result:
                    result["events"] = torch.zeros_like(state["events"])
            return result, payload.reshape(self.shape)
        if self._runtime == "sparse_calendar":
            result = dict(state)
            calendar, payload = pending_calendar_pop(
                _calendar_state(state, "pending_"),
                state["current_time_step"],
                width=self._width,
            )
            _put_calendar(result, "pending_", calendar)
            result["delivery_buffer"] = payload.reshape(1, self._width)
            if self._track_events:
                events, counts = pending_calendar_pop(
                    _calendar_state(state, "event_pending_"),
                    state["current_time_step"],
                    width=self._connections,
                )
                _put_calendar(result, "event_pending_", events)
                result["events"] = counts
            return result, payload.reshape(self.shape)
        due = dense_due(
            state["delivery_buffer"],
            state["current_time_step"],
            state.get("event_queue"),
        )
        result = dict(state)
        if self._track_events:
            result["events"] = due.event_counts
        return result, due.payload.reshape(self.shape)

    def _scheduled_gates(self, state, schedule):
        if not self._schedule.events:
            return None, None
        times, amplitudes = resolve_schedule_values(
            schedule.times,
            schedule.amplitudes,
            self._schedule.time_idx,
            self._schedule.amplitude_idx,
            time_source=schedule.time_source,
            weight_source=schedule.amplitude_source,
            time_dtype=self.dtype,
        )
        return netcon_tensor_schedule(
            self._schedule.indices,
            times,
            amplitudes,
            state["global_step"],
            self._dt,
            connections=self._connections,
            differentiable_times=self._diff_scheduled_times,
            width=self._sched_width,
        )

    def _enqueue_pending(self, state, weights, gates, *, counts=None):
        """Freeze enqueue-time payloads and atomically append both channels."""
        result = dict(state)
        future_slots = (state["current_time_step"] + self._delay_steps).remainder(
            self._depth
        )
        # Native append visits physical slots in sorted order and preserves
        # connection order within each slot. Retain that checkpoint ordering.
        order = torch.argsort(future_slots, stable=True)
        active_gates = gates != 0
        values = torch.where(active_gates, weights, 0) * torch.where(
            active_gates, gates, 0
        )
        previous = _calendar_state(state, "pending_")
        pending = pending_calendar_append(
            previous,
            future_slots[order],
            self._post_idx[order],
            values[order],
            ((gates != 0) & (values != 0))[order],
        )
        overflow = pending.overflow
        event_previous = event_pending = None
        if counts is not None:
            event_previous = _calendar_state(state, "event_pending_")
            event_pending = pending_calendar_append(
                event_previous,
                future_slots[order],
                order,
                counts[order],
                (counts != 0)[order],
            )
            overflow = overflow | event_pending.overflow

        def publish(before, after):
            return CalendarState(
                *(
                    torch.where(overflow, original, updated)
                    for original, updated in zip(before[:-1], after[:-1], strict=True)
                ),
                overflow,
            )

        _put_calendar(result, "pending_", publish(previous, pending))
        if event_pending is not None:
            _put_calendar(
                result, "event_pending_", publish(event_previous, event_pending)
            )
        return result

    def advance(self, weights, state, source_values, *, delays=None, schedule=None):
        """Observe the source, update transport history, and return next carry.

        Pass the state from :meth:`read_due` after delivering its payload and
        observing the source. Threshold metadata is frozen; gradients flow
        through explicit weights, source values, pending payloads, and schedule
        amplitudes. Triangular scheduling also differentiates explicit times.
        Interpolated-delay plans require explicit ``delays``; fixed-delay plans
        reject it. Delay values use the native fixed ring and are not scanned
        for a range on each step.
        """
        self._validate_source()
        self._validate_state(state)
        self._schedule.validate(schedule)
        self._validate_projection_parameters(weights, delays)
        self._validate_tensor(
            source_values,
            shape=self.source_shape,
            dtype=self.source_dtype,
            device=self.source_device,
            label="source_values",
        )
        if self._runtime in _HISTORY_RUNTIMES:
            metadata = self._history_metadata
            latch = state.get("bitpack_source_has_spiked")
            if latch is None:
                latch = metadata["bitpack_source_threshold"].new_empty(
                    (0,), dtype=torch.bool
                )
            has_spiked, gates = source_history_gates(
                source_values,
                latch,
                source_idx=metadata["bitpack_source_pre_idx"],
                mode=self._history_mode,
                threshold=metadata["bitpack_source_threshold"],
                diff_spiking=self._diff_spiking,
                tau=self._tau,
                dtype=self.source_dtype,
                binary=self._runtime == "bitpacked_history",
            )
            result = dict(state)
            if "bitpack_source_has_spiked" in result:
                result["bitpack_source_has_spiked"] = has_spiked
            result[self._history_name] = source_history_record(
                state[self._history_name],
                state["current_time_step"],
                (
                    gates.to(device=self.device)
                    if self._runtime == "bitpacked_history"
                    else gates.to(device=self.device, dtype=self.dtype)
                ),
                source_word_idx=(
                    None if self._diff_spiking else metadata["bitpack_source_word_idx"]
                ),
                source_bit_mask=(
                    None if self._diff_spiking else metadata["bitpack_source_bit_mask"]
                ),
            )
            if self._runtime == "bitpacked_history":
                scheduled_gates, _ = self._scheduled_gates(state, schedule)
                if scheduled_gates is not None:
                    result = self._enqueue_pending(result, weights, scheduled_gates)
            result["current_time_step"] = (state["current_time_step"] + 1).remainder(
                self._depth
            )
            result["global_step"] = state["global_step"] + 1
            return result
        event_queue = state.get("event_queue")
        selected = gather_source_gates(source_values, self._pre_idx)
        result = dict(state)
        if self._skip_thresholding:
            gates = selected
        else:
            result["has_spiked"], gates = threshold_transition(
                selected,
                state["has_spiked"],
                self._threshold,
                diff_spiking=self._diff_spiking,
                tau=self._tau,
                nan_mask=self._nan_mask if self._apply_masking else None,
            )
        gates = gates.to(device=self.device, dtype=self.dtype)
        scheduled_gates, scheduled_counts = self._scheduled_gates(state, schedule)
        if scheduled_gates is not None:
            combined = gates + scheduled_gates
        else:
            combined = gates
        counts = (
            connection_event_counts(gates, scheduled_counts)
            if self._track_events
            else None
        )
        if self._runtime == "sparse_calendar":
            result = self._enqueue_pending(result, weights, combined, counts=counts)
            result["current_time_step"] = (state["current_time_step"] + 1).remainder(
                self._depth
            )
            result["global_step"] = state["global_step"] + 1
            return result
        if self._diff_delays:
            updated = dense_interpolated_delay_transition(
                state["delivery_buffer"],
                state["current_time_step"],
                state["global_step"],
                weights,
                combined,
                delays,
                self._dt,
                post_idx=self._post_idx,
                depth=self._depth,
                width=self._width,
                taps=self._taps,
                sigma=self._sigma,
            )
        else:
            updated = dense_fixed_delay_transition(
                state["delivery_buffer"],
                state["current_time_step"],
                state["global_step"],
                weights,
                combined,
                post_idx=self._post_idx,
                delay_steps=self._delay_steps,
                depth=self._depth,
                width=self._width,
                event_queue=event_queue,
                event_counts=counts,
            )
        result.update(
            delivery_buffer=updated.queue,
            current_time_step=updated.slot,
            global_step=updated.global_step,
        )
        if self._track_events:
            result["event_queue"] = updated.event_queue
        return result

    def step(self, weights, state, source_gates, *, delays=None, schedule=None):
        """Return ``(next_state, due_payload)`` using prechosen source values.

        ``source_gates`` retains its original argument name for callers using
        explicit raw gates; finite thresholds interpret it as source values.
        Use the separate phases when reception can change those values.
        """
        delivery_state, payload = self.read_due(state, weights=weights, delays=delays)
        return (
            self.advance(
                weights,
                delivery_state,
                source_gates,
                delays=delays,
                schedule=schedule,
            ),
            payload,
        )

    def commit_state_(self, netcon, state):
        """Commit independent copies of validated carry into a compatible NetCon.

        This restores queues, counters, detector history, and tracked events.
        Align effective weights, delays, and floating schedule values separately when
        resuming native execution; they are parameters rather than carry.
        """
        if torch.compiler.is_compiling():
            raise FunctionalizationError("Commit NetCon state outside torch.compile.")
        self._validate_compatible(netcon)
        self._validate_state(state)
        self.check_overflow(state)
        self._validate_pending(state)
        with torch.inference_mode(False):
            copied = {
                name: value.clone(memory_format=torch.contiguous_format)
                for name, value in state.items()
            }
            if self._runtime != "dense":
                history = {
                    name: copied.pop(name)
                    for name in (
                        "source_gate_history",
                        "spike_history_packed",
                        "bitpack_source_has_spiked",
                    )
                    if name in copied
                }
                calendars = {}
                for prefix, name in (
                    ("pending_", "sparse_calendar"),
                    ("event_pending_", "sparse_event_calendar"),
                ):
                    calendars[name] = (
                        pending_calendar_to_legacy(_calendar_state(copied, prefix))
                        if f"{prefix}size" in copied
                        else {}
                    )
                    for field in _CALENDAR_FIELDS:
                        copied.pop(f"{prefix}{field}", None)
                copied["backend_state"] = {
                    "kind": self._runtime,
                    **calendars,
                    **history,
                }
            netcon.restore_dict_from_checkpoint(copied)
            return self._commit_pending_(netcon, state)

    def _commit_pending_(self, netcon, state):
        """Retain exact flat order after a validated legacy checkpoint restore."""
        if self._runtime in _CALENDAR_RUNTIMES:
            with torch.inference_mode(False):
                for prefix, mapping in (
                    ("pending_", netcon._sparse_calendar),
                    ("event_pending_", netcon._sparse_event_calendar),
                ):
                    if f"{prefix}size" in state:
                        mapping.replace_tensor_state(
                            CalendarState(
                                *(
                                    value.clone()
                                    for value in _calendar_state(state, prefix)
                                )
                            )
                        )
        return netcon


def make_functional_netcon(
    netcon, *, pending_capacity=None
) -> tuple[FunctionalNetCon, NetConTensors]:
    """Lower an initialized NetCon with explicit numerical source values."""
    projection = FunctionalNetCon(netcon, pending_capacity=pending_capacity)
    return projection, projection.extract()


__all__ = ["FunctionalNetCon", "NetConTensors", "make_functional_netcon"]
