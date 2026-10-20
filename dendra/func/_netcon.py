"""Explicit-state projections over NetCon's shared fixed-delay transport."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import NamedTuple

import torch

from dendra.models.core import Population
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.networks._event_delivery import (
    connection_event_counts,
    dense_due,
    dense_fixed_delay_transition,
    gather_source_gates,
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


class NetConTensors(NamedTuple):
    """Independent effective weights, dense carry, and optional schedule values."""

    weights: torch.Tensor
    state: dict[str, torch.Tensor]
    schedule: TensorScheduleTensors | None = None


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
    if netcon._checkpoint_runtime_kind() != "dense":
        raise FunctionalizationError(
            "Functional NetCon currently requires dense queues."
        )
    if netcon.training:
        if netcon.train_flags is None:
            raise FunctionalizationError(
                "Configure NetCon training before lowering it."
            )
        if netcon.train_flags[1]:
            raise FunctionalizationError(
                "Functional NetCon requires diff_delays=False."
            )
        if netcon.track_events and (netcon.train_flags[2] or netcon.train_flags[6]):
            raise FunctionalizationError(
                "Tracked NetCon events require hard spiking and scheduled-time semantics."
            )
    if netcon._sched_pre:
        raise FunctionalizationError(
            "Functional NetCon does not represent legacy presynaptic schedules."
        )
    allowed = (
        NetCon.advance_non_diff_dense_uniform,
        NetCon.advance_non_diff_dense_mixed,
        NetCon.advance_diff,
    )
    if (
        getattr(netcon.advance, "__self__", None) is not netcon
        or getattr(netcon.advance, "__func__", None) not in allowed
    ):
        raise FunctionalizationError(
            "Functional NetCon requires its canonical dense adapter."
        )


class FunctionalNetCon:
    """A frozen routing plan using the same pure transport core as NetCon.

    Supply full source-shaped numerical values and effective per-connection
    weights. Values use ``source_dtype``/``source_device`` (the initialized NetCon's
    presynaptic floating dtype/device), including when a NetStim stores hard
    spikes as booleans. Weights use the target ``dtype``/``device``.
    Finite thresholds detect rising edges using explicit ``has_spiked`` carry;
    NaN thresholds retain raw signed values and gradients. Training preserves
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

    Routing, quantized delays, timestep, and tensor schemas are copied at
    lowering. Reinitialization, calling ``set_diff_config``, mode changes, or
    changing schedule topology or bindings invalidates source-bound calls.
    Direct edits to source topology or detector buffers do not alter this frozen
    plan; extraction and commit reject incompatible metadata. Lower again to adopt changed routing,
    delays, or thresholds. Explicit Population batch axes are part of the
    source/target shapes; additional ``vmap`` axes stay implicit.
    """

    def __init__(self, netcon):
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
        self._physical_delays = _snapshot(_effective_tensor(netcon.delay_ms, "delay"))
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
        self._tau = float(netcon.train_flags[5]) if netcon.training else 0.1
        self._diff_scheduled_times = bool(netcon.training and netcon.train_flags[6])
        self._sched_width = float(netcon.train_flags[7]) if netcon.training else 1.0
        self._schedule = _TensorSchedulePlan(
            netcon, index_name="sched_con_idx", device=self.device, dtype=self.dtype
        )
        checkpoint = netcon.state_dict_for_checkpoint()
        self._state_schema = {
            name: (tuple(value.shape), value.dtype, value.device)
            for name, value in checkpoint.items()
        }
        self._validate_compatible(netcon)

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
        ):
            raise FunctionalizationError(
                "The source NetCon configuration changed; initialize and lower it again."
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
            raise ValueError("NetCon state must match its dense checkpoint schema")
        for name, (shape, dtype, device) in self._state_schema.items():
            self._validate_tensor(
                state[name],
                shape=shape,
                dtype=dtype,
                device=device,
                label=f"state.{name}",
            )

    def _validate_compatible(self, netcon):
        _eligible(netcon)
        source = _source_tensor(netcon)
        if (
            _mode(netcon) != self._mode
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
            or not torch.equal(netcon.inference_delay_steps, self._delay_steps)
            or not torch.equal(netcon.dt, self._dt)
            or not torch.equal(netcon.threshold, self._threshold)
            or not torch.equal(netcon.thresh_is_nan, self._nan_mask)
            or not torch.equal(
                _effective_tensor(netcon.delay_ms, "delay"), self._physical_delays
            )
        ):
            raise FunctionalizationError(
                "NetCon topology or detector does not match this frozen plan."
            )
        self._validate_tensor(
            _effective_tensor(netcon.weight, "weight"),
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="effective weights",
        )
        self._schedule.validate_compatible(netcon)
        if bool(torch.any(self._schedule.indices >= self._connections)):
            raise FunctionalizationError(
                "NetCon schedule connection indices are outside its topology."
            )
        self._validate_state(netcon.state_dict_for_checkpoint())
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
        """Snapshot effective weights, checkpoint carry, and schedule values."""
        if torch.compiler.is_compiling():
            raise FunctionalizationError("Extract NetCon tensors before torch.compile.")
        netcon = self._source if netcon is None else netcon
        self._validate_compatible(netcon)
        return NetConTensors(
            _snapshot(_effective_tensor(netcon.weight, "weight"), parameter=True),
            {
                name: _snapshot(value)
                for name, value in netcon.state_dict_for_checkpoint().items()
            },
            self._schedule.extract(netcon),
        )

    def read_due(self, state):
        """Return ``(delivery_state, payload)`` for one separate receive call.

        This publishes due tracked events in the returned state without clearing
        the queue or changing the detector latch, ring slot, or global step.
        """
        self._validate_source()
        self._validate_state(state)
        due = dense_due(
            state["delivery_buffer"],
            state["current_time_step"],
            state.get("event_queue"),
        )
        result = dict(state)
        if self._track_events:
            result["events"] = due.event_counts
        return result, due.payload.reshape(self.shape)

    def advance(self, weights, state, source_values, *, schedule=None):
        """Observe the current source, enqueue events, and return next carry.

        Pass the state from :meth:`read_due` after delivering its payload and
        observing the source. Threshold metadata is frozen; gradients flow
        through explicit weights, source values, pending payloads, and schedule
        amplitudes. Triangular scheduling also differentiates explicit times.
        """
        self._validate_source()
        self._validate_state(state)
        self._schedule.validate(schedule)
        self._validate_tensor(
            weights,
            shape=(self._connections,),
            dtype=self.dtype,
            device=self.device,
            label="weights",
        )
        self._validate_tensor(
            source_values,
            shape=self.source_shape,
            dtype=self.source_dtype,
            device=self.source_device,
            label="source_values",
        )
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
        scheduled_counts = None
        if self._schedule.events:
            times, amplitudes = resolve_schedule_values(
                schedule.times,
                schedule.amplitudes,
                self._schedule.time_idx,
                self._schedule.amplitude_idx,
                time_source=schedule.time_source,
                weight_source=schedule.amplitude_source,
                time_dtype=self.dtype,
            )
            scheduled_gates, scheduled_counts = netcon_tensor_schedule(
                self._schedule.indices,
                times,
                amplitudes,
                state["global_step"],
                self._dt,
                connections=self._connections,
                differentiable_times=self._diff_scheduled_times,
                width=self._sched_width,
            )
            combined = gates + scheduled_gates
        else:
            combined = gates
        counts = (
            connection_event_counts(gates, scheduled_counts)
            if self._track_events
            else None
        )
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

    def step(self, weights, state, source_gates, *, schedule=None):
        """Return ``(next_state, due_payload)`` using prechosen source values.

        ``source_gates`` retains its original argument name for callers using
        explicit raw gates; finite thresholds interpret it as source values.
        Use the separate phases when reception can change those values.
        """
        delivery_state, payload = self.read_due(state)
        return (
            self.advance(weights, delivery_state, source_gates, schedule=schedule),
            payload,
        )

    def commit_state_(self, netcon, state):
        """Commit independent copies of validated carry into a compatible NetCon.

        This restores queues, counters, detector history, and tracked events.
        Align effective weights and floating schedule values separately when
        resuming native execution; they are parameters rather than carry.
        """
        if torch.compiler.is_compiling():
            raise FunctionalizationError("Commit NetCon state outside torch.compile.")
        self._validate_compatible(netcon)
        self._validate_state(state)
        with torch.inference_mode(False):
            copied = {
                name: value.clone(memory_format=torch.contiguous_format)
                for name, value in state.items()
            }
            return netcon.restore_dict_from_checkpoint(copied)


def make_functional_netcon(netcon) -> tuple[FunctionalNetCon, NetConTensors]:
    """Lower one initialized dense NetCon with explicit numerical source values."""
    projection = FunctionalNetCon(netcon)
    return projection, projection.extract()


__all__ = ["FunctionalNetCon", "NetConTensors", "make_functional_netcon"]
