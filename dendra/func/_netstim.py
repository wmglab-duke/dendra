"""Explicit-state NetStim renewal over the shared network-layer transition."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import NamedTuple

import torch

from dendra.models.networks._heap_calendar import (
    heap_calendar_consume,
    heap_calendar_head,
    validate_heap_calendar,
)
from dendra.models.networks._netstim_renewal import (
    NetStimState,
    broadcast_netstim_time,
    netstim_initial_state,
    netstim_renewal,
    sample_netstim_exponential,
)
from dendra.models.networks._tensor_schedules import (
    netstim_tensor_schedule,
    resolve_schedule_values,
)
from dendra.models.networks.netstim import NetStim
from dendra.models.parametric import PositiveParam

from ._schedules import TensorScheduleTensors, _TensorSchedulePlan
from ._tensor_contracts import validate_tensor
from ._torch_parametrizations import canonical_type
from ._types import FunctionalizationError


class NetStimTensors(NamedTuple):
    """Independent parameters, calendar constants, and clock/RNG/cursor snapshots."""

    parameters: dict[str, torch.Tensor]
    constants: dict[str, torch.Tensor]
    state: dict[str, torch.Tensor]
    schedule: TensorScheduleTensors | None = None


class NetStimDraw(NamedTuple):
    """An exponential draw and its matching post-draw generator byte state.

    Sampling helpers produce replayable packets without touching the native
    NetStim or the input state. Keep the two fields together: manually supplied
    draws are accepted, but their RNG provenance cannot be inferred from values.
    """

    exp_rand: torch.Tensor
    rng_state: torch.Tensor


class NetStimOutputs(NamedTuple):
    """Hard boolean spikes and their differentiable straight-through gates."""

    spikes: torch.Tensor
    spike_gate: torch.Tensor


def _snapshot(value, *, parameter=False):
    with torch.inference_mode(False):
        result = value.detach().clone(memory_format=torch.contiguous_format)
        if parameter:
            result.requires_grad_(value.requires_grad)
        return result


def _eligible(source):
    if canonical_type(source) is not NetStim:
        raise FunctionalizationError("Expected an initialized canonical NetStim.")
    if not getattr(source, "_renewal_initialized", False):
        raise FunctionalizationError("Initialize the NetStim before lowering it.")
    if source._heap_nan_heaps is not None:
        raise FunctionalizationError(
            "Functional NetStim cannot lower NaN heap calendars; clear or replace "
            "their events before lowering."
        )
    interval = source.interval
    if (
        canonical_type(interval) is not PositiveParam
        or "forward" in interval.__dict__
        or "_compute" in interval.__dict__
        or interval._forward_hooks
        or interval._forward_pre_hooks
        or "forward" in source.__dict__
        or any(
            name in vars(source)
            or inspect.getattr_static(type(source), name)
            is not inspect.getattr_static(NetStim, name)
            for name in ("renewal_parameters", "_broadcast_to_state")
        )
        or source._forward_hooks
        or source._forward_pre_hooks
    ):
        raise FunctionalizationError(
            "Functional NetStim requires its canonical parameter and renewal adapters."
        )


class FunctionalNetStim:
    """A pure renewal plan with explicit random draws and checkpointable RNG.

    Lower an initialized :class:`~dendra.NetStim` with optional heap and tensor
    schedules. Effective ``interval``, ``start``, ``noise``, and ``tau`` use the
    full native state shape; ``max_spikes`` is an explicit integer constant.
    Parameters are independent leaf snapshots, so gradients never reach the
    source module.

    :meth:`step` calls the same pure transition as native NetStim. Draw one
    packet with :meth:`sample_step_draw` outside transformed/compiled code, then
    pass it into the step. Every native forward consumes one full-shape draw,
    including deterministic generators and calls that emit no spikes. RNG byte
    state travels with clock carry, while sampling remains outside the hot
    graph. ``bptt=False`` detaches the renewed clock as native execution does;
    emitted surrogate gates retain their current-step gradients.

    :meth:`initial_state` resets clocks using an explicit initialization packet.
    Its cold sampler reseeds when the lowered native NetStim has a fixed seed;
    with ``seed=None`` it continues the supplied state's RNG stream. A fully
    deterministic initialization consumes no draw. Initialization clocks are
    detached, matching native initialization.

    Configuration changes require lowering again. Runtime carry changes do not
    invalidate a plan. :meth:`commit_state_` transfers clocks and both RNG states
    without changing native parameters or constants; those must match explicit
    values when continuing the same trajectory natively. Tensor schedule values
    are explicit in :class:`TensorScheduleTensors`; routing and source indices
    are frozen at lowering, along with ``schedule_dt``. Set the native timestep
    before lowering. Heap calendars use non-differentiable times (Float64 on
    CPU/CUDA, simulation dtype on MPS) and CSR row offsets in ``constants``.
    Their absolute per-row cursors and any legacy head-only override belong to explicit carry.
    Calendar topology is fixed; changing it requires lowering again.
    Initialization preserves consumed cursors and clears legacy head overrides,
    matching native initialization.
    """

    def __init__(self, source):
        _eligible(source)
        self._source = source
        self._generation = source._renewal_generation
        self._config_generation = source._renewal_config_generation
        self._training = bool(source.training)
        self._seed = source.seed
        self._schedule_mode = (
            bool(source.diff_scheduled_times),
            float(source.sched_width),
        )
        self.shape = tuple(source.shape)
        self.device = source.next_stoch_time.device
        self.dtype = source.next_stoch_time.dtype
        self._heap_times = _snapshot(source.heap_times)
        self._heap_row_offsets = _snapshot(source.heap_row_offsets)
        self._heap_rows = source.numel()
        self._heap_depth = source.heap_depth
        self._dt = _snapshot(source.schedule_dt)
        self._schedule = _TensorSchedulePlan(
            source, index_name="sched_flat_idx", device=self.device, dtype=self.dtype
        )
        checkpoint = source.state_dict_for_checkpoint()
        self._state_schema = {
            name: (
                tuple(checkpoint[name].shape),
                checkpoint[name].dtype,
                checkpoint[name].device,
            )
            for name in (
                "next_stoch_time",
                "spike_counts",
                "t_last",
                "rng_state",
                "seeder_state",
                "heap_cursor",
                "heap_head_override",
                "heap_override_active",
            )
        }
        self._validate_compatible(source)

    def _validate_source(self):
        source = self._source
        if (
            canonical_type(source) is not NetStim
            or not source._renewal_initialized
            or source._renewal_generation != self._generation
            or source._renewal_config_generation != self._config_generation
            or bool(source.training) != self._training
            or source.seed != self._seed
            or tuple(source.shape) != self.shape
            or source.next_stoch_time.device != self.device
            or source.next_stoch_time.dtype != self.dtype
            or (bool(source.diff_scheduled_times), float(source.sched_width))
            != self._schedule_mode
        ):
            raise FunctionalizationError(
                "The source NetStim configuration changed; initialize and lower it again."
            )

    _validate_tensor = staticmethod(validate_tensor)

    def _validate_parameters(self, parameters, constants):
        if not isinstance(parameters, Mapping) or set(parameters) != {
            "interval",
            "start",
            "noise",
            "tau",
        }:
            raise ValueError(
                "NetStim parameters must contain interval, start, noise, and tau"
            )
        for name, value in parameters.items():
            self._validate_tensor(
                value,
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
                label=f"parameters.{name}",
            )
        if not isinstance(constants, Mapping) or set(constants) != {
            "max_spikes",
            "heap_times",
            "heap_row_offsets",
        }:
            raise ValueError(
                "NetStim constants must contain max_spikes, heap_times, and heap_row_offsets"
            )
        self._validate_tensor(
            constants["max_spikes"],
            shape=self.shape,
            dtype=torch.long,
            device=self.device,
            label="constants.max_spikes",
        )
        for name, expected in (
            ("heap_times", self._heap_times),
            ("heap_row_offsets", self._heap_row_offsets),
        ):
            value = constants[name]
            self._validate_tensor(
                value,
                shape=tuple(expected.shape),
                dtype=expected.dtype,
                device=expected.device,
                label=f"constants.{name}",
            )
            if value.requires_grad:
                raise ValueError(f"constants.{name} must not require gradients")

    def _validate_state(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self._state_schema):
            raise ValueError(
                "NetStim state must match its clock/RNG/calendar checkpoint schema"
            )
        for name, (shape, dtype, device) in self._state_schema.items():
            self._validate_tensor(
                state[name],
                shape=shape,
                dtype=dtype,
                device=device,
                label=f"state.{name}",
            )

    def _validate_draw(self, draw):
        if not isinstance(draw, NetStimDraw):
            raise TypeError("draw must be a NetStimDraw packet")
        self._validate_tensor(
            draw.exp_rand,
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            label="draw.exp_rand",
        )
        shape, dtype, device = self._state_schema["rng_state"]
        self._validate_tensor(
            draw.rng_state,
            shape=shape,
            dtype=dtype,
            device=device,
            label="draw.rng_state",
        )

    def _validate_compatible(self, source):
        _eligible(source)
        if (
            tuple(source.shape) != self.shape
            or source.next_stoch_time.device != self.device
            or source.next_stoch_time.dtype != self.dtype
            or bool(source.training) != self._training
            or source.seed != self._seed
            or (bool(source.diff_scheduled_times), float(source.sched_width))
            != self._schedule_mode
        ):
            raise FunctionalizationError(
                "NetStim configuration does not match this plan."
            )
        checkpoint = source.state_dict_for_checkpoint()
        self._validate_state({name: checkpoint[name] for name in self._state_schema})
        validate_heap_calendar(
            source.heap_times,
            source.heap_row_offsets,
            source.heap_cursor,
            rows=source.numel(),
            device=self.device,
            simulation_dtype=self.dtype,
        )
        if not torch.equal(source.heap_times, self._heap_times) or not torch.equal(
            source.heap_row_offsets, self._heap_row_offsets
        ):
            raise FunctionalizationError(
                "NetStim heap calendar does not match this frozen plan."
            )
        self._schedule.validate_compatible(source)
        if not torch.equal(source.schedule_dt, self._dt):
            raise FunctionalizationError(
                "NetStim schedule timestep does not match this plan."
            )
        if bool((self._schedule.indices >= source.numel()).any()):
            raise FunctionalizationError(
                "NetStim tensor schedule index is outside its source."
            )
        if source._rng.device != self.device:
            raise FunctionalizationError(
                "Initialize NetStim RNG on its state device first."
            )

    def extract(self, source=None) -> NetStimTensors:
        """Snapshot effective values and replayable carry without source mutation."""
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "Extract NetStim tensors before torch.compile."
            )
        source = self._source if source is None else source
        self._validate_compatible(source)
        # The native materializer bypasses canonical inference caches while
        # using the same broadcast/cast policy and parametrization getters.
        with torch.inference_mode(False), torch.enable_grad():
            values = source.renewal_parameters()
        parameters = {
            name: _snapshot(torch.broadcast_to(value, self.shape), parameter=True)
            for name, value in values.items()
        }
        constants = {
            "max_spikes": _snapshot(source.max_spikes),
            "heap_times": _snapshot(source.heap_times),
            "heap_row_offsets": _snapshot(source.heap_row_offsets),
        }
        checkpoint = source.state_dict_for_checkpoint()
        state = {name: _snapshot(checkpoint[name]) for name in self._state_schema}
        self._validate_parameters(parameters, constants)
        return NetStimTensors(
            parameters, constants, state, self._schedule.extract(source)
        )

    def _generator(self, state):
        generator = torch.Generator(device=self.device)
        # Tensor tape rows can be contiguous views with a nonzero storage
        # offset. Generator.set_state reads their backing storage from zero;
        # materialize the selected RNG bytes before crossing that boundary.
        generator.set_state(state["rng_state"].clone())
        return generator

    def _sample(self, state, *, initialize=False, parameters=None):
        if torch.compiler.is_compiling():
            raise FunctionalizationError("Sample NetStim draws outside torch.compile.")
        self._validate_source()
        self._validate_state(state)
        generator = self._generator(state)
        if initialize and self._seed is not None:
            generator.manual_seed(self._seed)
        consume = not initialize or bool(torch.any(parameters["noise"] > 0))
        with torch.inference_mode(False), torch.no_grad():
            if consume:
                exp_rand = sample_netstim_exponential(
                    generator, self.shape, device=self.device, dtype=self.dtype
                )
            else:
                exp_rand = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
            return NetStimDraw(exp_rand, generator.get_state())

    def sample_step_draw(self, state) -> NetStimDraw:
        """Draw once from an independent copy of the explicit RNG state."""
        return self._sample(state)

    def sample_initialization_draw(self, parameters, state) -> NetStimDraw:
        """Match native initialization's reseeding and conditional draw policy."""
        if not isinstance(parameters, Mapping) or "noise" not in parameters:
            raise ValueError("Initialization sampling requires parameters.noise")
        self._validate_tensor(
            parameters["noise"],
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            label="parameters.noise",
        )
        return self._sample(state, initialize=True, parameters=parameters)

    def initial_state(self, parameters, constants, state, draw):
        """Reset clocks/RNG, preserving consumed cursors and clearing legacy heads."""
        self._validate_source()
        self._validate_parameters(parameters, constants)
        self._validate_state(state)
        self._validate_draw(draw)
        clocks = netstim_initial_state(
            parameters["interval"],
            parameters["start"],
            parameters["noise"],
            draw.exp_rand,
        )
        return {
            "next_stoch_time": clocks.next_stoch_time,
            "spike_counts": clocks.spike_counts,
            "t_last": clocks.t_last,
            "rng_state": draw.rng_state,
            "seeder_state": state["seeder_state"],
            "heap_cursor": state["heap_cursor"],
            "heap_head_override": torch.full_like(
                state["heap_head_override"], torch.inf
            ),
            "heap_override_active": torch.zeros_like(state["heap_override_active"]),
        }

    def step(self, parameters, constants, state, t, draw, *, bptt=False, schedule=None):
        """Return next clock/RNG carry and hard/surrogate outputs for time ``t``.

        A plan with tensor events or bound sources requires an explicit
        ``schedule`` packet. Event values may change; routing and source-index
        maps retain the topology captured at lowering.
        """
        self._validate_source()
        self._validate_parameters(parameters, constants)
        self._validate_state(state)
        self._validate_draw(draw)
        self._schedule.validate(schedule)
        time = broadcast_netstim_time(
            t, self.shape, device=self.device, dtype=self.dtype
        )
        heap_head = heap_calendar_head(
            constants["heap_times"],
            constants["heap_row_offsets"][1:],
            state["heap_cursor"],
            dtype=self.dtype,
        ).reshape(self.shape)
        heap_head = torch.where(
            state["heap_override_active"], state["heap_head_override"], heap_head
        )
        scheduled_gate = scheduled_hard = None
        if self._schedule.active:
            times, amplitudes = resolve_schedule_values(
                schedule.times,
                schedule.amplitudes,
                self._schedule.time_idx,
                self._schedule.amplitude_idx,
                time_source=(
                    schedule.time_source if self._schedule.has_time_refs else None
                ),
                weight_source=(
                    schedule.amplitude_source
                    if self._schedule.has_amplitude_refs
                    else None
                ),
                time_dtype=self.dtype,
                weight_dtype=self.dtype,
                device=self.device,
            )
            scheduled_gate, scheduled_hard = netstim_tensor_schedule(
                self._schedule.indices,
                times,
                amplitudes,
                time,
                self._dt,
                shape=self.shape,
                differentiable_times=self._schedule_mode[0],
                width=self._schedule_mode[1],
            )
        transition = netstim_renewal(
            NetStimState(
                state["next_stoch_time"], state["spike_counts"], state["t_last"]
            ),
            parameters["interval"],
            parameters["noise"],
            constants["max_spikes"],
            time,
            draw.exp_rand,
            tau=parameters["tau"],
            bptt=bptt,
            next_sched_time=heap_head,
            scheduled_gate=scheduled_gate,
            scheduled_hard=scheduled_hard,
        )
        return {
            "next_stoch_time": (
                transition.state.next_stoch_time
                if bptt
                else transition.state.next_stoch_time.detach()
            ),
            "spike_counts": transition.state.spike_counts,
            "t_last": transition.state.t_last,
            "rng_state": draw.rng_state,
            "seeder_state": state["seeder_state"],
            "heap_cursor": heap_calendar_consume(
                constants["heap_times"],
                constants["heap_row_offsets"][1:],
                state["heap_cursor"],
                torch.broadcast_to(time, self.shape).reshape(self._heap_rows),
                transition.heap_fires.reshape(self._heap_rows),
                depth=self._heap_depth,
                dtype=self.dtype,
            ),
            "heap_head_override": state["heap_head_override"],
            "heap_override_active": state["heap_override_active"]
            & ~transition.heap_fires,
        }, NetStimOutputs(transition.spikes, transition.spike_gate)

    def commit_state_(self, source, state):
        """Transfer validated clocks, RNG, and cursors to a compatible native NetStim.

        Native training restores retain the incoming clock's autograd history;
        evaluation restores detach the clock, clearing any prior native history.
        Schedule values and bound sources are left as configured on ``source``;
        align them with the explicit packet before continuing a changed schedule.
        """
        if torch.compiler.is_compiling():
            raise FunctionalizationError("Commit NetStim state outside torch.compile.")
        self._validate_compatible(source)
        self._validate_state(state)
        validate_heap_calendar(
            self._heap_times,
            self._heap_row_offsets,
            state["heap_cursor"],
            rows=self._heap_row_offsets.numel() - 1,
            device=self.device,
            simulation_dtype=self.dtype,
        )
        # Validate generator byte contents before native runtime is touched.
        self._generator(state)
        torch.Generator().set_state(state["seeder_state"])
        with torch.inference_mode(False):
            copied = {
                name: value.clone(memory_format=torch.contiguous_format)
                for name, value in state.items()
            }
            checkpoint = source.state_dict_for_checkpoint()
            checkpoint.update(copied)
            source.restore_dict_from_checkpoint(checkpoint)
        return source


def make_functional_netstim(source) -> tuple[FunctionalNetStim, NetStimTensors]:
    """Return a pure renewal plan and independent state/parameter snapshots.

    ``source`` is an initialized NetStim with its timestep/schedules configured.
    Returns ``(FunctionalNetStim, NetStimTensors)``. Initialization and step draws
    remain explicit; full Network lowering binds this component automatically.
    """
    plan = FunctionalNetStim(source)
    return plan, plan.extract()


__all__ = [
    "FunctionalNetStim",
    "NetStimDraw",
    "NetStimOutputs",
    "NetStimTensors",
    "make_functional_netstim",
]
