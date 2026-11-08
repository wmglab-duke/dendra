"""Host-side Python schedulers for explicit-state functional transitions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch.utils import _pytree as pytree
from torch.utils.checkpoint import checkpoint

import dendra
from dendra.models.core import _duration_step_budget, _validate_time_scalar

from ._binding import BoundPopulation
from ._callbacks import (
    FunctionalCallbacks,
    _CallbackCollector,
    _collect_callback_emissions,
    _complete_callback_results,
    _concatenate_callback_parts,
    _host_callback_parts,
    _setup_callback_state,
)
from ._continuation import (
    _complete_bound_continuation,
    _resolve_bound_continuation,
)
from ._draws import (
    RuntimeDraws,
    _normalize_runtime_draws,
    _runtime_draws_binding,
    _with_runtime_draws,
)
from ._inputs import (
    Inputs,
    _index_runtime_inputs,
    _legacy_step_input,
    _population_inputs,
    _RuntimeInputs,
    _slice_runtime_inputs,
)
from ._network_binding import BoundNetwork
from ._population import FunctionalPopulation
from ._recording import _host_recording, _recording_span
from ._stimulation_clock import _accepted_bank_clock, _sampling_time_tape
from ._stimuli import FunctionalExtra, StimulusTensors
from ._tensor_contracts import _tensor_binding
from ._types import FunctionalizationError, RolloutInput, StepInput

_StepCallable = Callable[[Mapping[str, object], StepInput], tuple[object, object]]
_MISSING = object()
_OWNED_RUNTIME_DRAW_SAMPLER = FunctionalPopulation.sample_runtime_draws
_OWNED_RUNTIME_SAMPLERS = {
    name: getattr(FunctionalPopulation, name)
    for name in (
        "sample_noise_draw_tape",
        "sample_stimulation_draw_tape",
        "_sample_missing_runtime_draws",
    )
}


@dataclass(frozen=True)
class _OwnedStepBinding:
    """Exact public binding whose unchecked tensor transition Dendra owns."""

    functional: FunctionalPopulation
    parameters: Mapping[str, torch.Tensor]
    prepared: object
    compiled_chunk: object | None = None


@dataclass(frozen=True)
class _StepExecution:
    """A generic public step or its validated tensor-only fast path."""

    public_step: _StepCallable
    binding: _OwnedStepBinding | None = None
    prepared_values: Mapping[str, object] | None = None
    compiled_chunks: Mapping[int, object] | None = None
    checkpoint_parameter_binding: tuple | None = None

    def __call__(self, state, inputs: StepInput):
        if self.binding is None:
            return self.public_step(state, _legacy_step_input(inputs))
        if self.binding.compiled_chunk is None:
            return self.binding.functional._step_values(
                self.binding.parameters,
                self.prepared_values,
                state,
                inputs,
            )
        compiled_chunk = self.binding.compiled_chunk
        if self.compiled_chunks is not None:
            compiled_chunk = self.compiled_chunks[1]
        return compiled_chunk._step_values(
            self.binding.parameters,
            self.prepared_values,
            state,
            inputs,
        )


def _owned_step_binding(functional, step) -> _OwnedStepBinding | None:
    """Read explicit ownership; infer legacy partials only in their adapter."""
    if type(step) is BoundPopulation:
        if step.functional is not functional:
            return None
        return _OwnedStepBinding(
            functional, step._parameters, step.prepared, step.compiled_chunk
        )
    from ._runner_compat import legacy_owned_step_binding

    return legacy_owned_step_binding(functional, step)


def _prepare_step_execution(functional, step) -> _StepExecution:
    """Validate a visible functional binding once and expose its pure core."""
    binding = _owned_step_binding(functional, step)
    if binding is None:
        return _StepExecution(step)
    functional._validate_parameters(binding.parameters)
    prepared_values = functional._validate_prepared(
        binding.prepared,
        binding.parameters,
    )
    return _StepExecution(step, binding, prepared_values)


def _prepare_compiled_schedule(execution, steps, chunklength, *, callbacks):
    """Construct needed kernel specializations outside execution and replay.

    Host/checkpoint boundaries remain authoritative. Within each such span,
    consume full compiled chunks and at most one shorter tail. Callback kernels
    include every per-step update and return explicit carry and stacked samples.
    Empty schedules do not construct any new kernels.
    """
    binding = execution.binding
    if steps == 0 or binding is None or binding.compiled_chunk is None:
        return execution
    chunk = binding.compiled_chunk
    if chunk.steps == 1 and callbacks is None:
        return execution
    spans = {min(steps, chunklength)}
    if steps % chunklength:
        spans.add(steps % chunklength)
    sizes = set()
    for span in spans:
        sizes.add(min(chunk.steps, span))
        if span % chunk.steps:
            sizes.add(span % chunk.steps)
    kernels = {
        size: (
            chunk._runner_chunk(size)
            if callbacks is None
            else chunk._runner_callback_chunk(callbacks, size)
        )
        for size in sorted(sizes)
    }
    return replace(execution, compiled_chunks=kernels)


def _refresh_checkpoint_execution(
    execution: _StepExecution, state, *, source_validated=False
) -> _StepExecution:
    """Revalidate bindings after authored sampling and at checkpoint replay.

    Non-reentrant checkpointing invokes a chunk again during backward, after
    user code has had an opportunity to mutate captured tensors or the source
    model.  Rechecking at that replay boundary preserves the public freshness
    contract without putting Python validation back inside the timestep loop.
    An owned sampler boundary can reuse its immediate source validation while
    still checking independent preparation inputs after authored sampling.
    """
    binding = execution.binding
    if binding is None:
        return execution
    functional = binding.functional
    if not source_validated:
        functional._validate_source()
    functional._validate_parameters(binding.parameters)
    functional._validate_state(state)
    prepared_values = functional._validate_prepared(
        binding.prepared,
        binding.parameters,
    )
    if (
        execution.checkpoint_parameter_binding is not None
        and execution.checkpoint_parameter_binding
        != _tensor_binding(binding.parameters)
    ):
        raise FunctionalizationError(
            "parameters changed after checkpoint forward; run a new forward "
            "before backward"
        )
    return replace(execution, prepared_values=prepared_values)


def _validate_step_plan(functional, step) -> None:
    if type(step) is BoundPopulation:
        if step.functional is not functional:
            raise FunctionalizationError(
                "step belongs to a different FunctionalPopulation plan"
            )
        return
    from ._runner_compat import _validate_step_plan as validate_legacy_step

    validate_legacy_step(functional, step)


def _validate_visible_step_dt(
    functional,
    step,
    dt: float,
    *,
    prepared_values: Mapping[str, object] | None = None,
) -> None:
    """Check explicit preparation without inspecting callable metadata."""
    if type(step) is not BoundPopulation:
        from ._runner_compat import _validate_visible_step_dt as validate_legacy_dt

        return validate_legacy_dt(functional, step, dt, prepared_values=prepared_values)
    parameters, prepared = step._parameters, step.prepared
    if prepared_values is None:
        prepared_values = functional._validate_prepared(prepared, parameters)
    prepared_dt, _tangent = torch.autograd.forward_ad.unpack_dual(
        prepared_values["integrator"]["dt"]
    )
    prepared_value = float(
        prepared_dt.detach().to(device="cpu").to(torch.float64).item()
    )
    runner_value = float(torch.tensor(dt, dtype=prepared_dt.dtype, device="cpu").item())
    if prepared_value != runner_value:
        raise FunctionalizationError(
            "step clock advance does not match runner dt: bound prepared dt is "
            f"{prepared_value!r}, but runner dt becomes {runner_value!r} in "
            f"{prepared_dt.dtype}. Prepare and bind the step with constants "
            "containing the same dt passed to the host runner."
        )


def _validate_duration_remainder(state, *, expected: float | None = None) -> float:
    remainder = _duration_remainder(state)
    if remainder.requires_grad:
        raise FunctionalizationError(
            "state control.duration_remainder must not require gradients"
        )
    _primal, tangent = torch.autograd.forward_ad.unpack_dual(remainder)
    if tangent is not None:
        raise FunctionalizationError(
            "state control.duration_remainder must not carry a forward-AD tangent"
        )
    value = float(remainder.detach().item())
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "state control.duration_remainder must be finite and non-negative"
        )
    if expected is not None and value != expected:
        raise FunctionalizationError(
            "step changed host-owned state control.duration_remainder"
        )
    return value


def _validate_execution(functional, step, state) -> float:
    _validate_execution_context(functional, step)
    functional._validate_source()
    functional._validate_state(state)
    return _validate_duration_remainder(state)


def _validate_execution_context(functional, step) -> None:
    if not isinstance(functional, FunctionalPopulation):
        raise TypeError("functional must be a FunctionalPopulation")
    if not callable(step):
        raise TypeError("step must be callable")
    _validate_step_plan(functional, step)
    if torch.compiler.is_compiling():
        raise FunctionalizationError(
            "functional host schedulers cannot be passed through torch.compile; "
            "compile the supplied step callable instead"
        )
    if torch._C._are_functorch_transforms_active():
        raise FunctionalizationError(
            "functional host duration scheduling is not a torch.func transform; "
            "transform the supplied step callable instead"
        )


def _validate_chunklength(chunklength) -> int:
    if isinstance(chunklength, bool) or not isinstance(chunklength, int):
        raise ValueError("chunklength must be a positive integer")
    if chunklength <= 0:
        raise ValueError("chunklength must be a positive integer")
    return chunklength


def _validate_runner_dt(functional: FunctionalPopulation, dt) -> float:
    """Normalize the host scheduler clock without changing the transition."""
    return _validate_time_scalar(
        functional.dt if dt is None else dt,
        name="dt",
        positive=True,
    )


def _duration_remainder(state) -> torch.Tensor:
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    control = state.get("control")
    if not isinstance(control, Mapping) or "duration_remainder" not in control:
        raise KeyError("state must contain control.duration_remainder")
    remainder = control["duration_remainder"]
    if not torch.is_tensor(remainder):
        raise TypeError("state control.duration_remainder must be a Tensor")
    if tuple(remainder.shape) != ():
        raise ValueError("state control.duration_remainder must be a scalar Tensor")
    if remainder.device.type != "cpu" or remainder.dtype != torch.float64:
        raise ValueError("state control.duration_remainder must use cpu/torch.float64")
    return remainder


def _capture_clock(state) -> torch.Tensor:
    """Snapshot the primal model clock without joining its autograd graph."""
    clock = state["clock"]["t"]
    primal, _tangent = torch.autograd.forward_ad.unpack_dual(clock)
    return primal.detach().clone()


def _validate_clock_advance(
    start: torch.Tensor,
    state,
    *,
    steps: int,
    dt: float,
) -> None:
    """Verify that a bound transition and its host scheduler share one clock."""
    end, _tangent = torch.autograd.forward_ad.unpack_dual(state["clock"]["t"])

    # The host runners are intentionally outside compile and torch.func.  Do a
    # single synchronization per chunk, rather than one per step, and inspect
    # detached primals so this correctness guard does not alter reverse- or
    # forward-mode differentiation through the transition itself.  Moving to
    # CPU before float64 conversion also keeps this valid on MPS.
    clocks = torch.stack((start, end.detach())).to(device="cpu").to(torch.float64)
    start_value, end_value = clocks.tolist()
    observed_advance = end_value - start_value
    runner_dt = float(torch.tensor(dt, dtype=start.dtype, device="cpu").item())
    expected_advance = steps * runner_dt

    # A correct transition repeatedly evaluates fl(t + dt).  Each addition
    # contributes at most half one ULP at the largest magnitude traversed by
    # this chunk.  This is both substantially tighter than a tolerance scaled
    # directly by the absolute clock and the strongest distinction available
    # from clock motion alone: at a sufficiently coarse float32 clock, nearby
    # timesteps can round to the same representable increment.  Inspectable
    # public bindings are therefore checked exactly above as well.
    largest_clock = max(
        abs(start_value),
        abs(end_value),
        abs(start_value + expected_advance),
    )
    magnitude = torch.tensor(largest_clock, dtype=start.dtype, device="cpu")
    spacing = float(
        (
            torch.nextafter(
                magnitude, torch.full_like(magnitude, float("inf"), device="cpu")
            )
            - magnitude
        )
        .to(torch.float64)
        .item()
    )
    accumulation_budget = 0.5 * steps * spacing
    host_budget = 2.0 * max(
        math.ulp(observed_advance),
        math.ulp(expected_advance),
    )
    error = abs(observed_advance - expected_advance)
    if not math.isfinite(error) or error > accumulation_budget + host_budget:
        observed_dt = observed_advance / steps
        raise FunctionalizationError(
            "step clock advance does not match runner dt: observed "
            f"{observed_dt!r} per step across {steps} step(s), but runner dt "
            f"is {dt!r}. Prepare and bind the step with constants containing "
            "the same dt passed to the host runner."
        )


def _replace_duration_remainder(state, value: float):
    remainder = torch.tensor(value, device="cpu", dtype=torch.float64)
    updated = dict(state)
    updated_control = dict(state["control"])
    updated_control["duration_remainder"] = remainder
    updated["control"] = updated_control
    return updated


def _stage_duration(state, tstop, dt: float, *, name="tstop"):
    duration = _validate_time_scalar(tstop, name=name, positive=False)
    remainder = _duration_remainder(state)
    pending = float(remainder.detach().item())
    steps, next_remainder = _duration_step_budget(duration, dt, pending)
    return _replace_duration_remainder(state, next_remainder), steps


def _validate_rollout_input_container(inputs):
    return _population_inputs(inputs)


def _resolve_inputs(functional, inputs, steps: int | None):
    ve, intra, resolved_steps = functional._resolve_rollout_inputs(inputs, steps)
    return inputs._replace(ve=ve, intra=intra), resolved_steps


def _slice_inputs(inputs, start: int, stop: int):
    return _slice_runtime_inputs(inputs, start, stop)


def _canonical_runtime_sampling_available(functional, inputs):
    """Use the canonical sampler without bypassing legacy public overrides."""
    if (
        inputs.draws is not None
        or inputs.noise is not None
        or inputs.stimulation is not None
    ):
        return False
    if not (
        functional._noise_plan is not None
        and functional._noise_plan.runtime
        or getattr(functional, "_poisson_plan", None) is not None
    ):
        return False
    return all(
        getattr(getattr(functional, name), "__self__", None) is functional
        and getattr(getattr(functional, name), "__func__", None) is original
        for name, original in _OWNED_RUNTIME_SAMPLERS.items()
    )


def _normalize_sampled_inputs(functional, inputs, steps, *, qualify_draws=False):
    """Qualify canonical packets returned through a legacy sampler boundary."""
    legacy_canonical = isinstance(getattr(inputs, "noise", None), RuntimeDraws)
    inputs = _population_inputs(inputs)
    if inputs.draws is not None and (legacy_canonical or qualify_draws):
        # A replaced legacy sampler can return the canonical public packet and
        # execute authored code after sampling. Requalify before unchecked use.
        functional._validate_source()
        functional._runtime_draw_plan.validate(inputs.draws, steps=steps)
    return inputs


def _prepare_chunk_inputs(functional, state, inputs, start, stop, *, canonical=False):
    """Slice explicit inputs and sample missing packets at one host boundary."""
    chunk_steps = stop - start
    chunk_inputs = _slice_inputs(inputs, start, stop)
    if chunk_inputs.draws is not None:
        return chunk_inputs
    if canonical and _canonical_runtime_sampling_available(functional, chunk_inputs):
        public_sampler = functional.sample_runtime_draws
        owned_sampler = (
            getattr(public_sampler, "__self__", None) is functional
            and getattr(public_sampler, "__func__", None) is _OWNED_RUNTIME_DRAW_SAMPLER
        )
        sampler = (
            functional._sample_runtime_draws_source_validated
            if owned_sampler
            else public_sampler
        )
        packet = sampler(state, steps=chunk_steps)
        chunk_inputs = _with_runtime_draws(chunk_inputs, packet)
        # Public samplers and Mapping providers can execute authored Python.
        if not owned_sampler:
            functional._validate_source()
        functional._runtime_draw_plan.validate(chunk_inputs.draws, steps=chunk_steps)
        return chunk_inputs
    chunk_inputs = functional._sample_missing_runtime_draws(
        state, chunk_inputs, chunk_steps
    )
    if isinstance(chunk_inputs, _RuntimeInputs) and chunk_inputs.draws is not None:
        return _normalize_sampled_inputs(
            functional, chunk_inputs, chunk_steps, qualify_draws=True
        )
    if (
        functional._noise_plan is not None
        and functional._noise_plan.runtime
        and chunk_inputs.noise is None
    ):
        chunk_inputs = chunk_inputs._replace(
            noise=functional.sample_noise_draw_tape(state, chunk_steps)
        )
    if (
        getattr(functional, "_poisson_plan", None) is not None
        and chunk_inputs.stimulation is None
    ):
        chunk_inputs = chunk_inputs._replace(
            stimulation=functional.sample_stimulation_draw_tape(state, chunk_steps)
        )
    return _normalize_sampled_inputs(functional, chunk_inputs, chunk_steps)


def _prepare_extra(functional, extra, inputs):
    if extra is None:
        return None, None
    if functional.extra.enabled:
        raise ValueError(
            "runtime extra cannot be combined with extra bound by make_functional"
        )
    if inputs.ve is not None:
        raise ValueError("explicit ve cannot be combined with extra")
    plan = extra if isinstance(extra, FunctionalExtra) else functional.make_extra(extra)
    functional._validate_extra_plan(plan)
    from ._poisson_stimulation import randomized_nodes

    if any(randomized_nodes(waveform) for waveform in plan._waveforms._sources):
        raise FunctionalizationError(
            "Randomized Poisson extra must be bound through "
            "make_functional(..., extra=...) or dn.compile(..., extra=...); "
            "call-local extra has no explicit stimulation RNG carry."
        )
    return plan, plan.extract()


def _assemble_extra_chunk(
    functional,
    plan: FunctionalExtra | None,
    tensors: StimulusTensors | None,
    state,
    start: int,
    steps: int,
    total_steps: int,
    dt: float,
    *,
    runtime_extra_slot: bool = False,
    times: torch.Tensor | None = None,
    clock_dt: torch.Tensor | None = None,
):
    if plan is None:
        if runtime_extra_slot:
            return torch.zeros(
                (steps, *functional.shape),
                device=functional.device,
                dtype=functional.dtype,
            )
        return None
    if times is None:
        dt_tensor = (
            torch.tensor(dt, device=functional.device, dtype=functional.dtype)
            if clock_dt is None
            else clock_dt
        )
        # Host schedules sample waveforms at the clock accepted by each
        # transition. Reproduce that bounded recurrence using tensor additions:
        # start + arange * dt changes float32 pulse-edge ordering. Neither the
        # clock nor the waveform needs a per-step host read here. Network
        # schedules can supply their canonical origin/step clock explicitly.
        # Owned bindings supply prepared dt to retain timing derivatives;
        # generic callables retain the host scheduler's scalar fallback.
        clock = state["clock"]["t"]
        clocks = []
        for _ in range(steps):
            clocks.append(clock)
            clock = (clock.reshape(1) + dt_tensor.reshape(1)).squeeze(0)
        times = torch.stack(clocks) if clocks else clock.new_empty((0,))
    if plan.uses_waveforms:
        ve = plan.assemble_tensors(tensors, times)
    else:
        ve = plan.assemble(
            tensors.parameters,
            tensors.constants,
            times,
            temporal_slice=slice(start, start + steps),
            total_steps=total_steps,
        )
    if ve is None and runtime_extra_slot:
        return torch.zeros(
            (steps, *functional.shape),
            device=functional.device,
            dtype=functional.dtype,
        )
    return ve


def _validate_zero_step_extra(functional, plan, tensors, state, dt: float) -> None:
    """Validate tensor-temporal horizons even when no chunk will execute.

    Waveforms remain lazy on an empty simulation. Tensor time specifications,
    however, have an explicit call-local horizon and must contain zero samples
    when the resolved run has zero steps.
    """
    if plan is None or plan.uses_waveforms:
        return
    plan.assemble(
        tensors.parameters,
        tensors.constants,
        functional._stimulation_times(
            state,
            0,
            torch.tensor(dt, device=functional.device, dtype=functional.dtype),
        ),
        temporal_slice=slice(0, 0),
        total_steps=0,
    )


def _runtime_extra_checkpoint_binding(plan, tensors):
    """Bind declared runtime leaves and their authored owners for replay."""
    values = {
        f"{namespace}.{name}": value
        for namespace, leaves in (
            ("parameters", tensors.parameters),
            ("constants", tensors.constants),
            ("sources", plan._tensor_sources()),
        )
        for name, value in leaves.items()
    }
    # Checkpoint materializes inference operands into independent snapshots.
    # Their source owners have no version counter, so retain identity/schema
    # guards without requiring versioned source tensors for this public input.
    inference = tuple(
        (
            name,
            id(value),
            tuple(value.shape),
            value.dtype,
            value.device,
            value.requires_grad,
        )
        for name, value in sorted(values.items())
        if torch.is_inference(value)
    )
    return (
        tuple(
            (name, tuple(value.shape), value.dtype, value.device)
            for name, value in sorted(values.items())
        ),
        _tensor_binding(
            {
                name: value
                for name, value in values.items()
                if not torch.is_inference(value)
            }
        ),
        inference,
    )


def _execute_steps(step, state, inputs: RolloutInput, steps: int):
    auxiliary = None
    if isinstance(step, _StepExecution) and step.compiled_chunks is not None:
        binding = step.binding
        width = binding.compiled_chunk.steps
        for start in range(0, steps, width):
            stop = min(start + width, steps)
            kernel = step.compiled_chunks[stop - start]
            state, auxiliary = kernel._rollout_values(
                binding.parameters,
                step.prepared_values,
                state,
                _slice_inputs(inputs, start, stop),
            )
        return state, auxiliary
    for index in range(steps):
        result = step(
            state,
            _index_runtime_inputs(
                inputs.ve,
                inputs.intra,
                inputs.noise,
                inputs.stimulation,
                index,
                draws=inputs.draws,
                sampling_time=inputs.sampling_time,
            ),
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("step must return exactly (next_state, auxiliary)")
        state, auxiliary = result
    return state, auxiliary


def _execute_steps_with_callbacks(
    step,
    state,
    inputs: RolloutInput,
    steps: int,
    callbacks: FunctionalCallbacks,
    callback_state,
    *,
    retain=False,
):
    """Advance one chunk while threading pure callback carry.

    Compiled bindings execute joint model/callback chunks and concatenate their
    emissions. Eager callables update carry per step and stack emissions once.
    Both paths replay safely inside non-reentrant activation checkpointing.
    """
    auxiliary = None
    if isinstance(step, _StepExecution) and step.compiled_chunks is not None:
        binding = step.binding
        width = binding.compiled_chunk.steps
        parts = _host_callback_parts(callbacks, callback_state, []) if retain else []
        for start in range(0, steps, width):
            stop = min(start + width, steps)
            kernel = step.compiled_chunks[stop - start]
            state, auxiliary, carries, emitted = kernel._rollout_values(
                binding.parameters,
                step.prepared_values,
                state,
                _slice_inputs(inputs, start, stop),
                callback_state.carries,
            )
            callback_state = callbacks._wrap_compiled_update(
                callback_state, carries, emitted
            )
            _collect_callback_emissions(callbacks, callback_state, parts, emitted)
            del emitted
        return (
            state,
            auxiliary,
            callback_state,
            _concatenate_callback_parts(callbacks, parts, borrow_single=True),
        )
    emissions = []
    for index in range(steps):
        result = step(
            state,
            _index_runtime_inputs(
                inputs.ve,
                inputs.intra,
                inputs.noise,
                inputs.stimulation,
                index,
                draws=inputs.draws,
                sampling_time=inputs.sampling_time,
            ),
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("step must return exactly (next_state, auxiliary)")
        state, auxiliary = result
        callback_state, emitted = callbacks._update(
            callback_state,
            state,
            auxiliary,
        )
        if emitted is not None:
            emissions.append(emitted)
    emitted = callbacks._stack(emissions)
    if retain:
        parts = _host_callback_parts(callbacks, callback_state, [])
        if isinstance(parts, _CallbackCollector):
            _collect_callback_emissions(callbacks, callback_state, parts, emitted)
            emitted = parts.concatenate(borrow_single=True)
    return state, auxiliary, callback_state, emitted


def _prepare_callback_collection(
    functional,
    callbacks,
    callback_state,
    state,
    dt,
):
    """Validate one callback plan and establish its explicit carry/output."""
    if callbacks is None:
        if callback_state is not None:
            raise ValueError("callback_state requires callbacks")
        return None, None, []
    if not isinstance(callbacks, FunctionalCallbacks):
        raise TypeError(
            "callbacks must be a FunctionalCallbacks plan created by "
            "functional.make_callbacks(...)"
        )
    callbacks._validate_owner(functional)
    callbacks._validate_runtime_dt(dt)
    callback_state, parts = _setup_callback_state(callbacks, callback_state, state)
    return callbacks, callback_state, parts


def _attach_callback_results(
    auxiliary,
    callbacks,
    callback_state,
    callback_parts,
):
    if callbacks is None:
        return auxiliary
    if auxiliary is None:
        result = {}
    elif isinstance(auxiliary, Mapping):
        result = dict(auxiliary)
    else:
        raise TypeError(
            "callback-enabled steps must return a Mapping or None as auxiliary"
        )
    if "callbacks" in result:
        raise FunctionalizationError(
            "step auxiliary output uses the reserved 'callbacks' key"
        )
    result["callbacks"] = _complete_callback_results(
        callbacks, callback_state, callback_parts
    )
    return result


def _validate_returned_state(
    functional,
    state,
    expected_remainder: float,
    *,
    start_clock: torch.Tensor,
    steps: int,
    dt: float,
) -> None:
    functional._validate_state(state)
    _validate_duration_remainder(state, expected=expected_remainder)
    _validate_clock_advance(start_clock, state, steps=steps, dt=dt)


def _materialize_checkpoint_tree(value):
    """Preserve registered PyTrees and AD while making constants saveable.

    Public state and stimulus inputs also admit arbitrary key-addressed
    Mappings. Normalize only unregistered Mapping leaves to ordinary dicts;
    registered containers such as FunctionalCallbackState retain their static
    plan identity and tree structure.
    """

    inference_copies = {}

    def materialize(leaf):
        if torch.is_tensor(leaf):
            if torch.is_inference(leaf):
                if id(leaf) not in inference_copies:
                    with torch.inference_mode(False):
                        inference_copies[id(leaf)] = leaf.clone()
                return inference_copies[id(leaf)]
            return leaf
        if isinstance(leaf, Mapping):
            return {
                name: pytree.tree_map(materialize, item) for name, item in leaf.items()
            }
        return leaf

    return pytree.tree_map(materialize, value)


def _execution_spans(steps, chunklength):
    """Shared exact host spans, including a final short tail and zero work."""
    for start in range(0, steps, chunklength):
        yield start, min(start + chunklength, steps)


def _checkpoint_tensor_tree(function, operands):
    """Checkpoint explicit tensor operands, preserving genuine identity ties.

    Save each distinct tensor once and reconstruct its repeated tree positions
    from the same argument during replay. Aliased parameter leaves must not
    become independent saved-tensor objects after checkpoint unpacking.
    """
    leaves, spec = pytree.tree_flatten(_materialize_checkpoint_tree(operands))
    arguments, positions, identities = [], [], {}
    for leaf in leaves:
        if torch.is_tensor(leaf):
            index = identities.get(id(leaf))
            if index is None:
                index = len(arguments)
                identities[id(leaf)] = index
                arguments.append(leaf)
        else:
            index = len(arguments)
            arguments.append(leaf)
        positions.append(index)

    def execute(*values):
        unpacked = pytree.tree_unflatten([values[index] for index in positions], spec)
        return function(*unpacked)

    return checkpoint(execute, *arguments, use_reentrant=False)


def _execute_chunks(
    functional,
    step,
    state,
    inputs: RolloutInput,
    steps: int,
    chunklength: int,
    expected_remainder: float,
    dt: float,
    callbacks: FunctionalCallbacks | None,
    callback_state,
    callback_parts,
    extra_plan: FunctionalExtra | None = None,
    extra_tensors: StimulusTensors | None = None,
    runtime_extra_slot: bool = False,
):
    auxiliary = None
    sampling_origin = state["clock"]["t"]
    if functional.device.type == "mps" and (
        functional.intra.enabled
        or functional.extra.enabled
        or extra_plan is not None
        and extra_plan.uses_waveforms
    ):
        sampling_origin = sampling_origin.to(device="cpu")
    accepted_bank_clock = (
        _accepted_bank_clock(
            functional, inputs.draws is not None or inputs.stimulation is not None
        )
        and inputs.sampling_time is None
    )
    for start, stop in _execution_spans(steps, chunklength):
        start_clock = _capture_clock(state)
        chunk_inputs = _prepare_chunk_inputs(
            functional,
            state,
            inputs,
            start,
            stop,
            **(
                {"canonical": True} if type(step.public_step) is BoundPopulation else {}
            ),
        )
        if chunk_inputs.draws is not None and inputs.draws is None:
            step = _refresh_checkpoint_execution(step, state, source_validated=True)
        clock_dt = (
            step.prepared_values["integrator"]["dt"]
            if isinstance(step, _StepExecution) and step.prepared_values is not None
            else None
        )
        sampling_times = chunk_inputs.sampling_time
        if (
            sampling_times is None
            and not accepted_bank_clock
            and (
                functional.intra.enabled
                or functional.extra.enabled
                or extra_plan is not None
                and extra_plan.uses_waveforms
            )
        ):
            sampling_times = _sampling_time_tape(
                functional,
                sampling_origin,
                start,
                stop - start,
                dt,
                clock_dt=clock_dt,
            )
            if functional.intra.enabled or functional.extra.enabled:
                chunk_inputs = chunk_inputs._replace(sampling_time=sampling_times)
        extra_ve = _assemble_extra_chunk(
            functional,
            extra_plan,
            extra_tensors,
            state,
            start,
            stop - start,
            steps,
            dt,
            runtime_extra_slot=runtime_extra_slot,
            times=sampling_times,
            clock_dt=clock_dt,
        )
        if extra_ve is not None:
            chunk_inputs = chunk_inputs._replace(ve=extra_ve)
        if callbacks is None:
            state, auxiliary = _execute_steps(
                step,
                state,
                chunk_inputs,
                stop - start,
            )
        else:
            state, auxiliary, callback_state, emitted = _execute_steps_with_callbacks(
                step,
                state,
                chunk_inputs,
                stop - start,
                callbacks,
                callback_state,
                retain=True,
            )
            _collect_callback_emissions(
                callbacks,
                callback_state,
                callback_parts,
                emitted,
                retained=True,
                steps=stop - start,
            )
        _validate_returned_state(
            functional,
            state,
            expected_remainder,
            start_clock=start_clock,
            steps=stop - start,
            dt=dt,
        )
    if callbacks is not None:
        auxiliary = _attach_callback_results(
            auxiliary,
            callbacks,
            callback_state,
            callback_parts,
        )
    return state, auxiliary


def _execute_checkpointed_chunks(
    functional,
    execution,
    state,
    inputs,
    steps,
    chunklength,
    expected_remainder,
    dt,
    callbacks,
    callback_state,
    callback_parts,
    extra_plan,
    extra_tensors,
    runtime_extra_slot=False,
):
    """Execute a resolved host schedule with replay-time binding validation."""
    if execution.binding is not None:
        # Stimulation leaves do not contribute to physical preparation, but
        # they are still captured by the compiled transition during replay.
        # Snapshot every bound parameter once for this checkpointed invocation,
        # without adding a guard or fingerprint to ordinary run/longrun paths.
        execution = replace(
            execution,
            checkpoint_parameter_binding=_tensor_binding(execution.binding.parameters),
        )
    extra_binding = (
        _runtime_extra_checkpoint_binding(extra_plan, extra_tensors)
        if runtime_extra_slot and extra_plan is not None
        else None
    )
    full_draws = inputs.draws
    full_draws_binding = (
        None if full_draws is None else _runtime_draws_binding(full_draws)
    )
    auxiliary: Any = None
    sampling_origin = state["clock"]["t"]
    if functional.device.type == "mps" and (
        functional.intra.enabled
        or functional.extra.enabled
        or extra_plan is not None
        and extra_plan.uses_waveforms
    ):
        sampling_origin = sampling_origin.to(device="cpu")
    accepted_bank_clock = (
        _accepted_bank_clock(
            functional, inputs.draws is not None or inputs.stimulation is not None
        )
        and inputs.sampling_time is None
    )
    for start, stop in _execution_spans(steps, chunklength):
        chunk_steps = stop - start
        start_clock = _capture_clock(state)
        chunk_inputs = _prepare_chunk_inputs(
            functional,
            state,
            inputs,
            start,
            stop,
            **(
                {"canonical": True}
                if type(execution.public_step) is BoundPopulation
                else {}
            ),
        )
        source_draws = chunk_inputs.draws
        source_draws_binding = (
            None if source_draws is None else _runtime_draws_binding(source_draws)
        )
        sampling_times = chunk_inputs.sampling_time
        if (
            sampling_times is None
            and not accepted_bank_clock
            and (
                functional.intra.enabled
                or functional.extra.enabled
                or extra_plan is not None
                and extra_plan.uses_waveforms
            )
        ):
            sampling_times = _sampling_time_tape(
                functional,
                sampling_origin,
                start,
                chunk_steps,
                dt,
                clock_dt=(
                    execution.prepared_values["integrator"]["dt"]
                    if execution.prepared_values is not None
                    else None
                ),
            )
        checkpoint_inputs = _materialize_checkpoint_tree(
            (
                state,
                callback_state,
                chunk_inputs.ve,
                chunk_inputs.intra,
                chunk_inputs.noise,
                chunk_inputs.stimulation,
                chunk_inputs.draws,
                sampling_times,
                None if extra_tensors is None else extra_tensors.parameters,
                None if extra_tensors is None else extra_tensors.constants,
            )
        )

        def execute_chunk(
            state_in,
            callback_state_in,
            ve,
            intra,
            noise,
            stimulation,
            draws,
            sampling_times,
            extra_parameters,
            extra_constants,
            chunk_steps=chunk_steps,
            chunk_start=start,
            total_steps=steps,
            source_draws=source_draws,
            source_draws_binding=source_draws_binding,
        ):
            if (
                full_draws_binding is not None
                and _runtime_draws_binding(full_draws) != full_draws_binding
                or source_draws_binding is not None
                and _runtime_draws_binding(source_draws) != source_draws_binding
            ):
                raise FunctionalizationError(
                    "runtime draws changed after checkpoint forward; run a new forward before backward"
                )
            if extra_binding is not None and (
                _runtime_extra_checkpoint_binding(extra_plan, extra_tensors)
                != extra_binding
            ):
                raise FunctionalizationError(
                    "runtime extra tensors changed after checkpoint forward; "
                    "run a new forward before backward"
                )
            chunk_execution = _refresh_checkpoint_execution(execution, state_in)
            if extra_plan is not None or runtime_extra_slot:
                chunk_tensors = (
                    None
                    if extra_tensors is None
                    else StimulusTensors(
                        extra_parameters, extra_constants, extra_tensors.plan_token
                    )
                )
                ve = _assemble_extra_chunk(
                    functional,
                    extra_plan,
                    chunk_tensors,
                    state_in,
                    chunk_start,
                    chunk_steps,
                    total_steps,
                    dt,
                    runtime_extra_slot=runtime_extra_slot,
                    times=sampling_times,
                    clock_dt=(
                        chunk_execution.prepared_values["integrator"]["dt"]
                        if chunk_execution.prepared_values is not None
                        else None
                    ),
                )
            if callbacks is None:
                return _execute_steps(
                    chunk_execution,
                    state_in,
                    _RuntimeInputs(
                        ve=ve,
                        intra=intra,
                        noise=noise,
                        stimulation=stimulation,
                        draws=draws,
                        sampling_time=(
                            sampling_times
                            if functional.intra.enabled or functional.extra.enabled
                            else None
                        ),
                    ),
                    chunk_steps,
                )
            return _execute_steps_with_callbacks(
                chunk_execution,
                state_in,
                _RuntimeInputs(
                    ve=ve,
                    intra=intra,
                    noise=noise,
                    stimulation=stimulation,
                    draws=draws,
                    sampling_time=(
                        sampling_times
                        if functional.intra.enabled or functional.extra.enabled
                        else None
                    ),
                ),
                chunk_steps,
                callbacks,
                callback_state_in,
                retain=True,
            )

        checkpoint_result = _checkpoint_tensor_tree(execute_chunk, checkpoint_inputs)
        if callbacks is None:
            state, auxiliary = checkpoint_result
        else:
            state, auxiliary, callback_state, emitted = checkpoint_result
            _collect_callback_emissions(
                callbacks,
                callback_state,
                callback_parts,
                emitted,
                retained=True,
                steps=stop - start,
            )
        _validate_returned_state(
            functional,
            state,
            expected_remainder,
            start_clock=start_clock,
            steps=chunk_steps,
            dt=dt,
        )
    auxiliary = _attach_callback_results(
        auxiliary,
        callbacks,
        callback_state,
        callback_parts,
    )
    return state, auxiliary


def _population_timestep(functional, timestep):
    """Read an effective timestep while preserving nominal float32 durations."""
    primal, _tangent = torch.autograd.forward_ad.unpack_dual(timestep)
    value = float(primal.detach().to(device="cpu").to(torch.float64).item())
    nominal = functional.dt
    # Callback sample windows and physical durations use the authored Python
    # value. Retain it when it represents exactly the same prepared timestep.
    if float(torch.tensor(nominal, dtype=primal.dtype, device="cpu").item()) == value:
        return nominal
    return value


def _bound_runner_dt(functional, prepared_values):
    """Infer the prepared timestep while preserving nominal float32 durations."""
    return _population_timestep(functional, prepared_values["integrator"]["dt"])


def _run_schedule(
    functional,
    step,
    state,
    inputs,
    *,
    duration=_MISSING,
    steps=None,
    span=None,
    checkpointed=False,
    dt=None,
    extra=None,
    _runtime_extra_slot=False,
    callbacks=None,
    callback_state=None,
):
    """Validate an explicit binding and execute its canonical host schedule."""
    expected_remainder = _validate_execution(functional, step, state)
    return _run_validated_schedule(
        functional,
        step,
        state,
        inputs,
        expected_remainder=expected_remainder,
        duration=duration,
        steps=steps,
        span=span,
        checkpointed=checkpointed,
        dt=dt,
        extra=extra,
        _runtime_extra_slot=_runtime_extra_slot,
        callbacks=callbacks,
        callback_state=callback_state,
    )


def _run_source_validated(
    bound,
    state,
    *,
    inputs,
    steps,
    callbacks,
    callback_state,
    checkpoint_every=None,
    host_span_steps=None,
    extra=None,
    _runtime_extra_slot=False,
    duration=_MISSING,
    dt=None,
):
    """Schedule one qualified binding invocation after its source check.

    CompiledRunner and raw bindings use this path after qualifying their
    current invocation. It skips no state, schema, parameter, prepared-plan or
    callback checks, and carries no exemption into replay or later calls.
    """
    functional = bound.functional
    _validate_execution_context(functional, bound)
    functional._validate_state(state)
    expected_remainder = _validate_duration_remainder(state)
    return _run_validated_schedule(
        functional,
        bound,
        state,
        inputs,
        expected_remainder=expected_remainder,
        duration=duration,
        steps=steps,
        span=checkpoint_every if checkpoint_every is not None else host_span_steps,
        checkpointed=checkpoint_every is not None,
        dt=dt,
        extra=extra,
        _runtime_extra_slot=_runtime_extra_slot,
        callbacks=callbacks,
        callback_state=callback_state,
    )


def _prepare_schedule_inputs(
    functional, step, execution, inputs, *, span, extra, _runtime_extra_slot
):
    """Prepare inputs once before either horizon spelling is normalized."""
    if span is not None:
        span = _validate_chunklength(span)
    inputs = _validate_rollout_input_container(inputs)
    if _runtime_extra_slot and inputs.ve is not None:
        raise ValueError("explicit ve cannot be combined with a runtime extra slot")
    extra_plan, extra_tensors = _prepare_extra(functional, extra, inputs)
    if _runtime_extra_slot:
        # Raw containers and stimulus providers can execute authored Python
        # while binding. Requalify the source and preparation before capture.
        functional._validate_source()
        execution = _prepare_step_execution(functional, step)
    return execution, inputs, span, extra_plan, extra_tensors


def _run_validated_schedule(
    functional,
    step,
    state,
    inputs,
    *,
    expected_remainder,
    duration=_MISSING,
    steps=None,
    span=None,
    checkpointed=False,
    dt=None,
    extra=None,
    _runtime_extra_slot=False,
    callbacks=None,
    callback_state=None,
):
    """Normalize the explicit binding contract before common execution."""
    execution = _prepare_step_execution(functional, step)
    if dt is None and type(step) is BoundPopulation:
        dt = _bound_runner_dt(functional, execution.prepared_values)
    dt = _validate_runner_dt(functional, dt)
    _validate_visible_step_dt(
        functional, step, dt, prepared_values=execution.prepared_values
    )
    execution, inputs, span, extra_plan, extra_tensors = _prepare_schedule_inputs(
        functional,
        step,
        execution,
        inputs,
        span=span,
        extra=extra,
        _runtime_extra_slot=_runtime_extra_slot,
    )
    if duration is not _MISSING:
        state, steps = _stage_duration(state, duration, dt, name="duration")
        expected_remainder = _validate_duration_remainder(state)
    return _execute_schedule(
        functional,
        execution,
        state,
        inputs,
        expected_remainder=expected_remainder,
        steps=steps,
        span=span,
        checkpointed=checkpointed,
        dt=dt,
        extra_plan=extra_plan,
        extra_tensors=extra_tensors,
        _runtime_extra_slot=_runtime_extra_slot,
        callbacks=callbacks,
        callback_state=callback_state,
    )


@_host_recording
def _execute_schedule(
    functional,
    execution,
    state,
    inputs,
    *,
    expected_remainder,
    steps=None,
    span,
    checkpointed,
    dt,
    extra_plan,
    extra_tensors,
    _runtime_extra_slot,
    callbacks,
    callback_state,
    _recording_sessions=None,
):
    """Execute resolved scheduling operands without legacy horizon policies."""
    inputs, steps = _resolve_inputs(functional, inputs, steps)
    if inputs.draws is not None:
        if inputs.stimulation is not None:
            raise ValueError("draws cannot be combined with legacy stimulation packets")
        functional._runtime_draw_plan.validate(inputs.draws, steps=steps)
    else:
        if inputs.noise is not None or steps == 0:
            functional._validate_noise_draw(inputs.noise, steps=steps)
        if inputs.stimulation is not None or steps == 0:
            functional._validate_stimulation_draw(inputs.stimulation, steps=steps)
        functional._stimulation_rng_plan.validate_packets(
            inputs.noise, inputs.stimulation
        )
    chunklength = max(steps, 1) if span is None else span
    if not checkpointed:
        chunklength = _recording_span(callbacks, chunklength)
    use_checkpoint = checkpointed and torch.is_grad_enabled() and steps > 0
    if use_checkpoint:
        # Checkpoint cannot save inference tensors for backward. Preserve all
        # differentiable leaves while materializing these constant values.
        state = _materialize_checkpoint_tree(state)
    callbacks, callback_state, callback_parts = _prepare_callback_collection(
        functional, callbacks, callback_state, state, dt
    )
    if callbacks is not None:
        callback_parts = _host_callback_parts(
            callbacks, callback_state, callback_parts, storage=_recording_sessions
        )
    execution = _prepare_compiled_schedule(
        execution, steps, chunklength, callbacks=callbacks
    )
    if steps == 0:
        _validate_zero_step_extra(functional, extra_plan, extra_tensors, state, dt)
        state = functional._clone_state_tree(state)
        return state, _attach_callback_results(
            None, callbacks, callback_state, callback_parts
        )
    if use_checkpoint:
        return _execute_checkpointed_chunks(
            functional,
            execution,
            state,
            inputs,
            steps,
            chunklength,
            expected_remainder,
            dt,
            callbacks,
            callback_state,
            callback_parts,
            extra_plan,
            extra_tensors,
            _runtime_extra_slot,
        )
    return _execute_chunks(
        functional,
        execution,
        state,
        inputs,
        steps,
        chunklength,
        expected_remainder,
        dt,
        callbacks,
        callback_state,
        callback_parts,
        extra_plan,
        extra_tensors,
        _runtime_extra_slot,
    )


def _validate_network_duration_remainder(remainder):
    """Read host duration carry without staging a new Network frame."""
    if remainder.requires_grad:
        raise FunctionalizationError(
            "Network duration_remainder must not require gradients"
        )
    _, tangent = torch.autograd.forward_ad.unpack_dual(remainder)
    if tangent is not None:
        raise FunctionalizationError(
            "Network duration_remainder must not carry a forward-AD tangent"
        )
    pending = float(remainder.detach().cpu().item())
    if not math.isfinite(pending) or pending < 0:
        raise ValueError("Network duration_remainder must be finite and non-negative")
    return pending


def _network_duration_horizon(state, steps, duration, dt):
    """Resolve Network duration carry for both host execution interfaces."""
    remainder = state["clock"]["duration_remainder"]
    pending = _validate_network_duration_remainder(remainder)
    if duration is not None:
        duration = _validate_time_scalar(duration, name="duration", positive=False)
        steps, pending = _duration_step_budget(duration, dt, pending)
        state = {
            **state,
            "clock": {
                **state["clock"],
                "duration_remainder": remainder.new_tensor(pending),
            },
        }
    return state, steps


def _stage_network_duration(functional, state, duration):
    """Stage a convenience call through the canonical Network horizon rule."""
    functional._validate_source()
    functional._validate_state(state)
    return _network_duration_horizon(state, None, duration, functional.dt)


def _run_bound_network(
    bound,
    state,
    inputs,
    *,
    steps,
    duration,
    host_span_steps,
    checkpoint_every,
    dt,
    extra,
    callbacks,
    callback_state,
    draws=None,
):
    """Resolve the shared host contract over Network's existing chunk engine."""
    from ._network_bindings import _canonicalize_mapping_tree, _map_containers
    from ._network_rollout import run as run_network
    from ._runner_policy import _validate_host_call

    _validate_host_call()
    if draws is not None:
        draws = _normalize_runtime_draws(draws)
    (state, inputs), _ = _canonicalize_mapping_tree((state, inputs))
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise TypeError("Network extra must map Population names to bindings")
        extra = _map_containers(extra, dict)
        unknown = extra.keys() - bound.functional.populations.keys()
        if unknown:
            raise ValueError(
                f"extra names unknown Populations: {sorted(unknown, key=str)!r}"
            )
    functional = bound.functional
    functional._validate_source()
    functional._validate_state(state)
    runner_dt = _validate_time_scalar(
        functional.dt if dt is None else dt, name="dt", positive=True
    )
    if runner_dt != functional.dt:
        raise FunctionalizationError("runner dt must match the bound Network timestep")
    state, steps = _network_duration_horizon(state, steps, duration, runner_dt)
    options = {} if draws is None else {"draws": draws}
    return run_network(
        functional,
        bound._parameters,
        bound._constants,
        state,
        inputs,
        steps=steps,
        chunklength=(checkpoint_every or host_span_steps or 64),
        checkpoint_every=checkpoint_every,
        callbacks=callbacks,
        callback_state=callback_state,
        compiled_chunk=bound.compiled_chunk,
        _runtime_extra=extra,
        **options,
    )


def run(
    functional: (
        FunctionalPopulation
        | dendra.func.FunctionalNetwork
        | dendra.func.CompiledPopulationChunk
        | dendra.func.CompiledNetworkChunk
        | BoundPopulation
        | BoundNetwork
        | dendra.func.CompiledRunner
        | dendra.func.CompiledNetworkRunner
    ),
    step=_MISSING,
    state=_MISSING,
    inputs: Inputs | RolloutInput | None = None,
    *,
    steps: int | None = None,
    duration: float | None = None,
    checkpoint_every: int | None = None,
    host_span_steps: int | None = None,
    tstop: float | None = None,
    dt: float | None = None,
    extra=_MISSING,
    _runtime_extra_slot=False,
    callbacks: FunctionalCallbacks | None = None,
    callback_state=None,
    draws=None,
    parameters=_MISSING,
    constants=_MISSING,
    initialization=_MISSING,
):
    """Execute a compiled runner or an explicit functional boundary.

    Call ``run(program, ...)`` to start from the runner's configured initial state.
    Resume a previous run with ``run(program, continuation, ...)`` or
    ``run(program, state=continuation, ...)``. The runner supplies its configured
    timestep, callbacks, kernel width, and checkpoint policy.

    For explicit execution, use ``run(plan_or_kernel, continuation, ...)`` or
    ``run(bound, state_or_continuation, ...)``. A plan or kernel uses the
    continuation's sources for that call. A raw binding owns its sources, which
    must match a supplied continuation.

    Parameters
    ----------
    functional : compiled runner, functional plan, compiled chunk or binding
        Execution target. Runner calls without state restart their configured
        snapshot/fresh policy. Explicit plans/kernels require a Continuation.
    step, state : tensor tree or Continuation
        The second positional argument is the boundary for preferred calls;
        ``state=`` is also supported. A Continuation includes effective sources and
        callback carry. Prepared Population bindings accept explicit state/carry.
    inputs : Inputs or mapping, optional
        Deterministic time-leading voltage/current tapes. Networks use Population
        names as keys. Supplied tapes must match the selected accepted step count.
    steps, duration : int or float, optional
        Exact count or elapsed time in ms; mutually exclusive. Durations include
        fractional carry and publish its new remainder. Explicit counts leave it
        unchanged. Runners retain their default horizon when neither is supplied;
        explicit bindings can infer length from supplied tapes.
    draws : RuntimeDraws, optional
        Stochastic tape. Omission retains automatic host sampling.
    callbacks : FunctionalCallbacks, optional
        Prepared collection for explicit plans/bindings. Runners use their own
        configured collection. Results are in ``auxiliary["callbacks"]``.
    callback_state : FunctionalCallbackState, optional
        Carry for an explicit-state continuation. Incompatible with a Continuation,
        which already contains it.
    checkpoint_every, host_span_steps : int, optional
        Explicit-binding training-checkpoint or ordinary host span. They are
        mutually exclusive and independent of compiled kernel width. Runners use
        their construction-time checkpoint policy.
    parameters, constants, initialization : optional
        Runner-only call overrides. A Continuation already owns its sources;
        initialization overrides require a fresh no-state call.
    extra : stimulation specification or mapping, optional
        Call-local stimulation. Runners accept it only for declared VarInput slots;
        None supplies zero drive.
    dt : float, optional
        Explicit-binding timestep override, validated against prepared execution.
        Runners use their configured timestep.
    tstop : float, optional
        Duration in the supported four-positional compatibility form described below.

    Returns
    -------
    tuple
        ``(state, auxiliary)``. The final explicit state does not advance the
        native model. The auxiliary mapping contains callback results and,
        for runners/raw bindings/packet execution, a ``continuation`` carrying
        effective sources, state and callback phase.

    Notes
    -----
    ``run(functional, step_callable, state, inputs, tstop=...)`` remains supported.
    In that compatibility form supplied drive length takes precedence over tstop.
    Use :func:`initialize` for an explicit fresh boundary; execution from a packet
    never implicitly initializes it. Raw Population bindings own each forward's
    preparation graph and can reuse safe inference preparation. Gradients span
    all chunks and activation checkpoints. This host scheduler runs outside
    ``torch.compile`` and ``torch.func``; use combined plan tensor methods inside
    those transforms.
    """
    from ._convenience import CompiledRunner, _execute_population_runner
    from ._network_compile import CompiledNetworkRunner, _execute_network_runner

    runner_type = type(functional)
    if runner_type is CompiledRunner or runner_type is CompiledNetworkRunner:
        if step is not _MISSING and state is not _MISSING:
            raise TypeError("state was supplied more than once")
        selected_state = state if step is _MISSING else step
        if selected_state is _MISSING:
            selected_state = None
        for name, value in (
            ("checkpoint_every", checkpoint_every),
            ("host_span_steps", host_span_steps),
            ("tstop", tstop),
            ("dt", dt),
            ("callbacks", callbacks),
        ):
            if value is not None:
                raise TypeError(f"{name} cannot be supplied to run(program, ...)")
        if _runtime_extra_slot is not False:
            raise TypeError(
                "_runtime_extra_slot cannot be supplied to run(program, ...)"
            )
        options = {} if extra is _MISSING else {"extra": extra}
        execute = (
            _execute_population_runner
            if runner_type is CompiledRunner
            else _execute_network_runner
        )
        return execute(
            functional,
            selected_state,
            steps=steps,
            duration=duration,
            parameters=None if parameters is _MISSING else parameters,
            constants=None if constants is _MISSING else constants,
            inputs=inputs,
            callback_state=callback_state,
            initialization=None if initialization is _MISSING else initialization,
            draws=draws,
            **options,
        )
    if issubclass(runner_type, (CompiledRunner, CompiledNetworkRunner)):
        raise TypeError("run requires an exact CompiledRunner or CompiledNetworkRunner")
    for name, value in (
        ("parameters", parameters),
        ("constants", constants),
        ("initialization", initialization),
    ):
        if value is not _MISSING:
            raise TypeError(f"run() got an unexpected keyword argument {name!r}")
    if extra is _MISSING:
        extra = None

    from ._compiled import CompiledPopulationChunk
    from ._continuation import Continuation
    from ._network import FunctionalNetwork
    from ._network_rollout import CompiledNetworkChunk, NetworkRolloutInput

    packet_targets = (
        FunctionalPopulation,
        FunctionalNetwork,
        CompiledPopulationChunk,
        CompiledNetworkChunk,
    )
    packet = state if step is _MISSING else step
    packet_execution = (
        issubclass(runner_type, packet_targets)
        and issubclass(type(packet), Continuation)
        and not (step is not _MISSING and state is not _MISSING and callable(step))
    )
    if packet_execution or isinstance(functional, (BoundPopulation, BoundNetwork)):
        if packet_execution:
            if not any(runner_type is target for target in packet_targets):
                raise TypeError(
                    "packet execution requires an exact FunctionalPopulation, "
                    "FunctionalNetwork, CompiledPopulationChunk or CompiledNetworkChunk"
                )
            if type(packet) is not Continuation:
                raise TypeError("packet execution requires an exact Continuation")
        elif type(functional) not in (BoundPopulation, BoundNetwork):
            raise TypeError(
                "run requires an exact BoundPopulation or BoundNetwork binding"
            )
        if step is not _MISSING:
            if state is _MISSING:
                state = step
            elif inputs is None and (
                isinstance(
                    state, (Inputs, _RuntimeInputs, RolloutInput, NetworkRolloutInput)
                )
                or state is None
            ):
                inputs, state = state, step
            else:
                raise TypeError("state was supplied more than once")
        if state is _MISSING:
            raise TypeError("run requires state")
        bound = functional.bind(state) if packet_execution else functional
        if tstop is not None:
            raise ValueError(
                f"use duration instead of tstop with a {type(bound).__name__}"
            )
        if steps is not None and duration is not None:
            raise ValueError("steps and duration are mutually exclusive")
        if checkpoint_every is not None and host_span_steps is not None:
            raise ValueError(
                "checkpoint_every and host_span_steps are mutually exclusive"
            )
        for name, value in (
            ("checkpoint_every", checkpoint_every),
            ("host_span_steps", host_span_steps),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        state, callback_state = _resolve_bound_continuation(
            bound, state, callback_state, callbacks
        )
        if type(bound) is BoundNetwork:
            previous = (
                None
                if bound.compiled_chunk is None
                else bound.compiled_chunk.execution_report()
            )
            result = _run_bound_network(
                bound,
                state,
                inputs,
                steps=steps,
                duration=duration,
                host_span_steps=host_span_steps,
                checkpoint_every=checkpoint_every,
                dt=dt,
                extra=extra,
                callbacks=callbacks,
                callback_state=callback_state,
                draws=draws,
            )
            return _complete_bound_continuation(bound, result, previous)
        if draws is not None:
            inputs = _with_runtime_draws(inputs, draws)
        if bound._is_raw:
            from ._network_bindings import _canonicalize_mapping_tree

            _validate_execution_context(bound.functional, bound)
            (state, inputs), _ = _canonicalize_mapping_tree((state, inputs))
            bound.functional._validate_source()
            bound.functional._validate_parameters(bound._parameters)
            bound.functional._validate_state(state)
            previous = (
                None
                if bound.compiled_chunk is None
                else bound.compiled_chunk.execution_report()
            )
            prepared_bound = bound._prepared_binding(source_validated=True)
            result = _run_source_validated(
                prepared_bound,
                state,
                inputs=inputs,
                steps=steps,
                duration=_MISSING if duration is None else duration,
                dt=dt,
                host_span_steps=host_span_steps,
                checkpoint_every=checkpoint_every,
                extra=extra,
                _runtime_extra_slot=_runtime_extra_slot,
                callbacks=callbacks,
                callback_state=callback_state,
            )
            return _complete_bound_continuation(bound, result, previous)
        return _run_schedule(
            bound.functional,
            bound,
            state,
            inputs,
            duration=_MISSING if duration is None else duration,
            steps=steps,
            span=checkpoint_every if checkpoint_every is not None else host_span_steps,
            checkpointed=checkpoint_every is not None,
            dt=dt,
            extra=extra,
            _runtime_extra_slot=_runtime_extra_slot,
            callbacks=callbacks,
            callback_state=callback_state,
        )
    if any(
        value is not None
        for value in (steps, duration, checkpoint_every, host_span_steps)
    ):
        raise ValueError(
            "steps, duration and span options require run(bound, state, ...)"
        )
    if draws is not None:
        raise ValueError("draws requires run(bound, state, ...)")
    if step is _MISSING or state is _MISSING:
        raise TypeError("legacy run requires functional, step and state")
    from ._runner_compat import run_legacy_schedule

    return run_legacy_schedule(
        functional,
        step,
        state,
        inputs,
        duration=tstop,
        dt=dt,
        legacy_input_horizon=True,
        extra=extra,
        _runtime_extra_slot=_runtime_extra_slot,
        callbacks=callbacks,
        callback_state=callback_state,
    )


def longrun(
    functional: FunctionalPopulation,
    step: _StepCallable,
    state,
    tstop: float,
    chunklength: int,
    inputs: Inputs | RolloutInput | None = None,
    *,
    dt: float | None = None,
    extra=None,
    callbacks: FunctionalCallbacks | None = None,
    callback_state=None,
):
    """Run an explicit Population binding for a duration in host spans.

    Parameters
    ----------
    functional : FunctionalPopulation
        Lowered Population owning the binding and state schema.
    step : callable
        Exact binding returned by ``bind``, or a supported preparation/step partial.
    state : dict
        Initial explicit Population state.
    tstop : float
        Duration to advance from the supplied state, in milliseconds.
    chunklength : int
        Positive number of steps per host span, independent of compiled kernel width.
    inputs : Inputs or RolloutInput, optional
        Time-leading deterministic tapes and optional draw packets. Their length
        must match the resolved horizon.
    dt : float, optional
        Timestep in milliseconds; defaults to the plan's nominal timestep.
    extra : object, optional
        Extracellular field/Waveform bindings evaluated over the execution spans.
    callbacks : FunctionalCallbacks, optional
        Prepared callback collection for this Population.
    callback_state : FunctionalCallbackState, optional
        Matching carry to resume without duplicating the initial sample.

    Returns
    -------
    tuple
        ``(state, outputs)``: final explicit state and transition outputs, including
        callback results when configured. Gradients span all host spans.

    See Also
    --------
    run : Shared interface using ``run(bound, state, duration=...,
        host_span_steps=...)``.
    """
    from ._runner_compat import run_legacy_schedule

    return run_legacy_schedule(
        functional,
        step,
        state,
        inputs,
        duration=tstop,
        span=_validate_chunklength(chunklength),
        dt=dt,
        extra=extra,
        callbacks=callbacks,
        callback_state=callback_state,
    )


def longrun_checkpointed(
    functional: FunctionalPopulation,
    step: _StepCallable,
    state,
    tstop: float,
    chunklength: int,
    inputs: Inputs | RolloutInput | None = None,
    *,
    dt: float | None = None,
    extra=None,
    callbacks: FunctionalCallbacks | None = None,
    callback_state=None,
):
    """Run an explicit Population binding with activation checkpointing.

    Accepts the same arguments and returns as :func:`longrun`. ``tstop`` is the
    relative duration to advance; ``chunklength`` sets the number of steps per
    checkpoint span. Non-reentrant checkpointing retains gradients through the
    full simulation while replaying spans during backward.

    Steps and callbacks must be pure, deterministic and replay-safe. Replay
    reuses and validates prepared inputs; automatic sampling occurs outside the
    replayed transition. With gradients disabled, execution uses ordinary host
    spans. The shared spelling is
    ``run(bound, state, duration=..., checkpoint_every=...)``.
    """
    from ._runner_compat import run_legacy_schedule

    return run_legacy_schedule(
        functional,
        step,
        state,
        inputs,
        duration=tstop,
        span=_validate_chunklength(chunklength),
        checkpointed=True,
        dt=dt,
        extra=extra,
        callbacks=callbacks,
        callback_state=callback_state,
    )


__all__ = ["longrun", "longrun_checkpointed", "run"]
