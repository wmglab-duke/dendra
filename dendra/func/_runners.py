"""Host-side Python schedulers for explicit-state functional transitions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from dendra.models.core import _duration_step_budget, _validate_time_scalar

from ._binding import BoundPopulation
from ._callbacks import FunctionalCallbackResults, FunctionalCallbacks
from ._population import FunctionalPopulation
from ._stimuli import FunctionalExtra, StimulusTensors
from ._types import FunctionalizationError, RolloutInput, StepInput

_StepCallable = Callable[[Mapping[str, object], StepInput], tuple[object, object]]
_MISSING = object()


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

    def __call__(self, state, inputs: StepInput):
        if self.binding is None:
            return self.public_step(state, inputs)
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
    """Recognize only the ordinary, semantics-preserving public partial form.

    Wrapped callables and partial subclasses remain generic: bypassing an
    adapter whose invocation semantics we do not own would be unsafe. Exact
    :class:`CompiledPopulationChunk` instances are also owned and
    expose the same private tensor-only boundary. An explicit ``extra=None``
    is equivalent to the functional-step default; non-empty functional
    ``extra`` remains on the public path so its validation is unchanged.
    """
    if type(step) is BoundPopulation:
        if step.functional is not functional:
            return None
        return _OwnedStepBinding(
            functional, step._parameters, step.prepared, step.compiled_chunk
        )
    if type(step) is not partial:
        return None
    candidate = step.func
    is_functional_step = (
        getattr(candidate, "__self__", None) is functional
        and getattr(candidate, "__func__", None) is FunctionalPopulation.step
    )
    if len(step.args) != 2:
        return None
    keywords = step.keywords or {}
    if is_functional_step:
        if set(keywords) - {"extra"} or keywords.get("extra") is not None:
            return None
        return _OwnedStepBinding(functional, step.args[0], step.args[1])

    # Import lazily to preserve the existing population/compiled module split.
    from ._compiled import CompiledPopulationChunk

    if (
        not keywords
        and type(candidate) is CompiledPopulationChunk
        and candidate._functional is functional
    ):
        return _OwnedStepBinding(
            functional,
            step.args[0],
            step.args[1],
            compiled_chunk=candidate,
        )
    return None


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


def _refresh_checkpoint_execution(execution: _StepExecution, state) -> _StepExecution:
    """Revalidate captured bindings once for each checkpoint invocation.

    Non-reentrant checkpointing invokes a chunk again during backward, after
    user code has had an opportunity to mutate captured tensors or the source
    model.  Rechecking at that replay boundary preserves the public freshness
    contract without putting Python validation back inside the timestep loop.
    """
    binding = execution.binding
    if binding is None:
        return execution
    functional = binding.functional
    functional._validate_source()
    functional._validate_parameters(binding.parameters)
    functional._validate_state(state)
    prepared_values = functional._validate_prepared(
        binding.prepared,
        binding.parameters,
    )
    return replace(execution, prepared_values=prepared_values)


def _inspect_step_callable(step):
    """Unwrap public callable adapters while retaining bound positional args."""
    from ._compiled import CompiledPopulationChunk

    candidate = step
    bound_args = ()
    seen = set()
    while id(candidate) not in seen:
        seen.add(id(candidate))
        if isinstance(candidate, partial):
            # Inner partial arguments precede outer partial arguments in the
            # eventual invocation.
            bound_args = (*candidate.args, *bound_args)
            candidate = candidate.func
            continue
        original = getattr(candidate, "_torchdynamo_orig_callable", None)
        if original is None:
            original = getattr(candidate, "__wrapped__", None)
        if original is not None:
            candidate = original
            continue
        owner = getattr(candidate, "__self__", None)
        if (
            isinstance(owner, BoundPopulation)
            and getattr(candidate, "__func__", None) is BoundPopulation.__call__
        ):
            candidate = owner
            continue
        if (
            isinstance(owner, CompiledPopulationChunk)
            and getattr(candidate, "__func__", None) is CompiledPopulationChunk.__call__
        ):
            candidate = owner
            continue
        break
    return candidate, bound_args


def _validate_step_plan(functional, step) -> None:
    """Reject plan mismatches that are visible through public callables."""
    candidate, _bound_args = _inspect_step_callable(step)

    owner = getattr(candidate, "__self__", None)
    if isinstance(owner, FunctionalPopulation) and owner is not functional:
        raise FunctionalizationError(
            "step belongs to a different FunctionalPopulation plan"
        )

    # Import lazily to preserve the existing population/compiled module split.
    from ._compiled import CompiledPopulationChunk

    if isinstance(candidate, BoundPopulation):
        if candidate.functional is not functional:
            raise FunctionalizationError(
                "step belongs to a different FunctionalPopulation plan"
            )
        if candidate.steps != 1 and _owned_step_binding(functional, step) is None:
            raise FunctionalizationError(
                "multi-step compiled chunks require the exact binding returned "
                "by chunk.bind(); other adapters must expose a one-step callable"
            )
    if isinstance(candidate, CompiledPopulationChunk):
        if candidate._functional is not functional:
            raise FunctionalizationError(
                "step belongs to a different FunctionalPopulation plan"
            )
        if candidate.steps != 1 and _owned_step_binding(functional, step) is None:
            raise FunctionalizationError(
                "multi-step compiled chunks require the exact binding "
                "functools.partial(chunk, parameters, prepared); other adapters "
                "must expose a one-step callable"
            )


def _validate_visible_step_dt(
    functional,
    step,
    dt: float,
    *,
    prepared_values: Mapping[str, object] | None = None,
) -> None:
    """Compare runner dt exactly when a public prepared binding is visible."""
    candidate, bound_args = _inspect_step_callable(step)
    owner = getattr(candidate, "__self__", None)

    # Import lazily to preserve the existing population/compiled module split.
    from ._compiled import CompiledPopulationChunk

    if isinstance(candidate, BoundPopulation) and candidate.functional is functional:
        parameters, prepared = candidate._parameters, candidate.prepared
    elif (
        owner is functional
        and getattr(candidate, "__func__", None) is FunctionalPopulation.step
        and len(bound_args) >= 2
    ):
        parameters, prepared = bound_args[:2]
    elif (
        isinstance(candidate, CompiledPopulationChunk)
        and candidate._functional is functional
        and len(bound_args) >= 2
    ):
        parameters, prepared = bound_args[:2]
    else:
        return

    if prepared_values is None:
        prepared_values = functional._validate_prepared(prepared, parameters)
    prepared_dt, _tangent = torch.autograd.forward_ad.unpack_dual(
        prepared_values["integrator"]["dt"]
    )
    prepared_value = float(
        prepared_dt.detach().to(device="cpu", dtype=torch.float64).item()
    )
    runner_value = float(torch.tensor(dt, dtype=prepared_dt.dtype).item())
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
    functional._validate_source()
    functional._validate_state(state)
    return _validate_duration_remainder(state)


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
    clocks = torch.stack((start, end.detach())).to(device="cpu", dtype=torch.float64)
    start_value, end_value = clocks.tolist()
    observed_advance = end_value - start_value
    runner_dt = float(torch.tensor(dt, dtype=start.dtype).item())
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
    magnitude = torch.tensor(largest_clock, dtype=start.dtype)
    spacing = float(
        (
            torch.nextafter(magnitude, torch.full_like(magnitude, float("inf")))
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


def _validate_rollout_input_container(inputs: RolloutInput | None):
    if inputs is None:
        return RolloutInput()
    if not isinstance(inputs, RolloutInput):
        raise TypeError("inputs must be a RolloutInput or None")
    return inputs


def _resolve_inputs(functional, inputs: RolloutInput, steps: int | None):
    ve, intra, resolved_steps = functional._resolve_rollout_inputs(inputs, steps)
    return RolloutInput(ve=ve, intra=intra), resolved_steps


def _slice_inputs(inputs: RolloutInput, start: int, stop: int) -> RolloutInput:
    return RolloutInput(
        ve=None if inputs.ve is None else inputs.ve[start:stop],
        intra=None if inputs.intra is None else inputs.intra[start:stop],
    )


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
):
    if plan is None:
        return None
    dt_tensor = torch.tensor(dt, device=functional.device, dtype=functional.dtype)
    times = functional._stimulation_times(state, steps, dt_tensor)
    if plan.uses_waveforms:
        return plan.assemble_tensors(tensors, times)
    return plan.assemble(
        tensors.parameters,
        tensors.constants,
        times,
        temporal_slice=slice(start, start + steps),
        total_steps=total_steps,
    )


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
            StepInput(
                ve=None if inputs.ve is None else inputs.ve[index],
                intra=None if inputs.intra is None else inputs.intra[index],
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
        parts = []
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
            if callbacks._has_emissions(emitted):
                parts.append(emitted)
        return state, auxiliary, callback_state, callbacks._concatenate(parts)
    emissions = []
    for index in range(steps):
        result = step(
            state,
            StepInput(
                ve=None if inputs.ve is None else inputs.ve[index],
                intra=None if inputs.intra is None else inputs.intra[index],
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
    return (
        state,
        auxiliary,
        callback_state,
        callbacks._stack(emissions),
    )


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
    if callback_state is None:
        callback_state, initial_emission = callbacks._initialize(state)
        initial_part = callbacks._stack([initial_emission])
        parts = [initial_part] if callbacks._has_emissions(initial_part) else []
    else:
        callback_state = callbacks._validate_state(callback_state)
        parts = []
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
    stacked = callbacks._concatenate(callback_parts)
    callback_state = callbacks._capture_emission_schemas(
        callback_state,
        stacked,
    )
    outputs = callbacks._finalize(callback_state, stacked)
    callback_state = callbacks._materialize_state(callback_state)
    result["callbacks"] = FunctionalCallbackResults(
        state=callback_state,
        outputs=outputs,
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
    if torch.is_tensor(value):
        return value.clone() if torch.is_inference(value) else value
    if isinstance(value, Mapping):
        return {
            name: _materialize_checkpoint_tree(item) for name, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize_checkpoint_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_checkpoint_tree(item) for item in value)
    return value


def _execute_chunks(
    functional,
    step,
    state,
    inputs: RolloutInput,
    steps: int,
    chunklength: int,
    expected_remainder: float,
    dt: float,
    extra_plan: FunctionalExtra | None = None,
    extra_tensors: StimulusTensors | None = None,
):
    auxiliary = None
    for start in range(0, steps, chunklength):
        stop = min(start + chunklength, steps)
        start_clock = _capture_clock(state)
        chunk_inputs = _slice_inputs(inputs, start, stop)
        extra_ve = _assemble_extra_chunk(
            functional,
            extra_plan,
            extra_tensors,
            state,
            start,
            stop - start,
            steps,
            dt,
        )
        if extra_ve is not None:
            chunk_inputs = RolloutInput(ve=extra_ve, intra=chunk_inputs.intra)
        state, auxiliary = _execute_steps(
            step,
            state,
            chunk_inputs,
            stop - start,
        )
        _validate_returned_state(
            functional,
            state,
            expected_remainder,
            start_clock=start_clock,
            steps=stop - start,
            dt=dt,
        )
    return state, auxiliary


def _execute_chunks_with_callbacks(
    functional,
    step,
    state,
    inputs: RolloutInput,
    steps: int,
    chunklength: int,
    expected_remainder: float,
    dt: float,
    callbacks: FunctionalCallbacks,
    callback_state,
    callback_parts,
    extra_plan: FunctionalExtra | None = None,
    extra_tensors: StimulusTensors | None = None,
):
    auxiliary = None
    for start in range(0, steps, chunklength):
        stop = min(start + chunklength, steps)
        start_clock = _capture_clock(state)
        chunk_inputs = _slice_inputs(inputs, start, stop)
        extra_ve = _assemble_extra_chunk(
            functional,
            extra_plan,
            extra_tensors,
            state,
            start,
            stop - start,
            steps,
            dt,
        )
        if extra_ve is not None:
            chunk_inputs = RolloutInput(ve=extra_ve, intra=chunk_inputs.intra)
        state, auxiliary, callback_state, emitted = _execute_steps_with_callbacks(
            step,
            state,
            chunk_inputs,
            stop - start,
            callbacks,
            callback_state,
        )
        if callbacks._has_emissions(emitted):
            callback_parts.append(emitted)
        _validate_returned_state(
            functional,
            state,
            expected_remainder,
            start_clock=start_clock,
            steps=stop - start,
            dt=dt,
        )
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
):
    """Execute a resolved host schedule with replay-time binding validation."""
    auxiliary: Any = None
    for start in range(0, steps, chunklength):
        stop = min(start + chunklength, steps)
        chunk_steps = stop - start
        start_clock = _capture_clock(state)
        chunk_inputs = _slice_inputs(inputs, start, stop)
        chunk_inputs = RolloutInput(
            ve=(
                chunk_inputs.ve.clone()
                if chunk_inputs.ve is not None and torch.is_inference(chunk_inputs.ve)
                else chunk_inputs.ve
            ),
            intra=(
                chunk_inputs.intra.clone()
                if chunk_inputs.intra is not None
                and torch.is_inference(chunk_inputs.intra)
                else chunk_inputs.intra
            ),
        )

        def execute_chunk(
            state_in,
            callback_state_in,
            ve,
            intra,
            extra_parameters,
            extra_constants,
            chunk_steps=chunk_steps,
            chunk_start=start,
            chunk_stop=stop,
            total_steps=steps,
        ):
            if extra_plan is not None:
                times = functional._stimulation_times(
                    state_in,
                    chunk_steps,
                    torch.tensor(
                        dt,
                        device=functional.device,
                        dtype=functional.dtype,
                    ),
                )
                if extra_plan.uses_waveforms:
                    ve = extra_plan.assemble(
                        extra_parameters,
                        extra_constants,
                        times,
                    )
                else:
                    ve = extra_plan.assemble(
                        extra_parameters,
                        extra_constants,
                        times,
                        temporal_slice=slice(chunk_start, chunk_stop),
                        total_steps=total_steps,
                    )
            chunk_execution = _refresh_checkpoint_execution(execution, state_in)
            if callbacks is None:
                return _execute_steps(
                    chunk_execution,
                    state_in,
                    RolloutInput(ve=ve, intra=intra),
                    chunk_steps,
                )
            return _execute_steps_with_callbacks(
                chunk_execution,
                state_in,
                RolloutInput(ve=ve, intra=intra),
                chunk_steps,
                callbacks,
                callback_state_in,
            )

        checkpoint_result = checkpoint(
            execute_chunk,
            state,
            callback_state,
            chunk_inputs.ve,
            chunk_inputs.intra,
            None if extra_tensors is None else extra_tensors.parameters,
            None if extra_tensors is None else extra_tensors.constants,
            use_reentrant=False,
        )
        if callbacks is None:
            state, auxiliary = checkpoint_result
        else:
            state, auxiliary, callback_state, emitted = checkpoint_result
            if callbacks._has_emissions(emitted):
                callback_parts.append(emitted)
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


def _bound_runner_dt(functional, prepared_values):
    """Infer the prepared timestep while preserving nominal float32 durations."""
    primal, _tangent = torch.autograd.forward_ad.unpack_dual(
        prepared_values["integrator"]["dt"]
    )
    value = float(primal.detach().to(device="cpu", dtype=torch.float64).item())
    nominal = functional.dt
    # Callback sample windows and physical durations use the authored Python
    # value. Retain it when it represents exactly the same prepared timestep.
    if float(torch.tensor(nominal, dtype=primal.dtype).item()) == value:
        return nominal
    return value


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
    infer_bound_dt=False,
    legacy_input_horizon=False,
    duration_name="duration",
    extra=None,
    callbacks=None,
    callback_state=None,
):
    """Resolve one horizon and use the shared eager/checkpoint execution engine."""
    expected_remainder = _validate_execution(functional, step, state)
    execution = _prepare_step_execution(functional, step)
    if infer_bound_dt and dt is None:
        dt = _bound_runner_dt(functional, execution.prepared_values)
    dt = _validate_runner_dt(functional, dt)
    _validate_visible_step_dt(
        functional, step, dt, prepared_values=execution.prepared_values
    )
    if span is not None:
        span = _validate_chunklength(span)
    inputs = _validate_rollout_input_container(inputs)
    extra_plan, extra_tensors = _prepare_extra(functional, extra, inputs)
    has_drives = inputs.ve is not None or inputs.intra is not None
    if legacy_input_horizon:
        if has_drives:
            duration = _MISSING
        elif duration is None:
            raise ValueError("tstop must be provided when inputs contain no drives")
    if duration is not _MISSING:
        state, steps = _stage_duration(state, duration, dt, name=duration_name)
        expected_remainder = _validate_duration_remainder(state)
    inputs, steps = _resolve_inputs(functional, inputs, steps)
    chunklength = max(steps, 1) if span is None else span
    use_checkpoint = checkpointed and torch.is_grad_enabled() and steps > 0
    if use_checkpoint:
        # Checkpoint cannot save inference tensors for backward. Preserve all
        # differentiable leaves while materializing these constant values.
        state = _materialize_checkpoint_tree(state)
    callbacks, callback_state, callback_parts = _prepare_callback_collection(
        functional, callbacks, callback_state, state, dt
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
        )
    if callbacks is not None:
        return _execute_chunks_with_callbacks(
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
        extra_plan,
        extra_tensors,
    )


def run(
    functional: FunctionalPopulation | BoundPopulation,
    step=_MISSING,
    state=_MISSING,
    inputs: RolloutInput | None = None,
    *,
    steps: int | None = None,
    duration: float | None = None,
    checkpoint_every: int | None = None,
    host_span_steps: int | None = None,
    tstop: float | None = None,
    dt: float | None = None,
    extra=None,
    callbacks: FunctionalCallbacks | None = None,
    callback_state=None,
):
    """Schedule explicit functional execution, optionally with checkpointing.

    Preferred form: ``run(bound, state, inputs=..., steps=N)`` or
    ``run(bound, state, duration=..., checkpoint_every=H)``. Create ``bound``
    with ``fmodel.bind(parameters, prepared)`` or ``kernel.bind(...)``. Inputs
    alone infer the horizon; an explicit step count or duration must match
    their time axis. ``steps`` and ``duration`` are mutually exclusive.

    ``duration`` is an amount to advance, including the state's retained
    fractional remainder. Explicit or input-inferred steps leave that remainder
    unchanged. The default timestep comes from the validated preparation.
    ``checkpoint_every`` sets the number of steps between activation checkpoints;
    ``host_span_steps`` instead groups an ordinary run. These mutually exclusive
    options are independent of the compiled kernel width. Shorter kernel tails
    are cached, and gradients span the complete run.

    The legacy form ``run(functional, step, state, inputs, tstop=...)`` remains
    supported. Its explicit drive horizon takes precedence over ``tstop`` and
    its default timestep is the functional plan's nominal timestep. New horizon
    and span options require the preferred binding form. Legacy steps may be
    generic one-step callables, explicit bindings, or exact partial bindings.

    Callbacks observe every accepted state, including within compiled chunks.
    Results are returned under ``auxiliary["callbacks"]``; pass their explicit
    ``state`` as ``callback_state`` when resuming to avoid duplicate frames.
    This host scheduler stays outside torch.compile and torch.func boundaries.
    """
    if isinstance(functional, BoundPopulation):
        if type(functional) is not BoundPopulation:
            raise TypeError("run requires an exact BoundPopulation binding")
        bound = functional
        if step is not _MISSING:
            if state is _MISSING:
                state = step
            elif inputs is None and (isinstance(state, RolloutInput) or state is None):
                inputs, state = state, step
            else:
                raise TypeError("state was supplied more than once")
        if state is _MISSING:
            raise TypeError("run requires state")
        if tstop is not None:
            raise ValueError("use duration instead of tstop with a BoundPopulation")
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
            infer_bound_dt=True,
            extra=extra,
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
    if step is _MISSING or state is _MISSING:
        raise TypeError("legacy run requires functional, step and state")
    return _run_schedule(
        functional,
        step,
        state,
        inputs,
        duration=tstop,
        dt=dt,
        legacy_input_horizon=True,
        duration_name="tstop",
        extra=extra,
        callbacks=callbacks,
        callback_state=callback_state,
    )


def longrun(
    functional: FunctionalPopulation,
    step: _StepCallable,
    state,
    tstop: float,
    chunklength: int,
    inputs: RolloutInput | None = None,
    *,
    dt: float | None = None,
    extra=None,
    callbacks: FunctionalCallbacks | None = None,
    callback_state=None,
):
    """Run a duration in host spans, preserving the legacy calling convention.

    ``tstop`` is a duration to advance; supplied drives must match the resolved
    number of steps. ``chunklength`` sets host grouping independently of compiled
    kernel width. Exact bindings from ``bind`` and existing partial forms are
    accepted. The default timestep is the functional plan's nominal timestep.
    New code can use ``run(bound, state, duration=..., host_span_steps=...)``.
    """
    return _run_schedule(
        functional,
        step,
        state,
        inputs,
        duration=tstop,
        span=_validate_chunklength(chunklength),
        dt=dt,
        duration_name="tstop",
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
    inputs: RolloutInput | None = None,
    *,
    dt: float | None = None,
    extra=None,
    callbacks: FunctionalCallbacks | None = None,
    callback_state=None,
):
    """Run a duration with non-reentrant activation checkpointing.

    This preserves the legacy horizon, timestep and chunklength conventions.
    New code can use ``run(bound, state, duration=..., checkpoint_every=...)``.
    Steps and callbacks must be pure, deterministic and replay-safe. Gradients
    span the full simulation; replay reuses and revalidates prepared inputs.
    With gradients disabled, ordinary execution uses the same host spans.
    """
    return _run_schedule(
        functional,
        step,
        state,
        inputs,
        duration=tstop,
        span=_validate_chunklength(chunklength),
        checkpointed=True,
        dt=dt,
        duration_name="tstop",
        extra=extra,
        callbacks=callbacks,
        callback_state=callback_state,
    )


__all__ = ["longrun", "longrun_checkpointed", "run"]
