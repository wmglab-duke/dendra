"""Compatibility inference for legacy functional scheduling callables.

Canonical host scheduling consumes explicit bindings. These helpers only
interpret the existing callable-based entry points, retaining their ownership,
wrapper, and timestep rules at the compatibility boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING

import torch

from ._binding import BoundPopulation
from ._population import FunctionalPopulation
from ._types import FunctionalizationError

if TYPE_CHECKING:
    from ._runners import _OwnedStepBinding


def legacy_owned_step_binding(functional, step) -> _OwnedStepBinding | None:
    """Recognize only the ordinary, semantics-preserving public partial form.

    Wrapped callables and partial subclasses remain generic: bypassing an
    adapter whose invocation semantics we do not own would be unsafe. Exact
    :class:`CompiledPopulationChunk` instances are also owned and
    expose the same private tensor-only boundary. An explicit ``extra=None``
    is equivalent to the functional-step default; non-empty functional
    ``extra`` remains on the public path so its validation is unchanged.
    """
    from ._runners import _OwnedStepBinding

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
        if candidate.steps != 1 and legacy_owned_step_binding(functional, step) is None:
            raise FunctionalizationError(
                "multi-step compiled chunks require the exact binding returned "
                "by chunk.bind(); other adapters must expose a one-step callable"
            )
    if isinstance(candidate, CompiledPopulationChunk):
        if candidate._functional is not functional:
            raise FunctionalizationError(
                "step belongs to a different FunctionalPopulation plan"
            )
        if candidate.steps != 1 and legacy_owned_step_binding(functional, step) is None:
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
        if candidate._is_raw and prepared_values is None:
            # A visible adapter retains its own invocation semantics. Prepare
            # only this clock check rather than replacing the callable.
            prepared = candidate._prepared_binding().prepared
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


def validate_legacy_execution(functional, step, state):
    """Retain legacy callable inspection before host and source validation."""
    from . import _runners as engine

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
    return engine._validate_duration_remainder(state)


def run_legacy_schedule(
    functional,
    step,
    state,
    inputs,
    *,
    duration,
    span=None,
    checkpointed=False,
    dt=None,
    legacy_input_horizon=False,
    extra=None,
    _runtime_extra_slot=False,
    callbacks=None,
    callback_state=None,
):
    """Translate legacy callable, timestep, and horizon rules before execution.

    Legacy ``run`` lets supplied drives replace ``tstop``; the duration-based
    wrappers instead require those drives to cover their duration. Both use
    the functional plan's nominal timestep unless ``dt`` is explicit. Resolve
    these compatibility policies here so the execution engine receives a step
    count or an input-inferred horizon and no legacy policy flags.
    """
    from . import _runners as engine

    expected_remainder = validate_legacy_execution(functional, step, state)
    if type(step) is BoundPopulation and step._is_raw:
        step = step._prepared_binding()
    execution = engine._prepare_step_execution(functional, step)
    dt = engine._validate_runner_dt(functional, dt)
    _validate_visible_step_dt(
        functional, step, dt, prepared_values=execution.prepared_values
    )
    execution, inputs, span, extra_plan, extra_tensors = (
        engine._prepare_schedule_inputs(
            functional,
            step,
            execution,
            inputs,
            span=span,
            extra=extra,
            _runtime_extra_slot=_runtime_extra_slot,
        )
    )
    has_drives = (
        inputs.ve is not None
        or inputs.intra is not None
        or inputs.draws is not None
        or inputs.noise is not None
        or inputs.stimulation is not None
    )
    if legacy_input_horizon:
        if has_drives:
            duration = engine._MISSING
        elif duration is None:
            raise ValueError("tstop must be provided when inputs contain no drives")
    steps = None
    if duration is not engine._MISSING:
        state, steps = engine._stage_duration(state, duration, dt, name="tstop")
        expected_remainder = engine._validate_duration_remainder(state)
    return engine._execute_schedule(
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
