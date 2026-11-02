"""Read-only step budgeting and runtime sampling over explicit boundaries."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from dendra.models._execution_horizons import execution_call_horizon

from ._continuation import Continuation
from ._continuation_checkpoints import _cold_boundary
from ._types import FunctionalizationError


def _native_integrator(population):
    """Require a known scheduling implementation before consulting its spec."""
    import inspect

    from ._population import (
        _FUNCTIONAL_INTEGRATOR_HOOKS,
        _STANDARD_INTEGRATOR_FRESH_INITIALIZATION,
        _resolve_functional_integrator_spec,
    )

    integrator = population.integrator
    implementation = next(
        (
            candidate
            for candidate in _STANDARD_INTEGRATOR_FRESH_INITIALIZATION
            if type(integrator) is candidate
            or (
                type(integrator).__module__ == "dendra.models.integrators"
                and type(integrator).__qualname__ == candidate.__qualname__
                and type(integrator).__bases__ == (candidate,)
                and set(vars(type(integrator))) <= {"__module__", "__init__", "__doc__"}
            )
        ),
        None,
    )
    metadata = {
        "_PREPARED_WORKSPACE_SCHEMA",
        "_FUNCTIONAL_OPERATOR_KIND",
        "_FUNCTIONAL_STATE_NAMES",
        "_FUNCTIONAL_CRITICAL_METHODS",
        *_FUNCTIONAL_INTEGRATOR_HOOKS,
        "_call_kernel",
        "_kernel",
        "_sample_runtime_noises",
        "_advance_pre_current",
        "_advance_post_current",
    }
    if implementation is None or metadata.intersection(
        object.__getattribute__(integrator, "__dict__")
    ):
        raise FunctionalizationError(
            "Runtime draw sampling requires a canonical integrator implementation"
        )
    operator = inspect.getattr_static(implementation, "_FUNCTIONAL_OPERATOR_KIND")
    if _resolve_functional_integrator_spec(integrator, operator) is None:
        raise FunctionalizationError(
            "Runtime draw sampling requires canonical integrator scheduling hooks"
        )


def _native_timestep(model, dt, *, sampling=True):
    from dendra.models import Network
    from dendra.models.backend import Backend
    from dendra.models.core import _validate_time_scalar

    action = "sampling runtime draws" if sampling else "resolving steps"
    if isinstance(model, Network):
        from ._network import _NETWORK_SUBCLASS_METHODS, _NETWORK_SUBCLASS_SLOTS
        from ._population import (
            _subclass_contract_failures,
            _subclass_slot_contract_failures,
        )

        failures = _subclass_contract_failures(
            model, Network, _NETWORK_SUBCLASS_METHODS
        )
        failures.extend(
            _subclass_slot_contract_failures(model, Network, _NETWORK_SUBCLASS_SLOTS)
        )
        if failures:
            raise FunctionalizationError("; ".join(failures))
        from dendra.models.networks.net import step as native_step

        if any(
            getattr(model, name) is not native_step
            for name in ("_step", "_step_train", "_step_eval")
        ):
            failures.append("Network requires its canonical timestep phases")
        if failures:
            raise FunctionalizationError("; ".join(failures))
        if not model.built or model.dt is None or not model.populations:
            raise FunctionalizationError(f"Initialize the Network before {action}")
        populations = tuple(model.populations.values())
    else:
        populations = (model,)
    from ._population import _population_subclass_failures

    failures = [
        failure
        for population in populations
        for failure in _population_subclass_failures(population)
    ]
    if failures:
        raise FunctionalizationError("; ".join(failures))
    if any(
        not population.initialized or population.integrator is None
        for population in populations
    ):
        raise FunctionalizationError(f"Initialize every Population before {action}")
    if sampling:
        for population in populations:
            _native_integrator(population)
    if isinstance(model, Network):
        model._require_mode_runtime_ready()
        selected = _validate_time_scalar(model.dt, name="dt", positive=True)
        if (
            dt is not None
            and _validate_time_scalar(dt, name="dt", positive=True) != selected
        ):
            raise ValueError("dt must match the initialized Network timestep")
        if sampling:
            devices = {population.device() for population in populations}
            if len(devices) != 1 or (
                model.netstim is not None and model.netstim.device() not in devices
            ):
                raise FunctionalizationError(
                    "Runtime draw sampling currently requires a single simulation device"
                )
    else:
        selected = _validate_time_scalar(
            Backend.dt if dt is None else dt, name="dt", positive=True
        )
    return selected


def _functional_plan(target, *, operation="sample_runtime_draws"):
    from ._binding import BoundPopulation
    from ._compiled import CompiledPopulationChunk
    from ._convenience import CompiledRunner
    from ._network import FunctionalNetwork
    from ._network_binding import BoundNetwork
    from ._network_compile import CompiledNetworkRunner
    from ._network_rollout import CompiledNetworkChunk
    from ._population import FunctionalPopulation

    if type(target) in (FunctionalPopulation, FunctionalNetwork):
        return target
    if type(target) in (
        BoundPopulation,
        BoundNetwork,
        CompiledPopulationChunk,
        CompiledNetworkChunk,
        CompiledRunner,
        CompiledNetworkRunner,
    ):
        return target._functional
    raise TypeError(
        f"{operation} requires a native Population/Network, a functional "
        "plan, binding, compiled chunk or dn.compile runner"
    )


def _functional_duration_timestep(target, functional, continuation):
    """Read execution-owned timestep metadata without binding or preparing."""
    from ._binding import BoundPopulation
    from ._convenience import CompiledRunner
    from ._network import FunctionalNetwork
    from ._runners import _bound_runner_dt, _population_timestep

    if type(functional) is FunctionalNetwork:
        return functional.dt
    if type(target) is BoundPopulation:
        if target._constants is None:
            values = functional._validate_prepared(target._prepared, target._parameters)
            return _bound_runner_dt(functional, values)
        constants = target._constants
    elif continuation is not None:
        constants = continuation.constants
    elif type(target) is CompiledRunner:
        constants = target._constants
    else:
        return functional.dt
    functional._validate_constants(constants)
    return _population_timestep(functional, constants["dt"])


def _native_duration_steps(
    target, duration, selected_dt, *, operation="Duration draw sampling"
):
    """Read the canonical native budget without invoking authored overrides."""
    import inspect

    from dendra.models import Network, Population

    owner = Network if isinstance(target, Network) else Population
    if inspect.getattr_static(target, "_duration_budget") is not owner._duration_budget:
        raise FunctionalizationError(
            f"{operation} requires canonical duration budgeting"
        )
    steps, _ = owner._duration_budget(target, duration, selected_dt)
    return steps


def _functional_steps(target, functional, state, *, steps, duration, dt):
    """Share ordered state/metadata qualification and read-only duration math."""
    if dt is not None and duration is None:
        from dendra.models.core import _validate_time_scalar

        if _validate_time_scalar(dt, name="dt", positive=True) != functional.dt:
            raise ValueError("dt must match the functional plan timestep")
    continuation = state if isinstance(state, Continuation) else None
    state = continuation.state if continuation is not None else state
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping or Continuation")
    from ._network_bindings import _canonicalize_mapping_tree

    state, _ = _canonicalize_mapping_tree(state)
    # Finish public prepared/constant mapping reads before source checks.
    selected_dt = (
        _functional_duration_timestep(target, functional, continuation)
        if duration is not None
        else None
    )
    functional._validate_source()
    functional._validate_state(state)
    if duration is not None:
        from dendra.models.core import _duration_step_budget, _validate_time_scalar

        from ._network import FunctionalNetwork
        from ._runners import (
            _validate_duration_remainder,
            _validate_network_duration_remainder,
        )

        if (
            dt is not None
            and _validate_time_scalar(dt, name="dt", positive=True) != selected_dt
        ):
            raise ValueError("dt must match the functional execution timestep")
        pending = (
            _validate_network_duration_remainder(state["clock"]["duration_remainder"])
            if type(functional) is FunctionalNetwork
            else _validate_duration_remainder(state)
        )
        steps, _ = _duration_step_budget(duration, selected_dt, pending)
    return state, steps


def resolve_steps(target, state=None, *, steps=None, duration=None, dt=None):
    """Return the accepted timestep count without advancing an execution boundary.

    Supply exactly one explicit ``steps=`` or ``duration=`` request. ``steps``
    is a non-negative integer; ``duration`` is non-negative finite time in ms.
    Runner defaults and compiled chunk widths are not selected implicitly.
    The returned Python integer sizes deterministic :class:`Inputs` tapes and
    matches :func:`sample_runtime_draws` for the same boundary and request.

    Initialized native Populations and Networks use their current boundary and
    require ``state`` omitted. A Population uses ``dt`` or the native backend
    default; a Network uses its initialized timestep and requires a supplied
    ``dt`` to match. Native host scheduling and duration budgeting must retain
    their canonical implementations. Integrators need not support functional
    lowering or runtime draw sampling. The query never inspects stochastic
    distributions, constructs samplers or probes waveforms.

    Functional plans, bindings, compiled chunks and ``dn.compile`` runners
    require explicit state or a :class:`Continuation`. Population durations
    read the effective timestep from a binding, continuation constants, or
    runner defaults, in that order. Unbound plans/chunks with plain state use
    their nominal timestep; Networks use their fixed plan timestep. A supplied
    ``dt`` validates that timestep rather than overriding it. No preparation,
    cache refresh, native extraction or initialization is performed.

    Durations include retained fractional time using execution's exact budget
    arithmetic. Zero or substep durations can resolve to zero; existing pending
    time can also complete a step. Explicit counts do not spend pending time.
    Pass the original duration to execution so it publishes the updated carry::

        request = {"duration": 10.0}
        count = dn.func.resolve_steps(program, trial, **request)
        inputs = dn.func.Inputs(intra=current[:count])
        state, auxiliary = dn.func.run(program, trial, inputs=inputs, **request)

    The query leaves clocks, fractional carry, RNG, callbacks, parameter
    ownership and autograd history unchanged. It is a host operation outside
    compilation, function transforms and forward AD, including in inference
    mode. Stimulation bindings and their input tapes remain execution inputs.
    """
    _cold_boundary("resolve_steps")
    steps, duration = execution_call_horizon(None, None, steps, duration)
    if steps is None and duration is None:
        raise ValueError("resolve_steps requires steps or duration")

    from dendra.models import Network, Population

    with torch.inference_mode(False), torch.no_grad():
        if isinstance(target, (Population, Network)):
            if state is not None:
                raise TypeError("state must be omitted for native step resolution")
            selected_dt = _native_timestep(target, dt, sampling=False)
            if duration is not None:
                steps = _native_duration_steps(
                    target, duration, selected_dt, operation="Step resolution"
                )
        else:
            functional = _functional_plan(target, operation="resolve_steps")
            if state is None:
                raise TypeError("Functional step resolution requires explicit state")
            _, steps = _functional_steps(
                target, functional, state, steps=steps, duration=duration, dt=dt
            )
    return int(steps)


def sample_runtime_draws(
    target, state=None, *, steps=None, duration=None, dt=None, extra=None
):
    """Sample detached runtime draws without advancing a model or its RNG.

    Native initialized Populations and Networks sample from their current logical
    RNG boundary. A Population uses ``dt`` or the same backend default as its
    native execution methods. A Network uses its initialized timestep; a supplied
    ``dt`` must match. Pass call-local extracellular ``extra`` bindings again when
    replaying their Poisson draws. Sampling neither initializes nor lowers the
    native model. Cold invocation profiling uses disposable copies when needed;
    the source's parameters and numerical state remain unchanged.
    Compatible subclasses retain the canonical execution and lifecycle hooks.

    Functional Population/Network plans, bindings, compiled chunks and
    ``dn.compile`` runners require explicit ``state`` or a :class:`Continuation`.
    Stimulation bindings belong to the lowered plan. ``extra`` must be omitted,
    and a supplied ``dt`` must match the plan, or the effective execution
    timestep when sampling a duration. Runner snapshot/fresh initialization
    policies are not used. Duration sampling reads a Population's timestep from its
    binding, continuation constants, or runner defaults, in that order. An
    unbound plan or chunk with plain state uses the plan timestep. Network
    duration sampling uses its fixed plan timestep. No preparation is performed
    and source parameters/constants, callback carry and caches are unchanged.

    Omit ``steps`` and ``duration`` for one frame. Supply a non-negative integer
    ``steps`` or a non-negative finite ``duration`` in ms for a time-leading
    tape. They are mutually exclusive. Duration uses the same fractional carry
    and timestep budgeting as execution, without publishing a new remainder.
    Replay with the same duration and starting boundary; execution alone updates
    the carry. A duration accepting no complete steps produces fresh empty
    leaves without advancing any live or returned stream boundary. Explicit
    step counts do not spend fractional carry. The returned :class:`RuntimeDraws`
    uses runtime-only Poisson invocation rows and replays through native ``draws=`` or
    functional execution. Existing plan sampling methods retain their
    initialization-reserved packet layouts.

    Each physical stream has one isolated generator; requests are sampled in
    native phase order. Source/global RNG, pending logical boundaries, input tensors,
    waveform schedules and live parameter aliases remain unchanged. Outputs are
    detached ordinary tensors with mutation counters, including when called
    inside ``torch.inference_mode()``. Sampling is a host operation outside
    compilation, function transforms and forward AD.
    """
    _cold_boundary("sample_runtime_draws")
    steps, duration = execution_call_horizon(None, None, steps, duration)

    from dendra.models import Network, Population

    with torch.inference_mode(False), torch.no_grad():
        if isinstance(target, (Population, Network)):
            if state is not None:
                raise TypeError(
                    "state must be omitted for native runtime draw sampling"
                )
            selected_dt = _native_timestep(target, dt)
            if isinstance(target, Network) and extra is not None:
                if not isinstance(extra, Mapping):
                    raise TypeError(
                        "extra must map Population names to stimulation bindings"
                    )
                extra = dict(extra)
                unknown = extra.keys() - target.populations.keys()
                if unknown:
                    raise ValueError(
                        f"extra names unknown Populations: {sorted(unknown, key=str)!r}"
                    )
            from ._runtime_sampling import sample_native_runtime_draws

            if duration is not None:
                steps = _native_duration_steps(target, duration, selected_dt)
            return sample_native_runtime_draws(
                target, dt=selected_dt, steps=steps, extra=extra
            )

        functional = _functional_plan(target)
        if extra is not None:
            raise TypeError(
                "extra must be omitted for functional runtime draw sampling"
            )
        if state is None:
            raise TypeError("Functional runtime draw sampling requires explicit state")
        state, steps = _functional_steps(
            target, functional, state, steps=steps, duration=duration, dt=dt
        )
        return functional._runtime_draw_plan.sample_runtime(state, steps=steps)


__all__ = ["resolve_steps", "sample_runtime_draws"]
