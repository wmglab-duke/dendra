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
        _is_canonical_integrator_type,
        _resolve_functional_integrator_spec,
    )

    integrator = population.integrator
    integrator_type = type(integrator)
    implementation = next(
        (
            candidate
            for candidate in _STANDARD_INTEGRATOR_FRESH_INITIALIZATION
            if _is_canonical_integrator_type(integrator_type, candidate)
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
    """Return an explicit request's accepted timestep count without running it.

    Parameters
    ----------
    target : native model, functional plan, binding, compiled chunk or runner
        Native models must be initialized. Functional targets require ``state``.
    state : tensor tree or Continuation, optional
        Functional execution boundary; omitted for native models.
    steps : int, optional
        Non-negative exact count. Supply exactly one of ``steps`` or ``duration``.
    duration : float, optional
        Non-negative elapsed time in ms, combined with retained fractional time.
    dt : float, optional
        Native Population timestep; omission uses the native backend default.
        Native Networks use their initialized timestep. Functional callers normally
        omit it; an explicit value is validated against the target's timestep.

    Returns
    -------
    int
        Number of complete timesteps. Use it to size time-leading :class:`Inputs`.
        Runner horizon defaults and compiled widths are not inferred.

    Notes
    -----
    Population duration queries use the effective timestep from a binding,
    Continuation constants or runner defaults. An unbound plan/chunk with plain
    state uses its nominal timestep. Networks use their fixed plan timestep.
    Native queries require canonical host duration budgeting but not functional
    integrator or stochastic-sampler support.

    The query leaves state, fractional carry, RNG, callbacks and caches unchanged.
    It performs no initialization, preparation, lowering or waveform evaluation.
    Execute with the original ``duration=`` request so execution publishes the
    updated remainder; replacing it with ``steps=resolved_count`` preserves the old
    remainder instead. This is a host operation outside compilation, function
    transforms and forward AD.
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
    """Sample detached runtime draws without advancing the incoming boundary.

    Parameters
    ----------
    target : native model, functional plan, binding, compiled chunk or runner
        Initialized native model or compatible functional execution target.
    state : tensor tree or Continuation, optional
        Required for functional targets and omitted for native targets.
    steps : int, optional
        Non-negative tape length. Omit both horizon arguments for a single frame.
    duration : float, optional
        Non-negative elapsed time in ms. Produces a tape using the same effective
        timestep and fractional carry as execution. Mutually exclusive with steps.
    dt : float, optional
        Native Population timestep, defaulting to the native backend setting.
        A supplied Native Network or functional timestep must match its target.
    extra : stimulation specification or mapping, optional
        Native call-local extracellular bindings whose Poisson draws are sampled.
        Supply them again during replay. Omit for functional targets, whose
        stimulation bindings belong to the lowered plan.

    Returns
    -------
    RuntimeDraws
        One draw/RNG frame, or a time-leading tape for an explicit horizon. Zero
        accepted steps produces fresh empty leaves without advancing RNG carry.

    Notes
    -----
    Source state, global/source RNG, waveform schedules and preparation caches are
    unchanged. Sampling uses isolated generators in native request order and
    preserves shared streams. Outputs are detached versioned tensors, including
    when called under inference mode. Replay with the same boundary, duration and
    stimulation bindings so execution alone updates fractional time and RNG.

    Native sampling requires compatible canonical scheduling and a single
    simulation device. Functional runners require explicit state; their restart
    policy is not applied. This host operation runs outside compilation, function
    transforms and forward AD. Initialization uses
    :func:`sample_initialization_draws` and :class:`InitializationDraws` instead.
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
