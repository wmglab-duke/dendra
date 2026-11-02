"""Shared cold runtime sampling over native and explicit functional state."""

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


def _native_timestep(model, dt):
    from dendra.models import Network
    from dendra.models.backend import Backend
    from dendra.models.core import _validate_time_scalar

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
            raise FunctionalizationError(
                "Initialize the Network before sampling runtime draws"
            )
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
        raise FunctionalizationError(
            "Initialize every Population before sampling runtime draws"
        )
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


def _functional_plan(target):
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
        "sample_runtime_draws requires a native Population/Network, a functional "
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
                import inspect

                owner = Network if isinstance(target, Network) else Population
                if (
                    inspect.getattr_static(target, "_duration_budget")
                    is not owner._duration_budget
                ):
                    raise FunctionalizationError(
                        "Duration draw sampling requires canonical duration budgeting"
                    )
                steps, _ = owner._duration_budget(target, duration, selected_dt)
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
                _validate_network_duration_remainder(
                    state["clock"]["duration_remainder"]
                )
                if type(functional) is FunctionalNetwork
                else _validate_duration_remainder(state)
            )
            steps, _ = _duration_step_budget(duration, selected_dt, pending)
        return functional._runtime_draw_plan.sample_runtime(state, steps=steps)


__all__ = ["sample_runtime_draws"]
