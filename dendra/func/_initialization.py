"""Shared cold initialization over the existing pure model adapters."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ._continuation import Continuation, _make_continuation, _resolve_continuation
from ._initialization_draws import InitializationDraws
from ._network_bindings import _map_containers
from ._types import FunctionalizationError, InitializationInput, PopulationTensors

_UNSET = object()


def _initialization_input_mapping(inputs):
    """Snapshot external mappings without severing differentiable leaves."""
    if not isinstance(inputs, InitializationInput):
        raise TypeError("initialization must be an InitializationInput")
    mappings = []
    for name, values in (
        ("states", inputs.states),
        ("transforms", inputs.transforms),
        ("cache", inputs.cache),
    ):
        if values is not None and not isinstance(values, Mapping):
            raise TypeError(f"initialization {name} must be a mapping")
        mappings.append(None if values is None else dict(values))
    return InitializationInput(
        inputs.v_init, *mappings, inputs.random_draw, inputs.stimulation
    )


def _initialize_host(
    functional,
    parameters,
    constants,
    state,
    initialization,
    *,
    draws=None,
    _draws_capture=None,
):
    """Dispatch without changing existing caller validation or result contracts."""
    from ._network import FunctionalNetwork

    options = {}
    if draws is not None:
        options["draws"] = draws
    if _draws_capture is not None:
        options["_draws_capture"] = _draws_capture
    if isinstance(functional, FunctionalNetwork):
        return functional._initialize_host(
            parameters, constants, state, initialization, **options
        )
    return functional.initialize_host(
        parameters, constants, initialization, state=state, **options
    )


def _initialize_fresh(
    functional,
    parameters,
    constants,
    state,
    initialization,
    *,
    gradient,
    draws=None,
    _draws_capture=None,
):
    """Run one fresh initialization with ordinary, versioned tensor outputs."""
    with torch.inference_mode(False), torch.set_grad_enabled(gradient):
        return _initialize_host(
            functional,
            parameters,
            constants,
            state,
            initialization,
            draws=draws,
            _draws_capture=_draws_capture,
        )


def _runner_sources(
    runner, parameters, constants, initialization, *, fresh, inputs=None
):
    """Select raw frames in each runner's authored-container traversal order."""
    from ._network import FunctionalNetwork

    if isinstance(runner._functional, FunctionalNetwork):
        selected_parameters = (
            runner._parameters
            if parameters is None
            else _map_containers(parameters, dict)
        )
        selected_inputs = _map_containers(inputs, dict)
        selected_initialization = (
            _map_containers(
                runner._initialization_inputs
                if initialization is None
                else initialization,
                dict,
            )
            if fresh
            else None
        )
        selected_constants = (
            runner._constants if constants is None else _map_containers(constants, dict)
        )
        selected_constants = runner._functional._bind_parameter_constants(
            selected_parameters, selected_constants
        )
    else:
        selected_parameters = (
            runner._parameters if parameters is None else dict(parameters)
        )
        selected_inputs = inputs
        selected_initialization = (
            _initialization_input_mapping(
                runner._initialization_inputs
                if initialization is None
                else initialization
            )
            if fresh
            else None
        )
        selected_constants = runner._constants if constants is None else dict(constants)
    return (
        selected_parameters,
        selected_constants,
        selected_initialization,
        selected_inputs,
    )


def _initialize_runner(
    runner,
    parameters,
    constants,
    initialization,
    *,
    state=_UNSET,
    draws=None,
    _draws_capture=None,
):
    """Initialize selected sources without entering the runner's lifecycle."""
    state = runner._initial_state if state is _UNSET else _map_containers(state, dict)
    parameters, constants, selected, _ = _runner_sources(
        runner, parameters, constants, initialization, fresh=True
    )
    return _initialize_continuation(
        runner._functional,
        parameters,
        constants,
        state,
        selected,
        draws=draws,
        _draws_capture=_draws_capture,
    )


def _initialize_continuation(
    functional,
    parameters,
    constants,
    state,
    selected,
    *,
    draws=None,
    _draws_capture=None,
):
    """Publish effective initialization frames under the caller's gradient mode."""
    from ._network import FunctionalNetwork

    gradient = torch.is_grad_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(gradient):
        initialized = _initialize_fresh(
            functional,
            parameters,
            constants,
            state,
            selected,
            gradient=gradient,
            draws=draws,
            _draws_capture=_draws_capture,
        )
        parameters, constants = initialized.parameters, initialized.constants
        if isinstance(functional, FunctionalNetwork):
            constants = functional._bind_parameter_constants(parameters, constants)
        return _make_continuation(
            initialized.state, parameters, constants, lifecycle="unstarted"
        )


def initialize(
    functional,
    tensors=_UNSET,
    *,
    initialization=None,
    parameters=_UNSET,
    constants=_UNSET,
    draws=None,
):
    """Create a fresh functional execution boundary.

    Parameters
    ----------
    functional : compiled runner, FunctionalPopulation or FunctionalNetwork
        Receiving runner or lowered plan. Initialization does not advance its
        native source or update runner defaults.
    tensors : PopulationTensors, NetworkTensors or Continuation, optional
        Explicit source/state boundary. Plans require a matching tensor bundle or
        Continuation. A runner uses its captured defaults when omitted; a supplied
        Continuation supplies effective sources and current RNG carry instead.
    initialization : InitializationInput or NetworkInitializationInput, optional
        Initialization recipe override. Runners and Network plans retain defaults;
        a Population bundle supplies ``tensors.initialization``. A Population plan
        with a Continuation requires this argument explicitly.
    parameters, constants : mapping, optional
        Runner-only source overrides. Non-None overrides cannot accompany a
        Continuation, whose sources are already explicit.
    draws : InitializationDraws, optional
        Replay complete initialization draws. Omission samples missing draws from
        explicit RNG carry. Cannot be combined with component draw fields retained
        in an initialization recipe.

    Returns
    -------
    Continuation
        Fresh simulation state and effective parameter/constant mappings, with no
        callback carry. The first run starts fresh recordings and reductions.

    Notes
    -----
    Native initialization ordering, retained samples, reseeding and bound
    steady-state cache policies apply. Parameter-writing transforms operate on the
    supplied effective sources; initialization does not recover an earlier raw
    baseline. Tensor aliases and gradient history are retained without detaching
    sources. Use :func:`detach_continuation` between independent training trials
    when recurrent history should be cut.

    This host operation runs outside compilation, ``torch.func`` and forward AD.
    It follows the caller's gradient mode even for an inference-configured runner
    and creates versioned outputs inside inference mode. The lower-level pure
    plan ``initialize`` methods remain available for tensor transforms.
    """
    return _initialize_target(
        functional,
        tensors,
        initialization=initialization,
        parameters=parameters,
        constants=constants,
        draws=draws,
    )


def _initialize_target(
    functional,
    tensors,
    *,
    initialization,
    parameters,
    constants,
    draws=None,
    _draws_capture=None,
):
    """Resolve each public target once before the shared ordered execution."""
    from ._convenience import CompiledRunner
    from ._network_compile import CompiledNetworkRunner

    is_runner = type(functional) in (CompiledRunner, CompiledNetworkRunner)
    if is_runner:
        if tensors is not _UNSET and not isinstance(tensors, Continuation):
            raise TypeError(
                "tensors cannot be supplied when initializing a compiled runner"
            )
        if isinstance(tensors, Continuation):
            # Fresh initialization selects its own recipe and resets callbacks.
            _, parameters, constants, _ = _resolve_continuation(
                tensors,
                None if parameters is _UNSET else parameters,
                None if constants is _UNSET else constants,
                None,
                None,
            )
    else:
        for name, value in (("parameters", parameters), ("constants", constants)):
            if value is not _UNSET:
                raise TypeError(
                    f"initialize() got an unexpected keyword argument {name!r}"
                )
        if tensors is _UNSET:
            raise TypeError(
                "initialize() missing 1 required positional argument: 'tensors'"
            )
    if (
        torch.compiler.is_compiling()
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    ):
        raise FunctionalizationError(
            "initialize must run outside torch.compile, torch.func and forward AD"
        )

    if is_runner:
        return _initialize_runner(
            functional,
            None if parameters is _UNSET else parameters,
            None if constants is _UNSET else constants,
            initialization,
            state=tensors.state if isinstance(tensors, Continuation) else _UNSET,
            draws=draws,
            _draws_capture=_draws_capture,
        )

    from ._network import FunctionalNetwork, NetworkTensors
    from ._population import FunctionalPopulation

    if type(functional) is FunctionalPopulation:
        if type(tensors) is not PopulationTensors and not isinstance(
            tensors, Continuation
        ):
            raise TypeError(
                "tensors must be a PopulationTensors or Continuation for this FunctionalPopulation"
            )
        if isinstance(tensors, Continuation) and initialization is None:
            raise TypeError(
                "initialization is required when initializing a FunctionalPopulation from a Continuation"
            )
        selected = tensors.initialization if initialization is None else initialization
    elif type(functional) is FunctionalNetwork:
        if type(tensors) is not NetworkTensors and not isinstance(
            tensors, Continuation
        ):
            raise TypeError(
                "tensors must be a NetworkTensors or Continuation for this FunctionalNetwork"
            )
        selected = (
            functional.initialization_inputs
            if initialization is None
            else initialization
        )
    else:
        raise TypeError(
            "functional must be a FunctionalPopulation or FunctionalNetwork"
        )
    if tensors.state is None:
        raise ValueError("initialization requires explicit non-None state")

    # Finish authored container traversal before source qualification or sampling.
    parameters, constants, state, selected = _map_containers(
        (tensors.parameters, tensors.constants, tensors.state, selected), dict
    )
    if type(functional) is FunctionalPopulation:
        selected = _initialization_input_mapping(selected)
    else:
        constants = functional._bind_parameter_constants(parameters, constants)

    return _initialize_continuation(
        functional,
        parameters,
        constants,
        state,
        selected,
        draws=draws,
        _draws_capture=_draws_capture,
    )


def sample_initialization_draws(
    functional,
    tensors=_UNSET,
    *,
    initialization=None,
    parameters=_UNSET,
    constants=_UNSET,
):
    """Capture replayable stochastic inputs for one fresh initialization.

    Parameters
    ----------
    functional : initialized Population/Network, compiled runner or lowered plan
        Target whose initialization protocol determines draw ordering.
    tensors : PopulationTensors, NetworkTensors or Continuation, optional
        Required for a lowered plan, optional for a runner, and omitted for native
        targets. A Continuation supplies its current effective sources and RNG.
    initialization : InitializationInput or NetworkInitializationInput, optional
        Functional recipe override. A Population plan with a Continuation requires
        it. Native capture uses the model's default initialization options.
    parameters, constants : mapping, optional
        Runner-only source overrides, incompatible with a supplied Continuation.

    Returns
    -------
    InitializationDraws
        Base draws and resulting RNG states for replay with native
        ``model.initialize(draws=packet)`` or :func:`initialize`.

    Notes
    -----
    Capture leaves the source boundary, global RNG and runner caches unchanged.
    It follows initialization transforms and cache restoration under ``no_grad``
    to observe draw ordering; initialized values are discarded. Native capture
    uses an isolated model copy and requires an initialized compatible model.
    Newly sampled draws are detached. Explicit component draw values already in
    the recipe retain their Tensor identities and gradient flags.

    Capture and replay are separate initialization executions; each invokes
    authored transforms. Automatic initialization samples within a single call.
    Forced native rebuilding is unsupported with explicit draw replay. This host
    operation runs outside compilation, function transforms and forward AD.
    """
    from dendra.models import Network, Population

    if isinstance(functional, (Population, Network)):
        if tensors is not _UNSET:
            raise TypeError(
                "tensors must be omitted for native initialization sampling"
            )
        if (
            initialization is not None
            or parameters is not _UNSET
            or constants is not _UNSET
        ):
            raise TypeError(
                "Native initialization sampling uses the model's initialization defaults"
            )
        from ._native_initialization_draws import sample_native_initialization_draws

        return sample_native_initialization_draws(functional)
    captured = {}
    with torch.inference_mode(False), torch.no_grad():
        _initialize_target(
            functional,
            tensors,
            initialization=initialization,
            parameters=parameters,
            constants=constants,
            _draws_capture=captured,
        )
    return InitializationDraws(**captured)


__all__ = ["initialize", "sample_initialization_draws"]
