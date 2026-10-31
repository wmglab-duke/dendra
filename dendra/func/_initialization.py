"""Shared cold initialization over the existing pure model adapters."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ._continuation import _make_continuation
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
    runner, parameters, constants, initialization, *, draws=None, _draws_capture=None
):
    """Initialize from runner defaults without entering its execution lifecycle."""
    state = runner._initial_state
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
    """Initialize a runner or lowered model and return a Continuation.

    ``initialize(program)`` uses either compiled runner's raw defaults and
    initial clock/RNG snapshot. Optional ``parameters=`` and ``constants=``
    mappings override those sources; ``None`` retains the corresponding
    defaults. ``initialization=`` overrides its initialization-input packet.
    This explicit operation always runs fresh initialization, including bound
    caches and their RNG policies. The runner's snapshot/fresh policy continues
    to control ordinary no-state calls. Runner defaults, prepared caches and
    execution reports remain unchanged.

    For a lowered model, ``tensors`` is the matching ``PopulationTensors`` or
    ``NetworkTensors`` bundle returned by ``make_functional``. Its explicit
    state, parameters and constants
    supply the starting clock/RNG carry and raw sources; no native extraction is
    performed. The result retains effective initialization replacements, aliases
    and gradient history without copying or detaching Tensor leaves.

    ``initialization`` overrides the existing model-specific initialization
    packet. Omission uses ``tensors.initialization`` for a Population or
    ``functional.initialization_inputs`` for a Network. Bound caches and native
    ordering follow the existing initialization machinery. Missing draws are
    sampled from independent explicit RNG carry. Supplied draws are replayed,
    including draws retained in an already initialized Population bundle.
    ``draws=InitializationDraws(...)`` supplies the complete stochastic input
    separately from deterministic initialization values and replays without
    automatic sampling. It cannot be combined with legacy draw fields.
    Use :func:`sample_initialization_draws` to capture packets in native order.

    This host entry point runs outside ``torch.compile``, ``torch.func`` and
    forward AD. The pure ``functional.initialize`` methods and their existing
    result types remain available for transforms. Initialization creates ordinary
    Tensors even inside inference mode, while respecting the caller's gradient
    mode and the model's native training policy. For explicit runner
    initialization, the caller's gradient mode also applies independently of
    the runner's inference flag; that flag controls subsequent execution.

    The returned packet has no callback carry or recording history and advances
    no accepted timestep. Bind its effective sources, then start callbacks on the
    first run::

        initial = dn.func.initialize(functional, tensors)
        bound = kernel.bind(initial.parameters, constants=initial.constants)
        state, aux = dn.func.run(bound, initial, steps=100, callbacks=callbacks)

    Source models, bundle containers and initialization defaults are unchanged.
    Runtime ``Inputs``, ``RuntimeDraws`` and waveform recipes remain run inputs.
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
        if tensors is not _UNSET:
            raise TypeError(
                "tensors cannot be supplied when initializing a compiled runner"
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
            draws=draws,
            _draws_capture=_draws_capture,
        )

    from ._network import FunctionalNetwork, NetworkTensors
    from ._population import FunctionalPopulation

    if type(functional) is FunctionalPopulation:
        if type(tensors) is not PopulationTensors:
            raise TypeError(
                "tensors must be a PopulationTensors for this FunctionalPopulation"
            )
        selected = tensors.initialization if initialization is None else initialization
    elif type(functional) is FunctionalNetwork:
        if type(tensors) is not NetworkTensors:
            raise TypeError(
                "tensors must be a NetworkTensors for this FunctionalNetwork"
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
    """Capture replayable initialization draws from a runner or lowered model.

    Accepts the same targets and source overrides as :func:`initialize`.
    Sampling runs the existing host initializer under ``no_grad``, including
    pure transforms and cache restoration, to observe each sampling boundary
    in native order. The resulting packet includes the actual draw selections
    and intermediate RNG frames. Initialized model values are not published.
    Source modules, input tensors, runner defaults and global RNG are unchanged.
    Newly sampled tensors are detached. Supplied legacy draw fields are
    captured as provided, preserving their tensor identities and gradient flags.

    Replay with ``initialize(target, ..., draws=packet)``. Capture and replay
    are separate initialization executions; each invokes authored transforms
    once. Ordinary automatic initialization does not perform a capture pass.
    """
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
