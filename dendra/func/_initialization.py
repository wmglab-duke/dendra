"""Shared cold initialization over the existing pure model adapters."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ._continuation import Continuation
from ._network_bindings import _map_containers
from ._types import FunctionalizationError, InitializationInput, PopulationTensors


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


def _initialize_host(functional, parameters, constants, state, initialization):
    """Dispatch without changing existing caller validation or result contracts."""
    from ._network import FunctionalNetwork

    if isinstance(functional, FunctionalNetwork):
        return functional._initialize_host(parameters, constants, state, initialization)
    return functional.initialize_host(
        parameters, constants, initialization, state=state
    )


def initialize(functional, tensors, *, initialization=None):
    """Initialize a lowered Population or Network and return a Continuation.

    ``tensors`` is the matching ``PopulationTensors`` or ``NetworkTensors`` bundle
    returned by ``make_functional``. Its explicit state, parameters and constants
    supply the starting clock/RNG carry and raw sources; no native extraction is
    performed. The result retains effective initialization replacements, aliases
    and gradient history without copying or detaching Tensor leaves.

    ``initialization`` overrides the existing model-specific initialization
    packet. Omission uses ``tensors.initialization`` for a Population or
    ``functional.initialization_inputs`` for a Network. Bound caches and native
    ordering follow the existing initialization machinery. Missing draws are
    sampled from independent explicit RNG carry. Supplied draws are replayed,
    including draws retained in an already initialized Population bundle.

    This host entry point runs outside ``torch.compile``, ``torch.func`` and
    forward AD. The pure ``functional.initialize`` methods and their existing
    result types remain available for transforms. Initialization creates ordinary
    Tensors even inside inference mode, while respecting the caller's gradient
    mode and the model's native training policy.

    The returned packet has no callback carry or recording history and advances
    no accepted timestep. Bind its effective sources, then start callbacks on the
    first run::

        initial = dn.func.initialize(functional, tensors)
        bound = kernel.bind(initial.parameters, constants=initial.constants)
        state, aux = dn.func.run(bound, initial, steps=100, callbacks=callbacks)

    Source models, bundle containers and initialization defaults are unchanged.
    Runtime ``Inputs``, ``RuntimeDraws`` and waveform recipes remain run inputs.
    """
    if (
        torch.compiler.is_compiling()
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    ):
        raise FunctionalizationError(
            "initialize must run outside torch.compile, torch.func and forward AD"
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

    gradient = torch.is_grad_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(gradient):
        initialized = _initialize_host(
            functional, parameters, constants, state, selected
        )
        parameters, constants = initialized.parameters, initialized.constants
        if type(functional) is FunctionalNetwork:
            constants = functional._bind_parameter_constants(parameters, constants)
        return Continuation(initialized.state, parameters, constants)


__all__ = ["initialize"]
