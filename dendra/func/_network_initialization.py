"""Pure initialization of a frozen Network from explicit component inputs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import NamedTuple

import torch

from ._netstim import NetStimDraw
from ._network_bindings import _canonicalize_mapping_tree
from ._types import FunctionalizationError, InitializationInput


class NetworkInitializationInput(NamedTuple):
    """Initialization-only Population values, starting time, and optional draw.

    ``populations`` maps every authored Population name to an
    :class:`InitializationInput`. ``t`` is a scalar Tensor matching the Network
    clock's dtype and device. It is detached during initialization, matching
    native clock synchronization. ``netstim_draw`` is an explicit initialization
    packet; the host runner samples one when omitted. Step draw tapes remain
    separate runtime inputs.
    """

    populations: dict[str, InitializationInput]
    t: torch.Tensor
    netstim_draw: NetStimDraw | None = None


def initialization_alias_failures(functional):
    """Reject native retained materializations absent from the explicit packet."""
    from ._population import _initialization_transform_hooks

    order = {name: index for index, name in enumerate(functional.populations)}
    failures = []
    for name, plan in functional.populations.items():
        population = plan._population
        writes = {
            reference.removeprefix("parameters.")
            for phase in ("pre", "post")
            for hook in _initialization_transform_hooks(population, phase)
            for reference in population._initialization_transforms[
                hook.action_name
            ].writes
            if reference.startswith("parameters.")
        }
        for path, constant_paths in functional._parameter_constant_aliases:
            if path[:2] == ("populations", name) and path[2] in writes:
                bindings = ", ".join(
                    ".".join(map(str, target)) for target in constant_paths
                )
                failures.append(
                    f"Population {name!r} initialization writes a parameter shared "
                    f"with explicit Network constants ({bindings}). These writes "
                    "are currently unsupported; parameter replacements do not "
                    "replace constant bindings"
                )
        for group in functional._parameter_aliases:
            if not any(
                path[:2] == ("populations", name) and path[2] in writes
                for path in group
            ):
                continue
            for path in group:
                if (
                    path[0] == "populations"
                    and order[path[1]] < order[name]
                    and "._initial_sources." not in path[2]
                ):
                    failures.append(
                        f"Population {name!r} writes a shared runtime parameter of "
                        f"already initialized Population {path[1]!r}; native initialization "
                        "retains its earlier materialized coefficients. This ordering is "
                        "currently unsupported; initialize shared runtime parameters "
                        "before their first Population reader"
                    )
    return tuple(dict.fromkeys(failures))


def _replace_at(tree, path, value):
    """Replace one leaf in an owned container tree, retaining tensor history."""
    if not path:
        return value
    key, *remaining = path
    if isinstance(tree, Mapping):
        result = dict(tree)
        result[key] = _replace_at(tree[key], remaining, value)
        return result
    values = list(tree)
    values[key] = _replace_at(tree[key], remaining, value)
    return type(tree)(*values) if hasattr(tree, "_fields") else tuple(values)


def _propagate_population_parameters(functional, parameters, name, replacement):
    """Publish a Population transaction through shared Network parameter aliases."""
    from ._network import _at

    result = _replace_at(parameters, ("populations", name), replacement)
    for group in functional._parameter_aliases:
        changed = [
            path
            for path in group
            if path[:2] == ("populations", name)
            and _at(result, path) is not _at(parameters, path)
        ]
        if not changed:
            continue
        value = _at(result, changed[-1])
        if any(_at(result, path) is not value for path in changed[:-1]):
            raise FunctionalizationError(
                "Population initialization produced conflicting shared parameter replacements"
            )
        for path in group:
            result = _replace_at(result, path, value)
    return result


def _validate(functional, parameters, constants, state, inputs, draw, *, require_draw):
    from ._network import _validate_tree

    functional._validate_initialization_source()
    functional._validate_parameters(parameters)
    _validate_tree(constants, functional._constant_schema, "constants")
    functional._validate_state(state)
    if not isinstance(inputs, NetworkInitializationInput):
        raise TypeError("inputs must be a NetworkInitializationInput")
    if not isinstance(inputs.populations, Mapping) or set(inputs.populations) != set(
        functional.populations
    ):
        raise ValueError("initialization populations must match the Network")
    _validate_tree(inputs.t, functional._state_schema["clock"]["t"], "initialization.t")
    if not torch._C._are_functorch_transforms_active():
        if torch.compiler.is_compiling():
            torch._assert_async(
                torch.isfinite(inputs.t), "initialization.t must be finite"
            )
        elif not bool(torch.isfinite(inputs.t)):
            raise ValueError("initialization.t must be finite")
    if functional.netstim is None:
        if draw is not None:
            raise ValueError(
                "a Network without NetStim does not accept an initialization draw"
            )
    elif draw is None and require_draw:
        raise ValueError(
            "NetStim initialization requires an explicit draw; use sample_initialization_draw() outside transformed execution"
        )
    elif draw is not None:
        functional.netstim._validate_draw(draw)


def _initialize_components(functional, parameters, constants, state, inputs):
    """Complete the native phases preceding NetStim initialization."""
    from ._network import NetworkTensors

    populations = {}
    effective = parameters
    with nullcontext() if functional.training else torch.no_grad():
        # Native initialize completes each Population in authored order before
        # resetting connections. Parameter-writing transforms affect every
        # alias before the next Population observes its input frame.
        for name, plan in functional.populations.items():
            initialized = plan.initialize(
                effective["populations"][name],
                constants["populations"][name],
                inputs.populations[name],
            )
            effective = _propagate_population_parameters(
                functional, effective, name, initialized.parameters
            )
            populations[name] = initialized.state
        time = inputs.t.detach().clone()
        populations = {
            name: functional._snap_population_clock(value, time)
            for name, value in populations.items()
        }
        clock = {
            "t": time,
            "origin": time.clone(),
            "step": torch.zeros_like(state["clock"]["step"]),
            "duration_remainder": torch.zeros_like(
                state["clock"]["duration_remainder"]
            ),
        }
        synapses = {
            name: plan.initial_state(state["synapses"][name], t=time)
            for name, plan in functional.synapses.items()
        }
        continuous = {
            name: plan.initial_state(state["continuous_synapses"][name], t=time)
            for name, plan in functional.continuous_synapses.items()
        }
        initialized_state = {
            "populations": populations,
            "synapses": synapses,
            "continuous_synapses": continuous,
            "netstim": state["netstim"],
            "clock": clock,
        }
    return NetworkTensors(effective, constants, initialized_state)


def _finish_initialization(functional, packet, draw):
    from ._network import NetworkTensors

    state = dict(packet.state)
    if functional.netstim is not None:
        state["netstim"] = functional.netstim.initial_state(
            packet.parameters["netstim"]["parameters"],
            packet.constants["netstim"],
            state["netstim"],
            draw,
        )
    functional._validate_parameters(packet.parameters)
    functional._validate_state(state)
    return NetworkTensors(packet.parameters, packet.constants, state)


def initialize(functional, parameters, constants, state, inputs, *, netstim_draw=None):
    """Initialize one complete boundary without sampling or mutating operands."""
    (parameters, constants, state, inputs, netstim_draw), _ = (
        _canonicalize_mapping_tree((parameters, constants, state, inputs, netstim_draw))
    )
    if not isinstance(inputs, NetworkInitializationInput):
        raise TypeError("inputs must be a NetworkInitializationInput")
    if netstim_draw is not None and inputs.netstim_draw is not None:
        raise ValueError("supply the NetStim initialization draw only once")
    draw = inputs.netstim_draw if netstim_draw is None else netstim_draw
    _validate(functional, parameters, constants, state, inputs, draw, require_draw=True)
    packet = _initialize_components(functional, parameters, constants, state, inputs)
    return _finish_initialization(functional, packet, draw)


def initialize_host(functional, parameters, constants, state, inputs):
    """Cold sampling adapter over the same pure initialization phases."""
    (parameters, constants, state, inputs), _ = _canonicalize_mapping_tree(
        (parameters, constants, state, inputs)
    )
    if not isinstance(inputs, NetworkInitializationInput):
        raise TypeError("inputs must be a NetworkInitializationInput")
    _validate(
        functional,
        parameters,
        constants,
        state,
        inputs,
        inputs.netstim_draw,
        require_draw=False,
    )
    packet = _initialize_components(functional, parameters, constants, state, inputs)
    draw = inputs.netstim_draw
    if functional.netstim is not None and draw is None:
        # Native initialization samples last. Population transforms may have
        # changed a parameter shared with NetStim, including conditional noise.
        draw = functional.sample_initialization_draw(packet.parameters, packet.state)
    return _finish_initialization(functional, packet, draw)


__all__ = ["NetworkInitializationInput"]
