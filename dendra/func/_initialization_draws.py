"""Canonical draw inputs for one ordered initialization boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from ._types import InitializationInput, RandomParameterDraw, StimulationDraw


class InitializationDraws(NamedTuple):
    """Replayable base draws and RNG carry for fresh initialization.

    Obtain a complete packet with :func:`dendra.func.sample_initialization_draws`
    and pass it to :func:`dendra.func.initialize` through ``draws=``. For a
    Population, ``random`` contains its RAND/NOISE packet and ``stimulation``
    its Poisson packet. For a Network those fields map Population names to
    packets or ``None``; ``netstim`` and ``connections`` contain its existing
    component draw packets. Generator frames retain their native publication
    boundaries, including intermediate frames for shared streams.

    Sampling follows the ordered initializer, including transforms and cache
    restoration. It changes neither the native source nor the supplied carry.
    Distribution parameters keep their ordinary gradients during replay.
    Runtime execution uses the separate :class:`dendra.func.RuntimeDraws`.
    """

    random: RandomParameterDraw | Mapping[str, RandomParameterDraw | None] | None = None
    stimulation: StimulationDraw | Mapping[str, StimulationDraw | None] | None = None
    netstim: object = None
    connections: object = None


def _normalize_initialization_draws(draws):
    if not isinstance(draws, InitializationDraws):
        raise TypeError("draws must be an InitializationDraws packet or None")
    from ._network_bindings import _canonicalize_mapping_tree

    return _canonicalize_mapping_tree(draws)[0]


def _population_draw_types(random, stimulation):
    if random is not None and not isinstance(random, RandomParameterDraw):
        raise TypeError(
            "initialization random draws must be a RandomParameterDraw or None"
        )
    if stimulation is not None and not isinstance(stimulation, StimulationDraw):
        raise TypeError(
            "initialization stimulation draws must be a StimulationDraw or None"
        )


def _legacy_draw_conflict():
    raise ValueError("draws cannot be combined with legacy initialization draw packets")


def _with_population_initialization_draws(inputs, draws, *, random_draw=None):
    if draws is None:
        return inputs
    draws = _normalize_initialization_draws(draws)
    if not isinstance(inputs, InitializationInput):
        raise TypeError("inputs must be an InitializationInput")
    if (
        inputs.random_draw is not None
        or inputs.stimulation is not None
        or random_draw is not None
    ):
        _legacy_draw_conflict()
    if draws.netstim is not None or draws.connections is not None:
        raise ValueError(
            "Population initialization does not accept NetStim or connection draws"
        )
    _population_draw_types(draws.random, draws.stimulation)
    return inputs._replace(random_draw=draws.random, stimulation=draws.stimulation)


def _with_network_initialization_draws(functional, inputs, draws, *, netstim_draw=None):
    if draws is None:
        return inputs
    from ._connection_sampling import ConnectionSamplingDraw
    from ._netstim import NetStimDraw
    from ._network_initialization import NetworkInitializationInput

    draws = _normalize_initialization_draws(draws)
    if not isinstance(inputs, NetworkInitializationInput):
        raise TypeError("inputs must be a NetworkInitializationInput")
    if not isinstance(inputs.populations, Mapping) or set(inputs.populations) != set(
        functional.populations
    ):
        raise ValueError("initialization populations must match the Network")
    if (
        netstim_draw is not None
        or inputs.netstim_draw is not None
        or inputs.connection_draw is not None
    ):
        _legacy_draw_conflict()
    for population in inputs.populations.values():
        if not isinstance(population, InitializationInput):
            raise TypeError(
                "initialization populations must contain InitializationInput packets"
            )
        if population.random_draw is not None or population.stimulation is not None:
            _legacy_draw_conflict()
    families = []
    for name, values in (("random", draws.random), ("stimulation", draws.stimulation)):
        if values is None:
            values = dict.fromkeys(functional.populations)
        elif not isinstance(values, Mapping) or set(values) != set(
            functional.populations
        ):
            raise ValueError(
                f"initialization {name} draws must match the Network populations"
            )
        families.append(values)
    random, stimulation = families
    for name in functional.populations:
        _population_draw_types(random[name], stimulation[name])
    if draws.netstim is not None and not isinstance(draws.netstim, NetStimDraw):
        raise TypeError("initialization NetStim draws must be a NetStimDraw or None")
    if draws.connections is not None and not isinstance(
        draws.connections, ConnectionSamplingDraw
    ):
        raise TypeError(
            "initialization connection draws must be a ConnectionSamplingDraw or None"
        )
    return inputs._replace(
        populations={
            name: value._replace(
                random_draw=random[name], stimulation=stimulation[name]
            )
            for name, value in inputs.populations.items()
        },
        netstim_draw=draws.netstim,
        connection_draw=draws.connections,
    )


__all__ = ["InitializationDraws"]
