"""Shared cold entry points for Population and Network lowering."""

from __future__ import annotations

from dendra.models.core import Population, _validate_time_scalar
from dendra.models.networks.net import Network

from ._network import FunctionalNetwork, NetworkTensors
from ._population import FunctionalPopulation
from ._population import make_functional as _make_functional_population
from ._types import PopulationTensors


def _lower_network(network, *, dt=None, extra=None, pending_capacity=None):
    """Lower a typed Network and check an optional normalized timestep."""
    plan = FunctionalNetwork(network, extra=extra, pending_capacity=pending_capacity)
    tensors = plan.extract()
    if dt is not None and dt != plan.dt:
        raise ValueError(
            "dt must match the initialized Network timestep; initialize and compile again to change it"
        )
    return plan, tensors


def make_functional(
    population: Population | Network,
    *,
    dt: float,
    extra=None,
    pending_capacity=None,
    _bind_owned_cache=True,
    _retain_initialization_materializations=False,
    _network_stimulation_rng_scope=False,
) -> (
    tuple[FunctionalPopulation, PopulationTensors]
    | tuple[FunctionalNetwork, NetworkTensors]
):
    """Lower an initialized Population or Network into an explicit-state plan.

    ``dt`` is required. For a Network it must match the initialized timestep;
    lowering does not initialize the model or change its timestep. The returned
    pair is a :class:`FunctionalPopulation` and :class:`PopulationTensors`, or
    a :class:`FunctionalNetwork` and :class:`NetworkTensors`, respectively.

    Population extraction retains native raw parameter Tensor references.
    Network extraction owns detached snapshots and preserves shared parameter
    aliases. Both expose ``parameters``, ``constants`` and ``state`` for raw
    binding with ``plan.bind(parameters, constants=constants)`` and execution
    with :func:`run`. Lowering leaves native simulation state unchanged.

    ``extra`` binds extracellular stimulation: one ``(field, Waveform)`` pair
    or a sequence of contacts for a Population, or a mapping from Population
    names to those bindings for a Network. Shared field and waveform parameter
    aliases are preserved within the returned mapping.

    ``pending_capacity`` applies only to Network pending-event calendars. It is
    a positive entry count per calendar, or a connection-name mapping. Omitted
    capacities accommodate existing traffic and at least twice the edge count
    for active calendars. Overflow raises at host execution boundaries; retry
    from the input state with a larger-capacity plan.

    :func:`make_functional_network` is a compatibility entry point that uses the
    initialized Network timestep without a separate ``dt`` argument.
    """
    if isinstance(population, Network):
        if (
            _bind_owned_cache is not True
            or _retain_initialization_materializations is not False
            or _network_stimulation_rng_scope is not False
        ):
            raise TypeError(
                "Population-only lowering options cannot be used with a Network"
            )
        dt_value = _validate_time_scalar(dt, name="dt", positive=True)
        return _lower_network(
            population, dt=dt_value, extra=extra, pending_capacity=pending_capacity
        )
    if pending_capacity is not None:
        raise ValueError("pending_capacity applies only to Network lowering")
    return _make_functional_population(
        population,
        dt=dt,
        extra=extra,
        _bind_owned_cache=_bind_owned_cache,
        _retain_initialization_materializations=(
            _retain_initialization_materializations
        ),
        _network_stimulation_rng_scope=_network_stimulation_rng_scope,
    )


__all__ = ["make_functional"]
