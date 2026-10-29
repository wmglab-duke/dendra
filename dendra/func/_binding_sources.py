"""Construction-time selection of raw sources from explicit packets."""

from __future__ import annotations

_BINDING_UNSET = object()


def _packet_types():
    # Binding modules participate in Population/Network imports. Resolve these
    # only at construction, after the public types have been defined.
    from ._continuation import Continuation
    from ._network import NetworkTensors
    from ._types import PopulationTensors

    return PopulationTensors, NetworkTensors, Continuation


def _is_binding_packet(value):
    # Inspect the actual class, without invoking an authored __class__ getter.
    return issubclass(type(value), _packet_types())


def _binding_packet_sources(
    value, *, population, prepared=_BINDING_UNSET, constants=_BINDING_UNSET
):
    """Select only P/C; legacy inputs keep their existing construction path."""
    population_tensors, network_tensors, continuation = _packet_types()
    if not issubclass(type(value), (population_tensors, network_tensors, continuation)):
        return None
    if prepared is not _BINDING_UNSET or constants is not _BINDING_UNSET:
        raise TypeError("packet binding cannot be combined with prepared or constants")
    expected = population_tensors if population else network_tensors
    if type(value) not in (expected, continuation):
        owner = "Population" if population else "Network"
        raise TypeError(
            f"{owner} binding requires an exact {expected.__name__} or Continuation"
        )
    return value.parameters, value.constants


__all__ = []
