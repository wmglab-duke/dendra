"""Cold Population-scope validation shared by callback execution adapters."""

from __future__ import annotations


def _validate_population_callback_name(population):
    if not isinstance(population, str):
        raise TypeError("PopulationCallback population must be a string")
    if not population:
        raise ValueError("PopulationCallback population must not be empty")
    return population


def _population_callback_target(populations, population, *, error_type=ValueError):
    _validate_population_callback_name(population)
    if population not in populations:
        raise error_type(
            f"Unknown Network Population {population!r}; "
            f"available names are {tuple(populations)}"
        )
    return populations[population]


__all__ = []
