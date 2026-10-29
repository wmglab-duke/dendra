"""Bind Network observations into the existing functional callback machinery.

Qualified paths and Population scopes are cold binding adapters. Callback
updates, emissions, explicit carry, continuation tokens, and finalization use
the same operations as Population execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._callbacks import (
    LFP,
    AnomalyDetector,
    FunctionalCallbacks,
    Recorder,
    _ThresholdCallback,
)
from ._types import FunctionalizationError


@dataclass(frozen=True)
class _RecorderLeaf:
    public_path: tuple[str, ...]
    checkpoint_key: str


def _schema_paths(schema, path=()):
    if isinstance(schema, Mapping):
        for name, child in schema.items():
            yield from _schema_paths(child, (*path, name))
    elif schema is not None:
        yield path


class _NetworkRecorderLayout:
    """A cold Recorder naming adapter; it never executes a Network timestep."""

    def __init__(self, functional):
        self._functional = functional
        self.device = functional.device
        self._state_schema = functional._state_schema
        aliases = {}
        ambiguous = set()

        def add(name, path):
            if name in ambiguous:
                return
            previous = aliases.setdefault(name, path)
            if previous != path:
                aliases.pop(name)
                ambiguous.add(name)

        for path in _schema_paths(self._state_schema):
            add(".".join(path), path)
        add("t", ("clock", "t"))
        for name, population in functional.populations.items():
            for leaf in population._state_layout:
                path = ("populations", name, *leaf.public_path)
                add(f"{name}." + ".".join(leaf.public_path), path)
                if leaf.public_path[0] == "integrator":
                    add(f"{name}.{leaf.public_path[-1]}", path)
                if leaf.public_path == ("clock", "t"):
                    add(f"{name}.t", path)
                if leaf.checkpoint_key is not None:
                    add(f"{name}.{leaf.checkpoint_key}", path)
                    add(f"{name}.mech.{leaf.checkpoint_key}", path)
        self._state_layout = tuple(
            _RecorderLeaf(path, alias) for alias, path in aliases.items()
        )
        self.ambiguous = frozenset(ambiguous)

    @property
    def dt(self):
        return self._functional.dt


def _bind_network_region(functional, callback, paths):
    """Resolve one source Population, then reuse its physical Slice selector."""
    from dendra.models.slice import Slice

    if not isinstance(callback.region, Slice):
        raise FunctionalizationError(
            "Network Recorder region must be a physical Population Slice"
        )
    source = object.__getattribute__(callback.region, "root_model")
    owners = [
        (name, population)
        for name, population in functional.populations.items()
        if population._population is source
    ]
    if len(owners) != 1:
        raise FunctionalizationError(
            "Network Recorder region must belong to an exact source Population "
            "in this Network; select packed components through the packed "
            "Population's named Slices."
        )
    name, population = owners[0]
    prefix = ("populations", name)
    relative = []
    for state_name, path in zip(callback.states, paths, strict=True):
        if path == ("clock", "t"):
            relative.append(path)
        elif path[:2] == prefix:
            relative.append(path[2:])
        else:
            raise FunctionalizationError(
                f"Network Recorder region belongs to Population {name!r}; "
                f"{state_name!r} must select that same Population or the "
                "Network clock. Use separate Recorders for different Populations."
            )
    return callback._bind_region(population, tuple(relative))


def make_network_callbacks(functional, callbacks):
    """Bind qualified Recorders and pure custom callbacks to a Network plan.

    ``cell.v``, ``cell.mech.syn.g``, and global ``t`` are familiar qualified
    names. Fully qualified explicit state paths such as
    ``populations.cell.integrator.v`` are also accepted. The finalized result
    retains the selection strings supplied by the user.

    ``region`` selects a physical Slice of one exact source Population,
    including named components of a concatenated MultiPopulation. The qualified
    physical selections must belong to that Population; the Network or selected
    Population clock remains scalar. Bind Population-specific callbacks such
    as LFP, spike analysis, and anomaly detection with ``PopulationCallback``.
    Unscoped custom callbacks receive the complete Network state and auxiliary
    mapping; scoped ones receive the selected Population's values.
    """
    if not isinstance(callbacks, Mapping):
        raise TypeError(
            "callbacks must be a mapping of names to FunctionalCallback objects"
        )
    adapted = {}
    layout = None
    for name, callback in callbacks.items():
        if isinstance(callback, Recorder):
            if layout is None:
                layout = _NetworkRecorderLayout(functional)
            ambiguous = layout.ambiguous.intersection(callback.states)
            if ambiguous:
                raise FunctionalizationError(
                    f"Network Recorder selections {sorted(ambiguous)!r} are ambiguous; "
                    "use fully qualified explicit state paths."
                )
            if callback.region is None:
                adapted[name] = callback._bind(layout)
            else:
                paths = callback._resolve_paths(layout)
                region = _bind_network_region(functional, callback, paths)
                adapted[name] = callback._bind_paths(layout, paths, region)
        elif isinstance(callback, (LFP, _ThresholdCallback, AnomalyDetector)):
            raise FunctionalizationError(
                f"{type(callback).__name__} requires a Population-specific "
                "binding; wrap it in PopulationCallback(name, callback)."
            )
        else:
            adapted[name] = callback
    return FunctionalCallbacks(functional, adapted)


__all__ = ["make_network_callbacks"]
