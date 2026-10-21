"""Bind Network observations into the existing functional callback machinery.

Only selection names differ from Population callbacks. Recorder updates,
emissions, explicit carry, continuation tokens, and finalization are the same
operations used by Population execution.
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


def make_network_callbacks(functional, callbacks):
    """Bind qualified Recorders and pure custom callbacks to a Network plan.

    ``cell.v``, ``cell.mech.syn.g``, and global ``t`` are familiar qualified
    names. Fully qualified explicit state paths such as
    ``populations.cell.integrator.v`` are also accepted. The finalized result
    retains the selection strings supplied by the user.

    Population-specific region, LFP, threshold, and anomaly callbacks require
    a population selection contract and are not yet bound at Network level.
    Custom callbacks receive the complete Network state and auxiliary mapping.
    """
    if not isinstance(callbacks, Mapping):
        raise TypeError(
            "callbacks must be a mapping of names to FunctionalCallback objects"
        )
    adapted = {}
    layout = None
    for name, callback in callbacks.items():
        if isinstance(callback, Recorder):
            if callback.region is not None:
                raise FunctionalizationError(
                    "Network Recorder region selections are not yet supported; "
                    "select qualified state names or node_indices instead."
                )
            if layout is None:
                layout = _NetworkRecorderLayout(functional)
            ambiguous = layout.ambiguous.intersection(callback.states)
            if ambiguous:
                raise FunctionalizationError(
                    f"Network Recorder selections {sorted(ambiguous)!r} are ambiguous; "
                    "use fully qualified explicit state paths."
                )
            adapted[name] = callback._bind(layout)
        elif isinstance(callback, (LFP, _ThresholdCallback, AnomalyDetector)):
            raise FunctionalizationError(
                f"{type(callback).__name__} requires a Population-specific "
                "binding; use a pure Network FunctionalCallback instead."
            )
        else:
            adapted[name] = callback
    return FunctionalCallbacks(functional, adapted)


__all__ = ["make_network_callbacks"]
