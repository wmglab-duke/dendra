"""Immutable declarations for same-Population tensor input bindings."""

import keyword
from dataclasses import dataclass


def _input_path(path, *, target=False):
    if not isinstance(path, str):
        raise TypeError("Input binding paths must be strings.")
    parts = tuple(path.split("."))
    valid = len(parts) == 4 and parts[1] == "DE"
    if not target:
        valid = valid or len(parts) == 2
    if not valid or any(
        not part.isidentifier() or keyword.iskeyword(part) or part.startswith("_")
        for part in parts
    ):
        expected = (
            "mechanism.DE.State.input"
            if target
            else "mechanism.field or mechanism.DE.State.field"
        )
        raise ValueError(f"Input binding path {path!r} must have form {expected}.")
    return parts


@dataclass(frozen=True)
class InputBinding:
    """A read-only source path routed to one declared nested State input."""

    target: str
    source: str

    def __post_init__(self):
        _input_path(self.target, target=True)
        _input_path(self.source)
