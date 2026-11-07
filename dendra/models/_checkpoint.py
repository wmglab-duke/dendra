"""Preserve live native model bindings around activation-checkpoint replay."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext


def _copy_state_containers(value):
    """Copy nested containers while retaining every captured Tensor binding."""
    if isinstance(value, Mapping):
        return {key: _copy_state_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_state_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_state_containers(item) for item in value)
    return value


class _PreserveModelStateDuringRecompute:
    """Reusable, reentrant context for first- and higher-order replay."""

    def __init__(self, model):
        self._model = model
        self._previous = []

    def __enter__(self):
        self._previous.append(
            _copy_state_containers(self._model.state_dict_for_checkpoint())
        )
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        previous = self._previous.pop()
        self._model.restore_dict_from_checkpoint(previous)
        return False


def _checkpoint_recompute_contexts(model):
    """Return forward/recompute contexts without reading live state eagerly."""
    return nullcontext(), _PreserveModelStateDuringRecompute(model)


__all__ = []
