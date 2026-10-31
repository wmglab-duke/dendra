"""Host lifecycle for pure callbacks observing a mutable native model."""

from __future__ import annotations

from dataclasses import dataclass

from ._callbacks import (
    FunctionalCallbacks,
    _collect_callback_emissions,
    _complete_callback_results,
    _host_callback_parts,
    _setup_callback_state,
)


@dataclass(frozen=True)
class _NativeCallbackSession:
    callbacks: FunctionalCallbacks
    binding: object
    initial_callback_state: object

    def validate_dt(self, dt):
        self.callbacks._validate_runtime_dt(dt)

    def _snapshot(self):
        from dendra.models.core import _clone_nested_state

        # Native buffers can be mutated by a later step or checkpoint replay.
        # Own the frame before callbacks retain views or save operands for AD.
        return _clone_nested_state(self.binding.state())

    def start(self):
        model_state = self._snapshot() if self.initial_callback_state is None else None
        state, parts = _setup_callback_state(
            self.callbacks, self.initial_callback_state, model_state
        )
        return state, _host_callback_parts(self.callbacks, state, parts)

    def update(self, callback_state):
        model_state = self._snapshot()
        return self.callbacks._update(
            callback_state,
            model_state,
            self.binding.auxiliary(model_state),
        )

    def stack(self, emissions):
        return self.callbacks._stack(emissions)

    def collect(self, state, parts, stacked_emitted):
        return _collect_callback_emissions(
            self.callbacks, state, parts, stacked_emitted
        )

    def finish(self, state, parts):
        return _complete_callback_results(self.callbacks, state, parts)


def _bind_native_callbacks(population, callbacks, callback_state):
    if callbacks is None:
        if callback_state is not None:
            raise ValueError("functional_callback_state requires functional_callbacks")
        return None
    if not isinstance(callbacks, FunctionalCallbacks):
        raise TypeError(
            "functional_callbacks must be a FunctionalCallbacks plan "
            "created by functional.make_callbacks(...)"
        )
    return _NativeCallbackSession(
        callbacks, callbacks._bind_imperative(population), callback_state
    )


__all__ = []
