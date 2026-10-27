"""Declarations for call-local stimulation on compiled host runners."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VarInput:
    """Declare a required call-time ``extra`` in :func:`dendra.func.compile`.

    Pass ``extra=VarInput()`` when compiling, then supply ``extra=`` on every
    invocation. A specification contains a field/Waveform pair or a sequence
    of contacts. Supported deterministic waveform recipes and contact counts
    may change between invocations. ``extra=None`` supplies zero stimulation.
    """


_MISSING_RUNTIME_EXTRA = object()
_DEFAULT_RUNTIME_EXTRA_CHUNK_STEPS = 64


def _runtime_extra_chunk_steps(chunk_steps, *, steps, enabled):
    """Bound runtime assembly separately from a runner's complete horizon."""
    if chunk_steps is not None:
        if (
            isinstance(chunk_steps, bool)
            or not isinstance(chunk_steps, int)
            or chunk_steps < 1
        ):
            raise ValueError("chunk_steps must be a positive integer")
        if not enabled:
            raise ValueError("chunk_steps requires extra=VarInput()")
    if not enabled:
        return steps
    return min(steps, chunk_steps or _DEFAULT_RUNTIME_EXTRA_CHUNK_STEPS)


__all__ = ["VarInput"]
