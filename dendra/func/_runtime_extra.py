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


__all__ = ["VarInput"]
