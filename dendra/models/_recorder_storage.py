"""Host-only HDF5 collection shared with the native Recorder writer."""

from __future__ import annotations

from bisect import bisect_right
from numbers import Integral

import torch


class _RecorderHDF5Session:
    """Own one file run while numerical Recorder carry remains explicit.

    ``positions`` contains the accepted-step positions of the selected rows
    relative to each appended span. An initial frame is appended separately
    with ``steps=0, positions=(0,)``. A span without selected rows still
    advances the cache counter. The counter includes the initial frame, as
    native Recorder flushing does, but never changes the Recorder's ``i`` or
    in-memory history.
    """

    def __init__(self, recorder):
        if not recorder.cache_with_hdf5 or recorder.queue is None:
            raise RuntimeError("Recorder HDF5 storage is not open.")
        if recorder._hdf5_session is not None or recorder._native_hdf5_active:
            raise RuntimeError("Recorder already has an active HDF5 recording session.")
        if recorder.save_count:
            raise RuntimeError("Recorder has an unfinished native HDF5 run.")
        if (
            isinstance(recorder.cache_every, bool)
            or not isinstance(recorder.cache_every, Integral)
            or recorder.cache_every <= 0
        ):
            raise ValueError("Recorder HDF5 cache interval must be a positive integer.")
        self.recorder = recorder
        self.states = tuple(recorder.states)
        self.path = recorder.hdf5_path
        self.queue = recorder.queue
        self.cache_every = recorder.cache_every
        self.run_number = recorder.run_number
        self._counter = 0
        self._pending = {state: [] for state in self.states}
        self._attempted_write = False
        self._active = True
        recorder._hdf5_session = self

    def _validate(self):
        recorder = self.recorder
        if not self._active or recorder._hdf5_session is not self:
            raise RuntimeError("Recorder HDF5 recording session is closed.")
        if (
            not recorder.cache_with_hdf5
            or recorder.hdf5_path != self.path
            or recorder.queue is not self.queue
            or recorder.cache_every != self.cache_every
            or tuple(recorder.states) != self.states
            or recorder.run_number != self.run_number
        ):
            raise RuntimeError("Recorder HDF5 storage changed during execution.")

    def append(self, values, *, steps, positions):
        self._validate()
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
            raise ValueError("Recorder HDF5 span steps must be a non-negative integer.")
        positions = tuple(positions)
        initial = (
            steps == 0
            and len(positions) == 1
            and not isinstance(positions[0], bool)
            and isinstance(positions[0], Integral)
            and positions[0] == 0
        )
        if initial:
            if self._counter:
                raise ValueError(
                    "Recorder HDF5 initial frame must precede accepted steps."
                )
        elif any(
            isinstance(position, bool)
            or not isinstance(position, Integral)
            or position < 1
            or position > steps
            for position in positions
        ) or any(left >= right for left, right in zip(positions, positions[1:])):
            raise ValueError(
                "Recorder HDF5 sample positions must increase within the span."
            )
        if set(values) != set(self.states):
            raise ValueError("Recorder HDF5 frame must contain every configured state.")
        for state in self.states:
            value = values[state]
            if (
                not torch.is_tensor(value)
                or value.ndim < 1
                or value.shape[0] != len(positions)
            ):
                raise ValueError(
                    "Recorder HDF5 frame rows must match its sample positions."
                )

        def retain(start, stop):
            if start != stop:
                for state in self.states:
                    self._pending[state].append(
                        values[state].detach()[start:stop].clone()
                    )

        if initial:
            retain(0, 1)
            self._counter = 1
            return
        end = self._counter + int(steps)
        absolute = tuple(self._counter + int(position) for position in positions)
        start = 0
        first_boundary = (self._counter // self.cache_every + 1) * self.cache_every
        for boundary in range(first_boundary, end + 1, self.cache_every):
            stop = bisect_right(absolute, boundary)
            retain(start, stop)
            self._flush()
            start = stop
        retain(start, len(positions))
        self._counter = end

    def _flush(self):
        if not self.states or not self._pending[self.states[0]]:
            return
        values = {
            state: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            for state, parts in self._pending.items()
        }
        # Reserve this file identity even if enqueue fails after writing only
        # part of a frame; a later run must never reuse those dataset names.
        self._attempted_write = True
        self.recorder._write_hdf5_frame(values)
        self._pending = {state: [] for state in self.states}

    def finish(self):
        self._validate()
        try:
            self._flush()
        except BaseException:
            self.abort()
            raise
        self._release(advance=self._attempted_write)

    def abort(self):
        """Release unwritten samples without reusing an attempted file run."""
        if self._active:
            self._release(advance=self._attempted_write)

    def _release(self, *, advance):
        if self.recorder._hdf5_session is self:
            if advance:
                self.recorder.run_number = self.run_number + 1
            self.recorder.save_count = 0
            self.recorder._hdf5_session = None
        self._pending.clear()
        self._active = False


__all__ = []
