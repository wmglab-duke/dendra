"""Pure sampling rules shared by native and functional Recorders."""

from __future__ import annotations

import torch


def _recorder_stride(record_dt, step_dt):
    """Retain the native recording interval's integer conversion semantics."""
    return None if record_dt is None else int(record_dt / step_dt)


def _recorder_should_save(counter, stride):
    """Test a sample index, with zero denoting the initial boundary."""
    return True if stride is None else counter % stride == 0


def _recorder_indices(counter_end: torch.Tensor, size: int, stride):
    """Select a segment's dense emissions using its final accepted-step count.

    A fresh segment contains the initial frame at accepted-step count zero.
    Later samples are retained at positive multiples of the stride. Resumed
    segments contain only post-step emissions and continue the counter supplied
    by their caller without repeating the initial boundary.
    """
    indices = torch.arange(size, device=counter_end.device, dtype=torch.long)
    if stride is None:
        return indices
    counters = indices + counter_end - size + 1
    selected = _recorder_should_save(counters, stride)
    return indices[selected]


__all__ = []
