"""Pure sampling rules shared by native and functional Recorders."""

from __future__ import annotations

import torch


def _recorder_stride(record_dt, step_dt):
    """Retain the native recording interval's integer conversion semantics."""
    return None if record_dt is None else int(record_dt / step_dt)


def _recorder_should_save(counter, stride):
    """Test the counter before its accepted-step increment."""
    return True if stride is None else counter % stride == 0


def _recorder_indices(counter_end: torch.Tensor, size: int, stride):
    """Select a segment's dense emissions using its final accepted-step count.

    A fresh segment contains one initial frame before its post-step emissions.
    Its candidate counter is -1, so that boundary is retained independently of
    the stride. Resumed segments contain only post-step emissions and continue
    the counter supplied by their caller.
    """
    indices = torch.arange(size, device=counter_end.device, dtype=torch.long)
    if stride is None:
        return indices
    counters = indices + counter_end - size
    selected = (counters < 0) | _recorder_should_save(counters, stride)
    return indices[selected]


__all__ = []
