"""Pure threshold arithmetic shared by native and functional callbacks."""

from __future__ import annotations

import torch

_COUNT_DTYPE = torch.int32


def _restore_count(value, *, label):
    """Migrate supported legacy counters without truncation or overflow."""
    if not torch.is_tensor(value):
        raise TypeError(f"{label} must be a count Tensor")
    if value.dtype == _COUNT_DTYPE:
        return value
    if value.dtype == torch.float32:
        valid = (
            torch.isfinite(value)
            & (value >= 0)
            & (value < 2**31)
            & (value == torch.trunc(value))
        )
    elif value.dtype == torch.int64:
        valid = (value >= 0) & (value <= torch.iinfo(_COUNT_DTYPE).max)
    else:
        raise TypeError(f"{label} must use int32 or legacy float32/int64 counts")
    if not bool(torch.all(valid)):
        raise ValueError(
            f"{label} must contain non-negative integral counts representable in int32"
        )
    return value.to(dtype=_COUNT_DTYPE)


def _anomaly_flags(previous, voltage):
    """Accumulate non-finite voltages over the compartment axis only."""
    return previous | ~torch.isfinite(voltage).all(dim=-1)


def threshold_crossing(below, voltage, threshold):
    """Return the next below-threshold latch and upward crossing mask."""
    above = voltage >= threshold
    return ~above, above & below


def accumulate_count(count, spikes):
    """Add crossings while retaining the caller's counter dtype."""
    return count + spikes.to(count.dtype)


def activity_from_counts(count, at_least, partition=None, *, total=False):
    """Reduce the final checked-node axis using caller-validated partitions."""

    def reduce(values):
        measure = values.sum(dim=-1) if total else torch.count_nonzero(values, dim=-1)
        return measure >= at_least

    if partition is None:
        return reduce(count)
    return torch.stack(
        [reduce(segment) for segment in torch.split(count, partition, dim=-1)],
        dim=-1,
    )


__all__ = []
