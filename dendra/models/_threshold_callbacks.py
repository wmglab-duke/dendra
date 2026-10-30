"""Pure threshold arithmetic shared by native and functional callbacks."""

from __future__ import annotations

import torch


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
