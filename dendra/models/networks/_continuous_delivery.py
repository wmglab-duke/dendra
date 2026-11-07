"""Shared tensor transport for continuous projections.

Receiver calls and source observation belong to the caller: all-delayed
connections receive before sampling their source, while mixed connections
sample before delivering their combined immediate and queued payload. These
small phases let both execution adapters preserve that ordering.

All public tensor helpers are pure by default. Private ownership options let
the native adapter reuse inference buffers or clear and enqueue into one owned
training clone. Presence is independent of payload so a scheduled zero remains
a real delivery.
"""

from typing import NamedTuple

import torch

from ._event_delivery import (
    _clear_dense_slot,
    _enqueue_interpolated_values,
    _fixed_delay_indices,
)


class ContinuousDue(NamedTuple):
    payload: torch.Tensor
    mask: torch.Tensor


def gather_continuous_source(source, pre_idx):
    """Gather source values in edge order before transformation and weighting."""
    return source.reshape(source.numel()).index_select(0, pre_idx)


def continuous_due(queue, presence, slot):
    """Read independent payload and presence rows without changing the ring."""
    return ContinuousDue(
        queue.index_select(0, slot).squeeze(0),
        presence.index_select(0, slot).squeeze(0),
    )


def continuous_project(weighted, post_idx, *, width, direct=False):
    """Sum weighted edges within one projection into the target's flat shape.

    ``direct`` is a cold-validated identity route. Duplicate destinations sum
    even when the receiver later reduces different projections with min/max.
    The indexed low-precision CPU accumulation matches native ``index_add``.
    """
    if direct:
        return weighted
    queue = weighted.new_zeros((1, width))
    return _enqueue_interpolated_values(queue, post_idx, weighted).squeeze(0)


def continuous_clear(queue, presence, slot, *, _inplace=False, _clone_once=False):
    """Clear the due row, retaining both input rings under the default policy."""
    if _clone_once:
        queue = queue.clone(memory_format=torch.contiguous_format)
        presence = presence.clone(memory_format=torch.contiguous_format)
    owned = _inplace or _clone_once
    if owned:
        next_presence = presence.index_fill_(0, slot, False)
    else:
        rows = torch.arange(presence.shape[0], device=presence.device).reshape(
            presence.shape[0], 1
        )
        next_presence = torch.where(rows == slot.reshape(1, 1), False, presence)
    return (
        _clear_dense_slot(queue, slot, inplace=owned),
        next_presence,
    )


def continuous_enqueue(
    queue,
    presence,
    slot,
    weighted,
    *,
    post_idx,
    delay_steps,
    depth,
    width,
    connection_idx=None,
    uniform_delay=None,
    flat_offsets=None,
    _inplace=False,
):
    """Enqueue the delayed edges and set their independent presence bits.

    Delays and destinations describe only the delayed subset. If
    ``connection_idx`` is supplied, gather that subset from the full weighted
    edge vector; otherwise all weighted values are delayed. Integer delays
    wrap inside the fixed ring, including existing out-of-capacity policies.
    Immediate zero-delay edges must be projected and received separately.

    Optional uniform-delay and flat-offset metadata retain native routing
    specializations. Pure writes support independently mapped weights, rings,
    and pointers, including empty hidden vmap lanes.
    """
    values = (
        weighted if connection_idx is None else weighted.index_select(0, connection_idx)
    )
    destination = _fixed_delay_indices(
        slot,
        post_idx,
        delay_steps,
        depth=depth,
        width=width,
        flat_offsets=flat_offsets,
        uniform_delay=uniform_delay,
    )
    next_queue = _enqueue_interpolated_values(
        queue, destination, values, inplace=_inplace
    )
    flat_presence = (
        presence.view(presence.numel())
        if _inplace
        else presence.reshape(presence.numel())
    )
    destination = destination.reshape(destination.numel())
    if _inplace:
        flat_presence.index_fill_(0, destination, True)
        next_presence = presence
    else:
        # Boolean scatter_add is logical OR. Unlike index_fill's mapped-index
        # batching rule it also handles an empty mapped pointer dimension.
        next_presence = flat_presence.scatter_add(
            0, destination, torch.ones_like(values, dtype=torch.bool)
        ).reshape(presence.shape)
    return next_queue, next_presence


def continuous_counters(
    slot, global_step, *, depth, all_immediate=False, _inplace=False
):
    """Advance counters; an all-immediate projection leaves its slot untouched."""
    if _inplace:
        if not all_immediate:
            slot.add_(1).remainder_(depth)
        global_step.add_(1)
        return slot, global_step
    return (
        slot if all_immediate else (slot + 1).remainder(depth),
        global_step + 1,
    )
