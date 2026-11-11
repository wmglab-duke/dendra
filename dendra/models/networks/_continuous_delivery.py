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
    linear_delay_taps,
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


@torch.compiler.allow_in_graph
def _physical_delay_predicate(valid):
    """Expose all physical lanes of a nondifferentiable validation predicate."""
    functorch = torch._C._functorch
    while functorch.is_functorch_wrapped_tensor(valid):
        valid = functorch.get_unwrapped(valid)
    return valid


def continuous_delay_taps(delays, dt, *, max_delay, dtype):
    """Return lower offsets and upper fractions for bounded physical delays."""
    delays = delays.to(dtype)
    dt = dt.to(dtype)
    limit = dt.new_tensor(max_delay)
    # Parameter transforms can round an exact endpoint a few ulps outside it.
    # Snap these excursions back to the physical bounds before converting to steps.
    tolerance = 4 * torch.finfo(dtype).eps
    valid = (
        torch.isfinite(delays)
        & (delays >= dt - dt * tolerance)
        & (delays <= limit + limit * tolerance)
    ).all()
    message = (
        "Differentiable continuous delays must be finite and in [dt, max_delay_ms]."
    )
    if torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():
        torch._assert_async(_physical_delay_predicate(valid).all(), message)
    elif not bool(valid):
        raise ValueError(message)
    delays = torch.where(delays < dt, dt, delays)
    delays = torch.where(delays > limit, limit, delays)
    positions = delays / dt.reshape(1)
    # A numeric delay can round a few representable values away from
    # its integer position after parameter materialization and division.
    # Also bound the absolute correction so real low-precision fractions and
    # fractions at large offsets do not disappear.
    values = positions.detach()
    nearest = values.round()
    adjacent = torch.nextafter(nearest, values)
    correction = nearest - values
    distance = correction.abs()
    snap = (distance <= 4 * (nearest - adjacent).abs()) & (distance <= 1.0e-4)
    return linear_delay_taps(positions + torch.where(snap, correction, 0))


def continuous_enqueue_linear(
    queue,
    presence,
    slot,
    weighted,
    *,
    post_idx,
    delay_steps,
    fraction,
    depth,
    width,
    _inplace=False,
):
    """Enqueue weighted samples at the lower delay and the following step."""
    lower = (slot + delay_steps).remainder(depth)
    lower_destination = lower * width + post_idx
    upper_destination = (lower + 1).remainder(depth) * width + post_idx
    upper_active = fraction > 0
    # Keep the right derivative of a finite sample at an integer delay without
    # allowing an unused tap of an infinite sample to contaminate other traffic.
    upper_weighted = torch.where(upper_active | torch.isfinite(weighted), weighted, 0)
    queue = _enqueue_interpolated_values(
        queue, lower_destination, weighted * (1 - fraction), inplace=_inplace
    )
    queue = _enqueue_interpolated_values(
        queue, upper_destination, upper_weighted * fraction, inplace=_inplace
    )
    if _inplace:
        flat_presence = presence.view(presence.numel())
        flat_presence.index_fill_(0, lower_destination, True)
        flat_presence.scatter_add_(0, upper_destination, upper_active)
        return queue, presence
    flat_presence = presence.reshape(presence.numel()).scatter_add(
        0, lower_destination, torch.ones_like(lower_destination, dtype=torch.bool)
    )
    presence = flat_presence.scatter_add(0, upper_destination, upper_active).reshape(
        presence.shape
    )
    return queue, presence


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
