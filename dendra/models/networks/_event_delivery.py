"""Canonical dense event transport for NetCon adapters.

The public tensor functions are pure and retain gradients through gates,
weights, and already queued values. Detection, scheduling, receiver invocation,
and parameter materialization belong to the caller. In particular, read and
deliver the due row before detecting or gathering this step's source gates.

Private write helpers also accept caller-owned buffers so imperative inference
can reuse its large delay rings without copying them. Both ownership paths use
the same routing and scatter operations.
"""

from typing import NamedTuple

import torch


class DenseDue(NamedTuple):
    payload: torch.Tensor
    event_counts: torch.Tensor | None


class DenseTransition(NamedTuple):
    queue: torch.Tensor
    slot: torch.Tensor
    global_step: torch.Tensor
    event_queue: torch.Tensor | None


def dense_due(queue, slot, event_queue=None):
    """Read today's target payload and optional per-connection event counts.

    ``queue`` is ``[depth, target_numel]`` and ``slot`` is a one-element Long
    Tensor. An optional int32 event queue is ``[depth, connection_numel]``.
    The returned rows have independent gathered storage, so clearing the input
    ring later cannot change a payload retained by a receiver.
    """
    payload = queue.index_select(0, slot).squeeze(0)
    events = (
        None if event_queue is None else event_queue.index_select(0, slot).squeeze(0)
    )
    return DenseDue(payload, events)


def gather_source_gates(source_gates, pre_idx):
    """Gather raw source gates in connection order without changing values.

    Visible source axes include any explicit Population batch dimensions.
    Hidden vmap lanes remain hidden. Signed/fractional gates and their
    gradients are retained; thresholding and surrogate policies are separate.
    """
    return source_gates.reshape(source_gates.numel()).index_select(0, pre_idx)


def connection_event_counts(intrinsic_gates, scheduled_counts=None):
    """Count positive intrinsic gates and independently scheduled events."""
    counts = (intrinsic_gates > 0).to(torch.int32)
    return counts if scheduled_counts is None else counts + scheduled_counts


def _fixed_delay_indices(
    slot,
    post_idx,
    delay_steps,
    *,
    depth,
    width,
    flat_offsets=None,
    uniform_delay=None,
):
    """Resolve one destination per connection, preserving static fast paths."""
    if uniform_delay is not None:
        future = (slot + uniform_delay).remainder(depth)
        return future * width + post_idx
    if flat_offsets is not None:
        return (slot * width + flat_offsets).remainder(depth * width)
    future = (slot + delay_steps).remainder(depth)
    return future * width + post_idx


def _clear_dense_slot(queue, slot, *, inplace=False):
    """Clear a delivered row, optionally donating caller-owned storage."""
    if inplace:
        return queue.index_fill_(0, slot, 0)
    # index_fill's mapped-index batching rule loses the row dimension for an
    # empty hidden lane. A [depth, 1] row predicate expresses the same clear
    # without a full-size mask and remains valid for mapped or unmapped state.
    rows = torch.arange(queue.shape[0], device=queue.device).reshape(queue.shape[0], 1)
    return torch.where(rows == slot.reshape(1, 1), 0, queue)


def _enqueue_dense_values(queue, destination, values, *, inplace=False):
    """Scatter connection values into a ring using one canonical operation."""
    if inplace:
        # Owned inference requires flattenable storage. reshape could allocate
        # a copy for a restored noncontiguous view and silently discard writes.
        flat = queue.view(queue.numel())
        flat.scatter_add_(0, destination.reshape(destination.numel()), values)
        return queue
    flat = queue.reshape(queue.numel())
    # scatter_add has a native mapped-index rule, including empty lanes;
    # index_add falls back to stacking one result per mapped index lane.
    return flat.scatter_add(
        0, destination.reshape(destination.numel()), values
    ).reshape(queue.shape)


def _enqueue_interpolated_values(queue, destination, values, *, inplace=False):
    """Retain native interpolation's indexed accumulation precision.

    On CPU, scatter_add promotes float16/bfloat16 collision sums whereas the
    previous index_add rounded each addition to the queue dtype. index_put
    preserves that policy and handles empty mapped-index lanes under vmap.
    """
    if queue.device.type != "cpu" or queue.dtype not in (torch.float16, torch.bfloat16):
        return _enqueue_dense_values(queue, destination, values, inplace=inplace)
    flat = queue.view(queue.numel()) if inplace else queue.reshape(queue.numel())
    indices = (destination.reshape(destination.numel()),)
    if inplace:
        flat.index_put_(indices, values, accumulate=True)
        return queue
    return flat.index_put(indices, values, accumulate=True).reshape(queue.shape)


def dense_fixed_delay_transition(
    queue,
    slot,
    global_step,
    weights,
    connection_gates,
    *,
    post_idx,
    delay_steps,
    depth,
    width,
    event_queue=None,
    event_counts=None,
    _clone_once=False,
):
    """Clear today's row, enqueue this step's gates, and advance counters.

    ``weights``, ``connection_gates``, ``post_idx``, and ``delay_steps`` have
    one entry per connection. Delays are the already prepared integer steps
    (at least one), not physical milliseconds. ``depth`` and ``width`` are
    static queue dimensions. Multiplication happens at enqueue time: later
    weight changes do not rescale queued events.

    Tracking requires both ``event_queue`` and int32 ``event_counts``. Counts
    describe events independently of amplitude, including zero-weight events.
    Returned queue/counters retain the input schemas and no input is mutated.
    By default all writes are out of place, including when only weights or
    gates acquire a vmap lane while the input queue and pointers are unmapped.

    ``_clone_once`` is a private storage policy for native imperative training:
    clone each ring once, then write only into that owned output. It preserves
    the same pure input/output contract and gradients while avoiding a second
    ring allocation. Callers with hidden vmap lanes use the default policy,
    which permits independently mapped gates, weights, queues, and pointers.
    """
    if (event_queue is None) != (event_counts is None):
        raise ValueError("Event tracking requires both event_queue and event_counts.")

    destination = _fixed_delay_indices(
        slot, post_idx, delay_steps, depth=depth, width=width
    )
    output = (
        queue.clone(memory_format=torch.contiguous_format) if _clone_once else queue
    )
    cleared = _clear_dense_slot(output, slot, inplace=_clone_once)
    next_queue = _enqueue_dense_values(
        cleared, destination, weights * connection_gates, inplace=_clone_once
    )
    next_events = None
    if event_queue is not None:
        connection_idx = torch.arange(
            post_idx.numel(), device=post_idx.device, dtype=torch.long
        )
        event_destination = _fixed_delay_indices(
            slot,
            connection_idx,
            delay_steps,
            depth=depth,
            width=connection_idx.numel(),
        )
        output_events = (
            event_queue.clone(memory_format=torch.contiguous_format)
            if _clone_once
            else event_queue
        )
        cleared_events = _clear_dense_slot(output_events, slot, inplace=_clone_once)
        next_events = _enqueue_dense_values(
            cleared_events, event_destination, event_counts, inplace=_clone_once
        )
    return DenseTransition(
        next_queue,
        (slot + 1).remainder(depth),
        global_step + 1,
        next_events,
    )


def delay_interpolation(delays, dt, *, dtype, taps, sigma):
    """Return native delay offsets and linear or soft tap coefficients.

    Two-tap callers receive the lower integer offset and upper-tap fraction.
    Three-tap callers receive all three offsets and normalized coefficients.
    Delivery-time history reads and enqueue-time dense writes share this math
    while retaining their separate weighting and accumulation order.
    """
    lam = (delays.to(dtype) / dt.to(dtype).reshape(1)).clamp_min(1.0)
    lower = torch.floor(lam)
    lower_index = lower.to(torch.long)
    if taps == 2:
        return lower_index, lam - lower
    offsets = torch.stack([lower_index - 1, lower_index, lower_index + 1], dim=-1)
    centers = offsets.to(dtype)
    fractions = torch.softmax(
        -0.5 * ((lam.unsqueeze(-1) - centers) / (sigma + 1e-6)) ** 2,
        dim=-1,
    )
    return offsets, fractions


def dense_interpolated_delay_transition(
    queue,
    slot,
    global_step,
    weights,
    connection_gates,
    delays,
    dt,
    *,
    post_idx,
    depth,
    width,
    taps,
    sigma,
    _clone_once=False,
):
    """Enqueue physical delays with native two- or three-tap interpolation.

    ``delays`` contains one physical delay per connection, in the same units
    as scalar ``dt``. Delay positions are clamped to at least one timestep,
    then split between adjacent steps (two taps) or normalized Gaussian
    weights centered on ``floor(position) + [-1, 0, 1]`` (three taps).
    ``sigma`` is measured in timesteps and retains the native ``1e-6``
    denominator offset.

    Today's row must be delivered before this transition clears it. Tap
    destinations wrap modulo ``depth``, including the native three-tap
    current-row contribution when ``floor(position) == 1``. Such a value is
    delivered when the ring next visits that row. Weights, gates, delays, and
    timestep are evaluated at enqueue time; subsequent changes never move or
    rescale pending values.

    The default path is pure and supports independently mapped inputs under
    vmap, including empty lanes. ``_clone_once`` shares the fixed-delay
    transition's private native storage policy: clone one contiguous output
    ring, then clear and scatter only into the owned clone. No input is
    mutated. Fractional delays do not define integer event counts, so the
    returned ``event_queue`` is always ``None``.
    """
    offsets, fractions = delay_interpolation(
        delays, dt, dtype=queue.dtype, taps=taps, sigma=sigma
    )
    weighted_gates = weights * connection_gates
    output = (
        queue.clone(memory_format=torch.contiguous_format) if _clone_once else queue
    )
    next_queue = _clear_dense_slot(output, slot, inplace=_clone_once)
    if taps == 2:
        fraction = fractions
        row0 = (slot + offsets).remainder(depth)
        row1 = (row0 + 1).remainder(depth)
        next_queue = _enqueue_interpolated_values(
            next_queue,
            row0 * width + post_idx,
            weighted_gates * (1.0 - fraction),
            inplace=_clone_once,
        )
        next_queue = _enqueue_interpolated_values(
            next_queue,
            row1 * width + post_idx,
            weighted_gates * fraction,
            inplace=_clone_once,
        )
    else:
        rows = (slot + offsets).remainder(depth)
        destinations = rows * width + post_idx.unsqueeze(-1)
        values = weighted_gates.unsqueeze(-1) * fractions
        next_queue = _enqueue_interpolated_values(
            next_queue,
            destinations,
            values.reshape(values.numel()),
            inplace=_clone_once,
        )
    return DenseTransition(
        next_queue, (slot + 1).remainder(depth), global_step + 1, None
    )
