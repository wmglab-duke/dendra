"""Canonical dense fixed-delay event transport for NetCon adapters.

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
