"""Lazy CPU two-tap enqueue with a PyTorch adjoint."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

import torch

from ._two_tap_enqueue_ext import load_cpu_two_tap_enqueue

_OP_NAME = "dendra_native::ordered_two_tap_enqueue"
_SOURCE_OP_NAME = f"{_OP_NAME}_from_source"
_op = None
_registration_lock = Lock()
_ordered_vjp_sources = ContextVar("ordered_two_tap_vjp_sources", default=None)


@contextmanager
def _compile_two_tap_vjp(ordered_sources):
    """Enable the fused VJP during the local scan's autograd capture."""
    token = _ordered_vjp_sources.set(ordered_sources)
    try:
        yield
    finally:
        _ordered_vjp_sources.reset(token)


def _fake_enqueue(queue, slot, weights, gates, fraction, lower, upper, pre_idx=None):
    return torch.empty_like(queue, memory_format=torch.contiguous_format)


def _fake_read_row(queue, slot, advanced):
    return queue.new_empty((slot.numel(), *queue.shape[1:]))


def _fake_two_tap_gather(ring, offset, lower, upper):
    return ring.new_empty(lower.shape), ring.new_empty(upper.shape)


def _fake_two_tap_vjp(
    ring, offset, weights, gates, fraction, lower, upper, source_indices=None
):
    return tuple(ring.new_empty(value.shape) for value in (weights, gates, fraction))


def _fake_mutation(*args, **kwargs):
    return None


def _setup_context(ctx, inputs, output):
    # The queue transition is linear in its input queue. Only the active slot
    # and contribution inputs are needed for the adjoint.
    ctx.save_for_backward(*inputs[1:])


def _can_fuse_vjp(ring, slot, weights, gates, fraction, lower, upper, pre_idx):
    ordered_sources = _ordered_vjp_sources.get()
    if (
        ordered_sources is None
        or (pre_idx is not None and not ordered_sources)
        or not torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or ring.device.type != "cpu"
        or ring.dtype not in (torch.float32, torch.float64)
        or ring.ndim != 2
        or not all(type(size) is int and size > 0 for size in ring.shape)
    ):
        return False

    real = (weights, gates, fraction)
    indices = (slot, lower, upper) if pre_idx is None else (slot, lower, upper, pre_idx)
    if not all(
        value.device == ring.device
        and value.layout is torch.strided
        and value.is_contiguous()
        and not value.is_neg()
        and not torch._is_zerotensor(value)
        for value in (ring, *real, *indices)
    ):
        return False
    if not all(
        value.ndim == 1 and type(value.shape[0]) is int for value in (*real, *indices)
    ):
        return False
    return (
        all(value.dtype == ring.dtype for value in real)
        and all(value.dtype == torch.int64 for value in indices)
        and slot.shape == (1,)
        and weights.shape == fraction.shape == lower.shape == upper.shape
        and weights.shape == (gates if pre_idx is None else pre_idx).shape
    )


def _backward(ctx, grad_output):
    slot, weights, gates, fraction, lower, upper, *source_indices = ctx.saved_tensors
    pre_idx = source_indices[0] if source_indices else None
    needs_queue, _, needs_weights, needs_gates, needs_fraction = ctx.needs_input_grad[
        :5
    ]
    grad_queue = grad_output.index_fill(0, slot, 0) if needs_queue else None
    grad_weights = grad_gates = grad_fraction = None
    fused = (
        needs_weights
        and needs_gates
        and needs_fraction
        and _can_fuse_vjp(
            grad_output, slot, weights, gates, fraction, lower, upper, pre_idx
        )
    )
    if fused:
        vjp = torch.ops.dendra_native.delay_ring_two_tap_vjp.default
        grad_weights, grad_gates, grad_fraction = vjp(
            grad_output,
            slot * grad_output.shape[1],
            weights,
            gates,
            fraction,
            lower,
            upper,
            pre_idx,
        )
    elif needs_weights or needs_gates or needs_fraction:
        flat = grad_output.reshape(-1)
        slot_offset = slot * grad_output.shape[1]
        grad_lower = flat.index_select(0, (lower + slot_offset).remainder(flat.numel()))
        grad_upper = flat.index_select(0, (upper + slot_offset).remainder(flat.numel()))

        edge_gates = gates
        if pre_idx is not None and (needs_weights or needs_fraction):
            edge_gates = gates.index_select(0, pre_idx)

        if needs_weights or needs_gates:
            # Match the two products and the accumulation of their derivatives.
            grad_weighted = grad_upper * fraction + grad_lower * (1.0 - fraction)
            if needs_weights:
                grad_weights = grad_weighted * edge_gates
            if needs_gates:
                grad_gates = grad_weighted * weights
        if needs_fraction:
            weighted = weights * edge_gates
            grad_fraction = grad_upper * weighted - grad_lower * weighted
    if not fused and pre_idx is not None and needs_gates:
        # Use the same derivative as the gather at the original call site,
        # including the order of contributions from repeated source indices.
        grad_gates = torch.ops.aten.index_select_backward.default(
            grad_gates, list(gates.shape), 0, pre_idx
        )
    gradients = (grad_queue, None, grad_weights, grad_gates, grad_fraction, None, None)
    return gradients if pre_idx is None else (*gradients, None)


def get_cpu_two_tap_enqueue():
    """Load the dispatcher op and register its PyTorch adjoint once."""
    global _op
    if _op is None:
        with _registration_lock:
            if _op is None:
                load_cpu_two_tap_enqueue()
                for enqueue in (_OP_NAME, _SOURCE_OP_NAME):
                    torch.library.register_fake(enqueue, _fake_enqueue)
                    torch.library.register_fake(f"{enqueue}_", _fake_mutation)
                    torch.library.register_autograd(
                        enqueue, _backward, setup_context=_setup_context
                    )
                torch.library.register_fake(
                    "dendra_native::delay_ring_read_row", _fake_read_row
                )
                torch.library.register_fake(
                    "dendra_native::delay_ring_adjoint_update_", _fake_mutation
                )
                torch.library.register_fake(
                    "dendra_native::delay_ring_gather_two_tap", _fake_two_tap_gather
                )
                torch.library.register_fake(
                    "dendra_native::delay_ring_two_tap_vjp", _fake_two_tap_vjp
                )
                _op = torch.ops.dendra_native.ordered_two_tap_enqueue.default
    return _op
