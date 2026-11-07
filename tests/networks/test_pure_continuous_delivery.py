"""Independent routing and transform contracts for shared analog transport."""

from __future__ import annotations

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.networks._continuous_delivery import (
    continuous_clear,
    continuous_counters,
    continuous_due,
    continuous_enqueue,
    continuous_project,
    gather_continuous_source,
)

DTYPE = torch.float64
DEPTH, WIDTH = 5, 3
PARTITIONS = ((), (0, 0, 0), (2, 2, 2), (1, 2, 3), (0, 2, 0), (0, 1, 3))


def _metadata(delays, *, direct=False):
    delay = torch.tensor(delays, dtype=torch.long)
    post = (
        torch.arange(len(delays)) if direct else torch.tensor([0, 1, 0])[: len(delays)]
    )
    immediate = torch.nonzero(delay == 0).flatten()
    delayed = torch.nonzero(delay != 0).flatten()
    mask = torch.zeros(WIDTH, dtype=torch.bool)
    mask[post[immediate]] = True
    return delay, post, immediate, delayed, mask


def _step(queue, presence, slot, clock, weighted, metadata, *, direct=False):
    delays, post, immediate, delayed, immediate_mask = metadata
    payload = queue.new_zeros(WIDTH)
    mask = torch.zeros(WIDTH, dtype=torch.bool)
    if delayed.numel():
        payload, mask = continuous_due(queue, presence, slot)
        queue, presence = continuous_clear(queue, presence, slot)
    if immediate.numel():
        value = continuous_project(
            weighted.index_select(0, immediate),
            post.index_select(0, immediate),
            width=WIDTH,
            direct=direct,
        )
        payload = payload + value
        mask = mask | immediate_mask
    if delayed.numel():
        queue, presence = continuous_enqueue(
            queue,
            presence,
            slot,
            weighted,
            post_idx=post.index_select(0, delayed),
            delay_steps=delays.index_select(0, delayed),
            depth=DEPTH,
            width=WIDTH,
            connection_idx=delayed,
        )
    slot, clock = continuous_counters(
        slot,
        clock,
        depth=DEPTH,
        all_immediate=bool(immediate.numel() and not delayed.numel()),
    )
    return queue, presence, slot, clock, payload, mask


def _oracle(queue, presence, slot, clock, weighted, metadata):
    delays, post, _immediate, delayed, _mask = metadata
    queue, presence = queue.clone(), presence.clone()
    payload = queue.new_zeros(WIDTH)
    mask = torch.zeros(WIDTH, dtype=torch.bool)
    if len(delayed):
        payload = queue[slot].squeeze(0).clone()
        mask = presence[slot].squeeze(0).clone()
        queue[slot] = 0
        presence[slot] = False
    for edge in range(len(delays)):
        target, delay = int(post[edge]), int(delays[edge])
        if delay == 0:
            payload[target] = payload[target] + weighted[edge]
            mask[target] = True
        else:
            future = (int(slot) + delay) % DEPTH
            queue[future, target] = queue[future, target] + weighted[edge]
            presence[future, target] = True
    next_slot = slot if len(delays) and not len(delayed) else (slot + 1) % DEPTH
    return queue, presence, next_slot, clock + 1, payload, mask


def _assert_same(actual, expected):
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("delays", PARTITIONS)
def test_partitions_collisions_signed_zero_values_and_pending_gradients(delays):
    metadata = _metadata(delays)
    queue = torch.arange(DEPTH * WIDTH, dtype=DTYPE).reshape(DEPTH, WIDTH) / 8
    queue.requires_grad_()
    presence = queue.detach() > 0.5
    slot, clock = torch.tensor([3]), torch.tensor([7])
    weighted = torch.tensor([0.0, -2.0, 1.5], dtype=DTYPE)[: len(delays)]
    weighted.requires_grad_()
    original = tuple(
        value.clone() for value in (queue, presence, slot, clock, weighted)
    )
    actual = _step(queue, presence, slot, clock, weighted, metadata)
    expected = _oracle(queue, presence, slot, clock, weighted, metadata)
    _assert_same(actual, expected)
    _assert_same((queue, presence, slot, clock, weighted), original)
    coefficients = torch.arange(1, DEPTH * WIDTH + 1, dtype=DTYPE).reshape(DEPTH, WIDTH)
    actual_loss = (actual[0] * coefficients).sum() + actual[4].square().sum()
    expected_loss = (expected[0] * coefficients).sum() + expected[4].square().sum()
    actual_grads = torch.autograd.grad(
        actual_loss, (queue, weighted), allow_unused=True
    )
    expected_grads = torch.autograd.grad(
        expected_loss, (queue, weighted), allow_unused=True
    )
    for left, right in zip(actual_grads, expected_grads, strict=True):
        if right is None:
            assert left is None
        else:
            torch.testing.assert_close(left, right)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, DTYPE))
def test_collision_precision_matches_indexed_native_accumulation(dtype):
    # Every addition rounds in the destination dtype in the legacy CPU path.
    values = torch.tensor([2048.0, 1.0, -2048.0, 1.0], dtype=dtype)
    post = torch.zeros(4, dtype=torch.long)
    expected = torch.zeros(WIDTH, dtype=dtype).index_add_(0, post, values)
    torch.testing.assert_close(
        continuous_project(values, post, width=WIDTH), expected, rtol=0, atol=0
    )
    queue = torch.zeros((DEPTH, WIDTH), dtype=dtype)
    presence = torch.zeros_like(queue, dtype=torch.bool)
    queued, mask = continuous_enqueue(
        queue,
        presence,
        torch.tensor([0]),
        values,
        post_idx=post,
        delay_steps=torch.ones(4, dtype=torch.long),
        depth=DEPTH,
        width=WIDTH,
    )
    torch.testing.assert_close(queued[1], expected, rtol=0, atol=0)
    assert mask[1, 0] and not mask[0].any()


def test_direct_projection_keeps_identity_and_gather_preserves_fanout_gradients():
    source = torch.tensor([[1.5, -2.0]], dtype=DTYPE, requires_grad=True)
    selected = gather_continuous_source(source, torch.tensor([1, 0, 1]))
    projected = continuous_project(selected, torch.arange(3), width=3, direct=True)
    assert projected is selected
    torch.testing.assert_close(
        torch.autograd.grad(projected.sum(), source)[0], source.new_tensor([[1, 2]])
    )


@pytest.mark.parametrize("mapped", ("values", "queue", "slot", "presence"))
@pytest.mark.parametrize("lanes", (0, 3))
def test_vmap_independent_rings_values_and_pointers_including_empty(mapped, lanes):
    metadata = _metadata((0, 1, 3))
    queue = torch.arange(DEPTH * WIDTH, dtype=DTYPE).reshape(DEPTH, WIDTH) / 8
    presence = queue > 0.5
    slot, clock = torch.tensor([3]), torch.tensor([7])
    values = torch.tensor([0.0, -2.0, 1.5], dtype=DTYPE)
    args = [queue, presence, slot, values]
    index = {"queue": 0, "presence": 1, "slot": 2, "values": 3}[mapped]
    args[index] = args[index].expand(lanes, *args[index].shape).clone()
    dims = tuple(0 if position == index else None for position in range(4))

    def evaluate(q, p, s, v):
        return _step(q, p, s, clock, v, metadata)

    actual = torch.vmap(evaluate, in_dims=dims)(*args)
    for lane in range(lanes):
        operands = [
            value[lane] if dim == 0 else value for value, dim in zip(args, dims)
        ]
        expected = evaluate(*operands)
        _assert_same(tuple(value[lane] for value in actual), expected)
    assert all(value.shape[0] == lanes for value in actual)


def test_forward_reverse_ad_of_weighted_edges_and_pending_queue():
    metadata = _metadata((0, 1, 3))
    presence = torch.ones((DEPTH, WIDTH), dtype=torch.bool)

    def evaluate(queue, values):
        output = _step(
            queue, presence, torch.tensor([2]), torch.tensor([3]), values, metadata
        )
        return output[0], output[4]

    inputs = (
        torch.zeros((DEPTH, WIDTH), dtype=DTYPE),
        torch.tensor([0.0, -1.0, 2.0], dtype=DTYPE),
    )
    actual = torch.func.jacfwd(evaluate, argnums=(0, 1))(*inputs)
    expected = torch.func.jacrev(evaluate, argnums=(0, 1))(*inputs)
    for actual_part, expected_part in zip(actual, expected, strict=True):
        _assert_same(actual_part, expected_part)


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_fullgraph_transport_values_and_gradients_match_scalar_queue(backend):
    metadata = _metadata((0, 1, 3))
    queue = (
        torch.arange(DEPTH * WIDTH, dtype=DTYPE).reshape(DEPTH, WIDTH).requires_grad_()
    )
    values = torch.tensor([0.0, -2.0, 1.5], dtype=DTYPE, requires_grad=True)
    presence = queue.detach() > 0.5
    slot, clock = torch.tensor([3]), torch.tensor([7])

    def evaluate(q, v):
        return _step(q, presence, slot, clock, v, metadata)

    with torch_compiler_warning_context():
        options = {"compile_threads": 1} if backend == "inductor" else None
        actual = torch.compile(
            evaluate, backend=backend, fullgraph=True, options=options
        )(queue, values)
    expected = _oracle(queue, presence, slot, clock, values, metadata)
    _assert_same(actual, expected)
    actual_grads = torch.autograd.grad(
        actual[0].square().sum() + actual[4].sum(), (queue, values)
    )
    expected_grads = torch.autograd.grad(
        expected[0].square().sum() + expected[4].sum(), (queue, values)
    )
    _assert_same(actual_grads, expected_grads)


def test_native_owned_clear_keeps_due_rows_and_training_inputs_independent():
    queue = (
        torch.arange(DEPTH * WIDTH, dtype=DTYPE).reshape(DEPTH, WIDTH).T.contiguous().T
    )
    presence = queue > 0
    slot = torch.tensor([1])
    saved = queue.clone(), presence.clone()
    due = continuous_due(queue, presence, slot)
    next_queue, next_presence = continuous_clear(
        queue, presence, slot, _clone_once=True
    )
    _assert_same((queue, presence), saved)
    assert next_queue.is_contiguous() and next_presence.is_contiguous()
    assert next_queue.data_ptr() != queue.data_ptr()
    assert not next_queue[1].any() and not next_presence[1].any()
    continuous_clear(queue, presence, slot, _inplace=True)
    _assert_same(due, (saved[0][1], saved[1][1]))
