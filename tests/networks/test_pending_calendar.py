from copy import deepcopy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra.models.mod import expsyn
from dendra.models.networks._pending_calendar import (
    pending_calendar_append,
    pending_calendar_empty,
    pending_calendar_from_legacy,
    pending_calendar_pop,
    pending_calendar_to_legacy,
)
from dendra.models.networks.netcon import _restore_calendar_chunks

pytestmark = pytest.mark.cpu


def _empty(capacity=16, dtype=torch.float64):
    return pending_calendar_empty(capacity, device="cpu", dtype=dtype)


def _assert_same_state(left, right):
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def test_append_overflow_is_atomic_sticky_and_does_not_mutate_inputs():
    initial = _empty(3)
    original = tuple(value.clone() for value in initial)
    first = pending_calendar_append(
        initial, torch.tensor([2, 1]), torch.tensor([0, 1]), torch.tensor([4.0, 7.0])
    )
    _assert_same_state(initial, original)
    failed = pending_calendar_append(
        first, torch.tensor([1, 2]), torch.tensor([0, 0]), torch.tensor([8.0, 9.0])
    )
    assert failed.overflow.item()
    _assert_same_state(failed[:-1], first[:-1])
    later = pending_calendar_append(
        failed, torch.tensor([1]), torch.tensor([0]), torch.tensor([1.0])
    )
    _assert_same_state(later, failed)
    with pytest.raises(RuntimeError, match="overflow"):
        pending_calendar_to_legacy(failed)


def test_masked_nan_and_invalid_indices_never_enter_storage_or_payload():
    state = pending_calendar_append(
        _empty(4),
        torch.tensor([1, -999, 2, 999]),
        torch.tensor([0, -999, 1, 999]),
        torch.tensor([3.0, float("nan"), 5.0, float("inf")]),
        torch.tensor([True, False, True, False]),
    )
    assert state.size.item() == 2
    next_state, payload = pending_calendar_pop(state, torch.tensor([1]), width=2)
    torch.testing.assert_close(payload, torch.tensor([3.0, 0.0], dtype=torch.float64))
    assert next_state.size.item() == 1
    assert next_state.slots.tolist() == [2, 0, 0, 0]
    assert next_state.indices.tolist() == [1, 0, 0, 0]
    assert next_state.values.tolist() == [5.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_ordered_pop_matches_legacy_chunk_reduction_with_cancellation(dtype):
    generator = torch.Generator().manual_seed(31)
    chunks = []
    for _ in range(42):
        idx = torch.randint(0, 4, (7,), generator=generator)
        value = (torch.randn(7, generator=generator) * 7).to(dtype)
        chunks.append((idx, value))
    # Large positive/negative terms expose changes to accumulation order.
    chunks.extend(
        [(torch.tensor([0, 0, 0]), torch.tensor([1000.0, 0.25, -1000.0], dtype=dtype))]
    )
    state = pending_calendar_from_legacy({3: chunks}, device="cpu", dtype=dtype)
    base = torch.tensor([2.0, -3.0, 0.25, 8.0], dtype=dtype)
    expected = base.clone()
    for idx, value in chunks:
        expected.index_add_(0, idx, value)
    next_state, payload = pending_calendar_pop(state, 3, width=4, base=base)
    torch.testing.assert_close(payload, expected, rtol=0, atol=0)
    torch.testing.assert_close(base, torch.tensor([2.0, -3.0, 0.25, 8.0], dtype=dtype))
    assert next_state.size.item() == 0
    assert not bool(next_state.values.any())


def test_legacy_roundtrip_preserves_bucket_order_gradients_and_capacity_checks():
    values = torch.tensor([1.0, 3.0, 2.0, 4.0], requires_grad=True)
    state = pending_calendar_append(
        _empty(8), torch.tensor([1, 3, 2, 1]), torch.tensor([0, 0, 0, 0]), values
    )
    legacy = pending_calendar_to_legacy(state)
    assert list(legacy) == [1, 3, 2]
    assert legacy[1][0][1].tolist() == [1.0, 4.0]
    restored = pending_calendar_from_legacy(
        legacy, capacity=8, device="cpu", dtype=torch.float64
    )
    _, payload = pending_calendar_pop(restored, 1, width=1)
    payload.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="cannot hold"):
        pending_calendar_from_legacy(
            legacy, capacity=3, device="cpu", dtype=torch.float64
        )
    # Coarse cache translation merges buckets in their original dictionary
    # order, including the later contribution to the first live bucket.
    merged = _restore_calendar_chunks(
        legacy,
        old_dt=0.1,
        new_dt=1.0,
        depth=2,
        idx_device="cpu",
        idx_dtype=torch.long,
        val_device="cpu",
        val_dtype=torch.float64,
    )
    assert [value.item() for _, chunk in merged[0] for value in chunk] == [
        1.0,
        4.0,
        3.0,
        2.0,
    ]


def test_calendar_transitions_compile_vmap_and_preserve_value_gradients():
    slots, indices = torch.tensor([2, 1, 2]), torch.tensor([0, 1, 0])

    def transition(values, slot):
        state = pending_calendar_append(_empty(5), slots, indices, values)
        return pending_calendar_pop(state, slot, width=2)

    values = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], requires_grad=True)
    slots_to_pop = torch.tensor([2, 1])
    mapped = torch.vmap(transition)(values, slots_to_pop)
    compiled = torch.compile(torch.vmap(transition), backend="eager", fullgraph=True)
    actual = compiled(values, slots_to_pop)
    _assert_same_state(mapped[0], actual[0])
    torch.testing.assert_close(
        actual[1], torch.tensor([[8.0, 0.0], [0.0, 7.0]], dtype=torch.float64)
    )
    actual[1].sum().backward()
    torch.testing.assert_close(
        values.grad, torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    )


@pytest.mark.parametrize("lanes", [0, 2])
def test_mapped_calendar_indices_admit_empty_lanes_and_forward_derivatives(lanes):
    slots, indices = torch.tensor([1, 2]), torch.tensor([0, 1])

    def transition(values, slot):
        state = pending_calendar_append(_empty(4), slots, indices, values)
        return pending_calendar_pop(state, slot, width=2)[1]

    values = torch.ones((lanes, 2), dtype=torch.float64, requires_grad=True)
    selected = torch.ones(lanes, dtype=torch.long)
    mapped = torch.vmap(transition)
    result, tangent = torch.func.jvp(
        lambda value: mapped(value, selected), (values,), (torch.ones_like(values),)
    )
    assert result.shape == (lanes, 2)
    torch.testing.assert_close(tangent, result)
    result.sum().backward()
    assert values.grad.shape == values.shape


def test_zero_capacity_and_empty_candidates_are_valid():
    state = _empty(0)
    state = pending_calendar_append(
        state,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
    )
    next_state, payload = pending_calendar_pop(state, 0, width=0)
    _assert_same_state(next_state, state)
    assert payload.shape == (0,)


def _native():
    post = dn.Population(N=1, C=2, dtype=torch.float64)
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    net = dn.Network(
        {"post": post},
        netstim=dn.NetStim(N=2),
        track_netcon_events=True,
        netcon_delay_backend="sparse_calendar",
    )
    net.connect_one_to_one(
        net.netstim[:], post[:], post.mech.syn, threshold=None, weight=1.0, delay=0.3
    )
    net.build(0.1)
    net.init_synapses()
    return next(iter(net.synapses.values()))


def test_native_calendar_grows_and_matches_legacy_order_after_many_appends():
    native = _native()
    oracle = {}
    for step in range(42):
        slots = torch.tensor([3, 1, 2, 1])
        indices = torch.tensor([0, 0, 1, 0])
        values = torch.tensor([1.0, 1000.0, step + 0.25, -1000.0], dtype=torch.float64)
        native._append_sparse_payloads(slots, indices, values)
        for slot in torch.unique(slots).tolist():
            mask = slots == slot
            oracle.setdefault(slot, []).append((indices[mask], values[mask]))
    assert native._pending_calendar.size.item() == 168
    assert native._pending_calendar.values.numel() >= 168
    assert not native._pending_calendar.overflow.item()
    assert list(native._sparse_calendar) == list(oracle)
    for slot in oracle:
        scratch = torch.tensor([0.25, -3.0], dtype=torch.float64)
        expected = scratch.clone()
        for idx, value in oracle[slot]:
            expected.index_add_(0, idx, value)
        native._add_sparse_calendar_to_dense(slot, scratch)
        torch.testing.assert_close(scratch, expected, rtol=0, atol=0)
    assert not native._sparse_calendar


def test_native_legacy_mutation_deepcopy_and_separate_counts():
    native = _native()
    idx = torch.tensor([0, 0])
    native._sparse_calendar = {
        2: [(idx, torch.tensor([2.0, -2.0], dtype=torch.float64))]
    }
    native._sparse_event_calendar[2] = [
        (torch.tensor([0, 1]), torch.tensor([1, 1], dtype=torch.int32))
    ]
    copied = deepcopy(native._sparse_calendar)
    copied[2][0][1].zero_()
    assert native._sparse_calendar[2][0][1].tolist() == [2.0, -2.0]
    payload = native._pop_sparse_calendar_to_dense(2)
    native._pop_sparse_events(2)
    assert payload.tolist() == [0.0, 0.0]
    assert native.events.tolist() == [1, 1]
    native._sparse_calendar[1] = [(torch.tensor([0]), torch.tensor([7.0]))]
    native._sparse_calendar.clear()
    assert not native._sparse_calendar


def test_native_tensor_bridge_deepcopy_and_dtype_movement():
    native = _native()
    state = pending_calendar_append(
        _empty(8),
        torch.tensor([1, 3, 2, 1]),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=torch.float64),
    )
    native._sparse_calendar.replace_tensor_state(state)
    assert native._sparse_calendar.tensor_state is state
    copied = deepcopy(native)
    copied._sparse_calendar.clear()
    assert native._sparse_calendar
    assert not copied._sparse_calendar
    native._sparse_event_calendar[2] = [
        (torch.tensor([0]), torch.tensor([1], dtype=torch.int32))
    ]
    native.dtype = torch.float32
    native._move_sparse_calendar_to_device()
    assert native._pending_calendar.values.dtype == torch.float32
    assert native._pending_event_calendar.values.dtype == torch.int32
    assert native._pending_calendar.slots.dtype == torch.long
    assert native._pending_calendar.slots[:4].tolist() == [1, 3, 2, 1]


def test_native_idle_calendars_avoid_device_scalar_reads_after_restore_and_drain():
    class RejectScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("Idle calendar read a device scalar")
            return func(*args, **(kwargs or {}))

    native = _native()

    def check_idle():
        with RejectScalarRead():
            scratch = native._pop_sparse_calendar_to_dense(0)
            native._pop_sparse_events(0)
            native._add_sparse_calendar_to_dense(0, scratch)
            assert not native._sparse_calendar
            assert not native._sparse_event_calendar

    check_idle()
    native._append_sparse_payloads(
        torch.tensor([2, 2]), torch.tensor([0, 1]), torch.tensor([1.0, 2.0])
    )
    native._append_sparse_events(
        torch.tensor([2]), torch.tensor([1]), torch.tensor([1], dtype=torch.int32)
    )
    assert native._pending_calendar_size == 2
    assert native._pending_event_calendar_size == 1
    checkpoint = native.state_dict_for_checkpoint()
    native._clear_sparse_calendar()
    check_idle()
    native.restore_dict_from_checkpoint(checkpoint)
    assert native._pending_calendar_size == 2
    assert native._pending_event_calendar_size == 1
    native._pop_sparse_calendar_to_dense(2)
    native._pop_sparse_events(2)
    check_idle()
    native._sparse_calendar.replace_tensor_state(
        pending_calendar_append(
            _empty(2), torch.tensor([1]), torch.tensor([0]), torch.tensor([3.0])
        )
    )
    assert native._pending_calendar_size == 1
    native._add_sparse_calendar_to_dense(1, native.delivery_buffer.squeeze(0))
    check_idle()
