"""Independent checks for the native adapter around the shared heap calendar.

The ledger below deliberately uses Python heapq and the pre-calendar scalar
renewal formula rather than either of the shared tensor implementations. This
keeps native/functional parity from hiding a common error in consumption.
"""

from __future__ import annotations

import copy
import heapq
import math

import pytest
import torch
import torch.nn.functional as F

from dendra.models.networks.netstim import NetStim

DTYPE = torch.float64


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return copy.deepcopy(value)


def _assert_same(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            _assert_same(got, want)
    elif isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def _legacy_checkpoint(stim, heaps=None, *, head=None):
    """Produce a pre-calendar checkpoint, including its optional head-only form."""
    checkpoint = _clone(stim.state_dict_for_checkpoint())
    for key in tuple(checkpoint):
        if key.startswith("heap_"):
            del checkpoint[key]
    if head is not None:
        checkpoint.pop("sched_heaps", None)
        checkpoint["next_sched_time"] = torch.as_tensor(head, dtype=stim.dtype())
    elif heaps is not None:
        checkpoint["sched_heaps"] = tuple(tuple(row) for row in heaps)
    return checkpoint


def _pending(stim):
    # This compatibility view is cold: its purpose is to expose remaining events
    # in the legacy format, not to drive the runtime transition being tested.
    return [sorted(row) for row in stim._sched_heaps]


class _HeapLedger:
    """One scalar heap and differentiable scalar renewal clock per generator."""

    def __init__(self, stim, rows, *, clock=None, interval=None):
        self.shape = stim.shape
        self.dtype = stim.dtype()
        self.interval = (
            stim.interval().detach().clone() if interval is None else interval
        )
        self.clock = stim.next_stoch_time.detach().clone() if clock is None else clock
        self.noise = stim.noise.detach().clone()
        self.caps = stim.max_spikes.detach().clone()
        self.counts = stim.spike_counts.detach().clone()
        self.tau = stim.tau
        self.rows = [list(row) for row in rows]
        for row in self.rows:
            heapq.heapify(row)
        self.rng = torch.Generator()
        self.rng.set_state(stim.state_dict_for_checkpoint()["rng_state"])

    def advance(self, times, *, tensor_events=()):
        times = torch.broadcast_to(torch.as_tensor(times, dtype=self.dtype), self.shape)
        draws = (
            -torch.rand(self.shape, generator=self.rng, dtype=self.dtype)
            .clamp_min(torch.finfo(self.dtype).tiny)
            .log()
        )
        clocks, gates, spikes = [], [], []
        heads = []
        for flat, (time, clock, mean, noise, cap, count, draw) in enumerate(
            zip(
                times.reshape(-1),
                self.clock.reshape(-1),
                self.interval.reshape(-1),
                self.noise.reshape(-1),
                self.caps.reshape(-1),
                self.counts.reshape(-1),
                draws.reshape(-1),
            )
        ):
            row = self.rows[flat]
            head = time.new_tensor(row[0] if row else float("inf"))
            # This is the original scalar gate, with the minimum taken before
            # consuming the heap. A post-pop head changes its derivative.
            soft = torch.sigmoid((time - torch.minimum(clock, head)) / self.tau)
            hard = (time >= torch.minimum(clock, head)).to(self.dtype)
            legacy_gate = hard + (soft - soft.detach())
            allowed = bool(count < cap)
            stoch_due = bool(time >= clock.detach()) and allowed
            heap_due = bool(time >= head) and bool(torch.isfinite(head)) and allowed
            fired = stoch_due or heap_due
            gate = legacy_gate * allowed
            for target, event_time, amplitude, dt in tensor_events:
                if target == flat:
                    phase = (time - event_time) / dt
                    triangle = (1 - phase.abs()).clamp_min(0)
                    gate = gate + amplitude * triangle * allowed
                    fired = fired or (bool(triangle.detach() > 0) and allowed)
            clocks.append(
                clock
                + (mean * (1 - noise) + mean * noise * draw)
                * gate.new_tensor(stoch_due)
                * legacy_gate
                * allowed
            )
            gates.append(gate)
            spikes.append(fired)
            if heap_due:
                # Compare the original Python float after casting to simulation
                # dtype, exactly as the old adapter did in its while loop.
                while row and bool(time.new_tensor(row[0]) <= time):
                    heapq.heappop(row)
            heads.append(row[0] if row else float("inf"))
        self.clock = torch.stack(clocks).reshape(self.shape)
        spikes = torch.tensor(spikes).reshape(self.shape)
        self.counts = self.counts + spikes.long()
        return (
            spikes,
            torch.stack(gates).reshape(self.shape),
            times,
            times.new_tensor(heads).reshape(self.shape),
        )


def test_native_consumption_matches_independent_heapq_ledger_for_irregular_batch_times():
    stim = NetStim(
        N=2,
        interval=[[0.4, 0.6], [0.3, 0.5]],
        start=[[0.05, 0.15], [0.0, 0.2]],
        noise=[[0.0, 0.7], [0.5, 1.0]],
        max_spikes=[[6, 3], [2, 5]],
        seed=813,
        tau=0.2,
        dtype=DTYPE,
    ).initialize()
    rows = [[0.1, 0.1, 0.2, 0.9], [0.05, 0.4, 0.4], [0.2, 0.5], [0.1, 0.3, 1.2]]
    for flat, row in enumerate(rows):
        stim.schedule((flat // 2, flat % 2), row)
    ledger = _HeapLedger(stim, rows)
    # Repeat, skip, and move backwards independently in each batch. A call is
    # never deduplicated and never catches up more than one stochastic renewal.
    for per_batch in (
        [0.0, 0.1],
        [0.15, 0.4],
        [0.15, 0.4],
        [0.8, 0.2],
        [0.3, 1.4],
        [1.5, 1.4],
        [2.0, 2.5],
    ):
        times = torch.tensor(per_batch, dtype=DTYPE)
        want_spikes, want_gate, want_last, want_head = ledger.advance(times[:, None])
        stim(times, bptt=True)
        _assert_same(stim.spikes, want_spikes)
        torch.testing.assert_close(stim.spike_gate, want_gate)
        torch.testing.assert_close(stim.next_stoch_time, ledger.clock)
        _assert_same(stim.spike_counts, ledger.counts)
        _assert_same(stim.t_last, want_last)
        _assert_same(stim.next_sched_time, want_head)
        assert _pending(stim) == [sorted(row) for row in ledger.rows]
        _assert_same(
            stim.state_dict_for_checkpoint()["rng_state"], ledger.rng.get_state()
        )


def test_cap_blocks_consumption_and_reinitialization_preserves_remaining_events():
    stim = NetStim(start=100.0, interval=100.0, max_spikes=1, dtype=DTYPE).initialize()
    stim.schedule(0, [0.1, 0.1, 0.2, 0.3])
    assert bool(stim(0.1)[0])
    assert _pending(stim) == [[0.2, 0.3]]
    assert not bool(stim(0.7)[0])
    assert _pending(stim) == [[0.2, 0.3]]
    stim.initialize()
    assert _pending(stim) == [[0.2, 0.3]]
    assert not bool(stim(0.1)[0])
    assert bool(stim(0.7)[0])
    assert _pending(stim) == [[]]
    assert stim.spike_counts.tolist() == [1]


@pytest.mark.parametrize("heap_time", [0.1, 0.3, 0.5])
def test_native_mixed_schedule_clock_and_gradient_match_frozen_scalar_formula(
    heap_time,
):
    stim = (
        NetStim(
            N=2,
            interval=[0.7, 0.9],
            start=0.3,
            noise=[0.2, 0.6],
            tau=0.25,
            seed=43,
            dtype=DTYPE,
        )
        .unfreeze()
        .initialize()
    )
    native_clock = torch.tensor([0.3, 0.45], dtype=DTYPE, requires_grad=True)
    reference_clock = native_clock.detach().clone().requires_grad_()
    checkpoint = _legacy_checkpoint(stim, [[heap_time, heap_time, 0.8], [0.2, 0.6]])
    checkpoint["next_stoch_time"] = native_clock
    stim.restore_dict_from_checkpoint(checkpoint)
    native_time = torch.tensor([0.42], dtype=DTYPE, requires_grad=True)
    native_weight = torch.tensor([1.7], dtype=DTYPE, requires_grad=True)
    reference_time = native_time.detach().clone().requires_grad_()
    reference_weight = native_weight.detach().clone().requires_grad_()
    stim.schedule(0, native_time, weight=native_weight)
    reference_rho = stim.interval.rho.detach().clone().requires_grad_()
    ledger = _HeapLedger(
        stim,
        [[heap_time, heap_time, 0.8], [0.2, 0.6]],
        clock=reference_clock,
        interval=F.softplus(reference_rho, threshold=20.0),
    )
    native_loss, reference_loss = native_clock.new_zeros(()), native_clock.new_zeros(())
    for index, time in enumerate((0.35, 0.5, 0.5, 0.9)):
        want_spikes, want_gate, _, want_head = ledger.advance(
            time, tensor_events=((0, reference_time[0], reference_weight[0], 0.1),)
        )
        stim(time, bptt=True, dt=0.1)
        _assert_same(stim.spikes, want_spikes)
        torch.testing.assert_close(stim.spike_gate, want_gate)
        torch.testing.assert_close(stim.next_stoch_time, ledger.clock)
        _assert_same(stim.next_sched_time, want_head)
        native_loss = native_loss + (index + 1) * (
            stim.spike_gate.sum() + 0.3 * stim.next_stoch_time.sum()
        )
        reference_loss = reference_loss + (index + 1) * (
            want_gate.sum() + 0.3 * ledger.clock.sum()
        )
    got = torch.autograd.grad(
        native_loss, (native_clock, stim.interval.rho, native_time, native_weight)
    )
    want = torch.autograd.grad(
        reference_loss,
        (reference_clock, reference_rho, reference_time, reference_weight),
    )
    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual, expected)
    assert bool(got[2].abs().sum() > 0)
    assert bool(got[3].abs().sum() > 0)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("override", [0.2, 0.7])
def test_legacy_head_only_override_consumes_actual_heap_on_first_due_call(
    existing, override
):
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    if existing:
        stim.schedule(0, [0.4, 0.8])
    checkpoint = _legacy_checkpoint(stim, head=[override])
    stim.restore_dict_from_checkpoint(checkpoint)
    assert not bool(stim(override - 0.01)[0])
    assert stim.next_sched_time.item() == override
    assert bool(stim(override)[0])
    expected_pending = ([0.4, 0.8] if override < 0.4 else [0.8]) if existing else []
    assert _pending(stim) == [expected_pending]
    expected_head = expected_pending[0] if expected_pending else float("inf")
    assert stim.next_sched_time.item() == expected_head


def test_initialize_discards_legacy_head_override_but_preserves_real_pending_events():
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    stim.schedule(0, [0.4, 0.8])
    stim.restore_dict_from_checkpoint(_legacy_checkpoint(stim, head=[0.2]))
    stim.initialize()
    assert stim.next_sched_time.item() == 0.4
    assert not bool(stim(0.2)[0])


@pytest.mark.parametrize("through_parent", [False, True])
def test_calendar_retains_original_python_precision_through_module_dtype_changes(
    through_parent,
):
    first, second = 1.0 + 2**-30, 1.0 + 2**-29
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    stim.schedule(0, [first, second])
    module = torch.nn.ModuleList([stim]) if through_parent else stim
    module.float()
    assert _pending(stim) == [[first, second]]
    # Reinitialization rebuilds the native head, retaining the original event
    # precision rather than a calendar rounded by Module._apply.
    module.double()
    stim.initialize()
    assert stim.next_sched_time.item() == first
    assert not bool(stim(1.0)[0])
    assert bool(stim(first)[0])
    assert _pending(stim) == [[second]]
    restored = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    restored.restore_dict_from_checkpoint(_legacy_checkpoint(stim, [[second]]))
    assert restored.next_sched_time.item() == second


def test_float32_consumption_coalesces_distinct_float64_times_at_native_cast_boundary():
    first, second = 1.0 + 2**-30, 1.0 + 2**-29
    stim = NetStim(start=100.0, interval=100.0, dtype=torch.float32).initialize()
    stim.schedule(0, [first, second, 1.25])
    assert _pending(stim) == [[first, second, 1.25]]
    assert bool(stim(1.0)[0])
    assert _pending(stim) == [[1.25]]
    assert stim.next_sched_time.item() == 1.25


def test_float32_checkpoint_roundtrip_preserves_uncast_calendar_times():
    first, second = 1.0 + 2**-30, 1.0 + 2**-29
    stim = NetStim(start=100.0, interval=100.0, dtype=torch.float32).initialize()
    stim.schedule(0, [first, second])
    checkpoint = _clone(stim.state_dict_for_checkpoint())
    restored = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    restored.restore_dict_from_checkpoint(checkpoint)
    assert _pending(restored) == [[first, second]]
    assert restored.next_sched_time.item() == first
    assert not bool(restored(1.0)[0])
    assert bool(restored(first)[0])
    assert _pending(restored) == [[second]]


def test_cold_insert_clear_and_batch_only_rebuild_remaining_calendar_events():
    stim = NetStim(N=2, start=100.0, interval=100.0, dtype=DTYPE).initialize()
    stim.schedule(0, [0.1, 0.2, 0.8])
    stim.schedule(1, [0.1, 0.7])
    stim(0.25)
    assert _pending(stim) == [[0.8], [0.7]]
    # Default past filtering and an explicitly allowed overdue insertion retain
    # the legacy policy after part of the calendar has already been consumed.
    stim.schedule(0, [0.15, 0.5])
    stim.schedule(1, 0.15, allow_past=True)
    assert _pending(stim) == [[0.5, 0.8], [0.15, 0.7]]
    stim.batch(2)
    assert _pending(stim) == [[0.5, 0.8], [0.15, 0.7]] * 2
    stim.clear_schedule((1, 0))
    assert _pending(stim) == [[0.5, 0.8], [0.15, 0.7], [], [0.15, 0.7]]
    assert stim(torch.tensor([0.6, 0.8], dtype=DTYPE)).tolist() == [
        [True, True],
        [False, True],
    ]
    assert _pending(stim) == [[0.8], [0.7], [], []]


@pytest.mark.parametrize("bad_heaps", [(), ((0.1,), (0.2,)), ((object(),),)])
def test_invalid_legacy_restore_is_atomic_for_clock_calendar_counters_and_rng(
    bad_heaps,
):
    stim = NetStim(interval=0.5, noise=0.7, seed=91, dtype=DTYPE).initialize()
    stim.schedule(0, [0.1, 0.4, 0.6])
    stim(0.2)
    before = _clone(stim.state_dict_for_checkpoint())
    before_head = stim.next_sched_time.clone()
    before_spikes, before_gate = stim.spikes, stim.spike_gate
    corrupt = _legacy_checkpoint(stim, bad_heaps)
    corrupt["next_stoch_time"] = torch.tensor([12.0], dtype=DTYPE)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        stim.restore_dict_from_checkpoint(corrupt)
    _assert_same(stim.state_dict_for_checkpoint(), before)
    _assert_same(stim.next_sched_time, before_head)
    assert stim.spikes is before_spikes
    assert stim.spike_gate is before_gate


@pytest.mark.parametrize(
    "corruption",
    [
        "cursor_before_row",
        "cursor_after_row",
        "incorrect_offset",
        "missing_sentinel",
        "incomplete_calendar",
        "non_heap_layout",
        "wrong_layout_rows",
        "unsupported_time_dtype",
    ],
)
def test_invalid_modern_checkpoint_restore_is_atomic(corruption):
    stim = NetStim(interval=0.5, noise=0.7, seed=91, dtype=DTYPE).initialize()
    stim.schedule(0, [0.1, 0.2, 0.3])
    stim(0.15)
    before = _clone(stim.state_dict_for_checkpoint())
    before_head = stim.next_sched_time.clone()
    before_spikes, before_gate = stim.spikes, stim.spike_gate
    corrupt = _clone(before)
    if corruption == "cursor_before_row":
        corrupt["heap_cursor"][0] = -1
    elif corruption == "cursor_after_row":
        corrupt["heap_cursor"][0] = corrupt["heap_row_offsets"][-1] + 1
    elif corruption == "incorrect_offset":
        corrupt["heap_row_offsets"][-1] -= 1
    elif corruption == "missing_sentinel":
        corrupt["heap_times"][-1] = 2.0
    elif corruption == "incomplete_calendar":
        del corrupt["heap_row_offsets"]
    elif corruption == "non_heap_layout":
        corrupt["heap_layout"] = ((0.2, 0.1, 0.3),)
    elif corruption == "unsupported_time_dtype":
        corrupt["heap_times"] = corrupt["heap_times"].to(torch.long)
    else:
        corrupt["heap_layout"] = ()
    with pytest.raises(ValueError):
        stim.restore_dict_from_checkpoint(corrupt)
    _assert_same(stim.state_dict_for_checkpoint(), before)
    _assert_same(stim.next_sched_time, before_head)
    assert stim.spikes is before_spikes
    assert stim.spike_gate is before_gate


def test_failed_modern_restore_preserves_emitted_gate_autograd_history():
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    clock = torch.tensor([0.05], dtype=DTYPE, requires_grad=True)
    checkpoint = _legacy_checkpoint(stim, [[0.1]])
    checkpoint["next_stoch_time"] = clock
    stim.restore_dict_from_checkpoint(checkpoint)
    stim(0.1, bptt=True)
    original_spikes, original_gate = stim.spikes, stim.spike_gate
    corrupt = _clone(stim.state_dict_for_checkpoint())
    corrupt["heap_cursor"] = torch.tensor([999])
    with pytest.raises(ValueError):
        stim.restore_dict_from_checkpoint(corrupt)
    assert stim.spikes is original_spikes
    assert stim.spike_gate is original_gate
    (gradient,) = torch.autograd.grad(stim.spike_gate.sum(), (clock,))
    soft = torch.sigmoid(clock.new_tensor((0.1 - 0.05) / stim.tau))
    torch.testing.assert_close(gradient, (-soft * (1 - soft) / stim.tau).reshape(1))


@pytest.mark.parametrize(
    "events",
    [[float("nan"), 0.2, 0.1], [0.2, float("nan"), 0.1], [0.2, 0.1, float("nan")]],
)
def test_native_nan_heap_barriers_preserve_heapq_insertion_order(events):
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    stim.schedule(0, events, allow_past=True)
    # heapify is deliberately not used: insertion and heapify differ in the
    # presence of NaNs, and schedule historically used incremental heappush.
    row = []
    for value in events:
        heapq.heappush(row, value)
    count = 0
    for time in (0.15, 0.3, 0.3):
        head = row[0] if row else float("inf")
        fired = math.isfinite(head) and time >= head
        if fired:
            while row and row[0] <= time:
                heapq.heappop(row)
            count += 1
        stim(time)
        assert bool(stim.spikes[0]) is fired
        assert stim.spike_counts.item() == count
        _assert_same(
            stim.next_sched_time,
            torch.tensor([row[0] if row else float("inf")], dtype=DTYPE),
        )
        _assert_same(stim._sched_heaps, [row])


@pytest.mark.parametrize("consume_before_inserting_nan", [False, True])
def test_late_nan_insertion_retains_original_finite_heap_layout(
    consume_before_inserting_nan,
):
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    events = [5.0, 2.0, 3.0, 4.0]
    row = []
    for event in events:
        heapq.heappush(row, event)
    stim.schedule(0, events)
    if consume_before_inserting_nan:
        assert bool(stim(2.0)[0])
        heapq.heappop(row)
    stim.schedule(0, float("nan"), allow_past=True)
    heapq.heappush(row, float("nan"))
    # Sorting an earlier finite heap is numerically harmless until a later NaN
    # insertion. Its changed tree then moves the blocking NaN to another place.
    for time in (2.0, 3.0, 4.0, 5.0):
        fired = math.isfinite(row[0]) and row[0] <= time
        if fired:
            while row and row[0] <= time:
                heapq.heappop(row)
        stim(time)
        assert bool(stim.spikes[0]) is fired
        _assert_same(stim._sched_heaps, [row])


@pytest.mark.parametrize("with_events", [False, True])
def test_finite_native_forward_uses_cached_calendar_without_materializing_heaps(
    monkeypatch, with_events
):
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    if with_events:
        stim.schedule(0, [0.1, 0.1, 0.2, 0.7])

    def reject_materialization():
        raise AssertionError("Native forward must reuse its packed calendar.")

    monkeypatch.setattr(stim, "_materialize_heap_rows", reject_materialization)
    for time in (0.1, 0.3, 0.3, 0.2, 0.8):
        stim(time)


@pytest.mark.parametrize(
    "events", [[float("-inf"), 0.1], [0.1, float("inf")], [float("inf")]]
)
def test_nonfinite_heap_head_is_not_consumed_as_a_finite_heap_event(events):
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    stim.schedule(0, events, allow_past=True)
    row = []
    for value in events:
        heapq.heappush(row, value)
    for time in (0.15, 0.3):
        head = row[0] if row else float("inf")
        fired = math.isfinite(head) and time >= head
        if fired:
            while row and row[0] <= time:
                heapq.heappop(row)
        stim(time)
        assert bool(stim.spikes[0]) is fired
        _assert_same(
            stim.next_sched_time,
            torch.tensor([row[0] if row else float("inf")], dtype=DTYPE),
        )
        assert _pending(stim) == [sorted(row)]


@pytest.mark.parametrize("finite_head", [False, True])
def test_infinite_cutoff_can_pop_infinity_only_after_a_finite_head_fires(finite_head):
    stim = NetStim(start=100.0, interval=100.0, dtype=DTYPE).initialize()
    events = [0.1, float("inf")] if finite_head else [float("inf")]
    stim.schedule(0, events)
    stim(float("inf"))
    # Infinity alone is not a finite firing head. When a finite head does fire,
    # however, the old <= cutoff loop consumes every due entry, including +inf.
    assert _pending(stim) == ([[]] if finite_head else [[float("inf")]])
