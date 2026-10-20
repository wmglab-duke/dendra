"""Pure compact-calendar search, native caching, and transform contracts."""

import heapq
import importlib
import math

import pytest
import torch

from dendra.models.networks._heap_calendar import (
    heap_calendar_consume,
    heap_calendar_head,
    pack_heap_calendar,
    validate_heap_calendar,
)
from dendra.models.networks._netstim_renewal import NetStimState, netstim_renewal
from dendra.models.networks.netstim import NetStim

calendar_module = importlib.import_module("dendra.models.networks._heap_calendar")
netstim_module = importlib.import_module("dendra.models.networks.netstim")


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_upper_bound_search_matches_independent_heapq_across_uneven_rows(dtype):
    rows = [
        [((index * 17) % 73) / 10 for index in range(length)]
        for length in (0, 1, 2, 3, 4, 15, 64, 257)
    ]
    rows += [[float("-inf"), 0.0], [float("inf")], []]
    oracle = [list(row) for row in rows]
    for row in oracle:
        heapq.heapify(row)
    times, offsets, cursor, depth = pack_heap_calendar(rows, device="cpu")
    validate_heap_calendar(times, offsets, cursor, rows=len(rows), device="cpu")
    original_times = times.clone()
    generator = torch.Generator().manual_seed(23)
    for step in range(30):
        # Independent row clocks include backward and repeated observations.
        t = (torch.rand(len(rows), generator=generator, dtype=dtype) * 9).floor()
        available = torch.arange(len(rows)) % 3 != step % 3
        expected_heads = torch.tensor(
            [row[0] if row else float("inf") for row in oracle], dtype=dtype
        )
        head = heap_calendar_head(times, offsets[1:], cursor, dtype=dtype)
        torch.testing.assert_close(head, expected_heads, rtol=0, atol=0)
        fires = torch.isfinite(head) & (head <= t) & available
        old_cursor = cursor.clone()
        result = heap_calendar_consume(
            times, offsets[1:], cursor, t, fires, depth=depth, dtype=dtype
        )
        for index, row in enumerate(oracle):
            if bool(fires[index]):
                while row and torch.tensor(row[0], dtype=dtype) <= t[index]:
                    heapq.heappop(row)
            assert result[index] == offsets[index + 1] - len(row)
        torch.testing.assert_close(cursor, old_cursor, rtol=0, atol=0)
        torch.testing.assert_close(times, original_times, rtol=0, atol=0)
        cursor = result


@pytest.mark.parametrize("rows", ([], [[], [], []], [[], [0.1], []]))
def test_empty_rows_have_safe_heads_and_zero_work_when_fully_empty(rows):
    times, offsets, cursor, depth = pack_heap_calendar(rows, device="cpu")
    validate_heap_calendar(times, offsets, cursor, rows=len(rows), device="cpu")
    head = heap_calendar_head(times, offsets[1:], cursor, dtype=torch.float64)
    t = torch.full((len(rows),), 1.0, dtype=torch.float64)
    result = heap_calendar_consume(
        times,
        offsets[1:],
        cursor,
        t,
        torch.isfinite(head),
        depth=depth,
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, offsets[1:], rtol=0, atol=0)


def test_search_compares_in_simulation_dtype_without_changing_original_times():
    below = math.nextafter(1.0, 0.0)
    above = math.nextafter(1.0, float("inf"))
    times, offsets, cursor, depth = pack_heap_calendar(
        [[above, below, 1.0]], device="cpu"
    )
    for dtype, count in ((torch.float32, 3), (torch.float64, 2)):
        result = heap_calendar_consume(
            times,
            offsets[1:],
            cursor,
            torch.tensor([1.0], dtype=dtype),
            torch.tensor([True]),
            depth=depth,
            dtype=dtype,
        )
        assert result.item() == count
    assert times.tolist() == [below, 1.0, above, float("inf")]


def _calendar_transition(times, offsets, cursor, depth, clock, t):
    head = heap_calendar_head(times, offsets[1:], cursor, dtype=clock.dtype)
    transition = netstim_renewal(
        NetStimState(
            clock, torch.zeros_like(cursor), torch.full_like(clock, -float("inf"))
        ),
        torch.full_like(clock, 2.0),
        torch.zeros_like(clock),
        torch.full_like(cursor, 5),
        t,
        torch.ones_like(clock),
        tau=0.2,
        next_sched_time=head,
    )
    next_cursor = heap_calendar_consume(
        times,
        offsets[1:],
        cursor,
        t,
        transition.heap_fires,
        depth=depth,
        dtype=clock.dtype,
    )
    return next_cursor, transition.state.next_stoch_time, transition.spike_gate


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_shared_calendar_compiles_fullgraph_with_renewal_gradients(backend, dtype):
    times, offsets, cursor, depth = pack_heap_calendar(
        [[0.3, 0.3, 0.9], [], [0.2, 0.4], [1 + 2**-30, 1 + 2**-29]], device="cpu"
    )
    clock = torch.tensor([0.3, 0.5, 0.1, 100.0], dtype=dtype, requires_grad=True)
    t = torch.tensor([0.3, 0.4, 0.4, 1.0], dtype=dtype)

    def step(current):
        return _calendar_transition(times, offsets, cursor, depth, current, t)

    expected = step(clock)
    actual = torch.compile(step, backend=backend, fullgraph=True)(clock)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    expected_gradient = torch.autograd.grad(
        expected[1].sum() + expected[2].sum(), clock
    )[0]
    actual_gradient = torch.autograd.grad(actual[1].sum() + actual[2].sum(), clock)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    # Float32 comparison consumes both original Float64 events at t=1; a
    # widened comparison would silently leave those events pending.
    assert actual[0].tolist() == [2, 3, 5, 7 if dtype == torch.float32 else 5]


@pytest.mark.parametrize("lanes", (0, 3))
@pytest.mark.parametrize("mapped_cursor", (False, True))
def test_calendar_vmap_with_empty_lanes_and_independent_cursors(lanes, mapped_cursor):
    times, offsets, cursor, depth = pack_heap_calendar([[0.0, 0.1], []], device="cpu")
    cursors = cursor.repeat(lanes, 1) if mapped_cursor else cursor
    clocks = torch.full((lanes, 2), 1.0, dtype=torch.float64)
    t = torch.full_like(clocks, 0.1)
    actual = torch.vmap(
        lambda current, clock, time: _calendar_transition(
            times, offsets, current, depth, clock, time
        ),
        in_dims=(0 if mapped_cursor else None, 0, 0),
    )(cursors, clocks, t)
    assert actual[0].shape == (lanes, 2)
    assert actual[1].shape == (lanes, 2)
    torch.testing.assert_close(actual[0], torch.tensor([[2, 2]]).repeat(lanes, 1))


def test_native_forward_never_repacks_or_materializes_normal_heap_rows(monkeypatch):
    source = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    source.schedule(0, [0.1, 0.1, 0.2])
    times = source.heap_times

    def fail():
        raise AssertionError("Native forward must use the cached tensor calendar.")

    monkeypatch.setattr(source, "_materialize_heap_rows", fail)
    for time in (0.0, 0.1, 0.1, 0.3):
        source(time)
        assert source.heap_times is times
    assert source.heap_cursor.tolist() == [3]


@pytest.mark.parametrize("parent_route", (False, True))
def test_dtype_roundtrip_reconstructs_original_head_without_reinitialization(
    parent_route,
):
    time = 1.0 + 2**-30
    source = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    source.schedule(0, time)
    owner = torch.nn.Module() if parent_route else source
    if parent_route:
        owner.add_module("source", source)
    owner.float().double()
    assert source.heap_times.dtype == torch.float64
    assert source.next_sched_time.item() == time
    assert not bool(source(1.0)[0])
    assert bool(source(time)[0])


@pytest.mark.parametrize(
    "failure", ("sentinel", "nan", "order", "offset", "cursor", "dtype")
)
def test_cold_validator_rejects_malformed_calendar(failure):
    times, offsets, cursor, _ = pack_heap_calendar([[0.1, 0.2]], device="cpu")
    if failure == "sentinel":
        times[-1] = 0
    elif failure == "nan":
        times[0] = float("nan")
    elif failure == "order":
        times[0] = 0.3
    elif failure == "offset":
        offsets[0] = 1
    elif failure == "cursor":
        cursor[0] = 3
    else:
        times = times.float()
    with pytest.raises(ValueError):
        validate_heap_calendar(times, offsets, cursor, rows=1, device="cpu")


def test_nan_calendars_cannot_be_silently_sorted_into_the_pure_core():
    with pytest.raises(ValueError, match="NaN"):
        pack_heap_calendar([[float("nan"), 0.1]], device="cpu")


def _use_simulation_dtype_runtime(monkeypatch):
    """Exercise the MPS representation on CPU without requiring Metal."""

    def policy(device, simulation_dtype=torch.float32):
        return simulation_dtype

    monkeypatch.setattr(calendar_module, "heap_calendar_dtype", policy)
    monkeypatch.setattr(netstim_module, "heap_calendar_dtype", policy)


def test_calendar_dtype_policy_avoids_float64_on_mps():
    assert calendar_module.heap_calendar_dtype("mps") == torch.float32
    assert calendar_module.heap_calendar_dtype("mps", torch.float16) == torch.float16
    assert calendar_module.heap_calendar_dtype("mps", torch.bfloat16) == torch.bfloat16
    assert calendar_module.heap_calendar_dtype("cpu") == torch.float64
    assert calendar_module.heap_calendar_dtype("cuda") == torch.float64


@pytest.mark.parametrize("parent_move", (False, True))
def test_single_dtype_move_keeps_mixed_schedules_aligned_and_optimizer_leaves(
    parent_move,
):
    from dendra.func import make_functional_netstim

    source = NetStim(start=100.0, noise=0.0, seed=73, dtype=torch.float64).initialize()
    source.set_dt(0.1)
    source.schedule(0, [0.05, 0.15, 1.0 + 2**-30])
    event_time = torch.tensor([0.12], dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    time_source = torch.nn.Parameter(torch.tensor([0.21], dtype=torch.float64))
    weight_source = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
    source.schedule(0, event_time, weight=amplitude)
    source.bind_time_source(time_source).bind_weight_source(weight_source)
    source.schedule_time_ref(0, time_idx=0, weight=0.6)
    source.schedule_ref(0, times=0.23, weight_idx=0)
    optimizer = torch.optim.SGD([time_source, weight_source], lr=0.01)
    rng = source._rng.get_state().clone()
    seeder = source._seeder.get_state().clone()
    counts, cursor = source.spike_counts.clone(), source.heap_cursor.clone()
    owner = torch.nn.ModuleDict({"source": source}) if parent_move else source
    owner.float()
    assert source.sched_time_ms.dtype == torch.float32
    assert source.sched_weight.dtype == torch.float32
    assert source._sched_t_source is time_source
    assert source._sched_w_source is weight_source
    assert optimizer.param_groups[0]["params"][0] is time_source
    assert optimizer.param_groups[0]["params"][1] is weight_source
    assert time_source.is_leaf and weight_source.is_leaf
    assert event_time.dtype == amplitude.dtype == torch.float64
    assert event_time.is_leaf and amplitude.is_leaf
    torch.testing.assert_close(source._rng.get_state(), rng)
    torch.testing.assert_close(source._seeder.get_state(), seeder)
    torch.testing.assert_close(source.spike_counts, counts)
    torch.testing.assert_close(source.heap_cursor, cursor)
    plan, tensors = make_functional_netstim(source)
    state = tensors.state
    functional_loss, native_loss = 0, 0
    for index, time in enumerate((0.1, 0.2, 0.3)):
        draw = plan.sample_step_draw(state)
        state, output = plan.step(
            tensors.parameters,
            tensors.constants,
            state,
            time,
            draw,
            schedule=tensors.schedule,
        )
        source(time)
        torch.testing.assert_close(output.spikes, source.spikes)
        torch.testing.assert_close(output.spike_gate, source.spike_gate)
        torch.testing.assert_close(state["heap_cursor"], source.heap_cursor)
        torch.testing.assert_close(state["rng_state"], source._rng.get_state())
        functional_loss = functional_loss + (index + 1) * output.spike_gate.sum()
        native_loss = native_loss + (index + 1) * source.spike_gate.sum()
    actual = torch.autograd.grad(
        functional_loss,
        (
            tensors.schedule.times,
            tensors.schedule.amplitudes,
            tensors.schedule.time_source,
            tensors.schedule.amplitude_source,
        ),
    )
    expected = torch.autograd.grad(
        native_loss, (event_time, amplitude, time_source, weight_source)
    )
    for index, (gradient, reference) in enumerate(zip(actual, expected, strict=True)):
        if index < 2:
            gradient = gradient[:1].double()
        torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize("consume_rounded_events", (False, True))
def test_device_representation_roundtrip_recovers_original_times_and_cursor(
    monkeypatch, consume_rounded_events
):
    events = [0.25, 1.0 + 2**-30, 1.0 + 2**-29]
    source = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    source.schedule(0, events)
    source(0.25)
    assert source.heap_cursor.item() == 1
    with monkeypatch.context() as context:
        _use_simulation_dtype_runtime(context)
        source.to(dtype=torch.float32)
        assert source.heap_times.dtype == torch.float32
        assert source.next_sched_time.item() == 1.0
        from dendra.func import make_functional_netstim

        plan, tensors = make_functional_netstim(source)
        draw = plan.sample_step_draw(tensors.state)
        state, output = plan.step(
            tensors.parameters, tensors.constants, tensors.state, 1.0, draw
        )
        assert output.spikes.all()
        assert state["heap_cursor"].item() == 3
        if consume_rounded_events:
            source(1.0)
            assert source.heap_cursor.item() == 3
    source.double()
    assert source.heap_times.tolist() == events + [float("inf")]
    assert source.heap_times.dtype == torch.float64
    assert source.heap_cursor.item() == (3 if consume_rounded_events else 1)
    assert source.next_sched_time.item() == (
        float("inf") if consume_rounded_events else events[1]
    )


@pytest.mark.parametrize("save_float32", (False, True))
def test_checkpoint_import_rebuilds_destination_precision_from_original_layout(
    monkeypatch, save_float32
):
    events = [0.25, 1.0 + 2**-30, 1.0 + 2**-29]
    source = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    source.schedule(0, events)
    source(0.25)
    with monkeypatch.context() as context:
        if save_float32:
            _use_simulation_dtype_runtime(context)
            source.float()
        checkpoint = source.state_dict_for_checkpoint()
    target = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    with monkeypatch.context() as context:
        if not save_float32:
            _use_simulation_dtype_runtime(context)
            target.float()
        target.restore_dict_from_checkpoint(checkpoint)
        assert target.heap_times.dtype == (
            torch.float64 if save_float32 else torch.float32
        )
        assert target.heap_cursor.item() == 1
        assert target.next_sched_time.item() == (events[1] if save_float32 else 1.0)
        assert target.state_dict_for_checkpoint()["heap_layout"] == (tuple(events),)
    target.double()
    assert target.heap_times.tolist() == events + [float("inf")]
    assert not bool(target(1.0)[0])


def test_float32_checkpoint_without_original_layout_retains_stored_precision(
    monkeypatch,
):
    source = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    source.schedule(0, 1.0 + 2**-30)
    with monkeypatch.context() as context:
        _use_simulation_dtype_runtime(context)
        source.float()
        checkpoint = source.state_dict_for_checkpoint()
    del checkpoint["heap_layout"]
    target = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    target.restore_dict_from_checkpoint(checkpoint)
    torch.testing.assert_close(target.heap_times, checkpoint["heap_times"].double())
    assert target.next_sched_time.item() == 1.0


@pytest.mark.parametrize(
    "dtype, event",
    [(torch.float16, 1 + 2**-11 + 2**-30), (torch.bfloat16, 1 + 2**-8 + 2**-20)],
)
def test_low_precision_device_calendar_avoids_double_rounding_and_restores_originals(
    monkeypatch, dtype, event
):
    from dendra.func import make_functional_netstim

    direct = torch.tensor(event, dtype=dtype)
    if dtype == torch.float16:
        assert direct > torch.tensor(event, dtype=torch.float32).to(dtype)
    assert direct > 1.0
    source = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    source.schedule(0, event)
    with monkeypatch.context() as context:
        _use_simulation_dtype_runtime(context)
        source.to(dtype=dtype)
        assert source.heap_times.dtype == dtype
        assert source.next_sched_time.item() == direct.item()
        plan, tensors = make_functional_netstim(source)
        draw = plan.sample_step_draw(tensors.state)
        state, output = plan.step(
            tensors.parameters, tensors.constants, tensors.state, 1.0, draw
        )
        assert not output.spikes.any()
        assert not source(1.0).any()
        torch.testing.assert_close(state["heap_cursor"], source.heap_cursor)
        checkpoint = source.state_dict_for_checkpoint()
        source.to(dtype=torch.float32)
        assert (
            source.heap_times[0].item()
            == torch.tensor(event, dtype=torch.float32).item()
        )
    source.double()
    target = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    target.restore_dict_from_checkpoint(checkpoint)
    for current in (source, target):
        assert current.heap_times[0].item() == event
        assert current.next_sched_time.item() == event
        assert current.heap_cursor.item() == 0


@pytest.mark.parametrize("nan_heap", (False, True))
def test_device_representation_move_keeps_legacy_head_override(monkeypatch, nan_heap):
    source = NetStim(start=100.0, noise=0.0, dtype=torch.float64).initialize()
    source.schedule(0, [float("nan"), 1.25] if nan_heap else [1.0 + 2**-30])
    checkpoint = source.state_dict_for_checkpoint()
    checkpoint = {
        name: value
        for name, value in checkpoint.items()
        if not name.startswith("heap_") and name != "sched_heaps"
    }
    checkpoint["next_sched_time"] = torch.tensor([0.75], dtype=torch.float64)
    source.restore_dict_from_checkpoint(checkpoint)
    with monkeypatch.context() as context:
        _use_simulation_dtype_runtime(context)
        source.float()
        assert source.heap_times.dtype == torch.float32
        assert source.heap_override_active.all()
        assert source.next_sched_time.item() == 0.75
    source.double()
    assert source.heap_times.dtype == torch.float64
    assert source.heap_override_active.all()
    assert source.next_sched_time.item() == 0.75


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
@pytest.mark.parametrize("construct_on_mps", (False, True))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_mps_native_calendar_and_functional_cursors_keep_original_cold_precision(
    construct_on_mps,
    dtype,
):
    import dendra as dn

    event = 1.0 + 2**-30
    source = NetStim(
        start=100.0,
        noise=0.0,
        seed=73,
        device="mps" if construct_on_mps else "cpu",
        dtype=dtype,
    )
    source.schedule(0, [0.25, event])
    source.to("mps").initialize()
    assert source.heap_times.dtype == dtype
    assert source.heap_times.device.type == "mps"
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    state, output = plan.step(
        tensors.parameters, tensors.constants, tensors.state, 0.25, draw
    )
    source(0.25)
    torch.testing.assert_close(output.spikes, source.spikes)
    torch.testing.assert_close(state["heap_cursor"], source.heap_cursor)
    source.cpu().double()
    assert source.heap_times.tolist() == [0.25, event, float("inf")]
    assert source.heap_cursor.item() == 1
    assert source.next_sched_time.item() == event
