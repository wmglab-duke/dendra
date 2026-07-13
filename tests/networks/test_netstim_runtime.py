import pytest
import torch

from dendra.models.networks.netstim import NetStim

DT = 0.1
DTYPE = torch.float64


def _scheduled_stim(N=3):
    return (
        NetStim(
            N=N,
            interval=100.0,
            start=100.0,
            noise=0.0,
            max_spikes=10,
            dtype=DTYPE,
        )
        .set_dt(DT)
        .initialize()
    )


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def test_tensor_schedule_produces_weight_and_time_gradients():
    stim = _scheduled_stim()
    time = torch.tensor([0.2], dtype=DTYPE, requires_grad=True)
    weight = torch.tensor([2.0], dtype=DTYPE, requires_grad=True)
    stim.schedule(1, time, weight=weight)

    stim.forward(0.16, dt=DT)

    assert stim.spikes.tolist() == [False, True, False]
    assert stim.spike_gate.tolist() == pytest.approx([0.0, 1.2, 0.0])
    stim.spike_gate.sum().backward()
    assert weight.grad.item() == pytest.approx(0.6)
    assert time.grad.item() == pytest.approx(-20.0)


def test_reference_weight_and_time_schedules_read_bound_sources():
    weight_stim = _scheduled_stim()
    weight_source = torch.tensor([4.0], dtype=DTYPE, requires_grad=True)
    assert weight_stim.bind_weight_source(weight_source) is weight_stim
    weight_stim.schedule_ref(0, times_ms=0.2, weight_idx=0)
    weight_stim.forward(0.2, dt=DT)
    assert weight_stim.spike_gate.tolist() == pytest.approx([4.0, 0.0, 0.0])
    weight_stim.spike_gate.sum().backward()
    assert weight_source.grad.item() == pytest.approx(1.0)

    time_stim = _scheduled_stim()
    time_source = torch.tensor([0.2], dtype=DTYPE, requires_grad=True)
    assert time_stim.bind_time_source(time_source) is time_stim
    time_stim.schedule_time_ref(2, time_idx=0, weight=3.0)
    time_stim.forward(0.16, dt=DT)
    assert time_stim.spike_gate.tolist() == pytest.approx([0.0, 0.0, 1.8])
    time_stim.spike_gate.sum().backward()
    assert time_source.grad.item() == pytest.approx(-30.0)


def test_tensor_schedule_filters_past_events_and_keeps_metadata_aligned():
    stim = _scheduled_stim()
    stim.t_last.fill_(0.2)
    stim.schedule(
        [0, 2],
        torch.tensor([0.1, 0.3], dtype=DTYPE),
        weight=torch.tensor([5.0, 7.0], dtype=DTYPE),
    )

    assert stim.sched_flat_idx.tolist() == [2]
    assert stim.sched_abs_step.tolist() == [3]
    assert stim.sched_time_ms.tolist() == pytest.approx([0.3])
    assert stim.sched_weight.tolist() == pytest.approx([7.0])
    assert stim.sched_time_idx.tolist() == [-1]
    assert stim.sched_weight_idx.tolist() == [-1]


def test_reference_schedules_validate_binding_and_source_indices():
    stim = _scheduled_stim()
    with pytest.raises(RuntimeError, match="bind_weight_source"):
        stim.schedule_ref(0, times=0.1, weight_idx=0)
    with pytest.raises(RuntimeError, match="bind_time_source"):
        stim.schedule_time_ref(0, time_idx=0)

    stim.bind_weight_source(torch.ones(1, dtype=DTYPE))
    stim.bind_time_source(torch.ones(1, dtype=DTYPE))
    with pytest.raises(IndexError, match="weight_idx"):
        stim.schedule_ref(0, times=0.1, weight_idx=1)
    with pytest.raises(IndexError, match="weight_idx"):
        stim.schedule_ref(0, times=0.1, weight_idx=-2)
    with pytest.raises(IndexError, match="time_idx"):
        stim.schedule_time_ref(0, time_idx=1)
    with pytest.raises(IndexError, match="time_idx"):
        stim.schedule_time_ref(0, time_idx=-2)


def test_clear_schedule_can_remove_selected_or_all_events():
    stim = _scheduled_stim()
    stim.schedule([0, 1], [0.2, 0.3])
    stim.schedule(2, torch.tensor(0.4, dtype=DTYPE), weight=2.0)

    assert stim.clear_schedule([1, 2]) is stim
    assert stim.next_sched_time[0].item() == pytest.approx(0.2)
    assert torch.isinf(stim.next_sched_time[1:]).all()
    assert stim.sched_flat_idx.numel() == 0

    stim.schedule(0, torch.tensor(0.5, dtype=DTYPE), weight=2.0)
    stim.clear_schedule()
    assert stim.sched_flat_idx.numel() == 0
    assert torch.isinf(stim.next_sched_time).all()


def test_checkpoint_restore_replays_stochastic_rng_and_schedules():
    stim = NetStim(
        N=2,
        interval=0.5,
        start=0.0,
        noise=1.0,
        max_spikes=5,
        seed=1234,
        dtype=DTYPE,
    ).initialize()
    stim.schedule(1, 0.75)
    stim.forward(0.25)
    checkpoint = _clone_nested(stim.state_dict_for_checkpoint())

    first = []
    for time in (0.5, 0.75, 1.0, 1.5):
        stim.forward(time)
        first.append((stim.spikes.clone(), stim.next_stoch_time.clone()))

    stim.restore_dict_from_checkpoint(checkpoint)
    second = []
    for time in (0.5, 0.75, 1.0, 1.5):
        stim.forward(time)
        second.append((stim.spikes.clone(), stim.next_stoch_time.clone()))

    for (spikes_a, next_a), (spikes_b, next_b) in zip(first, second):
        assert torch.equal(spikes_a, spikes_b)
        assert torch.equal(next_a, next_b)


def test_checkpoint_restore_validates_schedule_heap_shape():
    stim = _scheduled_stim()
    checkpoint = _clone_nested(stim.state_dict_for_checkpoint())
    checkpoint["sched_heaps"] = checkpoint["sched_heaps"][:-1]

    with pytest.raises(ValueError, match="schedule heaps"):
        stim.restore_dict_from_checkpoint(checkpoint)


def test_checkpoint_restore_shape_failure_is_atomic():
    stim = _scheduled_stim()
    before = _clone_nested(stim.state_dict_for_checkpoint())
    corrupt = _clone_nested(before)
    corrupt["shape"] = (2, 3)

    with pytest.raises(ValueError, match="schedule heaps"):
        stim.restore_dict_from_checkpoint(corrupt)

    assert stim.shape == before["shape"]
    _assert_nested_equal(stim.state_dict_for_checkpoint(), before)


def test_batch_replicates_tensor_backed_schedules():
    stim = _scheduled_stim(N=2)
    stim.schedule(1, torch.tensor(0.2, dtype=DTYPE), weight=2.0)

    stim.batch((2, 3))

    assert stim.shape == (2, 3, 2)
    assert stim.sched_flat_idx.tolist() == [1, 3, 5, 7, 9, 11]
    assert stim.sched_weight.tolist() == pytest.approx([2.0] * 6)
