from __future__ import annotations

import copy

import pytest
import torch

from dendra.models.networks.netstim import NetStim

DT = 0.1
DTYPE = torch.float64


def _stim(n=3, *, batch=None):
    stim = NetStim(
        N=n,
        interval=100.0,
        start=100.0,
        noise=0.0,
        max_spikes=20,
        dtype=DTYPE,
    ).set_dt(DT)
    if batch is not None:
        stim.batch(batch)
    return stim.initialize()


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


def _assert_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            _assert_equal(got, want)
    else:
        assert actual == expected


def _runtime_schedule_state(stim):
    return {
        "checkpoint": _clone(stim.state_dict_for_checkpoint()),
        "next_sched_time": stim.next_sched_time.clone(),
        "spikes": stim.spikes.clone(),
        "spike_gate": stim.spike_gate.clone(),
    }


def test_constructor_broadcasts_multidimensional_parameters_over_generator_axis():
    interval = torch.tensor([[1.0], [2.0]], dtype=DTYPE)
    start = torch.tensor([[0.0, 0.1, 0.2]], dtype=DTYPE)
    max_spikes = torch.tensor([[1], [2]])
    stim = NetStim(
        N=3,
        interval=interval,
        start=start,
        noise=0.0,
        max_spikes=max_spikes,
        dtype=DTYPE,
    ).initialize()

    assert stim.shape == (2, 3)
    torch.testing.assert_close(
        stim.start,
        torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]], dtype=DTYPE),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        stim.interval(),
        torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=DTYPE),
    )
    assert stim.max_spikes.tolist() == [[1, 1, 1], [2, 2, 2]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"N": 0},
        {"N": 3, "interval": torch.ones((2, 2))},
        {"N": 3, "start": torch.ones((2, 2))},
        {"N": 3, "max_spikes": torch.ones((4, 2), dtype=torch.long)},
    ],
)
def test_constructor_rejects_invalid_generator_axis_without_partial_object(kwargs):
    with pytest.raises(ValueError):
        NetStim(dtype=DTYPE, **kwargs)


def test_multidimensional_coordinate_and_generator_indices_target_exact_elements():
    stim = _stim(batch=(2, 2)).set_diff_config(diff_scheduled_times=False)
    # A bare generator index fans out across every leading batch coordinate.
    stim.schedule(-1, 0.2, weight=torch.tensor(2.0, dtype=DTYPE))
    # Coordinate rows each target one state element and support negative indices.
    stim.schedule(
        torch.tensor([[0, 1, 0], [1, 0, -2]]),
        [0.2, 0.2],
        weight=[3.0, 5.0],
    )

    spikes = stim.forward(torch.full((2, 2), 0.2, dtype=DTYPE), dt=DT)
    expected_gate = torch.tensor(
        [[[0.0, 0.0, 2.0], [3.0, 0.0, 2.0]], [[0.0, 5.0, 2.0], [0.0, 0.0, 2.0]]],
        dtype=DTYPE,
    )
    torch.testing.assert_close(stim.spike_gate, expected_gate, atol=0.0, rtol=0.0)
    assert torch.equal(spikes, expected_gate != 0)


def test_legacy_schedule_broadcasts_one_index_over_many_times_and_one_time_over_indices():
    stim = _stim(n=2, batch=2)
    stim.schedule((1, 0), [0.1, 0.2])
    stim.schedule([(0, 1), (1, 1)], 0.2)

    assert stim.forward(0.1).tolist() == [[False, False], [True, False]]
    assert stim.forward(0.2).tolist() == [[False, True], [True, True]]
    assert torch.isinf(stim.next_sched_time).all()


def test_reference_schedules_broadcast_over_batches_and_read_live_sources():
    stim = _stim(n=2, batch=2).set_diff_config(diff_scheduled_times=False)
    weight_source = torch.tensor([2.0, 4.0], dtype=DTYPE, requires_grad=True)
    time_source = torch.tensor([0.2, 0.3], dtype=DTYPE, requires_grad=True)
    stim.bind_weight_source(weight_source)
    stim.bind_time_source(time_source)

    stim.schedule_ref(0, times=0.2, weight_idx=1)
    stim.schedule_time_ref([(0, 1), (1, 1)], time_idx=[0, 1], weight=[3.0, 5.0])
    # References are evaluated at delivery, not copied at scheduling time.
    with torch.no_grad():
        weight_source[1] = 6.0
        time_source.add_(0.1)

    stim.forward(0.2, dt=DT)
    torch.testing.assert_close(
        stim.spike_gate, torch.tensor([[6.0, 0.0], [6.0, 0.0]], dtype=DTYPE)
    )
    stim.forward(0.3, dt=DT)
    torch.testing.assert_close(
        stim.spike_gate, torch.tensor([[0.0, 3.0], [0.0, 0.0]], dtype=DTYPE)
    )
    stim.forward(0.4, dt=DT)
    torch.testing.assert_close(
        stim.spike_gate, torch.tensor([[0.0, 0.0], [0.0, 5.0]], dtype=DTYPE)
    )


def test_past_filtering_is_applied_per_batched_state_element():
    stim = _stim(n=2, batch=2)
    stim.t_last.copy_(torch.tensor([[0.0, 0.4], [0.0, 0.3]], dtype=DTYPE))
    stim.schedule(1, 0.35, weight=torch.tensor(2.0, dtype=DTYPE))

    assert stim.sched_flat_idx.tolist() == [3]
    assert stim.sched_time_ms.tolist() == pytest.approx([0.35])
    stim.set_diff_config(diff_scheduled_times=False)
    assert stim.forward(0.4, dt=DT).tolist() == [[False, False], [False, True]]


def test_allow_past_retains_events_for_explicit_replay():
    stim = _stim(n=1)
    stim.t_last.fill_(0.5)
    stim.schedule(0, 0.2, weight=2.0)
    assert stim.sched_flat_idx.numel() == 0

    stim.schedule(0, 0.2, weight=2.0, allow_past=True)
    assert stim.sched_flat_idx.tolist() == [0]
    stim.set_diff_config(diff_scheduled_times=False)
    # Exact scheduled-step semantics do not replay an event on a later grid step.
    assert not bool(stim.forward(0.5, dt=DT)[0])


@pytest.mark.parametrize(
    "operation",
    [
        lambda s: s.schedule_ref(0, times=0.2),
        lambda s: s.schedule_ref(0, weight_idx=0),
        lambda s: s.schedule_ref(0, times=0.2, times_ms=0.3, weight_idx=0),
        lambda s: s.schedule_ref(0, times=0.2, weight_idx=4),
        lambda s: s.schedule_time_ref(0, time_idx=4),
        lambda s: s.clear_schedule([0, (0, 1, 2)]),
    ],
)
def test_malformed_reference_and_clear_requests_are_atomic(operation):
    stim = _stim(n=3, batch=2)
    stim.bind_weight_source(torch.ones(1, dtype=DTYPE))
    stim.bind_time_source(torch.ones(1, dtype=DTYPE))
    stim.schedule((0, 0), 0.1)
    before = _runtime_schedule_state(stim)

    with pytest.raises((ValueError, IndexError)):
        operation(stim)

    _assert_equal(_runtime_schedule_state(stim), before)


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (lambda s: s.schedule((0, 1, 2), 0.2), IndexError),
        (lambda s: s.schedule(torch.zeros((2, 1), dtype=torch.long), 0.2), IndexError),
        (lambda s: s.schedule("bad", 0.2), TypeError),
        (lambda s: s.schedule(4, 0.2), IndexError),
        (lambda s: s.schedule([], 0.2, weight=2.0), ValueError),
        (lambda s: s.schedule([0, 1], [0.1, 0.2, 0.3], weight=2.0), ValueError),
        (lambda s: s.schedule([0, 1], 0.2, weight=[1.0, 2.0, 3.0]), ValueError),
        (lambda s: s.schedule(0, 0.2, times_ms=0.3), ValueError),
        (lambda s: s.schedule(0), ValueError),
    ],
)
def test_malformed_schedule_requests_are_atomic(operation, error):
    stim = _stim(n=3, batch=2)
    stim.schedule((0, 0), 0.1)
    before = _runtime_schedule_state(stim)

    with pytest.raises(error):
        operation(stim)

    _assert_equal(_runtime_schedule_state(stim), before)


@pytest.mark.parametrize(
    "time",
    [torch.zeros((2, 2)), torch.zeros((2, 3, 1, 1))],
)
def test_forward_rejects_ambiguous_multidimensional_time_shapes_atomically(time):
    stim = _stim(n=2, batch=(2, 3))
    before = _runtime_schedule_state(stim)
    with pytest.raises(ValueError, match="cannot be broadcast"):
        stim.forward(time)
    _assert_equal(_runtime_schedule_state(stim), before)


@pytest.mark.parametrize("batch", [(), 0, -2, (2, 0), "bad", 1.5])
def test_batch_rejects_invalid_dimensions_before_mutation(batch):
    stim = _stim(n=2)
    before = _runtime_schedule_state(stim)
    old_shape = stim.shape

    with pytest.raises((TypeError, ValueError)):
        stim.batch(batch)

    assert stim.shape == old_shape
    _assert_equal(_runtime_schedule_state(stim), before)


def test_tensor_batch_dimensions_preserve_generator_axis_and_schedule_targets():
    stim = _stim(n=2)
    stim.schedule(1, 0.2, weight=2.0)
    stim.batch(torch.tensor([2, 3]))

    assert stim.shape == (2, 3, 2)
    assert stim.sched_flat_idx.tolist() == [1, 3, 5, 7, 9, 11]
