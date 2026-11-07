"""Shared connection initialization retains native reset and clock policies."""

import pytest
import torch

import dendra as dn
from dendra.models.networks._connection_initialization import (
    continuous_initial_step,
    continuous_runtime_reset,
    event_initial_step,
    event_runtime_reset,
)
from dendra.models.networks.netcon import ContinuousCon, NetCon
from tests.func.test_continuous_con import _connection
from tests.func.test_netcon import _clone, _close, _netcon

pytestmark = pytest.mark.cpu


def _dirty(state):
    result = {
        name: value.new_full(value.shape, True if value.dtype is torch.bool else 2)
        for name, value in state.items()
    }
    result["current_time_step"] = torch.full_like(
        state["current_time_step"], state["delivery_buffer"].shape[0] - 1
    )
    return result


@pytest.mark.parametrize("tracked", (False, True))
@pytest.mark.parametrize("raw", (False, True))
@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("t", (0.0, 0.15, -0.15))
def test_event_initial_state_matches_native_and_preserves_parameters(
    tracked, raw, training, t
):
    net, native = _netcon(tracked=tracked, threshold=None if raw else 0.0, batch=2)
    if training:
        native.set_diff_config(
            diff_delays=False, diff_spiking=not tracked, diff_scheduled_times=False
        )
    native.train(training)
    native.initialize(reinit_weights=False, reinit_delays=False)
    projection, tensors = dn.func.make_functional_netcon(native)
    state = _dirty(tensors.state)
    state["delivery_buffer"] = state["delivery_buffer"] * tensors.weights.sum()
    before, source_before = _clone(state), _clone(native.state_dict_for_checkpoint())
    weights = native.weight.w.detach().clone()
    result = projection.initial_state(state, t=torch.tensor(t, dtype=torch.float64))
    _close(state, before)
    _close(native.state_dict_for_checkpoint(), source_before)
    assert all(not value.requires_grad for value in result.values())
    projection.commit_state_(native, state)
    net._reset_runtime_clock(t)
    native.initialize(reinit_weights=False, reinit_delays=False)
    _close(result, native.state_dict_for_checkpoint())
    torch.testing.assert_close(native.weight.w, weights)


@pytest.mark.parametrize("delays", ((0.0,) * 4, (0.1,) * 4, (0.0, 0.1, 0.0, 0.2)))
@pytest.mark.parametrize("t", (0.0, 0.15, -0.15))
def test_continuous_initial_state_matches_native_and_clears_presence(delays, t):
    net, native = _connection(delays, batch=2)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    state = _dirty(tensors.state)
    state["delivery_buffer"] = state["delivery_buffer"] * tensors.weights.sum()
    before, source_before = _clone(state), _clone(native.state_dict_for_checkpoint())
    result = projection.initial_state(state, t=torch.tensor(t, dtype=torch.float64))
    _close(state, before)
    _close(native.state_dict_for_checkpoint(), source_before)
    assert all(not value.requires_grad for value in result.values())
    projection.commit_state_(native, state)
    net._reset_runtime_clock(t)
    native.initialize(reinit_weights=False, reinit_delays=False)
    _close(result, native.state_dict_for_checkpoint())
    assert not result["delivery_mask"].any()


@pytest.mark.parametrize("kind", ("event", "continuous"))
@pytest.mark.parametrize("clear", (False, True))
def test_native_zero_and_shared_reset_keep_storage_and_optional_traffic(kind, clear):
    if kind == "event":
        _, native = _netcon(threshold=0.0)
        state = _dirty(native.state_dict_for_checkpoint())
        native.restore_dict_from_checkpoint(state)
        reset = event_runtime_reset
    else:
        _, native = _connection()
        state = _dirty(native.state_dict_for_checkpoint())
        native.restore_dict_from_checkpoint(state)
        reset = continuous_runtime_reset
    previous = _clone(state)
    storage = {
        name: value.data_ptr()
        for name, value in native._buffers.items()
        if torch.is_tensor(value)
    }
    expected = reset(state, clear_deliveries=clear)
    _close(state, previous)
    if kind == "event":
        native.zero(clear_delivery_buffers=clear)
    else:
        native.zero(clear_deliveries=clear)
    _close(native.state_dict_for_checkpoint(), expected)
    for name, address in storage.items():
        assert native._buffers[name].data_ptr() == address
    if not clear:
        torch.testing.assert_close(
            expected["delivery_buffer"], previous["delivery_buffer"]
        )
        if kind == "event":
            torch.testing.assert_close(expected["has_spiked"], previous["has_spiked"])
            torch.testing.assert_close(expected["global_step"], previous["global_step"])


def test_event_and_continuous_start_steps_retain_distinct_native_precision():
    time = torch.tensor(0.15, dtype=torch.float64)
    dt = torch.tensor(0.1, dtype=torch.float32)
    reference = torch.zeros(1, dtype=torch.long)
    event = event_initial_step(time, dt, reference)
    continuous = continuous_initial_step(time, dt, reference, dtype=torch.float32)
    assert event.item() == round(float(time) / float(dt)) == 1
    assert continuous.item() == 2


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_event_start_step_transfers_before_float64_clock_calculation():
    dt = torch.tensor(0.1, device="mps", dtype=torch.float32)
    reference = torch.zeros(1, device="mps", dtype=torch.long)
    for time in (
        0.15,
        torch.tensor(0.15, device="cpu", dtype=torch.float64),
        torch.tensor(0.15, device="mps", dtype=torch.float32),
    ):
        step = event_initial_step(time, dt, reference)
        assert step.device == reference.device and step.dtype == reference.dtype
        assert step.shape == reference.shape
        assert step.item() == round(float(time) / float(dt))


@pytest.mark.parametrize("t", (-1e20, 1e20))
def test_event_start_step_rejects_overflow_as_native_integer_conversion_does(t):
    with pytest.raises(OverflowError):
        event_initial_step(
            t,
            torch.tensor(0.1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.long),
        )


def test_event_start_step_rejects_nan_as_native_integer_conversion_does():
    with pytest.raises(ValueError, match="NaN"):
        event_initial_step(
            float("nan"),
            torch.tensor(0.1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.long),
        )


@pytest.mark.parametrize("kind", ("event", "continuous"))
@pytest.mark.parametrize("lanes", (0, 3))
def test_initial_state_vmap_supports_starting_time_and_empty_lanes(kind, lanes):
    _, native = _netcon() if kind == "event" else _connection()
    projection, tensors = (
        dn.func.make_functional_netcon(native)
        if kind == "event"
        else dn.func.make_functional_continuous_con(native)
    )
    times = torch.tensor([-0.15, 0.0, 0.35], dtype=torch.float64)[:lanes]
    actual = torch.vmap(lambda time: projection.initial_state(tensors.state, t=time))(
        times
    )
    assert actual["global_step"].shape == (lanes, 1)
    if lanes:
        expected = torch.stack(
            [
                projection.initial_state(tensors.state, t=time)["global_step"]
                for time in times
            ]
        )
        torch.testing.assert_close(actual["global_step"], expected)


@pytest.mark.parametrize("kind", ("event", "continuous"))
def test_initial_state_fullgraph_compilation(kind):
    _, native = _netcon() if kind == "event" else _connection()
    projection, tensors = (
        dn.func.make_functional_netcon(native)
        if kind == "event"
        else dn.func.make_functional_continuous_con(native)
    )
    execute = torch.compile(
        lambda state, t: projection.initial_state(state, t=t),
        backend="aot_eager",
        fullgraph=True,
    )
    t = torch.tensor(0.35, dtype=torch.float64)
    _close(execute(tensors.state, t), projection.initial_state(tensors.state, t=t))


@pytest.mark.parametrize("time", (-1e20, 1e20))
def test_compiled_event_initial_state_rejects_absolute_step_overflow(time):
    _, native = _netcon()
    projection, tensors = dn.func.make_functional_netcon(native)
    execute = torch.compile(
        lambda state, t: projection.initial_state(state, t=t),
        backend="aot_eager",
        fullgraph=True,
    )
    valid = torch.tensor(0.35, dtype=torch.float64)
    _close(
        execute(tensors.state, valid),
        projection.initial_state(tensors.state, t=valid),
    )
    with pytest.raises(RuntimeError, match="outside the Int64 range"):
        execute(tensors.state, valid.new_tensor(time))


@pytest.mark.parametrize(
    "kind,method",
    (
        ("event", "initialize"),
        ("event", "zero"),
        ("event", "detach"),
        ("event", "_clear_sparse_calendar"),
        ("continuous", "initialize"),
        ("continuous", "zero"),
        ("continuous", "detach"),
        ("continuous", "_configure_advance_impl"),
    ),
)
def test_custom_initialization_adapter_is_rejected_without_blocking_transition(
    kind, method, monkeypatch
):
    _, native = _netcon() if kind == "event" else _connection()
    projection, tensors = (
        dn.func.make_functional_netcon(native)
        if kind == "event"
        else dn.func.make_functional_continuous_con(native)
    )
    owner = NetCon if kind == "event" else ContinuousCon
    original = getattr(owner, method)
    monkeypatch.setattr(
        owner, method, lambda *args, **kwargs: original(*args, **kwargs)
    )
    projection._validate_source()
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        projection.initial_state(tensors.state, t=0.0)
