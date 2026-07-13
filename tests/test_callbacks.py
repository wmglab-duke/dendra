# test_callbacks.py
import math
import types

import matplotlib
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from dendra.models.callbacks import (
    LFP,
    Active,
    ActiveAL,
    ActiveALCount,
    AnomalyDetector,
    APCount,
    Callback,
    CallbackList,
    Raster,
    Recorder,
    RecorderLambda,
    ThresholdCallback,
    sliding_window_average,
)

matplotlib.use("Agg")


# ----------------------------------------------------------------------
# Minimal stand-ins that look like a real Dendra model/integrator
# ----------------------------------------------------------------------
class _DummyIntegrator:
    def __init__(self, n_ax, nc, device):
        self.mech = types.SimpleNamespace()  # touches Recorder mech path
        self.i_membrane = torch.zeros(n_ax, nc, device=device)
        self.imem = True  # LFP requires this attr
        self.mech.hh = types.SimpleNamespace(m=torch.zeros(n_ax, nc, device=device))


class _DummyModel:
    def __init__(self, n_ax=2, nc=8, device="cpu"):
        self._device = torch.device(device)
        self.n_ax, self.nc = n_ax, nc
        self.integrator = _DummyIntegrator(n_ax, nc, self._device)
        self.v = torch.zeros(n_ax, nc, device=self._device)  # membrane voltages
        self.t = 0.0

    # --- attributes that callbacks expect --------------------------------
    def device(self):  # AnomalyDetector / Recorder
        return self._device

    def n(self):  # APCount / Active / Recorder
        return self.n_ax

    # allow Recorder to see arbitrary attrs e.g. 'v'
    def __getattr__(self, item):
        if item == "v":
            return self.v
        if item == "i_membrane":
            return self.integrator.i_membrane
        raise AttributeError(item)


# ----------------------------------------------------------------------
# 1.  AnomalyDetector ----------------------------------------------------
# ----------------------------------------------------------------------
def test_anomaly_detector_flags_nan():
    model = _DummyModel()
    cb = AnomalyDetector()
    cb.pre_loop_hook(model)

    # step with normal data
    cb.post_step_hook(model)
    assert cb.rec.eq(False).all()

    # introduce a NaN in axon-1 → should flag only that row
    model.v[1, 3] = math.nan
    cb.post_step_hook(model)
    assert cb.rec.tolist() == [False, True]

    assert cb.numpy().tolist() == [False, True]
    cb.reset()
    assert cb.rec.eq(False).all()


# ----------------------------------------------------------------------
# 2.  Recorder (basic record, node-indexed, max_only) --------------------
# ----------------------------------------------------------------------
def test_recorder_basic_and_indexed():
    model = _DummyModel(n_ax=3, nc=6)
    rec = Recorder(states=["v"], node_indices=[1, 4])
    rec.pre_loop_hook(model)

    # run 5 time steps with changing voltages
    for _ in range(5):
        model.v = torch.randn_like(model.v)
        rec.post_step_hook(model)

    stack = rec.stack("v")  # shape: (steps+1, n_ax, nodes)
    assert stack.shape == (6, 3, 2)  # pre_loop adds the initial sample
    # make sure the values match what we stored last
    torch.testing.assert_close(stack[-1], model.v[:, [1, 4]])


def test_recorder_max_only():
    model = _DummyModel(n_ax=2, nc=4)
    rec = Recorder(states=["v"], max_only=True)
    rec.pre_loop_hook(model)

    for _ in range(4):
        model.v = torch.randn_like(model.v) * 10
        rec.post_step_hook(model)

    # With max_only the last dim should be 1 (max over nodes)
    assert rec.stack("v").shape == (5, 2, 1)


def test_recorder_mechanism_state_partition_sampling_and_reset():
    model = _DummyModel(n_ax=2, nc=6)
    rec = Recorder(
        states=["v", "hh.m"],
        max_only=True,
        dt=0.2,
        partition=[2, 4],
        sliding_window=2,
    )
    rec.dt = 0.1
    rec.pre_loop_hook(model)
    for value in range(1, 5):
        model.v.fill_(float(value))
        model.integrator.mech.hh.m.fill_(float(value + 10))
        rec.post_step_hook(model)

    assert rec.save_every == 2
    assert rec.stack("v").shape == (3, 2, 2)
    assert rec.stack().shape == (2, 2, 2)
    assert rec.numpy("hh.m").shape == (3, 2, 2)

    rec.post_loop_hook(model)
    assert rec.run_number == 1
    rec.reset()
    assert rec.i == 0
    assert all(not values for values in rec.rec.values())
    rec.close()


@pytest.mark.parametrize(
    "partition, message",
    [([[1, 1]], "1D"), ([], "non-empty"), ([1, 0], "positive")],
)
def test_recorder_rejects_invalid_partitions(partition, message):
    with pytest.raises(ValueError, match=message):
        Recorder(["v"], max_only=True, partition=partition)


def test_recorder_lambda_records_named_derived_values():
    model = _DummyModel(n_ax=2, nc=3)
    rec = RecorderLambda(
        {"mean": lambda m: m.v.mean(dim=-1), "max": lambda m: m.v.max(dim=-1).values}
    )
    rec.pre_loop_hook(model)
    model.v.fill_(2.0)
    rec.post_step_hook(model)

    assert rec.stack("mean").shape == (2, 2)
    assert rec.stack().shape == (2, 2, 2)
    assert set(rec.numpy()) == {"mean", "max"}
    assert rec.numpy("mean").shape == (2, 2)
    assert rec.numpy("missing").size == 0
    rec.reset()
    assert rec.rec == {}


def test_lfp_records_membrane_current_and_validates_imem():
    model = _DummyModel(n_ax=2, nc=3)
    weights = [torch.ones_like(model.v), 2 * torch.ones_like(model.v)]
    lfp = LFP(weights)
    lfp.pre_loop_hook(model)
    model.integrator.i_membrane.fill_(3.0)
    lfp.post_step_hook(model)

    assert lfp.lfp.shape == (2, 2)
    assert torch.equal(lfp.lfp[-1], torch.tensor([18.0, 36.0]))
    assert lfp.numpy().shape == (2, 2)
    assert lfp.t.numel() == 0
    lfp.reset()
    assert lfp._lfp == []

    model.integrator.imem = False
    with pytest.raises(RuntimeError, match="IMEM=1"):
        LFP(weights).pre_loop_hook(model)


# ----------------------------------------------------------------------
# 3.  APCount & Active ---------------------------------------------------
# ----------------------------------------------------------------------
def _drive_square_wave(model, node=0, amps=(1.0, -1.0, 1.0)):
    """Helper: toggle node 0 above/below threshold each call."""
    model.v.fill_(amps[0])
    yield
    model.v.fill_(amps[1])
    yield
    model.v.fill_(amps[2])
    yield


def test_apcount_counts_crossings():
    model = _DummyModel()
    cb = APCount(threshold=0.0, node_check=[0])
    cb.pre_loop_hook(model)

    # two full up–down transitions → 2 spikes expected
    gen = _drive_square_wave(model)
    next(gen)
    cb.post_step_hook(model)  # up   → crosses
    next(gen)
    cb.post_step_hook(model)  # down → resets
    next(gen)
    cb.post_step_hook(model)  # up   → crosses
    assert cb.record[0, 0].item() == 2


def test_active_sets_flag_once():
    model = _DummyModel()
    cb = Active(threshold=1.0, node_check=[0])
    cb.pre_loop_hook(model)

    # initially below threshold
    cb.post_step_hook(model)
    assert cb.is_active().eq(False).all()

    # cross once → becomes active and stays active
    model.v[0, 0] = 5.0
    cb.post_step_hook(model)
    assert cb.is_active()[0].item() is True
    model.v.fill_(-5.0)
    cb.post_step_hook(model)
    assert cb.is_active()[0].item() is True  # should remain latched


def test_callback_list_forwards_every_hook_in_order():
    events = []

    class TrackingCallback(Callback):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def pre_loop_hook(self, model):
            events.append((self.name, "pre_loop"))

        def pre_chunk_hook(self, model, timepoints=None):
            events.append((self.name, "pre_chunk", timepoints))

        def pre_step_hook(self, model):
            events.append((self.name, "pre_step"))

        def post_step_hook(self, model):
            events.append((self.name, "post_step"))

        def post_chunk_hook(self, model, timepoints=None):
            events.append((self.name, "post_chunk", timepoints))

        def post_loop_hook(self, model):
            events.append((self.name, "post_loop"))

    callbacks = CallbackList([TrackingCallback("a"), TrackingCallback("b")])
    model = _DummyModel()
    callbacks.pre_loop_hook(model)
    callbacks.pre_chunk_hook(model, [0, 1])
    callbacks.pre_step_hook(model)
    callbacks.post_step_hook(model)
    callbacks.post_chunk_hook(model, [0, 1])
    callbacks.post_loop_hook(model)

    assert len(callbacks) == 2
    assert [callback.name for callback in callbacks] == ["a", "b"]
    assert events[:2] == [("a", "pre_loop"), ("b", "pre_loop")]
    assert events[-2:] == [("a", "post_loop"), ("b", "post_loop")]


def test_threshold_callback_timing_negative_nodes_and_reset_helpers():
    model = _DummyModel(n_ax=1, nc=4)
    cb = ThresholdCallback(
        threshold=1.0,
        t_start_check=0.2,
        t_end_check=0.5,
        node_check=[0, -1],
        dt=0.1,
    )
    cb.pre_loop_hook(model)
    assert torch.equal(cb.node_check, torch.tensor([0, 3]))
    assert (cb.ind_start, cb.ind_end) == (2, 5)

    cb.record = torch.ones(1)
    cb.state_cache = torch.ones(1, dtype=torch.bool)
    assert cb.numpy().tolist() == [1.0]
    cb.reset()
    assert cb.i == 0 and cb.record is None and cb.state_cache is None

    cb.dt = 0.05
    assert (cb.ind_start, cb.ind_end) == (4, 10)


def test_active_variants_support_partitions_counts_and_inversion():
    record = torch.tensor([[1.0, 0.0, 2.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

    active_nodes = ActiveAL(at_least=2)
    active_nodes.record = record
    assert torch.equal(active_nodes.is_active(), torch.tensor([True, True]))
    assert torch.equal(
        active_nodes.is_active([2, 2]),
        torch.tensor([[False, False], [True, False]]),
    )
    assert active_nodes.numpy([2, 2]).shape == (2, 2)

    inverted = ActiveAL(at_least=2, inv=True)
    inverted.record = record
    assert torch.equal(inverted.is_active(), torch.tensor([False, False]))

    total = ActiveALCount(at_least=3)
    total.record = record
    assert torch.equal(total.is_active(), torch.tensor([True, False]))
    assert total.numpy().tolist() == [True, False]


@pytest.mark.parametrize(
    "partition, message",
    [([[2, 2]], "1D"), ([], "non-empty"), ([2, 1], "sum"), ([1, 3], "at least")],
)
def test_active_partition_validation(partition, message):
    cb = ActiveAL(at_least=2)
    cb.record = torch.ones((1, 4))
    with pytest.raises(ValueError, match=message):
        cb.is_active(partition)

    count = ActiveALCount(at_least=2)
    count.record = torch.ones((1, 4))
    with pytest.raises(ValueError, match=message):
        count.is_active(partition)


def test_raster_records_crossings_and_returns_plot_axes():
    model = _DummyModel(n_ax=2, nc=2)
    raster = Raster(threshold=0.0, node_check=[0], dt=0.1)
    raster.pre_loop_hook(model)

    model.v.fill_(-1.0)
    raster.post_step_hook(model)
    model.v[:, 0] = torch.tensor([1.0, -1.0])
    raster.post_step_hook(model)
    model.v[:, 0] = torch.tensor([-1.0, 1.0])
    raster.post_step_hook(model)

    assert raster.stack().shape == (3, 2, 1)
    assert raster.numpy().sum() == 2
    ax = raster.plot(np.array([1.0, 2.0]), "diameter", cmap="viridis")
    assert ax.get_xlabel() == "Time (ms)"
    assert ax.get_ylabel() == "diameter"


# ----------------------------------------------------------------------
# 4.  sliding_window_average property-based shape test -------------------
# ----------------------------------------------------------------------
@given(
    N=st.integers(min_value=3, max_value=15),
    C=st.integers(min_value=1, max_value=3),
    H=st.integers(min_value=1, max_value=3),
    W=st.integers(min_value=1, max_value=3),
    window=st.integers(min_value=1, max_value=5),
)
@settings(deadline=None)
def test_sliding_window_output_shape(N, C, H, W, window):
    x = torch.randn(N, C, H, W)
    with torch.no_grad():
        out = sliding_window_average(x, window)
    assert out.shape == x.shape


def test_sliding_window_known_values_and_validation():
    x = torch.arange(5.0).reshape(5, 1, 1, 1)
    out = sliding_window_average(x, 3).flatten()
    assert torch.allclose(out, torch.tensor([1 / 3, 1.0, 2.0, 3.0, 11 / 3]))
    assert torch.allclose(sliding_window_average(x.flatten(), 3), out)

    with pytest.raises(ValueError, match="at least 1"):
        sliding_window_average(x, 0)
    with pytest.raises(TypeError, match="PyTorch tensor"):
        sliding_window_average(np.arange(3.0), 2)
    with pytest.raises(ValueError, match="at least one sample"):
        sliding_window_average(torch.empty(0, 2), 2)


def _reference_sliding_window_average(x, window_size):
    pad_left = window_size // 2
    offsets = torch.arange(window_size, device=x.device) - pad_left
    indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1) + offsets
    indices = indices.clamp(0, x.shape[0] - 1)
    windows = x.index_select(0, indices.flatten()).reshape(
        x.shape[0], window_size, *x.shape[1:]
    )
    return windows.mean(dim=1)


@pytest.mark.parametrize("window_size", [1, 2, 3, 4, 8])
def test_sliding_window_matches_edge_replicated_reference(window_size):
    base = torch.arange(2 * 5 * 3, dtype=torch.float64).reshape(2, 5, 3)
    x = base.transpose(0, 1)
    assert not x.is_contiguous()

    actual = sliding_window_average(x, window_size)
    expected = _reference_sliding_window_average(x, window_size)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        (1, [0.0, 1.0, 2.0, 3.0, 4.0]),
        (2, [0.0, 0.5, 1.5, 2.5, 3.5]),
        (3, [1 / 3, 1.0, 2.0, 3.0, 11 / 3]),
        (4, [0.25, 0.75, 1.5, 2.5, 3.25]),
    ],
)
def test_sliding_window_odd_and_even_alignment(window_size, expected):
    x = torch.arange(5.0)
    torch.testing.assert_close(
        sliding_window_average(x, window_size), torch.tensor(expected)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_sliding_window_preserves_cpu_dtype(dtype):
    x = torch.linspace(-1, 1, 12, dtype=dtype).reshape(4, 3)
    result = sliding_window_average(x, 4)
    expected = _reference_sliding_window_average(x, 4)

    assert result.dtype == dtype
    assert result.device == x.device
    torch.testing.assert_close(result, expected)


def test_sliding_window_gradient_matches_reference():
    x = torch.randn(5, 2, 3, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn_like(x)

    result = sliding_window_average(x, 4)
    result.backward(upstream)
    actual_grad = x.grad.detach().clone()

    x.grad = None
    expected = _reference_sliding_window_average(x, 4)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, actual_grad)


@pytest.mark.parametrize("window_size", [1, 2, 5, 8])
def test_sliding_window_singleton_oversized_windows_are_identity(window_size):
    x = torch.tensor([[[-3.25, 7.5]]], dtype=torch.float64, requires_grad=True)

    result = sliding_window_average(x, window_size)

    torch.testing.assert_close(result, x)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sliding_window_preserves_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    x = torch.arange(30.0, device=device).reshape(5, 2, 3)
    result = sliding_window_average(x, 2)

    assert result.device == x.device
    torch.testing.assert_close(result, _reference_sliding_window_average(x, 2))
