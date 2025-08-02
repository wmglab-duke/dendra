# test_callbacks.py
import math
import types

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st


# ----------------------------------------------------------------------
# Minimal stand-ins that look like a real AxonML model/integrator
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
        raise AttributeError(item)


# ----------------------------------------------------------------------
# import the callbacks you want to test
# ----------------------------------------------------------------------
from axonml.models.callbacks import (
    Active,
    AnomalyDetector,
    APCount,
    Recorder,
    sliding_window_average,
)


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
    assert cb.record.eq(False).all()

    # cross once → becomes True and stays True
    model.v[0, 0] = 5.0
    cb.post_step_hook(model)
    assert cb.record[0].item() is True
    model.v.fill_(-5.0)
    cb.post_step_hook(model)
    assert cb.record[0].item() is True  # should remain latched


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
def test_sliding_window_output_shape(N, C, H, W, window):
    x = torch.randn(N, C, H, W)
    with torch.no_grad():
        out = sliding_window_average(x, window)
    assert out.shape == x.shape
