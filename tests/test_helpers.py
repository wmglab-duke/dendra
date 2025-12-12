# tests/test_helpers.py

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import axonml.helpers as H

# --------------------------------------------------------------------------
#  basic utilities
# --------------------------------------------------------------------------


def test_getenv_caching(monkeypatch):
    """getenv should return casted values and honour lru_cache semantics."""
    monkeypatch.setenv("AXONML_TEST_ENV", "42")
    H.getenv.cache_clear()

    assert H.getenv("AXONML_TEST_ENV", 0) == 42  # first read
    monkeypatch.setenv("AXONML_TEST_ENV", "314159")  # would change, but cached
    assert H.getenv("AXONML_TEST_ENV", 0) == 42  # still old

    H.getenv.cache_clear()
    assert H.getenv("AXONML_TEST_ENV", 0) == 314_159  # after clearing


def test_classproperty():
    class Foo:
        _v = 7

        @H.classproperty
        def v(cls):  # noqa: D401
            return cls._v

    f = Foo()
    assert Foo.v == 7 and f.v == 7
    Foo._v = 9
    assert Foo.v == 9 and f.v == 9


def test_contextvar_and_ctx():
    orig_debug = H.DEBUG.value
    with H.ctx(DEBUG=1):
        assert H.DEBUG.value == 1 and bool(H.DEBUG) is True
        assert H.DEBUG >= 1 and H.DEBUG > 0 and H.DEBUG.value == 1
    # restored
    assert H.DEBUG.value == orig_debug


def test_numpify():
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    a = H.numpify(t)
    assert isinstance(a, np.ndarray) and np.allclose(a, [1, 2])

    plain = H.numpify([3, 4])
    assert np.array_equal(plain, np.asarray([3, 4]))


# --------------------------------------------------------------------------
#  linear-algebra helpers (op_sc/op_mc/ve_from_s_t)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("multicontact", [False, True])
def test_ve_from_s_t_shapes_and_values(multicontact):
    # tiny, deterministic example
    space = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # (1, N_s)
    time = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)  # (1, N_t)
    if multicontact:
        space = space.unsqueeze(0)  # (1, 1, N_s) for multicontact
        time = time.unsqueeze(0)  # (1, 1, N_t) for multicontact
    n = 2
    got = H.ve_from_s_t(space, time, n, device="cpu", multicontact=multicontact)

    if multicontact:
        space = space.squeeze(0)  # (N_s,)
        time = time.squeeze(0)  # (N_t,)

    # expected outer-product: t_dim × n × s_dim
    expected = torch.stack(
        [space.repeat(n, 1) * t for t in time.squeeze()], dim=0
    )  # shape (N_t, n, N_s)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected)


# --------------------------------------------------------------------------
#  1-D interpolation  (scipy-like behaviour)
# --------------------------------------------------------------------------


def _numpy_interp_like(x, y, xnew, *, left=None, right=None):
    """Helper that mimics the left/right handling of helpers.interp1d."""
    if left is None:
        left = y[0]
    if right is None:
        right = y[-1]
    return np.interp(xnew, x, y, left=left, right=right)


float_arrays = (
    st.lists(
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
        unique=True,
    )
    .map(lambda xs: sorted(xs))
    .filter(  # every neighbour must survive FP32 quantisation
        lambda xs: min(
            np.float32(xs[i + 1]) - np.float32(xs[i])  # cast *then* diff
            for i in range(len(xs) - 1)
        )
        > 1e-6
    )
)


@given(
    float_arrays,
    st.lists(
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    ),
)
@settings(max_examples=80)
def test_interp1d_matches_numpy(x_list, y_list):
    """Inside the domain helpers.interp1d must match exact linear result."""
    assume_len = min(len(x_list), len(y_list))
    x = torch.tensor(x_list[:assume_len], dtype=torch.float32)
    y = torch.tensor(y_list[:assume_len], dtype=torch.float32)
    xnew = torch.linspace(float(x.min()), float(x.max()), steps=assume_len * 2)

    got = H.interp1d(x, y, xnew)
    want = torch.tensor(
        _numpy_interp_like(x.numpy(), y.numpy(), xnew.numpy()), dtype=torch.float32
    )

    assert torch.allclose(got, want, atol=2e-5, rtol=1e-4)


def test_interp1d_out_of_bounds_and_grad():
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=False)
    y = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
    xq = torch.tensor([-1.0, 0.5, 3.0])

    # --- vanilla version: clamps to edge values
    yq = H.interp1d(x, y, xq)
    assert (yq[0][0] == y[0]) and (yq[0][-1] == y[-1])

    # gradient flows
    yq.sum().backward()
    assert y.grad is not None and y.grad.shape == y.shape

    # --- zero-outside version
    yqz = H.interp1d_z(x, y.detach(), xq)
    assert yqz[0][0] == 0 and yqz[0][-1] == 0


# --------------------------------------------------------------------------
#  allow_tf32 (only check when CUDA available)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("state", [True, False])
def test_allow_tf32(state):
    """H.allow_tf32 should set *both* CUDA matmul and cuDNN flags consistently."""
    orig_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_cudnn = torch.backends.cudnn.allow_tf32

    try:
        H.allow_tf32(state)
        assert torch.backends.cuda.matmul.allow_tf32 is state
        assert torch.backends.cudnn.allow_tf32 is state
    finally:
        # always restore original state, even if assertions fail
        torch.backends.cuda.matmul.allow_tf32 = orig_matmul
        torch.backends.cudnn.allow_tf32 = orig_cudnn


# --------------------------------------------------------------------------
#  tic / toc timing helpers
# --------------------------------------------------------------------------


def test_tic_toc_and_stack_error(caplog):
    caplog.set_level("INFO", logger="axonml")

    H.tic("start")
    time.sleep(0.001)
    elapsed = H.toc("stop", log=False)
    assert 0 < elapsed < 1

    # Empty stack should trigger error log but not raise
    H.toc(log=False)
    assert any("tic() before you toc()" in rec.message for rec in caplog.records)
