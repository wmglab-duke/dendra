# tests/test_helpers.py

from __future__ import annotations

import re
import time

import numpy as np
import pytest
import torch

import dendra as dn
import dendra.helpers as H

# --------------------------------------------------------------------------
#  basic utilities
# --------------------------------------------------------------------------


def test_getenv_caching(monkeypatch):
    """getenv should return casted values and honour lru_cache semantics."""
    monkeypatch.setenv("DENDRA_TEST_ENV", "42")
    H.getenv.cache_clear()

    assert H.getenv("DENDRA_TEST_ENV", 0) == 42  # first read
    monkeypatch.setenv("DENDRA_TEST_ENV", "314159")  # would change, but cached
    assert H.getenv("DENDRA_TEST_ENV", 0) == 42  # still old

    H.getenv.cache_clear()
    assert H.getenv("DENDRA_TEST_ENV", 0) == 314_159  # after clearing


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


def test_runtime_contract_validation_context_is_normalized_and_atomic():
    original = H.RUNTIME_CONTRACT_VALIDATION.value

    with H.ctx(RUNTIME_CONTRACT_VALIDATION=" STRICT "):
        assert H.RUNTIME_CONTRACT_VALIDATION.value == "strict"
        assert H.current_runtime_contract_validation() == "strict"
        assert dn.RUNTIME_CONTRACT_VALIDATION is H.RUNTIME_CONTRACT_VALIDATION

    assert H.RUNTIME_CONTRACT_VALIDATION.value == original
    assert H.normalize_runtime_contract_validation("Versioned") == "versioned"

    with pytest.raises(ValueError, match="versioned.*strict.*initialize"):
        with H.ctx(RUNTIME_CONTRACT_VALIDATION="unsafe"):
            pass
    assert H.RUNTIME_CONTRACT_VALIDATION.value == original


def test_native_extension_policy_context_is_normalized_and_atomic():
    original = H.NATIVE_EXTENSION_POLICY.value

    with H.ctx(NATIVE_EXTENSION_POLICY=" REQUIRE "):
        assert H.NATIVE_EXTENSION_POLICY.value == "require"
        assert H.current_native_extension_policy() == "require"
        assert dn.NATIVE_EXTENSION_POLICY is H.NATIVE_EXTENSION_POLICY

    assert H.NATIVE_EXTENSION_POLICY.value == original
    assert H.normalize_native_extension_policy("Warn") == "warn"

    with pytest.raises(ValueError, match="fallback.*warn.*require"):
        with H.ctx(NATIVE_EXTENSION_POLICY="silent"):
            pass
    assert H.NATIVE_EXTENSION_POLICY.value == original


def test_explicit_model_dtype_overrides_context_default():
    with H.ctx(DTYPE="float32"):
        population = dn.Population(N=1, C=1, dtype=torch.float64)
        netstim = dn.NetStim(N=1, dtype=torch.float64)

    assert population.v.dtype == torch.float64
    assert netstim.interval.rho.dtype == torch.float64
    assert netstim.start.dtype == torch.float64


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
#  allow_tf32 (only check when CUDA available)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("state", [True, False])
def test_allow_tf32(state):
    """H.allow_tf32 should set CUDA matmul/cuDNN precision switches consistently."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", torch.__version__)
    version = tuple(map(int, match.groups())) if match else (0, 0, 0)

    if version >= (2, 9, 0):
        orig_matmul = torch.backends.cuda.matmul.fp32_precision
        orig_conv = torch.backends.cudnn.conv.fp32_precision
        orig_rnn = torch.backends.cudnn.rnn.fp32_precision
        expected = "tf32" if state else "ieee"

        try:
            H.allow_tf32(state)
            assert torch.backends.cuda.matmul.fp32_precision == expected
            assert torch.backends.cudnn.conv.fp32_precision == expected
            assert torch.backends.cudnn.rnn.fp32_precision == expected
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig_matmul
            torch.backends.cudnn.conv.fp32_precision = orig_conv
            torch.backends.cudnn.rnn.fp32_precision = orig_rnn
    else:
        orig_matmul = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn = torch.backends.cudnn.allow_tf32

        try:
            H.allow_tf32(state)
            assert torch.backends.cuda.matmul.allow_tf32 is state
            assert torch.backends.cudnn.allow_tf32 is state
        finally:
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul
            torch.backends.cudnn.allow_tf32 = orig_cudnn


# --------------------------------------------------------------------------
#  tic / toc timing helpers
# --------------------------------------------------------------------------


def test_tic_toc_and_stack_error(caplog):
    caplog.set_level("INFO", logger="dendra")

    H.tic("start")
    time.sleep(0.001)
    elapsed = H.toc("stop", log=False)
    assert 0 < elapsed < 1

    # Empty stack should trigger error log but not raise
    H.toc(log=False)
    assert any("tic() before you toc()" in rec.message for rec in caplog.records)
