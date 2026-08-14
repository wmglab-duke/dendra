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


def test_ctx_propagates_dtype_and_device_to_torch_factories():
    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()

    with dn.ctx(DTYPE=torch.float64, DEVICE="meta"):
        implicit = torch.zeros(5)
        inferred = torch.tensor([1.25, 2.5])
        explicit = torch.zeros(5, dtype=torch.float32, device="cpu")

        assert implicit.dtype == inferred.dtype == torch.float64
        assert implicit.device.type == inferred.device.type == "meta"
        assert explicit.dtype == torch.float32
        assert explicit.device.type == "cpu"
        assert H.current_dtype() == torch.float64
        assert H.current_device() == torch.device("meta")

    assert torch.get_default_dtype() == original_dtype
    assert torch.get_default_device() == original_device


def test_ctx_torch_defaults_nest_and_restore_after_an_exception():
    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()

    with pytest.raises(RuntimeError, match="leave outer context"):
        with H.ctx(DTYPE=torch.float64, DEVICE="meta"):
            assert torch.zeros(1).dtype == torch.float64
            assert torch.zeros(1).device.type == "meta"

            with H.ctx(DTYPE=torch.float32, DEVICE="cpu"):
                inner = torch.zeros(1)
                assert inner.dtype == torch.float32
                assert inner.device.type == "cpu"

            with H.ctx(DTYPE=None, DEVICE=None):
                defaults = torch.zeros(1)
                assert defaults.dtype == torch.float32
                assert defaults.device.type == "cpu"

            restored_outer = torch.zeros(1)
            assert restored_outer.dtype == torch.float64
            assert restored_outer.device.type == "meta"
            raise RuntimeError("leave outer context")

    assert torch.get_default_dtype() == original_dtype
    assert torch.get_default_device() == original_device


def test_invalid_torch_dtype_context_is_atomic():
    original_debug = H.DEBUG.value
    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()

    with pytest.raises(TypeError, match="floating-point"):
        with H.ctx(DEBUG=1, DEVICE="meta", DTYPE=torch.int64):
            pass

    assert H.DEBUG.value == original_debug
    assert torch.get_default_dtype() == original_dtype
    assert torch.get_default_device() == original_device


def test_ctx_preserves_bfloat16_when_torch_cannot_make_it_the_default(
    monkeypatch,
):
    original_dtype = torch.get_default_dtype()
    set_default_dtype = torch.set_default_dtype

    def reject_bfloat16(dtype):
        if dtype == torch.bfloat16:
            raise TypeError("bfloat16 has no corresponding complex dtype")
        set_default_dtype(dtype)

    monkeypatch.setattr(torch, "set_default_dtype", reject_bfloat16)

    with pytest.warns(UserWarning, match="cannot use torch.bfloat16"):
        with H.ctx(DTYPE=torch.bfloat16):
            assert H.current_dtype() == torch.bfloat16
            assert torch.zeros(1).dtype == original_dtype

    assert torch.get_default_dtype() == original_dtype


def test_ctx_instance_can_be_nested_and_reused_as_a_decorator():
    original_dtype = torch.get_default_dtype()
    context = H.ctx(DTYPE=torch.float64)

    with context:
        assert torch.zeros(1).dtype == torch.float64
        with context:
            assert torch.zeros(1).dtype == torch.float64
        assert torch.zeros(1).dtype == torch.float64

    @context
    def tensor_factory():
        return torch.zeros(1)

    assert tensor_factory().dtype == torch.float64
    assert tensor_factory().dtype == torch.float64
    assert torch.get_default_dtype() == original_dtype


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
