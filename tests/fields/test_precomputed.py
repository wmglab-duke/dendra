import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from axonml.models.fields.precomputed import (
    EfieldInterpolate3D,
    PreComputedExact,
    PreComputedInterpolate1D,
)

# -----------------------------------------------------------------------------
# Helper factories
# -----------------------------------------------------------------------------


def _rand_xyz(
    N: int, *, low: float = -1_000.0, high: float = 1_000.0, dtype=torch.float32
):
    """Random Nx3 coordinates in the given range."""
    rng = torch.empty(N, 3, dtype=dtype).uniform_(low, high)
    return rng


def _rand_vec(N: int, *, dtype=torch.float32):
    """Random Nx3 vectors (standard normal)."""
    rng = torch.empty(N, 3, dtype=dtype).normal_()
    return rng


# -----------------------------------------------------------------------------
# PreComputedExact
# -----------------------------------------------------------------------------


def test_precomputed_exact_get_in_memory():
    """`get_in_memory` should round-trip the data row-by-row."""
    data = np.arange(12).reshape(4, 3)
    pc = PreComputedExact(data, in_memory=True)
    for gid in range(data.shape[0]):
        np.testing.assert_array_equal(pc.get_in_memory(gid), data[gid])


# -----------------------------------------------------------------------------
# PreComputedInterpolate1D
# -----------------------------------------------------------------------------


def test_interpolate1d_exact_recovery():
    """Interpolating at sample points should reproduce the FEM vector exactly."""
    y = np.linspace(-5.0, 5.0, 11)
    fem = np.sin(y)
    data = fem[None, :]  # shape (1, len(y)) so gid==0
    pc = PreComputedInterpolate1D(
        data=data,
        y=y,
        method="linear",
        in_memory=True,
    )
    out = pc.interpolate(y, 0)
    np.testing.assert_allclose(out, fem, atol=1e-12, rtol=1e-12)


def test_interpolate1d_truncate_limit():
    """truncate > 0.4 should raise an error."""
    with pytest.raises(ValueError):
        PreComputedInterpolate1D(
            data=np.zeros((1, 10)),
            y=np.arange(10),
            truncate=0.5,  # > 0.4 → invalid
            in_memory=True,
        )


# -----------------------------------------------------------------------------
# EfieldInterpolate3D — core helper for manual IDW
# -----------------------------------------------------------------------------


def _manual_idw(dist2: np.ndarray, vecs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Reference inverse‑distance weighting (numpy implementation)."""
    w = 1.0 / (dist2 + eps)
    w = w / w.sum(axis=-1, keepdims=True)
    return (w[..., None] * vecs).sum(axis=-2)


# -----------------------------------------------------------------------------
# EfieldInterpolate3D — property‑based test against manual IDW
# -----------------------------------------------------------------------------


@given(
    N=st.integers(min_value=5, max_value=25),
    B=st.integers(min_value=1, max_value=3),
    K=st.integers(min_value=1, max_value=4),
    k=st.integers(min_value=1, max_value=5),
)
@settings(deadline=None, max_examples=25)
def test_idw_matches_manual(N, B, K, k):
    """`_interp` should match a pure‑numpy reference implementation."""
    k = min(k, N)
    xyz = _rand_xyz(N, dtype=torch.float64)
    efield = _rand_vec(N, dtype=torch.float64)
    interp = EfieldInterpolate3D(xyz, efield, k=k, eps=1e-9)

    xq = torch.randn(B, K, dtype=torch.float64)
    yq = torch.randn(B, K, dtype=torch.float64)
    zq = torch.randn(B, K, dtype=torch.float64)

    # Python name‑mangling: __interp → _Class__interp
    torch_out = interp._EfieldInterpolate3D__interp(xq, yq, zq)

    xyz_q = torch.stack((xq, yq, zq), dim=-1)  # (B, K, 3)
    dist2 = ((xyz_q.unsqueeze(-2) - xyz) ** 2).sum(-1)  # (B, K, N)
    dist2_k, idx = torch.topk(dist2, k, dim=-1, largest=False)
    vecs = efield[idx]  # (B, K, k, 3)

    ref = torch.as_tensor(
        _manual_idw(dist2_k.cpu().numpy(), vecs.cpu().numpy()), dtype=torch.float64
    )
    assert torch.allclose(torch_out, ref, rtol=1e-6, atol=1e-6)


# -----------------------------------------------------------------------------
# EfieldInterpolate3D — gradients
# -----------------------------------------------------------------------------


def test_efield_gradients():
    """Output should be differentiable w.r.t. query coords."""
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    efield = torch.eye(3, dtype=torch.float64)
    interp = EfieldInterpolate3D(xyz, efield, k=3, eps=1e-9)

    xq = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    yq = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    zq = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)

    out = interp._interp(xq, yq, zq)
    loss = out.pow(2).sum()
    loss.backward()

    for t in (xq, yq, zq):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()


# -----------------------------------------------------------------------------
# EfieldInterpolate3D — validation & error handling
# -----------------------------------------------------------------------------


def test_invalid_k_raises():
    xyz = _rand_xyz(5)
    efield = _rand_vec(5)
    with pytest.raises(ValueError):
        EfieldInterpolate3D(xyz, efield, k=0)  # k must be ≥1 or None


def test_invalid_chunksize_raises():
    xyz = _rand_xyz(5)
    efield = _rand_vec(5)
    with pytest.raises(ValueError):
        EfieldInterpolate3D(xyz, efield, chunksize=0)


def test_shape_mismatch_raises():
    xyz = _rand_xyz(3)
    efield = _rand_vec(3)
    interp = EfieldInterpolate3D(xyz, efield, k=3)

    x = torch.zeros(1, 2)
    y = torch.zeros(1, 2)
    z = torch.zeros(1, 3)  # mismatched shape
    with pytest.raises(RuntimeError):
        interp._interp(x, y, z)
