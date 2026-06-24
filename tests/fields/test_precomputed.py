import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from dendra.models.fields.precomputed import (
    EfieldInterpolate3DScattered,
    PreComputedInterpolate1D,
    PreComputedInterpolate3DScattered,
)


class DummyModel:
    def __init__(self, x, y=None, z=None, *, dtype=torch.float32, device="cpu"):
        self.x = torch.as_tensor(x, dtype=dtype, device=device)
        self.y = (
            torch.zeros_like(self.x)
            if y is None
            else torch.as_tensor(y, dtype=dtype, device=device)
        )
        self.z = (
            torch.zeros_like(self.x)
            if z is None
            else torch.as_tensor(z, dtype=dtype, device=device)
        )
        self.graph = None

    def device(self):
        return self.x.device

    def dtype(self):
        return self.x.dtype


# -----------------------------------------------------------------------------
# Helper factories
# -----------------------------------------------------------------------------


def _rand_xyz(
    N: int, *, low: float = -1_000.0, high: float = 1_000.0, dtype=torch.float32
):
    """Random Nx3 coordinates in the given range."""
    return torch.empty(N, 3, dtype=dtype).uniform_(low, high)


def _rand_vec(N: int, *, dtype=torch.float32):
    """Random Nx3 vectors (standard normal)."""
    return torch.empty(N, 3, dtype=dtype).normal_()


# -----------------------------------------------------------------------------
# PreComputedInterpolate1D
# -----------------------------------------------------------------------------


def test_interpolate1d_exact_recovery_at_sample_points():
    """Interpolating at tabulated coordinates should reproduce the LUT row."""
    x = np.linspace(-5.0, 5.0, 11)
    fem = np.sin(x)
    pc = PreComputedInterpolate1D(
        data=fem[None, :],
        x=x,
        outside="clamp",
    )
    model = DummyModel(
        torch.as_tensor(x, dtype=torch.float64).reshape(1, -1), dtype=torch.float64
    )
    out = pc(model)
    torch.testing.assert_close(
        out, torch.as_tensor(fem, dtype=torch.float64).reshape(1, -1)
    )


def test_interpolate1d_outside_zero_policy():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([[0.0, 1.0, 0.0]])
    pc = PreComputedInterpolate1D(data=y, x=x, outside="zero")
    model = DummyModel(torch.tensor([[-1.0, 0.5, 3.0]]))
    out = pc(model)
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.5, 0.0]]))


def test_interpolate1d_invalid_truncate_raises():
    """truncate accepts fractions or percentages below 100, not >=100%."""
    with pytest.raises(ValueError):
        PreComputedInterpolate1D(
            data=np.zeros((1, 10)),
            x=np.arange(10),
            truncate=100.0,
        )


# -----------------------------------------------------------------------------
# PreComputedInterpolate3DScattered scalar wrapper
# -----------------------------------------------------------------------------


def test_precomputed3d_scattered_scalar_shape():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    field = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    interp = PreComputedInterpolate3DScattered(
        xyz, field, method="nearest", k=1, knn_backend="torch"
    )
    model = DummyModel(
        x=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        y=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        z=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        dtype=torch.float64,
    )
    out = interp(model)
    assert out.shape == (1, 2)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0]], dtype=torch.float64))


# -----------------------------------------------------------------------------
# EfieldInterpolate3DScattered — property-based test against manual IDW
# -----------------------------------------------------------------------------


def _manual_idw(dist2: np.ndarray, vecs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Reference inverse-distance weighting (numpy implementation)."""
    w = 1.0 / (dist2 + eps)
    w = w / w.sum(axis=-1, keepdims=True)
    return (w[..., None] * vecs).sum(axis=-2)


@given(
    N=st.integers(min_value=5, max_value=25),
    B=st.integers(min_value=1, max_value=3),
    K=st.integers(min_value=1, max_value=4),
    k=st.integers(min_value=1, max_value=5),
)
@settings(deadline=None, max_examples=25)
def test_idw_matches_manual(N, B, K, k):
    """The public scattered E-field wrapper should match an explicit IDW reference."""
    k = min(k, N)
    xyz = _rand_xyz(N, dtype=torch.float64)
    efield = _rand_vec(N, dtype=torch.float64)
    interp = EfieldInterpolate3DScattered(
        xyz,
        efield,
        method="idw",
        power=2.0,
        k=k,
        eps=1e-9,
        knn_backend="torch",
    )

    xq = torch.randn(B, K, dtype=torch.float64)
    yq = torch.randn(B, K, dtype=torch.float64)
    zq = torch.randn(B, K, dtype=torch.float64)

    torch_out = interp._interp(xq, yq, zq)

    xyz_q = torch.stack((xq, yq, zq), dim=-1)  # (B, K, 3)
    dist2 = ((xyz_q.unsqueeze(-2) - xyz) ** 2).sum(-1)  # (B, K, N)
    dist2_k, idx = torch.topk(dist2, k, dim=-1, largest=False)
    vecs = efield[idx]  # (B, K, k, 3)

    ref = torch.as_tensor(
        _manual_idw(dist2_k.cpu().numpy(), vecs.cpu().numpy()), dtype=torch.float64
    )
    assert torch.allclose(torch_out, ref, rtol=1e-6, atol=1e-6)


# -----------------------------------------------------------------------------
# EfieldInterpolate3DScattered — gradients
# -----------------------------------------------------------------------------


def test_efield_gradients():
    """Output should be differentiable w.r.t. query coords inside a fixed kNN set."""
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    efield = torch.eye(3, dtype=torch.float64)
    interp = EfieldInterpolate3DScattered(
        xyz,
        efield,
        method="idw",
        power=2.0,
        k=3,
        eps=1e-9,
        knn_backend="torch",
    )

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
# EfieldInterpolate3DScattered — validation & error handling
# -----------------------------------------------------------------------------


def test_invalid_k_raises():
    xyz = _rand_xyz(5)
    efield = _rand_vec(5)
    with pytest.raises(ValueError):
        EfieldInterpolate3DScattered(xyz, efield, k=0)


def test_invalid_method_raises():
    xyz = _rand_xyz(5)
    efield = _rand_vec(5)
    with pytest.raises(ValueError):
        EfieldInterpolate3DScattered(xyz, efield, method="unsupported")


def test_shape_mismatch_raises():
    xyz = _rand_xyz(3)
    efield = _rand_vec(3)
    interp = EfieldInterpolate3DScattered(xyz, efield, k=3, knn_backend="torch")

    x = torch.zeros(1, 2)
    y = torch.zeros(1, 2)
    z = torch.zeros(1, 3)  # mismatched shape
    with pytest.raises(RuntimeError):
        interp._interp(x, y, z)
