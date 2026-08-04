import networkx as nx
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from dendra.models.fields.precomputed import (
    EfieldInterpolate3DMesh,
    EfieldInterpolate3DRect,
    EfieldInterpolate3DScattered,
    PreComputedInterpolate1D,
    PreComputedInterpolate3DMesh,
    PreComputedInterpolate3DRect,
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


def test_interpolate1d_population_row_bank_repeats_over_rank4_batches():
    x_table = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    data = x_table + torch.tensor([[0.0], [10.0]], dtype=torch.float64)
    field = PreComputedInterpolate1D(data=data, x=x_table, outside="clamp")
    query = torch.tensor([0.25, 1.5], dtype=torch.float64)
    query = query.reshape(1, 1, 1, 2).expand(2, 3, 2, 2).clone()
    model = DummyModel(query, dtype=torch.float64)

    out = field(model)
    expected = query + torch.tensor([0.0, 10.0], dtype=torch.float64).view(1, 1, 2, 1)

    assert out.shape == query.shape
    torch.testing.assert_close(out, expected.expand_as(query))


def test_interpolate1d_rejects_ambiguous_table_bank_without_indices():
    x_table = torch.arange(3.0, dtype=torch.float64).expand(3, -1).clone()
    field = PreComputedInterpolate1D(data=x_table, x=x_table, outside="clamp")
    model = DummyModel(torch.ones(2, 2, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match=r"requires D == 1, D == Q, or D == N"):
        field(model)


def test_interpolate1d_explicit_indices_still_map_arbitrary_table_banks():
    x_table = torch.arange(3.0, dtype=torch.float64).expand(3, -1).clone()
    data = x_table + torch.tensor([[0.0], [10.0], [20.0]], dtype=torch.float64)
    field = PreComputedInterpolate1D(data=data, x=x_table, outside="clamp")
    query = torch.ones(2, 2, 2, dtype=torch.float64)
    indices = torch.tensor([2, 1, 0, 2])

    out = field(DummyModel(query, dtype=torch.float64), indices=indices)
    expected = query.reshape(4, 2) + torch.tensor([20.0, 10.0, 0.0, 20.0])[:, None]

    torch.testing.assert_close(out, expected.reshape_as(query))


def test_interpolate1d_rejects_empty_explicit_indices():
    x_table = torch.arange(3.0, dtype=torch.float64).reshape(1, -1)
    field = PreComputedInterpolate1D(data=x_table, x=x_table, outside="clamp")
    model = DummyModel(torch.ones(2, 1, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="at least one LUT-row index"):
        field(model, indices=[])


def _chain_graph(n):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((i, i + 1) for i in range(n - 1))
    return graph


def _rank4_model(dtype=torch.float64):
    x = torch.tensor([0.0, 0.5, 1.0], dtype=dtype)
    x = x.reshape(1, 1, 1, 3).expand(2, 2, 1, 3).clone()
    model = DummyModel(x, torch.zeros_like(x), torch.zeros_like(x), dtype=dtype)
    model.graph = _chain_graph(3)
    return model


def _flatten_model(model):
    flat = DummyModel(
        model.x.reshape(-1, model.x.shape[-1]),
        model.y.reshape(-1, model.y.shape[-1]),
        model.z.reshape(-1, model.z.shape[-1]),
        dtype=model.x.dtype,
    )
    flat.graph = model.graph
    return flat


def test_rectilinear_and_scattered_scalar_fields_preserve_rank4_shape():
    axis = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    scalar_rect = PreComputedInterpolate3DRect(
        axis, axis, axis, torch.ones(2, 2, 2, dtype=torch.float64)
    )
    xyz = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    scalar_scattered = PreComputedInterpolate3DScattered(
        xyz,
        torch.ones(3, 1, dtype=torch.float64),
        method="nearest",
        k=1,
        knn_backend="torch",
    )
    model = _rank4_model()

    for field in (scalar_rect, scalar_scattered):
        out = field(model)
        flat = field(_flatten_model(model)).reshape_as(out)
        assert out.shape == model.x.shape
        torch.testing.assert_close(out, flat)


def test_rectilinear_and_scattered_vector_fields_preserve_rank4_shape():
    axis = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    rect_values = vector.view(1, 1, 1, 3).expand(2, 2, 2, 3).clone()
    vector_rect = EfieldInterpolate3DRect(axis, axis, axis, rect_values)
    xyz = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    vector_scattered = EfieldInterpolate3DScattered(
        xyz,
        vector.expand(3, 3).clone(),
        method="nearest",
        k=1,
        knn_backend="torch",
    )
    model = _rank4_model()

    for field in (vector_rect, vector_scattered):
        out = field(model)
        flat = field(_flatten_model(model)).reshape_as(out)
        assert out.shape == model.x.shape
        torch.testing.assert_close(out, flat)


class _ConstantMeshInterpolator(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values))

    def forward(self, xyz_q, *, squeeze=False):
        return self.values.reshape(1, -1).expand(xyz_q.shape[0], -1)


def test_mesh_scalar_and_vector_fields_preserve_rank4_shape():
    scalar = PreComputedInterpolate3DMesh(_ConstantMeshInterpolator([2.0]))
    vector = EfieldInterpolate3DMesh(_ConstantMeshInterpolator([1.0, 2.0, 3.0]))
    model = _rank4_model(dtype=torch.float32)

    scalar_out = scalar(model)
    vector_out = vector(model)
    scalar_flat = scalar(_flatten_model(model)).reshape_as(scalar_out)
    vector_flat = vector(_flatten_model(model)).reshape_as(vector_out)

    assert scalar_out.shape == model.x.shape
    assert vector_out.shape == model.x.shape
    torch.testing.assert_close(scalar_out, scalar_flat)
    torch.testing.assert_close(vector_out, vector_flat)
