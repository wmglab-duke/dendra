from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dendra.utils.interpolation import (
    PreparedInterp1dUniform,
    PreparedInterp3dFEM,
    PreparedInterp3dRect,
    PreparedInterp3dRectUniform,
    PreparedInterp3dScattered,
    interp1d,
    interp1d_uniform,
)

DTYPE = torch.float64


def _affine_grid(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    return 1.0 + 2.0 * xx - 3.0 * yy + 0.5 * zz


def _quadratic_grid(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    return xx.square() + yy.square() + zz.square()


def _cube_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=DTYPE,
    )


# ---------------------------------------------------------------------------
# Dedicated and functional uniform 1D paths
# ---------------------------------------------------------------------------


def test_prepared_interp1d_uniform_sort_outside_override_and_out_buffer():
    interp = PreparedInterp1dUniform(
        torch.tensor([2.0, 0.0, 1.0], dtype=DTYPE),
        torch.tensor([4.0, 0.0, 1.0], dtype=DTYPE),
        outside="clamp",
        sort_xy=True,
        check_uniform=True,
    )
    query = torch.tensor([-1.0, 0.5, 1.5, 3.0], dtype=DTYPE)

    assert torch.allclose(
        interp(query), torch.tensor([0.0, 0.5, 2.5, 4.0], dtype=DTYPE)
    )

    out = torch.empty_like(query)
    result = interp(query, outside="fill", fill_value=-7.0, out=out)
    assert result is out
    assert torch.allclose(out, torch.tensor([-7.0, 0.5, 2.5, -7.0], dtype=DTYPE))

    with pytest.raises(ValueError, match="out has shape"):
        interp(query, out=torch.empty(2, dtype=DTYPE))


def test_prepared_interp1d_uniform_batched_indices_and_learnable_gradient():
    x = torch.tensor([[0.0, 1.0, 2.0], [10.0, 12.0, 14.0]], dtype=DTYPE)
    y = 3.0 * x + 1.0
    interp = PreparedInterp1dUniform(x, y, learnable_y=True, check_uniform=True)
    query = torch.tensor([[11.0, 13.0], [0.5, 1.5]], dtype=DTYPE)

    got = interp(query, indices=[1, 0])
    want = torch.tensor([[34.0, 40.0], [2.5, 5.5]], dtype=DTYPE)
    assert torch.allclose(got, want)

    got.sum().backward()
    assert interp.y.grad is not None
    assert torch.all(interp.y.grad.sum(dim=1) > 0)


def test_prepared_interp1d_uniform_shared_query_selects_multiple_tables():
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=DTYPE)
    y = torch.tensor([[0.0, 1.0, 2.0], [10.0, 12.0, 14.0]], dtype=DTYPE)
    interp = PreparedInterp1dUniform(x, y, exact_clamp=False)

    got = interp(torch.tensor([0.25, 1.75], dtype=DTYPE), indices=[1, 0])
    assert got.shape == (2, 2)
    assert torch.allclose(got, torch.tensor([[10.5, 13.5], [0.25, 1.75]], dtype=DTYPE))


def test_prepared_interp1d_uniform_validation_paths():
    x = torch.tensor([0.0, 1.0, 3.0], dtype=DTYPE)
    y = torch.tensor([0.0, 1.0, 9.0], dtype=DTYPE)
    with pytest.raises(ValueError, match="not uniformly spaced"):
        PreparedInterp1dUniform(x, y, check_uniform=True)
    with pytest.raises(ValueError, match="outside"):
        PreparedInterp1dUniform(x, y, outside="bad")
    with pytest.raises(ValueError, match="same device and have the same dtype"):
        PreparedInterp1dUniform(x, y.float())
    with pytest.raises(TypeError, match="floating-point"):
        PreparedInterp1dUniform(torch.arange(3), torch.arange(3))
    with pytest.raises(ValueError, match="N >= 2"):
        PreparedInterp1dUniform(x[:1], y[:1])

    interp = PreparedInterp1dUniform(
        torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=DTYPE),
        torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="Provide indices"):
        interp(torch.tensor([0.5], dtype=DTYPE))
    with pytest.raises(ValueError, match="length 1"):
        interp(torch.tensor([[0.5]], dtype=DTYPE), indices=[0, 1])
    with pytest.raises(ValueError, match="values must be in"):
        interp(torch.tensor([[0.5]], dtype=DTYPE), indices=[2])
    with pytest.raises(ValueError, match="x_new must be 1D"):
        interp(torch.zeros((1, 1, 1), dtype=DTYPE))
    with pytest.raises(ValueError, match="match device and dtype"):
        interp(torch.tensor([0.5], dtype=torch.float32), indices=[0])
    with pytest.raises(ValueError, match="outside"):
        interp(torch.tensor([0.5], dtype=DTYPE), indices=[0], outside="bad")


def test_interp1d_uniform_broadcast_gradient_and_reshaped_queries():
    x = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE)
    y = torch.tensor(
        [[0.0, 1.0, 2.0], [10.0, 12.0, 14.0]],
        dtype=DTYPE,
        requires_grad=True,
    )
    query = torch.tensor([0.25, 1.5], dtype=DTYPE)

    got = interp1d_uniform(x, y, query, outside="clamp")
    assert torch.allclose(got, torch.tensor([[0.25, 1.5], [10.5, 13.0]], dtype=DTYPE))
    got.sum().backward()
    assert y.grad is not None

    query_2d = torch.tensor([[0.25, 0.75], [1.25, 1.75]], dtype=DTYPE)
    got_2d = interp1d_uniform(x, y.detach()[0], query_2d)
    assert got_2d.shape == query_2d.shape
    assert torch.allclose(got_2d, query_2d)


def test_functional_interp1d_nonuniform_broadcast_and_out_fallback():
    x = torch.tensor([[0.0, 1.0, 3.0], [0.0, 2.0, 5.0]], dtype=DTYPE)
    y = torch.tensor([0.0, 2.0, 6.0], dtype=DTYPE, requires_grad=True)
    query = torch.tensor([0.5, 2.0], dtype=DTYPE)

    # A wrongly sized output is intentionally ignored by this legacy API.
    wrong_out = torch.empty(1, dtype=DTYPE)
    got = interp1d(x, y, query, out=wrong_out, outside="clamp", uniform="never")
    assert got.shape == (2, 2)
    assert torch.allclose(got, torch.tensor([[1.0, 4.0], [0.5, 2.0]], dtype=DTYPE))
    assert got.data_ptr() != wrong_out.data_ptr()
    got.sum().backward()
    assert y.grad is not None

    with pytest.raises(ValueError, match="outside"):
        interp1d(x[0], y.detach(), query, outside="fill")
    with pytest.raises(ValueError, match="uniform"):
        interp1d(x[0], y.detach(), query, uniform="sometimes")


# ---------------------------------------------------------------------------
# Rectilinear 3D uniform paths and discrete Laplacians
# ---------------------------------------------------------------------------


def test_prepared_rect_uniform_sorts_axes_and_recovers_affine_channels():
    x = torch.tensor([2.0, 0.0, 1.0], dtype=DTYPE)
    y = torch.tensor([1.0, -1.0, 0.0], dtype=DTYPE)
    z = torch.tensor([4.0, 0.0, 2.0], dtype=DTYPE)
    scalar = _affine_grid(x, y, z)
    values = torch.stack([scalar, 2.0 * scalar - 4.0], dim=-1)
    interp = PreparedInterp3dRectUniform(
        x, y, z, values, sort_xyz=True, check_uniform=True
    )
    query = torch.tensor([[0.25, -0.5, 1.0], [1.5, 0.5, 3.0]], dtype=DTYPE)

    got = interp(query)
    expected_scalar = 1 + 2 * query[:, 0] - 3 * query[:, 1] + 0.5 * query[:, 2]
    expected = torch.stack([expected_scalar, 2 * expected_scalar - 4], dim=-1)
    assert torch.allclose(got, expected)


def test_prepared_rect_uniform_batched_indices_outside_and_gradient():
    axes = torch.tensor([[0.0, 1.0], [10.0, 12.0]], dtype=DTYPE)
    values = torch.stack(
        [
            _affine_grid(axes[0], axes[0], axes[0]),
            _affine_grid(axes[1], axes[1], axes[1]),
        ]
    )
    interp = PreparedInterp3dRectUniform(
        axes,
        axes,
        axes,
        values,
        learnable_values=True,
        check_uniform=True,
    )
    query = torch.tensor(
        [
            [[11.0, 11.0, 11.0], [30.0, 30.0, 30.0]],
            [[0.25, 0.5, 0.75], [-1.0, 0.0, 0.0]],
        ],
        dtype=DTYPE,
    )
    got = interp(query, indices=[1, 0], outside="fill", fill_value=-9.0)

    assert torch.allclose(
        got, torch.tensor([[[-4.5, -9.0], [0.375, -9.0]]], dtype=DTYPE).reshape(2, 2)
    )
    got.sum().backward()
    assert interp.values.grad is not None
    assert float(interp.values.grad.abs().sum()) > 0


def test_prepared_rect_uniform_shared_points_out_and_index_validation():
    axis = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=DTYPE)
    values = torch.stack(
        [torch.zeros((2, 2, 2), dtype=DTYPE), torch.ones((2, 2, 2), dtype=DTYPE)]
    )
    interp = PreparedInterp3dRectUniform(axis, axis, axis, values)
    query = torch.tensor([[0.5, 0.5, 0.5]], dtype=DTYPE)
    out = torch.empty((2, 1), dtype=DTYPE)

    assert interp(query, indices=[1, 0], out=out) is out
    assert torch.equal(out, torch.tensor([[1.0], [0.0]], dtype=DTYPE))

    with pytest.raises(ValueError, match="Provide indices"):
        interp(query)
    with pytest.raises(ValueError, match="non-empty"):
        interp(query, indices=[])
    with pytest.raises(ValueError, match="out has shape"):
        interp(query, indices=[0], out=torch.empty(2, dtype=DTYPE))


def test_prepared_rect_uniform_validation_paths():
    axis = torch.tensor([0.0, 1.0, 3.0], dtype=DTYPE)
    values = torch.zeros((3, 3, 3), dtype=DTYPE)
    with pytest.raises(ValueError, match="not uniformly spaced"):
        PreparedInterp3dRectUniform(axis, axis, axis, values, check_uniform=True)
    with pytest.raises(ValueError, match="outside"):
        PreparedInterp3dRectUniform(axis, axis, axis, values, outside="bad")
    with pytest.raises(ValueError, match="same ndim"):
        PreparedInterp3dRectUniform(axis, axis[None], axis, values)
    with pytest.raises(ValueError, match="first 3 dims"):
        PreparedInterp3dRectUniform(axis, axis, axis, values[:2])
    with pytest.raises(ValueError, match="at least 2"):
        PreparedInterp3dRectUniform(axis[:1], axis, axis, values[:1])

    interp = PreparedInterp3dRectUniform(axis, axis, axis, values)
    with pytest.raises(ValueError, match="xyz_new"):
        interp(torch.zeros((2, 2), dtype=DTYPE))
    with pytest.raises(ValueError, match="match device and dtype"):
        interp(torch.zeros((2, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="outside"):
        interp(torch.zeros((2, 3), dtype=DTYPE), outside="bad")


@pytest.mark.parametrize("boundary", ["replicate", "one-sided"])
def test_rect_laplacian_quadratic_known_answer_all_nodes(boundary: str):
    x = torch.tensor([0.0, 0.5, 2.0, 5.0], dtype=DTYPE)
    y = torch.tensor([-2.0, -0.5, 1.0, 4.0], dtype=DTYPE)
    z = torch.tensor([0.0, 1.0, 3.0, 7.0], dtype=DTYPE)
    interp = PreparedInterp3dRect(x, y, z, _quadratic_grid(x, y, z), uniform="never")

    lap = interp.laplacian_values(boundary=boundary)
    assert lap.shape == (4, 4, 4)
    assert torch.allclose(lap, torch.full_like(lap, 6.0), atol=1e-12)


def test_rect_laplacian_valid_and_zero_boundary_semantics():
    axis = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=DTYPE)
    interp = PreparedInterp3dRect(
        axis, axis, axis, _quadratic_grid(axis, axis, axis), uniform="never"
    )

    valid = interp.laplacian_values(boundary="valid")
    assert valid.shape == (2, 2, 2)
    assert torch.allclose(valid, torch.full_like(valid, 6.0))

    zero = interp.laplacian_values(boundary="zero")
    assert torch.allclose(
        zero[1:-1, 1:-1, 1:-1],
        torch.full((2, 2, 2), 6.0, dtype=DTYPE),
    )
    assert zero[0, 0, 0] == 0

    with pytest.raises(ValueError, match="boundary must"):
        interp.laplacian_values(boundary="periodic")


def test_rect_laplacian_batched_channels_known_answer():
    axes = torch.tensor([[0.0, 1.0, 3.0, 6.0], [-2.0, -1.0, 1.0, 5.0]], dtype=DTYPE)
    scalar = torch.stack([_quadratic_grid(row, row, row) for row in axes])
    values = torch.stack([scalar, 2.0 * scalar], dim=-1)
    interp = PreparedInterp3dRect(axes, axes, axes, values, uniform="never")

    valid = interp.laplacian_values(boundary="valid")
    assert valid.shape == (2, 2, 2, 2, 2)
    assert torch.allclose(valid[..., 0], torch.full_like(valid[..., 0], 6.0))
    assert torch.allclose(valid[..., 1], torch.full_like(valid[..., 1], 12.0))


@pytest.mark.parametrize("boundary", ["replicate", "one-sided"])
def test_rect_laplacian_batched_boundary_modes_known_answer(boundary: str):
    axes = torch.tensor([[0.0, 1.0, 3.0, 6.0], [-2.0, -1.0, 1.0, 5.0]], dtype=DTYPE)
    scalar = torch.stack([_quadratic_grid(row, row, row) for row in axes])
    interp = PreparedInterp3dRect(axes, axes, axes, scalar, uniform="never")

    lap = interp.laplacian_values(boundary=boundary)
    assert lap.shape == scalar.shape
    assert torch.allclose(lap, torch.full_like(lap, 6.0), atol=1e-12)


def test_rect_laplacian_batched_zero_boundary_semantics():
    axes = torch.tensor([[0.0, 1.0, 3.0, 6.0], [-2.0, -1.0, 1.0, 5.0]], dtype=DTYPE)
    scalar = torch.stack([_quadratic_grid(row, row, row) for row in axes])
    interp = PreparedInterp3dRect(axes, axes, axes, scalar, uniform="never")

    lap = interp.laplacian_values(boundary="zero")
    assert torch.allclose(
        lap[:, 1:-1, 1:-1, 1:-1],
        torch.full((2, 2, 2, 2), 6.0, dtype=DTYPE),
    )
    assert torch.equal(lap[:, 0, 0, 0], torch.zeros(2, dtype=DTYPE))


def test_rect_laplacian_valid_rejects_short_axes():
    axis = torch.tensor([0.0, 1.0], dtype=DTYPE)
    interp = PreparedInterp3dRect(axis, axis, axis, torch.zeros((2, 2, 2), dtype=DTYPE))
    with pytest.raises(ValueError, match="requires Nx, Ny, Nz >= 3"):
        interp.laplacian_values(boundary="valid")


# ---------------------------------------------------------------------------
# Scattered kNN interpolation
# ---------------------------------------------------------------------------


def test_scattered_nearest_known_answer_chunking_and_out_buffer():
    points = _cube_points()
    values = points[:, 0] + 2 * points[:, 1] + 4 * points[:, 2]
    interp = PreparedInterp3dScattered(
        points, values, method="nearest", knn_backend="torch", chunk_size=1
    )
    query = torch.tensor([[0.05, 0.1, 0.9], [0.9, 0.8, 0.1]], dtype=DTYPE)
    out = torch.empty(2, dtype=DTYPE)

    assert interp(query, out=out) is out
    assert torch.equal(out, torch.tensor([4.0, 3.0], dtype=DTYPE))
    with pytest.raises(ValueError, match="out has shape"):
        interp(query, out=torch.empty((2, 1), dtype=DTYPE))


def test_scattered_idw_exact_hit_symmetry_radius_and_gradient():
    points = _cube_points()
    values = (points[:, 0] + points[:, 1] + points[:, 2]).requires_grad_()
    interp = PreparedInterp3dScattered(
        points,
        values,
        method="idw",
        k=8,
        knn_backend="torch",
        learnable_values=True,
        radius=0.25,
        outside="fill",
        fill_value=-3.0,
    )
    query = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [3.0, 3.0, 3.0]], dtype=DTYPE
    )

    got = interp(query)
    assert torch.allclose(got, torch.tensor([0.0, -3.0, -3.0], dtype=DTYPE))

    inside = PreparedInterp3dScattered(
        points, values.detach(), method="idw", k=8, knn_backend="torch"
    )(query[1:2])
    assert torch.allclose(inside, torch.tensor([1.5], dtype=DTYPE))

    got[0].backward()
    assert interp.values.grad is not None
    assert interp.values.grad[0, 0] == 1


def test_scattered_mls_reproduces_affine_vector_field_and_query_gradient():
    points = _cube_points()
    scalar = 2.0 + 3.0 * points[:, 0] - points[:, 1] + 0.5 * points[:, 2]
    values = torch.stack([scalar, -2.0 * scalar], dim=1)
    interp = PreparedInterp3dScattered(
        points,
        values,
        method="mls",
        k=8,
        mls_reg=0.0,
        knn_backend="torch",
        recompute_d2=True,
        chunk_size=0,
    )
    query = torch.tensor([[0.2, 0.3, 0.4]], dtype=DTYPE, requires_grad=True)

    got = interp(query)
    expected_scalar = torch.tensor(2.5, dtype=DTYPE)
    assert torch.allclose(got, torch.tensor([[expected_scalar, -2 * expected_scalar]]))
    got.sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


@pytest.mark.parametrize("chunk_size", [4096, 0])
def test_scattered_empty_query_returns_empty_result(chunk_size: int):
    points = _cube_points()
    interp = PreparedInterp3dScattered(
        points,
        points[:, 0],
        method="idw",
        k=4,
        knn_backend="torch",
        chunk_size=chunk_size,
    )
    got = interp(torch.empty((0, 3), dtype=DTYPE))
    assert got.shape == (0,)


def test_scattered_backend_selection_and_radius_zero():
    points = _cube_points()
    interp = PreparedInterp3dScattered(
        points,
        points[:, :2],
        method="nearest",
        knn_backend="auto",
        radius=0.1,
        outside="zero",
    )
    interp._has_faiss = lambda: False
    assert interp._select_backend(torch.device("cpu")) == "torch"
    assert torch.equal(
        interp(torch.tensor([[2.0, 2.0, 2.0]], dtype=DTYPE)),
        torch.zeros((1, 2), dtype=DTYPE),
    )


def test_scattered_validation_paths():
    points = _cube_points()
    values = points[:, 0]
    with pytest.raises(ValueError, match="method"):
        PreparedInterp3dScattered(points, values, method="cubic")
    with pytest.raises(ValueError, match="outside"):
        PreparedInterp3dScattered(points, values, outside="clamp")
    with pytest.raises(ValueError, match="knn_backend"):
        PreparedInterp3dScattered(points, values, knn_backend="scipy")
    with pytest.raises(ValueError, match=r"shape \(N,3\)"):
        PreparedInterp3dScattered(points[:, :2], values)
    with pytest.raises(ValueError, match="values has N"):
        PreparedInterp3dScattered(points, values[:-1])
    with pytest.raises(ValueError, match="same dtype"):
        PreparedInterp3dScattered(points, values.float())
    with pytest.raises(ValueError, match="k must be"):
        PreparedInterp3dScattered(points, values, k=9)
    with pytest.raises(ValueError, match="requires k >= 4"):
        PreparedInterp3dScattered(points, values, method="mls", k=3)

    interp = PreparedInterp3dScattered(
        points, values, method="nearest", knn_backend="torch"
    )
    with pytest.raises(ValueError, match="xq must have shape"):
        interp(torch.zeros(3, dtype=DTYPE))
    with pytest.raises(ValueError, match="same dtype"):
        interp(torch.zeros((1, 3), dtype=torch.float32))


# ---------------------------------------------------------------------------
# Mesh-guided FEM interpolation with a deterministic one-tetrahedron mesh
# ---------------------------------------------------------------------------


class _FakeNodes:
    node_coord = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    def find_closest_node(self, points, return_index=False):
        delta = np.asarray(points)[:, None, :] - self.node_coord[None, :, :]
        distances = np.linalg.norm(delta, axis=-1)
        nearest = distances.argmin(axis=1)
        if return_index:
            return distances[np.arange(len(nearest)), nearest], nearest + 1
        return self.node_coord[nearest]


class _FakeMesh:
    def __init__(self):
        self.nodes = _FakeNodes()
        self.elm = SimpleNamespace(
            node_number_list=np.array([[1, 2, 3, 4]], dtype=np.int64),
            tag1=np.array([7], dtype=np.int64),
            tetrahedra=np.array([1], dtype=np.int64),
        )

    def find_tetrahedron_with_points(self, points, compute_baricentric=True):
        points = np.asarray(points)
        bary = np.column_stack(
            [1.0 - points.sum(axis=1), points[:, 0], points[:, 1], points[:, 2]]
        )
        inside = np.all(bary >= -1e-12, axis=1)
        tetrahedra = np.where(inside, 1, -1)
        return tetrahedra, bary

    def find_closest_element(
        self, points, return_index=False, elements_of_interest=None
    ):
        points = np.asarray(points)
        distance = np.linalg.norm(points - 0.25, axis=1)
        indices = np.ones(points.shape[0], dtype=np.int64)
        return (distance, indices) if return_index else indices


def test_fem_node_barycentric_numeric_and_nearest_fill_with_gradient():
    mesh = _FakeMesh()
    values = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=DTYPE)
    interp = PreparedInterp3dFEM(
        mesh, values, kind="node", learnable_values=True, squeeze=False
    )
    query = torch.tensor([[0.1, 0.2, 0.3], [2.0, 0.0, 0.0]], dtype=DTYPE)

    got = interp(query, out_fill=-5.0)
    assert got.shape == (2,)
    assert torch.allclose(got, torch.tensor([1.4, -5.0], dtype=DTYPE))

    nearest = interp(query, out_fill="nearest", squeeze=True)
    assert torch.allclose(nearest, torch.tensor([1.4, 1.0], dtype=DTYPE))
    nearest.sum().backward()
    assert interp.values.grad is not None
    assert torch.all(interp.values.grad >= 0)


def test_fem_element_assign_subset_out_and_empty_query():
    mesh = _FakeMesh()
    interp = PreparedInterp3dFEM(
        mesh,
        torch.tensor([[11.0, 12.0]], dtype=DTYPE),
        kind="element_assign",
        squeeze=False,
    )
    query = torch.tensor([[0.1, 0.2, 0.3], [2.0, 0.0, 0.0]], dtype=DTYPE)
    out = torch.empty((2, 2), dtype=DTYPE)

    assert interp(query, out_fill="nearest", out=out) is out
    assert torch.equal(out, torch.tensor([[11.0, 12.0], [11.0, 12.0]], dtype=DTYPE))

    filtered = interp(query[:1], th_indices=[2], out_fill=-4.0)
    assert torch.equal(filtered, torch.tensor([[-4.0, -4.0]], dtype=DTYPE))
    assert interp(torch.empty((0, 3), dtype=DTYPE).reshape(0, 3)).shape == (0, 2)

    with pytest.raises(ValueError, match="out has shape"):
        interp(query, out=torch.empty(2, dtype=DTYPE))


def test_fem_discontinuous_tag_state_uses_recovered_nodal_values():
    mesh = _FakeMesh()
    interp = PreparedInterp3dFEM(
        mesh,
        np.array([99.0]),
        kind="element_linear_discontinuous",
        tag_states=[
            {
                "tag": 7,
                "orig_elms": np.array([1]),
                "local_nodes": np.array([[0, 1, 2, 3]]),
                "node_values": np.array([0.0, 1.0, 2.0, 3.0]),
            }
        ],
    )
    got = interp(torch.tensor([[0.1, 0.2, 0.3]], dtype=DTYPE))
    assert torch.allclose(got, torch.tensor(1.4, dtype=DTYPE))


def test_fem_factory_and_validation_paths():
    mesh = _FakeMesh()
    node_data = SimpleNamespace(mesh=mesh, value=np.arange(4.0), field_name="v")
    element_data = SimpleNamespace(mesh=mesh, value=np.array([8.0]), field_name="e")

    node_interp = PreparedInterp3dFEM.from_NodeData(node_data, dtype=DTYPE)
    element_interp = PreparedInterp3dFEM.from_ElementData(
        element_data, method="assign", dtype=DTYPE
    )
    query = torch.tensor([[0.25, 0.25, 0.25]], dtype=DTYPE)
    assert torch.allclose(node_interp(query), torch.tensor(1.5, dtype=DTYPE))
    assert torch.allclose(element_interp(query), torch.tensor(8.0, dtype=DTYPE))

    with pytest.raises(ValueError, match="data.mesh is None"):
        PreparedInterp3dFEM.from_NodeData(SimpleNamespace(mesh=None, value=[1.0]))
    with pytest.raises(ValueError, match="method must"):
        PreparedInterp3dFEM.from_ElementData(element_data, method="cubic")
    with pytest.raises(ValueError, match="kind must"):
        PreparedInterp3dFEM(mesh, np.array([1.0]), kind="bad")
    with pytest.raises(ValueError, match="tag_states are required"):
        PreparedInterp3dFEM(mesh, np.array([1.0]), kind="element_linear_discontinuous")
    with pytest.raises(ValueError, match="one-based positive"):
        PreparedInterp3dFEM(mesh, np.arange(4.0), th_indices=[0])
    with pytest.raises(TypeError, match="floating-point"):
        PreparedInterp3dFEM(mesh, np.arange(4))

    interp = PreparedInterp3dFEM(mesh, np.arange(4.0))
    with pytest.raises(ValueError, match="points must have shape"):
        interp(torch.zeros(3, dtype=DTYPE))
    with pytest.raises(TypeError, match="points must be floating-point"):
        interp(torch.zeros((1, 3), dtype=torch.long))
