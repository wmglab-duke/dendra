from __future__ import annotations

import math
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from dendra.models.extcell import ExtCellAxon, ExtCellTree, gather_extcell
from dendra.models.fields.line import arbitrary_line, arc_line, helix_line, line3d
from dendra.models.fields.precomputed import (
    EfieldInterpolate3DMesh,
    EfieldInterpolate3DRect,
    EfieldInterpolate3DScattered,
    PreComputedInterpolate1D,
    PreComputedInterpolate3DMesh,
    PreComputedInterpolate3DRect,
    PreComputedInterpolate3DScattered,
)

# Linux Inductor currently imports torch.utils.mkldnn, whose upstream class
# decorators emit this deprecation once per test under newer PyTorch releases.
# Keep the warning-as-error contract for everything except that exact source.
pytestmark = pytest.mark.filterwarnings(
    "ignore:`torch\\.jit\\.script_method` is deprecated\\. Please switch to "
    "`torch\\.compile` or `torch\\.export`\\.:DeprecationWarning:torch\\.jit\\._script"
)


class DummyModel:
    def __init__(self, x, y=None, z=None, *, graph=None, dtype=torch.float64):
        self.x = torch.as_tensor(x, dtype=dtype)
        self.y = (
            torch.zeros_like(self.x) if y is None else torch.as_tensor(y, dtype=dtype)
        )
        self.z = (
            torch.zeros_like(self.x) if z is None else torch.as_tensor(z, dtype=dtype)
        )
        self.graph = graph

    def device(self):
        return self.x.device

    def dtype(self):
        return self.x.dtype


class ScalarMeshInterpolator(torch.nn.Module):
    """Small prepared-interpolator stand-in with the same public call contract."""

    def forward(self, xyz, *, squeeze=False):
        assert not squeeze
        return xyz.sum(dim=-1, keepdim=True)


class VectorMeshInterpolator(torch.nn.Module):
    def __init__(self, value=(2.0, 0.0, 0.0)):
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value, dtype=torch.float64))

    def forward(self, xyz, *, squeeze=False):
        assert not squeeze
        return self.value.to(xyz).expand(xyz.shape[0], -1)


class WrongWidthMeshInterpolator(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, xyz, *, squeeze=False):
        return xyz.new_zeros((xyz.shape[0], self.width))


def _affine_grid(dtype=torch.float64):
    x = torch.tensor([-1.0, 2.0], dtype=dtype)
    y = torch.tensor([0.0, 4.0], dtype=dtype)
    z = torch.tensor([-2.0, 3.0], dtype=dtype)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    values = 2.0 * xx - 3.0 * yy + 0.5 * zz + 7.0
    return x, y, z, values


def _tree_graph():
    graph = nx.DiGraph()
    for node, name in enumerate(("Cell.soma[0](0.5)", "Cell.axon[0](0.5)")):
        graph.add_node(
            node,
            name=name,
            L=10.0,
            diam=2.0,
            Ra=100.0,
            cm=1.0,
            area=math.pi * 20.0,
            x=100.0 * node,
            y=0.0,
            z=0.0,
        )
    graph.add_edge(0, 1, R_ohm=1e5, L=100.0, diff_geom_um=1.0)
    return graph


def test_line_source_matches_closed_form_and_preserves_batch_shape():
    source = line3d([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], rhoe=400.0)
    x = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    y = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float64)
    out = source.fn(x, y, torch.zeros_like(x))

    integral = 2.0 * torch.asinh(1.0 / y)
    expected = 1e4 * 400.0 / (4.0 * torch.pi * 2.0) * integral
    assert out.shape == x.shape
    torch.testing.assert_close(out, expected)


def test_line_source_is_invariant_to_straight_segment_subdivision():
    query = (torch.tensor([1.25]), torch.tensor([0.75]), torch.tensor([-0.5]))
    coarse = line3d([-2.0, 0.0, 0.0], [3.0, 0.0, 0.0], samples=2)
    fine = line3d([-2.0, 0.0, 0.0], [3.0, 0.0, 0.0], samples=17)
    torch.testing.assert_close(coarse.fn(*query), fine.fn(*query), rtol=2e-6, atol=1e-3)


def test_line_source_medium_regularization_gradient_and_forward_dtype():
    infinite = line3d([0, 0, 0], [10, 0, 0], min_distance=2.0)
    halfspace = line3d([0, 0, 0], [10, 0, 0], min_distance=2.0, medium="semi_infinite")
    x = torch.tensor([5.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    z = torch.zeros(1, dtype=torch.float64)
    regularized = infinite.fn(x, y, z)
    at_cutoff = infinite.fn(x, torch.full_like(y, 2.0), z)
    torch.testing.assert_close(regularized, at_cutoff)
    torch.testing.assert_close(halfspace.fn(x, y, z), 2.0 * regularized)

    outside_cutoff = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    value = infinite.fn(x, outside_cutoff, z)
    (gradient,) = torch.autograd.grad(value, outside_cutoff)
    assert torch.isfinite(gradient).all() and gradient.item() < 0.0

    model = DummyModel([[5.0]], [[3.0]], [[0.0]], dtype=torch.float64)
    forwarded = infinite(model)
    assert forwarded.dtype == torch.float64
    torch.testing.assert_close(forwarded, infinite.fn(model.x, model.y, model.z))


def test_line_geometry_helpers_and_state_dict_round_trip():
    helix = helix_line(0.0, 4.0, radius=2.0, orbits=1.0, phase=0.0, samples=5)
    torch.testing.assert_close(helix.xyz[0], torch.tensor([0.0, 2.0, 0.0]))
    torch.testing.assert_close(
        helix.xyz[-1], torch.tensor([4.0, 2.0, 0.0]), atol=1e-6, rtol=0
    )

    arc = arc_line(3.0, radius=2.0, orbit=0.5, samples=3)
    assert torch.all(arc.xyz[:, 0] == 3.0)
    polyline = arbitrary_line([[0, 0, 0], [1, 0, 0], [1, 1, 0]], rhoe=123.0)
    restored = arbitrary_line([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    restored.load_state_dict(polyline.state_dict())
    query = (torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([1.0]))
    torch.testing.assert_close(restored.fn(*query), polyline.fn(*query))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: arbitrary_line([[0, 0]]), "shape"),
        (lambda: arbitrary_line([[0, 0, 0]]), "at least two"),
        (lambda: arbitrary_line([[0, 0, 0], [0, 0, 0]]), "distinct"),
        (lambda: line3d([0, 0], [1, 0, 0]), "shape"),
        (lambda: line3d([0, 0, 0], [1, 0, 0], samples=1), "samples"),
        (lambda: helix_line(0, 1, 1, 1, samples=1), "samples"),
        (lambda: arc_line(0, 1, 1.1), "0 <= orbit <= 1"),
        (lambda: line3d([0, 0, 0], [1, 0, 0], medium="finite"), "medium"),
    ],
)
def test_line_source_validation(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_precomputed_1d_sorts_rows_selects_and_tiles_indices():
    x = torch.tensor([[2.0, 0.0, 1.0], [12.0, 10.0, 11.0]], dtype=torch.float64)
    values = torch.tensor([[4.0, 0.0, 2.0], [24.0, 20.0, 22.0]], dtype=torch.float64)
    field = PreComputedInterpolate1D(values, x, outside="clamp")
    model = DummyModel(
        [[0.5, 1.5], [10.5, 11.5], [0.25, 1.25], [10.25, 11.25]],
        dtype=torch.float64,
    )
    out = field(model, indices=[0, 1])
    expected = 2.0 * model.x
    torch.testing.assert_close(out, expected)

    with pytest.raises(ValueError, match="does not tile cleanly"):
        field(model, indices=[0, 1, 0])
    with pytest.raises(ValueError, match="sorted ascending"):
        PreComputedInterpolate1D(values, x, sort_xy=False)


def test_precomputed_1d_single_lut_broadcasts_across_model_batches():
    field = PreComputedInterpolate1D(
        data=torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64),
        x=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
    )
    model = DummyModel([[0.25, 1.25], [0.5, 1.5], [0.75, 1.75]], dtype=torch.float64)
    torch.testing.assert_close(field(model), 2.0 * model.x)


def test_precomputed_1d_truncation_modes_and_validation():
    x = torch.stack((torch.linspace(0.0, 10.0, 11), torch.linspace(0.0, 20.0, 11)))
    y = x.square()
    safe = PreComputedInterpolate1D(y, x, truncate=20, truncate_mode="safe")
    best = PreComputedInterpolate1D(y, x, truncate=0.2, truncate_mode="best_fit")
    assert safe._truncate_left == safe._truncate_right == 1
    assert best._truncate_left == best._truncate_right == 1

    with pytest.raises(ValueError, match="truncate_mode"):
        PreComputedInterpolate1D(y, x, truncate=0.2, truncate_mode="unknown")
    with pytest.raises(ValueError, match="N>=3"):
        PreComputedInterpolate1D(y[:, :2], x[:, :2], truncate=0.2)
    with pytest.raises(ValueError, match="strictly increasing"):
        PreComputedInterpolate1D(torch.ones(1, 3), torch.ones(1, 3), truncate=0.2)
    with pytest.raises(ValueError, match="data and x must be provided"):
        PreComputedInterpolate1D(data=None, x=x)
    with pytest.raises(ValueError, match="identical shape"):
        PreComputedInterpolate1D(torch.ones(2, 3), torch.ones(4))
    with pytest.raises(ValueError, match="2D tensors"):
        PreComputedInterpolate1D(torch.ones(1, 2, 3), torch.ones(1, 2, 3))
    assert PreComputedInterpolate1D._normalize_truncate(-1.0) == 0.0


def test_precomputed_1d_point_source_extrapolation_and_fallback():
    edge = 1.0 / math.sqrt(5.0)
    field = PreComputedInterpolate1D(
        data=torch.tensor([edge, 1.0, edge], dtype=torch.float64),
        x=torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64),
        outside="point_source",
        uniform="never",
    )
    model = DummyModel([[-4.0, -2.0, 0.0, 2.0, 4.0]], dtype=torch.float64)
    expected = torch.tensor(
        [[1.0 / math.sqrt(17.0), edge, 1.0, edge, 1.0 / math.sqrt(17.0)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(field(model), expected, rtol=1e-6, atol=1e-8)

    no_fit = PreComputedInterpolate1D(
        data=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        x=torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        outside="point_source",
    )
    torch.testing.assert_close(
        no_fit(DummyModel([[-2.0, 2.0]], dtype=torch.float64)),
        torch.zeros((1, 2), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="2D tensors"):
        field._point_source_fill(torch.ones(2), torch.ones(2))
    with pytest.raises(ValueError, match="must match"):
        field._point_source_fill(torch.ones(1, 2), torch.ones(1, 3))


def test_precomputed_1d_gradients_and_state_dict_round_trip():
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    values = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    field = PreComputedInterpolate1D(values, x, learnable_y=True)
    query = torch.tensor([[0.25, 1.5]], dtype=torch.float64, requires_grad=True)
    out = field(DummyModel(query, dtype=torch.float64))
    out.sum().backward()
    torch.testing.assert_close(
        query.grad, torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    )
    assert field.interp.y.grad is not None
    torch.testing.assert_close(
        field.interp.y.grad, torch.tensor([[0.75, 0.75, 0.5]], dtype=torch.float64)
    )

    restored = PreComputedInterpolate1D(torch.zeros_like(values), x, learnable_y=True)
    restored.load_state_dict(field.state_dict())
    torch.testing.assert_close(
        restored(DummyModel([[0.25, 1.5]], dtype=torch.float64)), out
    )

    torch.testing.assert_close(
        field._normalize_indices([0], Q=3, device=torch.device("cpu")),
        torch.zeros(3, dtype=torch.long),
    )
    torch.testing.assert_close(
        field._normalize_indices([0, 0, 0], Q=3, device=torch.device("cpu")),
        torch.zeros(3, dtype=torch.long),
    )


def _write_ascent_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows), header="header", comments="")


def test_precomputed_1d_from_ascent_resamples_variable_length_files(tmp_path):
    base = tmp_path / "samples/0/models/0/sims/0"
    _write_ascent_file(base / "fibersets_bases/0/0/0.dat", [[0.0], [0.001], [0.002]])
    _write_ascent_file(
        base / "fibersets_bases/0/0/1.dat", [[0.0], [0.001], [0.002], [0.003], [0.004]]
    )
    _write_ascent_file(base / "fibersets/0/0.dat", [[0, -1.0], [0, 0.0], [0, 1.0]])
    _write_ascent_file(
        base / "fibersets/0/1.dat",
        [[0, -2.0], [0, -1.0], [0, 0.0], [0, 1.0], [0, 2.0]],
    )

    field = PreComputedInterpolate1D.from_ascent(tmp_path, 0, 0, 0, 0)
    assert field.interp.D == 2 and field.interp.N == 4
    model = DummyModel([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]], dtype=torch.float64)
    out = field(model)
    torch.testing.assert_close(
        out[0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        out[1], torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    )


def test_precomputed_1d_from_ascent_validates_file_sets(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="must both be present"):
        PreComputedInterpolate1D.from_ascent(tmp_path, 0, 0, 0, 0)

    def fake_glob(pattern):
        if "fibersets_bases" in pattern:
            return ["field-0", "field-1"]
        return ["coords-0"]

    monkeypatch.setattr("dendra.models.fields.precomputed.glob.glob", fake_glob)
    monkeypatch.setattr(
        "dendra.models.fields.precomputed.np.loadtxt",
        lambda path, **kwargs: (
            np.array([0.0, 1.0, 2.0])
            if path.startswith("field")
            else np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        ),
    )
    with pytest.raises(ValueError, match="Number of data files"):
        PreComputedInterpolate1D.from_ascent(tmp_path, 0, 0, 0, 0)

    monkeypatch.setattr(
        "dendra.models.fields.precomputed.glob.glob",
        lambda pattern: ["field-0" if "fibersets_bases" in pattern else "coords-short"],
    )
    monkeypatch.setattr(
        "dendra.models.fields.precomputed.np.loadtxt",
        lambda path, **kwargs: (
            np.array([0.0, 1.0, 2.0])
            if path.startswith("field")
            else np.array([[0.0, 0.0], [0.0, 1.0]])
        ),
    )
    with pytest.raises(ValueError, match="same number of rows"):
        PreComputedInterpolate1D.from_ascent(tmp_path, 0, 0, 0, 0)


def test_rectilinear_scalar_wrapper_recovers_affine_field_and_gradients():
    x, y, z, values = _affine_grid()
    field = PreComputedInterpolate3DRect(x, y, z, values, uniform="never")
    xq = torch.tensor([[0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    yq = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    zq = torch.tensor([[0.5, -1.0]], dtype=torch.float64, requires_grad=True)
    model = DummyModel(xq, yq, zq, dtype=torch.float64)
    out = field(model)
    expected = 2.0 * model.x - 3.0 * model.y + 0.5 * model.z + 7.0
    torch.testing.assert_close(out, expected)
    gradients = torch.autograd.grad(out.sum(), (model.x, model.y, model.z))
    torch.testing.assert_close(gradients[0], torch.full_like(model.x, 2.0))
    torch.testing.assert_close(gradients[1], torch.full_like(model.y, -3.0))
    torch.testing.assert_close(gradients[2], torch.full_like(model.z, 0.5))


def test_rectilinear_learnable_values_and_state_dict():
    x, y, z, values = _affine_grid()
    field = PreComputedInterpolate3DRect(x, y, z, values, learnable_values=True)
    model = DummyModel([[0.5]], [[1.0]], [[0.0]], dtype=torch.float64)
    field(model).sum().backward()
    assert field.interpolator.values.grad is not None
    torch.testing.assert_close(
        field.interpolator.values.grad.sum(), torch.tensor(1.0, dtype=torch.float64)
    )

    restored = PreComputedInterpolate3DRect(
        x, y, z, torch.zeros_like(values), learnable_values=True
    )
    restored.load_state_dict(field.state_dict())
    torch.testing.assert_close(restored(model), field(model))


def test_rectilinear_efield_wrapper_matches_constant_field_quasipotential():
    graph = _tree_graph()
    axis = torch.tensor([0.0, 100.0], dtype=torch.float64)
    efield = torch.zeros(2, 2, 2, 3, dtype=torch.float64)
    efield[..., 0] = 2.0
    wrapper = EfieldInterpolate3DRect(axis, axis, axis, efield)
    model = DummyModel([[0.0, 100.0]], graph=graph, dtype=torch.float64)
    out = wrapper(model)
    torch.testing.assert_close(out, torch.tensor([[0.0, -0.2]], dtype=torch.float64))


def test_scattered_scalar_and_vector_wrappers_forward_and_validate():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]],
        dtype=torch.float64,
    )
    scalar = PreComputedInterpolate3DScattered(
        xyz,
        torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
        method="nearest",
        k=1,
        knn_backend="torch",
    )
    model = DummyModel([[0.0, 100.0]], graph=_tree_graph(), dtype=torch.float64)
    torch.testing.assert_close(
        scalar(model), torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    )

    vectors = torch.tensor([[2.0, 0.0, 0.0]] * 3, dtype=torch.float64)
    vector = EfieldInterpolate3DScattered(
        xyz, vectors, method="nearest", k=1, knn_backend="torch"
    )
    torch.testing.assert_close(
        vector(model), torch.tensor([[0.0, -0.2]], dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="shape"):
        PreComputedInterpolate3DScattered([[0.0, 0.0, 0.0]], [[1.0, 2.0]])
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        PreComputedInterpolate3DScattered([[0.0, 0.0]], [[1.0]])
    with pytest.raises(ValueError, match="same number of samples"):
        PreComputedInterpolate3DScattered(xyz, torch.ones(2, 1, dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        EfieldInterpolate3DScattered([[0.0, 0.0, 0.0]], [[1.0, 2.0]])
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        EfieldInterpolate3DScattered([[0.0, 0.0]], [[1.0, 2.0]])


def test_mesh_scalar_wrapper_coordinate_transform_forward_and_state():
    wrapper = PreComputedInterpolate3DMesh(
        ScalarMeshInterpolator(),
        coordinate_scale=2.0,
        coordinate_offset=[1.0, -1.0, 0.5],
    )
    model = DummyModel([[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], dtype=torch.float64)
    expected = 2.0 * (model.x + model.y + model.z) + 0.5
    torch.testing.assert_close(wrapper(model), expected)
    state = wrapper.state_dict()
    torch.testing.assert_close(
        state["_coordinate_scale"], torch.tensor(2.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        state["_coordinate_offset"],
        torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="length-3"):
        PreComputedInterpolate3DMesh(
            ScalarMeshInterpolator(), coordinate_offset=[1.0, 2.0]
        )
    bad = PreComputedInterpolate3DMesh(WrongWidthMeshInterpolator(2))
    with pytest.raises(RuntimeError, match="one component"):
        bad(model)


def test_mesh_efield_interpolation_and_forward_quasipotentials():
    wrapper = EfieldInterpolate3DMesh(VectorMeshInterpolator(), coordinate_scale=0.5)
    model = DummyModel([[0.0, 100.0]], graph=_tree_graph(), dtype=torch.float64)
    vectors = wrapper.interpolate_efield(model)
    torch.testing.assert_close(
        vectors, torch.tensor([[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        wrapper(model), torch.tensor([[0.0, -0.2]], dtype=torch.float64)
    )

    bad = EfieldInterpolate3DMesh(WrongWidthMeshInterpolator(2))
    with pytest.raises(RuntimeError, match=r"\(Q, 3\)"):
        bad.interpolate_efield(model)


def test_mesh_factories_validate_components_and_delegate(monkeypatch):
    scalar_prepared = ScalarMeshInterpolator()
    vector_prepared = VectorMeshInterpolator()
    scalar_calls = {}
    vector_calls = {}

    def make_scalar(data, **kwargs):
        scalar_calls.update(data=data, **kwargs)
        return scalar_prepared

    def make_vector(data, **kwargs):
        vector_calls.update(data=data, **kwargs)
        return vector_prepared

    monkeypatch.setattr(
        "dendra.models.fields.precomputed.PreparedInterp3dFEM.from_NodeData",
        make_scalar,
    )
    node_data = SimpleNamespace(nr_comp=1)
    scalar = PreComputedInterpolate3DMesh.from_node_data(
        node_data,
        coordinate_scale=1e-3,
        coordinate_offset=[1, 2, 3],
        dtype=torch.float64,
    )
    assert scalar.interpolator is scalar_prepared
    assert scalar_calls["data"] is node_data and scalar_calls["squeeze"] is False

    monkeypatch.setattr(
        "dendra.models.fields.precomputed.PreparedInterp3dFEM.from_ElementData",
        make_vector,
    )
    element_data = SimpleNamespace(nr_comp=3)
    vector = EfieldInterpolate3DMesh.from_element_data(element_data, method="assign")
    assert vector.interpolator is vector_prepared
    assert vector_calls["data"] is element_data and vector_calls["squeeze"] is False

    with pytest.raises(ValueError, match="scalar"):
        PreComputedInterpolate3DMesh.from_node_data(SimpleNamespace(nr_comp=3))
    with pytest.raises(ValueError, match="3 components"):
        EfieldInterpolate3DMesh.from_element_data(SimpleNamespace(nr_comp=1))


def test_gather_extcell_custom_values_and_defaults():
    graph = nx.DiGraph()
    graph.add_node(0, xraxial=[1.0, 2.0], xc=[3.0, 4.0], xg=[5.0, 6.0])
    graph.add_node(1)
    gathered = gather_extcell(graph)
    assert set(gathered) == {"xraxial", "xc", "xg"}
    assert gathered["xraxial"].shape == (1, 2, 2)
    torch.testing.assert_close(gathered["xraxial"][0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(gathered["xraxial"][0, 1], torch.full((2,), 1e9))
    torch.testing.assert_close(gathered["xc"][0, 1], torch.zeros(2))


def test_extcell_axon_geometry_buffers_and_graph_export():
    axon = ExtCellAxon(diameters=[5.0, 10.0], n_comp=3, dtype=torch.float64)
    assert axon.xraxial.shape == (2, 3, 2)
    for name in ("xraxial", "xc", "xg"):
        assert getattr(axon, name).dtype == axon.dtype() == torch.float64
        assert getattr(axon, name).device == axon.device()
    torch.testing.assert_close(
        axon.x[0], torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float64)
    )
    axon.xc[1, 2] = torch.tensor([7.0, 8.0], dtype=torch.float64)
    graphs = axon.assemble_graphs()
    assert graphs[1].nodes[2]["xc"] == [7.0, 8.0]
    with pytest.raises(ValueError, match="Only 2 layers"):
        ExtCellAxon(n_layers=1)


def test_extcell_tree_from_graph_broadcasts_extracellular_parameters():
    graph = _tree_graph()
    graph.nodes[0].update(xraxial=[1.0, 2.0], xc=[3.0, 4.0], xg=[5.0, 6.0])
    tree = ExtCellTree.from_graph(graph, N=2, dtype=torch.float64)
    assert tree.xraxial.shape == (2, 2, 2)
    for name in ("xraxial", "xc", "xg"):
        assert getattr(tree, name).dtype == tree.dtype() == torch.float64
        assert getattr(tree, name).device == tree.device()
    torch.testing.assert_close(
        tree.xraxial[:, 0],
        torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(tree.diff_parent_index, torch.tensor([-1, 0]))
    torch.testing.assert_close(
        tree.diff_geom_um,
        torch.tensor([[0.0, 1.0]], dtype=torch.float64).expand(2, -1),
    )

    original_second_cell = tree.x[1, 0].clone()
    tree.x[0, 0] = 99.0
    torch.testing.assert_close(tree.x[1, 0], original_second_cell)
    assert tree.assemble_graphs() == [graph]
    with pytest.raises(ValueError, match="Only 2 layers"):
        ExtCellTree(1, 2, graph, n_layers=1)
