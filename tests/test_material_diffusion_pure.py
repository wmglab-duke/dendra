"""Direct pure diffusion contracts and parity with cached imperative execution."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import DiffusionProcess, Material
from dendra.models.mechanisms import _spatial as spatial
from dendra.models.mechanisms._spatial import SpatialOperatorTree

pytestmark = pytest.mark.cpu
DTYPE = torch.float64
KEY = ("pool", "c")


@pytest.fixture(params=[1, 4], autouse=True)
def _cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    yield request.param
    torch.set_num_threads(previous)


def _fixture(kind, method, relaxation, *, interface="arithmetic"):
    c = torch.tensor([[1.3, 2.1, 0.4, 3.2]], dtype=DTYPE, requires_grad=True)
    D = torch.tensor([[0.25, 0.4, 0.3, 0.55]], dtype=DTYPE, requires_grad=True)
    dt = c.new_tensor(0.125, requires_grad=True)
    rate = c.new_tensor(0.17, requires_grad=True)
    target = c.new_tensor(0.8, requires_grad=True)
    volume_fraction = c.new_tensor(0.7, requires_grad=True)
    area_fraction = c.new_tensor(0.6, requires_grad=True)
    diam = c.new_tensor([[1.1, 1.7, 1.3, 2.2]], requires_grad=True)
    dx = c.new_tensor([[1.2, 2.3, 1.7, 3.4]], requires_grad=True)
    volume = c.new_tensor(
        [[1.1, 1.7, 0.0 if method == "implicit" else 1.3, 2.2]], requires_grad=True
    )
    native_edge = c.new_tensor([[0.4, 0.3, 0.0, 0.5]], requires_grad=True)

    class Diffuse(DiffusionProcess):
        DiffusionProcess.METHOD(
            method,
            solver="dense",
            interface_scheme=interface,
            volume_fraction=volume_fraction,
            area_fraction=area_fraction,
        )
        DiffusionProcess.DIFFUSE("pool", field="c", D=D)
        if relaxation:
            DiffusionProcess.RELAX("pool", field="c", rate=rate, target=target)
            DiffusionProcess.RELAX(
                "pool",
                field="c",
                rate=0.03,
                target=0.4,
                where=[True, False, True, False],
            )

    if kind == "tree":
        graph = nx.DiGraph()
        graph.add_nodes_from(range(4))
        graph.add_edges_from([(2, 0), (2, 1), (1, 3)])
        for parent, child in graph.edges:
            graph[parent][child]["diff_geom_um"] = float(native_edge[0, child].detach())
        population = SimpleNamespace(
            graph=graph,
            shape=tuple(c.shape),
            volume_i=volume,
            diff_geom_um=native_edge,
            diff_parent_index=torch.tensor([2, 2, -1, 1]),
        )
    else:
        population = SimpleNamespace(graph=None, shape=tuple(c.shape), diam=diam, dx=dx)
    material = Material("pool", tuple(c.shape), fields={"c": c.detach().clone()}).to(
        dtype=DTYPE
    )
    material._buffers["c"] = c
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE):
        process = Diffuse(
            name="diffuse",
            celsius=c.new_tensor(34.0),
            diameters=diam,
            shape=tuple(c.shape),
            shape_f=tuple(c.shape),
        ).to(dtype=DTYPE)
        process._bind_materials({"pool": material}.__getitem__, population=population)
        process.train()
        process.populate()
        process.set_dt(dt)
    values = {
        "c": c,
        "D": D,
        "dt": dt,
        "rate": rate,
        "target": target,
        "diam": diam,
        "dx": dx,
        "volume": volume,
        "native_edge": native_edge,
        "volume_fraction": volume_fraction,
        "area_fraction": area_fraction,
    }
    return process, material, values


def _reference(values, kind, method, relaxation, interface="arithmetic"):
    c, D, dt = (values[name] for name in ("c", "D", "dt"))
    if kind == "tree":
        parents, children = (2, 2, 1), (0, 1, 3)
        volume = values["volume"]
        edge = (
            values["native_edge"][..., list(children)]
            * 0.5
            * (D[..., list(parents)] + D[..., list(children)])
        )
    else:
        parents, children = (0, 1, 2), (1, 2, 3)
        area = torch.pi * (0.5 * values["diam"]).square()
        dx = values["dx"]
        volume = area * dx * values["volume_fraction"]
        if interface == "series":
            edge = values["area_fraction"] / (
                0.5 * dx[..., :-1] / (D[..., :-1] * area[..., :-1])
                + 0.5 * dx[..., 1:] / (D[..., 1:] * area[..., 1:])
            )
        else:
            edge = (
                0.5
                * (D[..., :-1] + D[..., 1:])
                * 0.5
                * (area[..., :-1] + area[..., 1:])
                * values["area_fraction"]
                / (0.5 * (dx[..., :-1] + dx[..., 1:]))
            )
    incidence = c.new_zeros((3, 4))
    for row, (parent, child) in enumerate(zip(parents, children)):
        incidence[row, parent] = 1.0
        incidence[row, child] = -1.0
    laplacian = torch.einsum("ei,be,ej->bij", incidence, edge, incidence)
    if method == "explicit":
        return c - dt * (laplacian @ c.unsqueeze(-1)).squeeze(-1) / volume
    matrix = torch.diag_embed(volume) + dt * laplacian
    rhs = volume * c
    if relaxation:
        mask = c.new_tensor([[True, False, True, False]], dtype=torch.bool)
        extra_rate = torch.where(mask, c.new_tensor(0.03), c.new_tensor(0.0))
        mass_rate = volume * (values["rate"] + extra_rate)
        matrix = matrix + torch.diag_embed(dt * mass_rate)
        rhs = rhs + dt * volume * (values["rate"] * values["target"] + extra_rate * 0.4)
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("kind", ["1d", "tree"])
@pytest.mark.parametrize(
    ("method", "relaxation"),
    [("implicit", False), ("implicit", True), ("explicit", False)],
)
def test_pure_diffusion_values_gradients_and_no_hidden_template_numerics(
    kind, method, relaxation, monkeypatch
):
    process, material, values = _fixture(kind, method, relaxation)
    fields = {KEY: values["c"]}
    dt = values["dt"]
    assert process.material_fields == (KEY,)
    assert process._buffers["_diffusion_literal_0_D"] is values["D"]
    if kind == "1d":
        assert process._buffers["_mp_dx"] is values["dx"]
        assert (
            process._buffers["_diffusion_literal_volume_fraction"]
            is values["volume_fraction"]
        )
        targets = tuple(
            values[name]
            for name in (
                "c",
                "D",
                "dt",
                "diam",
                "dx",
                "volume_fraction",
                "area_fraction",
            )
        )
    else:
        assert process._buffers["_mp_volume_i"] is values["volume"]
        assert process._buffers["_mp_diff_geom_um"] is values["native_edge"]
        targets = tuple(
            values[name] for name in ("c", "D", "dt", "volume", "native_edge")
        )
    if relaxation:
        targets = (*targets, values["rate"], values["target"])
    for operator in process._spatial_operators.values():
        operator._solve = operator._functional_solver()
        keep = (
            SpatialOperatorTree._FUNCTIONAL_TOPOLOGY_BUFFERS if kind == "tree" else ()
        )
        for name in tuple(operator._buffers):
            if name not in keep:
                del operator._buffers[name]

    def forbidden_resolver(*args, **kwargs):
        raise AssertionError("Pure diffusion must not resolve bound Material objects")

    monkeypatch.setattr(process, "_get_material", forbidden_resolver)
    object.__setattr__(process, "_mp_population", object())
    versions = {
        name: (id(value), value._version) for name, value in process.named_buffers()
    }
    prepared = process.prepare_materials(fields, dt)
    result = process.material_updates(fields, prepared)[KEY]
    expected = _reference(values, kind, method, relaxation)
    torch.testing.assert_close(result, expected, rtol=3e-12, atol=3e-13)
    weights = result.new_tensor([[0.3, -0.7, 1.1, 0.5]])
    gradients = torch.autograd.grad((result * weights).sum(), targets)
    references = torch.autograd.grad((expected * weights).sum(), targets)
    for actual, reference in zip(gradients, references):
        torch.testing.assert_close(actual, reference, rtol=3e-10, atol=3e-11)
        assert torch.isfinite(actual).all()
    assert versions == {
        name: (id(value), value._version) for name, value in process.named_buffers()
    }
    assert material._buffers["c"] is values["c"]
    assert all(
        torch.is_tensor(leaf) for leaf in torch.utils._pytree.tree_leaves(prepared)
    )


def test_pure_diffusion_series_geometry_and_live_parameters_match_independent_math():
    process, _material, values = _fixture("1d", "implicit", True, interface="series")
    for scale in (1.0, 1.3):
        process._buffers["diam"] = values["diam"] * scale
        current = dict(values, diam=values["diam"] * scale)
        prepared = process.prepare_materials({KEY: values["c"]}, values["dt"])
        result = process.material_updates({KEY: values["c"]}, prepared)[KEY]
        expected = _reference(current, "1d", "implicit", True, "series")
        torch.testing.assert_close(result, expected, rtol=3e-12, atol=3e-13)
        actual = torch.autograd.grad(
            result.square().sum(), (values["diam"], values["dx"])
        )
        reference = torch.autograd.grad(
            expected.square().sum(), (values["diam"], values["dx"])
        )
        for gradient, target in zip(actual, reference):
            torch.testing.assert_close(gradient, target, rtol=3e-10, atol=3e-11)


@pytest.mark.parametrize("kind", ["1d", "tree"])
def test_cached_imperative_diffusion_uses_shared_updates_without_repreparing(
    kind, monkeypatch
):
    process, material, values = _fixture(kind, "implicit", True)
    operators = tuple(process._spatial_operators.values())
    buffers = tuple(
        tuple(id(value) for value in operator._buffers.values())
        for operator in operators
    )

    def forbidden_preparation(*args, **kwargs):
        raise AssertionError(
            "Cached imperative diffusion must not prepare per timestep"
        )

    monkeypatch.setattr(process, "prepare_materials", forbidden_preparation)
    monkeypatch.setattr(process, "_diffusion_operator_inputs", forbidden_preparation)
    monkeypatch.setattr(process, "_diffusion_operand", forbidden_preparation)
    monkeypatch.setattr(
        process, "_refresh_named_transport_quantities", forbidden_preparation
    )
    expected = values["c"]
    for _step in range(3):
        expected = _reference(dict(values, c=expected), kind, "implicit", True)
        process.advance_materials(values["dt"])
        torch.testing.assert_close(
            material._buffers["c"], expected, rtol=3e-12, atol=3e-13
        )
    assert operators == tuple(process._spatial_operators.values())
    assert buffers == tuple(
        tuple(id(value) for value in operator._buffers.values())
        for operator in operators
    )


@pytest.mark.parametrize(
    ("kernel_name", "facade_name"),
    [("thomas_solve_cuda_t", "thomas_solve"), ("pcr_solve_cuda_t", "pcr_solve")],
)
def test_cuda_tridiagonal_facade_selection_requires_no_gpu_launch(
    kernel_name, facade_name, monkeypatch
):
    from dendra.models.integrators import _cuda_functional

    process, _material, _values = _fixture("1d", "implicit", False)
    operator = next(iter(process._spatial_operators.values()))
    kernel = getattr(spatial, kernel_name)
    if kernel is None:
        # CPU-only installations intentionally do not import Triton kernels.
        # A non-executable stand-in exercises the same identity-based route.
        def kernel(*args, **kwargs):
            raise AssertionError("Facade selection must never launch a CUDA kernel")

        monkeypatch.setattr(spatial, kernel_name, kernel)
    operator._solve = kernel
    buffers = {
        name: (id(value), value._version) for name, value in operator._buffers.items()
    }
    selected = operator._functional_solver()
    assert selected is getattr(_cuda_functional, facade_name)
    assert operator._solve is kernel
    assert buffers == {
        name: (id(value), value._version) for name, value in operator._buffers.items()
    }


def test_cuda_tree_facade_preserves_configured_threads_without_gpu_launch(monkeypatch):
    from dendra.models.integrators import _cuda_functional

    process, _material, _values = _fixture("tree", "implicit", False)
    operator = next(iter(process._spatial_operators.values()))
    kernel = spatial.dhs_solve_cuda
    if kernel is None:

        def kernel(*args, **kwargs):
            raise AssertionError("Facade selection must never launch a CUDA kernel")

        monkeypatch.setattr(spatial, "dhs_solve_cuda", kernel)
    launch = partial(kernel, threads=operator.threads)
    operator._solve = launch
    buffers = {
        name: (id(value), value._version) for name, value in operator._buffers.items()
    }
    selected = operator._functional_solver()
    assert isinstance(selected, partial)
    assert selected.func is _cuda_functional.dhs_solve
    assert selected.args == ()
    assert selected.keywords == {"threads": operator.threads}
    assert operator._solve is launch
    assert buffers == {
        name: (id(value), value._version) for name, value in operator._buffers.items()
    }
