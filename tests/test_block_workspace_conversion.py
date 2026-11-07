"""Block workspaces follow explicit Population dtype conversions."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas

DT = 0.01
pytestmark = pytest.mark.cpu


def _tree_graph():
    graph = nx.DiGraph()
    for node in range(3):
        graph.add_node(
            node,
            name=f"compartment.{node}",
            kind="compartment",
            L=10.0,
            diam=2.0,
            Ra=100.0,
            cm=1.0,
            area=100.0,
            xraxial=[1.5, 2.5],
            xc=[0.1, 0.2],
            xg=[1.0e-4, 2.0e-4],
        )
    graph.add_edge(0, 1, R_ohm=1.7e8, L=1.0)
    graph.add_edge(0, 2, R_ohm=2.3e8, L=1.0)
    return graph


def _model(kind, *, dtype=torch.float32):
    if kind in {"block_path", "block_tree"} and not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("CPU block solves require dendra-solvers")
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=dtype, IMEM=False):
        if kind == "block_path":
            model = dn.ExtCellAxon(
                [5.0],
                n_comp=3,
                v_init=-65.0,
                integrator=dn.bwd_euler_bt(method="thomas", imem=False),
            )
            model.xraxial.fill_(2.0)
            model.xc.fill_(0.1)
            model.xg.fill_(2.0e-4)
        elif kind == "block_tree":
            model = dn.ExtCellTree.from_graph(
                _tree_graph(),
                N=1,
                v_init=-65.0,
                integrator=dhs_bt(threads=2, imem=False),
            )
        elif kind == "scalar":
            model = dn.SingleCompartment(N=1, C=3, v_init=-65.0)
        else:
            assert kind == "cable"
            model = dn.Unmyelinated(
                [2.0],
                L=40.0,
                dx=10.0,
                v_init=-65.0,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
        model.insert(pas, g=1.0e-3, e=-70.0)
        model.initialize()
        model.eval()
        if hasattr(model, "vc"):
            shell = torch.linspace(-1.0, 0.5, model.v.numel(), dtype=dtype).reshape(
                model.shape
            )
            model.vc = torch.stack((model.v + shell, shell, -0.5 * shell), dim=-1)
        model.integrator._initialize(model, torch.tensor(DT, dtype=dtype), force=True)
    return model


def _workspace(model):
    return tuple(
        getattr(model.integrator, name)
        for name, _role in model.integrator._prepared_workspace_schema()
    )


def _carry(model):
    return tuple(getattr(model, name) for name in model.integrator.v_vars)


def _convert_dtype(model, dtype, conversion):
    if conversion == "to":
        return model.to(dtype=dtype)
    return model.float() if dtype == torch.float32 else model.double()


@pytest.mark.parametrize("kind", ["block_path", "block_tree"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("conversion", ["cast", "to"])
def test_same_dtype_conversion_preserves_block_workspace(kind, dtype, conversion):
    model, reference = _model(kind, dtype=dtype), _model(kind, dtype=dtype)
    before = tuple(value.clone() for value in _carry(model))
    before_workspace = _workspace(model)
    assert model.integrator.initialized
    assert all(value.dtype == dtype for value in before_workspace)

    assert _convert_dtype(model, dtype, conversion) is model

    assert model.integrator.initialized
    assert all(
        actual is expected
        for actual, expected in zip(_workspace(model), before_workspace, strict=True)
    )
    for actual, expected in zip(_carry(model), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE=dtype):
        model.step(dt=DT)
        reference.step(dt=DT)
    assert model.integrator.initialized
    assert all(value.dtype == dtype for value in _workspace(model))
    for actual, expected in zip(_carry(model), _carry(reference), strict=True):
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["block_path", "block_tree"])
@pytest.mark.parametrize("conversion", ["cast", "to"])
def test_block_workspace_precision_survives_public_dtype_roundtrip(kind, conversion):
    model = _model(kind)
    for dtype in (torch.float64, torch.float32):
        before = tuple(value.to(dtype=dtype) for value in _carry(model))
        _convert_dtype(model, dtype, conversion)
        assert not model.integrator.initialized
        for actual, expected in zip(_carry(model), before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE=dtype):
            model.step(dt=DT)
        assert model.integrator.initialized
        assert all(value.dtype == dtype for value in _workspace(model))
        assert all(value.dtype == dtype for value in _carry(model))
        assert all(torch.isfinite(value).all() for value in _carry(model))


@pytest.mark.parametrize("kind", ["scalar", "cable"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("conversion", ["cast", "to"])
def test_same_dtype_conversion_preserves_scalar_solver_workspace(
    kind, dtype, conversion
):
    model = _model(kind, dtype=dtype)
    before = _workspace(model)

    _convert_dtype(model, dtype, conversion)

    assert model.integrator.initialized
    assert all(
        actual is expected
        for actual, expected in zip(_workspace(model), before, strict=True)
    )


def test_conversion_accepts_legacy_integrator_without_workspace_schema():
    model = _model("scalar")
    legacy = torch.nn.Module()
    legacy.initialized = True
    legacy.register_buffer("coefficient", torch.ones(1, dtype=torch.float32))
    model.integrator = legacy

    model.float()
    assert legacy.initialized

    model.to(dtype=torch.float64)
    assert not legacy.initialized
    assert legacy.coefficient.dtype == model.v.dtype == torch.float64
