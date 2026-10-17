"""Tree workspace refreshes preserve unchanged native topology and code."""

from functools import partial

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.integrators import dhs_bt
from dendra.models.integrators import tree as tree_module
from dendra.models.integrators import tree_bt as block_module
from dendra.models.mod import pas
from tests.test_integrator_cache_lifecycle import _CountingCompiler

pytestmark = pytest.mark.cpu
DT = 0.025


@pytest.fixture
def compiler(monkeypatch):
    torch._dynamo.reset()
    value = _CountingCompiler(monkeypatch)
    yield value
    torch._dynamo.reset()


def _graph():
    graph = nx.DiGraph()
    for node in range(4):
        graph.add_node(
            node,
            L=10.0,
            diam=2.0,
            Ra=100.0,
            cm=1.0,
            area=100.0,
            xraxial=[1.5, 2.0],
            xc=[0.08, 0.12],
            xg=[1.0e-4, 2.0e-4],
        )
    for parent, child in ((0, 1), (0, 2), (2, 3)):
        graph.add_edge(parent, child, R_ohm=1.5e8, L=1.0)
    return graph


def _model(kind):
    solvers = pytest.importorskip("dendra_solvers")
    required = "DHSMultiPlan" if kind == "multi" else "DHSPlan"
    if not callable(getattr(solvers, required, None)):
        pytest.skip("requires owning native topology plans")
    if kind == "block" and not callable(getattr(solvers, "dhs_bt_solve_plan", None)):
        pytest.skip("requires block owning native topology plans")
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float64):
        if kind == "block":
            model = dn.ExtCellTree.from_graph(
                _graph(), N=1, dtype=torch.float64, integrator=dhs_bt(threads=2)
            )
            model.insert(pas, g=0.0002, e=-71.0)
        else:
            model = dn.Tree.from_graph(
                _graph(), N=1, dtype=torch.float64, integrator=dn.dhs(threads=2)
            )
            model.insert(pas, g=0.0002, e=-71.0)
            if kind == "multi":
                point = dn.SingleCompartment(N=1, C=1, dtype=torch.float64)
                point.insert(pas, g=0.0002, e=-71.0)
                model = dn.concat_models({"tree": model, "point": point}, threads=2)
        model.requires_grad_(False)
        model.eval()
        model.initialize()
    return model


@pytest.mark.parametrize("kind", ["scalar", "block", "multi"])
def test_tree_reinitialize_reuses_plan_and_compilation(compiler, kind):
    model = _model(kind)
    with dn.ctx(
        JIT=1,
        BACKEND=compiler.backend,
        FULLGRAPH=True,
        REQUIRE_GRAD=0,
        DEVICE="cpu",
        DTYPE=torch.float64,
    ):
        model.run(tstop=2 * DT, dt=DT)
        expected = model.v.detach().clone()
        integrator = model.integrator
        owner = integrator._cpu_topology_plan
        assert owner is not None
        kernels = dict(integrator._compiled_kernels)
        graphs, wrappers = len(compiler.graphs), len(compiler.wrappers)
        assert graphs and kernels

        for _ in range(2):
            model.initialize()
            model.run(tstop=2 * DT, dt=DT)
            torch.testing.assert_close(model.v, expected, rtol=0, atol=0)
            assert integrator._cpu_topology_plan is owner
            assert integrator._compiled_kernels == kernels
            assert len(compiler.graphs) == graphs
            assert len(compiler.wrappers) == wrappers


@pytest.mark.parametrize("kind", ["scalar", "block", "multi"])
def test_tree_preserved_code_reads_fresh_coefficients(compiler, kind):
    model = _model(kind)
    with dn.ctx(
        JIT=1,
        BACKEND=compiler.backend,
        FULLGRAPH=True,
        REQUIRE_GRAD=0,
        DEVICE="cpu",
        DTYPE=torch.float64,
    ):
        model.run(tstop=2 * DT, dt=DT)
        old_voltage = model.v.detach().clone()
        integrator = model.integrator
        owner = integrator._cpu_topology_plan
        old_workspace = tuple(
            getattr(integrator, name)
            for name, _role in integrator._prepared_workspace_schema()
        )
        kernels = dict(integrator._compiled_kernels)
        for population in model if kind == "multi" else (model,):
            population.cm_scale_param.set(2.0)
        model.initialize()
        model.run(tstop=2 * DT, dt=DT)
        actual = model.v.detach().clone()
        assert not torch.equal(actual, old_voltage)
        assert integrator._cpu_topology_plan is owner
        assert integrator._compiled_kernels == kernels
        assert all(
            getattr(integrator, name) is not previous
            for (name, _role), previous in zip(
                integrator._prepared_workspace_schema(), old_workspace, strict=True
            )
        )

    reference = _model(kind)
    for population in reference if kind == "multi" else (reference,):
        population.cm_scale_param.set(2.0)
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float64):
        reference.initialize()
        reference.run(tstop=2 * DT, dt=DT)
    torch.testing.assert_close(actual, reference.v, rtol=0, atol=0)


def test_tree_reinitialization_refreshes_autograd_without_recompilation(compiler):
    model, reference = _model("scalar"), _model("scalar")
    model.train()
    reference.train()
    model.cm_scale_param.rho.requires_grad_(True)
    reference.cm_scale_param.rho.requires_grad_(True)
    kernels = owner = None
    graphs = wrappers = None
    for scale in (1.0, 1.5):
        model.cm_scale_param.set(scale)
        reference.cm_scale_param.set(scale)
        with dn.ctx(
            JIT=1,
            BACKEND=compiler.backend,
            FULLGRAPH=True,
            REQUIRE_GRAD=1,
            DEVICE="cpu",
            DTYPE=torch.float64,
        ):
            model.initialize()
            model.run(tstop=2 * DT, dt=DT)
            gradient = torch.autograd.grad(model.v.sum(), model.cm_scale_param.rho)[0]
            assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
            if kernels is None:
                owner = model.integrator._cpu_topology_plan
                kernels = dict(model.integrator._compiled_kernels)
                graphs, wrappers = len(compiler.graphs), len(compiler.wrappers)
            else:
                assert model.integrator._cpu_topology_plan is owner
                assert model.integrator._compiled_kernels == kernels
                assert len(compiler.graphs) == graphs
                assert len(compiler.wrappers) == wrappers

        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64):
            reference.initialize()
            reference.run(tstop=2 * DT, dt=DT)
            expected_gradient = torch.autograd.grad(
                reference.v.sum(), reference.cm_scale_param.rho
            )[0]
        torch.testing.assert_close(model.v, reference.v, rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    "integrator_type, solver",
    [
        (tree_module._dhs, tree_module.dhs_solve_cuda),
        (tree_module._dhs_multi, tree_module.dhs_solve_multi_cuda),
        (block_module._dhs_bt, block_module.dhs_bt_solve_cuda),
    ],
)
def test_cuda_solver_binding_keeps_identical_selection_without_gpu(
    integrator_type, solver
):
    if not callable(solver):
        pytest.skip("CUDA solver bindings are unavailable")
    integrator = integrator_type.__new__(integrator_type)
    torch.nn.Module.__init__(integrator)
    integrator.threads = 2
    integrator._solve = None
    integrator._select_solver("cuda")
    selected = integrator._solve
    assert isinstance(selected, partial)
    assert selected.func is solver
    integrator._select_solver("cuda")
    assert integrator._solve is selected
    integrator.threads = 4
    integrator._select_solver("cuda")
    assert integrator._solve is not selected
    assert integrator._solve.keywords == {"threads": 4}
