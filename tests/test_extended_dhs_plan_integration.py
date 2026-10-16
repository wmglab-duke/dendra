"""Block/heterogeneous CPU plans are internal, fresh, and trace/AD preserving."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
import dendra.models.integrators.tree as tree_module
import dendra.models.integrators.tree_bt as block_module
from dendra.models.integrators import dhs_bt
from dendra.models.mod import hh

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not callable(getattr(tree_module.dendra_solvers, "DHSMultiPlan", None))
        or not callable(getattr(tree_module.dendra_solvers, "dhs_bt_solve_plan", None)),
        reason="requires extended owning-plan native solvers",
    ),
]
DT = 0.01
STEPS = 5


def _graph():
    graph = nx.DiGraph()
    for node in range(4):
        graph.add_node(
            node,
            L=8.0 + node,
            diam=1.5 + 0.2 * node,
            Ra=100.0,
            cm=1.0,
            area=120.0 + 15.0 * node,
            xraxial=[1.5, 2.0],
            xc=[0.08, 0.12],
            xg=[1.0e-4, 2.0e-4],
        )
    for parent, child in ((3, 0), (3, 2), (2, 1)):
        graph.add_edge(parent, child, R_ohm=1.5e8 + 1.0e7 * child)
    return graph


def _model(kind, *, batch=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        if kind == "block":
            model = dn.ExtCellTree.from_graph(
                _graph(),
                N=2,
                dtype=torch.float64,
                v_init=-64.0,
                integrator=dhs_bt(threads=2, imem=False),
            )
            model.insert(hh)
        else:
            tree = dn.Tree.from_graph(_graph(), N=2, dtype=torch.float64, v_init=-64.0)
            point = dn.SingleCompartment(N=1, C=1, dtype=torch.float64, v_init=-61.0)
            tree.insert(hh)
            point.insert(hh)
            model = dn.concat_models({"tree": tree, "point": point}, threads=2)
        if batch:
            model.batch(2)
        model.initialize()
        model.train()
        model.integrator._initialize(model, DT, force=True)
    assert model.integrator._cpu_topology_plan is not None
    return model


def _raw_only(model):
    model.integrator._cpu_plan_available = False
    model.integrator._cpu_topology_plan = None
    return model


@torch.no_grad()
def _trace(model, *, steps=STEPS):
    records = []
    for index in range(steps):
        stimulus = torch.full_like(model.v, (index + 1) * 1.0e-8)
        model.integrator.step(model, DT, intra=stimulus)
        channel = model.integrator.mech.mechanisms.hh
        state = (model.v, channel.m, channel.h, channel.n)
        if hasattr(model, "vc"):
            state += (model.vc,)
        records.append(tuple(value.detach().clone() for value in state))
    return tuple(torch.stack(values) for values in zip(*records, strict=True))


@pytest.mark.parametrize("kind", ("block", "multi"))
@pytest.mark.parametrize("batch", (False, True))
def test_imperative_full_state_traces_and_plan_reuse(kind, batch):
    model = _model(kind, batch=batch)
    reference = _raw_only(_model(kind, batch=batch))
    owner = model.integrator._cpu_topology_plan
    for actual, expected in zip(_trace(model), _trace(reference), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.integrator._cpu_topology_plan is owner


@pytest.mark.parametrize("kind", ("block", "multi"))
@pytest.mark.parametrize("rebind", (False, True))
def test_topology_mutation_and_rebinding_revalidate(kind, rebind):
    model = _model(kind)
    reference = _model(kind)
    integrator = model.integrator
    owner = integrator._cpu_topology_plan
    name = "layer_ptr" if kind == "block" else "_TILED_WARP_ROW_BASE"
    value = getattr(integrator, name)
    if rebind:
        setattr(integrator, name, value.clone())
    else:
        value.copy_(value.clone())
    for actual, expected in zip(_trace(model), _trace(reference), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert integrator._cpu_topology_plan is not owner


@pytest.mark.parametrize("kind", ("block", "multi"))
def test_invalid_topology_clears_owner_before_transition(kind):
    model = _model(kind)
    integrator = model.integrator
    name = "parent_idx" if kind == "block" else "P_cat"
    getattr(integrator, name).fill_(0)
    with pytest.raises(RuntimeError):
        integrator.step(model, DT)
    assert integrator._cpu_topology_plan is None
    assert not integrator._compiled_kernels


@pytest.mark.parametrize("kind", ("block", "multi"))
def test_versionless_metadata_and_old_package_capability_fall_back(kind, monkeypatch):
    model = _model(kind)
    reference = _raw_only(_model(kind))
    name = "layer_ptr" if kind == "block" else "_TILED_WARP_ROW_COUNT"
    with torch.inference_mode():
        setattr(model.integrator, name, getattr(model.integrator, name).clone())
    for actual, expected in zip(_trace(model), _trace(reference), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.integrator._cpu_topology_plan is None
    fresh = _model(kind)
    reference = _raw_only(_model(kind))
    module = block_module if kind == "block" else tree_module
    capability = "dhs_bt_solve_plan" if kind == "block" else "DHSMultiPlan"
    monkeypatch.setattr(module.dendra_solvers, capability, None)
    for actual, expected in zip(_trace(fresh), _trace(reference), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fresh.integrator._cpu_topology_plan is None


@pytest.mark.parametrize("kind", ("block", "multi"))
def test_custom_solver_not_bypassed(kind):
    model = _model(kind)
    reference = _model(kind)
    integrator = model.integrator
    original = integrator._solve
    calls = []

    def custom(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    integrator._solve = custom
    for actual, expected in zip(_trace(model), _trace(reference), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == STEPS
    assert integrator._cpu_topology_plan is None


@pytest.mark.parametrize("kind", ("block", "multi"))
def test_stimulus_initial_voltage_and_channel_gradients(kind):
    candidate = _model(kind, batch=True)
    baseline = _model(kind, batch=True)

    def evaluate(model, *, planned):
        functional, tensors = dn.func.make_functional(model, dt=DT)
        if not planned:
            functional._transition.solver = (
                tree_module.dendra_solvers.dhs_bt_solve
                if kind == "block"
                else tree_module.dendra_solvers.dhs_multi_solve
            )
        channel_name = tensors.parameter_name(
            "integrator.mech.mechanisms.hh.gnabar_param"
        )
        parameters = tensors.independent_parameters(trainable=(channel_name,))
        channel = parameters[channel_name]
        initial = tensors.state["integrator"]["v"].detach().clone().requires_grad_()
        stimulus = torch.linspace(
            -0.8e-8, 1.1e-8, STEPS * initial.numel(), dtype=initial.dtype
        ).reshape(STEPS, *initial.shape)
        stimulus.requires_grad_()
        state = torch.utils._pytree.tree_map(
            lambda x: x.detach().clone(), tensors.state
        )
        state["integrator"]["v"] = initial
        if "vc" in state["integrator"]:
            old = state["integrator"]["vc"]
            state["integrator"]["vc"] = torch.cat(
                ((initial + old[..., 1]).unsqueeze(-1), old[..., 1:]),
                dim=-1,
            )
        prepared = functional.prepare(parameters, tensors.constants)
        records = []
        for index in range(STEPS):
            state, _ = functional.step(
                parameters,
                prepared,
                state,
                dn.func.StepInput(intra=stimulus[index]),
            )
            records.append(state["integrator"]["v"])
        trace = torch.stack(records)
        gradients = torch.autograd.grad(
            trace.square().mean(), (stimulus, initial, channel)
        )
        for value in gradients:
            assert torch.isfinite(value).all()
            assert torch.count_nonzero(value), "gradient target was not exercised"
        return trace, gradients

    actual, grads = evaluate(candidate, planned=True)
    expected, expected_grads = evaluate(baseline, planned=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ("block", "multi"))
def test_functional_solver_has_independent_snapshot(kind):
    model = _model(kind)
    integrator = model.integrator
    solve = integrator._functional_solver()
    native = tree_module.dendra_solvers
    if kind == "block":
        metadata = (integrator.parent_idx, integrator.order, integrator.layer_ptr)
        shape = (integrator.B, integrator.K)
        dynamic = (
            torch.eye(3, dtype=torch.float64).expand(*shape, 3, 3).clone() * 3,
            torch.full((*shape, 3), 0.1, dtype=torch.float64),
            torch.randn(*shape, 3, dtype=torch.float64),
        )
        expected = native.dhs_bt_solve(*dynamic, *metadata)
    else:
        metadata = integrator._cpu_topology_sources()
        shape = (integrator._tiled_rows_total, integrator.K_stride)
        dynamic = (
            torch.full(shape, 3.0, dtype=torch.float64),
            torch.full(shape, 0.1, dtype=torch.float64),
            torch.randn(shape, dtype=torch.float64),
        )
        expected = native.dhs_multi_solve(*dynamic, *metadata)
    metadata[0].fill_(0)
    torch.testing.assert_close(solve(*dynamic, *metadata), expected, rtol=0, atol=0)
