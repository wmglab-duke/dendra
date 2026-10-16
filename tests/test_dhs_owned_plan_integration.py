"""The CPU topology fast path must not retain stale mutable solver indices."""

import copy
import pickle

import pytest
import torch

import dendra as dn
import dendra.models.integrators.tree as tree_module
from dendra.models.mod import pas

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not callable(getattr(tree_module.dendra_solvers, "DHSPlan", None)),
        reason="requires the owning-plan native solver candidate",
    ),
]
DT = 0.025


def _model(*, N=2):
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, IMEM=False):
        morphology = dn.Morphology(rhoa=100.0, cm=1.0)
        soma = morphology.section("soma", L=16.0, diam=14.0, nseg=1)
        branch = morphology.section("branch", L=60.0, diam=2.0, nseg=3)
        branch.connect(soma.at(1.0), child_end=0)
        model = dn.Tree.from_morphology(
            morphology,
            N=N,
            dtype=torch.float64,
            v_init=-64.0,
            integrator=dn.dhs(threads=32, imem=False),
        )
        model.insert(pas, g=4.0e-4, e=-71.0)
        model.requires_grad_(False)
        model.initialize()
        model.integrator._initialize(model, DT, force=True)
    assert model.integrator._cpu_topology_plan is not None
    return model


def _trace(model, steps=4, dt=DT):
    values = [model.v.detach().clone()]
    for _ in range(steps):
        model.integrator.step(model, dt)
        values.append(model.v.detach().clone())
    return torch.stack(values)


def test_unchanged_topology_reuses_one_owner_across_steps():
    model = _model()
    plan = model.integrator._cpu_topology_plan
    _trace(model)
    assert model.integrator._cpu_topology_plan is plan


@pytest.mark.parametrize("name", ["parent_idx", "order", "layer_ptr"])
@pytest.mark.parametrize("replacement", [False, True])
def test_mutation_or_rebinding_revalidates_before_next_step(name, replacement):
    model = _model()
    reference = copy.deepcopy(model)
    integrator = model.integrator
    old_plan = integrator._cpu_topology_plan
    value = getattr(integrator, name)
    if replacement:
        setattr(integrator, name, value.clone())
    else:
        value.copy_(value.clone())
    torch.testing.assert_close(_trace(model), _trace(reference), rtol=0, atol=0)
    assert integrator._cpu_topology_plan is not old_plan


def test_invalid_replacement_cannot_reuse_old_plan():
    model = _model()
    integrator = model.integrator
    integrator.parent_idx[0] = 0
    with pytest.raises(RuntimeError, match="parent|layer"):
        integrator.step(model, DT)
    assert integrator._cpu_topology_plan is None
    assert not integrator._compiled_kernels


def test_versionless_topology_falls_back_to_validated_raw_operator():
    model = _model()
    reference = copy.deepcopy(model)
    integrator = model.integrator
    with torch.inference_mode():
        integrator.layer_ptr = integrator.layer_ptr.clone()
    torch.testing.assert_close(_trace(model), _trace(reference), rtol=0, atol=0)
    assert integrator._cpu_topology_plan is None
    with torch.inference_mode():
        integrator.layer_ptr[-1] = 0
    with pytest.raises(RuntimeError, match="layer_ptr"):
        integrator.step(model, DT)


def test_old_native_package_capability_falls_back(monkeypatch):
    model = _model()
    reference = copy.deepcopy(model)
    monkeypatch.setattr(tree_module.dendra_solvers, "DHSPlan", None)
    torch.testing.assert_close(_trace(model), _trace(reference), rtol=0, atol=0)
    assert model.integrator._cpu_topology_plan is None


@pytest.mark.parametrize("during_initialize", [False, True])
def test_custom_solver_is_not_bypassed(monkeypatch, during_initialize):
    model = _model()
    reference = copy.deepcopy(model)
    integrator = model.integrator
    calls = []

    def custom_solver(*args):
        calls.append(True)
        return torch.ops.dendra_solvers.dhs_solve(*args)

    if during_initialize:

        def select_custom_solver(device):
            integrator._solve = custom_solver

        monkeypatch.setattr(integrator, "_select_solver", select_custom_solver)
        integrator._initialize(model, DT, force=True)
    else:
        integrator._solve = custom_solver
    torch.testing.assert_close(_trace(model), _trace(reference), rtol=0, atol=0)
    assert len(calls) == 4
    assert integrator._cpu_topology_plan is None
    with pytest.raises(RuntimeError, match="not yet transform-compatible"):
        integrator._functional_solver()


@pytest.mark.parametrize("mode", ["deepcopy", "pickle"])
def test_copy_and_pickle_plan_have_valid_owned_lifetimes(mode):
    model = _model()
    expected = copy.deepcopy(model)
    restored = copy.deepcopy(model)
    if mode == "deepcopy":
        assert (
            restored.integrator._cpu_topology_plan
            is model.integrator._cpu_topology_plan
        )
    else:
        # Serialize the new public plan, not the whole Population: configured
        # integrator classes and raw torch.ops bindings have legacy pickle limits.
        restored.integrator._refresh_cpu_topology_plan()
        restored.integrator._cpu_topology_plan = pickle.loads(
            pickle.dumps(model.integrator._cpu_topology_plan)
        )
        assert (
            restored.integrator._cpu_topology_plan
            is not model.integrator._cpu_topology_plan
        )
    owner = restored.integrator._cpu_topology_plan
    del model
    torch.testing.assert_close(_trace(restored), _trace(expected), rtol=0, atol=0)
    if mode == "pickle":
        assert restored.integrator._cpu_topology_plan is owner


@pytest.mark.parametrize("assign", [False, True])
def test_state_dict_load_reconstructs_plan_without_schema_changes(assign):
    # Multi-neuron mech.area is an expanded view; ordinary copy-loading that
    # legacy buffer is unsupported. Retain multi-neuron assign=True coverage.
    model = _model(N=2 if assign else 1)
    original = model.integrator._cpu_topology_plan
    state = copy.deepcopy(model.state_dict())
    assert not any("topology_plan" in name or "_cpu_plan" in name for name in state)
    model.load_state_dict(state, assign=assign)
    assert not model.integrator.initialized
    model.integrator._initialize(model, DT)
    assert model.integrator._cpu_topology_plan is not original
    assert torch.isfinite(_trace(model)).all()


def test_dtype_and_timestep_reinitialize_only_numeric_workspaces_and_plan():
    model = _model()
    old = model.integrator._cpu_topology_plan
    old_capacitance = model.integrator.cm_dt.detach().clone()
    model.to(dtype=torch.float32)
    assert not model.integrator.initialized
    model.integrator._initialize(model, DT * 2)
    assert model.integrator._cpu_topology_plan is not old
    assert model.integrator.cm_dt.dtype == torch.float32
    torch.testing.assert_close(model.integrator.cm_dt, (old_capacitance / 2).float())
    assert torch.isfinite(_trace(model, dt=DT * 2)).all()


def test_functional_solver_captures_independent_immutable_snapshot():
    model = _model()
    integrator = model.integrator
    solve = integrator._functional_solver()
    d = torch.full((2, integrator.K), 2.0, dtype=torch.float64, requires_grad=True)
    a = torch.full_like(d, 0.2, requires_grad=True)
    b = torch.randn_like(d, requires_grad=True)
    topology = (integrator.parent_idx, integrator.order, integrator.layer_ptr)
    reference = tree_module.dendra_solvers.dhs_solve(d, a, b, *topology)
    expected = torch.autograd.grad(
        reference.square().sum(), (d, a, b), retain_graph=True
    )
    integrator.parent_idx.fill_(0)
    actual = solve(d, a, b, *topology)
    observed = torch.autograd.grad(actual.square().sum(), (d, a, b))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for left, right in zip(observed, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
