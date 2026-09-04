"""Cross-topology contracts for functional Integrator lowering."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.func._population import _resolve_functional_integrator_spec
from dendra.models.integrators.implicit import _bwd_euler_sc, _bwd_euler_ub
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mod import pas

DT = 0.01

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _single_compartment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(
            N=2,
            C=3,
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        population.insert(pas, g=0.001, e=-70.0)
        population.initialize()
    return population


def _unmyelinated():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(pas, g=0.001, e=-70.0)
        population.initialize()
    return population


@pytest.mark.parametrize(
    ("factory", "operator_kind", "implementation"),
    [
        (_single_compartment, "scalar_point", _bwd_euler_sc),
        (_unmyelinated, "scalar_path", _bwd_euler_ub),
    ],
)
def test_public_integrator_wrappers_resolve_exact_private_contracts(
    factory,
    operator_kind,
    implementation,
):
    integrator = factory().integrator
    spec = _resolve_functional_integrator_spec(integrator, operator_kind)

    assert spec is not None
    assert spec.operator_kind == operator_kind
    assert spec.implementation is implementation
    assert spec.workspace_schema == integrator._prepared_workspace_schema()
    assert _resolve_functional_integrator_spec(integrator, "wrong_operator") is None


@pytest.mark.parametrize(
    ("schema", "error", "match"),
    [
        ((("cmdt", "parameter"), ("cmdt", "geometry")), ValueError, "duplicate"),
        ((("dt", "parameter"),), ValueError, "reserved entry 'dt'"),
        ((("cmdt", ""),), TypeError, "non-empty"),
        ((("cmdt", "parameter", "extra"),), TypeError, "string pairs"),
    ],
)
def test_prepared_workspace_schema_rejects_malformed_contracts(
    monkeypatch,
    schema,
    error,
    match,
):
    integrator = _single_compartment().integrator
    monkeypatch.setattr(integrator, "_PREPARED_WORKSPACE_SCHEMA", schema)

    with pytest.raises(error, match=match):
        integrator._prepared_workspace_schema()


@pytest.mark.parametrize(
    ("mode", "error", "match"),
    [
        ("not_mapping", TypeError, "must return a mapping"),
        ("missing", KeyError, "area"),
        ("extra", KeyError, "unexpected entries"),
        ("non_tensor", TypeError, "non-Tensor entries"),
    ],
)
def test_imperative_workspace_installation_is_transactional_for_bad_builders(
    monkeypatch,
    mode,
    error,
    match,
):
    population = _single_compartment()
    integrator = population.integrator
    original_builder = integrator._prepare_workspace
    snapshot = {
        name: (
            id(getattr(integrator, name)),
            getattr(integrator, name)._version,
            getattr(integrator, name).detach().clone(),
        )
        for name, _role in integrator._prepared_workspace_schema()
    }

    def malformed(dt, **inputs):
        workspace = original_builder(dt, **inputs)
        if mode == "not_mapping":
            return tuple(workspace.values())
        if mode == "missing":
            workspace.pop("area")
        elif mode == "extra":
            workspace["unexpected"] = torch.ones_like(workspace["cmdt"])
        elif mode == "non_tensor":
            workspace["area"] = object()
        return workspace

    monkeypatch.setattr(integrator, "_prepare_workspace", malformed)
    with pytest.raises(error, match=match):
        integrator.initialize(population, 2 * DT)

    for name, (identity, version, expected) in snapshot.items():
        actual = getattr(integrator, name)
        assert id(actual) == identity
        assert actual._version == version
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("operator_kind", ["scalar_point", "scalar_path"])
def test_shared_workspace_derivation_compiles_with_aot_backward(operator_kind):
    if operator_kind == "scalar_point":
        integrator = _single_compartment().integrator
        names = ("cmdt", "area")
        inputs = (
            torch.tensor(0.02, dtype=torch.float64),
            torch.linspace(1.1, 1.9, 6, dtype=torch.float64).reshape(2, 3),
            torch.linspace(0.7, 1.3, 6, dtype=torch.float64).reshape(2, 3),
        )

        def flattened(dt, cm, area):
            workspace = integrator._derive_prepared_workspace(
                dt,
                cm=cm,
                area=area,
            )
            return torch.cat(tuple(workspace[name].reshape(-1) for name in names))

    else:
        integrator = _unmyelinated().integrator
        names = tuple(name for name, _role in integrator._prepared_workspace_schema())
        inputs = (
            torch.tensor(0.02, dtype=torch.float64),
            torch.linspace(1.1, 1.9, 8, dtype=torch.float64).reshape(2, 4),
            torch.linspace(0.7, 1.3, 8, dtype=torch.float64).reshape(2, 4),
            torch.linspace(0.2, 0.6, 6, dtype=torch.float64).reshape(2, 3),
        )

        def flattened(dt, cm, area, edge_conductance):
            workspace = integrator._derive_prepared_workspace(
                dt,
                cm=cm,
                area=area,
                edge_conductance=edge_conductance,
            )
            return torch.cat(tuple(workspace[name].reshape(-1) for name in names))

    eager_inputs = tuple(value.clone().requires_grad_() for value in inputs)
    compiled_inputs = tuple(value.clone().requires_grad_() for value in inputs)
    weights = torch.linspace(
        0.5,
        1.5,
        flattened(*eager_inputs).numel(),
        dtype=torch.float64,
    )

    eager = flattened(*eager_inputs)
    eager_gradients = torch.autograd.grad((eager * weights).sum(), eager_inputs)
    compiled_fn = torch.compile(flattened, backend="aot_eager", fullgraph=True)
    compiled = compiled_fn(*compiled_inputs)
    compiled_gradients = torch.autograd.grad(
        (compiled * weights).sum(),
        compiled_inputs,
    )

    torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
        assert torch.isfinite(actual).all()


@pytest.mark.parametrize("factory", [_single_compartment, _unmyelinated])
@pytest.mark.parametrize(
    "hook",
    [
        "_prepare_workspace",
        "_voltage_update",
        "_step",
        "_PREPARED_WORKSPACE_SCHEMA",
    ],
)
def test_make_functional_rejects_integrator_instance_contract_overrides(
    factory,
    hook,
):
    population = factory()
    replacement = () if hook == "_PREPARED_WORKSPACE_SCHEMA" else lambda *a, **k: None
    setattr(population.integrator, hook, replacement)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="functionalization requires",
    ):
        dn.func.make_functional(population, dt=DT)


def test_make_functional_rejects_solver_selection_override():
    population = _unmyelinated()
    population.integrator._select_solver = lambda *_args, **_kwargs: None

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="functionalization requires",
    ):
        dn.func.make_functional(population, dt=DT)


def test_make_functional_rejects_current_frame_evaluator_override():
    population = _single_compartment()
    population.mech._evaluate_current_frame = lambda *args, **kwargs: None

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="standard MechanismHandler current-frame evaluator",
    ):
        dn.func.make_functional(population, dt=DT)


def test_make_functional_rejects_current_frame_evaluator_class_override(monkeypatch):
    population = _single_compartment()
    monkeypatch.setattr(
        MechanismHandler,
        "_evaluate_current_frame",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="standard MechanismHandler current-frame evaluator",
    ):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize(
    ("factory", "hook"),
    [
        (_single_compartment, "_step"),
        (_unmyelinated, "_step"),
        (_unmyelinated, "_select_solver"),
    ],
)
def test_integrator_contract_changes_invalidate_an_existing_plan(factory, hook):
    population = factory()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    setattr(population.integrator, hook, lambda *args, **kwargs: None)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure changed after make_functional",
    ):
        functional.prepare(tensors.parameters, tensors.constants)


def test_current_frame_evaluator_change_invalidates_an_existing_plan():
    population = _single_compartment()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    population.mech._evaluate_current_frame = lambda *args, **kwargs: None

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure changed after make_functional",
    ):
        functional.prepare(tensors.parameters, tensors.constants)
