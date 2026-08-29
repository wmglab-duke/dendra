"""Functional correctness oracles for initialization-derived workspaces."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State

DT = 0.03125
STEPS = 3
CELSIUS_COEFFICIENT = 0.01
DIAMETER_COEFFICIENT = 0.125


class _DerivedState(State):
    State.STATE("x")
    State.GLOBAL(state_scale=0.25)
    State.DERIVED_BUFFER("drive")
    State.DERIVATIVE("x' = drive")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}

    def derive_buffers(self):
        return {
            "drive": self.state_scale
            + CELSIUS_COEFFICIENT * self.celsius
            + DIAMETER_COEFFICIENT * self.diam
        }


class _DerivedWorkspaceMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_DerivedState)
    Mechanism.GLOBAL(mechanism_scale=1.0e-7, e=-55.0)
    Mechanism.DERIVED_BUFFER("conductance")
    Mechanism.CARRY("trace")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_buffers(self):
        return {
            "conductance": self.mechanism_scale
            * (1.0 + 0.001 * self.celsius)
            * (1.0 + 0.05 * self.diam)
        }

    def initial_values(self, v, values):
        del values
        return {"trace": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del v, dt
        return {"trace": values["conductance"] + values["x"]}

    def i(self, v):
        return self.conductance * (v - self.e)


class _MutatingDerivedWorkspace(Mechanism):
    Mechanism.GLOBAL(base=1.0e-8, e=-60.0)
    Mechanism.DERIVED_BUFFER("workspace")
    Mechanism.ASSIGNED("probe")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_buffers(self):
        return {"workspace": self.base * torch.ones_like(self.diam)}

    def assigned_values(self, v, values):
        # Deliberately invalid authored behavior: functional execution must at
        # least contain the mutation to its private transition workspace.
        del values
        if self.training:
            self.workspace.add_(self.base)
        return {"probe": torch.zeros_like(v)}

    def i(self, v):
        return self.workspace * (v - self.e)


class _RebindingDerivedWorkspace(_MutatingDerivedWorkspace):
    def assigned_values(self, v, values):
        del values
        if self.training:
            self.workspace = self.workspace + self.base
        return {"probe": torch.zeros_like(v)}


class _DeferredLayoutMechanism(Mechanism):
    Mechanism.CARRY("history", shape="deferred")
    Mechanism.DERIVED_BUFFER("routing", dtype=torch.long, shape="deferred")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_buffers(self):
        return {"routing": torch.arange(2, device=self.diam.device)}

    def initial_values(self, v, values):
        del values
        return {"history": torch.stack((v, v), dim=-1)}

    def advance(self, v, dt, values):
        del v
        return {"history": values["history"] + dt}

    def i(self, v):
        return 0.0 * v


def _model(mechanism=_DerivedWorkspaceMechanism):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(mechanism)
        if issubclass(mechanism, _MutatingDerivedWorkspace):
            # Initialization evaluates assigned_values once. Keep the deliberately
            # invalid mutation specific to the subsequent transition under test.
            model.eval()
        model.initialize()
        model.train()
    return model


def _deferred_model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(_DeferredLayoutMechanism)
        model.batch(2)
        model.initialize()
        model.train()
    return model


def _parameter_name(parameters, suffix):
    matches = [name for name in parameters if name.endswith(suffix)]
    assert len(matches) == 1
    return matches[0]


def _assert_every_leaf_close(actual, expected, *, rtol=1.0e-11, atol=1.0e-12):
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
        )


def test_derived_workspaces_are_prepared_static_tensors_not_explicit_carry():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    mechanism_name = "_DerivedWorkspaceMechanism"
    # state_name = "_DerivedState"
    assert set(tensors.state["mechanisms"][mechanism_name]) == {"x"}
    assert set(tensors.state["mechanism_buffers"][mechanism_name]) == {"trace"}
    assert "state_buffers" not in tensors.state

    prepared_values = prepared.values["mechanisms"]
    mechanism_path = "integrator.mech.mechanisms._DerivedWorkspaceMechanism.conductance"
    state_path = (
        "integrator.mech.mechanisms._DerivedWorkspaceMechanism.DE._DerivedState.drive"
    )
    expected_conductance = (
        model.mech._DerivedWorkspaceMechanism.mechanism_scale
        * (1.0 + 0.001 * model.celsius)
        * (1.0 + 0.05 * model.diam)
    )
    expected_drive = (
        model.mech._DerivedWorkspaceMechanism.DE._DerivedState.state_scale
        + CELSIUS_COEFFICIENT * model.celsius
        + DIAMETER_COEFFICIENT * model.diam
    )
    torch.testing.assert_close(
        prepared_values[mechanism_path], expected_conductance, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        prepared_values[state_path], expected_drive, rtol=0.0, atol=0.0
    )


def test_deferred_layouts_are_frozen_and_preserved_by_functional_step():
    model = _deferred_model()
    mechanism = model.mech._DeferredLayoutMechanism
    expected_shape = (*model.shape, 2)

    assert mechanism.history.shape == expected_shape
    assert mechanism.routing.shape == (2,)
    assert mechanism._carry_resolved_shapes == {"history": expected_shape}
    assert mechanism._derived_resolved_shapes == {"routing": (2,)}

    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    history = tensors.state["mechanism_buffers"]["_DeferredLayoutMechanism"]["history"]
    next_state, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    next_history = next_state["mechanism_buffers"]["_DeferredLayoutMechanism"][
        "history"
    ]

    assert history.shape == expected_shape
    assert next_history.shape == expected_shape
    torch.testing.assert_close(next_history, history + DT)


def test_parameter_temperature_and_geometry_substitutions_have_exact_gradients():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism_name = "_DerivedWorkspaceMechanism"
    # state_name = "_DerivedState"
    parameter_name = _parameter_name(
        tensors.parameters,
        "DE._DerivedState.state_scale_param",
    )

    def next_x(state_scale, celsius, diam):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[parameter_name] = state_scale
        parameters["celsius_param"] = celsius
        constants["diam"] = diam
        state, _aux = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )
        return state["mechanisms"][mechanism_name]["x"]

    state_scale = tensors.parameters[parameter_name]
    celsius = tensors.parameters["celsius_param"]
    diam = tensors.constants["diam"]
    changed = next_x(state_scale + 0.5, celsius - 5.0, diam * 1.25)
    expected = DT * (
        state_scale
        + 0.5
        + CELSIUS_COEFFICIENT * (celsius - 5.0)
        + DIAMETER_COEFFICIENT * diam * 1.25
    )
    torch.testing.assert_close(changed, expected, rtol=1.0e-12, atol=1.0e-13)

    gradients = torch.func.grad(
        lambda scale, temperature, diameter: next_x(scale, temperature, diameter).sum(),
        argnums=(0, 1, 2),
    )(state_scale, celsius, diam)
    torch.testing.assert_close(
        gradients[0],
        state_scale.new_tensor(DT * diam.numel()),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    torch.testing.assert_close(
        gradients[1],
        celsius.new_tensor(DT * CELSIUS_COEFFICIENT * diam.numel()),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    torch.testing.assert_close(
        gradients[2],
        torch.full_like(diam, DT * DIAMETER_COEFFICIENT),
        rtol=1.0e-12,
        atol=1.0e-13,
    )


def test_vmap_over_raw_parameter_temperature_and_geometry_matches_loop():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism_name = "_DerivedWorkspaceMechanism"
    parameter_name = _parameter_name(
        tensors.parameters,
        "DE._DerivedState.state_scale_param",
    )

    def run_lane(state_scale, celsius, diam):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[parameter_name] = state_scale
        parameters["celsius_param"] = celsius
        constants["diam"] = diam
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )[0]["mechanisms"][mechanism_name]["x"]

    state_scale = tensors.parameters[parameter_name]
    celsius = tensors.parameters["celsius_param"]
    diam = tensors.constants["diam"]
    scale_lanes = torch.stack(
        (torch.zeros_like(state_scale), state_scale, 1.5 * state_scale)
    )
    celsius_lanes = torch.stack((torch.zeros_like(celsius), celsius, celsius + 7.0))
    diameter_lanes = torch.stack((0.75 * diam, diam, 1.25 * diam))

    actual = torch.vmap(run_lane)(scale_lanes, celsius_lanes, diameter_lanes)
    expected = torch.stack(
        tuple(
            run_lane(scale, temperature, diameter)
            for scale, temperature, diameter in zip(
                scale_lanes,
                celsius_lanes,
                diameter_lanes,
                strict=True,
            )
        )
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-13)


def test_atomic_fullgraph_and_aot_backward_match_eager():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism_name = "_DerivedWorkspaceMechanism"
    parameter_name = _parameter_name(
        tensors.parameters,
        "DE._DerivedState.state_scale_param",
    )

    def next_x(state_scale, celsius, diam):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[parameter_name] = state_scale
        parameters["celsius_param"] = celsius
        constants["diam"] = diam
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )[0]["mechanisms"][mechanism_name]["x"]

    state_scale = tensors.parameters[parameter_name]
    celsius = tensors.parameters["celsius_param"]
    diam = tensors.constants["diam"].detach().clone().requires_grad_()
    eager = next_x(state_scale, celsius, diam)
    compiled = torch.compile(next_x, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(state_scale, celsius, diam)
    torch.testing.assert_close(actual, eager, rtol=0.0, atol=0.0)

    def loss(scale, temperature, diameter):
        return next_x(scale, temperature, diameter).square().mean()

    expected_loss = loss(state_scale, celsius, diam)
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (state_scale, celsius, diam),
    )
    compiled_loss = torch.compile(loss, backend="aot_eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual_loss = compiled_loss(state_scale, celsius, diam)
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (state_scale, celsius, diam),
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0.0, atol=0.0)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1.0e-11,
            atol=1.0e-12,
        )


def test_chained_fused_and_imperative_execution_match_every_carry_leaf():
    model = _model()
    imperative = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    chained = tensors.state
    for _ in range(STEPS):
        chained, _aux = functional.step(
            tensors.parameters,
            prepared,
            chained,
        )
    fused, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=STEPS,
    )

    dt = torch.as_tensor(DT, dtype=imperative.dtype(), device=imperative.device())
    imperative.integrator._initialize(
        imperative,
        dt,
        force=True,
        compile_scope="population",
    )
    for _ in range(STEPS):
        imperative.integrator.step(imperative, dt)
        imperative.t = imperative.t + dt
    expected = functional.extract(imperative).state

    _assert_every_leaf_close(chained, fused)
    _assert_every_leaf_close(chained, expected)


@pytest.mark.parametrize(
    "mechanism_type",
    [_MutatingDerivedWorkspace, _RebindingDerivedWorkspace],
)
def test_authored_derived_workspace_writes_fail_during_lowering(mechanism_type):
    model = _model(mechanism_type)
    source = getattr(model.mech, mechanism_type.__name__)
    original_source = source.workspace.clone()
    source_version = source.workspace._version

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="lowering.*mutation or rebinding.*read-only.*workspace",
    ):
        dn.func.make_functional(model, dt=DT)

    assert source.workspace._version == source_version
    torch.testing.assert_close(
        source.workspace,
        original_source,
        rtol=0.0,
        atol=0.0,
    )


def test_runtime_checkpoint_omits_reconstructible_derived_workspaces():
    model = _model()
    mechanism = model.mech._DerivedWorkspaceMechanism
    state = mechanism.DE._DerivedState
    dt = torch.as_tensor(DT, dtype=model.dtype(), device=model.device())
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    checkpoint = model.state_dict_for_checkpoint()

    assert "_DerivedWorkspaceMechanism.conductance" not in checkpoint["mech"]
    assert "_DerivedWorkspaceMechanism.DE._DerivedState.drive" not in checkpoint["mech"]
    assert "_DerivedWorkspaceMechanism.trace" in checkpoint["mech"]
    assert "_DerivedWorkspaceMechanism.x" in checkpoint["mech"]

    old_conductance = mechanism.conductance.clone()
    old_drive = state.drive.clone()
    with torch.no_grad():
        mechanism.mechanism_scale_param.mul_(2.0)
        state.state_scale_param.add_(0.5)
    model.initialize()
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    current_conductance = mechanism.conductance.clone()
    current_drive = state.drive.clone()
    assert not torch.equal(current_conductance, old_conductance)
    assert not torch.equal(current_drive, old_drive)

    model.restore_dict_from_checkpoint(checkpoint)
    torch.testing.assert_close(
        mechanism.conductance,
        current_conductance,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(state.drive, current_drive, rtol=0.0, atol=0.0)


def test_derived_builder_hook_identity_is_part_of_source_structure(monkeypatch):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def replacement(self):
        return {"conductance": 2.0 * self.mechanism_scale * self.diam}

    monkeypatch.setattr(_DerivedWorkspaceMechanism, "derive_buffers", replacement)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure changed|lower it again",
    ):
        functional.prepare(tensors.parameters, tensors.constants)
