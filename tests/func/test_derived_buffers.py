"""Functional correctness oracles for initialization-derived workspaces."""

from __future__ import annotations

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

    def derive_buffers(self):
        return {
            "drive": self.state_scale
            + CELSIUS_COEFFICIENT * self.celsius
            + DIAMETER_COEFFICIENT * self.diam
        }


class _DerivedWorkspaceMechanism(Mechanism):
    Mechanism.STATE(_DerivedState)
    Mechanism.INIT(x=0.0)
    Mechanism.GLOBAL(mechanism_scale=1.0e-7, e=-55.0)
    Mechanism.DERIVED_BUFFER("conductance")
    Mechanism.BUFFER("trace")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_buffers(self):
        return {
            "conductance": self.mechanism_scale
            * (1.0 + 0.001 * self.celsius)
            * (1.0 + 0.05 * self.diam)
        }

    def initial(self, v):
        self.trace = torch.zeros_like(v)

    def breakpoint(self, v):
        self.trace = self.conductance + self.x

    def i(self, v):
        return self.conductance * (v - self.e)


class _MutatingDerivedWorkspace(Mechanism):
    Mechanism.GLOBAL(base=1.0e-8, e=-60.0)
    Mechanism.DERIVED_BUFFER("workspace")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_buffers(self):
        return {"workspace": self.base * torch.ones_like(self.diam)}

    def breakpoint(self, v):
        # Deliberately invalid authored behavior: functional execution must at
        # least contain the mutation to its private transition workspace.
        if self.training:
            self.workspace.add_(self.base)

    def i(self, v):
        return self.workspace * (v - self.e)


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
        if mechanism is _MutatingDerivedWorkspace:
            # Initialization evaluates BREAKPOINT once. Keep the deliberately
            # invalid mutation specific to the subsequent transition under test.
            model.eval()
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


def test_inplace_derived_workspace_mutation_is_call_local():
    model = _model(_MutatingDerivedWorkspace)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    prepared_path = "integrator.mech.mechanisms._MutatingDerivedWorkspace.workspace"
    workspace = prepared.values["mechanisms"][prepared_path]
    original_prepared = workspace.clone()
    original_source = model.mech._MutatingDerivedWorkspace.workspace.clone()
    prepared_version = workspace._version

    first, _aux = functional.step(tensors.parameters, prepared, tensors.state)
    second, _aux = functional.step(tensors.parameters, prepared, tensors.state)

    assert workspace._version == prepared_version
    torch.testing.assert_close(workspace, original_prepared, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        model.mech._MutatingDerivedWorkspace.workspace,
        original_source,
        rtol=0.0,
        atol=0.0,
    )
    _assert_every_leaf_close(first, second, rtol=0.0, atol=0.0)
