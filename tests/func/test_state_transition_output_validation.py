"""Functional parity for State runtime output validation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State

DT = 0.0125
DTYPE = torch.float64

pytestmark = pytest.mark.cpu


class _FunctionalAssignedState(State):
    State.STATE("a")
    State.ASSIGNED("rate")
    State.DERIVATIVE("a' = rate")

    def state_defaults(self, v, values):
        del values
        return {"a": torch.zeros_like(v)}

    def assigned_values(self, v, values):
        del values
        return {"rate": 0.5 * torch.ones_like(v)}


class _FunctionalAdvanceState(State):
    State.STATE("b")
    State.CARRY("memory")
    State.DERIVATIVE("b' = 0.0 * b")

    def state_defaults(self, v, values):
        del values
        return {"b": torch.ones_like(v)}

    def advance(self, v, dt, values):
        del v
        return {
            "b": values["b"] + dt,
            "memory": values["memory"] + values["b"],
        }


class _FunctionalStateMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_FunctionalAssignedState, _FunctionalAdvanceState)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return 0.0 * v


class _BadAssignedState(State):
    State.STATE("a")
    State.ASSIGNED("rate")
    State.DERIVATIVE("a' = rate")

    def state_defaults(self, v, values):
        del values
        return {"a": torch.zeros_like(v)}

    def assigned_values(self, v, values):
        del v, values
        return {"rate": torch.tensor(1.0, dtype=DTYPE)}


class _BadAssignedMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_BadAssignedState)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return 0.0 * v


class _BadAdvanceState(State):
    State.STATE("a", "b")
    State.DERIVATIVE("a' = 0.0 * a", "b' = 0.0 * b")

    def state_defaults(self, v, values):
        del values
        return {"a": torch.zeros_like(v), "b": torch.ones_like(v)}

    def advance(self, v, dt, values):
        del v
        return {"a": values["a"] + dt}


class _BadAdvanceMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_BadAdvanceState)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return 0.0 * v


def _model(mechanism_type):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(mechanism_type)
        model.initialize()
        model.train()
    return model


def _assert_tree_close(actual, expected):
    actual_values, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_values, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(
        actual_values,
        expected_values,
        strict=True,
    ):
        torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize(
    ("mechanism_type", "error", "match"),
    [
        (_BadAssignedMechanism, ValueError, "assigned_values.*must match"),
        (_BadAdvanceMechanism, KeyError, "advance.*did not return.*b"),
    ],
)
def test_functional_lowering_uses_state_runtime_output_validation(
    mechanism_type,
    error,
    match,
):
    model = _model(mechanism_type)
    with pytest.raises(error, match=match):
        dn.func.make_functional(model, dt=DT)


def test_compiled_functional_step_skips_validation_then_eager_validates_once():
    model = _model(_FunctionalStateMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    transition_mechanism = functional._transition.population.mech.mechanisms[
        "_FunctionalStateMechanism"
    ]
    assigned_state = transition_mechanism.DE["_FunctionalAssignedState"]
    advance_state = transition_mechanism.DE["_FunctionalAdvanceState"]
    assert not assigned_state._assigned_schema_validated
    assert not advance_state._advance_schema_validated

    def step(state):
        return functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )[0]

    compiled = torch.compile(step, backend="eager", fullgraph=True)
    compiled_state = compiled(tensors.state)
    assert not assigned_state._assigned_schema_validated
    assert not advance_state._advance_schema_validated

    eager_state = step(tensors.state)
    assert assigned_state._assigned_schema_validated
    assert advance_state._advance_schema_validated
    _assert_tree_close(compiled_state, eager_state)


def test_functional_admission_rejects_postconstruction_state_hook_role_change():
    class _DefaultAdvanceState(State):
        State.STATE("value")
        State.DERIVATIVE("value' = 0.0 * value")

        def state_defaults(self, v, values):
            del values
            return {"value": torch.zeros_like(v)}

    class _DefaultAdvanceMechanism(Mechanism):
        Mechanism.STATE_BUNDLE(_DefaultAdvanceState)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.AFFINE("i")

        def i(self, v):
            return 0.0 * v

    model = _model(_DefaultAdvanceMechanism)
    original = _DefaultAdvanceState.advance

    def replacement(self, v, dt, values):
        del self, v, dt
        return {"value": values["value"]}

    _DefaultAdvanceState.advance = replacement
    try:
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="changed its role-based transition hooks after construction",
        ):
            dn.func.make_functional(model, dt=DT)
    finally:
        _DefaultAdvanceState.advance = original


@pytest.mark.parametrize("owner", ["mechanism", "state"])
def test_functional_admission_rejects_assigned_hook_without_declared_outputs(owner):
    class _NoAssignedState(State):
        State.STATE("value")
        State.DERIVATIVE("value' = 0.0 * value")

        def state_defaults(self, v, values):
            del values
            return {"value": torch.zeros_like(v)}

        if owner == "state":

            def assigned_values(self, v, values):
                del self, v, values
                return {}

    class _NoAssignedMechanism(Mechanism):
        Mechanism.STATE_BUNDLE(_NoAssignedState)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.AFFINE("i")

        if owner == "mechanism":

            def assigned_values(self, v, values):
                del self, v, values
                return {}

        def i(self, v):
            return 0.0 * v

    model = _model(_NoAssignedMechanism)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="authors assigned_values but declares no ASSIGNED outputs",
    ):
        dn.func.make_functional(model, dt=DT)
