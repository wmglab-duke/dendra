"""Focused contracts for State declarations and pure author hooks."""

from __future__ import annotations

import pytest
import torch

from dendra.models.mechanisms import Mechanism, State


class _OrderedBase(State):
    State.STATE("x", "y")
    State.CARRY("gain", dtype=torch.float64)
    State.ASSIGNED("rate", "offset")
    State.DERIVATIVE("x' = rate", "y' = offset")

    def assigned_values(self, v, values):
        return {
            "rate": values["gain"].to(v.dtype),
            "offset": torch.zeros_like(v),
        }

    def state_defaults(self, v, values):
        return {
            "x": torch.zeros_like(v) + values["celsius"] / 100.0,
            "y": torch.zeros_like(v) + values["diam"],
        }


class _OrderedChild(_OrderedBase):
    State.STATE("z", "x")
    State.CARRY("carry_flags", dtype=torch.bool, shape=(2,))
    State.ASSIGNED("scale", "rate")
    State.DERIVATIVE("z' = scale")

    def assigned_values(self, v, values):
        assigned = super().assigned_values(v, values)
        return {
            "rate": assigned["rate"],
            "offset": assigned["offset"],
            "scale": torch.zeros_like(v),
        }

    def initial_values(self, v, values):
        del values
        return {
            "gain": torch.ones_like(v, dtype=torch.float64),
            "carry_flags": torch.tensor([True, False], device=v.device),
        }


def _state(state_type=_OrderedChild):
    return state_type(
        torch.tensor(37.0, dtype=torch.float64),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        None,
        (3,),
        (3,),
    )


def test_state_declarations_preserve_authored_mro_order():
    assert _OrderedChild._state == ("x", "y", "z")
    assert _OrderedChild._carry == ("gain", "carry_flags")
    assert _OrderedChild._assigned == ("rate", "offset", "scale")


def test_state_carry_has_fixed_registered_shape_and_dtype():
    state = _state()

    assert state.gain.shape == (3,)
    assert state.gain.dtype is torch.float64
    assert state.carry_flags.shape == (2,)
    assert state.carry_flags.dtype is torch.bool
    assert state._carry_specs == {
        "gain": (torch.float64, "local"),
        "carry_flags": (torch.bool, (2,)),
    }

    outputs = state._derive_initial_values(
        torch.zeros(3, dtype=torch.float64),
        {},
        isolate=False,
    )
    assert outputs["gain"].shape == state.gain.shape
    assert outputs["carry_flags"].shape == state.carry_flags.shape
    assert outputs["carry_flags"].dtype is torch.bool


def test_explicit_state_carry_dtype_survives_module_conversion():
    state = _state().to(dtype=torch.float32)

    assert state.diam.dtype is torch.float32
    assert state.gain.dtype is torch.float64
    assert state.carry_flags.dtype is torch.bool


@pytest.mark.parametrize(
    ("dtype", "shape", "error", "message"),
    [
        ("float64", "local", TypeError, "CARRY dtype"),
        (None, "structural", ValueError, "CARRY shape"),
        (None, [1], TypeError, "CARRY shape"),
        (None, (True,), TypeError, "CARRY shape dimensions"),
        (None, (-1,), ValueError, "CARRY shape dimensions"),
    ],
)
def test_state_carry_rejects_ambiguous_storage_schemas(
    dtype,
    shape,
    error,
    message,
):
    with pytest.raises(error, match=message):

        class _InvalidCarrySchema(State):
            State.CARRY("carry", dtype=dtype, shape=shape)


def test_state_defaults_receive_explicit_temperature_and_geometry():
    state = _state(_OrderedBase)
    voltage = torch.full((3,), -65.0, dtype=torch.float64)

    defaults = state._derive_initial_state_values(voltage, require_complete=True)

    torch.testing.assert_close(defaults["x"], torch.full_like(voltage, 0.37))
    torch.testing.assert_close(defaults["y"], state.diam)


def test_default_advance_uses_assigned_values_and_returns_state_mapping():
    state = _state(_OrderedBase)
    voltage = torch.full((3,), -65.0, dtype=torch.float64)
    values = {
        "x": torch.zeros_like(voltage),
        "y": torch.ones_like(voltage),
        "gain": torch.full_like(voltage, 2.0),
        # A same-named parent input is stale for this State. The freshly
        # evaluated ASSIGNED output must own the generated solver slot.
        "rate": torch.full_like(voltage, 99.0),
    }

    advanced = state.advance(voltage, 0.25, values)

    torch.testing.assert_close(advanced["x"], torch.full_like(voltage, 0.5))
    torch.testing.assert_close(advanced["y"], values["y"])


def test_inherited_carry_schema_conflicts_fail_at_class_definition():
    with pytest.raises(ValueError, match="conflicting schemas"):

        class _InvalidCarry(_OrderedBase):
            State.CARRY("gain", dtype=torch.float32)


def test_state_value_cannot_be_redeclared_as_assigned():
    with pytest.raises(ValueError, match="State variables cannot also be ASSIGNED.*x"):

        class _StateAssignedCollision(State):
            State.STATE("x")
            State.ASSIGNED("x")


def test_state_assigned_can_still_shadow_an_explicit_parent_input():
    class _AssignedInputShadow(State):
        State.STATE("x")
        State.GLOBAL(rate=1.0)
        State.ASSIGNED("rate")
        State.DERIVATIVE("x' = rate")

        def assigned_values(self, v, values):
            return {"rate": values["rate"] + torch.ones_like(v)}

    state = _state(_AssignedInputShadow)
    voltage = torch.zeros(3, dtype=torch.float64)
    advanced = state.advance(
        voltage,
        0.25,
        {
            "x": torch.zeros_like(voltage),
            "rate": torch.full_like(voltage, 2.0),
        },
    )
    torch.testing.assert_close(advanced["x"], torch.full_like(voltage, 0.75))


def test_removed_ambiguous_declarations_are_not_public_api():
    assert not hasattr(State, "BUFFER")
    assert not hasattr(Mechanism, "BUFFER")
    assert not hasattr(Mechanism, "INIT")
    assert not hasattr(Mechanism, "states_dict")


@pytest.mark.parametrize(
    ("base", "hook", "replacement"),
    [
        (State, "breakpoint", "assigned_values"),
        (State, "inf", "state_defaults"),
        (State, "initial", "initial_values"),
        (State, "initial_outputs", "initial_values"),
        (State, "initialize", "initial_values"),
        (State, "solve", "advance"),
        (State, "_advance", "advance"),
        (Mechanism, "breakpoint", "assigned_values"),
        (Mechanism, "initial", "initial_values"),
        (Mechanism, "initial_outputs", "initial_values"),
        (Mechanism, "_advance", "advance"),
    ],
)
def test_removed_lifecycle_hooks_fail_at_class_definition(base, hook, replacement):
    def removed_hook(*args, **kwargs):
        del args, kwargs
        return {}

    with pytest.raises(TypeError, match=rf"{hook} -> {replacement}"):
        type(f"_Invalid_{base.__name__}_{hook}", (base,), {hook: removed_hook})


@pytest.mark.parametrize("attribute", ["method", "method_kwargs"])
def test_state_solver_policy_must_use_method_declaration(attribute):
    with pytest.raises(TypeError, match=r"State\.METHOD"):
        type(f"_InvalidDirect_{attribute}", (State,), {attribute: "cnexp"})
