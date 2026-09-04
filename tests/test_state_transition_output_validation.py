"""Runtime output contracts for authored State transition hooks."""

from __future__ import annotations

from types import MethodType

import pytest
import torch

from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms import _state as state_impl

DTYPE = torch.float64
SHAPE = (1, 3)


class _AssignedState(State):
    State.STATE("value")
    State.ASSIGNED("rate", "offset")
    State.DERIVATIVE("value' = rate + offset")

    def state_defaults(self, v, values):
        del values
        return {"value": torch.zeros_like(v)}

    def assigned_values(self, v, values):
        del values
        return {"rate": torch.ones_like(v), "offset": torch.zeros_like(v)}


class _AssignedMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_AssignedState)


class _MissingAssignedHookState(State):
    State.STATE("value")
    State.ASSIGNED("rate")
    State.DERIVATIVE("value' = rate")

    def state_defaults(self, v, values):
        del values
        return {"value": torch.zeros_like(v)}


class _MissingAssignedHookMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_MissingAssignedHookState)


class _AdvanceState(State):
    State.STATE("x", "y")
    State.CARRY("memory")
    State.DERIVATIVE("x' = 0.0 * x", "y' = 0.0 * y")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v), "y": torch.ones_like(v)}

    def advance(self, v, dt, values):
        del v
        return {
            "x": values["x"] + dt,
            "y": values["y"] - dt,
            "memory": values["x"] + values["y"],
        }


class _AdvanceMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_AdvanceState)


def _mechanism(mechanism_type):
    voltage = torch.full(SHAPE, -65.0, dtype=DTYPE)
    mechanism = mechanism_type(
        mechanism_type.__name__,
        torch.tensor(34.0, dtype=DTYPE),
        torch.ones(SHAPE, dtype=DTYPE),
        SHAPE,
        SHAPE,
    )
    mechanism._init_buffers_s(voltage)
    return mechanism, voltage


def _bad_assigned_output(kind, voltage):
    rate = torch.ones_like(voltage)
    offset = torch.zeros_like(voltage)
    if kind == "nonmapping":
        return (rate, offset)
    if kind == "missing":
        return {"rate": rate}
    if kind == "unknown":
        return {"rate": rate, "offset": offset, "extra": rate}
    if kind == "nontensor":
        return {"rate": rate, "offset": 0.0}
    if kind == "shape":
        return {"rate": rate, "offset": offset.sum()}
    if kind == "dtype":
        return {"rate": rate, "offset": offset.float()}
    if kind == "device":
        return {"rate": rate, "offset": torch.empty_like(offset, device="meta")}
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "error", "match"),
    [
        ("nonmapping", TypeError, "must return a mapping"),
        ("missing", KeyError, "did not return declared outputs.*offset"),
        ("unknown", KeyError, "returned undeclared outputs.*extra"),
        ("nontensor", TypeError, "must be a Tensor"),
        ("shape", ValueError, "must match"),
        ("dtype", ValueError, "must match"),
        ("device", ValueError, "must match"),
    ],
)
def test_imperative_state_assigned_values_validate_exact_schema_once(
    kind,
    error,
    match,
):
    mechanism, voltage = _mechanism(_AssignedMechanism)
    state = mechanism.DE["_AssignedState"]

    def malformed(self, v, values):
        del self, values
        return _bad_assigned_output(kind, v)

    state.assigned_values = MethodType(malformed, state)
    with pytest.raises(error, match=match):
        mechanism._advance_states(voltage, voltage.new_tensor(0.1))

    assert not state._assigned_schema_validated
    torch.testing.assert_close(mechanism.value, torch.zeros_like(voltage))


def test_declared_state_assigned_requires_an_authored_complete_hook():
    mechanism, voltage = _mechanism(_MissingAssignedHookMechanism)

    with pytest.raises(KeyError, match="did not return declared outputs.*rate"):
        mechanism._advance_states(voltage, voltage.new_tensor(0.1))


def test_state_assigned_mapping_order_is_semantically_irrelevant():
    mechanism, voltage = _mechanism(_AssignedMechanism)
    state = mechanism.DE["_AssignedState"]

    def reordered(self, v, values):
        del self, values
        return {"offset": torch.zeros_like(v), "rate": torch.ones_like(v)}

    state.assigned_values = MethodType(reordered, state)
    mechanism._advance_states(voltage, voltage.new_tensor(0.1))

    assert state._assigned_schema_validated
    torch.testing.assert_close(mechanism.value, torch.full_like(voltage, 0.1))


def _bad_advance_output(kind, voltage):
    x = torch.ones_like(voltage)
    y = 2.0 * torch.ones_like(voltage)
    if kind == "nonmapping":
        return (x, y)
    if kind == "missing":
        return {"x": x}
    if kind == "unknown":
        return {"x": x, "y": y, "extra": x}
    if kind == "nontensor":
        return {"x": x, "y": 2.0}
    if kind == "shape":
        return {"x": x, "y": y.sum()}
    if kind == "dtype":
        return {"x": x, "y": y.float()}
    if kind == "device":
        return {"x": x, "y": torch.empty_like(y, device="meta")}
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "error", "match"),
    [
        ("nonmapping", TypeError, "must return a mapping"),
        ("missing", KeyError, "did not return declared outputs.*y"),
        ("unknown", KeyError, "returned undeclared outputs.*extra"),
        ("nontensor", TypeError, "must be a Tensor"),
        ("shape", ValueError, "must match"),
        ("dtype", ValueError, "must match"),
        ("device", ValueError, "must match"),
    ],
)
def test_imperative_state_advance_validates_before_atomic_commit(
    kind,
    error,
    match,
):
    mechanism, voltage = _mechanism(_AdvanceMechanism)
    state = mechanism.DE["_AdvanceState"]
    before = (mechanism.x.clone(), mechanism.y.clone(), state.memory.clone())

    def malformed(self, v, dt, values):
        del self, dt, values
        return _bad_advance_output(kind, v)

    state.advance = MethodType(malformed, state)
    with pytest.raises(error, match=match):
        mechanism._advance_states(voltage, voltage.new_tensor(0.1))

    assert not state._advance_schema_validated
    torch.testing.assert_close(mechanism.x, before[0])
    torch.testing.assert_close(mechanism.y, before[1])
    torch.testing.assert_close(state.memory, before[2])


def test_state_advance_requires_state_but_allows_optional_carry():
    mechanism, voltage = _mechanism(_AdvanceMechanism)
    state = mechanism.DE["_AdvanceState"]

    def states_only(self, v, dt, values):
        del self, v
        return {"x": values["x"] + dt, "y": values["y"] - dt}

    state.advance = MethodType(states_only, state)
    mechanism._advance_states(voltage, voltage.new_tensor(0.25))

    assert state._advance_schema_validated
    assert state._advance_return_names == ("x", "y")
    torch.testing.assert_close(mechanism.x, torch.full_like(voltage, 0.25))
    torch.testing.assert_close(mechanism.y, torch.full_like(voltage, 0.75))
    torch.testing.assert_close(state.memory, torch.zeros_like(voltage))


def test_state_runtime_schemas_validate_only_on_first_eager_transition(monkeypatch):
    assigned_mechanism, voltage = _mechanism(_AssignedMechanism)
    advance_mechanism, _ = _mechanism(_AdvanceMechanism)
    calls = []
    original = state_impl._validate_runtime_outputs

    def counted(module, method_name, *args, **kwargs):
        calls.append((type(module).__name__, method_name))
        return original(module, method_name, *args, **kwargs)

    monkeypatch.setattr(state_impl, "_validate_runtime_outputs", counted)
    for _ in range(2):
        assigned_mechanism._advance_states(voltage, voltage.new_tensor(0.1))
        advance_mechanism._advance_states(voltage, voltage.new_tensor(0.1))

    assert calls == [
        ("_AssignedState", "assigned_values"),
        ("_AdvanceState", "advance"),
    ]
    assert assigned_mechanism.DE["_AssignedState"]._assigned_schema_validated
    assert advance_mechanism.DE["_AdvanceState"]._advance_schema_validated


def test_state_runtime_schema_cache_resets_after_dtype_conversion():
    class _FixedFloatAssignedState(State):
        State.STATE("value")
        State.ASSIGNED("rate")
        State.DERIVATIVE("value' = rate")

        def assigned_values(self, v, values):
            del values
            return {"rate": torch.ones(v.shape, device=v.device, dtype=torch.float32)}

    state = _FixedFloatAssignedState(
        torch.tensor(34.0, dtype=torch.float32),
        torch.ones(SHAPE, dtype=torch.float32),
        None,
        SHAPE,
        SHAPE,
    )
    voltage = torch.zeros(SHAPE, dtype=torch.float32)
    state.advance(voltage, voltage.new_tensor(0.1), {"value": voltage})
    assert state._assigned_schema_validated

    state.double()
    assert not state._assigned_schema_validated
    voltage = voltage.double()
    with pytest.raises(ValueError, match="must match.*torch.float64"):
        state.advance(voltage, voltage.new_tensor(0.1), {"value": voltage})
