"""Previous State and configured IC retain distinct ownership and validation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from tests.func.test_functional_initialization import (
    _assert_tensor_snapshot,
    _tensor_snapshot,
)

PROBE = "_PreviousProbe"
X = f"mechanisms.{PROBE}.x"
Y = f"mechanisms.{PROBE}.y"
pytestmark = pytest.mark.cpu


class _PreviousState(State):
    State.STATE("x", "y")
    State.DERIVATIVE("x' = 0.0 * x", "y' = 0.0 * y")


class _PreviousProbe(Mechanism):
    Mechanism.STATE_BUNDLE(_PreviousState)


class _BumpPrevious(torch.nn.Module):
    def forward(self, x, y):
        return x + 1.0, y + 2.0


class _ObservePrevious(torch.nn.Module):
    def forward(self, voltage, x, y):
        return (voltage + x + 2.0 * y,)


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(
        JIT=0,
        REQUIRE_GRAD=1,
        IMEM=False,
        DEVICE="cpu",
        DTYPE=torch.float64,
        CHECK_PURITY=False,
    ):
        yield


def _model():
    model = dn.SingleCompartment(
        N=1, C=2, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(_PreviousProbe, ic={"x": 0.4, "y": 0.8})
    model.initialize()
    model.register_pre_initialize_transform(
        "bump_previous",
        _BumpPrevious(),
        reads=(f"state.{X}", f"state.{Y}"),
        writes=(f"state.{X}", f"state.{Y}"),
    )
    model.register_pre_initialize_transform(
        "observe_previous",
        _ObservePrevious(),
        reads=("state.integrator.v", f"state.{X}", f"state.{Y}"),
        writes=("state.integrator.v",),
    )
    return model


def _lower(check_purity=False):
    with dn.ctx(CHECK_PURITY=check_purity):
        return dn.func.make_functional(_model(), dt=0.025)


@pytest.mark.parametrize("check_purity", (False, True))
def test_partial_state_uses_captured_previous_state_without_replacing_ic(check_purity):
    functional, tensors = _lower(check_purity)
    inputs = tensors.initialization._replace(
        states={
            X: tensors.state["integrator"]["v"].new_full((1, 2), 0.9),
            Y: tensors.state["integrator"]["v"].new_full((1, 2), 1.2),
        }
    )
    expected = functional.initialize(tensors.parameters, tensors.constants, inputs)
    actual = functional.initialize(
        tensors.parameters, tensors.constants, inputs, state={}
    )
    for key in ("x", "y"):
        torch.testing.assert_close(
            actual.state["mechanisms"][PROBE][key],
            expected.state["mechanisms"][PROBE][key],
        )
    torch.testing.assert_close(
        actual.state["integrator"]["v"],
        torch.full_like(actual.state["integrator"]["v"], -53.0),
    )
    torch.testing.assert_close(actual.state["mechanisms"][PROBE]["x"], inputs.states[X])
    torch.testing.assert_close(actual.state["mechanisms"][PROBE]["y"], inputs.states[Y])


@pytest.mark.parametrize("check_purity", (False, True))
def test_partial_previous_state_keeps_shared_ic_gradients_and_owned_outputs(
    check_purity,
):
    functional, tensors = _lower(check_purity)
    shared_ic = torch.full_like(tensors.initialization.states[X], 0.6).requires_grad_()
    previous_x = torch.full_like(shared_ic, 3.0).requires_grad_()
    previous_y = torch.full_like(shared_ic, 4.0).requires_grad_()
    snapshots = tuple(
        _tensor_snapshot(value) for value in (shared_ic, previous_x, previous_y)
    )
    inputs = tensors.initialization._replace(states={X: shared_ic, Y: shared_ic})
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        inputs,
        state={"mechanisms": {PROBE: {"x": previous_x, "y": previous_y}}},
    )
    x = initialized.state["mechanisms"][PROBE]["x"]
    y = initialized.state["mechanisms"][PROBE]["y"]
    voltage = initialized.state["integrator"]["v"]
    torch.testing.assert_close(x, shared_ic)
    torch.testing.assert_close(y, shared_ic)
    torch.testing.assert_close(voltage, torch.full_like(voltage, -44.0))
    gradients = torch.autograd.grad(
        (x + 2.0 * y + voltage).sum(), (shared_ic, previous_x, previous_y)
    )
    for gradient, coefficient in zip(gradients, (3.0, 1.0, 2.0), strict=True):
        torch.testing.assert_close(gradient, torch.full_like(gradient, coefficient))
    assert initialized.initialization.states[X] is shared_ic
    assert initialized.initialization.states[Y] is shared_ic
    assert len({value.untyped_storage().data_ptr() for value in (x, y, shared_ic)}) == 3
    with torch.no_grad():
        x.add_(10.0)
        y.add_(20.0)
    for value, snapshot in zip(
        (shared_ic, previous_x, previous_y), snapshots, strict=True
    ):
        _assert_tensor_snapshot(value, snapshot)


@pytest.mark.parametrize(
    "malformed", ("not_mapping", "missing_owner", "shape", "dtype")
)
def test_supplied_mechanism_state_is_validated_at_its_use_boundary(malformed):
    functional, tensors = _lower()
    category = {
        PROBE: {key: value for key, value in tensors.state["mechanisms"][PROBE].items()}
    }
    if malformed == "not_mapping":
        category = torch.zeros(())
        error, message = TypeError, r"state\['mechanisms'\] must be a mapping"
    elif malformed == "missing_owner":
        category = {}
        error, message = KeyError, r"state\['mechanisms'\] must contain exactly"
    elif malformed == "shape":
        category[PROBE]["x"] = category[PROBE]["x"][:, :1]
        error, message = (
            ValueError,
            r"state leaf 'mechanisms\._PreviousProbe\.x' has shape",
        )
    else:
        category[PROBE]["x"] = category[PROBE]["x"].float()
        error, message = (
            ValueError,
            r"state leaf 'mechanisms\._PreviousProbe\.x' must use",
        )
    with pytest.raises(error, match=message):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state={"mechanisms": category},
        )


def test_initialization_input_validation_precedes_previous_state_validation():
    functional, tensors = _lower()
    inputs = tensors.initialization._replace(
        states={
            **tensors.initialization.states,
            X: tensors.initialization.states[X][:, :1],
        }
    )
    with pytest.raises(
        ValueError, match=r"initial state 'mechanisms\._PreviousProbe\.x' has shape"
    ):
        functional.initialize(
            tensors.parameters, tensors.constants, inputs, state={"mechanisms": {}}
        )
