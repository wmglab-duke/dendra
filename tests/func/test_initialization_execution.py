"""Fresh execution and initialization diagnostics share an ordered tensor frame."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tensor_snapshot,
    _assert_tree_close,
    _module_snapshot,
    _tensor_snapshot,
)
from tests.func.test_initialization_parity import _live_state

DT = 0.025
PROBE = "_ExecutionProbe"
FIXED = f"mechanisms.{PROBE}.fixed"
GAIN = f"integrator.mech.mechanisms.{PROBE}.gain_param"
pytestmark = pytest.mark.cpu


class _ExecutionState(State):
    State.STATE("inferred", "fixed")
    State.DERIVATIVE("inferred' = 0.0 * inferred", "fixed' = 0.0 * fixed")

    def state_defaults(self, v, values):
        del values
        return {"inferred": 0.01 * v + 1.0}


class _ExecutionProbe(Mechanism):
    Mechanism.STATE_BUNDLE(_ExecutionState)
    Mechanism.GLOBAL(gain=0.2, rate=3.0, g=0.001, e=-61.0)
    Mechanism.CARRY("seed")
    Mechanism.TIMESTEP_BUFFER("coefficient")
    Mechanism.USEION("na", write=["ina"])
    Mechanism.AFFINE("ina")

    def derive_timestep_buffers(self, dt):
        return {"coefficient": self.gain + self.rate * dt}

    def initial_values(self, v, values):
        del v
        inferred = values["inferred"] + self.gain + self.coefficient
        return {"inferred": inferred, "seed": inferred + values["fixed"]}

    def ina_affine_terms(self, v):
        del v
        conductance = self.g * (
            1.0 + self.inferred + 0.1 * self.fixed + self.coefficient
        )
        return conductance, conductance * self.e


class _PreFrame(torch.nn.Module):
    def forward(self, voltage, gain, shift):
        return (voltage + gain + shift,)


class _PostFrame(torch.nn.Module):
    def forward(self, voltage, inferred, fixed, time, remainder, shift):
        return (
            voltage + shift * inferred,
            fixed + shift * inferred,
            time + shift,
            remainder + 2.0 * shift,
        )


class _ReadPostClock(torch.nn.Module):
    def forward(self, voltage, time, remainder):
        return (voltage + time + remainder,)


class _ZeroContextEffects(_ExecutionProbe):
    calls = 0

    Mechanism.DERIVED_BUFFER("bias")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active = False
        self.fail = False

    def derive_buffers(self):
        if self.active and self.dt == 0:
            _ZeroContextEffects.calls += 1
            random.random()
            np.random.random()
            if self.fail:
                torch.rand(())
                raise RuntimeError("zero-context execution failed")
        return {"bias": self.gain * torch.ones_like(self.diam)}


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


def _model(*, mechanism=_ExecutionProbe):
    probe = mechanism.__name__
    fixed = f"mechanisms.{probe}.fixed"
    gain = f"integrator.mech.mechanisms.{probe}.gain_param"
    model = dn.SingleCompartment(
        N=1,
        C=3,
        v_init=torch.tensor([-68.0, -64.0, -60.0]),
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(mechanism, ic={"fixed": 0.4})
    model.register_pre_initialize_transform(
        "pre_frame",
        _PreFrame(),
        reads=("state.integrator.v", f"parameters.{gain}"),
        writes=("state.integrator.v",),
        inputs={"shift": model.v.new_tensor(0.3)},
    )
    model.register_post_initialize_transform(
        "post_frame",
        _PostFrame(),
        reads=(
            "state.integrator.v",
            f"state.mechanisms.{probe}.inferred",
            f"state.{fixed}",
            "state.clock.t",
            "state.control.duration_remainder",
        ),
        writes=(
            "state.integrator.v",
            f"state.{fixed}",
            "state.clock.t",
            "state.control.duration_remainder",
        ),
        inputs={"shift": model.v.new_tensor(0.15)},
    )
    model.register_post_initialize_transform(
        "read_clock",
        _ReadPostClock(),
        reads=(
            "state.integrator.v",
            "state.clock.t",
            "state.control.duration_remainder",
        ),
        writes=("state.integrator.v",),
    )
    model.initialize()
    model.train()
    model.integrator._initialize(
        model, model.v.new_tensor(0.125), force=True, compile_scope="population"
    )
    model.t.fill_(1.25)
    model._duration_remainder.fill_(0.004)
    return model


def _arguments(tensors):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    parameters[GAIN] = (parameters[GAIN] + 0.07).requires_grad_()
    inputs = tensors.initialization._replace(
        v_init=(tensors.initialization.v_init - 0.4).detach().requires_grad_(),
        states={
            name: (value.detach().clone() + 0.2).requires_grad_()
            for name, value in tensors.initialization.states.items()
        },
        transforms={
            name: (value.detach().clone() + 0.05).requires_grad_()
            for name, value in tensors.initialization.transforms.items()
        },
    )
    return parameters, inputs


def _snapshot(tree):
    return [
        _tensor_snapshot(value)
        for value in torch.utils._pytree.tree_leaves(tree)
        if torch.is_tensor(value)
    ]


def _assert_snapshot(tree, snapshots):
    values = (
        value
        for value in torch.utils._pytree.tree_leaves(tree)
        if torch.is_tensor(value)
    )
    for value, snapshot in zip(values, snapshots, strict=True):
        _assert_tensor_snapshot(value, snapshot)


def _native(functional, parameters, inputs):
    model = _model()
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(parameters[name])
        model.set_v_init(inputs.v_init)
        model.mech.mechanisms[PROBE]._init_params["fixed"] = (
            inputs.states[FIXED].detach().clone()
        )
        for key, value in inputs.transforms.items():
            _, action, argument = key.split(".")
            registered = model._initialization_transforms[action]
            target = dict(
                zip(registered.input_names, registered.input_values(), strict=True)
            )[argument]
            target.copy_(value)
        model.initialize()
    return _live_state(functional, model)


def _loss(state):
    return (
        0.001 * state["integrator"]["v"].square().sum()
        + state["mechanisms"][PROBE]["inferred"].square().sum()
        + 0.1 * state["mechanism_buffers"][PROBE]["seed"].square().sum()
        + state["ions"]["na"]["ina"].square().sum()
    )


@pytest.mark.parametrize("check_purity", (False, True))
def test_initialization_orders_frames_and_owns_outputs(check_purity):
    model = _model()
    source = _module_snapshot(model)
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, inputs = _arguments(tensors)
    supplied = _snapshot((parameters, tensors.constants, inputs))
    # Execution retains the policy selected at lowering, even in another context.
    with dn.ctx(CHECK_PURITY=not check_purity):
        initialized = functional.initialize(parameters, tensors.constants, inputs)

    _assert_tree_close(initialized.state, _native(functional, parameters, inputs))
    pre = inputs.transforms["pre.pre_frame.shift"]
    post = inputs.transforms["post.post_frame.shift"]
    voltage = inputs.v_init + parameters[GAIN] + pre
    inferred = 0.01 * voltage + 1.0 + parameters[GAIN]
    fixed = inputs.states[FIXED]
    state = initialized.state
    torch.testing.assert_close(
        state["mechanisms"][PROBE]["fixed"], fixed + post * inferred
    )
    torch.testing.assert_close(state["mechanisms"][PROBE]["inferred"], inferred)
    torch.testing.assert_close(
        state["mechanism_buffers"][PROBE]["seed"], inferred + fixed
    )
    torch.testing.assert_close(
        state["integrator"]["v"], voltage + post * inferred + 3.0 * post
    )
    # Accepted ionic current is evaluated before post transforms with the fresh,
    # zero timestep workspace, although lowering followed nonzero preparation.
    g = parameters[f"integrator.mech.mechanisms.{PROBE}.g_param"]
    e = parameters[f"integrator.mech.mechanisms.{PROBE}.e_param"]
    torch.testing.assert_close(
        state["ions"]["na"]["ina"], g * (1 + inferred + 0.1 * fixed) * (voltage - e)
    )
    assert state["clock"]["t"].item() == 0
    assert state["control"]["duration_remainder"].item() == 0
    assert initialized.initialization.v_init is inputs.v_init
    assert initialized.initialization.states is inputs.states
    assert initialized.initialization.transforms is inputs.transforms
    _assert_snapshot((parameters, tensors.constants, inputs), supplied)
    _assert_module_snapshot(model, source)
    with torch.no_grad():
        for value in torch.utils._pytree.tree_leaves(state):
            value.add_(1.0)
    _assert_snapshot((parameters, tensors.constants, inputs), supplied)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("check_purity", (False, True))
def test_initialization_parameter_and_input_vjps_match_native(check_purity):
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(_model(), dt=DT)
    parameters, inputs = _arguments(tensors)
    initialized = functional.initialize(parameters, tensors.constants, inputs)
    targets = (
        parameters[GAIN],
        inputs.v_init,
        inputs.states[FIXED],
        *inputs.transforms.values(),
    )
    gradients = torch.autograd.grad(_loss(initialized.state), targets)
    epsilon = 1.0e-5
    for target, gradient in zip(targets, gradients, strict=True):
        direction = torch.linspace(
            0.7, 1.2, target.numel(), dtype=target.dtype
        ).reshape_as(target)

        def native_loss(sign):
            local_parameters = dict(parameters)
            local_states = dict(inputs.states)
            local_transforms = dict(inputs.transforms)
            replacement = target.detach() + sign * epsilon * direction
            if target is parameters[GAIN]:
                local_parameters[GAIN] = replacement
            if target is inputs.states[FIXED]:
                local_states[FIXED] = replacement
            for name, value in inputs.transforms.items():
                if target is value:
                    local_transforms[name] = replacement
            local_inputs = inputs._replace(
                v_init=replacement if target is inputs.v_init else inputs.v_init,
                states=local_states,
                transforms=local_transforms,
            )
            return _loss(_native(functional, local_parameters, local_inputs))

        expected = (native_loss(1.0) - native_loss(-1.0)) / (2.0 * epsilon)
        actual = (gradient * direction).sum()
        assert torch.isfinite(actual) and actual != 0
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-9)


@pytest.mark.parametrize("check_purity", (False, True))
@pytest.mark.parametrize("fail", (False, True), ids=("mutation", "exception"))
def test_initialization_audit_restores_class_and_rng_after_zero_context(
    fail, check_purity
):
    model = _model(mechanism=_ZeroContextEffects)
    owner = model.mech.mechanisms["_ZeroContextEffects"]
    owner.active = True
    owner.fail = fail
    source = _module_snapshot(model)
    calls = _ZeroContextEffects.calls
    torch_rng = torch.random.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(model, dt=DT)
    try:
        if check_purity:
            assert _ZeroContextEffects.calls == calls
            assert torch.equal(torch.random.get_rng_state(), torch_rng)
            assert random.getstate() == python_rng
            restored_numpy = np.random.get_state()
            assert restored_numpy[0] == numpy_rng[0]
            np.testing.assert_array_equal(restored_numpy[1], numpy_rng[1])
            assert restored_numpy[2:] == numpy_rng[2:]
        supplied = _snapshot(tensors)
        with dn.ctx(CHECK_PURITY=not check_purity):
            if check_purity:
                message = (
                    "could not be evaluated: zero-context execution failed"
                    if fail
                    else "class state"
                )
                with pytest.raises(dn.func.FunctionalizationError, match=message):
                    functional.initialize(
                        tensors.parameters, tensors.constants, tensors.initialization
                    )
            elif fail:
                with pytest.raises(RuntimeError, match="zero-context execution failed"):
                    functional.initialize(
                        tensors.parameters, tensors.constants, tensors.initialization
                    )
            else:
                initialized = functional.initialize(
                    tensors.parameters, tensors.constants, tensors.initialization
                )
                assert torch.isfinite(initialized.state["integrator"]["v"]).all()
            if not check_purity:
                assert random.getstate() != python_rng
                if fail:
                    assert not torch.equal(torch.random.get_rng_state(), torch_rng)
        _assert_snapshot(tensors, supplied)
        _assert_module_snapshot(model, source)
    finally:
        _ZeroContextEffects.calls = calls
        torch.random.set_rng_state(torch_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
