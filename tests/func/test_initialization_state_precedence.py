"""Pre-initialization State writes are transient until declared IC restoration."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_execution import _assert_snapshot, _snapshot
from tests.func.test_initialization_parity import _live_state

DT = 0.025
PROBE = "_PrecedenceProbe"
FIXED = f"mechanisms.{PROBE}.fixed"
GAIN = f"integrator.mech.mechanisms.{PROBE}.gain_param"
READ = "pre.read_previous.scale"
RETAIN = "pre.retain.scale"
DISCARD = "pre.discard.value"
POST = "post.update_fixed.shift"
pytestmark = pytest.mark.cpu


class _PrecedenceState(State):
    State.STATE("fixed", "inferred")
    State.DERIVATIVE("fixed' = 0.1", "inferred' = 0.2")

    def state_defaults(self, v, values):
        del values
        return {"inferred": 0.1 + 0.002 * v}


class _PrecedenceProbe(Mechanism):
    Mechanism.STATE_BUNDLE(_PrecedenceState)
    Mechanism.GLOBAL(gain=0.2, g=0.0001, e=-61.0)
    Mechanism.CARRY("seed")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        del v
        return {"seed": values["inferred"] + values["fixed"] + self.gain}

    def i(self, v):
        return self.g * (v - self.e)


class _ReadPrevious(torch.nn.Module):
    def forward(self, voltage, previous, gain, scale):
        return voltage + 0.1 * previous, previous + gain * scale


class _RetainRead(torch.nn.Module):
    def forward(self, voltage, intermediate, gain, scale):
        return (
            voltage + scale * intermediate,
            gain + 0.01 * intermediate.mean(),
            intermediate + scale,
        )


class _Overwrite(torch.nn.Module):
    def forward(self, voltage, value):
        return (torch.zeros_like(voltage) + value,)


class _PostState(torch.nn.Module):
    def forward(self, fixed, shift):
        return (fixed + shift,)


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=0,
            REQUIRE_GRAD=1,
            IMEM=False,
            DEVICE="cpu",
            DTYPE=torch.float64,
            CHECK_PURITY=False,
        ),
    ):
        yield


def _advance(model):
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    model.integrator.step(model, dt, None, None)
    model.t = model.t + dt


def _model(*, inferred_pre=False):
    model = dn.SingleCompartment(
        N=1,
        C=3,
        v_init=torch.tensor([-68.0, -64.0, -60.0]),
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(_PrecedenceProbe, ic={"fixed": 0.4})
    # Establish actual prior State before registering actions which read it.
    model.initialize()
    voltage = "state.integrator.v"
    fixed = f"state.{FIXED}"
    gain = f"parameters.{GAIN}"
    model.register_pre_initialize_transform(
        "read_previous",
        _ReadPrevious(),
        reads=(voltage, fixed, gain),
        writes=(voltage, fixed),
        inputs={"scale": model.v.new_tensor(0.3)},
    )
    model.register_pre_initialize_transform(
        "retain",
        _RetainRead(),
        reads=(voltage, fixed, gain),
        writes=(voltage, gain, fixed),
        inputs={"scale": model.v.new_tensor(0.2)},
    )
    model.register_pre_initialize_transform(
        "discard",
        _Overwrite(),
        reads=(voltage,),
        writes=(fixed,),
        inputs={"value": model.v.new_tensor(9.0)},
    )
    if inferred_pre:
        model.register_pre_initialize_transform(
            "inferred",
            _Overwrite(),
            reads=(voltage,),
            writes=(f"state.mechanisms.{PROBE}.inferred",),
            inputs={"value": model.v.new_tensor(12.0)},
        )
    model.register_post_initialize_transform(
        "update_fixed",
        _PostState(),
        reads=(fixed,),
        writes=(fixed,),
        inputs={"shift": model.v.new_tensor(0.1)},
    )
    model.initialize()
    model.train()
    _advance(model)
    return model


def _case(check_purity, continuation):
    model = _model()
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(model, dt=DT)
    previous = tensors.state
    if continuation:
        previous, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, previous
        )
    values = (
        (tensors.parameters[GAIN].detach() + 0.04).requires_grad_(),
        (tensors.initialization.v_init.detach() - 0.3).requires_grad_(),
        (tensors.initialization.states[FIXED].detach() + 0.35).requires_grad_(),
        *(
            tensors.initialization.transforms[name].detach().clone().requires_grad_()
            for name in (READ, RETAIN, DISCARD, POST)
        ),
        previous["mechanisms"][PROBE]["fixed"].detach().clone().requires_grad_(),
    )
    return model, functional, tensors, previous, values


def _call(functional, tensors, previous, values, *, continuation=True):
    gain, voltage, fixed, read, retain, discard, post, prior_fixed = values
    parameters = {**tensors.parameters, GAIN: gain}
    inputs = tensors.initialization._replace(
        v_init=voltage,
        states={**tensors.initialization.states, FIXED: fixed},
        transforms={
            READ: read,
            RETAIN: retain,
            DISCARD: discard,
            POST: post,
        },
    )
    state = pytree.tree_map(lambda value: value, previous)
    state["mechanisms"][PROBE]["fixed"] = prior_fixed
    return functional.initialize(
        parameters,
        tensors.constants,
        inputs,
        state=state if continuation else None,
    )


def _native(functional, values, *, continuation):
    gain, voltage, fixed, read, retain, discard, post, prior_fixed = values
    model = _model()
    if continuation:
        _advance(model)
    with torch.no_grad():
        dict(model.named_parameters())[GAIN].copy_(gain)
        model.set_v_init(voltage)
        owner = model.mech.mechanisms[PROBE]
        owner._init_params["fixed"] = fixed.detach().clone()
        owner.fixed.copy_(prior_fixed)
        for key, value in (
            (READ, read),
            (RETAIN, retain),
            (DISCARD, discard),
            (POST, post),
        ):
            _, name, _ = key.split(".")
            model._initialization_transforms[name].input_values()[0].copy_(value)
        model.initialize()
    return dict(model.named_parameters())[GAIN].clone(), _live_state(functional, model)


def _outputs(initialized):
    state = initialized.state
    return (
        initialized.parameters[GAIN],
        state["integrator"]["v"],
        state["mechanisms"][PROBE]["fixed"],
        state["mechanisms"][PROBE]["inferred"],
        state["mechanism_buffers"][PROBE]["seed"],
    )


def _native_outputs(functional, values, *, continuation=True):
    gain, state = _native(functional, values, continuation=continuation)
    return (
        gain,
        state["integrator"]["v"],
        state["mechanisms"][PROBE]["fixed"],
        state["mechanisms"][PROBE]["inferred"],
        state["mechanism_buffers"][PROBE]["seed"],
    )


def _loss(values):
    return sum(
        (index + 1) * value.square().mean() for index, value in enumerate(values)
    )


@pytest.mark.parametrize("check_purity", (False, True))
@pytest.mark.parametrize("continuation", (False, True), ids=("captured", "continued"))
def test_native_ic_overrides_pre_writes_after_ordered_previous_state_reads(
    check_purity, continuation
):
    model, functional, tensors, previous, values = _case(check_purity, continuation)
    source = _module_snapshot(model)
    supplied = _snapshot((tensors, previous, values))
    with dn.ctx(CHECK_PURITY=not check_purity):
        initialized = _call(
            functional, tensors, previous, values, continuation=continuation
        )
    native_gain, native_state = _native(functional, values, continuation=continuation)
    _assert_tree_close(initialized.state, native_state, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(initialized.parameters[GAIN], native_gain)
    gain, voltage, fixed, read, retain, _, post, prior_fixed = values
    intermediate = prior_fixed + gain * read
    torch.testing.assert_close(
        initialized.state["integrator"]["v"],
        voltage + 0.1 * prior_fixed + retain * intermediate,
    )
    torch.testing.assert_close(
        initialized.state["mechanisms"][PROBE]["fixed"], fixed + post
    )
    assert not torch.equal(prior_fixed, fixed)
    _assert_snapshot((tensors, previous, values), supplied)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("check_purity", (False, True))
def test_transient_pre_writes_keep_only_retained_gradient_routes(check_purity):
    _, functional, tensors, previous, values = _case(check_purity, True)
    outputs = _outputs(_call(functional, tensors, previous, values))
    gradients = torch.autograd.grad(
        _loss(outputs), values, allow_unused=True, materialize_grads=True
    )
    epsilon = 1.0e-5
    for index, (value, gradient) in enumerate(zip(values, gradients, strict=True)):
        direction = torch.linspace(
            0.7, 1.2, value.numel(), dtype=value.dtype
        ).reshape_as(value)
        losses = []
        for sign in (-1, 1):
            changed = list(values)
            changed[index] = value.detach() + sign * epsilon * direction
            losses.append(_loss(_native_outputs(functional, changed)))
        expected = (losses[1] - losses[0]) / (2 * epsilon)
        actual = (gradient * direction).sum()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        if index == 5:  # The final pre write has no reader before IC restoration.
            torch.testing.assert_close(
                gradient, torch.zeros_like(value), rtol=0, atol=0
            )
        else:
            assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


def test_initialization_precedence_under_jvp_and_vmap():
    _, functional, tensors, previous, values = _case(False, True)

    def outputs(*values):
        return _outputs(_call(functional, tensors, previous, values))

    directions = tuple(torch.ones_like(value) * 0.3 for value in values)
    primal, tangent = torch.func.jvp(outputs, values, directions)
    _assert_tree_close(
        primal, _native_outputs(functional, values), rtol=2e-13, atol=2e-13
    )
    epsilon = 1e-5
    positive = _native_outputs(
        functional,
        tuple(v + epsilon * d for v, d in zip(values, directions, strict=True)),
    )
    negative = _native_outputs(
        functional,
        tuple(v - epsilon * d for v, d in zip(values, directions, strict=True)),
    )
    _assert_tree_close(
        tangent,
        tuple((p - n) / (2 * epsilon) for p, n in zip(positive, negative, strict=True)),
        rtol=2e-7,
        atol=2e-8,
    )
    lanes = tuple(torch.stack((value, value + 0.05)) for value in values)
    actual = torch.vmap(outputs)(*lanes)
    expected = tuple(
        torch.stack(parts)
        for parts in zip(
            *(
                _native_outputs(functional, tuple(value[lane] for value in lanes))
                for lane in range(2)
            ),
            strict=True,
        )
    )
    _assert_tree_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_initialization_precedence_compiles_fullgraph():
    _, functional, tensors, previous, values = _case(False, True)

    def outputs(*values):
        return _outputs(_call(functional, tensors, previous, values))

    compiled = torch.compile(outputs, backend="aot_eager", fullgraph=True)
    _assert_tree_close(
        compiled(*values), _native_outputs(functional, values), rtol=2e-13, atol=2e-13
    )


def test_inferred_pre_state_write_keeps_current_admission_boundary():
    model = _model(inferred_pre=True)
    model.initialize()
    # Native defaults reconstruct inferred State after the transient pre write.
    torch.testing.assert_close(
        model.mech.mechanisms[PROBE].inferred,
        0.1 + 0.002 * model.v,
        rtol=0,
        atol=0,
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError, match="unsupported write paths.*inferred"
    ):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
