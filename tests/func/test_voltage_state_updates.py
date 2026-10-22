"""Nested State resets precede scnv advancement and remain explicit under dn.func."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import State, VoltageProcess
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_parity import _live_state


@pytest.fixture(autouse=True)
def _check_purity_diagnostics():
    """Exercise purity contracts explicitly; production defaults trust hooks."""
    with dn.ctx(CHECK_PURITY=True):
        yield


DT = 0.05
STEPS = 4
PATHS = ("v", "reset.owned", "reset.recovery", "reset.events", "t")
pytestmark = pytest.mark.cpu


class _ResettableState(State):
    State.STATE("owned", "recovery")
    State.ASSIGNED("external_drive")
    State.DERIVATIVE(
        "owned' = drive + external_drive - recovery", "recovery' = adaptation"
    )
    State.GLOBAL(drive=4.0, adaptation=0.5)
    synaptic_current = None

    def assigned_values(self, v, values):
        del values
        current = self.synaptic_current
        return {"external_drive": torch.zeros_like(v) if current is None else current}

    def state_defaults(self, v, values):
        del values
        return {"owned": v + 4, "recovery": torch.zeros_like(v) + 0.5}


class _StateVoltage(VoltageProcess):
    VoltageProcess.STATE_BUNDLE(_ResettableState)
    VoltageProcess.CARRY("events")
    VoltageProcess.GLOBAL(threshold=0.0, rest=-2.0, kick=1.25)

    def voltage_values(self, v, dt, values):
        del v, dt
        reset = values["owned"] >= self.threshold
        exposed = torch.where(
            reset, self.rest.expand_as(values["owned"]), values["owned"]
        )
        return exposed, {
            "owned": exposed,
            "recovery": values["recovery"] + reset.to(exposed.dtype) * self.kick,
            "events": values["events"] + reset.to(exposed.dtype),
        }


class _ImpureStateVoltage(_StateVoltage):
    def voltage_values(self, v, dt, values):
        self.owned = values["owned"] + 1
        return v, {"owned": self.owned}


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _model(mechanism=_StateVoltage):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64):
        model = dn.Population(
            N=1, C=3, dtype=torch.float64, integrator=dn.scnv(imem=False)
        )
        model.insert(mechanism.rename("reset"))
        model.batch(2)
        model.set_v_init(
            model.v.new_tensor([[[-5.0, -4.0, -3.0]], [[-4.5, -3.5, -2.5]]])
        )
        model.train()
        model.initialize()
    return model


def _observe(state):
    return {
        "v": state["integrator"]["v"],
        "reset.owned": state["mechanisms"]["reset"]["owned"],
        "reset.recovery": state["mechanisms"]["reset"]["recovery"],
        "reset.events": state["mechanism_buffers"]["reset"]["events"],
        "t": state["clock"]["t"],
    }


def _parameter(parameters, suffix):
    matches = [value for name, value in parameters.items() if name.endswith(suffix)]
    assert len(matches) == 1, suffix
    return matches[0]


def _reference(parameters, constants, state, steps=STEPS):
    """Independent original reset, then simultaneous linear State recurrence."""
    frame = _observe(state)
    history = [frame]
    threshold = _parameter(parameters, ".reset.threshold_param")
    rest = _parameter(parameters, ".reset.rest_param")
    kick = _parameter(parameters, ".reset.kick_param")
    drive = _parameter(parameters, ".drive_param")
    adaptation = _parameter(parameters, ".adaptation_param")
    dt = constants["dt"]
    for _ in range(steps):
        reset = frame["reset.owned"] >= threshold
        exposed = torch.where(
            reset, torch.zeros_like(frame["reset.owned"]) + rest, frame["reset.owned"]
        )
        recovery = frame["reset.recovery"] + reset.to(exposed.dtype) * kick
        frame = {
            "v": exposed,
            "reset.owned": exposed + dt * (drive - recovery),
            "reset.recovery": recovery + dt * adaptation,
            "reset.events": frame["reset.events"] + reset.to(exposed.dtype),
            "t": frame["t"] + dt,
        }
        history.append(frame)
    return {name: torch.stack([frame[name] for frame in history]) for name in PATHS}


def _runner(model, *, execution="scan", steps=STEPS):
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        initialization="snapshot",
        backend="eager",
        execution=execution,
        inference=execution == "while",
        callbacks={"trace": dn.func.Recorder(PATHS)},
    )


def _loss(trace):
    return sum(trace[name].square().mean() for name in PATHS[:-1])


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_nested_state_reset_matches_independent_and_imperative_traces(execution):
    model = _model()
    source = _module_snapshot(model)
    runner = _runner(model, execution=execution)
    parameters = _clone(dict(runner.parameters))
    targets = tuple(
        _parameter(parameters, suffix).requires_grad_()
        for suffix in (
            ".reset.rest_param",
            ".reset.kick_param",
            ".drive_param",
            ".adaptation_param",
        )
    )
    final, auxiliary = runner(parameters=parameters)
    trace = dict(auxiliary["callbacks"]["trace"])
    expected = _reference(parameters, runner.constants, runner.initial_state)
    _assert_tree_close(trace, expected, rtol=2e-10, atol=2e-10)
    if execution == "scan":
        actual_gradients = torch.autograd.grad(_loss(trace), targets)
        expected_gradients = torch.autograd.grad(_loss(expected), targets)
        for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, reference)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
    else:
        assert not trace["v"].requires_grad
    _assert_module_snapshot(model, source)

    reference = _model()
    dt = reference.v.new_tensor(DT)
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    frames = [_observe(_live_state(runner.functional, reference))]
    for _ in range(STEPS):
        reference.integrator.step(reference, dt)
        reference.t = reference.t + dt
        frames.append(_observe(_live_state(runner.functional, reference)))
    imperative = {
        name: torch.stack([frame[name] for frame in frames]) for name in PATHS
    }
    _assert_tree_close(trace, imperative, rtol=2e-10, atol=2e-10)
    assert not torch.equal(
        final["integrator"]["v"], final["mechanisms"]["reset"]["owned"]
    )


def test_nested_state_reset_explicit_continuation_and_commit_resume():
    model = _model()
    runner = _runner(model, steps=2)
    source = _module_snapshot(model)
    state, _ = runner()
    state = _clone(state)
    state["integrator"]["v"] = state["integrator"]["v"] + 17
    state["mechanisms"]["reset"]["owned"] = state["mechanisms"]["reset"]["owned"] + 2
    final, auxiliary = runner(state)
    expected = _reference(runner.parameters, runner.constants, state, steps=2)
    _assert_tree_close(
        dict(auxiliary["callbacks"]["trace"]), expected, rtol=2e-10, atol=2e-10
    )
    _assert_module_snapshot(model, source)

    runner.functional.commit_state_(model, final)
    _assert_tree_close(_live_state(runner.functional, model), final)
    expected_next, _ = runner.functional.prepare_and_step(
        runner.parameters, runner.constants, final
    )
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    model.integrator.step(model, dt)
    model.t = model.t + dt
    _assert_tree_close(
        _live_state(runner.functional, model), expected_next, rtol=2e-10, atol=2e-10
    )


def test_nested_state_output_does_not_authorize_mutating_the_hook_receiver():
    model = _model(_ImpureStateVoltage)
    source = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="mutat|impure|pure"):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
@pytest.mark.parametrize("binding", ("reference", "tensor"))
def test_nested_state_external_inputs_fail_before_evaluation(initialization, binding):
    model = _model()
    calls = []

    def getter():
        calls.append("evaluated")
        return torch.ones_like(model.v)

    state = model.mech.reset.DE["_ResettableState"]
    if binding == "reference":
        state.setreference("synaptic_current", getter)
    else:
        state.synaptic_current = torch.ones_like(model.v, requires_grad=True)
    source = _module_snapshot(model)
    with pytest.raises(
        dn.func.FunctionalizationError, match="live setreference.*synaptic_current"
    ):
        dn.compile(
            model, dt=DT, steps=1, initialization=initialization, backend="eager"
        )
    assert not calls
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("binding", ("reference", "tensor"))
def test_new_nested_state_external_input_invalidates_lowered_step_and_runner(binding):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    runner = _runner(model, steps=1)
    calls = []

    def getter():
        calls.append("evaluated")
        return torch.ones_like(model.v)

    state = model.mech.reset.DE["_ResettableState"]
    if binding == "reference":
        state.setreference("synaptic_current", getter)
    else:
        state.synaptic_current = torch.ones_like(model.v, requires_grad=True)
    source = _module_snapshot(model)
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Ss]tructur|changed|[Ss]chema"
    ):
        functional.prepare(tensors.parameters, tensors.constants)
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Ss]tructur|changed|[Ss]chema"
    ):
        functional.step(tensors.parameters, prepared, tensors.state)
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Ss]tructur|changed|[Ss]chema"
    ):
        runner()
    assert not calls
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("binding", ("reference", "tensor"))
def test_nested_state_external_input_remains_usable_imperatively(binding):
    model = _model()
    state = model.mech.reset.DE["_ResettableState"]
    current = torch.ones_like(model.v) * 7
    if binding == "reference":
        state.setreference("synaptic_current", lambda: current)
    else:
        state.synaptic_current = current
    model.mech.reset._advance_states(model.v, model.v.new_tensor(DT))
    torch.testing.assert_close(
        model.mech.reset.owned, model.v + 4 + DT * (4 + current - 0.5)
    )


def test_instance_none_shadows_a_nested_state_reference_without_evaluating_it():
    model = _model()
    calls = []

    def getter():
        calls.append("evaluated")
        return torch.ones_like(model.v)

    state = model.mech.reset.DE["_ResettableState"]
    state.setreference("synaptic_current", getter)
    state.synaptic_current = None
    runner = _runner(model, steps=1)
    _, auxiliary = runner()
    expected = _reference(
        runner.parameters, runner.constants, runner.initial_state, steps=1
    )
    _assert_tree_close(dict(auxiliary["callbacks"]["trace"]), expected)
    assert not calls


@pytest.mark.parametrize("name", ("cao", "cai", "nai"))
def test_shared_field_spelling_does_not_admit_an_undeclared_state_tensor(name):
    model = _model()
    state = model.mech.reset.DE["_ResettableState"]
    setattr(state, name, torch.ones_like(model.v, requires_grad=True))
    source = _module_snapshot(model)
    with pytest.raises(
        dn.func.FunctionalizationError, match=f"unregistered Tensor.*{name}"
    ):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, source)
