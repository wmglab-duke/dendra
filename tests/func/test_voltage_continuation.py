"""Changed fresh inputs and explicit voltage-process carry retain their graphs."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_voltage_processes import (
    _arguments,
    _clone,
    _close,
    _gradients,
    _imperative_reference,
    _key,
    _model,
    _runner,
)
from tests.test_voltage_process_pure import STEPS, _reference_fire

pytestmark = pytest.mark.cpu
VARIANT = "fire_r_d"


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _changed_arguments(runner, model):
    parameters, initialization, inputs, targets = _arguments(runner, model)
    for name, change in (
        ("threshold", 0.15),
        ("rest", -0.2),
        ("refractory", 0.03),
        ("tau_gate", 0.1),
    ):
        key = _key(parameters, f".{VARIANT}.{name}_param")
        parameters[key] = (parameters[key].detach() + change).requires_grad_()
        targets[key] = parameters[key]
    voltage = (initialization.v_init.detach() + 0.12).requires_grad_()
    initialization = initialization._replace(v_init=voltage)
    targets["initial_voltage"] = voltage
    return parameters, initialization, inputs, targets


def _slice_inputs(inputs, start, stop):
    return dn.func.RolloutInput(
        None if inputs.ve is None else inputs.ve[start:stop],
        None if inputs.intra is None else inputs.intra[start:stop],
    )


def _exercise_changed_fresh_continuation(kind, *, device="cpu", dtype=torch.float64):
    model = _model(VARIANT, kind=kind, device=device, dtype=dtype)
    whole = _runner(model, steps=STEPS)
    chunk = _runner(model, steps=STEPS // 2)
    source = _module_snapshot(model)
    parameters, initialization, inputs, targets = _changed_arguments(whole, model)
    state, auxiliary = whole(
        parameters=parameters, initialization=initialization, inputs=inputs
    )
    recorded = dict(auxiliary["callbacks"]["trace"])
    torch.testing.assert_close(recorded["v"][0], initialization.v_init)
    assert not torch.equal(recorded["v"][0], whole.initialization_inputs.v_init)
    for name, value in recorded.items():
        if name.startswith(f"{VARIANT}."):
            assert torch.count_nonzero(value[0]) == 0, name
    actual_gradients = _gradients(recorded, targets)

    parameters, initialization, inputs, targets = _changed_arguments(whole, model)
    first_state, first_auxiliary = chunk(
        parameters=parameters,
        initialization=initialization,
        inputs=_slice_inputs(inputs, 0, STEPS // 2),
    )
    second_state, second_auxiliary = chunk(
        first_state,
        parameters=first_auxiliary["parameters"],
        inputs=_slice_inputs(inputs, STEPS // 2, STEPS),
        callback_state=first_auxiliary["callbacks"].state,
    )
    joined = {
        name: torch.cat((first_auxiliary["callbacks"]["trace"][name], value))
        for name, value in second_auxiliary["callbacks"]["trace"].items()
    }
    _close(joined, recorded)
    _assert_tree_close(second_state, state, rtol=2e-8, atol=2e-9)
    _close(_gradients(joined, targets), actual_gradients)

    parameters, initialization, inputs, _targets = _changed_arguments(whole, model)
    imperative, imperative_gradients = _imperative_reference(
        model, whole, parameters, initialization, inputs
    )
    _close(recorded, imperative)
    _close(actual_gradients, imperative_gradients)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("kind", ("point", "multi_block"))
def test_changed_fresh_inputs_keep_full_and_split_voltage_gradient_graphs(
    kind, cpu_threads
):
    _exercise_changed_fresh_continuation(kind)


def _point_continuation(model, parameters, state, inputs, dt):
    """Independent original reset equations plus passive backward Euler."""
    voltage = state["integrator"]["v"]
    carry = state["mechanism_buffers"][VARIANT]
    process_parameters = {
        name: parameters[_key(parameters, f".{VARIANT}.{name}_param")]
        for name in ("threshold", "rest", "refractory", "tau_gate", "ste_scale")
    }
    conductance = parameters[_key(parameters, ".pas.g_param")]
    reversal = parameters[_key(parameters, ".pas.e_param")]
    cmdt = 1e-3 * model.cm.detach() * model.cm_scale.detach() / dt
    area = model.area.detach() * model.area_scale.detach()
    frames = [{"v": voltage, **{f"{VARIANT}.{k}": v for k, v in carry.items()}}]
    for drive in inputs.intra:
        reset, carry = _reference_fire(VARIANT, voltage, dt, process_parameters, carry)
        voltage = reset + (drive / area - conductance * (reset - reversal)) / (
            cmdt + conductance
        )
        frames.append({"v": voltage, **{f"{VARIANT}.{k}": v for k, v in carry.items()}})
    return {name: torch.stack([frame[name] for frame in frames]) for name in frames[0]}


def _override_carry(state, targets):
    state = _clone(state)
    voltage = state["integrator"]["v"].requires_grad_()
    targets["initial_voltage"] = voltage
    carry = state["mechanism_buffers"][VARIANT]
    indices = torch.arange(voltage.numel(), device=voltage.device).reshape(
        voltage.shape
    )
    carry["is_refractory"] = indices.remainder(3) != 0
    timer = torch.where(
        indices.remainder(2) == 0, voltage.new_tensor(0.02), voltage.new_tensor(0.12)
    )
    carry["time_refractory"] = timer.requires_grad_()
    for name in ("spike_gate", "reset_gate"):
        carry[name] = torch.full_like(voltage, 0.3).requires_grad_()
    targets.update(
        (f"initial_{name}", value)
        for name, value in carry.items()
        if value.is_floating_point()
    )
    return state


def test_explicit_bool_timer_and_gate_continuation_matches_independent_recurrence(
    cpu_threads,
):
    model = _model(VARIANT)
    runner = _runner(model, steps=STEPS // 2)
    source = _module_snapshot(model)
    previous, _ = runner()
    parameters, _initialization, inputs, targets = _changed_arguments(runner, model)
    inputs = _slice_inputs(inputs, STEPS // 2, STEPS)
    state = _override_carry(previous, targets)
    boundary = pytree.tree_map(torch.clone, state["mechanism_buffers"][VARIANT])
    _, auxiliary = runner(state, parameters=parameters, inputs=inputs)
    recorded = dict(auxiliary["callbacks"]["trace"])
    for name, value in boundary.items():
        torch.testing.assert_close(recorded[f"{VARIANT}.{name}"][0], value)
    actual_gradients = _gradients(recorded, targets)

    parameters, _initialization, inputs, targets = _changed_arguments(runner, model)
    inputs = _slice_inputs(inputs, STEPS // 2, STEPS)
    reference_state = _override_carry(previous, targets)
    independent = _point_continuation(
        model, parameters, reference_state, inputs, runner.constants["dt"]
    )
    _close({name: recorded[name] for name in independent}, independent)
    _close(actual_gradients, _gradients(independent, targets))
    for name in ("time_refractory", "spike_gate", "reset_gate"):
        assert torch.count_nonzero(actual_gradients[f"initial_{name}"])
    _assert_module_snapshot(model, source)
