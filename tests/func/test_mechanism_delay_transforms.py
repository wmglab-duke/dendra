"""Whole-model delay histories preserve transforms and compiler semantics."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_mechanism_delay_execution import (
    DT,
    _CircularLeak,
    _clone,
    _close,
    _DelayedLeak,
    _MultiCircularLeak,
    _MultiDelayedLeak,
    _network,
    _population,
)


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", (_DelayedLeak, _MultiDelayedLeak))
def test_population_shift_delay_transforms(mechanism):
    functional, tensors = dn.func.make_functional(_population(mechanism), dt=DT)
    name = tensors.parameter_name("*gain_param")

    def evaluate(gain, voltage, history, dt):
        parameters = {**tensors.parameters, name: gain}
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": voltage},
            "mechanism_buffers": {
                "delays": {
                    **tensors.state["mechanism_buffers"]["delays"],
                    "signal_delay_buffer": history,
                }
            },
        }
        final, _ = functional.prepare_and_rollout(
            parameters, {**tensors.constants, "dt": dt}, state, steps=4
        )
        return torch.cat(
            (
                final["integrator"]["v"].flatten(),
                final["mechanism_buffers"]["delays"]["delayed"].flatten(),
            )
        )

    operands = (
        tensors.parameters[name].detach(),
        tensors.state["integrator"]["v"],
        tensors.state["mechanism_buffers"]["delays"]["signal_delay_buffer"] + 0.25,
        tensors.constants["dt"],
    )
    argnums = tuple(range(len(operands)))
    reverse = torch.func.jacrev(evaluate, argnums=argnums)(*operands)
    forward = torch.func.jacfwd(evaluate, argnums=argnums)(*operands)
    _close(reverse, forward)
    assert all(value.abs().sum() > 0 for value in reverse)
    tangents = tuple(torch.ones_like(value) for value in operands)
    _, actual_tangent = torch.func.jvp(evaluate, operands, tangents)
    expected_tangent = sum(
        value.reshape(value.shape[0], -1).sum(1) for value in reverse
    )
    _close(actual_tangent, expected_tangent)
    batched = tuple(torch.stack((value, value + 0.01)) for value in operands)
    expected = torch.stack(
        [evaluate(*(value[index] for value in batched)) for index in range(2)]
    )
    _close(torch.vmap(evaluate)(*batched), expected)
    empty = torch.vmap(evaluate)(*(value[:0] for value in batched))
    assert empty.shape == (0, *expected.shape[1:])


@pytest.mark.cpu
@pytest.mark.parametrize(
    "mechanism", (_DelayedLeak, _CircularLeak, _MultiDelayedLeak, _MultiCircularLeak)
)
def test_native_jit_delay_trajectory_matches_python(mechanism):
    reference = _population(mechanism)
    with torch.no_grad():
        for _ in range(4):
            reference.step(DT)
    with dn.ctx(JIT=1):
        native = _population(mechanism)
        with torch.no_grad():
            for _ in range(4):
                native.step(DT)
    functional, tensors = dn.func.make_functional(reference, dt=DT)
    _close(functional.extract(native).state, tensors.state)


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", (_CircularLeak, _MultiCircularLeak))
def test_inductor_connected_circular_delays_checkpointed_gradients(mechanism):
    source, native = _network(mechanism), _network(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=False,
        trainable=("*left*gain_param", "*right*gain_param"),
        checkpoint_every=2,
        backend="inductor",
    )
    state = _clone(runner.initial_state)
    history = state["populations"]["left"]["mechanism_buffers"]["delays"][
        "signal_delay_buffer"
    ].requires_grad_()
    with torch.no_grad():
        history.fill_(0.25)
    native_history = history.detach().clone().requires_grad_()
    native.left.mech.delays.signal_delay_buffer = native_history
    for population in native.populations.values():
        population.mech.delays.populate()
    final, _ = runner(state)
    for _ in range(4):
        native.step()
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        sum(
            value["integrator"]["v"].square().sum()
            for value in final["populations"].values()
        ),
        (
            runner.get_parameter("*left*gain_param"),
            runner.get_parameter("*right*gain_param"),
            history,
        ),
    )
    expected = torch.autograd.grad(
        native.left.v.square().sum() + native.right.v.square().sum(),
        (
            native.left.mech.delays.gain_param,
            native.right.mech.delays.gain_param,
            native_history,
        ),
    )
    assert all(value.abs().sum() > 0 for value in expected)
    _close(actual, expected)
    assert runner.execution_report().strategy == "compiled_scan"


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", (_CircularLeak, _MultiCircularLeak))
def test_inductor_connected_circular_delays_while_recording_and_continuation(mechanism):
    source, native = _network(mechanism), _network(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=True,
        execution="while",
        backend="inductor",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.delays.delayed", "t"))},
    )
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    expected = [native.right.mech.delays.delayed.clone()]
    with torch.no_grad():
        for _ in range(8):
            native.step()
            expected.append(native.right.mech.delays.delayed.clone())
    _close(final, runner.functional.extract(native).state)
    observed = torch.cat(
        (
            first["callbacks"]["trace"]["right.delays.delayed"],
            second["callbacks"]["trace"]["right.delays.delayed"],
        )
    )
    _close(observed, torch.stack(expected))
    assert runner.execution_report().strategy == "compiled_while_loop"
