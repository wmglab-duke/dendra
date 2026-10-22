"""Whole-model TABLE paths preserve transforms and compiled execution."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_table_execution import (
    DT,
    LEARNABLE,
    MECHANISMS,
    _network,
    _owner,
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


def _trajectory(mechanism):
    functional, tensors = dn.func.make_functional(_population(mechanism), dt=DT)
    name = tensors.parameter_name("*curve_table.y")

    def evaluate(ordinates, voltage, dt):
        parameters = {**tensors.parameters, name: ordinates}
        constants = {**tensors.constants, "dt": dt}
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": voltage},
        }
        final, _ = functional.prepare_and_rollout(parameters, constants, state, steps=3)
        field = (
            final["mechanisms"]["tabulated"]["x"]
            if "tabulated" in final.get("mechanisms", {})
            else final["mechanism_buffers"]["tabulated"]["gain"]
        )
        return torch.cat((final["integrator"]["v"].flatten(), field.flatten()))

    return evaluate, (
        tensors.parameters[name].detach(),
        tensors.state["integrator"]["v"],
        tensors.constants["dt"],
    )


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", LEARNABLE)
def test_population_table_jacobians_vmap_and_empty_vmap(mechanism):
    evaluate, operands = _trajectory(mechanism)
    reverse = torch.func.jacrev(evaluate, argnums=(0, 1, 2))(*operands)
    forward = torch.func.jacfwd(evaluate, argnums=(0, 1, 2))(*operands)
    torch.testing.assert_close(reverse, forward, rtol=2e-9, atol=2e-10)
    assert all(value.abs().sum() > 0 for value in reverse)
    batched = tuple(torch.stack((value, value + 0.01)) for value in operands)
    expected = torch.stack(
        [evaluate(*(value[index] for value in batched)) for index in range(2)]
    )
    actual = torch.vmap(evaluate)(*batched)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    empty = torch.vmap(evaluate)(*(value[:0] for value in batched))
    assert empty.shape == (0, *expected.shape[1:])


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_native_jit_table_trajectory_matches_python_execution(mechanism):
    reference = _population(mechanism)
    with torch.no_grad():
        for _ in range(3):
            reference.step(DT)
    with dn.ctx(JIT=1):
        native = _population(mechanism)
        with torch.no_grad():
            for _ in range(3):
                native.step(DT)
    functional, tensors = dn.func.make_functional(reference, dt=DT)
    torch.testing.assert_close(
        functional.extract(native).state, tensors.state, rtol=2e-10, atol=2e-11
    )


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", LEARNABLE)
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_inductor_connected_table_network_recording_and_continuation(
    mechanism, execution
):
    source, native = _network(mechanism), _network(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        execution=execution,
        inference=True,
        backend="inductor",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
    )
    state, prefix = runner()
    final, suffix = runner(state, callback_state=prefix["callbacks"].state)
    expected = {
        "left.v": [native.left.v.clone()],
        "right.v": [native.right.v.clone()],
        "t": [native.t.clone()],
    }
    with torch.no_grad():
        for _ in range(8):
            native.step()
            for name in expected:
                value = (
                    native.t
                    if name == "t"
                    else native.populations[name.split(".")[0]].v
                )
                expected[name].append(value.clone())
    torch.testing.assert_close(
        final, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )
    for name, values in expected.items():
        actual = torch.cat(
            (prefix["callbacks"]["trace"][name], suffix["callbacks"]["trace"][name])
        )
        torch.testing.assert_close(actual, torch.stack(values), rtol=2e-10, atol=2e-11)
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", LEARNABLE)
def test_inductor_checkpointed_network_preserves_shared_table_gradients(mechanism):
    source, native = _network(mechanism, shared=True), _network(mechanism, shared=True)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=False,
        trainable=("*right*curve_table.y",),
        checkpoint_every=2,
        backend="inductor",
    )
    selected = runner.get_parameter("*left*curve_table.y")
    assert selected is runner.get_parameter("*right*curve_table.y")
    final, _ = runner()
    for _ in range(4):
        native.step()
    actual = torch.autograd.grad(
        sum(
            state["integrator"]["v"].square().sum()
            for state in final["populations"].values()
        ),
        selected,
    )[0]
    expected = torch.autograd.grad(
        sum(population.v.square().sum() for population in native.populations.values()),
        _owner(native.left).curve_table.y,
    )[0]
    assert expected.abs().sum() > 0
    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mechanism", LEARNABLE)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_table_continuation_matches_native(mechanism, dtype):
    with dn.ctx(DEVICE="cuda", DTYPE=dtype):
        source, native = _population(mechanism), _population(mechanism)
    runner = dn.compile(source, dt=DT, steps=3, backend="aot_eager")
    prefix, _ = runner()
    final, _ = runner(prefix)
    with torch.no_grad():
        for _ in range(6):
            native.step(DT)
    tolerance = 5e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(
        final, runner.functional.extract(native).state, rtol=tolerance, atol=tolerance
    )
