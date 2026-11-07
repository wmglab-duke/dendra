"""CUDA acceptance for exact plain Population point-system execution.

Run on a CUDA host with::

    python -m pytest tests/func/test_cuda_plain_population.py -W error -m cuda

The ordinary lane checks float32/float64 full recordings, all eleven gradient
sources (the usual eight plus point-model physics), purity, and fresh
initialization. Three slow cases exercise real Inductor scan, while-loop, and
initializer compilation. These tests require no native CPU solver package.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_functional_initialization import (
    AM1,
    _assert_module_snapshot,
    _assert_tensor_snapshot,
    _module_snapshot,
    _tensor_snapshot,
)
from tests.func.test_plain_population import (
    DT,
    RECORD,
    STEPS,
    _close,
    _close_gradients,
    _drives,
    _functional_result,
    _imperative_result,
    _model,
    _recorded,
    _trajectory,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_plain_population_full_recording_and_gradients_match_imperative(dtype):
    source = _model(device="cuda", dtype=dtype, batch_calls=(2,))
    reference = _model(
        model_class=dn.SingleCompartment,
        device="cuda",
        dtype=dtype,
        batch_calls=(2,),
    )
    snapshot = _module_snapshot(source)
    actual, gradients = _functional_result(source)
    expected, reference_gradients = _imperative_result(reference)
    assert source.shape == (2, 2, 3)
    assert len(gradients) == 11
    for value in gradients.values():
        assert value.dtype == dtype and value.device.type == "cuda"
        assert torch.isfinite(value).all()
    _close(actual, expected, dtype=dtype)
    _close_gradients(gradients, reference_gradients, dtype=dtype)
    _assert_module_snapshot(source, snapshot)


def _initializer_case(dtype, *, model_class=dn.Population):
    model = _model(
        model_class=model_class, device="cuda", dtype=dtype, batch_calls=(2,)
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert tensors.initialization is not None

    def initialize(voltage, rate):
        parameters = {**tensors.parameters, AM1: rate}
        return functional.initialize(
            parameters,
            tensors.constants,
            tensors.initialization._replace(v_init=voltage),
        ).state

    return model, tensors, initialize


def _initialization_loss(state):
    return (
        state["mechanisms"]["hh"]["m"].square().mean()
        + 0.003 * state["integrator"]["v"].square().mean()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_plain_population_fresh_initialization_matches_single_compartment(dtype):
    source, tensors, initialize = _initializer_case(dtype)
    _, _, initialize_reference = _initializer_case(
        dtype, model_class=dn.SingleCompartment
    )
    voltage = (
        torch.linspace(-73.0, -51.0, source.v.numel(), device="cuda", dtype=dtype)
        .reshape(source.shape)
        .requires_grad_()
    )
    rate = tensors.parameters[AM1].detach().clone().requires_grad_()
    snapshots = (_tensor_snapshot(voltage), _tensor_snapshot(rate))
    source_snapshot = _module_snapshot(source)
    actual = initialize(voltage, rate)
    gradients = torch.autograd.grad(_initialization_loss(actual), (voltage, rate))
    expected = initialize_reference(voltage, rate)
    reference_gradients = torch.autograd.grad(
        _initialization_loss(expected), (voltage, rate)
    )
    _close(actual, expected, dtype=dtype)
    _close_gradients(
        dict(zip(("initial_voltage", "rate"), gradients, strict=True)),
        dict(zip(("initial_voltage", "rate"), reference_gradients, strict=True)),
        dtype=dtype,
    )
    assert torch.count_nonzero(actual["clock"]["t"]) == 0
    assert torch.count_nonzero(actual["control"]["duration_remainder"]) == 0
    for value, snapshot in zip((voltage, rate), snapshots, strict=True):
        _assert_tensor_snapshot(value, snapshot)
    _assert_module_snapshot(source, source_snapshot)


@pytest.mark.slow
def test_cuda_plain_population_compiled_scan_preserves_full_recording_and_gradients():
    source = _model(device="cuda", dtype=torch.float32, batch_calls=(2,))
    snapshot = _module_snapshot(source)
    expected, expected_gradients = _functional_result(source)
    with torch_compiler_warning_context():
        actual, gradients = _functional_result(
            source, compiled=True, options={"cpp_wrapper": False}
        )
    _close(actual, expected, dtype=source.dtype())
    _close_gradients(gradients, expected_gradients, dtype=source.dtype())
    _assert_module_snapshot(source, snapshot)


@pytest.mark.slow
def test_cuda_plain_population_compiled_while_preserves_full_recording():
    source = _model(device="cuda", dtype=torch.float32, batch_calls=(2,))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    bath, intra = _drives(source)
    with torch.no_grad():
        expected = _trajectory(
            functional,
            tensors.parameters,
            tensors.constants,
            tensors.state,
            bath,
            intra,
        )
    snapshot = _module_snapshot(source)
    runner = dn.func.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution="while",
        inference=True,
        callbacks={"trace": dn.func.Recorder(RECORD)},
        backend="inductor",
    )
    assert not runner.kernel._compile_options.get("options", {}).get(
        "cpp_wrapper", False
    )
    with torch_compiler_warning_context():
        final, auxiliary = runner(inputs=dn.func.RolloutInput(ve=bath, intra=intra))
    _close(final, expected[-1], dtype=source.dtype())
    _close(
        dict(auxiliary["callbacks"]["trace"]), _recorded(expected), dtype=source.dtype()
    )
    assert runner.execution_report().strategy == "compiled_while_loop"
    _assert_module_snapshot(source, snapshot)


@pytest.mark.slow
def test_cuda_plain_population_compiled_initializer_preserves_state_and_gradients():
    source, tensors, initialize = _initializer_case(torch.float32)
    voltage = (
        torch.linspace(
            -72.0, -52.0, source.v.numel(), device="cuda", dtype=torch.float32
        )
        .reshape(source.shape)
        .requires_grad_()
    )
    rate = tensors.parameters[AM1].detach().clone().requires_grad_()
    snapshot = _module_snapshot(source)
    expected = initialize(voltage, rate)
    expected_gradients = torch.autograd.grad(
        _initialization_loss(expected), (voltage, rate)
    )
    compiled = torch.compile(
        initialize, backend="inductor", fullgraph=True, dynamic=False
    )
    with torch_compiler_warning_context():
        actual = compiled(voltage, rate)
        gradients = torch.autograd.grad(_initialization_loss(actual), (voltage, rate))
    _close(actual, expected, dtype=source.dtype())
    _close_gradients(
        dict(zip(("initial_voltage", "rate"), gradients, strict=True)),
        dict(zip(("initial_voltage", "rate"), expected_gradients, strict=True)),
        dtype=source.dtype(),
    )
    _assert_module_snapshot(source, snapshot)
