"""Run the CUDA acceptance harness on CPU to validate its factories/oracles.

Passing this file is deliberately not evidence of CUDA numerical correctness.
The actual device suite remains separately CUDA-marked and hardware-gated.
"""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from tests.func.test_cuda_population import (
    DT,
    KINDS,
    STEPS,
    _close,
    _close_gradients,
    _drives,
    _functional_result,
    _functional_trajectory,
    _imperative_result,
    _model,
    _record_fields,
    _recorded_states,
)

pytestmark = pytest.mark.cpu
NATIVE_KINDS = {"ub_thomas", "tree", "ext_axon", "ext_tree", "multi"}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_acceptance_helpers_match_cpu_imperative_oracle(kind, dtype):
    if kind in NATIVE_KINDS and not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("CPU acceptance oracle requires dendra-solvers")
    source = _model(kind, device="cpu", dtype=dtype)
    reference = _model(kind, device="cpu", dtype=dtype)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=dtype):
        functional, _, states, gradients = _functional_result(kind, source)
        expected_states, expected_gradients = _imperative_result(
            kind, reference, functional
        )
    assert len(states) == STEPS + 1
    assert len(gradients) == 9
    _close(states, expected_states, dtype=dtype)
    _close_gradients(gradients, expected_gradients, dtype=dtype)


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_cuda_recording_helpers_match_public_compiled_runner_on_cpu(execution):
    model = _model("sc", device="cpu", dtype=torch.float32)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)
    with torch.no_grad():
        expected = _functional_trajectory(
            functional, tensors.parameters, tensors.constants, tensors.state, ve, intra
        )
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=True,
        callbacks={"trace": dn.func.Recorder(_record_fields(tensors.state))},
        backend="eager",
    )
    with torch_compiler_warning_context():
        actual, auxiliary = runner(inputs=dn.func.RolloutInput(ve=ve, intra=intra))
    _close(actual, expected[-1], dtype=model.dtype())
    _close(
        dict(auxiliary["callbacks"]["trace"]),
        _recorded_states(expected),
        dtype=model.dtype(),
    )
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
