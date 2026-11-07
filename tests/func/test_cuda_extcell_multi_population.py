"""CUDA acceptance tests for packed finite-extracellular MultiPopulations.

Run on the CUDA host with::

    python -m pytest tests/func/test_cuda_extcell_multi_population.py -W error -m cuda

The ordinary lane compares full recordings and all physical/source gradients
with independently solved components. The slow lane additionally executes real
Inductor scan/while loops. No CPU native solver is required except for the
explicit float64 CPU-reference test.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_extcell_multi_population import (
    DT,
    RECORD,
    STEPS,
    _arguments,
    _close,
    _close_gradients,
    _compiled_recorded,
    _drives,
    _gradients,
    _model,
    _recorded,
    _standalone_recorded,
    _trajectory,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.mark.parametrize(
    ("dtype", "shell_resistance_scale"),
    [
        pytest.param(torch.float32, 100.0, id="float32-conditioned"),
        pytest.param(torch.float64, 100.0, id="float64-conditioned"),
        pytest.param(torch.float64, 1.0, id="float64-low-shell-resistance"),
    ],
)
@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_cuda_extcell_multi_full_recording_and_all_source_gradients(
    dtype, batch_calls, shell_resistance_scale
):
    # Retain the original low-resistance stress fixture in float64; the CPU
    # cross-precision test documents its common-mode conditioning in float32.
    model = _model(
        device="cuda",
        dtype=dtype,
        batch_calls=batch_calls,
        shell_resistance_scale=shell_resistance_scale,
    )
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cuda", DTYPE=dtype):
        functional, tensors = dn.func.make_functional(model, dt=DT)
        parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
        actual = _recorded(
            _trajectory(functional, parameters, constants, state, ve, intra)
        )
        actual_gradients = _gradients(actual, targets)
        parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
        expected = _standalone_recorded(model, parameters, constants, state, ve, intra)
        expected_gradients = _gradients(expected, targets)
    assert len(actual_gradients) == 21
    assert actual["v"].device.type == "cuda"
    assert actual["vc"].shape == (STEPS + 1, *model.shape, 3)
    _close(actual, expected, dtype=dtype)
    _close_gradients(actual_gradients, expected_gradients, dtype=dtype)


def test_cuda_extcell_multi_float32_matches_float64_cpu_full_trace_and_gradients():
    pytest.importorskip("dendra_solvers")
    results = []
    for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32)):
        model = _model(device=device, dtype=dtype, shell_resistance_scale=100.0)
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE=device, DTYPE=dtype):
            functional, tensors = dn.func.make_functional(model, dt=DT)
            parameters, constants, state, ve, intra, targets = _arguments(
                model, tensors
            )
            recorded = _recorded(
                _trajectory(functional, parameters, constants, state, ve, intra)
            )
            results.append((recorded, _gradients(recorded, targets)))
    expected, actual = results
    _close(actual[0], expected[0], dtype=torch.float32, cross_device=True)
    _close_gradients(actual[1], expected[1], dtype=torch.float32, cross_device=True)


@pytest.mark.slow
@pytest.mark.parametrize("execution", ["scan", "while"])
def test_cuda_extcell_multi_compiled_inference_preserves_full_recording(execution):
    model = _model(
        device="cuda",
        dtype=torch.float32,
        batch_calls=(2,),
        shell_resistance_scale=100.0,
    )
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cuda", DTYPE=torch.float32):
        functional, tensors = dn.func.make_functional(model, dt=DT)
        ve, intra = _drives(model)
        with torch.no_grad():
            states = _trajectory(
                functional,
                tensors.parameters,
                tensors.constants,
                tensors.state,
                ve,
                intra,
            )
        runner = dn.func.compile(
            model,
            dt=DT,
            steps=STEPS,
            execution=execution,
            inference=True,
            callbacks={"trace": dn.func.Recorder(RECORD)},
        )
        assert not runner.kernel._compile_options.get("options", {}).get(
            "cpp_wrapper", False
        )
        with torch_compiler_warning_context():
            final, auxiliary = runner(inputs=dn.func.RolloutInput(ve=ve, intra=intra))
    _close(final, states[-1], dtype=torch.float32)
    _close(
        dict(auxiliary["callbacks"]["trace"]), _recorded(states), dtype=torch.float32
    )
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


@pytest.mark.slow
def test_cuda_extcell_multi_compiled_scan_preserves_all_source_gradients():
    model = _model(device="cuda", dtype=torch.float32, shell_resistance_scale=100.0)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cuda", DTYPE=torch.float32):
        functional, tensors = dn.func.make_functional(model, dt=DT)
        parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
        states = _trajectory(functional, parameters, constants, state, ve, intra)
        expected = _recorded(states)
        expected_gradients = _gradients(expected, targets)
        parameters, constants, state, ve, intra, targets = _arguments(model, tensors)
        with torch_compiler_warning_context():
            final, actual = _compiled_recorded(
                functional, parameters, constants, state, ve, intra
            )
            gradients = _gradients(actual, targets)
    _close(final, states[-1], dtype=torch.float32)
    _close(actual, expected, dtype=torch.float32)
    _close_gradients(gradients, expected_gradients, dtype=torch.float32)
