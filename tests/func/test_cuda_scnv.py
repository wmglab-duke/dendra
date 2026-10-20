"""CUDA acceptance for voltage-skip carry, current phases, and compiled loops.

Host skips and successful collection do not constitute GPU validation.
"""

from __future__ import annotations

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_scnv import _exercise, _exercise_continuation

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_scnv_full_trace_and_gradient_oracle(dtype):
    trace = _exercise(device="cuda", dtype=dtype)
    assert all(value.device.type == "cuda" for value in trace.values())


def test_cuda_scnv_float32_matches_cpu_float64():
    expected = _exercise(device="cpu", dtype=torch.float64)
    actual = _exercise(device="cuda", dtype=torch.float32)
    for name, value in actual.items():
        reference = expected[name].detach().cpu()
        value = value.detach().to(device="cpu", dtype=torch.float64)
        if name == "i_membrane":
            value, reference = value * 1e6, reference * 1e6
        torch.testing.assert_close(value, reference, rtol=5e-4, atol=3e-5, msg=name)


def test_cuda_scnv_explicit_voltage_and_carry_continuation():
    _exercise_continuation(device="cuda")


@pytest.mark.slow
@pytest.mark.parametrize("execution,checkpoint_every", (("scan", 2), ("while", None)))
def test_cuda_scnv_inductor_checkpointed_scan_and_inference_while(
    execution, checkpoint_every
):
    _exercise(
        device="cuda",
        dtype=torch.float32,
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
