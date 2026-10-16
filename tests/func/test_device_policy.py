"""Functional device admission and compiler policy without GPU allocation."""

from types import SimpleNamespace

import pytest
import torch

from dendra.func._convenience import _inference_compile_options
from dendra.func._population import _functional_device_failures


@pytest.mark.parametrize("device", ["cpu", "cpu:0", "cuda", "cuda:1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_functional_device_admission_is_independent_of_hardware_availability(
    device, dtype
):
    assert _functional_device_failures(torch.device(device), dtype) == []


@pytest.mark.parametrize("device", ["mps", "meta", "xpu"])
def test_functional_devices_without_a_solver_contract_are_rejected(device):
    failures = _functional_device_failures(torch.device(device), torch.float32)
    assert failures == ["the supported functional devices are CPU and CUDA"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_functional_solver_precision_contract_remains_explicit(dtype):
    assert _functional_device_failures(torch.device("cuda"), dtype) == [
        "dtype must be float32 or float64"
    ]


def test_functional_device_and_dtype_failures_are_reported_together():
    assert _functional_device_failures(torch.device("meta"), torch.float16) == [
        "the supported functional devices are CPU and CUDA",
        "dtype must be float32 or float64",
    ]


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_cuda_convenience_runner_does_not_force_cpu_cpp_wrapper(inference, backend):
    functional = SimpleNamespace(device=torch.device("cuda:1"))
    keywords = {"fullgraph": True, "dynamic": False}
    actual = _inference_compile_options(functional, inference, backend, None, keywords)
    assert actual == {**keywords, "backend": backend}
    assert keywords == {"fullgraph": True, "dynamic": False}


def test_cuda_convenience_runner_preserves_explicit_compiler_options():
    functional = SimpleNamespace(device=torch.device("cuda"))
    options = {"cpp_wrapper": False, "max_autotune": False}
    keywords = {"fullgraph": True}
    actual = _inference_compile_options(functional, True, "inductor", options, keywords)
    assert actual == {"backend": "inductor", **keywords, "options": options}
    assert actual["options"] is not options
    assert options == {"cpp_wrapper": False, "max_autotune": False}
    assert keywords == {"fullgraph": True}
