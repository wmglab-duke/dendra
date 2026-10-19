"""CUDA acceptance for preconfigured, independently connected packed chemistry.

Run ``pytest tests/func/test_cuda_material_concatenation.py -W error -m cuda``.
Host skips are not CUDA validation. Tests include source/geometry ownership,
full recordings, the eight electrical gradients, and sparse chemical support.
"""

from __future__ import annotations

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_material_concatenation import (
    _exercise,
    _exercise_explicitly_batched_named_diffusion,
    _exercise_ion_sources,
)
from tests.test_material_concatenation import PROCESSES
from tests.test_material_concatenation import _exercise as _imperative_exercise
from tests.test_material_concatenation import _one_sided_exchange

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


@pytest.mark.parametrize("family", tuple(PROCESSES))
def test_cuda_material_concatenation_independent_traces_and_gradients(family):
    recorded, gradients = _imperative_exercise(
        family, device="cuda", dtype=torch.float64
    )
    assert all(value.device.type == "cuda" for value in recorded.values())
    assert all(value.device.type == "cuda" for value in gradients.values())
    functional, functional_gradients = _exercise(
        family, device="cuda", dtype=torch.float64
    )
    assert all(value.device.type == "cuda" for value in functional.values())
    assert all(value.device.type == "cuda" for value in functional_gradients.values())


@pytest.mark.parametrize("missing", ("material", "process"))
def test_cuda_packed_diffusion_sparse_component_support(missing):
    _exercise(
        "diffusion",
        device="cuda",
        dtype=torch.float64,
        **{f"missing_{missing}": True},
    )


def test_cuda_packed_diffusion_float32_matches_cpu_float64():
    expected, actual = [
        _exercise("named_diffusion", device=device, dtype=dtype, checkpoint_every=2)
        for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32))
    ]
    for actual_tree, expected_tree in zip(actual, expected, strict=True):
        assert actual_tree.keys() == expected_tree.keys()
        for name, value in actual_tree.items():
            reference = (
                expected_tree[name].detach().to(device="cpu", dtype=torch.float64)
            )
            value = value.detach().to(device="cpu", dtype=torch.float64)
            scale = reference.abs().amax().clamp_min(1e-10)
            torch.testing.assert_close(
                value / scale, reference / scale, rtol=6e-3, atol=6e-4, msg=name
            )


def test_cuda_packed_exchange_requires_both_endpoint_fields():
    _one_sided_exchange(device="cuda", dtype=torch.float64)


def test_cuda_preconfigured_ion_sources_styles_and_rate_gradients():
    _exercise_ion_sources(device="cuda", dtype=torch.float64)


def test_cuda_explicitly_batched_named_diffusion_full_traces_and_gradients():
    _exercise_explicitly_batched_named_diffusion(device="cuda")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("execution", "checkpoint_every"), (("scan", 2), ("while", None))
)
def test_cuda_inductor_packed_diffusion_scan_and_while(execution, checkpoint_every):
    recorded, gradients = _exercise(
        "named_diffusion",
        device="cuda",
        dtype=torch.float32,
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
    assert all(value.device.type == "cuda" for value in recorded.values())
    if execution == "while":
        assert not gradients and not recorded["v"].requires_grad
