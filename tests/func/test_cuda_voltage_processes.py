"""CUDA acceptance for pure voltage resets and explicit refractory carry.

Run ``pytest tests/func/test_cuda_voltage_processes.py -W error -m cuda``.
Host skips are not GPU validation. Float32 is checked against both the actual
CUDA imperative lifecycle and an independent CPU float64 gradient reference.
"""

from __future__ import annotations

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_voltage_continuation import _exercise_changed_fresh_continuation
from tests.func.test_voltage_processes import _exercise, _exercise_continuation
from tests.test_voltage_process_pure import FIRE_TYPES

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_fire_full_traces_and_gradients(variant, dtype):
    recorded, gradients = _exercise(variant, device="cuda", dtype=dtype)
    assert all(value.device.type == "cuda" for value in recorded.values())
    assert all(value.device.type == "cuda" for value in gradients.values())


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_cuda_fresh_snapshot_and_explicit_carry_continuation(variant):
    _exercise_continuation(variant, device="cuda", dtype=torch.float64)


@pytest.mark.parametrize("kind", ("point", "multi_block"))
def test_cuda_changed_fresh_initialization_and_differentiable_continuation(kind):
    _exercise_changed_fresh_continuation(kind, device="cuda", dtype=torch.float64)


@pytest.mark.parametrize(
    "kind",
    ("path", "myelinated", "block_path", "block_tree", "multi_scalar", "multi_block"),
)
def test_cuda_fire_geometry_and_packed_carry(kind):
    _exercise("fire_r_d", kind=kind, device="cuda", dtype=torch.float64)


def test_cuda_regional_and_ordered_voltage_processes():
    _exercise(
        "fire_r_d", regional=True, ordered=True, device="cuda", dtype=torch.float64
    )


@pytest.mark.parametrize("variant", tuple(FIRE_TYPES))
def test_cuda_float32_fire_matches_cpu_float64(variant):
    expected, actual = [
        _exercise(variant, device=device, dtype=dtype)
        for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32))
    ]
    for actual_tree, expected_tree in zip(actual, expected, strict=True):
        assert actual_tree.keys() == expected_tree.keys()
        for name, value in actual_tree.items():
            reference = expected_tree[name].detach().cpu()
            if value.dtype == torch.bool:
                assert torch.equal(value.cpu(), reference), name
                continue
            value = value.detach().to(device="cpu", dtype=torch.float64)
            scale = reference.abs().amax().clamp_min(1e-8)
            torch.testing.assert_close(
                value / scale, reference / scale, rtol=2e-3, atol=2e-4, msg=name
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("execution", "checkpoint_every"), (("scan", 2), ("while", None))
)
def test_cuda_inductor_fire_checkpointed_scan_and_while(execution, checkpoint_every):
    recorded, gradients = _exercise(
        "fire_r_d",
        device="cuda",
        dtype=torch.float32,
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
    assert recorded["fire_r_d.is_refractory"].dtype == torch.bool
    if execution == "while":
        assert not gradients and not recorded["v"].requires_grad
