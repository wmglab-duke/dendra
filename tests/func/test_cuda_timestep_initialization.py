"""Fresh timestep workspaces on a real CUDA device.

Run with ``pytest tests/func/test_cuda_timestep_initialization.py -W error -m cuda``.
Shared CPU fixtures distinguish literal-zero initialization workspaces from
runtime builders evaluated at zero dt, retain full state/current recordings,
and compare rollout gradients with explicit initialization and recurrence. Only
two tiny cases generate GPU Inductor kernels; there is no unrolled matrix.
"""

from __future__ import annotations

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_cuda_convenience_initialization import _cross_precision_close
from tests.func.test_timestep_initialization import (
    MECHANISM,
    STATE,
    STEPS,
    WORKSPACE_TARGETS,
    _exercise,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _assert_coverage(recorded, gradients, *, block, dtype):
    expected = {"v", "i_membrane", "hh.m", "hh.h", "hh.n", "t"}
    if block:
        expected.add("vc")
    expected.update(f"{MECHANISM}.{name}" for name in ("activation", "seed"))
    expected.add(f"{MECHANISM}.DE.{STATE}.state_seed")
    assert expected <= recorded.keys()
    for name, value in recorded.items():
        assert value.shape[0] == STEPS + 1, name
        assert torch.isfinite(value).all(), name
        if name != "t":
            assert value.dtype == dtype, name
            assert value.device.type == "cuda", name
    assert recorded["t"][0].item() == 0
    assert torch.count_nonzero(recorded["i_membrane"][0]) == 0
    if block:
        circuit = recorded["vc"]
        assert circuit.shape == (*recorded["v"].shape, 3)
        torch.testing.assert_close(circuit[0, ..., 0], recorded["v"][0], rtol=0, atol=0)
        assert torch.count_nonzero(circuit[0, ..., 1:]) == 0
    if not gradients:
        return
    standard = {
        "stimulus",
        "initial_voltage",
        "gnabar",
        "gkbar",
        "gl",
        "ena",
        "ek",
        "el",
    }
    assert standard | {"dt", "geometry", "celsius"} | set(WORKSPACE_TARGETS) == (
        gradients.keys()
    )
    for name, value in gradients.items():
        assert value.device.type == "cuda", name
        assert torch.isfinite(value).all(), name
        assert torch.count_nonzero(value), name


@pytest.mark.parametrize(
    ("kind", "checkpoint_every"), (("path", None), ("multi_block", 2))
)
def test_cuda_fresh_timestep_workspaces_full_traces_and_all_gradients(
    kind, checkpoint_every
):
    recorded, gradients = _exercise(
        kind,
        device="cuda",
        dtype=torch.float64,
        shell_resistance_scale=100.0,
        backend="eager",
        execution="scan",
        checkpoint_every=checkpoint_every,
    )
    _assert_coverage(
        recorded, gradients, block=kind == "multi_block", dtype=torch.float64
    )


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_cuda_fresh_timestep_float32_matches_float64_cpu_traces_and_gradients(kind):
    pytest.importorskip("dendra_solvers")
    results = []
    for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32)):
        results.append(
            _exercise(
                kind,
                device=device,
                dtype=dtype,
                shell_resistance_scale=100.0,
                backend="eager",
                execution="scan",
                checkpoint_every=2,
            )
        )
    expected, actual = results
    _assert_coverage(
        actual[0], actual[1], block=kind == "multi_block", dtype=torch.float32
    )
    _cross_precision_close(actual[0], expected[0])
    assert actual[1].keys() == expected[1].keys()
    for name, value in actual[1].items():
        reference = expected[1][name].detach().to(device="cpu", dtype=torch.float64)
        value = value.detach().to(device="cpu", dtype=torch.float64)
        scale = reference.abs().amax()
        assert torch.isfinite(scale) and scale > 0, name
        torch.testing.assert_close(
            value / scale,
            reference / scale,
            rtol=5e-3,
            atol=5e-4,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    (("path", "scan", 2), ("multi_block", "while", None)),
)
def test_cuda_inductor_fresh_timestep_scan_gradients_and_while_full_traces(
    kind, execution, checkpoint_every
):
    recorded, gradients = _exercise(
        kind,
        device="cuda",
        dtype=torch.float32,
        shell_resistance_scale=100.0,
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
    _assert_coverage(
        recorded, gradients, block=kind == "multi_block", dtype=torch.float32
    )
    if execution == "while":
        assert not gradients
        assert not recorded["v"].requires_grad
