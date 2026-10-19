"""Functional diffusion acceptance on a real CUDA device.

Run ``pytest tests/func/test_cuda_material_diffusion.py -W error -m cuda``.
Host collection/skips are not CUDA validation. Both native chain/Tree kernels,
live geometry, full state, fresh/checkpointed continuation, all HH gradients,
material gradients, vmap/JVP, and float32 versus CPU-float64 controls are covered.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_clearance_process import (
    NAI,
    POOL,
    STEPS,
    _close,
    _explicit_reference,
    _runner,
)
from tests.func.test_cuda_convenience_initialization import _cross_precision_close
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_material_diffusion import (
    _arguments,
    _exercise,
    _exercise_transforms,
    _gradients,
    _is_tree_kind,
    _matrix_oracle,
    _model,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _assert_coverage(recorded, gradients, *, dtype, steps=STEPS, block=False):
    assert {
        "v",
        "i_membrane",
        "t",
        "hh.m",
        "hh.h",
        "hh.n",
        POOL,
        NAI,
        "na_ion.ena",
        "na_ion.ina",
    } <= recorded.keys()
    for name, value in recorded.items():
        assert value.shape[0] == steps + 1, name
        assert torch.isfinite(value).all(), name
        if name != "t":
            assert value.device.type == "cuda" and value.dtype == dtype, name
    assert recorded["t"][0].item() == 0
    assert torch.count_nonzero(recorded["i_membrane"][0]) == 0
    if block:
        assert recorded["vc"].shape == (*recorded["v"].shape, 3)
        torch.testing.assert_close(
            recorded["vc"][0, ..., 0], recorded["v"][0], rtol=0, atol=0
        )
        assert torch.count_nonzero(recorded["vc"][0, ..., 1:]) == 0
    if gradients:
        assert {
            "stimulus",
            "initial_voltage",
            "gnabar",
            "gkbar",
            "gl",
            "ena",
            "ek",
            "el",
            "dt",
            "geometry",
            "initial_pool",
            "initial_nai",
        } <= gradients.keys()
        for name, value in gradients.items():
            assert value.device.type == "cuda", name
            assert torch.isfinite(value).all() and torch.count_nonzero(value), name


@pytest.mark.parametrize("kind", ("path", "tree", "block_path", "block_tree", "cable"))
def test_cuda_diffusion_native_full_traces_fresh_initialization_and_all_gradients(kind):
    block = kind.startswith("block_")
    recorded, gradients = _exercise(
        kind,
        device="cuda",
        dtype=torch.float64,
        shell_resistance_scale=100.0 if block else 1.0,
    )
    _assert_coverage(recorded, gradients, dtype=torch.float64, block=block)
    geometry = (
        {"chemical_volume", "chemical_edge_factor"}
        if _is_tree_kind(kind)
        else {"length"}
    )
    assert geometry <= gradients.keys()


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_cuda_diffusion_float32_matches_cpu_float64_traces_and_all_gradients(kind):
    pytest.importorskip("dendra_solvers")
    expected, actual = [
        _exercise(
            kind,
            geometry="area_distance",
            edge_diffusivity=True,
            device=device,
            dtype=dtype,
            checkpoint_every=2,
        )
        for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32))
    ]
    # Area and distance are primitive trainable geometry sources in this case.
    assert {
        "chemical_volume",
        "chemical_edge_area",
        "chemical_edge_distance",
        "edge_diffusivity",
    } <= actual[1].keys()
    _assert_coverage(actual[0], actual[1], dtype=torch.float32)
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


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_cuda_diffusion_checkpointed_continuation_full_traces_and_all_gradients(kind):
    model = _model(kind, geometry="factor", device="cuda", dtype=torch.float64)
    runner = _runner(model, checkpoint_every=2)
    source = _module_snapshot(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    state, auxiliary = runner(
        parameters=parameters, initialization=initialization, inputs=inputs
    )
    final, continued = runner(
        state,
        parameters=auxiliary["parameters"],
        inputs=inputs,
        callback_state=auxiliary["callbacks"].state,
    )
    recorded = {
        name: torch.cat((value, continued["callbacks"]["trace"][name]))
        for name, value in auxiliary["callbacks"]["trace"].items()
    }
    gradients = _gradients(recorded, targets)
    rp, ri, rx, rt = _arguments(runner, model)
    expected = _explicit_reference(
        runner,
        rp,
        ri,
        dn.func.RolloutInput(*(torch.cat((value, value)) for value in rx)),
        steps=2 * STEPS,
    )
    _assert_tree_close(final, expected[0], rtol=3e-9, atol=3e-10)
    reference_recorded = dict(expected[1]["callbacks"]["trace"])
    _close(recorded, reference_recorded)
    _close(gradients, _gradients(reference_recorded, rt))
    _matrix_oracle(model, recorded, parameters, runner.constants["dt"])
    _assert_coverage(recorded, gradients, dtype=torch.float64, steps=2 * STEPS)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_cuda_diffusion_vmap_empty_vmap_and_jvp_full_traces(kind):
    _exercise_transforms(kind, device="cuda", dtype=torch.float64)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    (("path", "scan", 2), ("tree", "while", None)),
)
def test_cuda_inductor_diffusion_checkpointed_scan_and_while_full_traces(
    kind, execution, checkpoint_every
):
    recorded, gradients = _exercise(
        kind,
        geometry="factor",
        device="cuda",
        dtype=torch.float32,
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
    _assert_coverage(recorded, gradients, dtype=torch.float32)
    if execution == "while":
        assert not gradients and not recorded["v"].requires_grad
