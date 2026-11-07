"""CUDA acceptance for functional electrode projections and physical regions.

Run with ``python -m pytest tests/func/test_cuda_lfp.py -W error -m cuda``.
The point-model tests do not require the native CPU solver package.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_lfp import DT, _eager, _fields, _inputs, _model
from tests.func.test_membrane_current import _model as _membrane_model

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _close(actual, expected, dtype):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=3.0e-5 if dtype == torch.float32 else 2.0e-10,
        atol=1.0e-12 if dtype == torch.float32 else 1.0e-18,
    )


def _close_gradients(actual, expected, dtype):
    for gradient, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(gradient).all() and torch.count_nonzero(reference) > 0
        scale = reference.abs().amax()
        torch.testing.assert_close(
            gradient / scale,
            reference / scale,
            rtol=2.0e-4 if dtype == torch.float32 else 2.0e-8,
            atol=2.0e-5 if dtype == torch.float32 else 2.0e-10,
        )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_lfp_full_projection_field_and_stimulus_gradients(dtype):
    model = _model(device="cuda", dtype=dtype, batch=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    fields = _fields(model, trainable=True)
    inputs = _inputs(model, trainable=True)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("i_membrane",)), "lfp": dn.func.LFP(fields)}
    )
    _, auxiliary = _eager(functional, tensors, callbacks, inputs)
    outputs = auxiliary["callbacks"]
    actual = outputs["lfp"]
    current = outputs["trace"]["i_membrane"]
    reference = torch.einsum("tbnc,knc->tbk", current, torch.stack(fields))
    assert actual.shape == (inputs.intra.shape[0] + 1, 2, 2)
    assert actual.device.type == "cuda" and actual.dtype == dtype
    _close(actual, reference, dtype)
    targets = (*fields, inputs.intra)
    actual_gradient = torch.autograd.grad(
        (actual * 1.0e6).square().mean(), targets, retain_graph=True
    )
    reference_gradient = torch.autograd.grad(
        (reference * 1.0e6).square().mean(), targets
    )
    _close_gradients(actual_gradient, reference_gradient, dtype)


@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_cuda_lfp_compiled_scan_gradients_and_while_projection(execution):
    dtype = torch.float32
    model = _model(device="cuda", dtype=dtype, batch=(2,))
    fields = _fields(model, trainable=execution == "scan")
    inputs = _inputs(model, steps=4, trainable=execution == "scan")
    specifications = {"lfp": dn.func.LFP(fields)}
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(specifications)
    _, expected_auxiliary = _eager(functional, tensors, callbacks, inputs)
    reference = expected_auxiliary["callbacks"]["lfp"]
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        execution=execution,
        inference=execution == "while",
        checkpoint_every=2 if execution == "scan" else None,
        callbacks=specifications,
        options={"compile_threads": 1},
    )
    with torch_compiler_warning_context():
        _, auxiliary = runner(inputs=inputs)
    actual = auxiliary["callbacks"]["lfp"]
    _close(actual, reference, dtype)
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    if execution == "scan":
        targets = (*fields, inputs.intra)
        expected_gradient = torch.autograd.grad(
            (reference * 1.0e6).square().mean(), targets
        )
        actual_gradient = torch.autograd.grad((actual * 1.0e6).square().mean(), targets)
        _close_gradients(actual_gradient, expected_gradient, dtype)


def test_cuda_recorder_region_preserves_extracellular_layers_and_current():
    model = _membrane_model("block_path", device="cuda", dtype=torch.float64)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    region = model[..., [2, 0, 2]]
    callbacks = functional.make_callbacks(
        {
            "all": dn.func.Recorder(("v", "vc", "i_membrane")),
            "region": dn.func.Recorder(("v", "vc", "i_membrane"), region=region),
        }
    )
    _, auxiliary = _eager(functional, tensors, callbacks, _inputs(model, steps=4))
    outputs = auxiliary["callbacks"]
    for name in ("v", "i_membrane"):
        torch.testing.assert_close(
            outputs["region"][name], outputs["all"][name][..., [2, 0, 2]]
        )
    torch.testing.assert_close(
        outputs["region"]["vc"], outputs["all"]["vc"][..., [2, 0, 2], :]
    )
    assert outputs["region"]["vc"].shape[-1] == 3
