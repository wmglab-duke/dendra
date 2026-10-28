"""Real compiler backends preserve raw-binding graph and cache lifetimes.

Run the MPS cases with PYTORCH_ENABLE_MPS_FALLBACK=0 on Apple hardware.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_runtime_extra_backends import DT, _environment, _model

CHUNK_STEPS = 2


@pytest.fixture
def mps_device():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", (
        "Run MPS backend tests with PYTORCH_ENABLE_MPS_FALLBACK=0"
    )
    return "mps"


def _case(device):
    model = _model(device)
    with torch.no_grad():
        # Leak reversal is -65 mV. A nonuniform displaced voltage exercises
        # both membrane conductance and diameter-dependent axial coupling.
        model.v.copy_(
            torch.linspace(
                -58.0,
                -52.0,
                model.v.numel(),
                device=model.device(),
                dtype=model.dtype(),
            ).reshape_as(model.v)
        )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = tensors.independent_parameters(trainable="*g_param")
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    constants["diam"].requires_grad_()
    conductance = next(
        value for name, value in parameters.items() if name.endswith("g_param")
    )
    state = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors.state
    )
    return functional, parameters, constants, state, conductance


def _kernel(functional, *, backend, execution="scan", steps=CHUNK_STEPS):
    keywords = {}
    if backend == "inductor":
        keywords["options"] = {"compile_threads": 1}
        if functional.device.type == "mps":
            keywords["options"]["max_fusion_unique_io_buffers"] = 30
    return functional.compile_rollout_chunk(
        steps, execution=execution, backend=backend, **keywords
    )


def _loss(state):
    voltage = state["integrator"]["v"]
    weights = torch.linspace(
        0.7, 1.3, voltage.shape[-1], device=voltage.device, dtype=voltage.dtype
    )
    return (voltage * weights).square().mean()


def _assert_state(actual, expected, device):
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    voltage = actual["integrator"]["v"]
    assert voltage.device.type == device
    assert voltage.dtype == torch.float32


def _training_case(device, backend):
    with _environment(device):
        functional, parameters, constants, state, conductance = _case(device)
        targets = (conductance, constants["diam"])
        kernel = _kernel(functional, backend=backend)
        bound = kernel.bind(parameters, constants=constants)
        for _ in range(2):
            for target in targets:
                target.grad = None
            actual, _ = dn.func.run(bound, state, steps=3, checkpoint_every=2)
            _loss(actual).backward()
            expected, _ = functional.prepare_and_rollout(
                parameters, constants, state, steps=3
            )
            expected_gradients = torch.autograd.grad(_loss(expected), targets)
            _assert_state(actual, expected, device)
            for target, expected_gradient in zip(
                targets, expected_gradients, strict=True
            ):
                observed = target.grad
                assert observed is not None
                assert observed.device.type == device
                assert torch.isfinite(observed).all()
                scale = expected_gradient.abs().amax()
                assert scale > 0
                assert torch.count_nonzero(observed) > 0
                torch.testing.assert_close(
                    observed / scale,
                    expected_gradient / scale,
                    rtol=4e-3,
                    atol=4e-4,
                )
            assert bound.prepared is None
            assert bound._inference_prepared is None
        assert kernel.execution_report().strategy == "compiled_scan"


@pytest.mark.cpu
@pytest.mark.slow
def test_cpu_inductor_raw_binding_checkpointed_backwards_use_fresh_graphs():
    _training_case("cpu", "inductor")


@pytest.mark.mps
def test_mps_aot_eager_raw_binding_checkpointed_backwards_use_fresh_graphs(mps_device):
    _training_case(mps_device, "aot_eager")


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_mps_inductor_raw_inference_reuses_preparation_and_continues(
    mps_device, execution
):
    with _environment(mps_device):
        functional, parameters, constants, state, _ = _case(mps_device)
        # The default inference loop needs a complete four-step block. Scan
        # also exercises host scheduling across two bounded two-step chunks.
        kernel = _kernel(
            functional,
            backend="inductor",
            execution="default" if execution == "while" else "scan",
            steps=4 if execution == "while" else CHUNK_STEPS,
        )
        bound = kernel.bind(parameters, constants=constants)
        with torch.inference_mode():
            first, _ = dn.func.run(bound, state, steps=4)
            cached = bound._inference_prepared
            assert cached is not None
            repeated, _ = dn.func.run(bound, state, steps=4)
            assert bound._inference_prepared is cached
            continued, _ = dn.func.run(bound, first, steps=4)
            assert bound._inference_prepared is cached
            expected_first, _ = functional.prepare_and_rollout(
                parameters, constants, state, steps=4
            )
            expected_continued, _ = functional.prepare_and_rollout(
                parameters, constants, state, steps=8
            )
        _assert_state(first, expected_first, mps_device)
        _assert_state(repeated, first, mps_device)
        _assert_state(continued, expected_continued, mps_device)
        assert not continued["integrator"]["v"].requires_grad
        assert not torch.is_inference(cached.values["integrator"]["diag_base"])
        assert bound.prepared is None
        assert kernel.execution_report().strategy == (
            "compiled_while_loop" if execution == "while" else "compiled_scan"
        )
