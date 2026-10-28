"""Shared drive packets retain gradients and inference on real CPU/MPS backends."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_raw_binding_backends import _assert_state, _case, _kernel, _loss
from tests.func.test_runtime_extra_backends import _environment


@pytest.fixture
def mps_device():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", (
        "Run MPS backend tests with PYTORCH_ENABLE_MPS_FALLBACK=0"
    )
    return "mps"


def _drives(functional, steps, *, grad=False):
    shape = (steps, *functional.shape)
    count = torch.Size(shape).numel()
    options = dict(dtype=functional.dtype, device=functional.device)
    ve = torch.linspace(-0.2, 0.3, count, **options).reshape(shape)
    intra = torch.linspace(0.8e-10, 1.2e-10, count, **options).reshape(shape)
    return dn.func.Inputs(ve.requires_grad_(grad), intra.requires_grad_(grad))


def _snapshot(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _training_case(device, backend):
    with _environment(device):
        functional, parameters, constants, state, conductance = _case(device)
        original = _snapshot(functional.extract())
        inputs = _drives(functional, 3, grad=True)
        targets = (conductance, constants["diam"], inputs.ve, inputs.intra)
        kernel = _kernel(functional, backend=backend)
        bound = kernel.bind(parameters, constants=constants)
        outputs = []
        for iteration in range(2):
            if iteration:
                with torch.no_grad():
                    inputs.ve.mul_(1.2)
                    inputs.intra.mul_(1.1)
            for target in targets:
                target.grad = None
            actual, _ = dn.func.run(
                bound, state, inputs=inputs, steps=3, checkpoint_every=2
            )
            _loss(actual).backward()
            expected, _ = functional.prepare_and_rollout(
                parameters, constants, state, dn.func.RolloutInput(*inputs), steps=3
            )
            reference = torch.autograd.grad(_loss(expected), targets)
            _assert_state(actual, expected, device)
            for target, expected_gradient in zip(targets, reference, strict=True):
                observed = target.grad
                assert observed is not None and observed.device.type == device
                assert torch.isfinite(observed).all()
                scale = expected_gradient.abs().amax()
                assert scale > 0 and torch.count_nonzero(observed) > 0
                torch.testing.assert_close(
                    observed / scale, expected_gradient / scale, rtol=4e-3, atol=4e-4
                )
            outputs.append(actual["integrator"]["v"].detach().clone())
            assert bound._inference_prepared is None
        assert not torch.equal(*outputs)
        assert kernel.execution_report().strategy == "compiled_scan"
        torch.testing.assert_close(functional.extract(), original)


@pytest.mark.cpu
@pytest.mark.slow
def test_cpu_inductor_shared_inputs_keep_checkpoint_and_live_input_gradients():
    _training_case("cpu", "inductor")


@pytest.mark.mps
def test_mps_aot_eager_shared_inputs_keep_checkpoint_and_live_input_gradients(
    mps_device,
):
    _training_case(mps_device, "aot_eager")


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_mps_inductor_shared_inputs_reuse_cache_and_resume_recording(
    mps_device, execution
):
    with _environment(mps_device):
        functional, parameters, constants, state, _ = _case(mps_device)
        source = _snapshot(functional.extract())
        inputs = _drives(functional, 8)
        frozen = _snapshot(inputs)
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
        kernel = _kernel(
            functional,
            backend="inductor",
            execution="default" if execution == "while" else "scan",
            steps=4 if execution == "while" else 2,
        )
        bound = kernel.bind(parameters, constants=constants)
        prefix_inputs = dn.func.Inputs(inputs.ve[:4], inputs.intra[:4])
        with torch.inference_mode():
            first, prefix = dn.func.run(
                bound, state, inputs=prefix_inputs, steps=4, callbacks=callbacks
            )
            cached = bound._inference_prepared
            assert cached is not None
            repeated, _ = dn.func.run(
                bound, state, inputs=prefix_inputs, steps=4, callbacks=callbacks
            )
            assert bound._inference_prepared is cached
            empty, middle = dn.func.run(
                bound,
                first,
                inputs=dn.func.Inputs(inputs.ve[:0], inputs.intra[:0]),
                steps=0,
                callbacks=callbacks,
                callback_state=prefix["callbacks"].state,
            )
            final, suffix = dn.func.run(
                bound,
                empty,
                inputs=dn.func.Inputs(inputs.ve[4:], inputs.intra[4:]),
                steps=4,
                callbacks=callbacks,
                callback_state=middle["callbacks"].state,
            )
            assert bound._inference_prepared is cached
            expected, auxiliary = functional.prepare_and_rollout(
                parameters,
                constants,
                state,
                dn.func.RolloutInput(*inputs),
                steps=8,
                callbacks=callbacks,
            )
        _assert_state(first, repeated, mps_device)
        _assert_state(final, expected, mps_device)
        for field in ("v", "t"):
            assert middle["callbacks"]["trace"][field].shape[0] == 0
            resumed = torch.cat(
                (
                    prefix["callbacks"]["trace"][field],
                    suffix["callbacks"]["trace"][field],
                )
            )
            assert resumed.shape[0] == 9
            torch.testing.assert_close(
                resumed, auxiliary["callbacks"]["trace"][field], rtol=3e-5, atol=2e-5
            )
        assert not final["integrator"]["v"].requires_grad
        assert kernel.execution_report().strategy == (
            "compiled_while_loop" if execution == "while" else "compiled_scan"
        )
        torch.testing.assert_close(inputs, frozen)
        torch.testing.assert_close(functional.extract(), source)
