"""Portable callback carry resumes through real CPU and MPS compiler paths."""

from __future__ import annotations

import io
import os

import pytest
import torch

import dendra as dn
from tests.func.test_inputs_backends import _drives, _snapshot
from tests.func.test_raw_binding_backends import _assert_state, _case, _kernel, _loss
from tests.func.test_runtime_extra_backends import _environment


@pytest.fixture
def mps_device():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    return "mps"


def _callbacks(functional):
    return functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("v", "t")),
            "counts": dn.func.APCount(node_check=[0, 1], threshold=-55.0),
        }
    )


def _checkpoint(callbacks, state, auxiliary):
    buffer = io.BytesIO()
    torch.save(
        {
            "state": _snapshot(state),
            "callbacks": callbacks.export_state(auxiliary["callbacks"].state),
        },
        buffer,
    )
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def _reconstruct(functional):
    return dn.func.make_functional(functional._population, dt=functional.dt)[0]


def _training_case(device, backend):
    with _environment(device):
        functional, parameters, constants, initial, conductance = _case(device)
        original = _snapshot(functional.extract())
        callbacks = _callbacks(functional)
        source = _kernel(functional, backend=backend, steps=2).bind(
            parameters, constants=constants
        )
        with torch.no_grad():
            prefix, prefix_aux = source(initial, callbacks=callbacks)
        saved = _checkpoint(callbacks, prefix, prefix_aux)
        target = _reconstruct(functional)
        target_callbacks = _callbacks(target)
        restored = target_callbacks.restore_state(saved["callbacks"])
        bound = _kernel(target, backend=backend, steps=2).bind(
            parameters, constants=constants
        )
        inputs = _drives(target, 2, grad=True)
        targets = (conductance, constants["diam"], inputs.ve, inputs.intra)
        actual, auxiliary = bound(
            saved["state"], inputs, callbacks=target_callbacks, callback_state=restored
        )
        trace = auxiliary["callbacks"]["trace"]["v"]
        (_loss(actual) + trace.square().mean()).backward()
        expected, reference_aux = target.prepare_and_rollout(
            parameters,
            constants,
            saved["state"],
            inputs,
            steps=2,
            callbacks=target_callbacks,
            callback_state=restored,
        )
        reference = torch.autograd.grad(
            _loss(expected) + reference_aux["callbacks"]["trace"]["v"].square().mean(),
            targets,
        )
        _assert_state(actual, expected, device)
        torch.testing.assert_close(
            dict(auxiliary["callbacks"]),
            dict(reference_aux["callbacks"]),
            rtol=3e-5,
            atol=2e-5,
        )
        assert trace.shape[0] == 2
        for value, expected_gradient in zip(targets, reference, strict=True):
            assert value.grad is not None and torch.isfinite(value.grad).all()
            scale = expected_gradient.abs().amax()
            assert scale > 0 and torch.count_nonzero(value.grad) > 0
            torch.testing.assert_close(
                value.grad / scale, expected_gradient / scale, rtol=4e-3, atol=4e-4
            )
        assert bound._inference_prepared is None
        assert bound.compiled_chunk.execution_report().strategy == "compiled_scan"
        torch.testing.assert_close(functional.extract(), original)


@pytest.mark.cpu
@pytest.mark.slow
def test_cpu_inductor_restored_callbacks_keep_suffix_training_gradients():
    _training_case("cpu", "inductor")


@pytest.mark.mps
def test_mps_aot_eager_restored_callbacks_keep_suffix_training_gradients(mps_device):
    _training_case(mps_device, "aot_eager")


def _inference_case(device, execution):
    with _environment(device):
        functional, parameters, constants, initial, _ = _case(device)
        original = _snapshot(functional.extract())
        callbacks = _callbacks(functional)
        target = _reconstruct(functional)
        target_callbacks = _callbacks(target)
        source = _kernel(
            functional, backend="inductor", execution=execution, steps=4
        ).bind(parameters, constants=constants)
        bound = _kernel(target, backend="inductor", execution=execution, steps=4).bind(
            parameters, constants=constants
        )
        with torch.inference_mode():
            prefix, prefix_aux = source(initial, callbacks=callbacks)
            # Export inside inference mode must still produce ordinary owned
            # tensors so restoration can later participate in training.
            saved = _checkpoint(callbacks, prefix, prefix_aux)
            restored = target_callbacks.restore_state(saved["callbacks"])
            final, suffix = bound(
                saved["state"], callbacks=target_callbacks, callback_state=restored
            )
            cached = bound._inference_prepared
            repeated, repeated_aux = bound(
                saved["state"], callbacks=target_callbacks, callback_state=restored
            )
            assert bound._inference_prepared is cached and cached is not None
            expected, whole = functional.prepare_and_rollout(
                parameters, constants, initial, steps=8, callbacks=callbacks
            )
        _assert_state(final, expected, device)
        _assert_state(repeated, final, device)
        torch.testing.assert_close(
            dict(suffix["callbacks"]), dict(repeated_aux["callbacks"])
        )
        for field in ("v", "t"):
            combined = torch.cat(
                (
                    prefix_aux["callbacks"]["trace"][field],
                    suffix["callbacks"]["trace"][field],
                )
            )
            assert combined.shape[0] == 9
            torch.testing.assert_close(
                combined, whole["callbacks"]["trace"][field], rtol=3e-5, atol=2e-5
            )
        torch.testing.assert_close(
            suffix["callbacks"]["counts"], whole["callbacks"]["counts"]
        )
        assert bound.compiled_chunk.execution_report().strategy == (
            "compiled_while_loop" if execution == "default" else "compiled_scan"
        )
        torch.testing.assert_close(functional.extract(), original)


@pytest.mark.cpu
@pytest.mark.slow
def test_cpu_inductor_restored_callback_while_resumes_recording_and_cache():
    _inference_case("cpu", "default")


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "default"))
def test_mps_inductor_restored_callbacks_resume_recording_and_cache(
    mps_device, execution
):
    _inference_case(mps_device, execution)
