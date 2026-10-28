"""Direct callback chunks retain autograd and continuation on CPU and MPS."""

from __future__ import annotations

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


def _objective(state, auxiliary):
    return _loss(state) + auxiliary["callbacks"]["trace"]["v"].square().mean()


def _training_case(device, backend):
    with _environment(device):
        functional, parameters, constants, state, conductance = _case(device)
        original = _snapshot(functional.extract())
        inputs = _drives(functional, 4, grad=True)
        targets = (conductance, constants["diam"], inputs.ve, inputs.intra)
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
        kernel = _kernel(functional, backend=backend, steps=2)
        bound = kernel.bind(parameters, constants=constants)
        for iteration in range(2):
            if iteration:
                with torch.no_grad():
                    inputs.ve.mul_(1.2)
                    inputs.intra.mul_(1.1)
            for target in targets:
                target.grad = None
            prefix, first_aux = bound(
                state,
                dn.func.Inputs(inputs.ve[:2], inputs.intra[:2]),
                callbacks=callbacks,
            )
            actual, last_aux = bound(
                prefix,
                dn.func.Inputs(inputs.ve[2:], inputs.intra[2:]),
                callbacks=callbacks,
                callback_state=first_aux["callbacks"].state,
            )
            # A resumed recorder emits only the suffix; retain both calls in
            # the objective so gradient ownership spans the boundary.
            trace = torch.cat(
                (
                    first_aux["callbacks"]["trace"]["v"],
                    last_aux["callbacks"]["trace"]["v"],
                )
            )
            (_loss(actual) + trace.square().mean()).backward()
            expected, reference_aux = functional.prepare_and_rollout(
                parameters, constants, state, inputs, steps=4, callbacks=callbacks
            )
            reference = torch.autograd.grad(
                _objective(expected, reference_aux), targets
            )
            _assert_state(actual, expected, device)
            assert trace.shape[0] == 5
            torch.testing.assert_close(
                trace, reference_aux["callbacks"]["trace"]["v"], rtol=3e-5, atol=2e-5
            )
            for target, expected_gradient in zip(targets, reference, strict=True):
                observed = target.grad
                assert observed is not None and observed.device.type == device
                assert torch.isfinite(observed).all()
                scale = expected_gradient.abs().amax()
                assert scale > 0 and torch.count_nonzero(observed) > 0
                torch.testing.assert_close(
                    observed / scale, expected_gradient / scale, rtol=4e-3, atol=4e-4
                )
            assert bound._inference_prepared is None
        assert kernel.execution_report().strategy == "compiled_scan"
        assert kernel.execution_report().callbacks
        torch.testing.assert_close(functional.extract(), original)


@pytest.mark.cpu
@pytest.mark.slow
def test_cpu_inductor_direct_callbacks_keep_continuation_and_live_gradients():
    _training_case("cpu", "inductor")


@pytest.mark.mps
def test_mps_aot_eager_direct_callbacks_keep_continuation_and_live_gradients(
    mps_device,
):
    _training_case(mps_device, "aot_eager")


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "default"))
def test_mps_inductor_direct_callbacks_reuse_cache_and_resume(mps_device, execution):
    with _environment(mps_device):
        functional, parameters, constants, state, _ = _case(mps_device)
        original = _snapshot(functional.extract())
        inputs = _drives(functional, 8)
        frozen = _snapshot(inputs)
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
        kernel = _kernel(functional, backend="inductor", execution=execution, steps=4)
        bound = kernel.bind(parameters, constants=constants)
        prefix_inputs = dn.func.Inputs(inputs.ve[:4], inputs.intra[:4])
        with torch.inference_mode():
            first, prefix = bound(state, prefix_inputs, callbacks=callbacks)
            cached = bound._inference_prepared
            assert cached is not None
            repeated, _ = bound(state, prefix_inputs, callbacks=callbacks)
            assert bound._inference_prepared is cached
            final, suffix = bound(
                first,
                dn.func.Inputs(inputs.ve[4:], inputs.intra[4:]),
                callbacks=callbacks,
                callback_state=prefix["callbacks"].state,
            )
            assert bound._inference_prepared is cached
            expected, auxiliary = functional.prepare_and_rollout(
                parameters, constants, state, inputs, steps=8, callbacks=callbacks
            )
        _assert_state(first, repeated, mps_device)
        _assert_state(final, expected, mps_device)
        for field in ("v", "t"):
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
            "compiled_while_loop" if execution == "default" else "compiled_scan"
        )
        assert kernel.execution_report().callbacks
        torch.testing.assert_close(inputs, frozen)
        torch.testing.assert_close(functional.extract(), original)
