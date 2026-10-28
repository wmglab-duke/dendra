"""Real CPU/MPS kernels share raw-bound continuation and graph ownership."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from tests.func.test_continuation import _drive, _joined, _snapshot
from tests.func.test_continuation_backends import (
    _assert_result,
    _compile,
    _environment,
    _inputs,
    _loss,
)
from tests.func.test_lowering_api import G, _target, _voltage


@pytest.fixture
def mps_device():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    return "mps"


def _training(network, device, backend):
    with _environment(device):
        runner = _compile(network, device, backend)
        functional = runner.functional
        bound = runner.kernel.bind(runner.parameters, constants=runner.constants)
        native = _snapshot(functional.extract().state)
        current, inputs = _drive(bound, network, 5, requires_grad=True)
        _, first = dn.func.run(
            bound,
            runner.initial_state,
            inputs=_inputs(network, current[:3]),
            steps=3,
            checkpoint_every=2,
            callbacks=runner.callbacks,
        )
        packet = first["continuation"]
        conductance = _target(packet.parameters, network)
        assert conductance is _target(bound.parameters, network)
        if network:
            assert conductance is packet.parameters["populations"]["right"][G]
        actual, last = dn.func.run(
            bound,
            packet,
            inputs=_inputs(network, current[3:]),
            steps=2,
            checkpoint_every=1,
            callbacks=runner.callbacks,
        )
        trace = _joined(first, last)
        observed = torch.autograd.grad(
            _loss(actual, trace, network), (conductance, current), retain_graph=True
        )
        expected, reference = functional.prepare_and_rollout(
            packet.parameters,
            packet.constants,
            runner.initial_state,
            inputs,
            steps=5,
            callbacks=runner.callbacks,
        )
        gradients = torch.autograd.grad(
            _loss(expected, reference["callbacks"]["trace"], network),
            (conductance, current),
        )
        _assert_result(
            actual, expected, trace, reference["callbacks"]["trace"], network, device, 5
        )
        for actual_gradient, expected_gradient in zip(observed, gradients, strict=True):
            scale = expected_gradient.abs().amax()
            assert torch.isfinite(actual_gradient).all()
            assert scale > 0 and torch.count_nonzero(actual_gradient) > 0
            torch.testing.assert_close(
                actual_gradient / scale,
                expected_gradient / scale,
                rtol=4e-3,
                atol=4e-4,
            )
        assert runner.kernel.execution_report().strategy == "compiled_scan"
        torch.testing.assert_close(functional.extract().state, native)


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize("network", (False, True))
def test_cpu_inductor_bound_packets_preserve_checkpointed_training_and_tails(network):
    _training(network, "cpu", "inductor")


@pytest.mark.mps
@pytest.mark.parametrize("network", (False, True))
def test_mps_aot_bound_packets_preserve_checkpointed_training_and_tails(
    network, mps_device
):
    _training(network, mps_device, "aot_eager")


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("network", (False, True))
def test_mps_inductor_while_bound_packets_preserve_recording_and_inference_cache(
    network, mps_device
):
    with _environment(mps_device):
        runner = _compile(network, mps_device, "inductor", inference=True)
        functional = runner.functional
        bound = runner.kernel.bind(runner.parameters, constants=runner.constants)
        native = _snapshot(functional.extract().state)
        current, inputs = _drive(bound, network, 8)
        with torch.inference_mode():
            prefix, first = dn.func.run(
                bound,
                runner.initial_state,
                inputs=_inputs(network, current[:4]),
                steps=4,
                callbacks=runner.callbacks,
            )
            cached = None if network else bound._inference_prepared
            actual, last = dn.func.run(
                bound,
                first["continuation"],
                inputs=_inputs(network, current[4:]),
                steps=4,
                callbacks=runner.callbacks,
            )
            if not network:
                assert cached is not None and bound._inference_prepared is cached
            expected, reference = functional.prepare_and_rollout(
                first["continuation"].parameters,
                first["continuation"].constants,
                runner.initial_state,
                inputs,
                steps=8,
                callbacks=runner.callbacks,
            )
            repeated, _ = dn.func.run(
                bound,
                runner.initial_state,
                inputs=_inputs(network, current[:4]),
                steps=4,
                callbacks=runner.callbacks,
            )
        trace = _joined(first, last)
        _assert_result(
            actual,
            expected,
            trace,
            reference["callbacks"]["trace"],
            network,
            mps_device,
            8,
        )
        torch.testing.assert_close(repeated, prefix, rtol=3e-5, atol=2e-5)
        assert last["callbacks"]["trace"]["left.v" if network else "v"].shape[0] == 4
        assert not _voltage(actual, network).requires_grad
        assert runner.kernel.execution_report().strategy == "compiled_while_loop"
        torch.testing.assert_close(functional.extract().state, native)
