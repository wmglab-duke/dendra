"""Shared packets preserve real CPU/MPS training and while-loop continuation."""

from __future__ import annotations

import os

import pytest
import torch

from tests.func.test_continuation import _drive, _joined, _runner, _snapshot
from tests.func.test_lowering_api import G, _target, _voltage
from tests.func.test_runtime_extra_backends import _environment


@pytest.fixture
def mps_device():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    return "mps"


def _compile(network, device, backend, *, inference=False):
    options = {}
    if backend == "inductor":
        options = {"compile_threads": 1}
        if device == "mps":
            options["max_fusion_unique_io_buffers"] = 30
    return _runner(
        network,
        steps=4 if inference else 2,
        inference=inference,
        execution="while" if inference else "scan",
        backend=backend,
        disable=False,
        checkpoint_every=None if inference else 1,
        options=options,
    )


def _loss(state, trace, network):
    return _voltage(state, network).square().mean() + sum(
        value.square().mean() for name, value in trace.items() if name != "t"
    )


def _assert_result(actual, expected, auxiliary, reference, network, device, steps):
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(auxiliary, dict(reference), rtol=3e-5, atol=2e-5)
    voltage = _voltage(actual, network)
    assert voltage.device.type == device and voltage.dtype is torch.float32
    assert all(value.shape[0] == steps + 1 for value in auxiliary.values())


def _inputs(network, current):
    import dendra as dn

    packet = dn.func.Inputs(intra=current)
    return {"right": packet} if network else packet


def _training_case(network, device, backend):
    with _environment(device):
        runner = _compile(network, device, backend)
        original = _snapshot(runner.functional.extract().state)
        current, inputs = _drive(runner, network, 4, requires_grad=True)
        first, auxiliary = runner(inputs=_inputs(network, current[:2]))
        continuation = auxiliary["continuation"]
        conductance = _target(continuation.parameters, network)
        assert conductance is _target(runner.parameters, network)
        if network:
            assert conductance is continuation.parameters["populations"]["right"][G]
        actual, last = runner(continuation, inputs=_inputs(network, current[2:]))
        trace = _joined(auxiliary, last)
        actual_gradients = torch.autograd.grad(
            _loss(actual, trace, network), (conductance, current), retain_graph=True
        )
        expected, reference = runner.functional.prepare_and_rollout(
            continuation.parameters,
            continuation.constants,
            runner.initial_state,
            inputs,
            steps=4,
            callbacks=runner.callbacks,
        )
        reference_gradients = torch.autograd.grad(
            _loss(expected, reference["callbacks"]["trace"], network),
            (conductance, current),
        )
        _assert_result(
            actual, expected, trace, reference["callbacks"]["trace"], network, device, 4
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, reference_gradients, strict=True
        ):
            assert torch.isfinite(actual_gradient).all()
            scale = expected_gradient.abs().amax()
            assert scale > 0 and torch.count_nonzero(actual_gradient) > 0
            torch.testing.assert_close(
                actual_gradient / scale, expected_gradient / scale, rtol=4e-3, atol=4e-4
            )
        assert (
            auxiliary["callbacks"]["trace"]["left.v" if network else "v"].shape[0] == 3
        )
        assert last["callbacks"]["trace"]["left.v" if network else "v"].shape[0] == 2
        assert runner.execution_report().strategy == "compiled_scan"
        torch.testing.assert_close(runner.functional.extract().state, original)


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize("network", (False, True))
def test_cpu_inductor_packet_continuation_spans_training_graphs(network):
    _training_case(network, "cpu", "inductor")


@pytest.mark.mps
@pytest.mark.parametrize("network", (False, True))
def test_mps_aot_packet_continuation_spans_training_graphs(network, mps_device):
    _training_case(network, mps_device, "aot_eager")


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("network", (False, True))
def test_mps_inductor_while_packet_resumes_recording_and_reuses_the_existing_cache_policy(
    network, mps_device
):
    with _environment(mps_device):
        runner = _compile(network, mps_device, "inductor", inference=True)
        original = _snapshot(runner.functional.extract().state)
        current, inputs = _drive(runner, network, 8)
        prefix, first = runner(inputs=_inputs(network, current[:4]))
        continuation = first["continuation"]
        cached = None if network else runner._prepared
        actual, last = runner(continuation, inputs=_inputs(network, current[4:]))
        if not network:
            assert runner._prepared is cached and cached is not None
        with torch.inference_mode():
            expected, reference = runner.functional.prepare_and_rollout(
                continuation.parameters,
                continuation.constants,
                runner.initial_state,
                inputs,
                steps=8,
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
        assert last["callbacks"]["trace"]["left.v" if network else "v"].shape[0] == 4
        assert not _voltage(actual, network).requires_grad
        assert runner.execution_report().strategy == "compiled_while_loop"
        torch.testing.assert_close(runner.functional.extract().state, original)
        repeated, _ = runner(inputs=_inputs(network, current[:4]))
        torch.testing.assert_close(repeated, prefix, rtol=3e-5, atol=2e-5)
