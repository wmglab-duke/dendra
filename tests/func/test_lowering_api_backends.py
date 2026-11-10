"""Canonical Network lowering retains real CPU/MPS execution and continuation."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn, pas
from tests.func.test_inputs_backends import _snapshot
from tests.func.test_lowering_api import DT, G, _network
from tests.func.test_runtime_extra_backends import _environment


@pytest.fixture
def mps_device():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    return "mps"


def _lower(device, *, inference=False):
    model = _network()
    if inference:
        model.eval()
        model.initialize(DT)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert type(functional) is dn.func.FunctionalNetwork
    assert type(tensors) is dn.func.NetworkTensors
    assert functional.device.type == device
    for population in functional.populations.values():
        assert population.dtype is torch.float32
        assert population.device.type == device
    shared = tensors.parameters["populations"]["left"][G]
    assert shared is tensors.parameters["populations"]["right"][G]
    assert shared is not model.left.mech.leak.g_param
    assert shared.data_ptr() != model.left.mech.leak.g_param.data_ptr()
    return model, functional, tensors


def _callbacks(functional):
    return functional.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    )


def _kernel(functional, backend, *, execution="scan", steps=2):
    keywords = {}
    if backend == "inductor":
        keywords["options"] = {"compile_threads": 1}
        if functional.device.type == "mps":
            keywords["options"]["max_fusion_unique_io_buffers"] = 30
    return functional.compile_rollout_chunk(
        steps, execution=execution, backend=backend, **keywords
    )


def _inputs(model, steps, *, requires_grad=False):
    current = torch.linspace(
        -2e-6,
        4e-6,
        steps * model.right.v.numel(),
        device=model.right.device(),
        dtype=model.right.dtype(),
    ).reshape(steps, *model.right.shape)
    current.requires_grad_(requires_grad)
    return current, {"right": dn.func.Inputs(intra=current)}


def _joined(prefix, suffix):
    return {
        name: torch.cat((prefix["callbacks"]["trace"][name], value))
        for name, value in suffix["callbacks"]["trace"].items()
    }


def _loss(state, trace):
    return sum(
        value["integrator"]["v"].square().mean()
        for value in state["populations"].values()
    ) + sum(trace[name].square().mean() for name in ("left.v", "right.v"))


def _assert_state(actual, expected, device):
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    for population in actual["populations"].values():
        voltage = population["integrator"]["v"]
        assert voltage.device.type == device
        assert voltage.dtype is torch.float32


def _assert_trace(actual, expected, device, steps):
    torch.testing.assert_close(actual, dict(expected), rtol=3e-5, atol=2e-5)
    for name, values in actual.items():
        assert values.shape[0] == steps + 1
        if name != "t":
            assert values.device.type == device
            assert values.dtype is torch.float32


def _training_case(device, backend):
    with _environment(device):
        model, functional, tensors = _lower(device)
        original = _snapshot(functional.extract())
        parameters = tensors.independent_parameters(trainable="*g_param")
        conductance = parameters["populations"]["left"][G]
        assert conductance is parameters["populations"]["right"][G]
        current, inputs = _inputs(model, 4, requires_grad=True)
        frozen_inputs = _snapshot(inputs)
        callbacks = _callbacks(functional)
        kernel = _kernel(functional, backend)
        bound = kernel.bind(parameters, constants=tensors.constants)
        prefix, first = bound(
            tensors.state,
            {"right": dn.func.Inputs(intra=current[:2])},
            callbacks=callbacks,
        )
        actual, last = bound(
            prefix,
            {"right": dn.func.Inputs(intra=current[2:])},
            callbacks=callbacks,
            callback_state=first["callbacks"].state,
        )
        trace = _joined(first, last)
        _loss(actual, trace).backward()
        expected, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            inputs,
            steps=4,
            callbacks=callbacks,
        )
        reference = torch.autograd.grad(
            _loss(expected, auxiliary["callbacks"]["trace"]), (conductance, current)
        )
        _assert_state(actual, expected, device)
        _assert_trace(trace, auxiliary["callbacks"]["trace"], device, 4)
        assert first["callbacks"]["trace"]["left.v"].shape[0] == 3
        assert last["callbacks"]["trace"]["left.v"].shape[0] == 2
        for target, expected_gradient in zip(
            (conductance, current), reference, strict=True
        ):
            observed = target.grad
            assert observed is not None and observed.device.type == device
            assert torch.isfinite(observed).all()
            scale = expected_gradient.abs().amax()
            assert scale > 0 and torch.count_nonzero(observed) > 0
            torch.testing.assert_close(
                observed / scale, expected_gradient / scale, rtol=4e-3, atol=4e-4
            )
        report = kernel.execution_report()
        assert report.strategy == "compiled_scan" and report.callbacks
        torch.testing.assert_close(functional.extract(), original)
        torch.testing.assert_close(inputs, frozen_inputs)


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_mps_inductor_netstim_drives_expsyn(execution, mps_device):
    dt, steps = 0.125, 8
    with _environment(mps_device):
        cell = dn.SingleCompartment(N=1, C=1, v_init=-65.0)
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("exc"), e=0.0, tau=2.0)
        stimulus = dn.NetStim(
            N=1, start=dt, interval=2 * dt, noise=0, max_spikes=2, seed=0
        )
        network = dn.Network({"cell": cell}, netstim=stimulus).eval()
        network.connect_one_to_one(
            stimulus[0], cell[0, 0], cell.mech.exc, weight=0.01, delay=dt
        )
        network.initialize(dt)
        program = dn.compile(
            network, dt=dt, steps=steps, execution=execution, backend="inductor"
        )
        with torch.inference_mode():
            state, _ = program()
            initial_voltage = cell.v.clone()
            network.run(steps=steps)

        voltage = state["populations"]["cell"]["integrator"]["v"]
        assert voltage.device.type == mps_device and voltage.dtype is torch.float32
        assert not torch.equal(cell.v, initial_voltage)
        torch.testing.assert_close(voltage, cell.v, rtol=3e-5, atol=2e-5)
        assert stimulus.spike_counts.item() == 2
        torch.testing.assert_close(
            state["netstim"]["spike_counts"], stimulus.spike_counts
        )
        torch.testing.assert_close(state["clock"]["t"], network.t)
        assert state["clock"]["t"].device.type == "cpu"
        assert state["clock"]["t"].dtype is torch.float64
        assert program.execution_report().strategy == (
            "compiled_scan" if execution == "scan" else "compiled_while_loop"
        )


@pytest.mark.cpu
@pytest.mark.slow
def test_cpu_inductor_canonical_network_lowering_retains_scan_gradients():
    _training_case("cpu", "inductor")


@pytest.mark.mps
def test_mps_aot_eager_canonical_network_lowering_retains_scan_gradients(mps_device):
    _training_case(mps_device, "aot_eager")


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_mps_inductor_canonical_network_lowering_resumes_inference_recording(
    mps_device, execution
):
    with _environment(mps_device):
        model, functional, tensors = _lower(mps_device, inference=True)
        original = _snapshot(functional.extract())
        parameters = tensors.independent_parameters()
        assert (
            parameters["populations"]["left"][G]
            is parameters["populations"]["right"][G]
        )
        current, inputs = _inputs(model, 8)
        frozen_inputs = _snapshot(inputs)
        callbacks = _callbacks(functional)
        kernel = _kernel(functional, "inductor", execution=execution, steps=4)
        bound = kernel.bind(parameters, constants=tensors.constants)
        with torch.inference_mode():
            first_inputs = {"right": dn.func.Inputs(intra=current[:4])}
            prefix, first = bound(tensors.state, first_inputs, callbacks=callbacks)
            repeated, _ = bound(tensors.state, first_inputs, callbacks=callbacks)
            final, suffix = bound(
                prefix,
                {"right": dn.func.Inputs(intra=current[4:])},
                callbacks=callbacks,
                callback_state=first["callbacks"].state,
            )
            expected, auxiliary = functional.prepare_and_rollout(
                parameters,
                tensors.constants,
                tensors.state,
                inputs,
                steps=8,
                callbacks=callbacks,
            )
        _assert_state(prefix, repeated, mps_device)
        _assert_state(final, expected, mps_device)
        trace = _joined(first, suffix)
        _assert_trace(trace, auxiliary["callbacks"]["trace"], mps_device, 8)
        assert first["callbacks"]["trace"]["right.v"].shape[0] == 5
        assert suffix["callbacks"]["trace"]["right.v"].shape[0] == 4
        assert not final["populations"]["right"]["integrator"]["v"].requires_grad
        report = kernel.execution_report()
        assert report.strategy == (
            "compiled_while_loop" if execution == "while" else "compiled_scan"
        )
        assert report.callbacks
        torch.testing.assert_close(functional.extract(), original)
        torch.testing.assert_close(inputs, frozen_inputs)
