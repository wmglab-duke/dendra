"""Small real-backend runtime-extra checks with independently assembled drives.

Run the MPS cases with PYTORCH_ENABLE_MPS_FALLBACK=0 on Apple hardware.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas

DT = 0.01
STEPS = 8
CHUNK_STEPS = 4


@contextmanager
def _environment(device):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float32, DEVICE=device),
        ):
            yield
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture
def mps_device():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", (
        "Run MPS backend tests with PYTORCH_ENABLE_MPS_FALLBACK=0"
    )
    return "mps"


def _model(device):
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-65.0,
        device=device,
        dtype=torch.float32,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas, g=1.0e-4, e=-65.0)
    model.initialize()
    model.t.fill_(0.175)
    model.train()
    return model


def _field(model):
    positions = torch.linspace(
        -1.0, 1.0, model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(model.shape)
    return positions.square() + 0.2 * positions


def _runner(model, *, execution="scan", backend="inductor", inference=True):
    keywords = {}
    if backend == "inductor":
        keywords["options"] = {
            "compile_threads": 1,
            "cpp_wrapper": model.device().type == "cpu",
        }
        if model.device().type == "mps":
            keywords["options"]["max_fusion_unique_io_buffers"] = 30
    return dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        extra=dn.VarInput(),
        chunk_steps=CHUNK_STEPS,
        execution=execution,
        inference=inference,
        checkpoint_every=None if inference else 7,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend=backend,
        **keywords,
    )


def _tape(runner, extra):
    # Evaluate the authored recipes directly at each recurrent pre-step clock;
    # this reference does not use FunctionalExtra or the runner's assembly.
    clock = runner.initial_state["clock"]["t"]
    dt = clock.new_tensor(DT)
    frames = []
    for _ in range(STEPS):
        frame = torch.zeros_like(runner.initial_state["integrator"]["v"])
        if extra is not None:
            for field, waveform in extra:
                frame = frame + field * waveform(clock)
        frames.append(frame)
        clock = clock + dt
    return torch.stack(frames)


def _reference(runner, extra):
    functional = runner.functional
    prepared = functional.prepare(runner.parameters, runner.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    return dn.func.run(
        functional.bind(runner.parameters, prepared),
        runner.initial_state,
        steps=STEPS,
        inputs=dn.func.RolloutInput(ve=_tape(runner, extra)),
        callbacks=callbacks,
    )


def _trace(result):
    return dict(result[1]["callbacks"]["trace"])


def _assert_result(actual, expected, device):
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(_trace(actual), _trace(expected), rtol=3e-5, atol=2e-5)
    for value in _trace(actual).values():
        assert value.shape[0] == STEPS + 1
        assert value.device.type == device
        assert value.dtype == torch.float32
    torch.testing.assert_close(
        actual[0]["integrator"]["v"], _trace(actual)["v"][-1], rtol=0, atol=0
    )


def _inference_case(device, execution):
    with _environment(device):
        model = _model(device)
        runner = _runner(model, execution=execution)
        first = [(_field(model), dn.sin(amp=0.3, freq=0.9, phase=0.2))]
        second = [
            (_field(model).flip(-1), dn.cos(amp=-0.2, freq=0.55, phase=-0.15)),
            (_field(model) * 0.4, dn.mono_rect(amp=0.25, delay=0.19, pw=0.04)),
        ]
        results = []
        for extra in (first, second, None):
            actual = runner(extra=extra)
            with torch.no_grad():
                expected = _reference(runner, extra)
            _assert_result(actual, expected, device)
            assert not actual[0]["integrator"]["v"].requires_grad
            assert runner.execution_report().strategy == (
                "compiled_scan" if execution == "scan" else "compiled_while_loop"
            )
            results.append(actual)
        _assert_result(runner(extra=first), results[0], device)
        assert not torch.equal(_trace(results[0])["v"], _trace(results[2])["v"])
        assert not torch.equal(_trace(results[1])["v"], _trace(results[2])["v"])


@pytest.mark.cpu
@pytest.mark.slow
def test_cpu_inductor_runtime_extra_recipes_and_zero_match_explicit_drives():
    _inference_case("cpu", "scan")


@pytest.mark.mps
@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_mps_inductor_runtime_extra_recipes_and_zero_match_explicit_drives(
    mps_device, execution
):
    _inference_case(mps_device, execution)


def _differentiable_extra(model, scale_value):
    scale = model.v.new_tensor(scale_value, requires_grad=True)
    first = dn.sin(amp=0.3, freq=0.9, phase=0.2)
    second = dn.cos(amp=-0.2, freq=0.55, phase=-0.15)
    extra = [
        (_field(model) * scale, first),
        (_field(model).flip(-1) * 0.4, second),
    ]
    return extra, (scale, first.amp, first.freq, first.phase, second.amp)


def _gradients(result, targets):
    voltage = _trace(result)["v"]
    weights = torch.linspace(
        0.7, 1.3, voltage.shape[-1], device=voltage.device, dtype=voltage.dtype
    )
    loss = (voltage * weights).square().mean() + 0.1 * voltage.sin().mean()
    gradients = torch.autograd.grad(loss, targets)
    for value in gradients:
        assert value.device.type == "mps"
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) > 0
    return gradients


@pytest.mark.mps
def test_mps_aot_eager_runtime_extra_checkpoint_gradients_match_explicit_drives(
    mps_device,
):
    with _environment(mps_device):
        model = _model(mps_device)
        runner = _runner(model, backend="aot_eager", inference=False)
        for scale_value in (0.8, 1.2):
            extra, targets = _differentiable_extra(model, scale_value)
            actual = runner(extra=extra)
            actual_gradients = _gradients(actual, targets)
            reference_extra, reference_targets = _differentiable_extra(
                model, scale_value
            )
            expected = _reference(runner, reference_extra)
            expected_gradients = _gradients(expected, reference_targets)
            _assert_result(actual, expected, mps_device)
            for actual_gradient, expected_gradient in zip(
                actual_gradients, expected_gradients, strict=True
            ):
                scale = expected_gradient.abs().amax()
                torch.testing.assert_close(
                    actual_gradient / scale,
                    expected_gradient / scale,
                    rtol=4e-3,
                    atol=4e-4,
                )
