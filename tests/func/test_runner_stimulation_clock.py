"""Compiled host chunks must sample bound waveforms at the recurrent clock."""

from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from dendra.units import nA

DT = 0.01
STEPS = 8


def _case(stimulus_kind):
    # Repeated float32 additions reach this pulse; start + arange * dt misses
    # it. A pulse at the final step also makes the extracellular effect visible
    # before subsequent axial relaxation can hide a sampling error.
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float32):
        model = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float32,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-4, e=-65.0)
        waveform = dn.mono_rect(
            amp=0.2 * nA if stimulus_kind == "intra" else 1.0,
            delay=1000.070068359375,
            pw=0.001,
        )
        extra = None
        if stimulus_kind == "intra":
            model[:, 1].inject(waveform)
        else:
            field = torch.linspace(-1.0, 1.0, model.v.numel()).reshape(model.shape)
            extra = (field, waveform)
        model.initialize()
        model.t.fill_(1000.0)
        model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    amplitude_name = next(
        name
        for name in parameters
        if name.startswith(f"stimulation.{stimulus_kind}.") and name.endswith(".amp")
    )
    amplitude = parameters[amplitude_name].requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    return functional, tensors, parameters, prepared, amplitude


@pytest.mark.parametrize("stimulus_kind", ["intra", "extra"])
def test_compiled_runner_bound_waveforms_preserve_clock_and_gradients(stimulus_kind):
    functional, tensors, parameters, prepared, amplitude = _case(stimulus_kind)
    expected = tensors.state
    for _ in range(STEPS):
        expected, _auxiliary = functional.step(parameters, prepared, expected)
    expected_voltage = expected["integrator"]["v"]
    expected_gradient = torch.autograd.grad(
        expected_voltage.square().mean(), amplitude
    )[0]
    assert torch.isfinite(expected_gradient).all()
    assert expected_gradient.abs().item() > 0.0

    chunk = functional.compile_rollout_chunk(STEPS, backend="aot_eager")
    step = partial(chunk, parameters, prepared)
    with torch_compiler_warning_context():
        for runner in (dn.func.run, dn.func.longrun, dn.func.longrun_checkpointed):
            if runner is dn.func.run:
                actual, _auxiliary = runner(
                    functional, step, tensors.state, tstop=STEPS * DT
                )
            else:
                actual, _auxiliary = runner(
                    functional, step, tensors.state, STEPS * DT, STEPS
                )
            actual_voltage = actual["integrator"]["v"]
            actual_gradient = torch.autograd.grad(
                actual_voltage.square().mean(), amplitude
            )[0]
            torch.testing.assert_close(actual_voltage, expected_voltage)
            torch.testing.assert_close(actual_gradient, expected_gradient)
            torch.testing.assert_close(actual["clock"]["t"], expected["clock"]["t"])


@pytest.mark.parametrize("stimulus_kind", ["intra", "extra"])
def test_compiled_runner_callbacks_do_not_change_bound_pulse_sampling(stimulus_kind):
    functional, tensors, parameters, prepared, _amplitude = _case(stimulus_kind)
    chunk = functional.compile_rollout_chunk(STEPS, backend="eager")
    step = partial(chunk, parameters, prepared)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})

    with torch.no_grad(), torch_compiler_warning_context():
        ordinary, _auxiliary = dn.func.run(
            functional, step, tensors.state, tstop=STEPS * DT
        )
        recorded, auxiliary = dn.func.run(
            functional,
            step,
            tensors.state,
            tstop=STEPS * DT,
            callbacks=callbacks,
        )
    trace = auxiliary["callbacks"]["trace"]
    assert trace["v"].shape[0] == STEPS + 1
    torch.testing.assert_close(ordinary["integrator"]["v"], recorded["integrator"]["v"])
    torch.testing.assert_close(trace["v"][-1], ordinary["integrator"]["v"])
