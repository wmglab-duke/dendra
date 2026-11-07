"""Compiled SDE scans own tape views across training and continuation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_state_sde_execution import (
    DT,
    STEPS,
    _clone,
    _close,
    _HeunLeak,
    _OULeak,
    _population,
    _slice,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _objective(state, recorded):
    return (
        0.001 * state["integrator"]["v"].square().mean()
        + state["mechanisms"]["sde"]["x"].square().mean()
        + recorded["sde.x"].square().mean()
    )


@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_inductor_sde_training_tape_views_recording_and_checkpoint_continuation(
    mechanism, checkpoint_every
):
    source, native = _population(mechanism), _population(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*sigma_param",),
        execution="scan",
        backend="inductor",
        checkpoint_every=checkpoint_every,
        callbacks={"trace": dn.func.Recorder(("v", "sde.x"))},
    )
    functional = runner.functional
    source_before = _clone(functional.extract())
    snapshot = _clone(runner.initial_state)
    global_rng = torch.get_rng_state().clone()
    tape = functional.sample_noise_draw_tape(snapshot, 2 * STEPS)
    saved_tape = _clone(tape)
    prefix_input = dn.func.RolloutInput(noise=_slice(tape, 0, STEPS))
    prefix, first = runner(inputs=prefix_input)
    # A call without state repeats the snapshot, including its RNG frame.
    restarted, restarted_aux = runner(inputs=prefix_input)
    _close(restarted, prefix)
    _close(restarted_aux["callbacks"]["trace"], first["callbacks"]["trace"])
    continued, second = runner(
        prefix,
        inputs=dn.func.RolloutInput(noise=_slice(tape, STEPS, 2 * STEPS)),
        callback_state=first["callbacks"].state,
    )
    recorded = {
        name: torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        for name in ("v", "sde.x")
    }
    expected = {"v": [native.v.clone()], "sde.x": [native.mech.sde.x.clone()]}
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step(DT)
            expected["v"].append(native.v.clone())
            expected["sde.x"].append(native.mech.sde.x.clone())
    for name, values in expected.items():
        _close(recorded[name], torch.stack(values))
    _close(continued, functional.extract(native).state)

    # Compare both the state objective and gradients through the Recorder with
    # an eager trajectory using exactly the same draws and RNG-state tape.
    sigma = runner.get_parameter("*sigma_param")
    sigma_name = next(
        name for name in runner.parameters if name.endswith("sigma_param")
    )
    eager_sigma = sigma.detach().clone().requires_grad_()
    parameters = {**runner.parameters, sigma_name: eager_sigma}
    prepared = functional.prepare(parameters, runner.constants)
    eager, eager_aux = functional.rollout(
        parameters,
        prepared,
        snapshot,
        steps=2 * STEPS,
        inputs=dn.func.RolloutInput(noise=tape),
        callbacks=functional.make_callbacks(
            {"trace": dn.func.Recorder(("v", "sde.x"))}
        ),
    )
    _close(continued, eager)
    _close(recorded, eager_aux["callbacks"]["trace"])
    actual_gradient = torch.autograd.grad(_objective(continued, recorded), sigma)[0]
    eager_gradient = torch.autograd.grad(
        _objective(eager, eager_aux["callbacks"]["trace"]), eager_sigma
    )[0]
    assert torch.isfinite(actual_gradient).all()
    assert actual_gradient.abs().sum() > 0
    _close(actual_gradient, eager_gradient)
    _close(tape, saved_tape)
    _close(runner.initial_state, snapshot)
    _close(functional.extract(), source_before)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    assert runner.execution_report().strategy == "compiled_scan"
    assert runner.execution_report().grad_enabled
