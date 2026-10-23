#!/usr/bin/env python3
"""Replay randomized stimulation with explicit draws and independent RNG carry."""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas
from dendra.units import nA

DT = 0.025
STEPS = 8
DTYPE = torch.float64


def make_population():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        population = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(pas, g=0.001, e=-65.0)
        # region waveform
        waveform = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=torch.Generator().manual_seed(17),
        )
        population[:, 1].inject(waveform)
        # endregion waveform
        population.train()
        population.initialize()
        population.detach()
    return population, waveform


def run_demo():
    population, waveform = make_population()
    native, native_waveform = make_population()
    source_rng = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    # region explicit-draws
    functional, tensors = dn.func.make_functional(population, dt=DT)
    draws = functional.sample_stimulation_draw_tape(tensors.state, STEPS)
    inputs = dn.func.RolloutInput(stimulation=draws)
    final, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        steps=STEPS,
    )
    replay, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        steps=STEPS,
    )
    torch.testing.assert_close(final, replay)
    # endregion explicit-draws
    for _ in range(STEPS):
        native.step(DT)
    torch.testing.assert_close(final["integrator"]["v"], native.v)

    # region gradients
    amplitude = next(
        value
        for name, value in tensors.parameters.items()
        if name.endswith("waveform.amp")
    )
    actual = torch.autograd.grad(final["integrator"]["v"].square().mean(), amplitude)[0]
    expected = torch.autograd.grad(
        native.v.square().mean(), native_waveform.waveform.amp
    )[0]
    torch.testing.assert_close(actual, expected)
    assert actual.abs().sum() > 0
    # endregion gradients

    # region compiled
    runner = dn.compile(
        population,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*waveform.amp",),
        checkpoint_every=4,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        backend="aot_eager",
    )
    prefix, first = runner()
    restarted, _ = runner()
    torch.testing.assert_close(prefix, restarted)
    continued, second = runner(prefix, callback_state=first["callbacks"].state)
    assert second["callbacks"]["trace"]["t"][-1] > first["callbacks"]["trace"]["t"][-1]
    # endregion compiled
    torch.testing.assert_close(waveform.generator.get_state(), source_rng)
    torch.testing.assert_close(torch.get_rng_state(), global_rng)
    torch.testing.assert_close(prefix["integrator"]["v"], final["integrator"]["v"])
    assert torch.isfinite(continued["integrator"]["v"]).all()
    print(
        "Explicit Poisson replay, gradients, compiled continuation, and RNG isolation passed."
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_demo()
