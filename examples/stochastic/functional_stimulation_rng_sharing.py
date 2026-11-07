#!/usr/bin/env python3
"""Replay Poisson stimulation and model NOISE from one shared generator.

Run from the repository root:
    python examples/stochastic/functional_stimulation_rng_sharing.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.units import nA

DT = 0.025
STEPS = 8
DTYPE = torch.float64


class NoisyLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=17, rng_name="stream")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.eta


def make_population():
    population = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-45.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    population.insert(NoisyLeak.rename("leak"))
    population.train()
    population.initialize()
    # region borrowed-generator
    stream = population.mech.leak.stream
    waveform = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=stream.rng,
    )
    population[:, 1].inject(waveform)
    # endregion borrowed-generator
    population.initialize()
    population.detach()
    return population, waveform


def run_demo():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        population, waveform = make_population()
        native, native_waveform = make_population()
        source_rng = waveform.generator.get_state().clone()
        global_rng = torch.get_rng_state().clone()
        functional, tensors = dn.func.make_functional(population, dt=DT)
        # region paired-tapes
        noise = functional.sample_noise_draw_tape(tensors.state, STEPS)
        stimulation = functional.sample_stimulation_draw_tape(tensors.state, STEPS)
        inputs = dn.func.RolloutInput(noise=noise, stimulation=stimulation)
        final, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs,
            steps=STEPS,
        )
        # endregion paired-tapes
        replay, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs,
            steps=STEPS,
        )
        torch.testing.assert_close(final, replay)
        for _ in range(STEPS):
            native.step(DT)
        torch.testing.assert_close(final["integrator"]["v"], native.v)
        # region authoritative-carry
        assert len(final["random"]["rng"]) == 1
        assert not final["stimulation"].get("rng", {})
        frame = next(iter(final["random"]["rng"].values()))
        torch.testing.assert_close(frame["cpu"], native_waveform.generator.get_state())
        # endregion authoritative-carry
        amplitude = next(
            value
            for name, value in tensors.parameters.items()
            if name.endswith("waveform.amp")
        )
        actual = torch.autograd.grad(
            final["integrator"]["v"].square().mean(), amplitude
        )[0]
        expected = torch.autograd.grad(
            native.v.square().mean(), native_waveform.waveform.amp
        )[0]
        torch.testing.assert_close(actual, expected)
        assert actual.abs().sum() > 0

        # region compiled-continuation
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
        continued, second = runner(prefix, callback_state=first["callbacks"].state)
        # endregion compiled-continuation
        torch.testing.assert_close(prefix, final)
        for _ in range(STEPS):
            native.step(DT)
        torch.testing.assert_close(continued["integrator"]["v"], native.v)
        assert (
            second["callbacks"]["trace"]["t"][-1] > first["callbacks"]["trace"]["t"][-1]
        )
        torch.testing.assert_close(waveform.generator.get_state(), source_rng)
        torch.testing.assert_close(torch.get_rng_state(), global_rng)
        print(
            "Shared Poisson/NOISE stream: native draw order, replay, gradients, "
            "single RNG carry, and compiled continuation passed."
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_demo()
