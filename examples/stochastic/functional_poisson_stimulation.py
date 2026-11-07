#!/usr/bin/env python3
"""Replay randomized stimulation with explicit draws and independent RNG carry."""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import pas
from dendra.models.stim.waveform import Waveform
from dendra.units import nA

DT = 0.025
STEPS = 8
DTYPE = torch.float64


# region custom-waveform
class OrderedPoissonWaveform(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETER(gain=0.7)

    def __init__(self, first, second, **kwargs):
        super().__init__(**kwargs)
        self.first, self.second = first, second

    def fn(self, times):
        second = self.second.fn(times)
        first = self.first.fn(times)
        repeated = self.first.fn(times)
        return self.gain * (second + first + 0.25 * repeated)


# endregion custom-waveform


# region repeated-mechanism
class TwiceDrivenLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.ASSIGNED("i_inj")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def assigned_values(self, v, values):
        del values
        first = self.injection_values(v, t=self.t)
        second = self.injection_values(v, t=self.t)
        return {"i_inj": first + 0.25 * second}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * self.i_inj


# endregion repeated-mechanism


def make_population(*, mechanism=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        population = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(TwiceDrivenLeak if mechanism else pas, g=0.001, e=-65.0)
        # region waveform
        generator = torch.Generator().manual_seed(17)
        pulse = dn.mono_rect(amp=0.2 if mechanism else 0.2 * nA, pw=0.045)
        first = pulse.poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=generator,
        )
        second = pulse.poisson(
            interval=0.08,
            n=5,
            start=-0.02,
            noise=0.7,
            randomize_every_call=True,
            generator=generator,
        )
        waveform = OrderedPoissonWaveform(first, second)
        population[:, 1].inject(waveform)
        # endregion waveform
        population.train()
        population.initialize()
        population.detach()
    return population, waveform


def run_demo(*, mechanism=False):
    population, waveform = make_population(mechanism=mechanism)
    native, native_waveform = make_population(mechanism=mechanism)
    source_rng = waveform.first.generator.get_state().clone()
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
    gain = next(
        value for name, value in tensors.parameters.items() if name.endswith(".gain")
    )
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().mean(), (amplitude, gain)
    )
    expected = torch.autograd.grad(
        native.v.square().mean(),
        (native_waveform.first.waveform.amp, native_waveform.gain),
    )
    torch.testing.assert_close(actual, expected)
    assert all(gradient.abs().sum() > 0 for gradient in actual)
    # endregion gradients

    # region compiled
    runner = dn.compile(
        population,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*waveform.amp", "*gain"),
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
    torch.testing.assert_close(waveform.first.generator.get_state(), source_rng)
    torch.testing.assert_close(torch.get_rng_state(), global_rng)
    torch.testing.assert_close(prefix["integrator"]["v"], final["integrator"]["v"])
    assert torch.isfinite(continued["integrator"]["v"]).all()
    owner = "Mechanism" if mechanism else "Intracellular"
    print(
        f"{owner} Poisson invocations: replay, gradients, compiled continuation, "
        "and RNG isolation passed."
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_demo()
    run_demo(mechanism=True)
