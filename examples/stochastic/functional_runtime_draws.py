#!/usr/bin/env python3
"""Replay canonical runtime draws for Populations and Networks.

Run from the repository root::

    python examples/stochastic/functional_runtime_draws.py
    python examples/stochastic/functional_runtime_draws.py --backend aot_eager

The example shares a generator between model NOISE and randomized Poisson
stimulation. It checks native trajectories, explicit/automatic execution,
checkpoint continuation, recordings, gradients, and source RNG immutability.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.units import nA

DT = 0.025
STEPS = 4
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
    pulse = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=population.mech.leak.stream.rng,
    )
    population[:, 1].inject(pulse)
    population.initialize()
    population.detach()
    return population, pulse


def make_network():
    left, left_pulse = make_population()
    right, right_pulse = make_population()
    right.mech.leak.stream = left.mech.leak.stream
    right_pulse.generator = left.mech.leak.stream.rng
    network = dn.Network({"left": left, "right": right})
    network.train()
    network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network, (left_pulse, right_pulse)


def clone_tree(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def slice_draws(value, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda tensor: tensor[start:stop], value)


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)


def population_demo():
    source, pulse = make_population()
    native, native_pulse = make_population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    original = clone_tree(tensors.state)
    source_rng = pulse.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()

    # region sample-population
    draws = functional.sample_runtime_draws(tensors.state, steps=2 * STEPS)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.bind(tensors.parameters, prepared)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    prefix, first = dn.func.run(
        bound,
        tensors.state,
        draws=slice_draws(draws, 0, STEPS),
        steps=STEPS,
        checkpoint_every=2,
        callbacks=callbacks,
    )
    # endregion sample-population
    # region continuation
    final, second = dn.func.run(
        bound,
        prefix,
        draws=slice_draws(draws, STEPS, 2 * STEPS),
        steps=STEPS,
        checkpoint_every=2,
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    continued_draws = functional.sample_runtime_draws(prefix, steps=STEPS)
    assert_close(continued_draws, slice_draws(draws, STEPS, 2 * STEPS))
    # endregion continuation
    automatic, automatic_aux = dn.func.run(
        bound, tensors.state, steps=2 * STEPS, checkpoint_every=2, callbacks=callbacks
    )
    assert_close(final, automatic)
    for name, values in automatic_aux["callbacks"]["trace"].items():
        joined = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        assert_close(joined, values)
        assert joined.shape[0] == 2 * STEPS + 1
    for _ in range(2 * STEPS):
        native.step(DT)
    assert_close(final, functional.extract(native).state)
    amplitude = next(
        value
        for name, value in tensors.parameters.items()
        if name.endswith("waveform.amp")
    )
    actual = torch.autograd.grad(final["integrator"]["v"].square().mean(), amplitude)[0]
    expected = torch.autograd.grad(native.v.square().mean(), native_pulse.waveform.amp)[
        0
    ]
    assert_close(actual, expected)
    assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    assert_close(functional.extract().state, original)
    assert_close(pulse.generator.get_state(), source_rng)
    assert_close(torch.get_rng_state(), global_rng)
    print(
        "Population: canonical replay, automatic sampling, native gradients, and continuation passed."
    )


def network_demo(*, backend="eager"):
    source, pulses = make_network()
    native, _ = make_network()
    functional, tensors = dn.func.make_functional_network(source)
    original = clone_tree(tensors.state)
    source_rng = pulses[0].generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    # region sample-network
    draws = functional.sample_runtime_draws(tensors.state, steps=2 * STEPS)
    kernel = functional.compile_rollout_chunk(3, backend=backend)
    bound = kernel.bind(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    )
    final, auxiliary = dn.func.run(
        bound,
        tensors.state,
        steps=2 * STEPS,
        draws=draws,
        checkpoint_every=2,
        callbacks=callbacks,
    )
    # endregion sample-network
    automatic, automatic_aux = dn.func.run(
        bound, tensors.state, steps=2 * STEPS, checkpoint_every=2, callbacks=callbacks
    )
    assert_close(final, automatic)
    assert_close(
        dict(auxiliary["callbacks"]["trace"]), dict(automatic_aux["callbacks"]["trace"])
    )
    for _ in range(2 * STEPS):
        native.step()
    assert_close(final, functional.extract(native).state)
    assert_close(functional.extract().state, original)
    assert_close(pulses[0].generator.get_state(), source_rng)
    assert_close(torch.get_rng_state(), global_rng)
    print(
        "Network: native shared-stream ordering, compiled replay, recordings, and checkpoints passed."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("eager", "aot_eager", "inductor"), default="eager"
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        population_demo()
        network_demo(backend=arguments.backend)


if __name__ == "__main__":
    main()
