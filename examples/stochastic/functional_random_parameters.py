"""Compile sampled heterogeneity with explicit draws and reproducible restarts.

Run from the repository root: python examples/stochastic/functional_random_parameters.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT = 0.125
STEPS = 4


# region declaration
class HeterogeneousLeak(Mechanism):
    Mechanism.RANGERAND(
        "g",
        distribution="lognormal",
        mu=-7.0,
        sigma=0.2,
        seed=17,
        reseed_on_initialize=False,
    )
    Mechanism.GLOBAL(e=-65.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e


# endregion declaration


def make_population(*, mechanism=HeterogeneousLeak):
    cell = dn.SingleCompartment(
        N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    cell.insert(mechanism)
    cell.train()
    cell.initialize()
    return cell


def run_population_demo():
    source, native = make_population(), make_population()
    # region population-runner
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("g_mu_param",),
        callbacks={"trace": dn.func.Recorder(("v",))},
        backend="aot_eager",
    )
    _, auxiliary = runner()
    voltage = auxiliary["callbacks"]["trace"]["v"]
    mu = runner.get_parameter("g_mu_param")
    gradient = torch.autograd.grad(voltage.square().mean(), mu)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # endregion population-runner
    # RAND initialization advances the retained stream; its seed does not
    # reseed every initialize. Both models begin at the same captured suffix.
    native.initialize()
    expected = [native.v.clone()]
    for _ in range(STEPS):
        native.step(DT)
        expected.append(native.v.clone())
    torch.testing.assert_close(voltage, torch.stack(expected))
    return voltage.detach()


def make_network():
    network = dn.Network({"left": make_population(), "right": make_population()})
    network.train()
    network.initialize(DT)
    return network


def run_network_demo():
    source, native = make_network(), make_network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        execution="while",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        backend="aot_eager",
    )
    # region network-draws
    functional = runner.functional
    draws = functional.sample_population_initialization_draws(
        runner.parameters, runner.initial_state
    )
    initial = runner.initialization_inputs
    initial = initial._replace(
        populations={
            name: values._replace(random_draw=draws[name])
            for name, values in initial.populations.items()
        }
    )
    prefix, first = runner(initialization=initial)
    # Sampling and execution leave the source and input RNG packets unchanged.
    replay, replay_auxiliary = runner(initialization=initial)
    torch.testing.assert_close(
        replay["populations"]["left"]["integrator"]["v"],
        prefix["populations"]["left"]["integrator"]["v"],
    )
    torch.testing.assert_close(
        first["callbacks"]["trace"]["left.v"],
        replay_auxiliary["callbacks"]["trace"]["left.v"],
    )
    # endregion network-draws
    # region continuation
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    voltage = torch.cat(
        (first["callbacks"]["trace"]["left.v"], second["callbacks"]["trace"]["left.v"])
    )
    assert voltage.shape == (2 * STEPS + 1, 2, 2)
    # endregion continuation
    with torch.no_grad():
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
        expected = [native.left.v.clone()]
        for _ in range(2 * STEPS):
            native.step()
            expected.append(native.left.v.clone())
    torch.testing.assert_close(voltage, torch.stack(expected))
    torch.testing.assert_close(
        final["populations"]["left"]["integrator"]["v"], native.left.v
    )
    assert source.t.item() == 0.0
    return voltage


# region custom-distribution
def uniform_draw(rng, shape, *, device, dtype):
    return rng.rand(shape, device=device, dtype=dtype)


def squared_uniform(parameters, draw):
    return parameters["low"] + parameters["span"] * draw.square()


dn.register_random_distribution(
    "squared_uniform_example",
    {"low": 0.0004, "span": 0.001},
    {"low": "positive", "span": "positive"},
    draw_sampler=uniform_draw,
    transform=squared_uniform,
    overwrite=True,
)


class CustomLeak(Mechanism):
    Mechanism.RANGERAND("g", distribution="squared_uniform_example", seed=31)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * -65.0


# endregion custom-distribution


def run_custom_demo():
    source, native = (
        make_population(mechanism=CustomLeak),
        make_population(mechanism=CustomLeak),
    )
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("v",))},
        backend="aot_eager",
    )
    _, auxiliary = runner()
    native.initialize()
    expected = [native.v.clone()]
    for _ in range(STEPS):
        native.step(DT)
        expected.append(native.v.clone())
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"]["v"], torch.stack(expected)
    )


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        population = run_population_demo()
        network = run_network_demo()
        run_custom_demo()
    print("Population sampled-start voltage shape:", tuple(population.shape))
    print("continued Network sampled-start voltage shape:", tuple(network.shape))


if __name__ == "__main__":
    main()
