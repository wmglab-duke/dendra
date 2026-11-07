"""Execute detached runtime NOISE with replayable draw and RNG tapes.

Run from the repository root: python examples/stochastic/functional_runtime_noise.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT = 0.125
STEPS = 4


# region declaration
class NoisyLeak(Mechanism):
    Mechanism.RANGE(g=0.001)
    Mechanism.GLOBAL(e=-65.0)
    Mechanism.RANGENOISE(
        "eta",
        distribution="normal",
        mu=0.0,
        sigma=0.00002,
        scale="white",
        seed=17,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e - self.eta


# endregion declaration


def make_population():
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(NoisyLeak)
    population.train()
    population.initialize()
    return population


def run_population_demo():
    source, native = make_population(), make_population()
    # region population-runner
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("g_param", "eta_mu_param"),
        callbacks={"trace": dn.func.Recorder(("v",))},
        backend="aot_eager",
    )
    _, auxiliary = runner()
    voltage = auxiliary["callbacks"]["trace"]["v"]
    conductance = runner.get_parameter("g_param")
    noise_mean = runner.get_parameter("eta_mu_param")
    conductance_gradient, noise_gradient = torch.autograd.grad(
        voltage.square().mean(),
        (conductance, noise_mean),
        allow_unused=True,
    )
    assert torch.isfinite(conductance_gradient).all()
    assert conductance_gradient.abs().sum() > 0
    assert noise_gradient is None or torch.count_nonzero(noise_gradient) == 0
    # endregion population-runner
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
        execution="while",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        backend="aot_eager",
    )
    # region network-tapes
    functional = runner.functional
    tapes = functional.sample_population_noise_tapes(runner.initial_state, STEPS)
    inputs = dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=tape)
            for name, tape in tapes.items()
            if tape is not None
        }
    )
    prefix, first = runner(inputs=inputs)
    replay, replay_auxiliary = runner(inputs=inputs)
    torch.testing.assert_close(
        first["callbacks"]["trace"]["left.v"],
        replay_auxiliary["callbacks"]["trace"]["left.v"],
    )
    torch.testing.assert_close(
        prefix["populations"]["left"]["integrator"]["v"],
        replay["populations"]["left"]["integrator"]["v"],
    )
    # endregion network-tapes
    # region continuation
    # The host runner samples the next bounded tape from prefix's RNG carry.
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    voltage = torch.cat(
        (first["callbacks"]["trace"]["left.v"], second["callbacks"]["trace"]["left.v"])
    )
    assert voltage.shape == (2 * STEPS + 1, 2, 2)
    # endregion continuation
    with torch.no_grad():
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


def run_pure_population_demo():
    source, native = make_population(), make_population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    # region population-tape
    tape = functional.sample_noise_draw_tape(tensors.state, STEPS)
    final, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(noise=tape),
        steps=STEPS,
    )
    # endregion population-tape
    for _ in range(STEPS):
        native.step(DT)
    torch.testing.assert_close(final["integrator"]["v"], native.v)


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        population = run_population_demo()
        network = run_network_demo()
        run_pure_population_demo()
    print("Population noisy voltage shape:", tuple(population.shape))
    print("continued Network noisy voltage shape:", tuple(network.shape))


if __name__ == "__main__":
    main()
