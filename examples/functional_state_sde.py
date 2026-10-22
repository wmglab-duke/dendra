"""Run State SDEs with pathwise gradients and replayable draw/RNG tapes.

Run from the repository root: python examples/functional_state_sde.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism, State

DT = 0.125
STEPS = 4


# region ou-state
class OrnsteinUhlenbeck(State):
    State.STATE("x")
    State.RANGE(theta=0.7, mu=0.2, sigma=0.15)
    State.DERIVATIVE("x' = theta * (mu - x)")
    State.DIFFUSION("x = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


# endregion ou-state


# region heun-state
class MultiplicativeState(State):
    State.STATE("x")
    State.RANGE(theta=0.7, mu=0.2, sigma=0.15)
    State.ASSIGNED("diffusion")
    State.DERIVATIVE("x' = theta * (mu - x)")
    State.DIFFUSION("x = diffusion")
    State.METHOD("euler_heun", average_drift=True)

    def assigned_values(self, v, values):
        del v
        return {"diffusion": self.sigma * values["x"]}

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


# endregion heun-state


class DrivenLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + 0.002 * self.x


class OULeak(DrivenLeak):
    Mechanism.STATE_BUNDLE(OrnsteinUhlenbeck)


class HeunLeak(DrivenLeak):
    Mechanism.STATE_BUNDLE(MultiplicativeState)


def make_population(mechanism=OULeak):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism.rename("stochastic"))
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def run_population_demo():
    source, native = make_population(), make_population()
    # Align both values and RNG streams using the ordinary native checkpoint.
    native.restore_dict_from_checkpoint(source.state_dict_for_checkpoint())
    # region population-runner
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*theta_param", "*mu_param", "*sigma_param"),
        callbacks={"trace": dn.func.Recorder(("v", "stochastic.x"))},
        checkpoint_every=2,
        backend="aot_eager",
    )
    _, auxiliary = runner()
    trace = auxiliary["callbacks"]["trace"]
    theta, sigma = (
        runner.get_parameter(f"*{name}_param") for name in ("theta", "sigma")
    )
    gradients = torch.autograd.grad(
        trace["stochastic.x"].square().mean(), (theta, sigma)
    )
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    # endregion population-runner
    with torch.no_grad():
        expected = [native.mech.stochastic.x.clone()]
        for _ in range(STEPS):
            native.step(DT)
            expected.append(native.mech.stochastic.x.clone())
    torch.testing.assert_close(trace["stochastic.x"], torch.stack(expected))
    return trace["stochastic.x"].detach()


def run_explicit_heun_demo():
    functional, tensors = dn.func.make_functional(make_population(HeunLeak), dt=DT)
    parameters = tensors.independent_parameters(trainable=("*sigma_param",))
    state = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors.state
    )
    constants = dict(tensors.constants)
    dt = constants["dt"].detach().clone().requires_grad_()
    constants["dt"] = dt
    initial = state["mechanisms"]["stochastic"]["x"].requires_grad_()
    # region explicit-tape
    tape = functional.sample_noise_draw_tape(state, STEPS)
    prepared = functional.prepare(parameters, constants)
    final, _ = functional.rollout(
        parameters,
        prepared,
        state,
        dn.func.RolloutInput(noise=tape),
        steps=STEPS,
    )
    sigma = parameters[tensors.parameter_name("*sigma_param")]
    gradients = torch.autograd.grad(
        final["mechanisms"]["stochastic"]["x"].square().mean(), (sigma, dt, initial)
    )
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    # endregion explicit-tape
    return final["mechanisms"]["stochastic"]["x"].detach()


def make_network():
    network = dn.Network(
        {"left": make_population(), "right": make_population(HeunLeak)}
    )
    network.train()
    network.initialize(DT)
    return network


def run_network_demo():
    source, native = make_network(), make_network()
    native.restore_dict_from_checkpoint(source.state_dict_for_checkpoint())
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution="while",
        inference=True,
        callbacks={
            "trace": dn.func.Recorder(("left.v", "left.stochastic.x", "right.v", "t"))
        },
        backend="aot_eager",
    )
    # region network-tapes
    tapes = runner.functional.sample_population_noise_tapes(runner.initial_state, STEPS)
    inputs = dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
        }
    )
    prefix, first = runner(inputs=inputs)
    replay, repeated = runner(inputs=inputs)
    torch.testing.assert_close(prefix, replay)
    torch.testing.assert_close(
        first["callbacks"]["trace"], repeated["callbacks"]["trace"]
    )
    # endregion network-tapes
    # region continuation
    # Host sampling starts from the returned SDE and RNG boundary.
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    recorded = torch.cat(
        (
            first["callbacks"]["trace"]["left.stochastic.x"],
            second["callbacks"]["trace"]["left.stochastic.x"],
        )
    )
    # endregion continuation
    with torch.no_grad():
        expected = [native.left.mech.stochastic.x.clone()]
        for _ in range(2 * STEPS):
            native.step()
            expected.append(native.left.mech.stochastic.x.clone())
    torch.testing.assert_close(recorded, torch.stack(expected))
    torch.testing.assert_close(
        final["populations"]["left"]["integrator"]["v"], native.left.v
    )
    assert source.t.item() == 0.0
    return recorded


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        population = run_population_demo()
        heun = run_explicit_heun_demo()
        network = run_network_demo()
    print("Recorded OU state:", tuple(population.shape))
    print("Explicit multiplicative Heun state:", tuple(heun.shape))
    print("Continued Network OU state:", tuple(network.shape))


if __name__ == "__main__":
    main()
