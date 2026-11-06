"""Sample MaterialProcess heterogeneity and runtime noise with explicit RNG carry.

Run: python examples/stochastic/functional_material_stochasticity.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import MaterialProcess
from dendra.models.mod import pas

DT = 0.125
STEPS = 3


# region process
class NoisyTransfer(MaterialProcess):
    MaterialProcess.GLOBAL(gain=0.8)
    MaterialProcess.RANGERAND(
        "rate",
        distribution="lognormal",
        mu=-1.0,
        sigma=0.15,
        seed=17,
        rng_name="stream",
        resample_on_initialize=True,
    )
    MaterialProcess.RANGENOISE(
        "eta",
        distribution="normal",
        mu=0.0,
        sigma=0.1,
        seed=17,
        rng_name="stream",
        scale="standard",
    )
    MaterialProcess.PHASE("transport")
    material_fields = (("pool", "amount"), ("reservoir", "amount"))

    def prepare_materials(self, fields, dt):
        like = fields[("pool", "amount")]
        rate = self._resolve_quantity("rate", like)
        gain = self._resolve_quantity("gain", like)
        noise = self._resolve_quantity("eta", like).clone()
        return (-torch.expm1(-dt * rate * gain * torch.exp(noise)),)

    def material_updates(self, fields, prepared):
        (fraction,) = prepared
        pool, reservoir = fields[("pool", "amount")], fields[("reservoir", "amount")]
        transferred = fraction * pool
        return self._mask_material_updates(
            fields,
            {
                ("pool", "amount"): pool - transferred,
                ("reservoir", "amount"): reservoir + transferred,
            },
        )


# endregion process


def make_network():
    populations = {}
    shared_mean = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
    for name in ("left", "right"):
        cell = dn.SingleCompartment(N=1, C=2, integrator=dn.bwd_euler_sc(imem=False))
        cell.material("pool", fields={"amount": 2.0})
        cell.material("reservoir", fields={"amount": 0.5})
        cell.insert(pas)
        cell.insert(NoisyTransfer.rename("transfer"), rate_mu=shared_mean)
        populations[name] = cell
    network = dn.Network(populations)
    network.train()
    network.initialize(DT)
    return network


def noise_inputs(functional, state):
    tapes = functional.sample_population_noise_tapes(state, STEPS)
    return dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
        }
    )


def amount(state, name):
    return state["populations"][name]["materials"]["reservoir"]["amount"]


def run_demo():
    source, native = make_network(), make_network()
    # region compile
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*rate_mu_param", "*gain_param", "*eta_mu_param"),
        checkpoint_every=2,
        backend="aot_eager",
    )
    shared_mean = runner.get_parameter("*left*rate_mu_param")
    assert shared_mean is runner.get_parameter("*right*rate_mu_param")
    # endregion compile
    # region initialization
    functional = runner.functional
    draws = functional.sample_population_initialization_draws(
        runner.parameters, runner.initial_state
    )
    initial = runner.initialization_inputs._replace(
        populations={
            name: values._replace(random_draw=draws[name])
            for name, values in runner.initialization_inputs.populations.items()
        }
    )
    initialized = functional.initialize(
        runner.parameters, runner.constants, runner.initial_state, initial
    )
    # endregion initialization
    # region continuation
    prefix, first = runner(
        initialization=initial, inputs=noise_inputs(functional, initialized.state)
    )
    final, _ = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        inputs=noise_inputs(functional, prefix),
    )
    # endregion continuation
    # region gradients
    mean_gradient, noise_gradient = torch.autograd.grad(
        sum(amount(final, name).sum() for name in ("left", "right")),
        (shared_mean, runner.get_parameter("*left*eta_mu_param")),
        allow_unused=True,
    )
    assert torch.isfinite(mean_gradient).all() and mean_gradient.abs().sum() > 0
    assert noise_gradient is None or torch.count_nonzero(noise_gradient) == 0
    # endregion gradients

    native.initialize(DT)
    for _ in range(2 * STEPS):
        native.step()
    expected = functional.extract(native)
    for name in ("left", "right"):
        torch.testing.assert_close(amount(final, name), amount(expected.state, name))
        pool = final["populations"][name]["materials"]["pool"]["amount"]
        torch.testing.assert_close(
            pool + amount(final, name), torch.full_like(pool, 2.5)
        )
    native_gradient = torch.autograd.grad(
        sum(
            native.populations[name].mech.materials["reservoir"].amount.sum()
            for name in ("left", "right")
        ),
        native.left.mech.material_processes["transfer"].rate_mu_param,
    )[0]
    torch.testing.assert_close(mean_gradient, native_gradient)
    for actual, reference in zip(
        torch.utils._pytree.tree_leaves(final["populations"]["left"]["random"]["rng"]),
        torch.utils._pytree.tree_leaves(
            expected.state["populations"]["left"]["random"]["rng"]
        ),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference)
    assert source.t.item() == 0.0
    return mean_gradient


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        gradient = run_demo()
    print("Material values, shared RAND gradients, and continued RNG match native.")
    print("Shared distribution mean gradient:", gradient.item())


if __name__ == "__main__":
    main()
