"""Sample on both sides of pure initialization transforms.

Run: python examples/stochastic/functional_stochastic_initialization_ordering.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT = 0.125
STEPS = 3
MU = "trial_value_mu_param"
G = "integrator.mech.mechanisms.leak.g"


# region declarations
class TrialPopulation(dn.SingleCompartment):
    dn.SingleCompartment.GLOBALRAND(
        "trial_value", distribution="lognormal", mu=-6.0, sigma=0.1, seed=17
    )
    dn.SingleCompartment.GLOBALNOISE(
        "trial_noise", distribution="normal", mu=-6.0, sigma=0.1, seed=19
    )


class RandomLeak(Mechanism):
    Mechanism.GLOBALRAND("g", distribution="lognormal", mu=-6.0, sigma=0.1, seed=23)
    Mechanism.GLOBAL(e=-65.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e


class ShiftMean(torch.nn.Module):
    def forward(self, mean, shift):
        return (mean + shift,)


# endregion declarations


def make_network():
    cells = {}
    for name in ("left", "right"):
        cell = TrialPopulation(
            N=1, C=1, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(RandomLeak.rename("leak"))
        cells[name] = cell
    network = dn.Network(cells)
    network.train()
    network.initialize(DT)
    # region sharing
    mean = network.left.trial_value_mu_param
    for cell in network.populations.values():
        cell.trial_value_mu_param = mean
        cell.trial_noise_mu_param = mean
        cell.mech.mechanisms["leak"].g_mu_param = mean
        cell.register_pre_initialize_transform(
            "shift_mean",
            ShiftMean(),
            reads=(f"parameters.{MU}",),
            writes=(f"parameters.{MU}",),
            inputs={"shift": torch.tensor(0.2, dtype=torch.float64)},
        )
    # endregion sharing
    return network


def voltage(state, name):
    return state["populations"][name]["integrator"]["v"]


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source, native = make_network(), make_network()
        # region compile
        runner = dn.compile(
            source,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            inference=False,
            trainable=(f"populations.*.{MU}",),
            checkpoint_every=2,
            backend="aot_eager",
        )
        shared = runner.get_parameter(f"populations.left.{MU}")
        assert shared is runner.get_parameter(f"populations.right.{MU}")
        # endregion compile
        # region draws
        functional = runner.functional
        packets = functional.sample_population_initialization_draws(
            runner.parameters, runner.initial_state
        )
        inputs = runner.initialization_inputs._replace(
            populations={
                name: value._replace(random_draw=packets[name])
                for name, value in runner.initialization_inputs.populations.items()
            }
        )
        initialized = functional.initialize(
            runner.parameters, runner.constants, runner.initial_state, inputs
        )
        # endregion draws
        # region ordering
        for index, name in enumerate(("left", "right")):
            constants = initialized.constants["populations"][name]
            torch.testing.assert_close(
                constants["random.parameters.trial_value.mu"], shared + 0.2 * index
            )
            torch.testing.assert_close(
                constants[f"random.parameters.{G}.mu"],
                shared + 0.2 * (index + 1),
            )
        # endregion ordering
        # region continuation
        prefix, auxiliary = runner(initialization=inputs)
        final, _ = runner(
            prefix,
            parameters=auxiliary["parameters"],
            constants=auxiliary["constants"],
        )
        gradient = torch.autograd.grad(
            sum(voltage(final, name).square().sum() for name in ("left", "right")),
            shared,
        )[0]
        # endregion continuation
        native.initialize(DT)
        for _ in range(2 * STEPS):
            native.step()
        expected = functional.extract(native)
        for name in ("left", "right"):
            torch.testing.assert_close(voltage(final, name), native.populations[name].v)
            constants = auxiliary["constants"]["populations"][name]
            torch.testing.assert_close(
                constants["random.values.trial_value"],
                native.populations[name].trial_value,
            )
            torch.testing.assert_close(
                constants[f"random.values.{G}"],
                native.populations[name].mech.mechanisms["leak"].g,
            )
            torch.testing.assert_close(
                final["populations"][name]["runtime_noise"]["trial_noise"],
                expected.state["populations"][name]["runtime_noise"]["trial_noise"],
            )
            for actual, reference in zip(
                torch.utils._pytree.tree_leaves(
                    final["populations"][name]["random"]["rng"]
                ),
                torch.utils._pytree.tree_leaves(
                    expected.state["populations"][name]["random"]["rng"]
                ),
                strict=True,
            ):
                torch.testing.assert_close(actual, reference)
        native_gradient = torch.autograd.grad(
            sum(
                native.populations[name].v.square().sum() for name in ("left", "right")
            ),
            native.left.trial_value_mu_param,
        )[0]
        torch.testing.assert_close(gradient, native_gradient)
        assert source.t.item() == 0.0
    print(
        "Ordered samples, shared-parameter gradients, and RNG continuation match native."
    )


if __name__ == "__main__":
    main()
