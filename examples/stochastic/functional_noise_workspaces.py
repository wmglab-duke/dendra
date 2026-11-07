"""Keep noise-dependent workspaces at their native preparation boundaries.

Run: python examples/stochastic/functional_noise_workspaces.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT = 0.125
STEPS = 3
G = "integrator.mech.mechanisms.leak.g_param"


# region declaration
class NoisyLeak(Mechanism):
    Mechanism.GLOBAL(g=0.001, e=-65.0)
    Mechanism.RANGENOISE("eta", mu=1.0, sigma=0.1, seed=17)
    Mechanism.DERIVED_BUFFER("conductance")
    Mechanism.TIMESTEP_BUFFER("increment")
    Mechanism.CARRY("accumulator")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_buffers(self):
        return {"conductance": self.g * self.eta}

    def derive_timestep_buffers(self, dt):
        return {"increment": dt * self.conductance * (1.0 + self.eta)}

    def initial_values(self, v, values):
        del values
        return {"accumulator": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del v, dt
        return {"accumulator": values["accumulator"] + self.increment}

    def i_affine_terms(self, v):
        del v
        return self.conductance, self.conductance * self.e


# endregion declaration


def make_network():
    cells = {}
    for name in ("left", "right"):
        cell = dn.SingleCompartment(
            N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(NoisyLeak.rename("leak"))
        cells[name] = cell
    network = dn.Network(cells)
    network.initialize(DT)
    cells["right"].mech.leak.g_param = cells["left"].mech.leak.g_param
    network.train()
    network.initialize(DT)
    return network


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
            trainable=(f"populations.*.{G}",),
            backend="aot_eager",
            callbacks={"trace": dn.func.Recorder(("left.v", "right.v"))},
        )
        shared = runner.get_parameter(f"populations.left.{G}")
        assert shared is runner.get_parameter(f"populations.right.{G}")
        # endregion compile
        # region initialization
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
        # endregion initialization
        # region execution
        tapes = functional.sample_population_noise_tapes(initialized.state, STEPS)
        rollout = dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
            }
        )
        prefix, auxiliary = runner(inputs=rollout, initialization=inputs)
        final, _ = runner(
            prefix,
            parameters=auxiliary["parameters"],
            constants=auxiliary["constants"],
        )
        # endregion execution
        native.initialize(DT)
        held = {
            name: (
                population.mech.leak.conductance.clone(),
                population.mech.leak.increment.clone(),
            )
            for name, population in native.populations.items()
        }
        for _ in range(2 * STEPS):
            native.step()
        # region verification
        loss = sum(
            final["populations"][name]["mechanism_buffers"]["leak"]["accumulator"].sum()
            for name in ("left", "right")
        )
        gradient = torch.autograd.grad(loss, shared)[0]
        native_loss = sum(
            population.mech.leak.accumulator.sum()
            for population in native.populations.values()
        )
        native_gradient = torch.autograd.grad(
            native_loss, native.left.mech.leak.g_param
        )[0]
        torch.testing.assert_close(gradient, native_gradient)
        for name, population in native.populations.items():
            torch.testing.assert_close(
                final["populations"][name]["integrator"]["v"], population.v
            )
            torch.testing.assert_close(population.mech.leak.conductance, held[name][0])
            torch.testing.assert_close(population.mech.leak.increment, held[name][1])
        # endregion verification
    print("Held noise workspaces, shared gradients, and native continuation agree.")


if __name__ == "__main__":
    main()
