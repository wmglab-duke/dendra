"""Share a native RNG stream across functional Network populations.

Run: python examples/functional_network_rng_sharing.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT = 0.125
STEPS = 3
RAW = "integrator.mech.mechanisms.leak.g_mu_param"


# region declaration
class NoisyLeak(Mechanism):
    Mechanism.GLOBALRAND("g", mu=0.001, sigma=0.00005, seed=17, rng_name="stream")
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=19, rng_name="stream")
    Mechanism.GLOBAL(e=-65.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e - self.eta.clone()


# endregion declaration


def make_network():
    cells = {}
    for name in ("left", "right"):
        cell = dn.SingleCompartment(
            N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(NoisyLeak.rename("leak"))
        cell.initialize()
        cells[name] = cell
    # region sharing
    left, right = cells["left"].mech.leak, cells["right"].mech.leak
    right.stream = left.stream
    right.g_mu_param = left.g_mu_param
    network = dn.Network(cells)
    network.train()
    left.stream.reseed(1203)
    network.initialize(DT)
    # endregion sharing
    return network


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source, native = make_network(), make_network()
        source_voltage = source.left.v.detach().clone()
        source_rng = source.left.mech.leak.stream.get_extra_state()
        # region compile
        runner = dn.compile(
            source,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            inference=False,
            trainable=(f"populations.*.{RAW}",),
            backend="aot_eager",
            callbacks={"trace": dn.func.Recorder(("left.v", "right.v"))},
        )
        functional = runner.functional
        assert len(runner.initial_state["rng"]) == 1
        shared = runner.get_parameter(f"populations.left.{RAW}")
        assert shared is runner.get_parameter(f"populations.right.{RAW}")
        # endregion compile
        # region initialization
        packets = functional.sample_population_initialization_draws(
            runner.parameters, runner.initial_state
        )
        initialization = runner.initialization_inputs._replace(
            populations={
                name: values._replace(random_draw=packets[name])
                for name, values in runner.initialization_inputs.populations.items()
            }
        )
        initial = functional.initialize(
            runner.parameters, runner.constants, runner.initial_state, initialization
        )
        # endregion initialization
        # region continuation
        tapes = functional.sample_population_noise_tapes(initial.state, STEPS)
        inputs = dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
            }
        )
        prefix, auxiliary = runner(inputs=inputs, initialization=initialization)
        final, _ = runner(
            prefix,
            parameters=auxiliary["parameters"],
            constants=auxiliary["constants"],
        )
        # endregion continuation
        native.initialize(DT)
        for _ in range(2 * STEPS):
            native.step()
        # region verification
        for name, cell in native.populations.items():
            torch.testing.assert_close(
                final["populations"][name]["integrator"]["v"], cell.v
            )
        loss = sum(
            state["integrator"]["v"].square().sum()
            for state in final["populations"].values()
        )
        gradient = torch.autograd.grad(loss, shared)[0]
        native_loss = sum(cell.v.square().sum() for cell in native.populations.values())
        expected = torch.autograd.grad(native_loss, native.left.mech.leak.g_mu_param)[0]
        torch.testing.assert_close(gradient, expected)
        torch.testing.assert_close(source.left.v, source_voltage)
        torch.testing.assert_close(
            source.left.mech.leak.stream.get_extra_state(), source_rng
        )
        functional.commit_state_(native, final)
        assert native.left.mech.leak.stream is native.right.mech.leak.stream
        # endregion verification
    print("Shared stream ordering, gradients, and native continuation agree.")


if __name__ == "__main__":
    main()
