"""Parametrize shared Ion/Material initializers and differentiate a Network.

Run: python examples/functional_initialization_parametrizations.py
"""

from __future__ import annotations

import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT = 0.125
STEPS = 2


# region recipe
class InitialValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor(140.0, dtype=torch.float64))

    def forward(self):
        return self.raw


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("offset", torch.tensor(0.5, dtype=torch.float64))

    def forward(self, value):
        return value * self.factor + self.offset


def make_source():
    source = InitialValue()
    register_parametrization(source, "raw", Scale())
    return source


# endregion recipe


class SodiumLeak(Mechanism):
    Mechanism.GLOBAL(g=0.001)
    Mechanism.USEION("na", read=["ena"], write=["ina"])
    Mechanism.USEMATERIAL("pool", read=["amount"])
    Mechanism.AFFINE("ina")

    def ina_affine_terms(self, v):
        del v
        conductance = self.g + 0.00001 * self.amount
        return conductance, conductance * self.ena


def make_population(source):
    population = dn.SingleCompartment(
        N=1, C=2, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    # region attach
    population.concentrations(nao0=source)
    population.material("pool", fields={"amount": source}, conserved=False)
    population.ion_style("na", 1, 1)
    population.insert(SodiumLeak)
    # endregion attach
    population.train()
    with torch.no_grad():
        population.initialize()
    population.detach()
    return population


def make_network():
    source = make_source()
    network = dn.Network(
        {"left": make_population(source), "right": make_population(source)}
    )
    network.train()
    with torch.no_grad():
        network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network, source


def loss(left, right):
    return left.square().sum() + right.square().sum()


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        network, source = make_network()
        native, native_source = make_network()
        # region compile
        runner = dn.compile(
            network,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            inference=False,
            trainable=("*o_init.parametrizations.raw.*",),
            checkpoint_every=1,
            callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
            backend="aot_eager",
        )
        original = runner.get_parameter("*left*o_init.parametrizations.raw.original")
        factor = runner.get_parameter("*left*o_init.parametrizations.raw.0.factor")
        assert original is runner.get_parameter(
            "*right*o_init.parametrizations.raw.original"
        )
        print("Trainable parameters:", tuple(runner.trainable_parameters))
        # endregion compile
        # region continue
        prefix, first = runner()
        final, second = runner(
            prefix,
            parameters=first["parameters"],
            constants=first["constants"],
            callback_state=first["callbacks"].state,
        )
        left = final["populations"]["left"]["integrator"]["v"]
        right = final["populations"]["right"]["integrator"]["v"]
        gradients = torch.autograd.grad(loss(left, right), (original, factor))
        # endregion continue

        native.initialize(DT)
        expected = [native.left.v.clone()]
        for _ in range(2 * STEPS):
            native.step()
            expected.append(native.left.v.clone())
        native_gradients = torch.autograd.grad(
            loss(native.left.v, native.right.v), tuple(native_source.parameters())
        )
        torch.testing.assert_close(left, native.left.v)
        torch.testing.assert_close(right, native.right.v)
        for actual, expected_gradient in zip(gradients, native_gradients, strict=True):
            torch.testing.assert_close(actual, expected_gradient)
            assert torch.isfinite(actual).all() and actual.abs().sum() > 0
        recorded = torch.cat(
            (
                first["callbacks"]["trace"]["left.v"],
                second["callbacks"]["trace"]["left.v"],
            )
        )
        torch.testing.assert_close(recorded, torch.stack(expected))
        assert network.t.item() == 0.0
        assert source.parametrizations.raw.original.item() == 140.0
    print("Shared-source values, gradients, recording, and continuation match native.")


if __name__ == "__main__":
    main()
