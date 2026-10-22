"""Train Ion/Material initializers and continue a compiled Network.

Run from the repository root: python examples/functional_initialization_sources.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.parametric import PositiveParam

DT = 0.125
STEPS = 4


# region custom-source
class AffineInitializer(torch.nn.Module):
    def __init__(self, initial=1.25, offset=0.5):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float64))
        self.register_buffer("offset", torch.as_tensor(offset, dtype=torch.float64))

    def forward(self):
        return self.raw + self.offset


# endregion custom-source


# region mechanism
class SodiumLeak(Mechanism):
    Mechanism.GLOBAL(g=0.001)
    Mechanism.USEION("na", read=["ena"], write=["ina"])
    Mechanism.USEMATERIAL("pool", read=["amount"])
    Mechanism.AFFINE("ina")

    def ina_affine_terms(self, v):
        del v
        conductance = self.g + 0.00001 * self.amount
        return conductance, conductance * self.ena


# endregion mechanism


def make_population(*, sodium=None, amount=None):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    # region attach
    if sodium is None:
        sodium = PositiveParam(
            torch.tensor(140.0, dtype=torch.float64), requires_grad=True
        )
    if amount is None:
        amount = AffineInitializer()
    population.concentrations(nao0=sodium)
    population.material("pool", fields={"amount": amount}, conserved=False)
    population.ion_style("na", 1, 1)
    population.insert(SodiumLeak)
    population.train()
    population.initialize()
    # endregion attach
    return population


def run_population_demo():
    source, native = make_population(), make_population()
    # region train
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*ions.na.o_init.rho", "*pool._initial_sources.amount.raw"),
        checkpoint_every=2,
        callbacks={"trace": dn.func.Recorder(("v",))},
        backend="aot_eager",
    )
    final, auxiliary = runner()
    sodium_raw = runner.get_parameter("*ions.na.o_init.rho")
    material_raw = runner.get_parameter("*pool._initial_sources.amount.raw")
    gradients = torch.autograd.grad(
        final["integrator"]["v"].square().mean(), (sodium_raw, material_raw)
    )
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    # endregion train
    native.initialize()
    expected = [native.v.clone()]
    for _ in range(STEPS):
        native.step(DT)
        expected.append(native.v.clone())
    trace = auxiliary["callbacks"]["trace"]["v"]
    torch.testing.assert_close(trace, torch.stack(expected))
    assert source.t.item() == 0.0
    return trace.detach()


def make_network():
    # region shared
    sodium = PositiveParam(torch.tensor(140.0, dtype=torch.float64), requires_grad=True)
    network = dn.Network(
        {
            "left": make_population(sodium=sodium),
            "right": make_population(sodium=sodium),
        }
    )
    network.train()
    network.initialize(DT)
    # endregion shared
    return network


def run_network_demo():
    source, native = make_network(), make_network()
    # region continue
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*ions.na.o_init.rho",),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        backend="aot_eager",
    )
    assert runner.get_parameter("*left*ions.na.o_init.rho") is runner.get_parameter(
        "*right*ions.na.o_init.rho"
    )
    prefix, first = runner()
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    # endregion continue
    native.initialize(DT)
    expected = [native.left.v.clone()]
    for _ in range(2 * STEPS):
        native.step()
        expected.append(native.left.v.clone())
    trace = torch.cat(
        (first["callbacks"]["trace"]["left.v"], second["callbacks"]["trace"]["left.v"])
    )
    torch.testing.assert_close(trace, torch.stack(expected))
    torch.testing.assert_close(
        final["populations"]["right"]["integrator"]["v"], native.right.v
    )
    return trace.detach()


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        population_trace = run_population_demo()
        network_trace = run_network_demo()
    print("Fresh Population trace:", tuple(population_trace.shape))
    print("Continued shared-source Network trace:", tuple(network_trace.shape))


if __name__ == "__main__":
    main()
