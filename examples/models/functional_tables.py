"""Differentiate TABLE values and continue compiled Population and Network runs.

Run from the repository root: python examples/models/functional_tables.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State

DT = 0.125
STEPS = 4


# region mechanism-table
class TabulatedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.CARRY("gain")
    Mechanism.TABLE("curve", -100.0, 20.0, 17, learnable=True)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def curve(self, v):
        return 0.5 + torch.sigmoid((v + 50.0) / 15.0)

    def initial_values(self, v, values):
        del values
        return {"gain": self.curve(v)}

    def advance(self, v, dt, values):
        del dt, values
        return {"gain": self.curve(v)}

    def i_affine_terms(self, v):
        del v
        conductance = self.g * self.gain
        return conductance, conductance * self.e


# endregion mechanism-table


# region state-table
class TabulatedGate(State):
    State.STATE("x")
    State.RANGE(tau=2.0)
    State.ASSIGNED("target")
    State.TABLE("curve", -100.0, 20.0, 17, learnable=True)
    State.DERIVATIVE("x' = (target - x) / tau")
    State.METHOD("cnexp")

    def curve(self, v):
        return 0.5 + torch.sigmoid((v + 50.0) / 15.0)

    def assigned_values(self, v, values):
        del values
        return {"target": self.curve(v)}

    def state_defaults(self, v, values):
        del values
        return {"x": self.curve(v)}


class GatedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.STATE_BUNDLE(TabulatedGate)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        conductance = self.g * self.x
        return conductance, conductance * self.e


# endregion state-table


def make_population(mechanism=TabulatedLeak):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism.rename("tabulated"))
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def run_training_demo():
    source, native = make_population(), make_population()
    # region train-table
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*curve_table.y",),
        checkpoint_every=2,
        callbacks={"trace": dn.func.Recorder(("v", "tabulated.gain"))},
        backend="aot_eager",
    )
    selected = runner.get_parameter("*curve_table.y")
    final, auxiliary = runner()
    gradient = torch.autograd.grad(final["integrator"]["v"].square().mean(), selected)[
        0
    ]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # endregion train-table
    with torch.no_grad():
        for _ in range(STEPS):
            native.step(DT)
    torch.testing.assert_close(final["integrator"]["v"], native.v)
    assert source.t.item() == 0.0
    return auxiliary["callbacks"]["trace"]["v"].detach()


def run_network_demo():
    network = dn.Network(
        {"left": make_population(), "right": make_population(GatedLeak)}
    )
    network.train()
    network.initialize(DT)
    # region network-continuation
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=True,
        execution="while",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.tabulated.x"))},
        backend="aot_eager",
    )
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    assert final["clock"]["t"].item() == 2 * STEPS * DT
    # endregion network-continuation
    return torch.cat(
        (first["callbacks"]["trace"]["left.v"], second["callbacks"]["trace"]["left.v"])
    )


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        population_trace = run_training_demo()
        network_trace = run_network_demo()
    print("Population voltage trace:", tuple(population_trace.shape))
    print("Continued Network voltage trace:", tuple(network_trace.shape))


if __name__ == "__main__":
    main()
