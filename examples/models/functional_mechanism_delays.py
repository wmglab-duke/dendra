"""Differentiate mechanism delay histories and continue a compiled Network.

Run from the repository root: python examples/models/functional_mechanism_delays.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import graded_syn

DT = 0.125
STEPS = 4


# region single-stream
class DelayedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.CARRY("delayed")
    Mechanism.CARRY("signal_delay_buffer", shape="deferred")
    Mechanism.CARRY("signal_delay_ptr", shape=(), dtype=torch.long)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def __init__(self, *args, **kwargs):
        Mechanism.__init__(self, *args, **kwargs)
        self.register_delayed_state(
            "signal",
            self.g.expand(self.shape_f),
            delay_steps=2,
            mode="shift",
            insert_axis=0,
        )

    def initial_values(self, v, values):
        return {"delayed": torch.zeros_like(v), **self.initial_delay_values(values)}

    def advance(self, v, dt, values):
        del dt
        source = torch.sigmoid((v + 55.0) / 15.0)
        delayed, updates = self.delayed_state_values("signal", source, values)
        return {**updates, "delayed": delayed}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * self.delayed


# endregion single-stream


# region multi-stream
class PathwayLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.CARRY("delayed")
    Mechanism.CARRY("paths_delay_buffer", shape="deferred")
    Mechanism.CARRY("paths_delay_ptr", shape=(), dtype=torch.long)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")
    supports_population_axis_multistream_delays = True

    def __init__(self, *args, **kwargs):
        Mechanism.__init__(self, *args, **kwargs)
        payload = torch.stack((self.g.expand(self.shape_f),) * 3, dim=0)
        self.register_delayed_states(
            "paths",
            payload,
            delay_steps=(0, 1, 3),
            mode="shift",
            stream_axis=0,
            delay_axis=0,
        )

    def initial_values(self, v, values):
        return {"delayed": torch.zeros_like(v), **self.initial_delay_values(values)}

    def advance(self, v, dt, values):
        del dt
        source = torch.sigmoid((v + 55.0) / 15.0)
        payload = torch.stack((source, 1.3 * source, 0.7 * source), dim=0)
        delayed, updates = self.delayed_states_values("paths", payload, values)
        return {**updates, "delayed": delayed.sum(dim=0)}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * self.delayed


# endregion multi-stream


def make_population(mechanism=DelayedLeak):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-47.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism.rename("delays"))
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def run_training_demo():
    source, native = make_population(), make_population()
    # region train-delay
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*gain_param",),
        checkpoint_every=2,
        callbacks={"trace": dn.func.Recorder(("v", "delays.delayed"))},
        backend="aot_eager",
    )
    final, auxiliary = runner()
    gradient = torch.autograd.grad(
        final["integrator"]["v"].square().mean(), runner.get_parameter("*gain_param")
    )[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # endregion train-delay
    with torch.no_grad():
        for _ in range(STEPS):
            native.step(DT)
    torch.testing.assert_close(final["integrator"]["v"], native.v)
    assert source.t.item() == 0.0
    return auxiliary["callbacks"]["trace"]["v"].detach()


def run_network_demo():
    left, right = make_population(), make_population(PathwayLeak)
    right.insert(graded_syn.rename("graded"), e=-10.0)
    network = dn.Network({"left": left, "right": right})
    network.connect_continuous_one_to_one(
        left[:],
        right[:],
        right.mech.graded,
        pre_var="mech.delays.delayed",
        input="g_pre",
        weight=0.001,
        delay=0.0,
        transform=torch.sigmoid,
    )
    network.eval()
    network.initialize(DT)
    # region network-continuation
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=True,
        execution="while",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.delays.delayed"))},
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
