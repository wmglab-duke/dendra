#!/usr/bin/env python3
"""Train connected native Network segments with explicit gradient boundaries.

Run from the repository root::

    python examples/training/shared_network_detach.py

The bounded CPU example uses a delayed continuous connection, shared Population
parameters and separately owned callback carry. Each segment has its own
backward pass and optimizer update while numerical history and Recorder cadence
continue across run, checkpointed longrun and longrun.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas
from dendra.models.parametric import PositiveParam

DT = 0.025
SEGMENTS = (("run", 3), ("checkpointed", 4), ("longrun", 3))


class VoltageEnergy(dn.func.FunctionalCallback):
    """Keep an independent cumulative summary for one Population."""

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "energy": voltage.new_zeros(()),
            "steps": voltage.new_zeros((), dtype=torch.int64),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return {
            "energy": carry["energy"] + (state["integrator"]["v"] / 65).square().mean(),
            "steps": carry["steps"] + 1,
        }, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


def make_network():
    conductance = torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float64))
    weight = PositiveParam(torch.tensor(0.0005, dtype=torch.float64))
    populations = {}
    for name, reversal in (("left", -55.0), ("right", -75.0)):
        population = dn.SingleCompartment(
            N=1, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas.rename("leak"), g=conductance, e=reversal)
        populations[name] = population
    populations["right"].insert(graded_syn.rename("analog"), e=0.0)
    network = dn.Network(populations)
    network.connect_continuous_one_to_one(
        network.left[:],
        network.right[:],
        network.right.mech.analog,
        input="g_pre",
        transform=release,
        weight=weight,
        delay=DT,
    )
    network.train()
    network.initialize(DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        train_delay_backend="dense",
    )
    return network, conductance, weight


def advance(network, method, steps, observations, carry):
    options = {"callbacks": observations, "callback_state": carry}
    if method == "checkpointed":
        hook_loss, results = network.longrun_checkpointed(
            steps * DT, chunklength=2, safe_checkpoint=True, **options
        )
        assert hook_loss is None
        return results
    if method == "longrun":
        return network.longrun(steps * DT, chunklength=2, **options)
    return network.run(steps * DT, **options)


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        network, conductance, weight = make_network()
        # region preparation
        scope = dn.callbacks.PopulationCallback
        observations = dn.func.make_callbacks(
            network,
            {
                "trace": dn.Recorder(("left.v", "right.v", "t"), dt=2 * DT),
                "left_energy": scope("left", VoltageEnergy()),
                "right_energy": scope("right", VoltageEnergy()),
            },
        )
        optimizer = torch.optim.SGD(
            [
                {"params": [conductance], "lr": 1e-8},
                {"params": [weight.rho], "lr": 1e-3},
            ]
        )
        # endregion preparation
        carry = None
        previous_voltage = None
        accepted = 0
        times = []
        for method, steps in SEGMENTS:
            optimizer.zero_grad()
            results = advance(network, method, steps, observations, carry)
            loss = (
                results["left_energy"]["energy"]
                + results["right_energy"]["energy"]
                + (results["trace"]["right.v"] / 65).square().mean()
            )
            if previous_voltage is not None:
                assert (
                    torch.autograd.grad(
                        loss, previous_voltage, allow_unused=True, retain_graph=True
                    )[0]
                    is None
                )
            previous_voltage = network.right.v
            before = {
                name: population.v.detach().clone()
                for name, population in network.populations.items()
            }
            clock = network.t.detach().clone()
            # region boundary
            loss.backward()
            optimizer.step()
            network.detach()
            carry = observations.detach_state(results.state)
            # endregion boundary
            for parameter in (conductance, weight.rho):
                assert parameter.grad is not None
                assert (
                    torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                )
                assert any(parameter is source for source in network.parameters())
            assert network.left.mech.leak.g_param is conductance
            assert network.right.mech.leak.g_param is conductance
            for name, population in network.populations.items():
                torch.testing.assert_close(population.v, before[name])
                assert population.v.grad_fn is None
            torch.testing.assert_close(network.t, clock)
            assert carry.plan_token is results.state.plan_token
            accepted += steps
            for name in ("left_energy", "right_energy"):
                torch.testing.assert_close(carry[name], results[name])
                assert carry[name]["energy"].grad_fn is None
                assert carry[name]["steps"].item() == accepted
            assert carry["trace"].item() == accepted
            times.append(results["trace"]["t"].detach())
        torch.testing.assert_close(
            torch.cat(times), torch.arange(0, accepted + 1, 2, dtype=torch.float64) * DT
        )
        torch.testing.assert_close(network.t, network.t.new_tensor(accepted * DT))
        print(
            "Native Network boundaries: connected gradients, shared source ownership, "
            "numerical history and callback cadence passed"
        )


if __name__ == "__main__":
    main()
