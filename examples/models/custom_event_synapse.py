#!/usr/bin/env python3
"""Implement one event synapse and use it in a Network and with dn.func.

Run from the repository root::

    python examples/models/custom_event_synapse.py

All times below are in ms. Conductance state and received weights are bare
numbers in µS; the point-process current is in nA. The checks use a single
deterministic event and a few timesteps, without plotting or compiling.
"""

# region imports
import torch

import dendra as dn
from dendra.models.mechanisms import PointProcess, State, Synapse
from dendra.models.mod import pas

# endregion imports


# region custom-synapse
class ExponentialConductance(State):
    State.STATE("g")
    State.RANGE(tau=2.0)  # ms; choose a strictly positive time constant.
    State.DERIVATIVE("g' = -g / tau")

    def state_defaults(self, v, values):
        return {"g": torch.zeros_like(v)}  # Preserve local shape/device/dtype.


class CustomExpSyn(PointProcess, Synapse):
    """Lumped conductance synapse: weights and g in µS, e in mV, i in nA."""

    PointProcess.STATE_BUNDLE(ExponentialConductance)
    PointProcess.RANGE(e=0.0)
    PointProcess.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)  # Outward-positive current.

    def net_receive_values(self, weights, dt, values):
        # The adapter handles imperative writes; this hook returns replacements.
        # ODE decay runs during the population step, not once per receive call.
        return {"states": {"g": values["g"] + weights}}


# endregion custom-synapse


# region population
def make_cells(*, gradients=False):
    with dn.ctx(DEVICE="cpu", DTYPE=torch.float64, JIT=0, REQUIRE_GRAD=int(gradients)):
        cells = dn.SingleCompartment(
            N=1,
            C=1,
            v_init=-65.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        cells.insert(pas, g=0.001, e=-65.0)  # Distributed leak: g in S/cm².
        cells.insert(CustomExpSyn.rename("syn"), tau=2.0, e=0.0)
        cells.train(gradients)
        cells.initialize()
    return cells


# endregion population


# region network
def network_demo():
    dt = 0.1
    cells = make_cells()
    with dn.ctx(DEVICE="cpu", DTYPE=torch.float64, JIT=0, REQUIRE_GRAD=0):
        source = dn.NetStim(N=1, start=0.0, interval=100.0, noise=0.0, max_spikes=1)
        net = dn.Network({"cells": cells}, netstim=source)
        net.connect_one_to_one(
            net.netstim[:],
            net.cells[:],
            net.cells.mech.syn,
            threshold=None,  # NetStim supplies an event flag, not a voltage.
            weight=0.05,  # 0.05 µS; do not multiply by dendra.units.uS.
            delay=dt,
        )
        net.eval()
        net.initialize(dt)

    net.step()  # Detect the t=0 event and schedule its future delivery.
    torch.testing.assert_close(net.cells.mech.syn.g, torch.zeros_like(cells.v))
    net.step()  # Deliver 0.05 µS, then advance the ODE and membrane voltage.
    expected = torch.full_like(cells.v, 0.05) * torch.exp(cells.v.new_tensor(-dt / 2.0))
    torch.testing.assert_close(net.cells.mech.syn.g, expected)
    net.step()  # A zero-payload receive does not add conductance; decay continues.
    torch.testing.assert_close(
        net.cells.mech.syn.g, expected * torch.exp(cells.v.new_tensor(-dt / 2.0))
    )
    return net


# endregion network


# region functional
def functional_demo():
    cells = make_cells(gradients=True)
    original_voltage = cells.v.detach().clone()
    functional, tensors = dn.func.make_functional(cells, dt=0.1)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    initial_g = tensors.state["mechanisms"]["syn"]["g"]
    payload = torch.full_like(initial_g, 0.05, requires_grad=True)

    received = functional.receive(
        tensors.parameters, prepared, tensors.state, target="syn", payload=payload
    )
    torch.testing.assert_close(received["mechanisms"]["syn"]["g"], initial_g + payload)
    torch.testing.assert_close(received["clock"]["t"], tensors.state["clock"]["t"])
    state, _ = functional.step(tensors.parameters, prepared, received)

    decay = torch.exp(-tensors.constants["dt"] / 2.0)
    conductance = state["mechanisms"]["syn"]["g"]
    torch.testing.assert_close(conductance, (initial_g + payload) * decay)
    (gradient,) = torch.autograd.grad(conductance.sum(), payload)
    torch.testing.assert_close(gradient, torch.ones_like(payload) * decay)
    # Functional delivery and stepping leave the original population unchanged.
    torch.testing.assert_close(cells.mech.syn.g, torch.zeros_like(initial_g))
    torch.testing.assert_close(cells.v, original_voltage)
    return state, gradient


# endregion functional


def main():
    network_demo()
    _, gradient = functional_demo()
    print("Custom synapse: Network delivery and conductance decay checks passed.")
    print(f"Functional payload gradient: {gradient.item():.8f}")


if __name__ == "__main__":
    main()
