#!/usr/bin/env python3
"""Lower an ordinary NetCon and compose delivery with a functional population.

Run from the repository root::

    python examples/networks/functional_event_delivery.py

The standard Network and the functional projection share NetCon's transport
core. This example supplies dimensionless source gates explicitly; it does not
implement another event detector or NetStim scheduler.
"""

# region declarations
from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import expsyn, pas


class GateInput(Mechanism):
    Mechanism.CARRY("gate")


# endregion declarations


# region standard-network
def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        pre = dn.SingleCompartment(N=2, C=1, dtype=torch.float64)
        pre.insert(GateInput.rename("source"))
        post = dn.SingleCompartment(N=2, C=1, dtype=torch.float64)
        post.insert(pas, g=0.001, e=-65.0)
        post.insert(expsyn.rename("syn"), tau=2.0, e=0.0)
        network = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            pre[:],
            post[:],
            post.mech.syn,
            pre_var="mech.source.gate",
            threshold=None,
            weight=torch.tensor([0.002, 0.004], dtype=torch.float64),  # µS
            delay=torch.tensor([0.1, 0.3], dtype=torch.float64),  # ms
        )
        network.train()
        network.initialize(dt=0.1)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
            train_delay_backend="dense",
        )
    return network


# endregion standard-network


def run_demo():
    # region lower-projection
    network = make_network()
    connection = next(iter(network.synapses.values()))
    projection, connection_tensors = dn.func.make_functional_netcon(connection)
    population, population_tensors = dn.func.make_functional(network.post, dt=0.1)
    prepared = population.prepare(
        population_tensors.parameters, population_tensors.constants
    )
    weights = connection_tensors.weights.detach().clone().requires_grad_()
    connection_state = connection_tensors.state
    population_state = population_tensors.state
    # endregion lower-projection

    reference = make_network()
    native_connection = next(iter(reference.synapses.values()))
    gates = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.0],
            [0.25, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    ).reshape(8, 2, 1)
    native_weights = native_connection.weight.w
    original_voltage = network.post.v.clone()
    original_queue = connection.delivery_buffer.clone()

    for gate in gates:
        # region functional-step
        connection_state, payload = projection.step(weights, connection_state, gate)
        population_state = population.receive(
            population_tensors.parameters,
            prepared,
            population_state,
            target="syn",
            payload=payload,
        )
        population_state, _ = population.step(
            population_tensors.parameters, prepared, population_state
        )
        # endregion functional-step

        # The ordinary Network consumes the same gates through its source field.
        reference.pre.mech.source.gate = gate
        reference.step()
        torch.testing.assert_close(
            population_state["integrator"]["v"], reference.post.v
        )
        torch.testing.assert_close(
            population_state["mechanisms"]["syn"]["g"], reference.post.mech.syn.g
        )
        torch.testing.assert_close(
            connection_state["delivery_buffer"], native_connection.delivery_buffer
        )

    loss = population_state["integrator"]["v"].square().sum()
    (gradient,) = torch.autograd.grad(loss, weights)
    (native_gradient,) = torch.autograd.grad(
        reference.post.v.square().sum(), native_weights
    )
    torch.testing.assert_close(gradient, native_gradient)
    assert (
        torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient) == weights.numel()
    )
    torch.testing.assert_close(network.post.v, original_voltage)
    torch.testing.assert_close(connection.delivery_buffer, original_queue)
    return population_state, connection_state, gradient


def main():
    _, connection_state, gradient = run_demo()
    print("Shared NetCon core: native and functional queue/voltage checks passed.")
    print("Connection steps:", connection_state["global_step"].item())
    print("Weight gradient:", gradient.tolist())


if __name__ == "__main__":
    main()
