#!/usr/bin/env python3
"""Preserve receive-before-detect ordering with shared NetCon threshold detection.

Run from the repository root::

    python examples/networks/functional_threshold_detection.py
"""

# region declarations
from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mechanisms import Synapse

DT = 0.1


class Relay(Synapse):
    """A diagnostic relay whose reception changes its source signal."""

    Synapse.GLOBAL(scale=1.0)
    Synapse.CARRY("level", "calls")

    def initial_values(self, v, values):
        return {"level": torch.full_like(v, -0.5)}

    def net_receive_values(self, weights, dt, values):
        level = torch.where(
            values["level"] > 2, values["level"] - 4, values["level"] + 1
        )
        return {
            "carry": {
                "level": level + weights * self.scale,
                "calls": values["calls"] + 1,
            }
        }


# endregion declarations


# region standard-network
def make_network(surrogate):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64):
        cell = dn.SingleCompartment(N=1, C=1, integrator=dn.scnv(imem=False))
        cell.insert(Relay.rename("relay"))
        network = dn.Network(
            {"cell": cell},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            cell[:],
            cell[:],
            cell.mech.relay,
            pre_var="mech.relay.level",
            threshold=0.0,
            weight=0.2,
            delay=DT,
            allow_autapses=True,
        )
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=surrogate,
            diff_scheduled_times=False,
            tau=0.23,
            train_delay_backend="dense",
        )
    return network


# endregion standard-network


def run_demo(surrogate):
    model, native = make_network(surrogate), make_network(surrogate)
    connection = next(iter(model.synapses.values()))
    native_connection = next(iter(native.synapses.values()))
    projection, connection_tensors = dn.func.make_functional_netcon(connection)
    population, tensors = dn.func.make_functional(model.cell, dt=DT)
    prepared = population.prepare(tensors.parameters, tensors.constants)
    weights = connection_tensors.weights.detach().clone().requires_grad_()
    native_weights = weights.detach().clone().requires_grad_()
    native_connection.weight.w = native_weights
    connection_state, state = connection_tensors.state, tensors.state
    initial_level = (
        state["mechanism_buffers"]["relay"]["level"].clone().requires_grad_()
    )
    native_level = initial_level.detach().clone().requires_grad_()
    state["mechanism_buffers"]["relay"]["level"] = initial_level
    native.cell.mech.relay.level = native_level

    for index in range(8):
        # region split-step
        connection_state, payload = projection.read_due(connection_state)
        state = population.receive(
            tensors.parameters,
            prepared,
            state,
            target="relay",
            payload=payload,
        )
        # Observe the source after this receiver, then detect and enqueue.
        source_values = state["mechanism_buffers"]["relay"]["level"]
        connection_state = projection.advance(weights, connection_state, source_values)
        state, _ = population.step(tensors.parameters, prepared, state)
        # endregion split-step

        native.step()
        relay = state["mechanism_buffers"]["relay"]
        torch.testing.assert_close(relay["level"], native.cell.mech.relay.level)
        torch.testing.assert_close(relay["calls"], native.cell.mech.relay.calls)
        for name, value in native_connection.state_dict_for_checkpoint().items():
            torch.testing.assert_close(connection_state[name], value)
        if index == 0:
            # Reception moves the level from -0.5 to +0.5 before detection.
            assert connection_state["has_spiked"].all()
            torch.testing.assert_close(
                connection_state["delivery_buffer"].sum(), weights.sum()
            )

    loss = state["mechanism_buffers"]["relay"]["level"].square().sum()
    gradients = torch.autograd.grad(loss, (weights, initial_level))
    native_gradients = torch.autograd.grad(
        native.cell.mech.relay.level.square().sum(),
        (native_weights, native_level),
    )
    for actual, expected in zip(gradients, native_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    # The original model and projection were only read during lowering.
    torch.testing.assert_close(
        model.cell.mech.relay.level, torch.full_like(initial_level, -0.5)
    )
    assert not torch.count_nonzero(connection.delivery_buffer)
    return gradients


def main():
    for surrogate in (False, True):
        gradients = run_demo(surrogate)
        print(
            f"{'Surrogate' if surrogate else 'Hard'} detector: native ordering, "
            f"carry, queues, and gradients match; dloss/dweight={gradients[0].item():.6f}"
        )


if __name__ == "__main__":
    main()
