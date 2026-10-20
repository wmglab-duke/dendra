"""Composing projections preserves the native per-connection receive boundary."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Synapse

DT = 0.125


class ReceiveLedger(Synapse):
    Synapse.GLOBAL(scale=1.0)
    Synapse.CARRY("calls", "ledger")

    def net_receive_values(self, weights, dt, values):
        return {
            "carry": {
                "calls": values["calls"] + 1,
                "ledger": 2 * values["ledger"] + weights * self.scale + 1,
            }
        }


def _network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64):
        source_a = dn.SingleCompartment(
            N=1, C=1, v_init=0.0, integrator=dn.scnv(imem=False)
        )
        source_b = dn.SingleCompartment(
            N=1, C=1, v_init=0.0, integrator=dn.scnv(imem=False)
        )
        post = dn.SingleCompartment(N=1, C=1, integrator=dn.scnv(imem=False))
        post.insert(ReceiveLedger.rename("receiver"))
        network = dn.Network(
            {"source_a": source_a, "source_b": source_b, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for source, weight in ((source_a, 0.2), (source_b, 0.5)):
            network.connect_one_to_one(
                source[:],
                post[:],
                post.mech.receiver,
                threshold=None,
                weight=weight,
                delay=DT,
            )
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
            train_delay_backend="dense",
        )
    return network


@pytest.mark.parametrize("compiled", (False, True))
def test_two_projections_keep_ordered_zero_receives_and_native_gradients(compiled):
    network = _network()
    connections = tuple(network.synapses.values())
    assert len(connections) == 2
    plans, projection_tensors = zip(
        *(dn.func.make_functional_netcon(connection) for connection in connections),
        strict=True,
    )
    post, population_tensors = dn.func.make_functional(network.post, dt=DT)
    weights = tuple(
        t.weights.detach().clone().requires_grad_() for t in projection_tensors
    )
    native_weights = tuple(
        c.weight.w.detach().clone().requires_grad_() for c in connections
    )
    for connection, weight in zip(connections, native_weights, strict=True):
        connection.weight.w = weight
    gates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.5], [0.0, 0.0], [-0.2, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    native_gates = gates.detach().clone().requires_grad_()
    projection_states = tuple(t.state for t in projection_tensors)
    population_state = population_tensors.state

    def step(connection_states, state, gate_row, values):
        next_connections = []
        for i, plan in enumerate(plans):
            next_connection, payload = plan.step(
                values[i], connection_states[i], gate_row[i].reshape(1, 1)
            )
            next_connections.append(next_connection)
            state = post.prepare_and_receive(
                population_tensors.parameters,
                population_tensors.constants,
                state,
                target="receiver",
                payload=payload,
            )
        state, _ = post.prepare_and_step(
            population_tensors.parameters, population_tensors.constants, state
        )
        return tuple(next_connections), state

    advance = (
        torch.compile(step, backend="aot_eager", fullgraph=True) if compiled else step
    )
    for index, gate in enumerate(gates):
        projection_states, population_state = advance(
            projection_states, population_state, gate, weights
        )
        network.source_a.v = native_gates[index, 0].reshape(1, 1)
        network.source_b.v = native_gates[index, 1].reshape(1, 1)
        network.step()
        observed = population_state["mechanism_buffers"]["receiver"]
        torch.testing.assert_close(observed["calls"], network.post.mech.receiver.calls)
        torch.testing.assert_close(
            observed["ledger"], network.post.mech.receiver.ledger
        )
        for connection_state, connection in zip(
            projection_states, connections, strict=True
        ):
            native_state = connection.state_dict_for_checkpoint()
            for name in connection_state:
                torch.testing.assert_close(connection_state[name], native_state[name])

    assert population_state["mechanism_buffers"]["receiver"]["calls"].item() == 2 * len(
        gates
    )
    actual = torch.autograd.grad(
        population_state["mechanism_buffers"]["receiver"]["ledger"].sum(),
        (gates, *weights),
    )
    expected = torch.autograd.grad(
        network.post.mech.receiver.ledger.sum(), (native_gates, *native_weights)
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)
