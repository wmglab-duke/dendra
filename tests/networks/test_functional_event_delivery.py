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


class ThresholdRelay(Synapse):
    Synapse.GLOBAL(scale=1.0)
    Synapse.CARRY("level", "mirror", "calls", "ledger")

    def initial_values(self, v, values):
        return {"level": torch.full_like(v, -0.5), "mirror": torch.full_like(v, -0.5)}

    def net_receive_values(self, weights, dt, values):
        level = torch.where(
            values["level"] > 2, values["level"] - 4, values["level"] + 1
        )
        return {
            "carry": {
                "level": level + weights * self.scale,
                "mirror": level + weights * self.scale,
                "calls": values["calls"] + 1,
                "ledger": 2 * values["ledger"] + weights,
            }
        }


def _recurrent_network(surrogate):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64):
        cell = dn.SingleCompartment(N=1, C=1, integrator=dn.scnv(imem=False))
        cell.insert(ThresholdRelay.rename("relay"))
        network = dn.Network(
            {"cell": cell},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
            track_netcon_events=not surrogate,
        )
        # Different source fields keep these two standard connection groups
        # distinct. Both fields follow the receiver's newly computed level.
        for field, threshold, weight in (("level", 0.0, 0.2), ("mirror", 1.5, 0.3)):
            network.connect_one_to_one(
                cell[:],
                cell[:],
                cell.mech.relay,
                pre_var=f"mech.relay.{field}",
                threshold=threshold,
                weight=weight,
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


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("surrogate", (False, True))
def test_split_phases_observe_receiver_changes_in_native_connection_order(
    compiled, surrogate
):
    model, native = _recurrent_network(surrogate), _recurrent_network(surrogate)
    connections = tuple(model.synapses.values())
    native_connections = tuple(native.synapses.values())
    assert len(connections) == 2
    plans, carries = zip(
        *(dn.func.make_functional_netcon(connection) for connection in connections),
        strict=True,
    )
    functional, tensors = dn.func.make_functional(model.cell, dt=DT)
    weights = tuple(
        carry.weights.detach().clone().requires_grad_() for carry in carries
    )
    native_weights = tuple(
        weight.detach().clone().requires_grad_() for weight in weights
    )
    for connection, weight in zip(native_connections, native_weights, strict=True):
        connection.weight.w = weight
    state = tensors.state
    initial_level = (
        state["mechanism_buffers"]["relay"]["level"].clone().requires_grad_()
    )
    native_level = initial_level.detach().clone().requires_grad_()
    state["mechanism_buffers"]["relay"]["level"] = initial_level
    native.cell.mech.relay.level = native_level
    connection_states = tuple(carry.state for carry in carries)

    def advance(states, population_state, values):
        updated = []
        for field, plan, carry, weight in zip(
            ("level", "mirror"), plans, states, values, strict=True
        ):
            carry, payload = plan.read_due(carry)
            population_state = functional.prepare_and_receive(
                tensors.parameters,
                tensors.constants,
                population_state,
                target="relay",
                payload=payload,
            )
            observed = population_state["mechanism_buffers"]["relay"][field]
            updated.append(plan.advance(weight, carry, observed))
        population_state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, population_state
        )
        return tuple(updated), population_state

    evaluate = (
        torch.compile(advance, backend="aot_eager", fullgraph=True)
        if compiled
        else advance
    )
    for index in range(8):
        connection_states, state = evaluate(connection_states, state, weights)
        native.step()
        observed = state["mechanism_buffers"]["relay"]
        for name in ("level", "calls", "ledger"):
            torch.testing.assert_close(
                observed[name], getattr(native.cell.mech.relay, name)
            )
        for carry, connection in zip(
            connection_states, native_connections, strict=True
        ):
            for name, value in connection.state_dict_for_checkpoint().items():
                torch.testing.assert_close(carry[name], value)
        if index == 0:
            # Both zero-payload receptions change the source. The second
            # observation reaches its threshold exactly. Pre-reading either
            # source before reception would miss these first-step events.
            torch.testing.assert_close(
                observed["level"], initial_level.new_tensor([[1.5]])
            )
            for carry, weight in zip(connection_states, weights, strict=True):
                assert carry["has_spiked"].all()
                torch.testing.assert_close(carry["delivery_buffer"].sum(), weight.sum())

    loss = state["mechanism_buffers"]["relay"]["level"].square().sum()
    loss = loss + 0.01 * state["mechanism_buffers"]["relay"]["ledger"].square().sum()
    native_loss = native.cell.mech.relay.level.square().sum()
    native_loss = native_loss + 0.01 * native.cell.mech.relay.ledger.square().sum()
    actual = torch.autograd.grad(loss, (initial_level, *weights))
    expected = torch.autograd.grad(native_loss, (native_level, *native_weights))
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-8, atol=2e-9)
