#!/usr/bin/env python3
"""Shared interpolated delays, connected gradients, and checkpoint replay.

Run from the repository root: python examples/networks/functional_delay_interpolation.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import exp2syn, expsyn, pas

DT = 0.05
STEPS = 16
SPLIT = 7


class GateInput(Mechanism):
    Mechanism.CARRY("gate")


def make_network(*, taps=2, synapse="expsyn", batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        pre = dn.SingleCompartment(N=2, C=1)
        pre.insert(GateInput.rename("source"))
        post = dn.SingleCompartment(N=2, C=1)
        post.insert(pas, g=0.001, e=-65.0)
        kind = expsyn if synapse == "expsyn" else exp2syn
        kinetics = {"tau": 2.0} if synapse == "expsyn" else {"tau1": 0.3, "tau2": 2.0}
        post.insert(kind.rename("event"), e=0.0, **kinetics)
        # region declarations
        network = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            pre[:],
            post[:],
            post.mech.event,
            pre_var="mech.source.gate",
            threshold=None,
            weight=torch.tensor([0.002, 0.003]),
            delay=torch.tensor([0.065, 0.135]),  # Effective physical delays, ms.
        )
        if batch is not None:
            network.batch(batch)
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=True,
            taps=taps,
            sigma=0.35,
            diff_scheduled_times=True,
            train_delay_backend="dense",
        )
        # endregion declarations
        connection = next(iter(network.synapses.values()))
        repeats = connection._n_conn // 2
        connection.schedule(
            con_indices=torch.arange(connection._n_conn),
            times_ms=torch.tensor([0.115, 0.265]).repeat(repeats),
            weight=torch.tensor([0.3, -0.2]).repeat(repeats),
        )
    return network


def clone_carry(tree):
    """Independent storage retaining BPTT history for this fixed-parameter run."""
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def assert_carry(actual, expected):
    left, left_spec = pytree.tree_flatten(actual)
    right, right_spec = pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for value, reference in zip(left, right, strict=True):
        torch.testing.assert_close(value, reference)


def run_demo(*, taps=2, synapse="expsyn", batch=None, compiled=False):
    model = make_network(taps=taps, synapse=synapse, batch=batch)
    reference = make_network(taps=taps, synapse=synapse, batch=batch)
    resumed = make_network(taps=taps, synapse=synapse, batch=batch)
    connection = next(iter(model.synapses.values()))
    # region lower
    projection, delivery = dn.func.make_functional_netcon(connection)
    population, cell = dn.func.make_functional(model.post, dt=DT)
    weights, delays = delivery.weights, delivery.delays
    carry = delivery.state, cell.state
    # endregion lower

    def step(weights, delays, carry, gate):
        queue, state = carry
        # region step
        queue, payload = projection.read_due(queue)
        state = population.prepare_and_receive(
            cell.parameters, cell.constants, state, target="event", payload=payload
        )
        queue = projection.advance(
            weights, queue, gate, delays=delays, schedule=delivery.schedule
        )
        state, _ = population.prepare_and_step(cell.parameters, cell.constants, state)
        # endregion step
        return queue, state

    execute = (
        torch.compile(step, backend="aot_eager", fullgraph=True) if compiled else step
    )
    values = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.0],
            [0.25, 1.0],
            [0.0, 0.0],
            [0.0, -0.2],
            [0.5, 0.0],
            [0.0, 0.4],
            [0.0, 0.0],
            [0.2, 0.3],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    ).reshape(STEPS, 2, 1)
    if batch is not None:
        values = values.unsqueeze(1).repeat(1, batch, 1, 1)
    gates = values.clone().requires_grad_()
    native_gates = values.clone().requires_grad_()
    original = clone_carry(carry)
    origin = model.t.detach().clone()
    native_connection = next(iter(reference.synapses.values()))
    trajectory = []
    saved = None

    def assert_native(state, network):
        nc = next(iter(network.synapses.values()))
        assert_carry(state[0], projection.extract(nc).state)
        assert_carry(state[1], population.extract(network.post).state)

    for index in range(STEPS):
        carry = execute(weights, delays, carry, gates[index])
        reference.pre.mech.source.gate = native_gates[index]
        reference.step()
        assert_native(carry, reference)
        trajectory.append(clone_carry(carry))
        if index + 1 == SPLIT:
            # region checkpoint
            saved = {
                "integer_step": index + 1,
                "origin": origin.clone(),
                "carry": clone_carry(carry),
                "weights": weights,
                "delays": delays,
                "schedule": delivery.schedule,
            }
            # endregion checkpoint

    # region replay
    replay = saved["carry"]
    for index in range(saved["integer_step"], STEPS):
        replay = execute(saved["weights"], saved["delays"], replay, gates[index])
        assert_carry(replay, trajectory[index])
    # endregion replay

    # Align the standard Network's owner clock, then deliberately alter its
    # queue/receiver state before restoring the saved component carry.
    for index in range(SPLIT):
        resumed.pre.mech.source.gate = values[index]
        resumed.step()
    resumed_connection = next(iter(resumed.synapses.values()))
    resumed_connection.advance()
    # region commit
    projection.commit_state_(resumed_connection, saved["carry"][0])
    population.commit_state_(resumed.post, saved["carry"][1])
    for index in range(saved["integer_step"], STEPS):
        resumed.pre.mech.source.gate = values[index]
        resumed.step()
        assert_native(trajectory[index], resumed)
    # endregion commit

    targets = weights, delays, gates
    loss = (
        carry[1]["integrator"]["v"].square().sum()
        + carry[0]["delivery_buffer"].square().sum()
    )
    gradients = torch.autograd.grad(loss, targets, retain_graph=True)
    replay_loss = (
        replay[1]["integrator"]["v"].square().sum()
        + replay[0]["delivery_buffer"].square().sum()
    )
    replay_gradients = torch.autograd.grad(replay_loss, targets)
    native_loss = (
        reference.post.v.square().sum()
        + native_connection.delivery_buffer.square().sum()
    )
    native_gradients = torch.autograd.grad(
        native_loss,
        (native_connection.weight.w, native_connection.delay_ms.w, native_gates),
    )
    for gradient, replay_gradient, native_gradient in zip(
        gradients, replay_gradients, native_gradients, strict=True
    ):
        torch.testing.assert_close(gradient, native_gradient)
        torch.testing.assert_close(gradient, replay_gradient)
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    assert_carry(saved["carry"], trajectory[SPLIT - 1])
    assert_carry(
        (projection.extract().state, population.extract(model.post).state), original
    )
    torch.testing.assert_close(reference.t, origin + origin.new_tensor(STEPS) * DT)
    return gradients


def main():
    for taps in (2, 3):
        gradients = run_demo(taps=taps)
        print(
            f"{taps}-tap: native trajectories, delay gradients, and checkpoint replay match."
        )
        print("dloss/deffective_delays:", gradients[1].tolist())


if __name__ == "__main__":
    main()
