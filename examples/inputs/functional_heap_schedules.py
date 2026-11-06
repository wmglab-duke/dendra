#!/usr/bin/env python3
"""Consume shared heap calendars, checkpoint explicit carry, and resume both APIs.

Run from the repository root: python examples/inputs/functional_heap_schedules.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.05
STEPS = 12
SPLIT = 6


def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        stimulus = dn.NetStim(
            N=2,
            interval=[0.2, 0.3],
            noise=[0.4, 0.8],
            start=[0.0, 0.05],
            max_spikes=20,
            tau=0.05,
            seed=71,
        )
        post = dn.SingleCompartment(N=2, C=1)
        post.insert(pas, g=0.001, e=-65.0)
        post.insert(expsyn.rename("event"), tau=2.0, e=0.0)
        network = dn.Network(
            {"post": post},
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            stimulus[:],
            post[:],
            post.mech.event,
            weight=torch.tensor([0.002, 0.003]),
            delay=DT,
        )
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=True,
            train_delay_backend="dense",
        )
        # region declarations
        # Python unit-amplitude events select consuming heap storage.
        # Coincident events coalesce when the next sampled time is due.
        stimulus.schedule([0, 0, 0, 1, 1, 1], [0.075, 0.075, 0.14, 0.025, 0.24, 0.29])
        # Differentiable tensor events coexist with the heap calendar.
        stimulus.schedule(
            [0, 1],
            times=[0.12, 0.37],
            weight=[0.2, -0.1],
            differentiable=True,
        )
        # endregion declarations
    return network


def clone_carry(tree):
    """Copy storage while retaining BPTT history within this fixed-parameter run."""
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def functional_step(
    generator, source, projection, delivery, population, cell, prepared, carry, time
):
    source_state, queue, state = carry
    # region step
    draw = generator.sample_step_draw(source_state)  # Cold, independent sampling.
    source_state, output = generator.step(
        source.parameters,
        source.constants,
        source_state,
        time,
        draw,
        bptt=True,
        schedule=source.schedule,
    )
    queue, payload = projection.read_due(queue)
    state = population.receive(
        cell.parameters,
        prepared,
        state,
        target="event",
        payload=payload,
    )
    queue = projection.advance(delivery.weights, queue, output.spike_gate)
    state, _ = population.step(cell.parameters, prepared, state)
    # endregion step
    return (source_state, queue, state), output


def assert_native(carry, network):
    source_state, queue, state = carry
    checkpoint = network.netstim.state_dict_for_checkpoint()
    for name in source_state:
        torch.testing.assert_close(source_state[name], checkpoint[name])
    connection = next(iter(network.synapses.values()))
    for name, value in connection.state_dict_for_checkpoint().items():
        torch.testing.assert_close(queue[name], value)
    torch.testing.assert_close(state["integrator"]["v"], network.post.v)
    torch.testing.assert_close(
        state["mechanisms"]["event"]["g"], network.post.mech.event.g
    )


def assert_carry(actual, expected):
    for value, reference in zip(
        pytree.tree_leaves(actual), pytree.tree_leaves(expected), strict=True
    ):
        torch.testing.assert_close(value, reference)


def run_demo():
    model, reference, resumed = make_network(), make_network(), make_network()
    # Share the complete initial source checkpoint, including its lifecycle RNG.
    for network in (reference, resumed):
        network.netstim.restore_dict_from_checkpoint(
            model.netstim.state_dict_for_checkpoint()
        )
    native_connection = next(iter(model.synapses.values()))
    # region lower
    generator, source = dn.func.make_functional_netstim(model.netstim)
    projection, delivery = dn.func.make_functional_netcon(native_connection)
    population, cell = dn.func.make_functional(model.post, dt=DT)
    prepared = population.prepare(cell.parameters, cell.constants)
    carry = source.state, delivery.state, cell.state
    # heap_times and heap_row_offsets are fixed non-differentiable constants;
    # heap_cursor and optional legacy head overrides travel with source.state.
    # endregion lower
    original_rng = model.netstim.state_dict_for_checkpoint()["rng_state"].clone()
    origin = model.t.detach().clone()
    trajectory = []
    saved = None
    for index in range(STEPS):
        time = origin + origin.new_tensor(index) * DT
        carry, output = functional_step(
            generator,
            source,
            projection,
            delivery,
            population,
            cell,
            prepared,
            carry,
            time,
        )
        reference.step()
        assert_native(carry, reference)
        torch.testing.assert_close(output.spike_gate, reference.netstim.spike_gate)
        trajectory.append(clone_carry(carry))
        if index + 1 == SPLIT:
            # region checkpoint
            saved = {
                "integer_step": index + 1,
                "origin": origin.clone(),
                "carry": clone_carry(
                    carry
                ),  # Source clocks/cursor/RNG, queue, population.
                "source": source,  # Fixed parameter/constant/schedule snapshots.
                "delivery": delivery,
                "cell": cell,
            }
            # endregion checkpoint

    # region functional_resume
    replay = saved["carry"]
    for index in range(saved["integer_step"], STEPS):
        time = saved["origin"] + saved["origin"].new_tensor(index) * DT
        replay, _ = functional_step(
            generator,
            saved["source"],
            projection,
            saved["delivery"],
            population,
            saved["cell"],
            prepared,
            replay,
            time,
        )
        assert_carry(replay, trajectory[index])
    # endregion functional_resume

    # Bring the Network owner clock to the same complete timestep boundary.
    for _ in range(SPLIT):
        resumed.step()
    # Deliberately alter its source cursor, clock and RNG before restoring.
    resumed.netstim(1000.0, bptt=True)
    saved_source, saved_queue, saved_population = saved["carry"]
    # region native_resume
    generator.commit_state_(resumed.netstim, saved_source)
    resumed_connection = next(iter(resumed.synapses.values()))
    projection.commit_state_(resumed_connection, saved_queue)
    population.commit_state_(resumed.post, saved_population)
    for index in range(saved["integer_step"], STEPS):
        resumed.step()
        assert_native(trajectory[index], resumed)
    # endregion native_resume
    torch.testing.assert_close(
        resumed.netstim.state_dict_for_checkpoint()["seeder_state"],
        carry[0]["seeder_state"],
    )

    inputs = source.parameters["interval"], delivery.weights
    loss = carry[2]["integrator"]["v"].square().sum()
    gradients = torch.autograd.grad(loss, inputs, retain_graph=True)
    replay_gradients = torch.autograd.grad(
        replay[2]["integrator"]["v"].square().sum(), inputs
    )
    reference_connection = next(iter(reference.synapses.values()))
    native_gradients = torch.autograd.grad(
        reference.post.v.square().sum(),
        (reference.netstim.interval.rho, reference_connection.weight.w),
    )
    for gradient, replay_gradient in zip(gradients, replay_gradients, strict=True):
        torch.testing.assert_close(gradient, replay_gradient)
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    # Effective functional intervals are independent values; native intervals
    # are PositiveParam transforms of rho, so compare through that transform.
    torch.testing.assert_close(
        gradients[0] * torch.sigmoid(reference.netstim.interval.rho),
        native_gradients[0],
    )
    torch.testing.assert_close(gradients[1], native_gradients[1])
    assert torch.equal(
        model.netstim.state_dict_for_checkpoint()["rng_state"], original_rng
    )
    assert not model.netstim.spike_counts.any()
    torch.testing.assert_close(model.netstim.heap_cursor, source.state["heap_cursor"])
    assert not torch.equal(saved_source["heap_cursor"], source.state["heap_cursor"])
    assert_carry(saved["carry"], trajectory[SPLIT - 1])
    return gradients


def main():
    gradients = run_demo()
    print(
        "Heap calendars: native, functional replay, clocks, cursors, queues and RNG match."
    )
    print("dloss/dinterval:", gradients[0].tolist())
    print("dloss/dweights:", gradients[1].tolist())


if __name__ == "__main__":
    main()
