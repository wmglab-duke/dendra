#!/usr/bin/env python3
"""Shared tensor schedules, indexed bindings, gradients, and explicit carry.

Run from the repository root: python examples/inputs/functional_tensor_schedules.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.05


def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        stimulus = dn.NetStim(N=2, start=10.0, seed=71)
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
            sched_width=1.3,
            train_delay_backend="dense",
        )
        connection = next(iter(network.synapses.values()))
        # region declarations
        times = torch.tensor([0.275, 0.335], dtype=torch.float64, requires_grad=True)
        amplitudes = torch.tensor([0.6, -0.2], dtype=torch.float64, requires_grad=True)
        # The same vectors are shared by source and connection schedules.
        for owner in (stimulus, connection):
            owner.bind_time_source(times)
            owner.bind_weight_source(amplitudes)
        stimulus.schedule(
            [0, 1],
            times=[0.065, 0.115],
            weight=[0.35, 0.2],
            differentiable=True,
        )
        stimulus.schedule_ref([1, 0], times=[0.165, 0.215], weight_idx=[1, 0])
        stimulus.schedule_time_ref([0, 1], time_idx=[0, 1], weight=[0.2, 0.3])
        connection.schedule(
            con_indices=[0, 1],
            times_ms=[0.085, 0.135],
            weight=torch.tensor([-0.1, 0.0], dtype=torch.float64),
        )
        connection.schedule_ref(
            con_indices=[1, 0], times_ms=[0.185, 0.235], weight_idx=[1, 0]
        )
        connection.schedule_time_ref(
            con_indices=[0, 1],
            time_idx=[0, 1],
            weight=torch.tensor([0.4, 0.2], dtype=torch.float64),
        )
        # endregion declarations
    return network, times, amplitudes


def run_demo():
    model, _, _ = make_network()
    reference, reference_times, reference_amplitudes = make_network()
    native_connection = next(iter(model.synapses.values()))
    # region lower
    generator, source = dn.func.make_functional_netstim(model.netstim)
    projection, delivery = dn.func.make_functional_netcon(native_connection)
    population, cell = dn.func.make_functional(model.post, dt=DT)
    # Extraction creates independent snapshots. Restore intentional sharing
    # explicitly, so both schedules differentiate the same optimizer leaves.
    times = source.schedule.time_source
    amplitudes = source.schedule.amplitude_source
    source_schedule = source.schedule
    connection_schedule = delivery.schedule._replace(
        time_source=times,
        amplitude_source=amplitudes,
    )
    source_state, queue, state = source.state, delivery.state, cell.state
    # endregion lower
    prepared = population.prepare(cell.parameters, cell.constants)
    original_rng = model.netstim.state_dict_for_checkpoint()["rng_state"].clone()
    origin = model.t.detach().clone()
    saved = None
    for index in range(12):
        # Match Network's origin + integer_step * dt clock without drift.
        time = origin + origin.new_tensor(index) * DT
        # region step
        draw = generator.sample_step_draw(source_state)  # Cold RNG sampling.
        source_state, output = generator.step(
            source.parameters,
            source.constants,
            source_state,
            time,
            draw,
            bptt=True,
            schedule=source_schedule,
        )
        queue, payload = projection.read_due(queue)
        state = population.receive(
            cell.parameters,
            prepared,
            state,
            target="event",
            payload=payload,
        )
        queue = projection.advance(
            delivery.weights,
            queue,
            output.spike_gate,
            schedule=connection_schedule,
        )
        state, _ = population.step(cell.parameters, prepared, state)
        # endregion step
        reference.step()
        torch.testing.assert_close(state["integrator"]["v"], reference.post.v)
        torch.testing.assert_close(output.spike_gate, reference.netstim.spike_gate)
        torch.testing.assert_close(output.spikes, reference.netstim.spikes)
        reference_connection = next(iter(reference.synapses.values()))
        for name, value in reference_connection.state_dict_for_checkpoint().items():
            torch.testing.assert_close(queue[name], value)
        source_checkpoint = reference.netstim.state_dict_for_checkpoint()
        for name in ("next_stoch_time", "spike_counts", "t_last", "rng_state"):
            torch.testing.assert_close(source_state[name], source_checkpoint[name])
        if index == 5:
            # region checkpoint
            saved = {
                "next_time": origin + origin.new_tensor(index + 1) * DT,
                "source_state": source_state,
                "queue": queue,
                "population_state": state,
                "source_schedule": source_schedule,
                "connection_schedule": connection_schedule,
                "weights": delivery.weights,
            }
            # endregion checkpoint

    gradients = torch.autograd.grad(
        state["integrator"]["v"].square().sum(),
        (times, amplitudes, delivery.weights),
    )
    reference_gradients = torch.autograd.grad(
        reference.post.v.square().sum(),
        (reference_times, reference_amplitudes, reference_connection.weight.w),
    )
    for gradient, expected in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(gradient, expected)
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)
    assert torch.equal(
        model.netstim.state_dict_for_checkpoint()["rng_state"], original_rng
    )
    assert not model.netstim.spike_counts.any()
    assert saved["next_time"] == 6 * DT
    assert (
        saved["source_state"]["spike_counts"].sum() < source_state["spike_counts"].sum()
    )
    return gradients


def main():
    gradients = run_demo()
    print(
        "Tensor schedules: native trajectories, queues, RNG state, and shared-source gradients match."
    )
    print("dloss/dshared_times:", gradients[0].tolist())
    print("dloss/dshared_amplitudes:", gradients[1].tolist())


if __name__ == "__main__":
    main()
