#!/usr/bin/env python3
"""One shared NetStim drives two functional projections and resumes natively.

Run from the repository root::

    python examples/functional_netstim_renewal.py
"""

# region network
from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.05


def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64):
        source = dn.NetStim(
            N=2,
            interval=[0.2, 0.3],
            noise=[0.4, 0.8],
            start=[0.0, 0.05],
            max_spikes=20,
            tau=0.05,
            seed=71,
            dtype=torch.float64,
        )
        targets = {}
        for name in ("a", "b"):
            cell = dn.SingleCompartment(N=2, C=1, dtype=torch.float64)
            cell.insert(pas, g=0.001, e=-65.0)
            cell.insert(expsyn.rename("syn"), tau=2.0, e=0.0)
            targets[name] = cell
        network = dn.Network(
            targets,
            netstim=source,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for name, weight in (("a", 0.002), ("b", 0.004)):
            target = targets[name]
            network.connect_one_to_one(
                source[:],
                target[:],
                target.mech.syn,
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


# endregion network


def run_demo():
    model, reference, resumed = make_network(), make_network(), make_network()
    # region lower
    generator, generator_tensors = dn.func.make_functional_netstim(model.netstim)
    parameters = generator_tensors.parameters
    constants = generator_tensors.constants
    source_state = generator_tensors.state
    # Reset purely using the same seeded initialization draw as native NetStim.
    initial_draw = generator.sample_initialization_draw(parameters, source_state)
    source_state = generator.initial_state(
        parameters, constants, source_state, initial_draw
    )
    # endregion lower
    original_source = model.netstim.state_dict_for_checkpoint()
    connections = tuple(model.synapses.values())
    plans, connection_tensors = zip(
        *(dn.func.make_functional_netcon(connection) for connection in connections),
        strict=True,
    )
    populations, population_tensors = zip(
        *(dn.func.make_functional(getattr(model, name), dt=DT) for name in ("a", "b")),
        strict=True,
    )
    prepared = tuple(
        plan.prepare(tensors.parameters, tensors.constants)
        for plan, tensors in zip(populations, population_tensors, strict=True)
    )
    connection_states = tuple(tensors.state for tensors in connection_tensors)
    population_states = tuple(tensors.state for tensors in population_tensors)
    weights = tuple(tensors.weights for tensors in connection_tensors)
    reference_weights = tuple(
        connection.weight.w for connection in reference.synapses.values()
    )
    time = model.t.detach().clone()

    for index in range(16):
        # region sample
        # Cold sampling uses an independent Generator restored from explicit state.
        draw = generator.sample_step_draw(source_state)
        # endregion sample
        # region generate
        source_state, output = generator.step(
            parameters,
            constants,
            source_state,
            time,
            draw,
            bptt=True,
        )
        # endregion generate
        updated_connections, received_populations = [], []
        for i, plan in enumerate(plans):
            # region project
            carry, payload = plan.read_due(connection_states[i])
            received = populations[i].receive(
                population_tensors[i].parameters,
                prepared[i],
                population_states[i],
                target="syn",
                payload=payload,
            )
            # Both projections consume this one generator invocation's gates.
            carry = plan.advance(weights[i], carry, output.spike_gate)
            # endregion project
            updated_connections.append(carry)
            received_populations.append(received)
        connection_states = tuple(updated_connections)
        population_states = tuple(
            plan.step(tensors.parameters, buffers, state)[0]
            for plan, tensors, buffers, state in zip(
                populations,
                population_tensors,
                prepared,
                received_populations,
                strict=True,
            )
        )
        reference.step()
        _assert_native(reference, source_state, connection_states, population_states)

        if index < 8:
            resumed.step()  # Align the Network and population clocks at the split.
        if index == 7:
            # Move this source ahead to demonstrate restoring clock AND stream.
            resumed.netstim(1000.0, bptt=False)
            # region commit
            generator.commit_state_(resumed.netstim, source_state)
            for plan, native, state in zip(
                plans,
                resumed.synapses.values(),
                connection_states,
                strict=True,
            ):
                plan.commit_state_(native, state)
            for name, plan, state in zip(
                ("a", "b"),
                populations,
                population_states,
                strict=True,
            ):
                plan.commit_state_(getattr(resumed, name), state)
            # endregion commit
        elif index >= 8:
            resumed.step()
            _assert_native(
                resumed,
                source_state,
                connection_states,
                population_states,
                check_seeder=True,
            )
        time = time + DT

    loss = sum(state["integrator"]["v"].square().sum() for state in population_states)
    gradients = torch.autograd.grad(loss, (parameters["interval"], *weights))
    reference_loss = reference.a.v.square().sum() + reference.b.v.square().sum()
    native_gradients = torch.autograd.grad(
        reference_loss, (reference.netstim.interval.rho, *reference_weights)
    )
    torch.testing.assert_close(
        gradients[0] * torch.sigmoid(reference.netstim.interval.rho),
        native_gradients[0],
    )
    for actual, expected in zip(gradients[1:], native_gradients[1:], strict=True):
        torch.testing.assert_close(actual, expected)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.count_nonzero(gradients[0])
    # Functional sampling and execution leave the source model untouched.
    untouched_source = model.netstim.state_dict_for_checkpoint()
    assert torch.equal(untouched_source["rng_state"], original_source["rng_state"])
    assert torch.equal(source_state["seeder_state"], original_source["seeder_state"])
    assert not model.netstim.spike_counts.any()
    return gradients


def _assert_native(
    network, source_state, connection_states, population_states, *, check_seeder=False
):
    for name in ("next_stoch_time", "spike_counts", "t_last"):
        torch.testing.assert_close(source_state[name], getattr(network.netstim, name))
    checkpoint = network.netstim.state_dict_for_checkpoint()
    assert torch.equal(source_state["rng_state"], checkpoint["rng_state"])
    if check_seeder:
        assert torch.equal(source_state["seeder_state"], checkpoint["seeder_state"])
    for carry, connection in zip(
        connection_states, network.synapses.values(), strict=True
    ):
        for name, value in connection.state_dict_for_checkpoint().items():
            torch.testing.assert_close(carry[name], value)
    for name, state in zip(("a", "b"), population_states, strict=True):
        torch.testing.assert_close(state["integrator"]["v"], getattr(network, name).v)
        torch.testing.assert_close(
            state["mechanisms"]["syn"]["g"],
            getattr(network, name).mech.syn.g,
        )


def main():
    gradients = run_demo()
    print(
        "Shared NetStim: native trajectories, queues, clock/RNG continuation, and gradients match."
    )
    print("dloss/dinterval:", gradients[0].tolist())


if __name__ == "__main__":
    main()
