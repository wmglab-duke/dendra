#!/usr/bin/env python3
"""Lower an ordinary recurrent Network into one pure connected timestep.

Run from the repository root: python examples/functional_network_lowering.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import expsyn, graded_syn, pas

DT = 0.1
STEPS = 8
SPLIT = 3


class Release(torch.nn.Module):
    """An ordinary memoryless module with an explicit trainable coefficient."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    def forward(self, voltage):
        return torch.sigmoid(self.scale * (voltage + 60.0))


def make_network():
    # region standard_network
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        left = dn.SingleCompartment(N=2, C=1)
        right = dn.SingleCompartment(N=2, C=1)
        left.insert(pas, g=0.001, e=-55.0)
        left.insert(graded_syn.rename("graded"), e=0.0)
        right.insert(pas, g=0.001, e=-65.0)
        right.insert(expsyn.rename("event"), tau=2.0, e=0.0)
        network = dn.Network(
            {"left": left, "right": right},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            left[:],
            right[:],
            right.mech.event,
            pre_var="v",
            threshold=-64.0,
            weight=torch.tensor([0.003, 0.004]),  # expsyn weights: µS.
            delay=torch.tensor([0.1, 0.3]),  # ms.
        )
        network.connect_continuous_one_to_one(
            right[:],
            left[:],
            left.mech.graded,
            pre_var="v",
            input="g_pre",
            weight=torch.tensor([0.0005, 0.0008]),  # S/cm² for release gates.
            delay=torch.tensor([0.0, 0.2]),  # Immediate and delayed edges.
            transform=Release(),
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
    # endregion standard_network
    return network


def clone_carry(tree):
    """Save live carry while retaining the prefix's autograd history."""
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def assert_carry(actual, expected):
    actual_leaves, actual_spec = pytree.tree_flatten(actual)
    expected_leaves, expected_spec = pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for value, reference in zip(actual_leaves, expected_leaves, strict=True):
        torch.testing.assert_close(value, reference)


def functional_loss(state):
    return sum(
        population["integrator"]["v"].square().sum()
        for population in state["populations"].values()
    ) + sum(
        connection["delivery_buffer"].square().sum()
        for section in ("synapses", "continuous_synapses")
        for connection in state[section].values()
    )


def native_loss(network):
    return sum(pop.v.square().sum() for pop in network.populations.values()) + sum(
        connection.delivery_buffer.square().sum()
        for section in (network.synapses, network.continuous_synapses)
        for connection in section.values()
    )


def run_demo(*, compiled=False):
    model = make_network()
    reference = make_network()
    resumed = make_network()
    # region lower
    plan, tensors = dn.func.make_functional_network(model)
    parameters, constants, state = tensors
    execute = (
        torch.compile(plan.prepare_and_step, backend="aot_eager", fullgraph=True)
        if compiled
        else plan.prepare_and_step
    )
    # endregion lower
    original = clone_carry(state)
    trajectory = []
    saved = None
    for index in range(STEPS):
        # region step
        draw = plan.sample_step_draw(state)  # None for this Network: no NetStim.
        state, _ = execute(parameters, constants, state, netstim_draw=draw)
        # endregion step
        reference.step()
        assert_carry(state, plan.extract(reference).state)
        trajectory.append(clone_carry(state))
        if index + 1 == SPLIT:
            # region checkpoint
            saved = {
                "step": index + 1,
                "state": clone_carry(state),
                "parameters": parameters,
                "constants": constants,
            }
            # endregion checkpoint

    replay = saved["state"]
    for index in range(saved["step"], STEPS):
        replay, _ = execute(
            saved["parameters"],
            saved["constants"],
            replay,
            netstim_draw=plan.sample_step_draw(replay),
        )
        assert_carry(replay, trajectory[index])

    # The matching native model has the same parameters, but need not first run
    # the prefix: a complete Network commit includes its authoritative clock.
    # region commit
    plan.commit_state_(resumed, saved["state"])
    # endregion commit
    for index in range(saved["step"], STEPS):
        resumed.step()
        assert_carry(trajectory[index], plan.extract(resumed).state)

    event = next(iter(parameters["synapses"].values()))
    continuous = next(iter(parameters["continuous_synapses"].values()))
    targets = (
        event["weight.weights.0.param.rho"],
        continuous["weight.weights.0.param.rho"],
        continuous["transform_parameters"]["scale"],
    )
    gradients = torch.autograd.grad(functional_loss(state), targets, retain_graph=True)
    replay_gradients = torch.autograd.grad(functional_loss(replay), targets)
    native_event = next(iter(reference.synapses.values()))
    native_continuous = next(iter(reference.continuous_synapses.values()))
    native_gradients = torch.autograd.grad(
        native_loss(reference),
        (
            native_event.weight.weights[0].param.rho,
            native_continuous.weight.weights[0].param.rho,
            native_continuous.transform.scale,
        ),
    )
    for value, replay_value, native_value in zip(
        gradients, replay_gradients, native_gradients, strict=True
    ):
        torch.testing.assert_close(value, native_value)
        torch.testing.assert_close(value, replay_value)
        assert torch.isfinite(value).all()
    assert_carry(tensors.state, original)
    assert_carry(plan.extract().state, original)
    return state, gradients


def main():
    state, gradients = run_demo()
    print("Functional Network: native trajectories, gradients, and checkpoints match.")
    print("Completed steps:", state["clock"]["step"].item())
    print("Gradient norms:", [value.norm().item() for value in gradients])


if __name__ == "__main__":
    main()
