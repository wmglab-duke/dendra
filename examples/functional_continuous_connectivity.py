#!/usr/bin/env python3
"""Shared continuous projections, connected gradients, and saved-state replay.

Run from the repository root: python examples/functional_continuous_connectivity.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import graded_syn, pas

DT = 0.1
STEPS = 8
SPLIT = 3


class GateInput(Mechanism):
    Mechanism.CARRY("gate")


# region transforms
class ReleaseCurve(torch.nn.Module):
    """A custom memoryless transform with trainable parameters and a buffer."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.1, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))
        self.register_buffer("offset", torch.tensor(0.05, dtype=torch.float64))

    def forward(self, gate):
        return torch.sigmoid(self.scale * gate + self.bias) + self.offset


def signed_release(gate):
    return torch.tanh(gate)


# endregion transforms


def make_network(*, batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        sources = [dn.SingleCompartment(N=2, C=1) for _ in range(2)]
        for source in sources:
            source.insert(GateInput.rename("source"))
        post = dn.SingleCompartment(N=2, C=1)
        post.insert(pas, g=0.001, e=-65.0)
        post.insert(graded_syn.rename("graded"), e=-10.0)
        # region declarations
        network = dn.Network({"pre_a": sources[0], "pre_b": sources[1], "post": post})
        for source, weight, delay, transform in zip(
            sources,
            ([0.002, 0.003], [0.001, 0.004]),  # S/cm² for dimensionless gates.
            ([0.0, 0.2], [0.1, 0.3]),  # Mixed immediate/delayed, then all delayed.
            (ReleaseCurve(), signed_release),
            strict=True,
        ):
            network.connect_continuous_one_to_one(
                source[:],
                post[:],
                post.mech.graded,
                pre_var="mech.source.gate",
                input="g_pre",
                weight=torch.tensor(weight),
                delay=torch.tensor(delay),
                transform=transform,
            )
        # endregion declarations
        if batch is not None:
            network.batch(batch)
        network.train()
        network.initialize(DT)
    return network


def clone_carry(tree):
    """Clone live carry while retaining gradients through the saved prefix."""
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def assert_carry(actual, expected):
    left, left_spec = pytree.tree_flatten(actual)
    right, right_spec = pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for value, reference in zip(left, right, strict=True):
        torch.testing.assert_close(value, reference)


def run_demo(*, batch=None, compiled=False):
    model = make_network(batch=batch)
    reference = make_network(batch=batch)
    resumed = make_network(batch=batch)
    # region lower
    lowered = tuple(
        dn.func.make_functional_continuous_con(connection)
        for connection in model.continuous_synapses.values()
    )
    projections, connections = zip(*lowered, strict=True)
    population, cell = dn.func.make_functional(model.post, dt=DT)
    weights = tuple(connection.weights for connection in connections)
    transform_parameters = tuple(
        connection.transform_parameters for connection in connections
    )
    carry = tuple(connection.state for connection in connections), cell.state
    # endregion lower

    def step(weights, transform_parameters, carry, source_values):
        queues, state = carry
        # region step
        state, scratch = population.reset_continuous_inputs(state)
        next_queues = []
        for index, projection in enumerate(projections):
            queue = queues[index]
            if not projection.has_connections:
                queue, _, _ = projection.advance(
                    weights[index],
                    queue,
                    source_values[index],
                    transform_parameters=transform_parameters[index],
                )
                next_queues.append(queue)
                continue
            if projection.receive_before_source:
                _, payload, mask = projection.read_due(queue)
                state, scratch = population.prepare_and_continuous_receive(
                    cell.parameters,
                    cell.constants,
                    state,
                    scratch,
                    target="graded",
                    payload=payload,
                    input=projection.input,
                    reduce=projection.reduce,
                    mask=mask,
                )
            queue, payload, mask = projection.advance(
                weights[index],
                queue,
                source_values[index],
                transform_parameters=transform_parameters[index],
            )
            if not projection.receive_before_source:
                state, scratch = population.prepare_and_continuous_receive(
                    cell.parameters,
                    cell.constants,
                    state,
                    scratch,
                    target="graded",
                    payload=payload,
                    input=projection.input,
                    reduce=projection.reduce,
                    mask=mask,
                )
            next_queues.append(queue)
        state, _ = population.prepare_and_step(cell.parameters, cell.constants, state)
        # endregion step
        return tuple(next_queues), state

    execute = (
        torch.compile(step, backend="aot_eager", fullgraph=True) if compiled else step
    )
    values = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.0, 0.5], [0.4, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.25, 1.0], [0.5, 0.25]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, -0.1], [0.2, 0.0]],
            [[0.5, 0.0], [0.0, 0.4]],
            [[0.0, 0.4], [0.3, 0.0]],
        ],
        dtype=torch.float64,
    ).reshape(STEPS, 2, 2, 1)
    if batch is not None:
        values = values.unsqueeze(2).repeat(1, 1, batch, 1, 1)
    gates = values.clone().requires_grad_()
    native_gates = values.clone().requires_grad_()
    original = clone_carry(carry)
    trajectory = []
    saved = None

    def assert_native(current, network):
        for projection, queue, connection in zip(
            projections, current[0], network.continuous_synapses.values(), strict=True
        ):
            assert_carry(queue, projection.extract(connection).state)
        assert_carry(current[1], population.extract(network.post).state)

    for index in range(STEPS):
        carry = execute(weights, transform_parameters, carry, gates[index])
        for source, gate in zip(
            (reference.pre_a, reference.pre_b), native_gates[index], strict=True
        ):
            source.mech.source.gate = gate
        reference.step()
        assert_native(carry, reference)
        trajectory.append(clone_carry(carry))
        if index + 1 == SPLIT:
            # region checkpoint
            saved = {
                "step": index + 1,
                "carry": clone_carry(carry),
                "weights": weights,
                "transform_parameters": transform_parameters,
            }
            # endregion checkpoint

    # region replay
    replay = saved["carry"]
    for index in range(saved["step"], STEPS):
        replay = execute(
            saved["weights"], saved["transform_parameters"], replay, gates[index]
        )
        assert_carry(replay, trajectory[index])
    # endregion replay

    # Synchronize the native owner clock before committing a complete boundary.
    for index in range(SPLIT):
        for source, gate in zip(
            (resumed.pre_a, resumed.pre_b), values[index], strict=True
        ):
            source.mech.source.gate = gate
        resumed.step()
    # region commit
    for projection, connection, queue in zip(
        projections,
        resumed.continuous_synapses.values(),
        saved["carry"][0],
        strict=True,
    ):
        projection.commit_state_(connection, queue)
    population.commit_state_(resumed.post, saved["carry"][1])
    # endregion commit
    for index in range(SPLIT, STEPS):
        for source, gate in zip(
            (resumed.pre_a, resumed.pre_b), values[index], strict=True
        ):
            source.mech.source.gate = gate
        resumed.step()
        assert_native(trajectory[index], resumed)

    def loss(current):
        return current[1]["integrator"]["v"].square().sum() + sum(
            queue["delivery_buffer"].square().sum() for queue in current[0]
        )

    transform_trainables = tuple(
        value
        for mapping in transform_parameters
        if mapping is not None
        for value in mapping.values()
        if value.requires_grad
    )
    targets = (*weights, *transform_trainables, gates)
    gradients = torch.autograd.grad(loss(carry), targets, retain_graph=True)
    replay_gradients = torch.autograd.grad(loss(replay), targets)
    native_connections = tuple(reference.continuous_synapses.values())
    native_loss = reference.post.v.square().sum() + sum(
        connection.delivery_buffer.square().sum() for connection in native_connections
    )
    native_transform_parameters = tuple(
        value
        for connection in native_connections
        if isinstance(connection.transform, torch.nn.Module)
        for value in connection.transform.parameters()
        if value.requires_grad
    )
    native_gradients = torch.autograd.grad(
        native_loss,
        (
            *(connection.weight.w for connection in native_connections),
            *native_transform_parameters,
            native_gates,
        ),
    )
    for gradient, replay_gradient, native_gradient in zip(
        gradients, replay_gradients, native_gradients, strict=True
    ):
        torch.testing.assert_close(gradient, native_gradient)
        torch.testing.assert_close(gradient, replay_gradient)
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    assert_carry(
        (
            tuple(projection.extract().state for projection in projections),
            population.extract(model.post).state,
        ),
        original,
    )
    assert reference.post.mech.graded._continuous_reset_count.item() == STEPS
    return gradients


def main():
    gradients = run_demo()
    print("Shared continuous core: native trajectories, gradients, and replay match.")
    print(
        "Weight/transform gradients:",
        [gradient.tolist() for gradient in gradients[:-1]],
    )


if __name__ == "__main__":
    main()
