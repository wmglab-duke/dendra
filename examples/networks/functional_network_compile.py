#!/usr/bin/env python3
"""Compile an ordinary recurrent Network with shared trainable connections.

Run from the repository root:
    python examples/networks/functional_network_compile.py
or:
    python -m examples.networks.functional_network_compile
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn

if __package__:
    from .functional_network_rollout import DT, assert_tree, make_network
else:
    from functional_network_rollout import DT, assert_tree, make_network

STEPS = 3
TOTAL_STEPS = 2 * STEPS


def clone_tree(tree):
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def slice_inputs(inputs, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], inputs)


def objective(state, trace):
    return (
        sum(
            population["integrator"]["v"].square().mean()
            for population in state["populations"].values()
        )
        + 0.01 * trace["right.v"].square().mean()
    )


def run_demo():
    # region compile
    model = make_network()  # Ordinary, initialized two-Population Network.
    event_connections = tuple(model.synapses.values())
    event_connections[1].weight.weights[0].param = (
        event_connections[0].weight.weights[0].param
    )
    event_connections[1].weight.n = event_connections[0].weight.n
    model.initialize(DT)
    event_names = tuple(
        f"synapses.{name}.weight.weights.0.param.rho" for name in model.synapses
    )
    callbacks = {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        inference=False,
        initialization="snapshot",
        trainable=(
            event_names[0],
            "transform_parameters.scale",
            "netstim.parameters.noise",
        ),
        callbacks=callbacks,
        checkpoint_every=2,
        backend="aot_eager",
    )
    # endregion compile
    # region parameters
    shared_weight = runner.get_parameter(event_names[0])
    assert shared_weight is runner.get_parameter(event_names[1])
    assert isinstance(shared_weight, torch.nn.Parameter)
    assert (
        shared_weight.data_ptr()
        != event_connections[0].weight.weights[0].param.rho.data_ptr()
    )
    # Both qualified names refer to one independent optimizer leaf.
    assert (
        runner.named_parameters[event_names[0]]
        is runner.named_parameters[event_names[1]]
    )
    # endregion parameters
    source = clone_tree(runner.functional.extract().state)
    initial = clone_tree(runner.initial_state)

    # region inputs
    voltage = initial["populations"]["left"]["integrator"]["v"]
    current = voltage.new_full((TOTAL_STEPS, *voltage.shape), 0.02).requires_grad_()
    inputs = {"left": dn.func.Inputs(intra=current)}
    draws = runner.functional.sample_runtime_draws(initial, steps=TOTAL_STEPS)
    # endregion inputs
    # region restart
    prefix_inputs = slice_inputs(inputs, 0, STEPS)
    prefix_draws = slice_inputs(draws, 0, STEPS)
    prefix, prefix_aux = runner(inputs=prefix_inputs, draws=prefix_draws)
    repeated, repeated_aux = runner(inputs=prefix_inputs, draws=prefix_draws)
    assert_tree(repeated, prefix)
    assert_tree(repeated_aux["callbacks"]["trace"], prefix_aux["callbacks"]["trace"])
    # endregion restart
    # region continuation
    final, final_aux = runner(
        prefix_aux["continuation"],
        inputs=slice_inputs(inputs, STEPS, TOTAL_STEPS),
        draws=slice_inputs(draws, STEPS, TOTAL_STEPS),
    )
    trace = {
        name: torch.cat((prefix_aux["callbacks"]["trace"][name], values))
        for name, values in final_aux["callbacks"]["trace"].items()
    }
    assert trace["right.v"].shape[0] == TOTAL_STEPS + 1
    # endregion continuation
    # region reference
    plan = runner.functional
    reference_callbacks = plan.make_callbacks(callbacks)
    expected, expected_aux = plan.prepare_and_rollout(
        runner.parameters,
        runner.constants,
        runner.initial_state,
        inputs,
        draws=draws,
        callbacks=reference_callbacks,
    )
    expected_trace = expected_aux["callbacks"]["trace"]
    assert_tree(final, expected)
    assert_tree(trace, expected_trace)
    # endregion reference
    # region gradients
    targets = (
        shared_weight,
        runner.get_parameter("transform_parameters.scale"),
        runner.get_parameter("netstim.parameters.noise"),
        current,
    )
    gradients = torch.autograd.grad(objective(final, trace), targets)
    oracle = torch.autograd.grad(objective(expected, expected_trace), targets)
    for value, reference in zip(gradients, oracle, strict=True):
        torch.testing.assert_close(value, reference)
        assert torch.isfinite(value).all()
    # endregion gradients
    # region automatic-draws
    automatic, automatic_aux = runner(inputs=prefix_inputs)
    assert_tree(automatic, prefix)
    assert_tree(automatic_aux["callbacks"]["trace"], prefix_aux["callbacks"]["trace"])
    # endregion automatic-draws
    # Another backward starts from fresh preparation and the same snapshot.
    new_gradient = torch.autograd.grad(
        objective(automatic, automatic_aux["callbacks"]["trace"]), shared_weight
    )[0]
    assert torch.isfinite(new_gradient).all()
    assert_tree(runner.initial_state, initial)
    assert_tree(runner.functional.extract().state, source)
    # region report
    report = runner.execution_report()
    assert report.strategy == "compiled_scan"
    assert runner.kernel.capabilities.compiled_reverse_mode == "first_order"
    # endregion report
    return final, trace, gradients, report


def main():
    state, trace, gradients, report = run_demo()
    print(
        "Compiled Network: snapshot, continuation, shared parameters and gradients verified."
    )
    print("Completed steps:", state["clock"]["step"].item())
    print("Recorded samples:", trace["right.v"].shape[0])
    print("Gradient norms:", [value.norm().item() for value in gradients])
    print("Last dispatch:", report.strategy, "steps:", report.steps)


if __name__ == "__main__":
    main()
