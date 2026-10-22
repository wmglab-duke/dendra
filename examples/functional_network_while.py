"""Run connected inference with shared while loops, recordings, and continuation.

Run from the repository root: python examples/functional_network_while.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn

if __package__:
    from .functional_network_rollout import DT, assert_tree, make_network, slice_inputs
else:
    from functional_network_rollout import DT, assert_tree, make_network, slice_inputs

STEPS = 7


def clone_tree(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def run_demo(*, initialization="snapshot", backend="aot_eager"):
    model = make_network()  # Two Populations, recurrent event/continuous traffic.
    connections = tuple(model.synapses.values())
    connections[1].weight.w = connections[0].weight.w
    with torch.no_grad():
        for _ in range(2):
            model.step()
    # region compile
    callbacks = {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution="while",
        inference=True,
        initialization=initialization,
        callbacks=callbacks,
        backend=backend,
    )
    reference = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution="scan",
        inference=True,
        initialization=initialization,
        callbacks=callbacks,
        backend=backend,
    )
    # endregion compile
    source = clone_tree(runner.functional.extract().state)
    initial = clone_tree(runner.initial_state)
    names = tuple(model.synapses)
    assert (
        runner.parameters["synapses"][names[0]]["weights"]
        is runner.parameters["synapses"][names[1]]["weights"]
    )
    # region inputs
    voltage = initial["populations"]["left"]["integrator"]["v"]
    current = (
        torch.linspace(0.0, 0.02, 2 * STEPS, dtype=voltage.dtype, device=voltage.device)
        .reshape(2 * STEPS, *([1] * voltage.ndim))
        .expand(2 * STEPS, *voltage.shape)
    )
    inputs = dn.func.NetworkRolloutInput(
        populations={"left": dn.func.RolloutInput(intra=current)},
        netstim=(
            runner.functional.sample_draw_tape(initial, 2 * STEPS)
            if initialization == "snapshot"
            else None
        ),
    )
    # endregion inputs
    first_inputs = slice_inputs(inputs, 0, STEPS)
    prefix, first = runner(inputs=first_inputs)
    expected, expected_aux = reference(inputs=first_inputs)
    assert_tree(prefix, expected)
    assert_tree(first["callbacks"]["trace"], expected_aux["callbacks"]["trace"])
    repeated, _ = runner(inputs=first_inputs)
    assert_tree(repeated, prefix)
    # region continuation
    final, second = runner(
        prefix,
        inputs=slice_inputs(inputs, STEPS, 2 * STEPS),
        callback_state=first["callbacks"].state,
    )
    expected, expected_aux = reference(
        expected,
        inputs=slice_inputs(inputs, STEPS, 2 * STEPS),
        callback_state=expected_aux["callbacks"].state,
    )
    trace = {
        name: torch.cat((first["callbacks"]["trace"][name], values))
        for name, values in second["callbacks"]["trace"].items()
    }
    # endregion continuation
    assert_tree(final, expected)
    assert_tree(second["callbacks"]["trace"], expected_aux["callbacks"]["trace"])
    assert trace["right.v"].shape[0] == 2 * STEPS + 1
    assert not trace["right.v"].requires_grad
    # region report
    report = runner.execution_report()
    assert report.strategy == "compiled_while_loop"
    assert report.requested_execution == "while"
    assert runner.kernel.capabilities.compiled_reverse_mode == "unsupported"
    # endregion report
    # region checkpoint
    resumed = make_network()
    shared = {}
    for name, connection in resumed.synapses.items():
        weight = runner.parameters["synapses"][name]["weights"]
        if id(weight) not in shared:
            shared[id(weight)] = weight.detach().clone()
        connection.weight.w = shared[id(weight)]
    runner.functional.commit_state_(resumed, final)
    with torch.no_grad():
        continued, _ = runner.functional.prepare_and_step(
            runner.parameters,
            runner.constants,
            final,
            netstim_draw=runner.functional.sample_step_draw(final),
        )
        resumed.step()
    assert_tree(continued, runner.functional.extract(resumed).state)
    # endregion checkpoint
    # region tail
    tail = runner.functional.compile_rollout_chunk(
        2, execution="while", backend=backend
    )
    with torch.no_grad():
        runner.functional.run(
            runner.parameters,
            runner.constants,
            final,
            steps=2,
            compiled_chunk=tail,
            chunklength=2,
        )
    assert tail.execution_report().strategy == "compiled_unrolled"
    # endregion tail
    assert_tree(runner.initial_state, initial)
    assert_tree(runner.functional.extract().state, source)
    return final, trace, report


def main():
    for initialization in ("snapshot", "fresh"):
        state, trace, report = run_demo(initialization=initialization)
        print(initialization, report.strategy, "step", state["clock"]["step"].item())
        print("Recorded samples:", trace["right.v"].shape[0])


if __name__ == "__main__":
    main()
