#!/usr/bin/env python3
"""Fresh Network initialization, explicit random draws, and continued gradients.

Run from the repository root:
    python examples/initialization/functional_network_initialization.py
or:
    python -m examples.initialization.functional_network_initialization
"""

from __future__ import annotations

import torch

import dendra as dn

if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.networks.functional_network_compile import (
    clone_tree,
    objective,
    slice_inputs,
)
from examples.networks.functional_network_rollout import DT, assert_tree, make_network

STEPS = 3


def run_demo():
    # region fresh-compile
    model = make_network()
    event_names = tuple(
        f"synapses.{name}.weight.weights.0.param.rho" for name in model.synapses
    )
    callbacks = {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=(event_names[0],),
        callbacks=callbacks,
        checkpoint_every=2,
        backend="aot_eager",
    )
    # endregion fresh-compile
    source = clone_tree(runner.functional.extract().state)
    # region initialization-inputs
    defaults = runner.initialization_inputs
    populations = dict(defaults.populations)
    initial_voltage = populations["left"].v_init.detach().clone().requires_grad_()
    populations["left"] = populations["left"]._replace(v_init=initial_voltage)
    draw = runner.functional.sample_initialization_draw(
        runner.parameters, runner.initial_state
    )
    initialization = defaults._replace(populations=populations, netstim_draw=draw)
    # endregion initialization-inputs
    # region explicit-initialization
    initialized = runner.functional.initialize(
        runner.parameters,
        runner.constants,
        runner.initial_state,
        initialization,
    )
    inputs = dn.func.NetworkRolloutInput(
        netstim=runner.functional.sample_draw_tape(initialized.state, 2 * STEPS)
    )
    # endregion explicit-initialization
    # region fresh-continuation
    prefix, prefix_aux = runner(
        initialization=initialization, inputs=slice_inputs(inputs, 0, STEPS)
    )
    final, final_aux = runner(
        prefix,
        parameters=prefix_aux["parameters"],
        constants=prefix_aux["constants"],
        inputs=slice_inputs(inputs, STEPS, 2 * STEPS),
        callback_state=prefix_aux["callbacks"].state,
    )
    trace = {
        name: torch.cat((prefix_aux["callbacks"]["trace"][name], values))
        for name, values in final_aux["callbacks"]["trace"].items()
    }
    # endregion fresh-continuation
    expected, expected_aux = runner.functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        inputs,
        callbacks=runner.functional.make_callbacks(callbacks),
    )
    expected_trace = expected_aux["callbacks"]["trace"]
    assert_tree(final, expected)
    assert_tree(trace, expected_trace)
    # region initialization-gradients
    targets = (initial_voltage, runner.get_parameter(event_names[0]))
    gradients = torch.autograd.grad(objective(final, trace), targets)
    oracle = torch.autograd.grad(objective(expected, expected_trace), targets)
    for value, reference in zip(gradients, oracle, strict=True):
        torch.testing.assert_close(value, reference)
        assert torch.isfinite(value).all()
    # endregion initialization-gradients
    assert_tree(runner.functional.extract().state, source)
    return final, trace, gradients


def main():
    final, trace, gradients = run_demo()
    print("Fresh Network initialization and continued checkpoint gradients verified.")
    print("Completed steps:", final["clock"]["step"].item())
    print("Recorded samples:", trace["right.v"].shape[0])
    print("Gradient norms:", [value.norm().item() for value in gradients])


if __name__ == "__main__":
    main()
