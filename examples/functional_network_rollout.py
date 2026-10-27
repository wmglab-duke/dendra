#!/usr/bin/env python3
"""Run, record, checkpoint, and resume an ordinary recurrent Network purely.

Run from the repository root: python examples/functional_network_rollout.py
For the compiled scan variant, import this module and call run_demo(compiled=True).
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
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    def forward(self, voltage):
        return torch.sigmoid(self.scale * (voltage + 60.0))


# region reducer
class VoltageEnergy(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["populations"]["right"]["integrator"]["v"]
        return voltage.new_zeros(()), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["populations"]["right"]["integrator"]["v"]
        return carry + voltage.square().mean(), None

    def finalize(self, carry, emissions):
        del emissions
        return carry


# endregion reducer


def make_network():
    # region network
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False, DEVICE="cpu"):
        left = dn.SingleCompartment(N=2, C=1)
        right = dn.SingleCompartment(N=2, C=1)
        left.insert(pas, g=0.001, e=-55.0)
        left.insert(graded_syn.rename("graded"), e=0.0)
        right.insert(pas, g=0.001, e=-65.0)
        right.insert(expsyn.rename("event"), tau=2.0, e=0.0)
        stimulus = dn.NetStim(
            N=2, interval=[0.2, 0.3], noise=0.6, start=0.0, max_spikes=20, seed=71
        )
        stimulus.noise.requires_grad_()
        network = dn.Network(
            {"left": left, "right": right},
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            left[:],
            right[:],
            right.mech.event,
            pre_var="v",
            threshold=-64.0,
            weight=torch.tensor([0.003, 0.004]),
            delay=torch.tensor([0.1, 0.3]),
        )
        network.connect_one_to_one(
            stimulus[:], right[:], right.mech.event, weight=0.002, delay=DT
        )
        network.connect_continuous_one_to_one(
            right[:],
            left[:],
            left.mech.graded,
            pre_var="v",
            input="g_pre",
            weight=torch.tensor([0.0005, 0.0008]),
            delay=torch.tensor([0.0, 0.2]),
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
    # endregion network
    return network


def clone_tree(tree):
    # Cloning preserves the prefix graph; detach only for independent saved runs.
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def assert_tree(actual, expected):
    values, spec = pytree.tree_flatten(actual)
    references, reference_spec = pytree.tree_flatten(expected)
    assert spec == reference_spec
    for value, reference in zip(values, references, strict=True):
        torch.testing.assert_close(value, reference)


def slice_inputs(inputs, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], inputs)


def objective(state, auxiliary):
    return (
        sum(
            population["integrator"]["v"].square().mean()
            for population in state["populations"].values()
        )
        + 0.01 * auxiliary["callbacks"]["trace"]["right.v"].square().mean()
    )


def run_demo(*, compiled=False):
    model = make_network()
    # region inputs
    plan, tensors = dn.func.make_functional_network(model)
    parameters, constants, initial = tensors
    left_v = initial["populations"]["left"]["integrator"]["v"]
    right_v = initial["populations"]["right"]["integrator"]["v"]
    left_current = left_v.new_full((STEPS, *left_v.shape), 0.02).requires_grad_()
    right_current = right_v.new_full((STEPS, *right_v.shape), -0.01)
    draws = plan.sample_draw_tape(initial, STEPS)
    inputs = dn.func.NetworkRolloutInput(
        populations={
            "left": dn.func.RolloutInput(intra=left_current),
            "right": dn.func.RolloutInput(intra=right_current),
        },
        netstim=draws,
    )
    # endregion inputs
    original = clone_tree(initial)
    # region callbacks
    callbacks = plan.make_callbacks(
        {
            "trace": dn.func.Recorder(("left.v", "right.v", "right.mech.event.g", "t")),
            "energy": VoltageEnergy(),
        }
    )
    # endregion callbacks
    # region eager
    prepared = plan.prepare(parameters, constants)
    expected, expected_aux = plan.rollout(
        parameters, prepared, initial, inputs, callbacks=callbacks
    )
    # endregion eager
    # region chunk
    chunk = (
        plan.compile_rollout_chunk(3, execution="scan", backend="aot_eager")
        if compiled
        else None
    )
    # endregion chunk
    # region run
    bound = (
        plan.bind(parameters, constants)
        if chunk is None
        else chunk.bind(parameters, constants)
    )
    final, auxiliary = dn.func.run(
        bound,
        initial,
        inputs=inputs,
        checkpoint_every=4,
        callbacks=callbacks,
    )
    trace = auxiliary["callbacks"]["trace"]
    assert trace["right.v"].shape[0] == STEPS + 1
    # endregion run
    assert_tree(final, expected)
    assert_tree(trace, expected_aux["callbacks"]["trace"])

    # region continuation
    prefix, prefix_aux = dn.func.run(
        bound,
        initial,
        inputs=slice_inputs(inputs, 0, SPLIT),
        callbacks=callbacks,
    )
    saved = {
        "state": clone_tree(prefix),
        "callbacks": clone_tree(prefix_aux["callbacks"].state),
        "step": SPLIT,
    }
    resumed, resumed_aux = dn.func.run(
        bound,
        saved["state"],
        inputs=slice_inputs(inputs, saved["step"], STEPS),
        callbacks=callbacks,
        callback_state=saved["callbacks"],
    )
    joined = {
        name: torch.cat((prefix_aux["callbacks"]["trace"][name], values))
        for name, values in resumed_aux["callbacks"]["trace"].items()
    }
    # endregion continuation
    assert_tree(resumed, expected)
    assert_tree(joined, trace)
    torch.testing.assert_close(
        resumed_aux["callbacks"]["energy"], auxiliary["callbacks"]["energy"]
    )

    # region gradients
    event = next(iter(parameters["synapses"].values()))
    continuous = next(iter(parameters["continuous_synapses"].values()))
    targets = (
        event["weight.weights.0.param.rho"],
        continuous["weight.weights.0.param.rho"],
        continuous["transform_parameters"]["scale"],
        parameters["netstim"]["parameters"]["noise"],
        left_current,
    )
    gradients = torch.autograd.grad(objective(final, auxiliary), targets)
    reference_gradients = torch.autograd.grad(
        objective(expected, expected_aux), targets
    )
    for value, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(value, reference)
        assert torch.isfinite(value).all()
    # endregion gradients

    # A complete boundary also resumes natively, including the NetStim RNG.
    # The next step has no explicit input current in either execution path.
    native = make_network()
    plan.commit_state_(native, final)
    next_state, _ = plan.prepare_and_step(
        parameters, constants, final, netstim_draw=plan.sample_step_draw(final)
    )
    native.step()
    assert_tree(next_state, plan.extract(native).state)
    assert_tree(initial, original)
    assert_tree(plan.extract().state, original)
    return final, trace, gradients


def main():
    state, trace, gradients = run_demo()
    print(
        "Network rollout: checkpointed values, recordings, and gradients match eager."
    )
    print("Completed steps:", state["clock"]["step"].item())
    print("Recorded samples:", trace["right.v"].shape[0])
    print("Gradient norms:", [value.norm().item() for value in gradients])


if __name__ == "__main__":
    main()
