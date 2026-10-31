#!/usr/bin/env python3
"""Observe native and functional Networks with the same pure callback API.

Run from the repository root::

    python examples/native_network_functional_callbacks.py

The bounded CPU example connects unequal Populations and continues observations
across native run, checkpointed longrun and longrun. It compares scoped results,
Recorder cadence and parameter gradients with an independent functional Network,
and checks live-state restoration after a callback-only objective's backward.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT = 0.025
FIRST, CHECKPOINTED, LAST = 3, 3, 2
STEPS = FIRST + CHECKPOINTED + LAST
G = "integrator.mech.mechanisms.leak.g_param"


# region analysis
class VoltageSummary(dn.func.FunctionalCallback):
    """Accumulate post-step energy for the selected Population."""

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "initial_voltage": voltage,
            "energy": voltage.new_zeros(()),
            "steps": voltage.new_zeros((), dtype=torch.int64),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "initial_voltage": carry["initial_voltage"],
            "energy": carry["energy"] + voltage.square().mean(),
            "steps": carry["steps"] + 1,
        }, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


# endregion analysis


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


def make_network():
    # region network
    left = dn.SingleCompartment(
        N=1, C=2, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    right = dn.SingleCompartment(
        N=1, C=3, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    left.insert(pas.rename("leak"), g=0.001, e=-55.0)
    right.insert(pas.rename("leak"), g=0.001, e=-75.0)
    right.insert(graded_syn.rename("analog"), e=0.0)
    network = dn.Network({"left": left, "right": right})
    network.connect_continuous_one_to_one(
        left[:, 0],
        right[:, 0],
        right.mech.analog,
        pre_var="v",
        input="g_pre",
        weight=0.0005,
        delay=0.0,
        transform=release,
    )
    network.train()
    network.initialize(DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        train_delay_backend="dense",
    )
    # endregion network
    return network


def make_callbacks(target):
    # region callbacks
    scope = dn.callbacks.PopulationCallback
    return dn.func.make_callbacks(
        target,
        {
            "trace": dn.Recorder(("left.v", "right.v", "t")),
            "coarse": dn.Recorder(("left.v", "right.v", "t"), dt=2 * DT),
            "right_trace": scope("right", dn.Recorder(("v", "t"), dt=2 * DT)),
            "summary": scope("right", VoltageSummary()),
        },
    )
    # endregion callbacks


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        network = make_network()
        # region continuation
        callbacks = make_callbacks(network)
        first = network.run(FIRST * DT, callbacks=callbacks)
        # region checkpoint
        hook_loss, boundary_state, middle = network.longrun_checkpointed(
            CHECKPOINTED * DT,
            chunklength=2,
            callbacks=callbacks,
            callback_state=first.state,
            safe_checkpoint=True,
            return_final_state=True,
        )
        # With no native loss hook, gradients can use only pure observations.
        assert hook_loss is None and isinstance(boundary_state, dict)
        # endregion checkpoint
        last = network.longrun(
            LAST * DT,
            chunklength=2,
            callbacks=callbacks,
            callback_state=middle.state,
        )
        # endregion continuation
        joined = {
            name: {
                field: torch.cat(
                    tuple(result[name][field] for result in (first, middle, last))
                )
                for field in first[name]
            }
            for name in ("trace", "coarse", "right_trace")
        }
        assert joined["trace"]["left.v"].shape == (STEPS + 1, 1, 2)
        assert joined["trace"]["right.v"].shape == (STEPS + 1, 1, 3)
        assert joined["coarse"]["right.v"].shape == (STEPS // 2 + 1, 1, 3)
        torch.testing.assert_close(
            joined["right_trace"]["v"], joined["coarse"]["right.v"]
        )
        torch.testing.assert_close(joined["right_trace"]["t"], joined["coarse"]["t"])
        for field in ("left.v", "right.v", "t"):
            torch.testing.assert_close(
                joined["coarse"][field], joined["trace"][field][::2]
            )
        assert last["summary"]["steps"].item() == STEPS
        torch.testing.assert_close(
            last["summary"]["initial_voltage"], torch.full_like(network.right.v, -65.0)
        )

        # An independent Network supplies the functional value and gradient oracle.
        reference_network = make_network()
        reference, tensors = dn.func.make_functional_network(reference_network)
        reference_callbacks = make_callbacks(reference)
        final, auxiliary = reference.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=STEPS,
            callbacks=reference_callbacks,
        )
        expected = auxiliary["callbacks"]
        for name, trace in joined.items():
            torch.testing.assert_close(trace, expected[name])
        torch.testing.assert_close(last["summary"], expected["summary"])
        for name, population in network.populations.items():
            torch.testing.assert_close(
                population.v, final["populations"][name]["integrator"]["v"]
            )

        # region gradients
        native_loss = last["summary"]["energy"] + sum(
            joined["trace"][name].square().mean() for name in ("left.v", "right.v")
        )
        functional_loss = expected["summary"]["energy"] + sum(
            expected["trace"][name].square().mean() for name in ("left.v", "right.v")
        )
        native_parameters = tuple(
            network.populations[name].mech.leak.g_param for name in ("left", "right")
        )
        functional_parameters = tuple(
            tensors.parameters["populations"][name][G] for name in ("left", "right")
        )
        final_voltage = {
            name: population.v.detach().clone()
            for name, population in network.populations.items()
        }
        final_time = network.t.detach().clone()
        observed = torch.autograd.grad(native_loss, native_parameters)
        wanted = torch.autograd.grad(functional_loss, functional_parameters)
        for gradient, reference_gradient in zip(observed, wanted, strict=True):
            torch.testing.assert_close(gradient, reference_gradient)
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for name, population in network.populations.items():
            torch.testing.assert_close(population.v, final_voltage[name])
        torch.testing.assert_close(network.t, final_time)
        # endregion gradients
        print(
            "Native Network pure callbacks: scopes, cadence, continuation, "
            "gradients and checkpoint replay passed"
        )


if __name__ == "__main__":
    main()
