#!/usr/bin/env python3
"""Use one pure analysis with native and functional Population execution.

Run from the repository root::

    python examples/native_functional_callbacks.py

The small CPU example continues pure callback carry across native run,
longrun and checkpointed longrun, compares functional results, and checks
recording cadence, gradients and checkpoint replay isolation.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025
FIRST, MIDDLE, LAST = 3, 2, 3
STEPS = FIRST + MIDDLE + LAST


# region analysis
class VoltageSummary(dn.func.FunctionalCallback):
    """Retain the initial voltage and accumulate post-step voltage energy."""

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "initial_voltage": voltage,
            "energy": voltage.new_zeros(()),
            "steps": voltage.new_zeros((), dtype=torch.int64),
        }, None

    def update(self, carry, state, auxiliary):
        del state
        voltage = auxiliary["v"]
        return {
            "initial_voltage": carry["initial_voltage"],
            "energy": carry["energy"] + voltage.square().mean(),
            "steps": carry["steps"] + 1,
        }, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


# endregion analysis


class NativeVoltageLoss(dn.callbacks.Callback):
    """An independent replay-safe native loss hook."""

    def post_step_hook(self, model):
        return model.v.square().mean()


def make_model():
    model = dn.SingleCompartment(
        N=1, C=2, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=-55.0)
    model.train()
    model.initialize()
    return model


def make_callbacks(target):
    return dn.func.make_callbacks(
        target,
        {
            "summary": VoltageSummary(),
            "trace": dn.Recorder(("v", "t")),
            "coarse": dn.Recorder(("v", "t"), dt=2 * DT),
        },
        dt=DT,
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        model = make_model()
        # region binding
        callbacks = make_callbacks(model)
        first = model.run(
            tstop=FIRST * DT,
            dt=DT,
            callbacks=callbacks,
        )
        second = model.longrun(
            tstop=MIDDLE * DT,
            chunklength=2,
            dt=DT,
            callbacks=callbacks,
            callback_state=first.state,
        )
        # endregion binding
        # region checkpoint
        native_loss, last = model.longrun_checkpointed(
            tstop=LAST * DT,
            chunklength=2,
            dt=DT,
            callbacks=[NativeVoltageLoss()],
            functional_callbacks=callbacks,
            callback_state=second.state,
            safe_checkpoint=True,
        )
        # A native hook list coexists with the prepared pure collection through
        # functional_callbacks; their loss and observation outputs stay separate.
        energy = last["summary"]["energy"]
        # endregion checkpoint
        assert last["summary"]["steps"].item() == STEPS
        torch.testing.assert_close(
            last["summary"]["initial_voltage"], torch.full_like(model.v, -65.0)
        )
        joined = {
            name: {
                field: torch.cat(
                    tuple(result[name][field] for result in (first, second, last))
                )
                for field in ("v", "t")
            }
            for name in ("trace", "coarse")
        }
        assert joined["trace"]["v"].shape == (STEPS + 1, 1, 2)
        assert joined["coarse"]["v"].shape == (STEPS // 2 + 1, 1, 2)
        torch.testing.assert_close(joined["coarse"]["v"], joined["trace"]["v"][::2])

        # The same callback definition observes an independent functional model.
        reference_model = make_model()
        reference, tensors = dn.func.make_functional(reference_model, dt=DT)
        reference_callbacks = make_callbacks(reference)
        final, auxiliary = reference.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=STEPS,
            callbacks=reference_callbacks,
        )
        wanted = auxiliary["callbacks"]
        torch.testing.assert_close(model.v, final["integrator"]["v"])
        for name, trace in joined.items():
            torch.testing.assert_close(trace, wanted[name])
        torch.testing.assert_close(last["summary"], wanted["summary"])

        # region gradients
        reference_native_loss = (
            wanted["trace"]["v"][-LAST:].square().flatten(1).mean(1).sum()
        )
        torch.testing.assert_close(native_loss, reference_native_loss)
        final_voltage = model.v.detach().clone()
        observed = torch.autograd.grad(energy + native_loss, model.mech.leak.g_param)[0]
        expected = torch.autograd.grad(
            wanted["summary"]["energy"] + reference_native_loss,
            reference_model.mech.leak.g_param,
        )[0]
        torch.testing.assert_close(observed, expected)
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0
        torch.testing.assert_close(model.v, final_voltage)
        # endregion gradients
        print(
            "Native pure callbacks: joined results, cadence, gradients and replay passed"
        )


if __name__ == "__main__":
    main()
