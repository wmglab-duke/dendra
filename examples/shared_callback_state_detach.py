#!/usr/bin/env python3
"""Train native Population segments with explicit pure callback boundaries.

Run from the repository root::

    python examples/shared_callback_state_detach.py

The bounded CPU example continues run, step, checkpointed longrun and longrun.
It checks model/callback gradient boundaries separately, trainable LFP fields,
source ownership, reducer totals and Recorder cadence.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025
SEGMENTS = (("run", 3), ("step", 1), ("checkpointed", 3), ("longrun", 2))


class VoltageEnergy(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "energy": voltage.new_zeros(()),
            "steps": voltage.new_zeros((), dtype=torch.int64),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return {
            "energy": carry["energy"] + (state["integrator"]["v"] / 65).square().mean(),
            "steps": carry["steps"] + 1,
        }, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


def make_population():
    model = dn.SingleCompartment(
        N=1,
        C=2,
        v_init=-65.0,
        integrator=dn.bwd_euler_sc(imem=True),
    )
    model.insert(pas.rename("leak"), g=0.001, e=-55.0)
    model[:, 0].inject(dn.constant(value=1e-6))
    model.train()
    model.initialize()
    return model


def advance(model, method, steps, callbacks, carry):
    options = {"dt": DT, "callbacks": callbacks, "callback_state": carry}
    if method == "step":
        assert steps == 1
        return dn.step(model, **options)
    if method == "checkpointed":
        hook_loss, results = model.longrun_checkpointed(
            tstop=steps * DT, chunklength=2, safe_checkpoint=True, **options
        )
        assert hook_loss is None
        return results
    if method == "longrun":
        return model.longrun(tstop=steps * DT, chunklength=2, **options)
    return model.run(tstop=steps * DT, **options)


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=True):
        model = make_population()
        conductance = model.mech.leak.g_param
        field = torch.tensor([[1.0, 0.5]], dtype=torch.float64, requires_grad=True)
        # region preparation
        observations = dn.func.make_callbacks(
            model,
            {
                "trace": dn.Recorder(("v", "t"), dt=2 * DT),
                "currents": dn.Recorder(("i_membrane",)),
                "lfp": dn.callbacks.LFP([field, field]),
                "summary": VoltageEnergy(),
            },
            dt=DT,
        )
        optimizer = torch.optim.SGD(
            [
                {"params": [conductance], "lr": 1e-8},
                {"params": [field], "lr": 1e-4},
            ]
        )
        # endregion preparation
        carry = None
        previous_voltage = None
        times = []
        accepted = 0
        for method, steps in SEGMENTS:
            optimizer.zero_grad()
            results = advance(model, method, steps, observations, carry)
            loss = (
                (results["trace"]["v"] / 65).square().mean()
                + results["summary"]["energy"]
                + (results["lfp"] * 1e6).square().mean()
            )
            current = results["currents"]["i_membrane"]
            reference = (current * field).sum((-1, -2)).unsqueeze(-1).expand(-1, 2)
            torch.testing.assert_close(results["lfp"], reference)
            observed = torch.autograd.grad(loss, field, retain_graph=True)[0]
            expected = torch.autograd.grad(
                (reference * 1e6).square().mean(), field, retain_graph=True
            )[0]
            torch.testing.assert_close(observed, expected)
            if previous_voltage is not None:
                assert (
                    torch.autograd.grad(
                        loss, previous_voltage, allow_unused=True, retain_graph=True
                    )[0]
                    is None
                )
            previous_voltage = model.v
            # region boundary
            loss.backward()
            optimizer.step()
            model.detach()  # The native Population owns its simulation state.
            carry = observations.detach_state(results.state)
            # endregion boundary
            for parameter in (conductance, field):
                assert parameter.grad is not None
                assert (
                    torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                )
            assert model.mech.leak.g_param is conductance
            assert model.v.grad_fn is None
            assert carry.plan_token is results.state.plan_token
            assert carry["summary"]["energy"].grad_fn is None
            torch.testing.assert_close(carry["summary"], results["summary"])
            accepted += steps
            assert carry["summary"]["steps"].item() == accepted
            assert carry["trace"].item() == accepted
            times.append(results["trace"]["t"].detach())
        torch.testing.assert_close(
            torch.cat(times), torch.arange(0, accepted + 1, 2, dtype=torch.float64) * DT
        )
        torch.testing.assert_close(model.t, model.t.new_tensor(accepted * DT))
        print(
            "Native callback boundaries: model/source ownership, local gradients, "
            "LFP field refresh and continued sampling passed"
        )


if __name__ == "__main__":
    main()
