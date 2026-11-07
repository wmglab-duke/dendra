#!/usr/bin/env python3
"""Train continued segments while retaining optimizer-owned source tensors.

Run from the repository root::

    python examples/training/functional_continuation_detach.py

The bounded CPU example covers Population and Network runners, shared model
parameters, trainable LFP fields, callback cadence and a cumulative reducer.
Each segment has its own backward pass and optimizer update. The next segment
retains numerical history while its gradients stop at the explicit boundary.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT, SEGMENT_STEPS, SEGMENTS = 0.025, 3, 3


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


def cell(compartments, reversal, *, conductance):
    model = dn.SingleCompartment(
        N=1,
        C=compartments,
        v_init=-65.0,
        integrator=dn.bwd_euler_sc(imem=True),
    )
    model.insert(pas.rename("leak"), g=conductance, e=reversal)
    return model


def make_model(network):
    conductance = torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float64))
    observed = cell(2, -55.0, conductance=conductance)
    model = observed
    if network:
        other = cell(1, -75.0, conductance=conductance)
        model = dn.Network({"observed": observed, "other": other})
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def make_program(model, field, network):
    def scope(callback):
        return (
            dn.callbacks.PopulationCallback("observed", callback)
            if network
            else callback
        )

    voltage = "observed.v" if network else "v"
    return dn.compile(
        model,
        dt=DT,
        steps=SEGMENT_STEPS,
        chunk_steps=2,
        backend="eager",
        inference=False,
        trainable="*g_param",
        callbacks={
            "trace": dn.Recorder((voltage, "t"), dt=2 * DT),
            "currents": scope(dn.Recorder(("i_membrane",))),
            "lfp": scope(dn.callbacks.LFP([field, field])),
            "summary": scope(VoltageEnergy()),
        },
    )


def run_demo(network):
    model = make_model(network)
    field = torch.tensor([[1.0, 0.5]], dtype=torch.float64, requires_grad=True)
    program = make_program(model, field, network)
    # Model defaults are optimizer-owned leaves; the LFP configuration retains
    # the separate trainable field source, including its two contact aliases.
    model_parameters = dn.func.inspect_sources(program).trainable_leaves
    assert len(model_parameters) == 1
    optimizer = torch.optim.SGD(
        [
            {"params": model_parameters, "lr": 1e-8},
            {"params": [field], "lr": 1e-4},
        ]
    )
    continuation = None
    previous_drive = None
    times = []
    source_identities = {
        entry.path: entry.tensor for entry in dn.func.inspect_sources(program).entries
    }
    # region training
    for segment in range(SEGMENTS):
        drive = (
            torch.linspace(
                (segment + 1) * 1e-6,
                (segment + 2) * 1e-6,
                SEGMENT_STEPS * field.numel(),
                dtype=torch.float64,
            )
            .reshape(SEGMENT_STEPS, *field.shape)
            .requires_grad_()
        )
        inputs = dn.func.Inputs(intra=drive)
        if network:
            inputs = {"observed": inputs}
        optimizer.zero_grad()
        _, auxiliary = program(continuation, inputs=inputs)
        results = auxiliary["callbacks"]
        voltage = "observed.v" if network else "v"
        loss = (
            (results["trace"][voltage] / 65).square().mean()
            + results["summary"]["energy"]
            + (results["lfp"] * 1e6).square().mean()
        )
        # endregion training
        current = results["currents"]["i_membrane"]
        reference = (current * field).sum((-1, -2)).unsqueeze(-1).expand(-1, 2)
        torch.testing.assert_close(results["lfp"], reference)
        observed_field_gradient = torch.autograd.grad(loss, field, retain_graph=True)[0]
        expected_field_gradient = torch.autograd.grad(
            (reference * 1e6).square().mean(), field, retain_graph=True
        )[0]
        torch.testing.assert_close(observed_field_gradient, expected_field_gradient)
        if previous_drive is not None:
            assert (
                torch.autograd.grad(
                    loss, previous_drive, allow_unused=True, retain_graph=True
                )[0]
                is None
            )
        # region boundary
        loss.backward()
        optimizer.step()
        continuation = dn.func.detach_continuation(program, auxiliary["continuation"])
        # endregion boundary
        for parameter in (*model_parameters, field, drive):
            assert parameter.grad is not None
            assert (
                torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            )
        for entry in dn.func.inspect_sources(continuation).entries:
            assert entry.tensor is source_identities[entry.path]
        assert continuation.callback_state.plan_token is results.state.plan_token
        assert continuation.callback_state["summary"]["energy"].grad_fn is None
        assert (
            continuation.callback_state["summary"]["steps"].item()
            == (segment + 1) * SEGMENT_STEPS
        )
        assert (
            continuation.callback_state["trace"].item() == (segment + 1) * SEGMENT_STEPS
        )
        times.append(results["trace"]["t"].detach())
        previous_drive = drive
    torch.testing.assert_close(
        torch.cat(times),
        torch.arange(0, SEGMENT_STEPS * SEGMENTS + 1, 2, dtype=torch.float64) * DT,
    )
    assert model.t.item() == 0.0
    print(
        f"{'Network' if network else 'Population'}: optimizer identity, local gradients, "
        "LFP field refresh and continued callback phase passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=True):
        for network in (False, True):
            run_demo(network)


if __name__ == "__main__":
    main()
