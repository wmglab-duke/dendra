#!/usr/bin/env python3
"""Prepare named pure callbacks directly from a model or an existing plan.

Run from the repository root::

    python examples/shared_callback_preparation.py

The bounded CPU example covers a Population and a scoped Network analysis.
It compares native observations with an independent functional plan, checks
gradients and cadence, then transfers detached callback carry between plans.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025
PREFIX, SUFFIX, FOLLOWUP = 3, 3, 2
STEPS = PREFIX + SUFFIX
G = "integrator.mech.mechanisms.leak.g_param"


class VoltageEnergy(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {"energy": voltage.new_zeros(())}, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return {
            "energy": carry["energy"] + state["integrator"]["v"].square().mean()
        }, None

    def finalize(self, carry, emissions):
        del emissions
        return carry

    def checkpoint_configuration(self):
        return {
            "kind": "example.preparation_voltage_energy",
            "version": 1,
            "fields": {},
        }


def make_model(network):
    specifications = (
        (("left", 2, -55.0), ("right", 3, -75.0)) if network else (("cell", 2, -55.0),)
    )
    populations = {}
    for name, compartments, reversal in specifications:
        population = dn.SingleCompartment(
            N=1,
            C=compartments,
            v_init=-65.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        population.insert(pas.rename("leak"), g=0.001, e=reversal)
        populations[name] = population
    model = dn.Network(populations) if network else populations["cell"]
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def specifications(network):
    scope = dn.callbacks.PopulationCallback
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    return {
        "trace": dn.Recorder(fields, dt=2 * DT),
        "summary": scope("right", VoltageEnergy()) if network else VoltageEnergy(),
    }


def run_demo(network):
    model = make_model(network)
    # region native
    # Population preparation needs dt; a Network supplies its initialized dt.
    clock = {} if network else {"dt": DT}
    observations = dn.func.make_callbacks(model, specifications(network), **clock)
    first = model.run(tstop=PREFIX * DT, **clock, functional_callbacks=observations)
    last = model.longrun(
        tstop=SUFFIX * DT,
        chunklength=2,
        **clock,
        functional_callbacks=observations,
        functional_callback_state=first.state,
    )
    # endregion native
    samples = {
        field: torch.cat((first["trace"][field], value))
        for field, value in last["trace"].items()
    }
    torch.testing.assert_close(
        samples["t"], torch.arange(0, STEPS + 1, 2, dtype=torch.float64) * DT
    )

    reference_model = make_model(network)
    # region functional
    functional, tensors = dn.func.make_functional(reference_model, dt=DT)
    # An existing plan is reused rather than lowered again.
    reference_callbacks = dn.func.make_callbacks(functional, specifications(network))
    final, auxiliary = dn.func.run(
        functional.bind(tensors),
        tensors.state,
        steps=STEPS,
        callbacks=reference_callbacks,
    )
    # endregion functional
    expected = auxiliary["callbacks"]
    torch.testing.assert_close(samples, expected["trace"])
    torch.testing.assert_close(last["summary"], expected["summary"])
    targets = (
        tuple(model.populations[name].mech.leak.g_param for name in ("left", "right"))
        if network
        else (model.mech.leak.g_param,)
    )
    reference_targets = (
        tuple(tensors.parameters["populations"][name][G] for name in ("left", "right"))
        if network
        else (tensors.parameters[G],)
    )
    voltage_fields = tuple(field for field in samples if field != "t")
    objective = last["summary"]["energy"] + sum(
        samples[field].square().mean() for field in voltage_fields
    )
    reference_objective = expected["summary"]["energy"] + sum(
        expected["trace"][field].square().mean() for field in voltage_fields
    )
    observed = torch.autograd.grad(objective, targets)
    wanted = torch.autograd.grad(reference_objective, reference_targets)
    for gradient, reference_gradient in zip(observed, wanted, strict=True):
        torch.testing.assert_close(gradient, reference_gradient)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    # region portable
    # Live carry belongs to one collection. Transfer detached carry explicitly
    # when using an equivalent collection bound to another execution plan.
    packet = observations.export_state(last.state)
    restored = reference_callbacks.restore_state(packet)
    with torch.no_grad():
        continued = model.run(
            tstop=FOLLOWUP * DT,
            **clock,
            functional_callbacks=observations,
            functional_callback_state=last.state,
        )
        _, resumed = dn.func.run(
            functional.bind(tensors),
            final,
            steps=FOLLOWUP,
            callbacks=reference_callbacks,
            callback_state=restored,
        )
    # endregion portable
    torch.testing.assert_close(continued["trace"], resumed["callbacks"]["trace"])
    torch.testing.assert_close(continued["summary"], resumed["callbacks"]["summary"])
    assert continued["trace"]["t"].numel() == 1
    print(
        f"{'Network' if network else 'Population'}: shared callback preparation, "
        "cadence, gradients and portable carry passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            run_demo(network)


if __name__ == "__main__":
    main()
