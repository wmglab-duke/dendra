#!/usr/bin/env python3
"""Reuse prepared pure callbacks when handing functional state to native models.

Run from the repository root::

    python examples/callbacks/shared_callback_rebinding.py

The bounded CPU example covers a Population and a two-Population Network.
It checks cadenced recordings, a custom reducer without portable checkpoint
hooks, native/functional values and receiving-Parameter gradient ownership.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025
PREFIX, SUFFIX = 3, 4
G = "integrator.mech.mechanisms.leak.g_param"


# region reducer
class VoltageEnergy(dn.func.FunctionalCallback):
    """Accumulate post-step energy without a serialization protocol."""

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "energy": voltage.new_zeros(()),
            "steps": voltage.new_zeros((), dtype=torch.int64),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "energy": carry["energy"] + (voltage / 65).square().mean(),
            "steps": carry["steps"] + 1,
        }, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


# endregion reducer


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


def configurations(network):
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    summary = (
        dn.callbacks.PopulationCallback("right", VoltageEnergy())
        if network
        else VoltageEnergy()
    )
    return {"trace": dn.Recorder(fields, dt=2 * DT), "summary": summary}


def native_parameters(model, network):
    return (
        tuple(population.mech.leak.g_param for population in model.populations.values())
        if network
        else (model.mech.leak.g_param,)
    )


def functional_parameters(program, network):
    return (
        tuple(values[G] for values in program.parameters["populations"].values())
        if network
        else (program.parameters[G],)
    )


def voltage(state, network):
    return (
        {name: value["integrator"]["v"] for name, value in state["populations"].items()}
        if network
        else state["integrator"]["v"]
    )


def native_voltage(model, network):
    return (
        {name: population.v for name, population in model.populations.items()}
        if network
        else model.v
    )


def objective(results):
    return results["summary"]["energy"] + sum(
        (value / 65).square().mean()
        for field, value in results["trace"].items()
        if field != "t"
    )


def demonstrate(network):
    source = make_model(network)
    # region handoff
    program = dn.compile(
        source,
        dt=DT,
        steps=PREFIX,
        callbacks=configurations(network),
        trainable="*g_param",
        inference=False,
        backend="eager",
        disable=True,
    )
    _, prefix = dn.func.run(program)
    packet = prefix["continuation"]

    receiver = make_model(network)
    receiving_parameters = native_parameters(receiver, network)
    dn.func.commit_continuation_(program, receiver, packet)
    carry = program.callbacks.detach_state(packet.callback_state)
    clock = {} if network else {"dt": DT}
    suffix = receiver.run(
        steps=SUFFIX,
        **clock,
        callbacks=program.callbacks,
        callback_state=carry,
    )
    # endregion handoff
    assert all(
        actual is original
        for actual, original in zip(
            native_parameters(receiver, network), receiving_parameters, strict=True
        )
    )
    assert carry.plan_token is packet.callback_state.plan_token
    assert suffix["summary"]["steps"].item() == PREFIX + SUFFIX

    # Detach the functional reference at the same boundary, so its suffix has
    # the same local parameter derivatives as the receiving native model.
    detached = dn.func.detach_continuation(program, packet)
    expected_state, expected = dn.func.run(program, detached, steps=SUFFIX)
    torch.testing.assert_close(suffix["trace"], expected["callbacks"]["trace"])
    torch.testing.assert_close(suffix["summary"], expected["callbacks"]["summary"])
    torch.testing.assert_close(
        native_voltage(receiver, network), voltage(expected_state, network)
    )

    # An uninterrupted native simulation independently checks values and phase.
    reference = make_model(network)
    reference_callbacks = dn.func.make_callbacks(
        reference, configurations(network), dt=DT
    )
    native = reference.run(
        steps=PREFIX + SUFFIX, **clock, callbacks=reference_callbacks
    )
    joined = {
        field: torch.cat((prefix["callbacks"]["trace"][field], value))
        for field, value in suffix["trace"].items()
    }
    torch.testing.assert_close(joined, native["trace"])
    torch.testing.assert_close(suffix["summary"], native["summary"])
    torch.testing.assert_close(
        native_voltage(receiver, network), native_voltage(reference, network)
    )
    torch.testing.assert_close(
        joined["t"], torch.arange(0, PREFIX + SUFFIX + 1, 2, dtype=torch.float64) * DT
    )

    # region gradients
    prefix_voltage = voltage(packet.state, network)
    prefix_tensor = prefix_voltage["left"] if network else prefix_voltage
    foreign_sources = (
        *native_parameters(source, network),
        *functional_parameters(program, network),
        prefix_tensor,
    )
    gradients = torch.autograd.grad(
        objective(suffix), (*receiving_parameters, *foreign_sources), allow_unused=True
    )
    receiving_gradients = gradients[: len(receiving_parameters)]
    assert all(gradient is None for gradient in gradients[len(receiving_parameters) :])
    expected_gradients = torch.autograd.grad(
        objective(expected["callbacks"]), functional_parameters(program, network)
    )
    for actual, wanted in zip(receiving_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, wanted)
        assert actual.isfinite().all() and actual.abs().sum() > 0
    # endregion gradients
    print(
        f"{'Network' if network else 'Population'}: prepared callback reuse, "
        "cadence, reducer totals, native/functional values and receiver gradients passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            demonstrate(network)


if __name__ == "__main__":
    main()
