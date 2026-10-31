#!/usr/bin/env python3
"""Return functional Population and Network continuations to native execution.

Run from the repository root::

    python examples/functional_native_handoff.py

The bounded CPU example compares suffix values and verifies that native
gradients belong to receiving Parameters after detached value transfer.
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas

DT, STEPS = 0.025, 3
G = "integrator.mech.mechanisms.leak.g_param"


def make_model(network):
    def cell():
        population = dn.SingleCompartment(
            N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas.rename("leak"), g=0.001, e=-45.0)
        return population

    model = cell()
    if network:
        right = cell()
        model = dn.Network({"left": model, "right": right})
        right.mech.leak.g_param = model.left.mech.leak.g_param
    model.train()
    model.initialize(DT)
    return model


def raw_source(model, network):
    return model.left.mech.leak.g_param if network else model.mech.leak.g_param


def voltage(value, network):
    if network:
        return torch.cat(
            [
                part["integrator"]["v"].flatten()
                for part in value["populations"].values()
            ]
        )
    return value["integrator"]["v"]


def demonstrate(network):
    program = dn.compile(
        make_model(network),
        dt=DT,
        steps=STEPS,
        trainable="*g_param",
        backend="eager",
        disable=True,
    )
    parameters = dict(program.parameters)
    if network:
        parameters["populations"] = {
            name: dict(values) for name, values in parameters["populations"].items()
        }
        original = parameters["populations"]["left"][G]
    else:
        original = parameters[G]
    gain = original.new_tensor(1.4, requires_grad=True)
    selected = original * gain
    if network:
        for local in parameters["populations"].values():
            local[G] = selected
    else:
        parameters[G] = selected

    # region handoff
    _, first = dn.func.run(program, parameters=parameters)
    continuation = first["continuation"]
    receiver = make_model(network)
    receiving_parameter = raw_source(receiver, network)
    dn.func.commit_continuation_(program, receiver, continuation)
    assert raw_source(receiver, network) is receiving_parameter
    torch.testing.assert_close(receiving_parameter, selected)
    if network:
        assert receiver.right.mech.leak.g_param is receiving_parameter
    # endregion handoff

    expected, _ = dn.func.run(program, continuation)
    # region resume
    if network:
        receiver.run(STEPS * DT)
    else:
        receiver.run(tstop=STEPS * DT, dt=DT)
    actual = (
        torch.cat([part.v.flatten() for part in receiver.populations.values()])
        if network
        else receiver.v
    )
    torch.testing.assert_close(actual, voltage(expected, network))
    native_gradient, functional_gradient = torch.autograd.grad(
        actual.square().sum(), (receiving_parameter, gain), allow_unused=True
    )
    assert native_gradient.isfinite().all() and native_gradient.abs().sum() > 0
    assert functional_gradient is None
    # endregion resume
    print(
        ("Network" if network else "Population")
        + ": native values, source identity, aliases and detached gradient ownership passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        demonstrate(False)
        demonstrate(True)


if __name__ == "__main__":
    main()
