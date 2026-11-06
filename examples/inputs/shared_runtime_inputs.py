#!/usr/bin/env python3
"""Use the same deterministic drive packets in native and functional execution.

Run from the repository root::

    python examples/inputs/shared_runtime_inputs.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas
from dendra.units import nA

DT, STEPS = 0.025, 5


def population(*, initialize=True):
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-55.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas.rename("leak"), g=0.001, e=-70.0)
    model.train()
    if initialize:
        model.initialize()
    return model


def tapes(model):
    count = STEPS * model.v.numel()
    ve = (
        torch.linspace(-0.2, 0.3, count, dtype=model.dtype(), device=model.device())
        .reshape(STEPS, *model.shape)
        .requires_grad_()
    )
    intra = (
        torch.linspace(
            0.8e-10, 1.2e-10, count, dtype=model.dtype(), device=model.device()
        )
        .reshape(STEPS, *model.shape)
        .requires_grad_()
    )
    return dn.func.Inputs(ve=ve, intra=intra)


def population_example():
    # region population
    model = population()
    inputs = tapes(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    expected, _ = dn.func.run(functional, initial, inputs=inputs, steps=STEPS)
    reference = torch.autograd.grad(expected["integrator"]["v"].square().mean(), inputs)
    # A Population run infers its horizon from the tapes when tstop is omitted.
    model.run(dt=DT, inputs=inputs)
    torch.testing.assert_close(model.v, expected["integrator"]["v"])
    actual = torch.autograd.grad(model.v.square().mean(), inputs)
    for gradient, wanted in zip(actual, reference, strict=True):
        torch.testing.assert_close(gradient, wanted)
        assert gradient.isfinite().all() and gradient.abs().sum() > 0
    # endregion population
    # region population_helpers
    stepped = population()
    for index in range(STEPS):
        frame = dn.func.Inputs(ve=inputs.ve[index], intra=inputs.intra[index])
        if index % 2 == 0:
            dn.step(stepped, dt=DT, inputs=frame)
        else:
            dn.step_population(stepped, dt=DT, inputs=frame)
    torch.testing.assert_close(stepped.v, model.v)
    gradients = torch.autograd.grad(stepped.v.square().mean(), inputs)
    for gradient, wanted in zip(gradients, reference, strict=True):
        torch.testing.assert_close(gradient, wanted)
    # endregion population_helpers
    print(
        "Population: native/functional runs and step helpers match input values and gradients"
    )


def network_example():
    # region network
    network = dn.Network(
        {"left": population(initialize=False), "right": population(initialize=False)}
    )
    network.train()
    network.initialize(DT)
    left, right = tapes(network.left), tapes(network.right)
    inputs = {
        "left": dn.func.Inputs(intra=left.intra),
        "right": dn.func.Inputs(ve=right.ve),
    }
    functional, tensors = dn.func.make_functional(network, dt=DT)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    expected, _ = dn.func.run(functional, initial, inputs=inputs, steps=STEPS)
    reference_loss = sum(
        state["integrator"]["v"].square().mean()
        for state in expected["populations"].values()
    )
    reference = torch.autograd.grad(reference_loss, (left.intra, right.ve))
    network.longrun(STEPS * DT, chunklength=3, inputs=inputs)
    for name, model in network.populations.items():
        torch.testing.assert_close(
            model.v, expected["populations"][name]["integrator"]["v"]
        )
    loss = network.left.v.square().mean() + network.right.v.square().mean()
    actual = torch.autograd.grad(loss, (left.intra, right.ve))
    for gradient, wanted in zip(actual, reference, strict=True):
        torch.testing.assert_close(gradient, wanted)
        assert gradient.isfinite().all() and gradient.abs().sum() > 0
    # endregion network
    # region network_helpers
    stepped = dn.Network(
        {"left": population(initialize=False), "right": population(initialize=False)}
    )
    stepped.train()
    stepped.initialize(DT)
    for index in range(STEPS):
        frames = {
            "left": dn.func.Inputs(intra=left.intra[index]),
            "right": dn.func.Inputs(ve=right.ve[index]),
        }
        if index % 2 == 0:
            dn.step(stepped, inputs=frames)
        else:
            dn.step_network(stepped, inputs=frames)
    for name, model in stepped.populations.items():
        torch.testing.assert_close(model.v, network.populations[name].v)
    loss = stepped.left.v.square().mean() + stepped.right.v.square().mean()
    gradients = torch.autograd.grad(loss, (left.intra, right.ve))
    for gradient, wanted in zip(gradients, reference, strict=True):
        torch.testing.assert_close(gradient, wanted)
    # endregion network_helpers
    print(
        "Network: named inputs, chunk tails and step helpers match values and gradients"
    )


def registered_injection_example():
    # region injection
    model = population(initialize=False)
    waveform = dn.mono_rect(amp=0.2 * nA, pw=STEPS * DT)
    model[:, 1].inject(waveform)
    model.initialize()
    # Omitting intra retains the waveform registered through Slice.inject.
    ve = torch.zeros(STEPS, *model.shape, dtype=model.dtype())
    model.run(dt=DT, inputs=dn.func.Inputs(ve=ve))
    model.v.square().mean().backward()
    assert waveform.amp.grad is not None and waveform.amp.grad.abs().sum() > 0
    # endregion injection
    print("Slice.inject: omitted intra preserves the registered waveform")


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        population_example()
        network_example()
        registered_injection_example()


if __name__ == "__main__":
    main()
