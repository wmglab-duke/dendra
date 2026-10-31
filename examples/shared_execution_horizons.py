#!/usr/bin/env python3
"""Choose exact steps or elapsed duration for native and functional execution.

Run from the repository root::

    python examples/shared_execution_horizons.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas
from dendra.units import nA

DT = 0.025


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


def population_example():
    # region population
    model = population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    expected, _ = dn.func.run(functional, initial, steps=4)
    parameter = model.mech.leak.g_param
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), parameter
    )[0]

    model.run(steps=3, dt=DT)
    # This duration is smaller than a timestep, so it remains pending.
    model.longrun(duration=0.6 * DT, chunklength=2, dt=DT)
    torch.testing.assert_close(model.t, model.t.new_tensor(3 * DT))
    # Together the two durations accept one additional complete step.
    model.longrun_checkpointed(
        duration=0.4 * DT, chunklength=2, dt=DT, safe_checkpoint=True
    )
    torch.testing.assert_close(model.v, expected["integrator"]["v"])
    actual = torch.autograd.grad(model.v.square().mean(), parameter)[0]
    torch.testing.assert_close(actual, wanted)
    assert actual.isfinite().all() and actual.abs().sum() > 0
    # endregion population
    print("Population: mixed horizons match four functional steps and gradients")


def network_example():
    # region network
    network = dn.Network(
        {"left": population(initialize=False), "right": population(initialize=False)}
    )
    network.train()
    network.initialize(DT)
    network.run(steps=3)
    network.longrun(duration=2.5 * DT, chunklength=2)
    # Exact steps leave the pending half-step available to a duration call.
    network.longrun_checkpointed(steps=2, chunklength=2, safe_checkpoint=True)
    network.run(duration=0.5 * DT)
    torch.testing.assert_close(network.t, network.t.new_tensor(8 * DT))
    # endregion network
    print("Network: exact steps retain pending physical duration")


def injection_example():
    # region injection
    model = population(initialize=False)
    waveform = dn.mono_rect(amp=0.2 * nA, pw=5 * DT)
    model[:, 1].inject(waveform)
    model.initialize()
    voltage = torch.zeros(5, *model.shape, dtype=model.dtype(), device=model.device())
    # The explicit horizon and voltage tape both contain five accepted steps.
    # Omitted intra keeps the waveform registered through Slice.inject active.
    model.run(steps=5, dt=DT, inputs=dn.func.Inputs(ve=voltage))
    model.v.square().mean().backward()
    assert waveform.amp.grad is not None and waveform.amp.grad.abs().sum() > 0
    # endregion injection
    print("Slice.inject: explicit horizons preserve waveform gradients")


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        population_example()
        network_example()
        injection_example()


if __name__ == "__main__":
    main()
