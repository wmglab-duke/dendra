#!/usr/bin/env python3
"""Lower RK2/RK4 cable Populations directly and inside a Network."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import graded_syn, pas
from dendra.units import nA

DT = 0.0001
STEPS = 4
DTYPE = torch.float64


def release(voltage):
    return torch.sigmoid(0.03 * (voltage + 65.0))


def run_population_demo():
    # region standalone
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        population = dn.Unmyelinated(
            [2.0],
            L=30.0,
            dx=10.0,
            v_init=torch.tensor([-66.0, -65.0, -64.0], dtype=DTYPE),
            integrator=dn.rk2(),
            dtype=DTYPE,
        )
        population.insert(pas, g=0.001, e=-70.0)
        population.initialize(DT)

    functional, tensors = dn.func.make_functional(population, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    next_state, _ = functional.step(tensors.parameters, prepared, tensors.state)

    runner = dn.compile(
        population,
        dt=DT,
        steps=STEPS,
        inference=False,
        backend="aot_eager",
    )
    final, _ = runner()
    # endregion standalone
    torch.testing.assert_close(population.v, tensors.state["integrator"]["v"])
    population.step(DT)
    torch.testing.assert_close(next_state, functional.extract(population).state)
    for _ in range(STEPS - 1):
        population.step(DT)
    torch.testing.assert_close(final, runner.functional.extract(population).state)

    # region geometry
    geometry = dict(tensors.constants)
    diameter = geometry["diam"].detach().clone().requires_grad_()
    geometry["diam"] = diameter
    prepared_geometry = functional.prepare(tensors.parameters, geometry)
    changed, _ = functional.step(tensors.parameters, prepared_geometry, tensors.state)
    (diameter_gradient,) = torch.autograd.grad(
        changed["integrator"]["v"].square().sum(), diameter
    )
    assert torch.isfinite(diameter_gradient).all()
    assert diameter_gradient.abs().sum() > 0
    # endregion geometry


def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        # region models
        midpoint = dn.Unmyelinated(
            [2.0],
            L=30.0,
            dx=10.0,
            v_init=-65.0,
            integrator=dn.rk2(),
            dtype=DTYPE,
        )
        classical = dn.Unmyelinated(
            [2.0],
            L=30.0,
            dx=10.0,
            v_init=-65.0,
            integrator=dn.rk4(),
            dtype=DTYPE,
        )
        for population in (midpoint, classical):
            population.insert(pas, g=0.001, e=-65.0)
        classical.insert(graded_syn.rename("syn"), e=0.0)
        midpoint[:, 1].inject(dn.mono_rect(amp=0.1 * nA, pw=0.0002))
        network = dn.Network({"midpoint": midpoint, "classical": classical})
        network.connect_continuous_one_to_one(
            midpoint[:],
            classical[:],
            classical.mech.syn,
            pre_var="v",
            input="g_pre",
            weight=0.0002,
            delay=DT,
            transform=release,
        )
        # endregion models
        network.train()
        network.initialize(DT)
        midpoint.integrator.ssd.weight.requires_grad_()
    return network


def run_network_demo():
    network, native = make_network(), make_network()
    source = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), network.state_dict()
    )
    # region compile
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="snapshot",
        inference=False,
        trainable=("*integrator.ssd.weight", "*waveforms.*.amp"),
        callbacks={"trace": dn.func.Recorder(("midpoint.v", "classical.v", "t"))},
        checkpoint_every=2,
        backend="aot_eager",
    )
    final, auxiliary = runner()
    # endregion compile
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(final, runner.functional.extract(native).state)

    # region gradients
    stencil = runner.get_parameter("populations.midpoint.integrator.ssd.weight")
    amplitude = runner.get_parameter("waveforms.0.amp")
    objective = final["populations"]["classical"]["integrator"]["v"].square().mean()
    gradients = torch.autograd.grad(objective, (stencil, amplitude))
    for value in gradients:
        assert torch.isfinite(value).all() and value.abs().sum() > 0
    torch.testing.assert_close(gradients[0], gradients[0].flip(-1))
    # endregion gradients

    # region continuation
    repeated, _ = runner()
    torch.testing.assert_close(repeated, final)
    continued, next_auxiliary = runner(
        final,
        callback_state=auxiliary["callbacks"].state,
    )
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(continued, runner.functional.extract(native).state)
    assert (
        next_auxiliary["callbacks"]["trace"]["t"][-1]
        > auxiliary["callbacks"]["trace"]["t"][-1]
    )
    # endregion continuation
    torch.testing.assert_close(network.state_dict(), source)
    print(
        "RK2/RK4 native parity, geometry/stencil gradients, restart, and continuation passed."
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_population_demo()
    run_network_demo()
