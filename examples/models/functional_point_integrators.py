#!/usr/bin/env python3
"""Compile Population and SingleCompartment with explicit and two-level methods."""

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
        population = dn.Population(
            N=2,
            C=3,
            v_init=torch.tensor([-66.0, -62.0, -65.0], dtype=DTYPE),
            integrator=dn.euler(),
            dtype=DTYPE,
        )
        population.insert(pas, g=0.0004, e=-70.0)
        population.initialize()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    next_state, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    # endregion standalone
    original = population.v.detach().clone()
    torch.testing.assert_close(original, tensors.state["integrator"]["v"])
    population.step(DT)
    torch.testing.assert_close(next_state, functional.extract(population).state)


def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        # region models
        leak = torch.nn.Parameter(torch.tensor(0.0004, dtype=DTYPE))
        source = dn.Population(N=2, C=1, v_init=-64.0, integrator=dn.rk4(), dtype=DTYPE)
        target = dn.SingleCompartment(
            N=2,
            C=1,
            v_init=-66.0,
            integrator=dn.dfh(conv=True, beta=0.9),
            dtype=DTYPE,
        )
        for population in (source, target):
            population.insert(pas, g=leak, e=-70.0)
        target.insert(graded_syn.rename("syn"), e=0.0)
        source[:, 0].inject(dn.mono_rect(amp=0.1 * nA, pw=0.0002))
        network = dn.Network({"source": source, "target": target})
        network.connect_continuous_one_to_one(
            source[:],
            target[:],
            target.mech.syn,
            pre_var="v",
            input="g_pre",
            weight=0.0002,
            delay=DT,
            transform=release,
        )
        network.train()
        network.initialize(DT)
        # endregion models
    return network


def run_network_demo():
    network, native = make_network(), make_network()
    frozen = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), network.state_dict()
    )
    # region compile
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*pas.g_param", "*waveforms.*.amp"),
        checkpoint_every=2,
        callbacks={"trace": dn.func.Recorder(("source.v", "target.v", "t"))},
        backend="aot_eager",
    )
    final, auxiliary = runner()
    # endregion compile
    native.initialize(DT)
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(final, runner.functional.extract(native).state)

    # region gradients
    leak = runner.get_parameter(
        "populations.source.integrator.mech.mechanisms.pas.g_param"
    )
    assert leak is runner.get_parameter(
        "populations.target.integrator.mech.mechanisms.pas.g_param"
    )
    amplitude = runner.get_parameter("waveforms.0.amp")
    objective = final["populations"]["target"]["integrator"]["v"].square().mean()
    gradients = torch.autograd.grad(objective, (leak, amplitude))
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    # endregion gradients

    # region continuation
    repeated, _ = runner()
    torch.testing.assert_close(repeated, final)
    continued, _ = runner(
        final,
        parameters=auxiliary["parameters"],
        constants=auxiliary["constants"],
        callback_state=auxiliary["callbacks"].state,
    )
    # endregion continuation
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(continued, runner.functional.extract(native).state)
    torch.testing.assert_close(network.state_dict(), frozen)
    print(
        "Point integrator native parity, shared gradients, restart, and continuation passed."
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_population_demo()
    run_network_demo()
