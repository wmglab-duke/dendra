#!/usr/bin/env python3
"""Compile a connected scalar/block SPD Network with shared parameters."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import graded_syn, pas
from dendra.units import nA

DT = 0.005
STEPS = 4
DTYPE = torch.float64


def release(voltage):
    return torch.sigmoid(0.03 * (voltage + 65.0))


def make_network():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        # region models
        leak = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE))
        cable = dn.Unmyelinated(
            [2.0],
            L=30.0,
            dx=10.0,
            v_init=torch.tensor([-66.0, -64.0, -62.0], dtype=DTYPE),
            integrator=dn.bwd_euler_ub(method="spd"),
            dtype=DTYPE,
        )
        shell = dn.ExtCellAxon(
            [5.0],
            n_comp=3,
            v_init=-65.0,
            integrator=dn.bwd_euler_bt(method="spd"),
            dtype=DTYPE,
        )
        shell.xraxial.fill_(2.0)
        shell.xc.fill_(0.12)
        shell.xg.fill_(0.0002)
        for population in (cable, shell):
            population.insert(pas, g=leak, e=-70.0)
        # endregion models
        shell.insert(graded_syn.rename("syn"), e=0.0)
        cable[:, 1].inject(dn.mono_rect(amp=0.1 * nA, pw=0.03))
        network = dn.Network({"cable": cable, "shell": shell})
        network.connect_continuous_one_to_one(
            cable[:],
            shell[:],
            shell.mech.syn,
            pre_var="v",
            input="g_pre",
            weight=0.0001,
            delay=DT,
            transform=release,
        )
        network.train()
        network.initialize(DT)
    return network


def main():
    torch.set_num_threads(1)
    network, native = make_network(), make_network()
    before = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), network.state_dict()
    )
    # region compile
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*pas.g_param", "*cm_param*", "*rhoa_param*"),
        checkpoint_every=2,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("cable.v", "shell.v", "t"))},
    )
    final, auxiliary = runner()
    # endregion compile
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(final, runner.functional.extract(native).state)

    # region gradients
    cable_leak = runner.get_parameter(
        "populations.cable.integrator.mech.mechanisms.pas.g_param"
    )
    shell_leak = runner.get_parameter(
        "populations.shell.integrator.mech.mechanisms.pas.g_param"
    )
    assert cable_leak is shell_leak
    capacitance = runner.get_parameter("populations.cable.cm_param")
    objective = final["populations"]["shell"]["integrator"]["v"].square().mean()
    gradients = torch.autograd.grad(objective, (cable_leak, capacitance))
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    # endregion gradients

    # region continuation
    repeated, _ = runner()
    torch.testing.assert_close(repeated, final)
    continued, _ = runner(
        final,
        parameters=auxiliary.get("parameters", runner.parameters),
        constants=auxiliary.get("constants", runner.constants),
        callback_state=auxiliary["callbacks"].state,
    )
    # endregion continuation
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(continued, runner.functional.extract(native).state)
    torch.testing.assert_close(network.state_dict(), before)
    print(
        "SPD native parity, shared gradients, checkpointed scan, restart, and continuation passed."
    )


if __name__ == "__main__":
    main()
