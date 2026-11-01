#!/usr/bin/env python3
"""Inspect native raw sources and form optimizer groups without lowering.

Run from the repository root::

    python examples/native_source_inspection.py
"""

from __future__ import annotations

import re

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025


def make_cell(initial, reversal):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=initial, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=reversal)
    return model


def make_network():
    network = dn.Network(
        {"left": make_cell(-60.0, -45.0), "right": make_cell(-65.0, -70.0)}
    )
    network.right.mech.leak.g_param = network.left.mech.leak.g_param
    network.train()
    network.initialize(DT)
    return network


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        cell = make_cell(-60.0, -45.0)
        # Before building, only the Tensor sources already registered on the
        # model are visible. Inspection itself does not build the model.
        before_build = dn.func.inspect_sources(cell)
        print("Registered Population sources before initialization:")
        print(before_build)
        cell.initialize()
        after_build = dn.func.inspect_sources(cell)
        assert any(
            entry.tensor is cell.mech.leak.g_param for entry in after_build.entries
        )

        # region inspect
        network = make_network()
        report = dn.func.inspect_sources(network)
        print("\nNative Network sources:")
        print(report)
        conductance = report.select("glob:parameters.*.g_param")
        reversal = report.select(re.compile(r"\.e_param$"))
        print("\nSelected conductance sources and their aliases:")
        print(conductance)
        assert len(conductance.trainable_leaves) == 1
        assert conductance.trainable_leaves[0] is network.left.mech.leak.g_param
        assert network.left.mech.leak.g_param is network.right.mech.leak.g_param
        # endregion inspect

        # region optimizers
        conductance_ids = {id(value) for value in conductance.trainable_leaves}
        reversal_ids = {id(value) for value in reversal.trainable_leaves}
        assert conductance_ids.isdisjoint(reversal_ids)
        optimizer = torch.optim.Adam(
            [
                {"params": conductance.trainable_leaves, "lr": 1e-4},
                {"params": reversal.trainable_leaves, "lr": 1e-3},
            ]
        )
        optimizer.zero_grad(set_to_none=True)
        network.run(2 * DT)
        loss = network.left.v.square().mean() + network.right.v.square().mean()
        loss.backward()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                assert parameter.grad is not None and parameter.grad.isfinite().all()
                assert parameter.grad.abs().sum() > 0
        optimizer.step()
        network.detach()
        # endregion optimizers
        print("\nNative source identity, shared aliases and optimizer groups passed")


if __name__ == "__main__":
    main()
