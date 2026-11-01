#!/usr/bin/env python3
"""Select effective sources for optimizer groups and separate optimizers.

Run from the repository root::

    python examples/functional_source_selection.py
    python examples/functional_source_selection.py --backend aot_eager

The short CPU examples check captured source metadata, complete alias groups,
identity-disjoint optimizer leaves, gradients and updates. They do not measure
performance or establish numerical convergence.
"""

from __future__ import annotations

import argparse
import re

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025
LEAK = "parameters.*.leak.g_param"
OTHER = re.compile(r"^parameters\..*\.other\.g_param$")


def make_population(voltage):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=voltage, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=-55.0)
    model.insert(pas.rename("other"), g=0.0005, e=-70.0)
    model.build()
    return model


def make_model(network):
    left = make_population(-60.0)
    if network:
        right = make_population(-64.0)
        right.mech.leak.g_param = left.mech.leak.g_param
        model = dn.Network({"left": left, "right": right})
    else:
        model = left
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def identities(values):
    return tuple(id(value) for value in values)


def loss(state, network):
    states = state["populations"].values() if network else (state,)
    return sum(local["integrator"]["v"].square().mean() for local in states)


def optimize_once(program, optimizers, leaves, network):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    before = tuple(value.detach().clone() for value in leaves)
    state, auxiliary = dn.func.run(program)
    effective = dn.func.inspect_sources(auxiliary["continuation"])
    assert identities(effective.select((LEAK, OTHER)).trainable_leaves) == identities(
        leaves
    )
    loss(state, network).backward()
    assert all(
        value.grad is not None
        and torch.isfinite(value.grad).all()
        and value.grad.abs().sum() > 0
        for value in leaves
    )
    for optimizer in optimizers:
        optimizer.step()
    assert all(
        torch.isfinite(value).all() and not torch.equal(value, previous)
        for value, previous in zip(leaves, before, strict=True)
    )


def run_demo(network, backend):
    program = dn.compile(
        make_model(network),
        dt=DT,
        steps=5,
        chunk_steps=2,
        inference=False,
        trainable="*.g_param",
        backend=backend,
    )

    # region selection
    report = dn.func.inspect_sources(program)
    leak = report.select(LEAK)
    other = report.select(OTHER)
    selected = report.select((OTHER, LEAK))
    assert identities(selected.trainable_leaves) == identities(
        leak.trainable_leaves + other.trainable_leaves
    )
    assert {id(p) for p in leak.trainable_leaves}.isdisjoint(
        id(p) for p in other.trainable_leaves
    )
    assert len(leak.trainable_leaves) == 1
    assert len(other.trainable_leaves) == (2 if network else 1)
    shared = leak.trainable_leaves[0]
    assert identities(leak.entries) == identities(
        entry for entry in report.entries if entry.tensor is shared
    )
    if network:
        paths = tuple(entry.path for entry in leak.entries)
        assert any(path.startswith("parameters.populations.left.") for path in paths)
        assert any(path.startswith("parameters.populations.right.") for path in paths)
        assert leak.entries[0].aliases == paths[1:]
    print(leak)
    # endregion selection

    # region optimizer_groups
    grouped = torch.optim.Adam(
        [
            {"params": leak.trainable_leaves, "lr": 1e-6},
            {"params": other.trainable_leaves, "lr": 2e-6},
        ]
    )
    optimize_once(program, (grouped,), selected.trainable_leaves, network)
    # endregion optimizer_groups

    # region separate_optimizers
    leak_optimizer = torch.optim.Adam(leak.trainable_leaves, lr=1e-6)
    other_optimizer = torch.optim.SGD(other.trainable_leaves, lr=1e-8)
    optimize_once(
        program,
        (leak_optimizer, other_optimizer),
        selected.trainable_leaves,
        network,
    )
    # endregion separate_optimizers
    print(
        f"{'Network' if network else 'Population'}: complete aliases, "
        "optimizer groups and separate optimizers passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            run_demo(network, args.backend)


if __name__ == "__main__":
    main()
