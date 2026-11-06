"""Select and inspect independent training parameters in compiled simulations.

Run from the repository root with ``python examples/training/functional_parameter_selection.py``.
"""

from __future__ import annotations

import re

import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.05
DTYPE = torch.float64


def make_cell():
    cell = dn.SingleCompartment(
        N=2,
        C=1,
        v_init=-60.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    cell.insert(pas.rename("leak"), g=0.001, e=-70.0)
    cell.insert(pas.rename("other"), g=0.0005, e=-65.0)
    return cell


def optimize_once(runner, *, trace_field="v"):
    # region inspect
    selected = runner.trainable_parameters
    for name, value in selected.items():
        print(name, tuple(value.shape), value.dtype, value.requires_grad)
    # Every alias is visible; an optimizer receives each Tensor object once.
    unique = {id(value): value for value in selected.values()}
    optimizer = torch.optim.Adam(list(unique.values()), lr=1e-4)
    optimizer.zero_grad()
    _, auxiliary = runner()
    loss = auxiliary["callbacks"]["trace"][trace_field].square().mean()
    loss.backward()
    for value in unique.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    optimizer.step()
    # endregion inspect
    return {
        "names": tuple(selected),
        "optimizer_leaves": len(unique),
        "gradient_norms": {
            name: value.grad.norm().item() for name, value in selected.items()
        },
    }


def population_demo(*, backend="aot_eager"):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu", IMEM=False):
        model = make_cell()
        model.train()
        model.initialize()
    # region population-select
    _, tensors = dn.func.make_functional(model, dt=DT)
    conductances = tensors.find_parameters("*.g_param", within="model")
    assert len(conductances) == 2
    selectors = ("*.leak.g_param", re.compile(r"\.other\.(?:g|e)_param$"))
    independent = tensors.independent_parameters(selectors, within="model")
    assert sum(value.requires_grad for value in independent.values()) == 3
    runner = dn.compile(
        model,
        dt=DT,
        steps=4,
        inference=False,
        trainable=selectors,
        callbacks={"trace": dn.func.Recorder(("v",))},
        backend=backend,
    )
    assert tuple(runner.find_parameters("*.g_param")) == tuple(conductances)
    assert len(runner.trainable_parameters) == 3
    # endregion population-select
    return optimize_once(runner)


def network_demo(*, backend="aot_eager"):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu", IMEM=False):
        cell = make_cell()
        cell.insert(expsyn.rename("first"), tau=1.0, e=0.0)
        cell.insert(expsyn.rename("second"), tau=1.2, e=0.0)
        stimulus = dn.NetStim(N=2, interval=0.15, start=0.0, noise=0.0, seed=19)
        network = dn.Network({"cell": cell}, netstim=stimulus)
        for target, delay in ((cell.mech.first, DT), (cell.mech.second, 2 * DT)):
            network.connect_one_to_one(
                stimulus[:], cell[:], target, weight=0.002, delay=delay
            )
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
        )
        first, second = network.synapses.values()
        second.weight.weights[0].param = first.weight.weights[0].param
        network.initialize(DT)
    # region network-select
    runner = dn.compile(
        network,
        dt=DT,
        steps=6,
        inference=False,
        trainable=(
            "synapses.*.weight.weights.*.param.rho",
            r"re:^populations\.cell\..*\.leak\.g_param$",
        ),
        callbacks={"trace": dn.func.Recorder(("cell.v",))},
        backend=backend,
    )
    weights = runner.find_parameters("synapses.*.weight.weights.*.param.rho")
    assert len(weights) == 2
    assert len({id(value) for value in weights.values()}) == 1
    assert len(runner.trainable_parameters) == 3
    # endregion network-select
    return optimize_once(runner, trace_field="cell.v")


def run_demo(*, backend="aot_eager"):
    results = {
        "Population": population_demo(backend=backend),
        "Network": network_demo(backend=backend),
    }
    for name, result in results.items():
        print(
            f"{name}:",
            len(result["names"]),
            "selected names,",
            result["optimizer_leaves"],
            "optimizer leaves",
        )
    return results


if __name__ == "__main__":
    run_demo()
