#!/usr/bin/env python3
"""Use native activity configurations in functional simulations.

Run from the repository root::

    python examples/functional_activity_configuration.py
    python examples/functional_activity_configuration.py --backend aot_eager
    python examples/functional_activity_configuration.py --execution while
    python examples/functional_activity_configuration.py --section network

The CPU example covers a Population and a Network, activity by distinct sites
or total events, inversion, static groups, recording gradients and portable
continuation. Scan is differentiable; while is inference-only.
"""

from __future__ import annotations

import argparse
import io

import torch

import dendra as dn
from dendra.models.mod import hh

DT = 0.025
STEPS = 8
PREFIX = 4


def make_model(network):
    population = dn.SingleCompartment(
        N=1,
        C=3,
        celsius=6.3,
        v_init=-63.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    population.insert(hh)
    model = dn.Network({"cell": population}) if network else population
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def make_callbacks(network):
    # region configuration
    selection = {
        "threshold": -64.0,
        "node_check": [0, 1, -1],
        "t_start_check": DT,
        "t_end_check": 7 * DT,
    }
    configured = {
        "counts": dn.callbacks.APCount(**selection),
        "active": dn.callbacks.Active(**selection),
        "inactive": dn.callbacks.Active(**selection, inv=True),
        "sites": dn.callbacks.ActiveAL(**selection, at_least=2),
        "events": dn.callbacks.ActiveALCount(**selection, at_least=2),
    }
    callbacks = dict(configured)
    # Static functional groups cover the selected-slot axis: [0] and [1, -1].
    callbacks["groups"] = dn.func.ActiveAL(**selection, at_least=1, partition=(1, 2))
    if network:
        callbacks = {
            name: dn.func.PopulationCallback("cell", callback)
            for name, callback in callbacks.items()
        }
    field = "cell.v" if network else "v"
    callbacks["trace"] = dn.Recorder((field, "t"))
    # endregion configuration
    return configured, callbacks, field


def compile_program(model, callbacks, *, backend, execution):
    options = {"checkpoint_every": 3} if execution == "scan" else {}
    return dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        chunk_steps=PREFIX,
        callbacks=callbacks,
        backend=backend,
        execution=execution,
        inference=execution == "while",
        **options,
    )


def inputs(network, drive):
    tape = dn.func.Inputs(intra=drive)
    return {"cell": tape} if network else tape


def run_demo(*, network, backend, execution):
    model = make_model(network)
    configured, callbacks, field = make_callbacks(network)
    program = compile_program(model, callbacks, backend=backend, execution=execution)
    population = model.populations["cell"] if network else model
    training = execution == "scan"
    drive = (
        torch.linspace(1e-6, 3e-6, STEPS * population.v.numel(), dtype=torch.float64)
        .reshape(STEPS, *population.v.shape)
        .requires_grad_(training)
    )

    # region continuation
    _, first = program(steps=PREFIX, inputs=inputs(network, drive[:PREFIX]))
    actual, last = program(
        first["continuation"],
        steps=STEPS - PREFIX,
        inputs=inputs(network, drive[PREFIX:]),
    )
    expected, whole = program(inputs=inputs(network, drive))
    torch.testing.assert_close(actual, expected)
    for name in ("counts", "active", "inactive", "sites", "events", "groups"):
        torch.testing.assert_close(last["callbacks"][name], whole["callbacks"][name])
    joined_voltage = torch.cat(
        (first["callbacks"]["trace"][field], last["callbacks"]["trace"][field])
    )
    torch.testing.assert_close(joined_voltage, whole["callbacks"]["trace"][field])
    # endregion continuation

    # region reductions
    results = whole["callbacks"]
    counts = results["counts"]
    fired_sites = counts > 0
    assert counts.dtype == torch.int32 and counts.shape == (1, 3)
    torch.testing.assert_close(results["active"], fired_sites.any(dim=-1))
    torch.testing.assert_close(results["inactive"], ~results["active"])
    # Repeated events at one site contribute once to sites, every time to events.
    torch.testing.assert_close(results["sites"], fired_sites.sum(dim=-1) >= 2)
    torch.testing.assert_close(results["events"], counts.sum(dim=-1) >= 2)
    expected_groups = torch.stack(
        (fired_sites[..., :1].any(dim=-1), fired_sites[..., 1:].any(dim=-1)),
        dim=-1,
    )
    torch.testing.assert_close(results["groups"], expected_groups)
    for name in ("active", "inactive", "sites", "events", "groups"):
        assert results[name].dtype == torch.bool and not results[name].requires_grad
    # endregion reductions
    for callback in configured.values():
        assert callback.record is callback.state_cache is None
        assert callback.i == 0
    assert model.t.item() == 0.0

    if training:
        gradient = torch.autograd.grad(joined_voltage.square().mean(), drive)[0]
        reference = torch.autograd.grad(results["trace"][field].square().mean(), drive)[
            0
        ]
        torch.testing.assert_close(gradient, reference)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    # region portable
    buffer = io.BytesIO()
    torch.save(program.export_continuation(first["continuation"]), buffer)
    buffer.seek(0)
    _, receiving_callbacks, _ = make_callbacks(network)
    receiver = compile_program(
        make_model(network), receiving_callbacks, backend=backend, execution=execution
    )
    restored = receiver.restore_continuation(torch.load(buffer, weights_only=True))
    resumed, suffix = receiver(
        restored,
        steps=STEPS - PREFIX,
        inputs=inputs(network, drive[PREFIX:].detach()),
    )
    torch.testing.assert_close(resumed, actual)
    for name in ("counts", "active", "inactive", "sites", "events", "groups"):
        torch.testing.assert_close(suffix["callbacks"][name], last["callbacks"][name])
    # endregion portable

    print(
        f"{'Network' if network else 'Population'} {execution}: shared activity "
        "configuration, reductions and portable continuation passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    parser.add_argument("--execution", choices=("all", "scan", "while"), default="all")
    parser.add_argument(
        "--section", choices=("all", "population", "network"), default="all"
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            section = "network" if network else "population"
            for execution in ("scan", "while"):
                if arguments.section in ("all", section) and arguments.execution in (
                    "all",
                    execution,
                ):
                    run_demo(
                        network=network, backend=arguments.backend, execution=execution
                    )


if __name__ == "__main__":
    main()
