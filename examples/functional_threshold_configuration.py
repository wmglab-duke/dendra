#!/usr/bin/env python3
"""Use native spike-analysis configurations in functional simulations.

Run from the repository root::

    python examples/functional_threshold_configuration.py
    python examples/functional_threshold_configuration.py --backend aot_eager
    python examples/functional_threshold_configuration.py --execution while
    python examples/functional_threshold_configuration.py --section network

The CPU example covers a Population and a Network. It checks native callback
configuration ownership, windowed counts and raster events, live gradients,
and portable continuation without resetting the detection history.
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


def make_population():
    model = dn.SingleCompartment(
        N=1,
        C=2,
        celsius=6.3,
        v_init=-63.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(hh)
    model.train()
    return model


def make_model(network):
    population = make_population()
    model = dn.Network({"cell": population}) if network else population
    model.train()
    model.initialize(DT)
    return model


def make_callbacks(network):
    # region configuration
    selection = {
        "threshold": -64.0,
        "node_check": [0, -1],
        "t_start_check": DT,
        "t_end_check": 7 * DT,
    }
    configured = {
        "counts": dn.callbacks.APCount(**selection),
        "spikes": dn.callbacks.Raster(**selection),
        "anomaly": dn.callbacks.AnomalyDetector(),
    }
    callbacks = {
        name: dn.func.PopulationCallback("cell", callback) if network else callback
        for name, callback in configured.items()
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
        chunk_steps=4,
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
    joined_spikes = torch.cat(
        (first["callbacks"]["spikes"], last["callbacks"]["spikes"])
    )
    joined_voltage = torch.cat(
        (
            first["callbacks"]["trace"][field],
            last["callbacks"]["trace"][field],
        )
    )
    torch.testing.assert_close(joined_spikes, whole["callbacks"]["spikes"])
    torch.testing.assert_close(joined_voltage, whole["callbacks"]["trace"][field])
    for name in ("counts", "anomaly"):
        torch.testing.assert_close(last["callbacks"][name], whole["callbacks"][name])
    # endregion continuation

    results = whole["callbacks"]
    assert results["counts"].dtype == torch.int32
    assert results["counts"].element_size() == 4
    assert not results["counts"].requires_grad
    assert results["spikes"].dtype == torch.bool
    assert results["anomaly"].dtype == torch.bool
    assert results["counts"].shape == (1, 2)
    assert results["spikes"].shape == (6, 1, 2)  # Checked callback indices 1 through 6.
    assert results["anomaly"].shape == (1,)
    assert first["callbacks"]["spikes"].shape[0] == 3
    assert last["callbacks"]["spikes"].shape[0] == 3
    torch.testing.assert_close(
        results["counts"], results["spikes"].sum(dim=0).to(results["counts"].dtype)
    )
    assert configured["counts"].record is configured["counts"].state_cache is None
    assert configured["spikes"].record is configured["spikes"].state_cache is None
    assert configured["anomaly"].rec is None
    assert configured["counts"].i == configured["spikes"].i == 0
    assert model.t.item() == 0.0

    if training:
        # region gradients
        gradient = torch.autograd.grad(joined_voltage.square().mean(), drive)[0]
        reference = torch.autograd.grad(results["trace"][field].square().mean(), drive)[
            0
        ]
        torch.testing.assert_close(gradient, reference)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        # endregion gradients

    # region portable
    saved = program.export_continuation(first["continuation"])
    buffer = io.BytesIO()
    torch.save(saved, buffer)
    buffer.seek(0)
    _, receiving_callbacks, _ = make_callbacks(network)
    receiver = compile_program(
        make_model(network),
        receiving_callbacks,
        backend=backend,
        execution=execution,
    )
    restored = receiver.restore_continuation(torch.load(buffer, weights_only=True))
    resumed, suffix = receiver(
        restored,
        steps=STEPS - PREFIX,
        inputs=inputs(network, drive[PREFIX:].detach()),
    )
    torch.testing.assert_close(resumed, actual)
    for name in ("counts", "spikes", "anomaly"):
        torch.testing.assert_close(suffix["callbacks"][name], last["callbacks"][name])
    # endregion portable

    print(
        f"{'Network' if network else 'Population'} {execution}: shared threshold "
        "configuration, callback ownership and portable continuation passed"
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
