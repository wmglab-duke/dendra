#!/usr/bin/env python3
"""Use native LFP configurations in functional simulations.

Run from the repository root::

    python examples/functional_lfp_configuration.py
    python examples/functional_lfp_configuration.py --backend aot_eager
    python examples/functional_lfp_configuration.py --execution while
    python examples/functional_lfp_configuration.py --section network

The CPU example covers a batched Population and a Network. It checks lead-field
projections, configuration ownership, independent fresh backward passes, and
live and portable continuation with scan and inference-only while execution.
"""

from __future__ import annotations

import argparse
import io

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.1
STEPS = 6
PREFIX = 4


def make_model(network):
    population = dn.SingleCompartment(
        N=2,
        C=3,
        v_init=-65.0,
        integrator=dn.bwd_euler_sc(imem=True),
    )
    population.insert(pas, g=0.001, e=-55.0)
    model = dn.Network({"cell": population}) if network else population
    model.batch(2)
    model.train()
    model.initialize(DT)
    return model


def make_callbacks(network, fields):
    # region configuration
    configured = dn.callbacks.LFP(fields)
    observed = dn.func.PopulationCallback("cell", configured) if network else configured
    current = "cell.i_membrane" if network else "i_membrane"
    callbacks = {
        "lfp": observed,
        "trace": dn.Recorder((current, "t")),
    }
    # endregion configuration
    return configured, callbacks, current


def compile_program(model, callbacks, *, backend, execution):
    options = {"checkpoint_every": 2} if execution == "scan" else {}
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
    training = execution == "scan"
    # One independent, trainable field per contact, shared across batch lanes.
    fields = [
        torch.tensor(values, dtype=torch.float64, requires_grad=training)
        for values in (
            [[1.0, 0.5, 0.25], [0.75, 0.5, 0.125]],
            [[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]],
        )
    ]
    configured, callbacks, current_name = make_callbacks(network, fields)
    source_buffer = configured.v_unit
    source_rule = configured.rule
    program = compile_program(model, callbacks, backend=backend, execution=execution)
    population = model.populations["cell"] if network else model
    drive = torch.linspace(
        1e-6, 3e-6, STEPS * population.v.numel(), dtype=torch.float64
    ).reshape(STEPS, *population.v.shape)

    # region continuation
    _, first = program(steps=PREFIX, inputs=inputs(network, drive[:PREFIX]))
    actual, last = program(
        first["continuation"],
        steps=STEPS - PREFIX,
        inputs=inputs(network, drive[PREFIX:]),
    )
    expected, whole = program(inputs=inputs(network, drive))
    torch.testing.assert_close(actual, expected)
    joined = torch.cat((first["callbacks"]["lfp"], last["callbacks"]["lfp"]))
    torch.testing.assert_close(joined, whole["callbacks"]["lfp"])
    assert first["callbacks"]["lfp"].shape == (PREFIX + 1, 2, 2)
    assert last["callbacks"]["lfp"].shape == (STEPS - PREFIX, 2, 2)
    # endregion continuation

    # region projection
    current_mA = whole["callbacks"]["trace"][current_name]
    potential_mV = whole["callbacks"]["lfp"]
    projected = torch.einsum("tbnc,knc->tbk", current_mA, torch.stack(fields))
    torch.testing.assert_close(potential_mV, projected)
    assert potential_mV.shape == (STEPS + 1, 2, 2)  # Frames, batch, contacts.
    # endregion projection
    assert configured.v_unit is source_buffer
    assert configured.rule == source_rule
    assert model.t.item() == 0.0

    if training:
        # region gradients
        gradient = torch.autograd.grad(joined.square().sum(), fields)
        reference = torch.autograd.grad(potential_mV.square().sum(), fields)
        for actual_gradient, expected_gradient in zip(gradient, reference):
            torch.testing.assert_close(actual_gradient, expected_gradient)
            assert torch.isfinite(actual_gradient).all()
            assert actual_gradient.abs().sum() > 0

        # A fresh call builds a new field stack after the previous backward.
        _, fresh = program(inputs=inputs(network, drive))
        fresh_gradient = torch.autograd.grad(
            fresh["callbacks"]["lfp"].square().sum(), fields
        )
        for actual_gradient, expected_gradient in zip(fresh_gradient, reference):
            torch.testing.assert_close(actual_gradient, expected_gradient)
        # endregion gradients
    else:
        assert not potential_mV.requires_grad

    # region portable
    buffer = io.BytesIO()
    torch.save(program.export_continuation(first["continuation"]), buffer)
    buffer.seek(0)
    receiving_fields = [
        field.detach().clone().requires_grad_(training) for field in fields
    ]
    _, receiving_callbacks, _ = make_callbacks(network, receiving_fields)
    receiver = compile_program(
        make_model(network), receiving_callbacks, backend=backend, execution=execution
    )
    restored = receiver.restore_continuation(torch.load(buffer, weights_only=True))
    resumed, suffix = receiver(
        restored,
        steps=STEPS - PREFIX,
        inputs=inputs(network, drive[PREFIX:]),
    )
    torch.testing.assert_close(resumed, actual)
    restored_lfp = suffix["callbacks"]["lfp"]
    torch.testing.assert_close(restored_lfp, last["callbacks"]["lfp"])
    # Portable carry is detached from the receiving collection's field Tensors.
    if training and restored_lfp.requires_grad:
        restored_gradient = torch.autograd.grad(
            restored_lfp.sum(), receiving_fields, allow_unused=True
        )
        assert all(gradient is None for gradient in restored_gradient)
    # endregion portable

    print(
        f"{'Network' if network else 'Population'} {execution}: shared LFP "
        "configuration, batch projection and portable continuation passed"
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
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=True):
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
