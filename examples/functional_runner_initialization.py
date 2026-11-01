#!/usr/bin/env python3
"""Initialize compiled runners before starting recording or a rollout.

Run from the repository root::

    python examples/functional_runner_initialization.py
    python examples/functional_runner_initialization.py --backend aot_eager

The short CPU Population and Network examples check effective sources, aliases,
initialization gradients, recording continuation and portable initialization
checkpoints. They do not time execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025
G = "integrator.mech.mechanisms.leak.g_param"


class Scale(torch.nn.Module):
    def forward(self, value, factor):
        return (value * factor,)


def make_population(voltage, reversal):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=voltage, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=reversal)
    model.build()
    return model


def make_model(network):
    left = make_population(-60.0, -55.0)
    if network:
        right = make_population(-65.0, -70.0)
        right.mech.leak.g_param = left.mech.leak.g_param
        model = dn.Network({"left": left, "right": right})
    else:
        model = left
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    left.register_pre_initialize_transform(
        "scale",
        Scale(),
        reads=(f"parameters.{G}",),
        writes=(f"parameters.{G}",),
        inputs={"factor": left.v.new_tensor(1.5)},
    )
    return model


def conductance(parameters, network):
    return parameters["populations"]["left"][G] if network else parameters[G]


def initialization_voltage(inputs, network):
    selected = inputs.populations["left"] if network else inputs
    voltage = selected.v_init.detach().clone().requires_grad_()
    selected = selected._replace(v_init=voltage)
    if network:
        inputs = inputs._replace(populations={**inputs.populations, "left": selected})
    else:
        inputs = selected
    return inputs, voltage


def run_demo(network, backend):
    names = ("left.v", "right.v", "t") if network else ("v", "t")
    program = dn.compile(
        make_model(network),
        dt=DT,
        steps=5,
        chunk_steps=2,
        inference=False,
        trainable="*.g_param",
        callbacks={"trace": dn.func.Recorder(names)},
        backend=backend,
    )
    # The automatic no-state policy is the default snapshot policy.
    raw = conductance(program.parameters, network)
    original = raw.detach().clone()
    gain = raw.new_tensor(1.2, requires_grad=True)
    replacement = raw * gain
    if network:
        parameters = {
            **program.parameters,
            "populations": {
                name: {**local, G: replacement}
                for name, local in program.parameters["populations"].items()
            },
        }
    else:
        parameters = {**program.parameters, G: replacement}
    inputs, voltage = initialization_voltage(program.initialization_inputs, network)

    # region initialize_runner
    initial = dn.func.initialize(program, parameters=parameters, initialization=inputs)
    assert initial.callback_state is None
    assert program.execution_report() is None
    print(dn.func.inspect_sources(initial))
    effective = conductance(initial.parameters, network)
    torch.testing.assert_close(effective, raw * gain * 1.5)
    assert effective.grad_fn is not None
    if network:
        assert effective is initial.parameters["populations"]["right"][G]
    torch.testing.assert_close(conductance(program.parameters, network), original)
    # endregion initialize_runner

    # region run
    _, first = program(initial, steps=2)
    final, last = program(first["continuation"], steps=3)
    expected, whole = program(initial, steps=5)
    torch.testing.assert_close(final, expected)
    joined = {
        name: torch.cat((first["callbacks"]["trace"][name], values))
        for name, values in last["callbacks"]["trace"].items()
    }
    torch.testing.assert_close(joined, whole["callbacks"]["trace"])
    assert first["callbacks"]["trace"]["t"].shape == (3,)
    assert last["callbacks"]["trace"]["t"].shape == (3,)
    assert joined["t"].shape == (6,)
    field = "left.v" if network else "v"
    torch.testing.assert_close(joined[field][0], voltage)
    gradients = torch.autograd.grad(joined[field].square().mean(), (raw, gain, voltage))
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    # endregion run

    # region initial_checkpoint
    with TemporaryDirectory() as directory:
        path = Path(directory) / "initial.pt"
        saved = dn.func.export_continuation(program, initial)
        assert saved["callbacks"] is None
        torch.save(saved, path)
        restored = dn.func.restore_continuation(
            program, torch.load(path, weights_only=True)
        )
    assert restored.callback_state is None
    restored_final, recorded = program(restored, steps=5)
    torch.testing.assert_close(restored_final, expected)
    torch.testing.assert_close(
        recorded["callbacks"]["trace"], whole["callbacks"]["trace"]
    )
    assert recorded["callbacks"]["trace"]["t"].shape == (6,)
    assert recorded["continuation"].callback_state is not None
    _, continued = program(recorded["continuation"], steps=0)
    assert continued["callbacks"]["trace"]["t"].numel() == 0
    # endregion initial_checkpoint
    print(
        f"{'Network' if network else 'Population'}: runner initialization, "
        "effective aliases, live gradients, recording and disk continuation passed"
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
