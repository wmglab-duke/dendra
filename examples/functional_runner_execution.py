#!/usr/bin/env python3
"""Execute compiled Population and Network programs through dn.func.run.

Run from the repository root::

    python examples/functional_runner_execution.py
    python examples/functional_runner_execution.py --backend aot_eager

The short CPU examples check direct/canonical equivalence, explicit initialized
continuation, source gradients and portable restoration. They do not measure
performance or establish numerical convergence.
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
    model.initialize(DT)
    return model


def conductance(parameters, network):
    return parameters["populations"]["left"][G] if network else parameters[G]


def trace(auxiliary):
    return auxiliary["callbacks"]["trace"]


def joined_trace(first, last):
    return {
        name: torch.cat((trace(first)[name], value))
        for name, value in trace(last).items()
    }


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

    # region restart
    expected, reference = dn.func.run(program)
    direct, direct_auxiliary = program()
    torch.testing.assert_close(direct, expected)
    torch.testing.assert_close(trace(direct_auxiliary), trace(reference))
    # endregion restart

    # region continuation
    initial = dn.func.initialize(program)
    raw = conductance(initial.parameters, network)
    if network:
        assert raw is initial.parameters["populations"]["right"][G]
    first_state, first = dn.func.run(program, initial, steps=2)
    final, last = dn.func.run(program, first["continuation"], steps=3)
    torch.testing.assert_close(final, expected)
    recording = joined_trace(first, last)
    torch.testing.assert_close(recording, trace(reference))
    assert trace(first)["t"].shape == (3,)
    assert trace(last)["t"].shape == (3,)

    unchanged, empty = dn.func.run(program, state=first["continuation"], steps=0)
    torch.testing.assert_close(unchanged, first_state)
    assert all(value.shape[0] == 0 for value in trace(empty).values())
    field = "left.v" if network else "v"
    (gradient,) = torch.autograd.grad(recording[field].square().mean(), raw)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # endregion continuation

    # region checkpoint
    with TemporaryDirectory() as directory:
        path = Path(directory) / "initial.pt"
        torch.save(dn.func.export_continuation(program, initial), path)
        restored = dn.func.restore_continuation(
            program, torch.load(path, weights_only=True)
        )
        assert restored.callback_state is None
        _, resumed_first = dn.func.run(program, restored, steps=2)

        path = Path(directory) / "active.pt"
        torch.save(
            dn.func.export_continuation(program, resumed_first["continuation"]), path
        )
        active = dn.func.restore_continuation(
            program, torch.load(path, weights_only=True)
        )
        if network:
            assert (
                conductance(active.parameters, True)
                is active.parameters["populations"]["right"][G]
            )
        resumed_final, resumed_last = dn.func.run(program, active, duration=3 * DT)
    torch.testing.assert_close(resumed_final, expected)
    torch.testing.assert_close(
        joined_trace(resumed_first, resumed_last), trace(reference)
    )
    # endregion checkpoint
    print(
        f"{'Network' if network else 'Population'}: canonical/direct execution, "
        "live gradients and portable continuation passed"
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
