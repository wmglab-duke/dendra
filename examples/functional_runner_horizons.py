#!/usr/bin/env python3
"""Choose step or duration horizons with a reusable compiled runner.

Run from the repository root::

    python examples/functional_runner_horizons.py
    python examples/functional_runner_horizons.py --backend aot_eager

The short CPU Population and Network examples check restart, per-call overrides,
fractional duration continuation and Recorder phase. They do not measure
performance or establish numerical convergence.
"""

from __future__ import annotations

import argparse

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025


def make_population(voltage):
    model = dn.SingleCompartment(
        N=1,
        C=1,
        v_init=voltage,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(pas.rename("leak"), g=0.001, e=-70.0)
    return model


def make_model(*, network):
    if network:
        model = dn.Network(
            {"left": make_population(-60.0), "right": make_population(-65.0)}
        )
    else:
        model = make_population(-60.0)
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def run_demo(*, network, backend):
    model = make_model(network=network)
    names = ("left.v", "right.v", "t") if network else ("v", "t")
    options = {
        "dt": DT,
        "chunk_steps": 2,
        "callbacks": {"trace": dn.func.Recorder(names, dt=2 * DT)},
        "backend": backend,
    }

    # region defaults
    by_steps = dn.compile(model, steps=5, **options)
    by_duration = dn.compile(model, duration=2.5 * DT, **options)

    expected, reference = by_steps()  # Five accepted steps from the snapshot.
    first_state, first = by_duration()  # Two steps, with half a step retained.
    repeated, _ = by_duration()  # No state means restart, using the same default.
    torch.testing.assert_close(repeated, first_state)

    final, second = by_duration(first["continuation"])  # Three more steps.
    torch.testing.assert_close(final, expected)
    # endregion defaults

    # Recorder phase follows accepted steps across the duration boundary.
    joined = {
        name: torch.cat((first["callbacks"]["trace"][name], values))
        for name, values in second["callbacks"]["trace"].items()
    }
    torch.testing.assert_close(joined, reference["callbacks"]["trace"])
    torch.testing.assert_close(
        joined["t"],
        torch.tensor((0.0, 2 * DT, 4 * DT), dtype=torch.float64),
    )

    # region overrides
    overridden, _ = by_duration(steps=5)  # Replaces the default duration.
    torch.testing.assert_close(overridden, expected)
    shorter, _ = by_steps(duration=2.5 * DT)  # Replaces the default step count.
    torch.testing.assert_close(shorter, first_state)

    unchanged, empty = by_duration(first["continuation"], duration=0.0)
    torch.testing.assert_close(unchanged, first_state)
    assert all(value.shape[0] == 0 for value in empty["callbacks"]["trace"].values())
    # endregion overrides

    # Explicit steps preserve an outstanding fractional duration. Adding the
    # other half step afterwards therefore advances one complete solver step.
    stepped, extra = by_duration(first["continuation"], steps=1)
    completed, _ = by_duration(extra["continuation"], duration=0.5 * DT)
    four_steps, _ = by_steps(steps=4)
    torch.testing.assert_close(completed, four_steps)
    clock = stepped["clock"]["t"]
    torch.testing.assert_close(clock, clock.new_full(clock.shape, 3 * DT))
    assert by_steps.kernel.steps == by_duration.kernel.steps == 2
    print(
        f"{'Network' if network else 'Population'}: default/override horizons, "
        "fractional continuation and Recorder phase passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("eager", "aot_eager", "inductor"), default="eager"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        run_demo(network=False, backend=args.backend)
        run_demo(network=True, backend=args.backend)


if __name__ == "__main__":
    main()
