#!/usr/bin/env python3
"""Carry native callback reductions and recording phase into functional execution.

Run from the repository root::

    python examples/functional_callback_capture.py

The bounded CPU example covers Populations and Networks, explicit plans and
compiled programs, a cadenced Recorder, cumulative APCount and portable replay.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.models.mod import pas

DT, PREFIX, SUFFIX = 0.025, 4, 4


def make_model(network):
    def cell(reversal):
        population = dn.SingleCompartment(
            N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas.rename("leak"), g=0.001, e=reversal)
        return population

    model = cell(-45.0)
    if network:
        model = dn.Network({"left": model, "right": cell(-55.0)})
        model.right.mech.leak.g_param = model.left.mech.leak.g_param
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def configurations(network):
    counts = dn.func.APCount(threshold=-59.5, node_check=[0], dt=DT)
    if network:
        counts = dn.callbacks.PopulationCallback("left", counts)
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    return {
        "trace": dn.Recorder(paths, dt=3 * DT),
        "counts": counts,
    }


def prepare(native, network, compiled):
    if compiled:
        target = dn.compile(
            native,
            dt=DT,
            steps=SUFFIX,
            inference=False,
            callbacks=configurations(network),
            backend="eager",
            disable=True,
        )
        # Runners infer their configured collection for capture and execution.
        return target, target.callbacks, {}
    target, _ = dn.func.make_functional(native, dt=DT)
    callbacks = target.make_callbacks(configurations(network))
    return target, callbacks, {"callbacks": callbacks}


def native_run(native, network, callbacks, steps):
    clock = {} if network else {"dt": DT}
    return native.run(steps=steps, callbacks=callbacks, **clock)


def voltage(state, network):
    if network:
        return torch.cat(
            [
                part["integrator"]["v"].flatten()
                for part in state["populations"].values()
            ]
        )
    return state["integrator"]["v"]


def demonstrate(network, compiled, directory):
    native = make_model(network)
    target, callbacks, callback_arguments = prepare(native, network, compiled)

    # region capture_callbacks
    # Native and functional execution use this exact prepared collection.
    prefix = native_run(native, network, callbacks, PREFIX)
    continuation = dn.func.capture_continuation(
        target, callback_state=prefix.state, **callback_arguments
    )
    final, suffix = dn.func.run(
        target, continuation, steps=SUFFIX, **callback_arguments
    )
    # Carry retains the reducer total and Recorder's post-step counter.
    assert suffix["callbacks"]["trace"]["t"].numel() == 1
    torch.testing.assert_close(
        suffix["callbacks"]["trace"]["t"], native.t.new_tensor([6 * DT])
    )
    # endregion capture_callbacks

    reference = make_model(network)
    clock = {} if network else {"dt": DT}
    reference_callbacks = dn.func.make_callbacks(
        reference, configurations(network), **clock
    )
    whole = native_run(reference, network, reference_callbacks, PREFIX + SUFFIX)
    expected_voltage = (
        torch.cat([part.v.flatten() for part in reference.populations.values()])
        if network
        else reference.v
    )
    torch.testing.assert_close(voltage(final, network), expected_voltage)
    joined = {
        path: torch.cat((prefix["trace"][path], value))
        for path, value in suffix["callbacks"]["trace"].items()
    }
    torch.testing.assert_close(joined, whole["trace"])
    torch.testing.assert_close(suffix["callbacks"]["counts"], whole["counts"])
    assert prefix["counts"].sum().item() > 0
    assert joined["t"].numel() == 3  # Frames 0, 3 and 6, with one initial sample.

    # region checkpoint_callbacks
    # The captured model and built-in callback carry can be saved immediately.
    saved = dn.func.export_continuation(target, continuation, **callback_arguments)
    path = directory / f"{'network' if network else 'population'}-{compiled}.pt"
    torch.save(saved, path)
    receiver, _, receiving_arguments = prepare(make_model(network), network, compiled)
    restored = dn.func.restore_continuation(
        receiver, torch.load(path, weights_only=True), **receiving_arguments
    )
    replay_final, replay = dn.func.run(
        receiver, restored, steps=SUFFIX, **receiving_arguments
    )
    # Portable restoration attaches the receiving collection's identity.
    torch.testing.assert_close(replay_final, final)
    torch.testing.assert_close(
        replay["callbacks"]["trace"], suffix["callbacks"]["trace"]
    )
    torch.testing.assert_close(
        replay["callbacks"]["counts"], suffix["callbacks"]["counts"]
    )
    # endregion checkpoint_callbacks

    # Omitting carry retains the ordinary fresh-callback boundary behavior.
    fresh = dn.func.capture_continuation(target)
    _, restarted = dn.func.run(target, fresh, steps=0, **callback_arguments)
    torch.testing.assert_close(
        restarted["callbacks"]["trace"]["t"], native.t.reshape(1)
    )
    assert restarted["callbacks"]["counts"].sum().item() == 0
    print(
        f"{'Network' if network else 'Population'} "
        f"{'program' if compiled else 'plan'}: native callback capture, "
        "cumulative counts, recording phase and portable replay passed"
    )


def main():
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
        TemporaryDirectory(prefix="dendra-callback-capture-") as temporary,
    ):
        for network in (False, True):
            for compiled in (False, True):
                demonstrate(network, compiled, Path(temporary))


if __name__ == "__main__":
    main()
