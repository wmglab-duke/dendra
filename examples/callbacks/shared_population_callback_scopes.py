#!/usr/bin/env python3
"""Share Population callback scopes between native and functional Networks.

Run from the repository root::

    python examples/callbacks/shared_population_callback_scopes.py
    python examples/callbacks/shared_population_callback_scopes.py --backend aot_eager

The CPU example compares native and functional scoped observations, Recorder
cadence, scan gradients, continuation, and portable restoration into the
compatible dn.func.PopulationCallback entry point.
"""

from __future__ import annotations

import argparse
import io

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.1
STEPS = 6
PREFIX = 3


def make_network():
    populations = {}
    for name, reversal in (("left", -55.0), ("right", -75.0)):
        population = dn.SingleCompartment(
            N=1, C=2, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas, g=0.001, e=reversal)
        populations[name] = population
    network = dn.Network(populations)
    network.train()
    network.initialize(DT)
    return network


def make_callbacks(scope=dn.callbacks.PopulationCallback):
    # region scopes
    observations = {
        "left_trace": dn.Recorder(("v", "t"), dt=2 * DT),
        "right_trace": dn.Recorder(("v", "t"), dt=2 * DT),
        "counts": dn.callbacks.APCount(threshold=-100.0, node_check=[0, -1]),
        "anomaly": dn.callbacks.AnomalyDetector(),
    }
    callbacks = {
        "left_trace": scope("left", observations["left_trace"]),
        "right_trace": scope("right", observations["right_trace"]),
        "counts": scope("left", observations["counts"]),
        "anomaly": scope("right", observations["anomaly"]),
    }
    # endregion scopes
    return observations, callbacks


def compile_program(network, callbacks, backend):
    return dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        chunk_steps=PREFIX,
        checkpoint_every=2,
        inference=False,
        execution="scan",
        trainable="*g_param",
        callbacks=callbacks,
        backend=backend,
    )


def run_demo(backend):
    native = make_network()
    observations, callbacks = make_callbacks()
    # region native
    native.run(STEPS * DT, callbacks=list(callbacks.values()))
    left_voltage = observations["left_trace"].stack("v")
    native_counts = observations["counts"].record
    # Native results belong to the supplied inner callbacks.
    assert callbacks["counts"].callback is observations["counts"]
    # endregion native
    native_traces = {
        name: {field: observations[name].stack(field) for field in ("v", "t")}
        for name in ("left_trace", "right_trace")
    }
    history = {name: observations[name].rec for name in native_traces}
    assert left_voltage.shape == (4, 1, 2)
    torch.testing.assert_close(native_counts, torch.ones((1, 2), dtype=torch.int32))
    assert not observations["anomaly"].rec.any()

    source = make_network()
    # Binding the same configurations imports no native recording history.
    program = compile_program(source, callbacks, backend)
    # region continuation
    _, first = program(steps=PREFIX)
    actual, last = program(first["continuation"], steps=STEPS - PREFIX)
    expected, whole = program()
    for name in ("left", "right"):
        torch.testing.assert_close(
            actual["populations"][name]["integrator"]["v"],
            expected["populations"][name]["integrator"]["v"],
        )
    joined = {
        name: {
            field: torch.cat(
                (first["callbacks"][name][field], last["callbacks"][name][field])
            )
            for field in ("v", "t")
        }
        for name in ("left_trace", "right_trace")
    }
    for name, trace in joined.items():
        for field, value in trace.items():
            torch.testing.assert_close(value, whole["callbacks"][name][field])
            torch.testing.assert_close(value, native_traces[name][field])
        assert trace["v"].shape == (4, 1, 2)
    torch.testing.assert_close(last["callbacks"]["counts"], native_counts)
    assert not last["callbacks"]["anomaly"].any()
    # endregion continuation
    assert not torch.equal(joined["left_trace"]["v"], joined["right_trace"]["v"])
    assert source.t.item() == 0.0
    for name, original in history.items():
        assert observations[name].rec is original
        torch.testing.assert_close(
            observations[name].stack("v"), native_traces[name]["v"]
        )
    assert observations["counts"].record is native_counts

    targets = tuple(program.trainable_parameters.values())
    loss = sum(trace["v"].square().mean() for trace in joined.values())
    reference_loss = sum(
        whole["callbacks"][name]["v"].square().mean() for name in joined
    )
    gradient = torch.autograd.grad(loss, targets)
    reference = torch.autograd.grad(reference_loss, targets)
    for observed, wanted in zip(gradient, reference, strict=True):
        torch.testing.assert_close(observed, wanted)
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0

    # region portable
    buffer = io.BytesIO()
    torch.save(program.export_continuation(first["continuation"]), buffer)
    buffer.seek(0)
    _, legacy_callbacks = make_callbacks(dn.func.PopulationCallback)
    receiver = compile_program(make_network(), legacy_callbacks, backend)
    restored = receiver.restore_continuation(torch.load(buffer, weights_only=True))
    _, suffix = receiver(restored, steps=STEPS - PREFIX)
    for name in ("left_trace", "right_trace"):
        for field in ("v", "t"):
            torch.testing.assert_close(
                suffix["callbacks"][name][field], last["callbacks"][name][field]
            )
    for name in ("counts", "anomaly"):
        torch.testing.assert_close(suffix["callbacks"][name], last["callbacks"][name])
    # endregion portable
    print(
        "Shared Population scopes: native parity, cadence, gradients "
        "and continuation passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        run_demo(arguments.backend)


if __name__ == "__main__":
    main()
