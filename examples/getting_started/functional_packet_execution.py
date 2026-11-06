#!/usr/bin/env python3
"""Run explicit Population/Network continuations with eager plans or chunks.

Run from the repository root::

    python examples/getting_started/functional_packet_execution.py
    python examples/getting_started/functional_packet_execution.py --backend aot_eager

The short CPU examples check shared sources, recording, live gradients and
portable checkpoint continuation. They do not measure execution speed.
"""

from __future__ import annotations

import argparse

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT, PREFIX, SUFFIX = 0.025, 2, 5
G = "integrator.mech.mechanisms.leak.g_param"


def cell(initial, reversal):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=initial, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=reversal)
    return model


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


def make_model(network):
    left = cell(-60.0, -55.0)
    if network:
        right = cell(-65.0, -70.0)
        right.insert(graded_syn.rename("continuous"), e=0.0)
        model = dn.Network({"left": left, "right": right})
        right.mech.leak.g_param = left.mech.leak.g_param
        model.connect_continuous_one_to_one(
            left[:],
            right[:],
            right.mech.continuous,
            pre_var="v",
            input="g_pre",
            weight=0.0005,
            delay=0.0,
            transform=release,
        )
    else:
        model = left
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def conductance(packet, network):
    local = packet.parameters["populations"]["left"] if network else packet.parameters
    return local[G]


def check_sources(packet, expected, network):
    assert conductance(packet, network) is expected
    if network:
        assert packet.parameters["populations"]["right"][G] is expected


def trace_loss(auxiliary):
    return sum(
        values.square().mean()
        for name, values in auxiliary["callbacks"]["trace"].items()
        if name != "t"
    )


def demo(network, backend):
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    functional, tensors = dn.func.make_functional(make_model(network), dt=DT)
    parameters = tensors.independent_parameters(trainable="*.g_param")
    initial = dn.func.initialize(functional, tensors._replace(parameters=parameters))
    source = conductance(initial, network)
    assert source.requires_grad and source.is_leaf
    check_sources(initial, source, network)

    # region packet_run
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(paths)})
    _, first = dn.func.run(functional, initial, steps=PREFIX, callbacks=callbacks)
    kernel = functional.compile_rollout_chunk(3, backend=backend)
    final, last = dn.func.run(
        kernel,
        state=first["continuation"],
        steps=SUFFIX,
        checkpoint_every=4,
        callbacks=callbacks,
    )
    # endregion packet_run
    check_sources(first["continuation"], source, network)
    check_sources(last["continuation"], source, network)

    # A single run and the resumed run have the same final state and samples.
    expected, whole = dn.func.run(
        kernel, initial, steps=PREFIX + SUFFIX, callbacks=callbacks
    )
    torch.testing.assert_close(final, expected)
    for name, suffix in last["callbacks"]["trace"].items():
        joined = torch.cat((first["callbacks"]["trace"][name], suffix))
        torch.testing.assert_close(joined, whole["callbacks"]["trace"][name])
        assert joined.shape[0] == PREFIX + SUFFIX + 1
        assert suffix.shape[0] == SUFFIX  # Continuation does not repeat the boundary.

    # The suffix's loss remains connected to the original optimization leaf.
    gradient = torch.autograd.grad(trace_loss(last), source)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    # region packet_restore
    saved = dn.func.export_continuation(
        functional, first["continuation"], callbacks=callbacks
    )
    receiving, _ = dn.func.make_functional(make_model(network), dt=DT)
    receiving_callbacks = receiving.make_callbacks({"trace": dn.func.Recorder(paths)})
    restored = dn.func.restore_continuation(
        receiving, saved, callbacks=receiving_callbacks
    )
    receiving_kernel = receiving.compile_rollout_chunk(3, backend=backend)
    restored_final, restored_aux = dn.func.run(
        receiving_kernel, restored, steps=SUFFIX, callbacks=receiving_callbacks
    )
    # endregion packet_restore
    torch.testing.assert_close(restored_final, final)
    torch.testing.assert_close(
        dict(restored_aux["callbacks"]["trace"]), dict(last["callbacks"]["trace"])
    )
    restored_source = conductance(restored, network)
    assert restored_source is not source and restored_source.is_leaf
    check_sources(restored, restored_source, network)
    check_sources(restored_aux["continuation"], restored_source, network)
    # Portable restoration creates independent raw and retained source leaves.
    leaves = dn.func.inspect_sources(restored).trainable_leaves
    gradients = torch.autograd.grad(
        trace_loss(restored_aux), (*leaves, source), allow_unused=True
    )
    assert all(value is None or torch.isfinite(value).all() for value in gradients[:-1])
    assert any(value is not None and value.abs().sum() > 0 for value in gradients[:-1])
    assert gradients[-1] is None  # Portable checkpoints cut the earlier graph.
    print(
        f"{'Network' if network else 'Population'}: direct packet execution, "
        "shared sources, recording and live/restored gradients passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        demo(False, args.backend)
        demo(True, args.backend)


if __name__ == "__main__":
    main()
