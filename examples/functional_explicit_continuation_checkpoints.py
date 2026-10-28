#!/usr/bin/env python3
"""Save and resume lowered Populations and Networks without a convenience runner.

Run from the repository root::

    python examples/functional_explicit_continuation_checkpoints.py
    python examples/functional_explicit_continuation_checkpoints.py --backend aot_eager

The short CPU examples check portable continuation, recording and suffix
gradients. They do not measure performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT, STEPS = 0.025, 2
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
    model.initialize(DT)
    return model


def demo(network, backend, directory):
    paths = ("left.v", "right.v", "t") if network else ("v", "t")

    # region lower
    functional, tensors = dn.func.make_functional(make_model(network), dt=DT)
    parameters = tensors.independent_parameters(trainable="*.g_param")
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(paths)})
    kernel = functional.compile_rollout_chunk(STEPS, backend=backend)
    bound = kernel.bind(parameters, constants=tensors.constants)
    _, first = dn.func.run(bound, tensors.state, steps=STEPS, callbacks=callbacks)
    # endregion lower

    # region export
    saved = dn.func.export_continuation(
        functional, first["continuation"], callbacks=callbacks
    )
    path = directory / ("network.pt" if network else "population.pt")
    torch.save(saved, path)
    # endregion export

    # region restore
    receiving, _ = dn.func.make_functional(make_model(network), dt=DT)
    receiving_callbacks = receiving.make_callbacks({"trace": dn.func.Recorder(paths)})
    restored = dn.func.restore_continuation(
        receiving,
        torch.load(path, weights_only=True),
        callbacks=receiving_callbacks,
    )
    # Restoration creates owned source leaves. Bind these exact objects.
    receiving_kernel = receiving.compile_rollout_chunk(STEPS, backend=backend)
    resumed = receiving_kernel.bind(restored.parameters, constants=restored.constants)
    final, last = dn.func.run(
        resumed, restored, steps=STEPS, callbacks=receiving_callbacks
    )
    # endregion restore

    expected, whole = dn.func.run(
        bound, tensors.state, steps=2 * STEPS, callbacks=callbacks
    )
    torch.testing.assert_close(final, expected)
    for name, suffix in last["callbacks"]["trace"].items():
        joined = torch.cat((first["callbacks"]["trace"][name], suffix))
        torch.testing.assert_close(joined, whole["callbacks"]["trace"][name])
        assert joined.shape[0] == 2 * STEPS + 1
        assert suffix.shape[0] == STEPS

    # region gradients
    report = dn.func.inspect_sources(restored)
    print(report)
    leaves = report.trainable_leaves
    assert leaves and all(value.is_leaf for value in leaves)
    assert len(leaves) == len({id(value) for value in leaves})
    if network:
        left = restored.parameters["populations"]["left"][G]
        right = restored.parameters["populations"]["right"][G]
        assert left is right  # The tied source remains one trainable leaf.
    trace = last["callbacks"]["trace"]
    loss = sum(value.square().mean() for name, value in trace.items() if name != "t")
    gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
    assert all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    )
    assert any(
        gradient is not None and torch.count_nonzero(gradient) for gradient in gradients
    )
    # endregion gradients
    print(
        f"{'Network' if network else 'Population'}: explicit portable continuation, "
        "joined recording, aliases and suffix gradients passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
        TemporaryDirectory(prefix="dendra-explicit-continuation-") as temporary,
    ):
        for network in (False, True):
            demo(network, args.backend, Path(temporary))


if __name__ == "__main__":
    main()
