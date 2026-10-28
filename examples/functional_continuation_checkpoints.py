#!/usr/bin/env python3
"""Save effective training sources and resume through a reconstructed runner.

Run from the repository root::

    python examples/functional_continuation_checkpoints.py
    python examples/functional_continuation_checkpoints.py --backend aot_eager

The short CPU examples check portable continuation, not performance.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT, STEPS = 0.025, 2
G = "integrator.mech.mechanisms.leak.g_param"


class Scale(torch.nn.Module):
    def forward(self, value, factor):
        return (value * factor,)


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


def cell(initial, reversal):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=initial, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=reversal)
    return model


def make_model(network):
    if network:
        left, right = cell(-60.0, -55.0), cell(-65.0, -70.0)
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
        owner = left
    else:
        model = owner = cell(-60.0, -70.0)
    model.train()
    model.initialize(DT)
    # Fresh initialization replaces this raw source. A checkpoint must retain
    # the effective result rather than the runner's pre-initialization default.
    owner.register_pre_initialize_transform(
        "scale",
        Scale(),
        reads=(f"parameters.{G}",),
        writes=(f"parameters.{G}",),
        inputs={"factor": owner.v.new_tensor(1.5)},
    )
    return model


def program(model, network, backend, *, steps=STEPS):
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    return dn.compile(
        model,
        dt=DT,
        steps=steps,
        initialization="fresh",
        inference=False,
        trainable="*g_param",
        callbacks={"trace": dn.func.Recorder(paths)},
        backend=backend,
    )


def replace(tree, original, replacement):
    """Copy raw mapping containers and retain the other Tensor identities."""
    if isinstance(tree, Mapping):
        return {
            name: replace(value, original, replacement) for name, value in tree.items()
        }
    if isinstance(tree, tuple):
        values = tuple(replace(value, original, replacement) for value in tree)
        return type(tree)(*values) if hasattr(tree, "_fields") else values
    return replacement if tree is original else tree


def conductance(continuation, network):
    values = continuation.parameters
    return values["populations"]["left"][G] if network else values[G]


def demo(network, backend, directory):
    model = make_model(network)
    source = program(model, network, backend)
    raw = next(iter(source.trainable_parameters.values()))
    gain = raw.new_tensor(1.2, requires_grad=True)
    overrides = replace(source.parameters, raw, raw * gain)
    _, first = source(parameters=overrides)
    original = first["continuation"]
    effective = conductance(original, network)
    assert effective.grad_fn is not None
    torch.testing.assert_close(effective, raw * gain * 1.5)

    # region source_inspection
    defaults = dn.func.inspect_sources(source)  # dn.compile runner defaults
    current = dn.func.inspect_sources(original)  # Effective sources for this call
    print(defaults)
    print(current)
    assert any(entry.tensor is raw for entry in defaults.entries)
    assert any(
        entry.tensor is effective and entry.requires_grad and not entry.is_leaf
        for entry in current.entries
    )
    # The original optimizer leaves raw and gain remain caller-owned. Inspection
    # reports the nonleaf result; it does not traverse its upstream graph.
    # endregion source_inspection

    # region save
    packet = source.export_continuation(first["continuation"])
    path = directory / ("network.pt" if network else "population.pt")
    torch.save(packet, path)
    # endregion save

    # region restore
    receiver = program(make_model(network), network, backend)
    restored = receiver.restore_continuation(torch.load(path, weights_only=True))
    final, last = receiver(restored)
    # endregion restore

    # region trainability
    saved_g = conductance(restored, network)
    assert saved_g.requires_grad and saved_g.is_leaf and saved_g.grad_fn is None
    assert saved_g is not effective and saved_g.data_ptr() != effective.data_ptr()
    torch.testing.assert_close(saved_g, effective)
    if network:
        assert saved_g is restored.parameters["populations"]["right"][G]
    report = dn.func.inspect_sources(restored)
    leaves = report.trainable_leaves
    assert len(leaves) == len({id(value) for value in leaves})
    assert any(value is saved_g for value in leaves)
    print(report)
    if network:
        assert any(
            entry.tensor is saved_g and entry.aliases for entry in report.entries
        )
    loss = sum(
        value.square().mean()
        for name, value in last["callbacks"]["trace"].items()
        if name != "t"
    )
    gradients = torch.autograd.grad(
        loss,
        (*leaves, gain),
        allow_unused=True,
        retain_graph=True,  # Reuse this loss for the optimizer demonstration below.
    )
    suffix_gradients, prefix_gradient = gradients[:-1], gradients[-1]
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in suffix_gradients
    )
    nonzero = {
        id(value)
        for value, gradient in zip(leaves, suffix_gradients, strict=True)
        if gradient is not None and torch.count_nonzero(gradient)
    }
    assert nonzero
    assert prefix_gradient is None  # Portable export cuts the prefix graph.
    if network:
        # The suffix uses saved retained constants. Inspection lists eligible
        # leaves; differentiating this loss establishes which receive gradients.
        assert any(
            entry.domain == "constants" and id(entry.tensor) in nonzero
            for entry in report.entries
        )
        assert id(saved_g) not in nonzero
    else:
        assert id(saved_g) in nonzero
    # endregion trainability

    whole = program(model, network, backend, steps=2 * STEPS)
    expected, auxiliary = whole(parameters=overrides)
    torch.testing.assert_close(final, expected)
    for name, suffix in last["callbacks"]["trace"].items():
        joined = torch.cat((first["callbacks"]["trace"][name], suffix))
        torch.testing.assert_close(joined, auxiliary["callbacks"]["trace"][name])
        assert joined.shape[0] == 2 * STEPS + 1
        assert suffix.shape[0] == STEPS
    assert receiver.initial_state["clock"]["t"].item() == 0

    # region optimizer
    optimizer = torch.optim.Adam(report.trainable_leaves, lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # endregion optimizer

    print(
        f"{'Network' if network else 'Population'}: portable effective sources, "
        "source inspection, joined recording and suffix optimization passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
        TemporaryDirectory(prefix="dendra-continuation-checkpoint-") as temporary,
    ):
        for network in (False, True):
            demo(network, args.backend, Path(temporary))


if __name__ == "__main__":
    main()
