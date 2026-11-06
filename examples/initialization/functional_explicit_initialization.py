#!/usr/bin/env python3
"""Initialize and resume lowered Populations and Networks with explicit sources.

Run from the repository root::

    python examples/initialization/functional_explicit_initialization.py
    python examples/initialization/functional_explicit_initialization.py --backend aot_eager

The short CPU examples verify initialization ownership, shared parameters,
recording, live gradients and portable continuation. They do not time execution.
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


class Scale(torch.nn.Module):
    def forward(self, value, factor):
        return (value * factor,)


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


def trace_loss(auxiliary):
    return sum(
        value.square().mean()
        for name, value in auxiliary["callbacks"]["trace"].items()
        if name != "t"
    )


def demo(network, backend, directory):
    paths = ("left.v", "right.v", "t") if network else ("v", "t")
    functional, tensors = dn.func.make_functional(make_model(network), dt=DT)

    # region initialize
    parameters = tensors.independent_parameters(trainable="*.g_param")
    raw = conductance(parameters, network)
    gain = raw.new_tensor(1.2, requires_grad=True)
    replacement = raw * gain
    if network:
        for local in parameters["populations"].values():
            local[G] = replacement  # Keep the raw parameter tie explicit.
    else:
        parameters[G] = replacement
    initial = dn.func.initialize(functional, tensors._replace(parameters=parameters))
    assert initial.callback_state is None
    assert initial.state["clock"]["t"].item() == 0
    effective = conductance(initial.parameters, network)
    torch.testing.assert_close(effective, raw * gain * 1.5)
    assert effective.grad_fn is not None
    if network:
        assert effective is initial.parameters["populations"]["right"][G]
    # endregion initialize

    # region run
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(paths)})
    kernel = functional.compile_rollout_chunk(STEPS, backend=backend)
    # Binding takes sources; the run still receives state and callbacks explicitly.
    bound = kernel.bind(initial)
    _, first = dn.func.run(bound, initial, steps=STEPS, callbacks=callbacks)
    live_final, live_last = dn.func.run(
        bound, first["continuation"], steps=STEPS, callbacks=callbacks
    )
    expected, whole = dn.func.run(bound, initial, steps=2 * STEPS, callbacks=callbacks)
    torch.testing.assert_close(live_final, expected)
    gradients = torch.autograd.grad(trace_loss(live_last), (raw, gain))
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    # endregion run

    # region checkpoint
    saved = dn.func.export_continuation(
        functional, first["continuation"], callbacks=callbacks
    )
    path = directory / ("network.pt" if network else "population.pt")
    torch.save(saved, path)
    receiving, _ = dn.func.make_functional(make_model(network), dt=DT)
    receiving_callbacks = receiving.make_callbacks({"trace": dn.func.Recorder(paths)})
    restored = dn.func.restore_continuation(
        receiving,
        torch.load(path, weights_only=True),
        callbacks=receiving_callbacks,
    )
    receiving_kernel = receiving.compile_rollout_chunk(STEPS, backend=backend)
    resumed = receiving_kernel.bind(restored)
    final, last = dn.func.run(
        resumed, restored, steps=STEPS, callbacks=receiving_callbacks
    )
    # endregion checkpoint
    torch.testing.assert_close(final, expected)
    for name, suffix in last["callbacks"]["trace"].items():
        joined = torch.cat((first["callbacks"]["trace"][name], suffix))
        torch.testing.assert_close(joined, whole["callbacks"]["trace"][name])
        assert joined.shape[0] == 2 * STEPS + 1
        assert suffix.shape[0] == STEPS

    # region detached_gradients
    report = dn.func.inspect_sources(restored)
    print(report)
    leaves = report.trainable_leaves
    gradients = torch.autograd.grad(
        trace_loss(last), (*leaves, gain), allow_unused=True
    )
    assert all(value is None or torch.isfinite(value).all() for value in gradients[:-1])
    assert any(value is not None and value.abs().sum() > 0 for value in gradients[:-1])
    assert gradients[-1] is None  # The portable boundary cuts the earlier graph.
    # endregion detached_gradients
    print(
        f"{'Network' if network else 'Population'}: explicit initialization, "
        "effective aliases, live gradients and portable continuation passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with (
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
        TemporaryDirectory(prefix="dendra-explicit-initialization-") as temporary,
    ):
        for network in (False, True):
            demo(network, args.backend, Path(temporary))


if __name__ == "__main__":
    main()
