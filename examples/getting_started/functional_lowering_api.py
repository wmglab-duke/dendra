#!/usr/bin/env python3
"""Use one explicit lowering, binding and scheduling flow for both model kinds.

Run from the repository root::

    python examples/getting_started/functional_lowering_api.py
    python examples/getting_started/functional_lowering_api.py --backend aot_eager

The short CPU examples verify source inspection, ownership, gradients and
continuation.
"""

from __future__ import annotations

import argparse

import torch

import dendra as dn
from dendra.models.mod import graded_syn, pas

DT, STEPS, WIDTH = 0.025, 4, 2


def cell(*, initial=-60.0, reversal=-70.0):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=initial, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=reversal)
    return model


def make_population():
    model = cell()
    model.train()
    model.initialize()
    return model


def release(voltage):
    return torch.sigmoid(0.1 * (voltage + 60.0))


def make_network():
    left = cell(reversal=-55.0)
    right = cell(initial=-65.0)
    right.insert(graded_syn.rename("continuous"), e=0.0)
    network = dn.Network({"left": left, "right": right})
    # Raw parameter ties remain aliases in the extracted Network snapshot.
    right.mech.leak.g_param = left.mech.leak.g_param
    network.connect_continuous_one_to_one(
        left[:],
        right[:],
        right.mech.continuous,
        pre_var="v",
        input="g_pre",
        weight=0.0005,
        delay=0.0,
        transform=release,
    )
    network.train()
    network.initialize(DT)
    return network


def check_extraction(model, functional, tensors):
    if isinstance(tensors, dn.func.NetworkTensors):
        assert isinstance(functional, dn.func.FunctionalNetwork)
        left = tensors.parameters["populations"]["left"]
        right = tensors.parameters["populations"]["right"]
        name = next(name for name in left if name.endswith("g_param"))
        assert left[name] is right[name]
        assert left[name] is not model.populations["left"].mech.leak.g_param
        # The implicit-timestep adapter has the same concrete return types.
        compatible, legacy = dn.func.make_functional_network(model)
        assert isinstance(compatible, type(functional))
        torch.testing.assert_close(legacy.state, tensors.state)
    else:
        assert isinstance(functional, dn.func.FunctionalPopulation)
        name = next(name for name in tensors.parameters if name.endswith("g_param"))
        assert tensors.parameters[name] is model.mech.leak.g_param


def source_signature(report):
    """Compare source metadata without comparing Tensor contents."""
    return tuple(
        (
            entry.path,
            id(entry.tensor),
            entry.aliases,
            entry.requires_grad,
            entry.is_leaf,
        )
        for entry in report.entries
    )


def run_model(model, *, paths, backend):
    # region lower_and_bind
    functional, tensors = dn.func.make_functional(model, dt=DT)
    # The extracted bundle reports exactly its stored parameter/constant frames.
    extracted = dn.func.inspect_sources(tensors)
    print(extracted)
    parameters = tensors.independent_parameters(trainable="*.g_param")
    selected = tensors._replace(parameters=parameters)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(paths)})
    eager = functional.bind(selected)
    kernel = functional.compile_rollout_chunk(WIDTH, backend=backend)
    bound = kernel.bind(selected)
    # A raw binding reports the sources it consumes, including canonical
    # parameter-owned readers in a Network's constant frame.
    bound_report = dn.func.inspect_sources(bound)
    print(bound_report)
    # endregion lower_and_bind
    check_extraction(model, functional, tensors)
    extracted_signature = source_signature(extracted)
    bound_signature = source_signature(bound_report)
    assert source_signature(dn.func.inspect_sources(eager)) == bound_signature
    assert all(
        value.requires_grad and value.is_leaf for value in bound_report.trainable_leaves
    )
    assert len(bound_report.trainable_leaves) == len(
        {id(value) for value in bound_report.trainable_leaves}
    )

    # region run_and_continue
    _, first = dn.func.run(bound, tensors.state, steps=WIDTH, callbacks=callbacks)
    final, second = dn.func.run(
        bound,
        first["continuation"],
        steps=STEPS - WIDTH,
        callbacks=callbacks,
    )
    expected, whole = dn.func.run(
        eager, tensors.state, steps=STEPS, callbacks=callbacks
    )
    # endregion run_and_continue
    # region inspect_continuation
    current = dn.func.inspect_sources(second["continuation"])
    print(current)
    assert source_signature(current) == bound_signature
    # endregion inspect_continuation
    assert source_signature(dn.func.inspect_sources(tensors)) == extracted_signature
    assert source_signature(dn.func.inspect_sources(bound)) == bound_signature
    torch.testing.assert_close(final, expected)
    for name, suffix in second["callbacks"]["trace"].items():
        joined = torch.cat((first["callbacks"]["trace"][name], suffix))
        torch.testing.assert_close(joined, whole["callbacks"]["trace"][name])
        assert joined.shape[0] == STEPS + 1  # One initial boundary frame.
    torch.testing.assert_close(functional.extract().state, tensors.state)

    targets = bound_report.trainable_leaves
    trace = whole["callbacks"]["trace"]
    loss = sum(value.square().mean() for name, value in trace.items() if name != "t")
    gradients = torch.autograd.grad(loss, targets, allow_unused=True)
    assert targets
    assert all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    )
    by_identity = {
        id(value): gradient for value, gradient in zip(targets, gradients, strict=True)
    }
    conductances = {
        id(entry.tensor)
        for entry in bound_report.entries
        if entry.domain == "parameters" and entry.path.endswith("leak.g_param")
    }
    assert conductances
    for identity in conductances:
        gradient = by_identity[identity]
        assert gradient is not None and gradient.abs().sum() > 0
    print(
        f"{type(functional).__name__}: shared lowering/bind/run inspection, extraction ownership, "
        "recording continuation and training gradients passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="eager")
    args = parser.parse_args()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        run_model(make_population(), paths=("v", "t"), backend=args.backend)
        run_model(
            make_network(), paths=("left.v", "right.v", "t"), backend=args.backend
        )


if __name__ == "__main__":
    main()
