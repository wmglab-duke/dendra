#!/usr/bin/env python3
"""Use one explicit lowering, binding and scheduling flow for both model kinds.

Run from the repository root::

    python examples/functional_lowering_api.py
    python examples/functional_lowering_api.py --backend aot_eager

The short CPU examples check ownership, gradients and continuation, not speed.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils import _pytree as pytree

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
    model.initialize(DT)
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


def run_model(model, *, paths, backend):
    # region lower_and_bind
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = tensors.independent_parameters(trainable="*.g_param")
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(paths)})
    kernel = functional.compile_rollout_chunk(WIDTH, backend=backend)
    bound = kernel.bind(parameters, constants=tensors.constants)
    # endregion lower_and_bind
    check_extraction(model, functional, tensors)

    # region run_and_continue
    prefix, first = dn.func.run(bound, tensors.state, steps=WIDTH, callbacks=callbacks)
    final, second = dn.func.run(
        bound,
        prefix,
        steps=STEPS - WIDTH,
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    expected, whole = dn.func.run(
        bound, tensors.state, steps=STEPS, callbacks=callbacks
    )
    # endregion run_and_continue
    torch.testing.assert_close(final, expected)
    for name, suffix in second["callbacks"]["trace"].items():
        joined = torch.cat((first["callbacks"]["trace"][name], suffix))
        torch.testing.assert_close(joined, whole["callbacks"]["trace"][name])
        assert joined.shape[0] == STEPS + 1  # One initial boundary frame.
    torch.testing.assert_close(functional.extract().state, tensors.state)

    targets = tuple(
        {
            id(value): value
            for value in pytree.tree_leaves(parameters)
            if isinstance(value, torch.Tensor) and value.requires_grad
        }.values()
    )
    trace = whole["callbacks"]["trace"]
    loss = sum(value.square().mean() for name, value in trace.items() if name != "t")
    gradients = torch.autograd.grad(loss, targets)
    assert targets
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    print(
        f"{type(functional).__name__}: shared lowering/bind/run, extraction ownership, "
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
