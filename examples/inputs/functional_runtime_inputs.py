"""Shared deterministic inputs, gradients, recording and continuation.

Run from the repository with ``python examples/inputs/functional_runtime_inputs.py``.
Use ``--backend aot_eager`` for AOT compilation of the bounded chunks.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas

DT, STEPS = 0.025, 6


def _population():
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=torch.linspace(-58.0, -52.0, 5),
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas, g=0.001, e=-70.0)
    model.train()
    model.initialize()
    return dn.func.make_functional(model, dt=DT)


def _network():
    cells = {}
    for name, count in (("left", 1), ("right", 2)):
        cell = dn.SingleCompartment(
            N=count,
            C=1,
            v_init=-55.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        cell.insert(pas, g=0.001, e=-70.0)
        cells[name] = cell
    model = dn.Network(cells)
    model.train()
    model.initialize(DT)
    return dn.func.make_functional_network(model)


def _drives(voltage, *, cable):
    shape = (STEPS, *voltage.shape)
    intra = voltage.new_full(shape, 0.02).requires_grad_()
    ve = (
        torch.linspace(
            -0.1, 0.2, intra.numel(), dtype=voltage.dtype, device=voltage.device
        )
        .reshape(shape)
        .requires_grad_()
        if cable
        else None
    )
    return dn.func.Inputs(ve=ve, intra=intra)


def _loss(state):
    populations = state.get("populations", {"cell": state})
    return sum(
        value["integrator"]["v"].square().mean() for value in populations.values()
    )


def _check(functional, tensors, *, backend):
    network = isinstance(functional, dn.func.FunctionalNetwork)
    parameters = tensors.independent_parameters(trainable="*g_param")
    constants = tensors.constants
    initial = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors.state
    )
    if network:
        inputs = {
            name: _drives(state["integrator"]["v"], cable=False)
            for name, state in initial["populations"].items()
        }
        legacy = dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(packet.ve, packet.intra)
                for name, packet in inputs.items()
            }
        )
        fields = ("left.v", "right.v", "t")
    else:
        inputs = _drives(initial["integrator"]["v"], cable=True)
        legacy = dn.func.RolloutInput(inputs.ve, inputs.intra)
        fields = ("v", "t")
    kernel = functional.compile_rollout_chunk(2, backend=backend)
    bound = kernel.bind(parameters, constants=constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(fields)})
    final, auxiliary = dn.func.run(
        bound,
        initial,
        inputs=inputs,
        steps=STEPS,
        checkpoint_every=3,
        callbacks=callbacks,
    )
    reference, _ = functional.prepare_and_rollout(
        parameters, constants, initial, legacy, steps=STEPS
    )
    torch.testing.assert_close(final, reference)
    targets = tuple(
        {
            id(value): value
            for value in pytree.tree_leaves((parameters, inputs))
            if torch.is_tensor(value) and value.requires_grad
        }.values()
    )
    actual_gradient = torch.autograd.grad(_loss(final), targets)
    expected_gradient = torch.autograd.grad(_loss(reference), targets)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0
        for value in actual_gradient
    )
    with torch.no_grad():
        head_inputs = pytree.tree_map_only(
            torch.Tensor, lambda value: value[:3], inputs
        )
        tail_inputs = pytree.tree_map_only(
            torch.Tensor, lambda value: value[3:], inputs
        )
        first, head = dn.func.run(
            bound, initial, inputs=head_inputs, callbacks=callbacks
        )
        resumed, tail = dn.func.run(
            bound,
            first,
            inputs=tail_inputs,
            callbacks=callbacks,
            callback_state=head["callbacks"].state,
        )
        torch.testing.assert_close(resumed, final)
        for name, trace in auxiliary["callbacks"]["trace"].items():
            combined = torch.cat(
                (head["callbacks"]["trace"][name], tail["callbacks"]["trace"][name])
            )
            torch.testing.assert_close(combined, trace)
            assert combined.shape[0] == STEPS + 1
    label = "Network" if network else "Population"
    print(f"{label}: shared inputs, gradients, recording and continuation passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="eager", choices=("eager", "aot_eager", "inductor")
    )
    options = parser.parse_args()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        _check(*_population(), backend=options.backend)
        _check(*_network(), backend=options.backend)


if __name__ == "__main__":
    main()
