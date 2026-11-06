"""Reuse raw Population and Network bindings across optimizer iterations.

Run with ``python examples/training/functional_raw_bindings.py`` from the repository.
The eager compiler backend keeps this small example quick to execute.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas

DT = 0.025


def _population():
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-55.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas, g=0.001, e=-70.0)
    model.train()
    model.initialize()
    return dn.func.make_functional(model, dt=DT)


def _network():
    cells = {}
    for name, voltage in (("left", -55.0), ("right", -60.0)):
        cell = dn.SingleCompartment(
            N=1,
            C=1,
            v_init=voltage,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        cell.insert(pas, g=0.001, e=-70.0)
        cells[name] = cell
    model = dn.Network(cells)
    model.train()
    model.initialize(DT)
    # Both populations use one trainable source, including after functionalization.
    cells["right"].mech.pas.g_param = cells["left"].mech.pas.g_param
    model.initialize(DT)
    return dn.func.make_functional_network(model)


def _loss(state):
    populations = state.get("populations", {"cell": state})
    return sum(
        population["integrator"]["v"].square().mean()
        for population in populations.values()
    )


def _train(label, functional, tensors, *, backend):
    parameters = tensors.independent_parameters(trainable="*g_param")
    constants = tensors.constants
    named = tensors._replace(parameters=parameters).trainable_parameters
    targets = tuple(
        {id(value): value for value in named.values() if value.requires_grad}.values()
    )
    optimizer = torch.optim.SGD(targets, lr=1.0e-8)
    state = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors.state
    )

    kernel = functional.compile_rollout_chunk(2, backend=backend)
    bound = kernel.bind(parameters, constants=constants)
    for iteration in range(3):
        # An independent run prepares its own autograd graph. The binding and
        # kernel survive optimizer updates, and the initial state is explicit.
        optimizer.zero_grad(set_to_none=True)
        final, _ = dn.func.run(bound, state, steps=5, host_span_steps=3)
        loss = _loss(final)
        loss.backward()
        assert all(
            value.grad is not None and torch.isfinite(value.grad).all()
            for value in targets
        )
        print(f"{label} iteration {iteration}: loss={loss.item():.6f}")
        optimizer.step()

    # The same operands also bind the eager transition with the same API.
    eager = functional.bind(parameters, constants=constants)
    with torch.no_grad():
        observed, _ = dn.func.run(bound, state, steps=5)
        expected, _ = dn.func.run(eager, state, steps=5)
        for actual, reference in zip(
            pytree.tree_leaves(observed), pytree.tree_leaves(expected), strict=True
        ):
            torch.testing.assert_close(actual, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="eager", choices=("eager", "aot_eager", "inductor")
    )
    options = parser.parse_args()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        _train("Population", *_population(), backend=options.backend)
        _train("Network", *_network(), backend=options.backend)


if __name__ == "__main__":
    main()
