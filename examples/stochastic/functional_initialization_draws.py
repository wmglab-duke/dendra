#!/usr/bin/env python3
"""Capture and replay initialization draws for Populations and Networks.

Run from the repository root::

    python examples/stochastic/functional_initialization_draws.py
    python examples/stochastic/functional_initialization_draws.py --backend aot_eager
    python examples/stochastic/functional_initialization_draws.py --section network

The bounded CPU example checks explicit versus automatic initialization, source
RNG isolation, shared parameters, scan gradients and portable continuation.
"""

from __future__ import annotations

import argparse
import io

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT = 0.1
STEPS = 4
PREFIX = 2


class RandomLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-55.0)
    Mechanism.GLOBALRAND("offset", mu=0.00001, sigma=0.000001, seed=17)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.offset


def make_model(network):
    populations = {}
    for name in ("left", "right") if network else ("cell",):
        population = dn.SingleCompartment(
            N=1, C=2, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(RandomLeak.rename("leak"))
        populations[name] = population
    if network:
        model = dn.Network(populations)
        populations["right"].mech.leak.g_param = populations["left"].mech.leak.g_param
    else:
        model = populations["cell"]
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def compile_program(model, network, backend):
    return dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        chunk_steps=PREFIX,
        checkpoint_every=PREFIX,
        inference=False,
        execution="scan",
        trainable="*g_param",
        callbacks={
            "trace": dn.Recorder(("left.v", "right.v", "t") if network else ("v", "t"))
        },
        backend=backend,
    )


def run_demo(network, backend):
    model = make_model(network)
    program = compile_program(model, network, backend)
    global_rng = torch.get_rng_state().clone()
    # region draws
    draws = dn.func.sample_initialization_draws(program)
    initial = dn.func.initialize(program, draws=draws)
    automatic = dn.func.initialize(program)
    torch.testing.assert_close(initial.state, automatic.state)
    torch.testing.assert_close(initial.parameters, automatic.parameters)
    torch.testing.assert_close(initial.constants, automatic.constants)
    # Capturing draws returns no new simulation state and starts no callbacks.
    assert isinstance(draws, dn.func.InitializationDraws)
    assert initial.callback_state is None
    # endregion draws
    torch.testing.assert_close(torch.get_rng_state(), global_rng)
    assert model.t.item() == 0.0
    if network:
        shared = "integrator.mech.mechanisms.leak.g_param"
        assert (
            initial.parameters["populations"]["left"][shared]
            is initial.parameters["populations"]["right"][shared]
        )

    # The same API accepts explicit lowered models and their tensor bundles.
    # region explicit
    functional, tensors = dn.func.make_functional(make_model(network), dt=DT)
    explicit_draws = dn.func.sample_initialization_draws(functional, tensors)
    explicit = dn.func.initialize(functional, tensors, draws=explicit_draws)
    torch.testing.assert_close(explicit.state, initial.state)
    # endregion explicit

    # region continuation
    _, first = program(initial, steps=PREFIX)
    actual, last = program(first["continuation"], steps=STEPS - PREFIX)
    expected, whole = program(initial)
    fields = ("left.v", "right.v") if network else ("v",)
    joined = {
        field: torch.cat(
            (first["callbacks"]["trace"][field], last["callbacks"]["trace"][field])
        )
        for field in fields
    }
    torch.testing.assert_close(actual, expected)
    for field, trace in joined.items():
        torch.testing.assert_close(trace, whole["callbacks"]["trace"][field])
    targets = tuple(program.trainable_parameters.values())
    gradients = torch.autograd.grad(
        sum(value.square().mean() for value in joined.values()),
        targets,
        retain_graph=True,
    )
    reference = torch.autograd.grad(
        sum(whole["callbacks"]["trace"][field].square().mean() for field in fields),
        targets,
    )
    for observed, wanted in zip(gradients, reference, strict=True):
        torch.testing.assert_close(observed, wanted)
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0
    # endregion continuation

    buffer = io.BytesIO()
    torch.save(program.export_continuation(first["continuation"]), buffer)
    buffer.seek(0)
    receiver = compile_program(make_model(network), network, backend)
    restored = receiver.restore_continuation(torch.load(buffer, weights_only=True))
    resumed, suffix = receiver(restored, steps=STEPS - PREFIX)
    torch.testing.assert_close(resumed, actual)
    for field in fields:
        torch.testing.assert_close(
            suffix["callbacks"]["trace"][field], last["callbacks"]["trace"][field]
        )
    print(
        f"{'Network' if network else 'Population'}: initialization draw replay, "
        "gradients and portable continuation passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    parser.add_argument(
        "--section", choices=("all", "population", "network"), default="all"
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            section = "network" if network else "population"
            if arguments.section in ("all", section):
                run_demo(network, arguments.backend)


if __name__ == "__main__":
    main()
