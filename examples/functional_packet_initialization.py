#!/usr/bin/env python3
"""Start independent training trials from effective sources and current RNG carry.

Run from the repository root::

    python examples/functional_packet_initialization.py
    python examples/functional_packet_initialization.py --backend aot_eager

The bounded CPU example covers Population and Network runners, optimizer-owned
parameter aliases, fresh callback histories, initialization draw replay and
portable continuation. Each trial has an independent backward pass.
"""

from __future__ import annotations

import argparse
import io

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT, STEPS, TRIALS = 0.1, 4, 3
G = "integrator.mech.mechanisms.leak.g_param"


class RandomLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-55.0)
    Mechanism.GLOBALRAND(
        "offset",
        mu=0.00001,
        sigma=0.000001,
        seed=17,
        reseed_on_initialize=False,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.offset


def make_model(network):
    populations = {}
    for name in ("left", "right") if network else ("cell",):
        population = dn.SingleCompartment(
            N=1, C=1, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
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


def conductance(packet, network):
    return (
        packet.parameters["populations"]["left"][G] if network else packet.parameters[G]
    )


def run_demo(network, backend):
    model = make_model(network)
    names = ("left.v", "right.v", "t") if network else ("v", "t")
    program = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        chunk_steps=2,
        checkpoint_every=2,
        inference=False,
        trainable="*g_param",
        callbacks={"trace": dn.Recorder(names)},
        backend=backend,
    )
    # region training
    trial = dn.func.initialize(program)
    parameters = dn.func.inspect_sources(trial).trainable_leaves
    optimizer = torch.optim.SGD(parameters, lr=1e-8)
    for _ in range(TRIALS):
        optimizer.zero_grad(set_to_none=True)
        _, auxiliary = dn.func.run(program, trial)
        field = "left.v" if network else "v"
        voltage = auxiliary["callbacks"]["trace"][field]
        loss = (voltage - (-63.0)).square().mean()
        loss.backward()
        optimizer.step()

        finished = dn.func.detach_continuation(program, auxiliary["continuation"])
        trial = dn.func.initialize(program, finished)
        # endregion training
        assert len(parameters) == 1
        assert conductance(trial, network) is parameters[0]
        assert parameters[0].grad is not None
        assert torch.isfinite(parameters[0].grad).all()
        assert parameters[0].grad.abs().sum() > 0
        if network:
            assert (
                conductance(trial, True) is trial.parameters["populations"]["right"][G]
            )
        assert trial.callback_state is None
        torch.testing.assert_close(
            auxiliary["callbacks"]["trace"]["t"],
            torch.arange(STEPS + 1, dtype=torch.float64) * DT,
        )
        torch.testing.assert_close(voltage[0], torch.full_like(voltage[0], -65.0))

    # Capture reads the continuation's current sources and RNG carry, without
    # advancing that packet. Replay and automatic sampling select the same draws.
    # region replay
    draws = dn.func.sample_initialization_draws(program, trial)
    replayed = dn.func.initialize(program, trial, draws=draws)
    automatic = dn.func.initialize(program, trial)
    torch.testing.assert_close(replayed.state, automatic.state)
    torch.testing.assert_close(replayed.parameters, automatic.parameters)
    torch.testing.assert_close(replayed.constants, automatic.constants)
    # endregion replay
    assert replayed.callback_state is None
    assert conductance(replayed, network) is parameters[0]

    # A portable checkpoint restores independent leaves and the selected RNG
    # position. Restarting either packet follows the same initialization protocol.
    buffer = io.BytesIO()
    torch.save(dn.func.export_continuation(program, trial), buffer)
    buffer.seek(0)
    restored = dn.func.restore_continuation(
        program, torch.load(buffer, weights_only=True)
    )
    restarted = dn.func.initialize(program, restored)
    torch.testing.assert_close(restarted.state, automatic.state)
    assert conductance(restarted, network) is not parameters[0]
    assert conductance(restarted, network).requires_grad
    assert model.t.item() == 0.0
    print(
        f"{'Network' if network else 'Population'}: independent training trials, "
        "optimizer identity, fresh recording, draw replay and checkpoint restart passed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aot_eager"), default="eager")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            run_demo(network, args.backend)


if __name__ == "__main__":
    main()
