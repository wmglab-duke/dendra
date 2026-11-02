#!/usr/bin/env python3
"""Sample matching draw tapes for elapsed-duration execution.

Run from the repository root::

    python examples/shared_runtime_draw_horizons.py

The bounded CPU example covers Population and Network execution, retained
fractional time, read-only sampling, gradients and native/functional checkpoints.
"""

from __future__ import annotations

import io

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import expsyn

DT = 0.025
PENDING = 0.5 * DT
DURATION = 2.5 * DT
STEPS = 3


class Gate(State):
    State.STATE("x")
    State.RANGE(theta=0.7, sigma=0.03)
    State.DERIVATIVE("x' = theta * (0.3 - x)")
    State.DIFFUSION("x = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class NoisyLeak(Mechanism):
    Mechanism.STATE_BUNDLE(Gate)
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00001, seed=31)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + 0.002 * self.x - self.eta.clone()


def make_model(network):
    cells = {}
    for name in ("left", "right") if network else ("cell",):
        cell = dn.SingleCompartment(
            N=2, C=1, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(NoisyLeak.rename("leak"))
        if network:
            cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        cells[name] = cell
    if network:
        stimulus = dn.NetStim(
            N=2, interval=0.2, start=0.0, noise=0.5, max_spikes=20, seed=37
        )
        model = dn.Network(
            cells,
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for cell in cells.values():
            model.connect_one_to_one(
                stimulus[:], cell[:], cell.mech.syn, weight=0.003, delay=DT
            )
    else:
        model = cells["cell"]
    model.train()
    model.initialize(DT) if network else model.initialize()
    if network:
        model.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=False,
            train_delay_backend="dense",
        )
    return model


def clone(value):
    return pytree.tree_map_only(torch.Tensor, lambda leaf: leaf.detach().clone(), value)


def assert_equal(actual, expected):
    leaves, structure = pytree.tree_flatten(actual)
    references, expected_structure = pytree.tree_flatten(expected)
    assert structure == expected_structure
    for value, reference in zip(leaves, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def voltage_loss(state, network):
    frames = state["populations"].values() if network else (state,)
    return sum(frame["integrator"]["v"].square().mean() for frame in frames)


def demonstrate(network):
    model = make_model(network)
    native_options = {} if network else {"dt": DT}
    # A substep duration has no numerical or stochastic step to execute.
    empty = dn.func.sample_runtime_draws(model, duration=PENDING, **native_options)
    model.run(duration=PENDING, draws=empty, **native_options)
    torch.testing.assert_close(model.t, model.t.new_tensor(0.0))
    checkpoint = clone(model.state_dict_for_checkpoint())
    pending = checkpoint["duration_remainder"]
    torch.testing.assert_close(pending, pending.new_tensor(PENDING))

    program = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        chunk_steps=2,
        inference=False,
        trainable="*g_param",
        backend="eager",
        disable=True,
    )
    boundary = dn.func.capture_continuation(program)
    # region duration
    # The pending half-step plus this duration accepts three complete steps.
    draws = dn.func.sample_runtime_draws(program, boundary, duration=DURATION)
    current = (
        torch.linspace(0.2e-9, 0.7e-9, STEPS, dtype=torch.float64)
        .reshape(STEPS, 1, 1)
        .requires_grad_()
    )
    inputs = dn.func.Inputs(intra=current)
    drives = {"left": inputs, "right": inputs} if network else inputs
    expected, auxiliary = dn.func.run(
        program, boundary, duration=DURATION, inputs=drives, draws=draws
    )
    # endregion duration
    torch.testing.assert_close(
        draws, dn.func.sample_runtime_draws(model, duration=DURATION, **native_options)
    )
    assert_equal(model.state_dict_for_checkpoint(), checkpoint)
    wanted = torch.autograd.grad(voltage_loss(expected, network), current)[0]

    # Portable continuation restoration retains the same pending time and RNG.
    buffer = io.BytesIO()
    torch.save(dn.func.export_continuation(program, boundary), buffer)
    buffer.seek(0)
    restored = dn.func.restore_continuation(
        program, torch.load(buffer, weights_only=True)
    )
    torch.testing.assert_close(
        dn.func.sample_runtime_draws(program, restored, duration=DURATION), draws
    )
    with torch.no_grad():
        repeated, _ = dn.func.run(
            program, restored, duration=DURATION, inputs=drives, draws=draws
        )
    torch.testing.assert_close(repeated, expected)

    # Sampling does not consume the half-step; execution commits its result once.
    model.longrun_checkpointed(
        duration=DURATION,
        chunklength=2,
        inputs=drives,
        draws=draws,
        safe_checkpoint=True,
        **native_options,
    )
    cells = model.populations.values() if network else (model,)
    objective = sum(cell.v.square().mean() for cell in cells)
    observed = torch.autograd.grad(objective, current)[0]
    torch.testing.assert_close(observed, wanted)
    assert observed.isfinite().all() and observed.abs().sum() > 0
    torch.testing.assert_close(program.functional.extract(model).state, expected)
    torch.testing.assert_close(model.t, model.t.new_tensor(STEPS * DT))
    remainder = model.state_dict_for_checkpoint()["duration_remainder"]
    torch.testing.assert_close(remainder, torch.zeros_like(remainder))

    receiver = make_model(network)
    receiver.restore_dict_from_checkpoint(checkpoint)
    resumed_draws = dn.func.sample_runtime_draws(
        receiver, duration=DURATION, **native_options
    )
    torch.testing.assert_close(resumed_draws, draws)
    receiver.longrun(
        duration=DURATION,
        chunklength=1,
        inputs=drives,
        draws=resumed_draws,
        **native_options,
    )
    torch.testing.assert_close(program.functional.extract(receiver).state, expected)
    assert auxiliary["continuation"].callback_state is None
    print(
        f"{'Network' if network else 'Population'}: duration tapes, fractional carry, "
        "gradients and checkpoint continuation passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        for network in (False, True):
            demonstrate(network)


if __name__ == "__main__":
    main()
