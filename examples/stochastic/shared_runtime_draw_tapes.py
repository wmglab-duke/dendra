#!/usr/bin/env python3
"""Replay a stochastic Population or Network across native run methods.

Run from the repository root::

    python examples/stochastic/shared_runtime_draw_tapes.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import expsyn

DT, STEPS = 0.025, 7


class Gate(State):
    State.STATE("x")
    State.RANGE(theta=0.7, mu=0.2, sigma=0.03)
    State.DERIVATIVE("x' = theta * (mu - x)")
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


def clone(value):
    return pytree.tree_map_only(torch.Tensor, lambda leaf: leaf.detach().clone(), value)


def sliced(value, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda leaf: leaf[start:stop], value)


def assert_equal(actual, expected):
    leaves, structure = pytree.tree_flatten(actual)
    saved, saved_structure = pytree.tree_flatten(expected)
    assert structure == saved_structure
    for value, reference in zip(leaves, saved, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def make_model(network):
    populations = {}
    for name in ("left", "right") if network else ("cell",):
        cell = dn.SingleCompartment(
            N=2, C=1, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(NoisyLeak.rename("leak"))
        if network:
            cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        populations[name] = cell
    if network:
        stimulus = dn.NetStim(
            N=2, interval=0.2, start=0.0, noise=0.5, max_spikes=20, seed=37
        )
        model = dn.Network(
            populations,
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for cell in populations.values():
            model.connect_one_to_one(
                stimulus[:], cell[:], cell.mech.syn, weight=0.003, delay=DT
            )
    else:
        model = populations["cell"]
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


def configurations(network):
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    return {"trace": dn.Recorder(fields, dt=2 * DT)}


def inputs(current, network):
    packet = dn.func.Inputs(intra=current)
    return {"left": packet, "right": packet} if network else packet


def demonstrate(network):
    model = make_model(network)
    # region sample
    functional, _ = dn.func.make_functional(model, dt=DT)
    boundary = dn.func.capture_continuation(functional, model)
    draws = functional.sample_runtime_draws(boundary.state, steps=STEPS)
    current = torch.linspace(0.2e-9, 0.7e-9, STEPS, dtype=model.t.dtype).reshape(
        STEPS, 1, 1
    )
    current.requires_grad_()
    drives = inputs(current, network)
    expected, _ = dn.func.run(
        functional, boundary, steps=STEPS, inputs=drives, draws=draws
    )
    # endregion sample

    native_options = {} if network else {"dt": DT}
    callbacks = dn.func.make_callbacks(model, configurations(network), dt=DT)
    # region prefix
    prefix = model.run(
        steps=2,
        inputs=sliced(drives, 0, 2),
        draws=sliced(draws, 0, 2),
        callbacks=callbacks,
        **native_options,
    )
    checkpoint = clone(model.state_dict_for_checkpoint())
    callback_checkpoint = callbacks.export_state(prefix.state)
    loss, result = model.longrun_checkpointed(
        steps=STEPS - 2,
        chunklength=3,
        inputs=sliced(drives, 2, STEPS),
        draws=sliced(draws, 2, STEPS),
        callbacks=callbacks,
        callback_state=prefix.state,
        safe_checkpoint=True,
        **native_options,
    )
    assert loss is None
    # endregion prefix
    torch.testing.assert_close(functional.extract(model).state, expected)

    # region restore
    receiver = make_model(network)
    receiver.restore_dict_from_checkpoint(checkpoint)
    receiving_callbacks = dn.func.make_callbacks(
        receiver, configurations(network), dt=DT
    )
    receiving_state = receiving_callbacks.restore_state(callback_checkpoint)
    resumed = receiver.longrun(
        steps=STEPS - 2,
        chunklength=2,
        inputs=sliced(drives, 2, STEPS),
        draws=sliced(draws, 2, STEPS),
        callbacks=receiving_callbacks,
        callback_state=receiving_state,
        **native_options,
    )
    torch.testing.assert_close(resumed.outputs, result.outputs)
    torch.testing.assert_close(functional.extract(receiver).state, expected)
    # endregion restore

    cells = model.populations.values() if network else (model,)
    objective = sum(cell.v.square().mean() for cell in cells)
    final = clone(model.state_dict_for_checkpoint())
    gradient = torch.autograd.grad(objective, current)[0]
    assert gradient.isfinite().all() and gradient.abs().sum() > 0
    assert_equal(model.state_dict_for_checkpoint(), final)
    joined = torch.cat((prefix["trace"]["t"], result["trace"]["t"]))
    torch.testing.assert_close(
        joined, torch.arange(0, STEPS + 1, 2, dtype=model.t.dtype) * DT
    )
    print(
        f"{'Network' if network else 'Population'}: draw-tape replay, callback continuation, checkpoint restoration and gradients passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        for network in (False, True):
            demonstrate(network)


if __name__ == "__main__":
    main()
