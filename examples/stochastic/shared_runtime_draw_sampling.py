#!/usr/bin/env python3
"""Sample runtime randomness directly from native Populations and Networks.

Run from the repository root::

    python examples/stochastic/shared_runtime_draw_sampling.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import expsyn

DT, STEPS = 0.025, 6


class Gate(State):
    State.STATE("x")
    State.RANGE(theta=0.7, sigma=0.03)
    State.RANGENOISE("forcing", mu=0.0, sigma=0.01, seed=31, rng_name="stream")
    State.DERIVATIVE("x' = theta * (0.3 - x) + forcing")
    State.DIFFUSION("x = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class InjectedLeak(Mechanism):
    Mechanism.STATE_BUNDLE(Gate)
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=37, rng_name="stream")
    Mechanism.ASSIGNED("i_inj")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def assigned_values(self, v, values):
        del values
        return {"i_inj": self.injection_values(v, t=self.t)}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.eta.clone() + self.gain * (
            self.i_inj + self.x
        )


def make_model(network):
    cells = {}
    for name in ("left", "right") if network else ("cell",):
        cell = dn.SingleCompartment(
            N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(InjectedLeak.rename("leak"))
        if network:
            cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        cell.train()
        cell.initialize()
        owner = cell.mech.leak
        gate = owner.DE.Gate
        gate.stream = owner.stream
        gate.x_dW_rng = owner.stream
        pulse = dn.mono_rect(amp=0.2, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=owner.stream.rng,
        )
        cell[:, 0].inject(pulse + pulse)
        cells[name] = cell
    if network:
        stimulus = dn.NetStim(
            N=1, interval=0.2, start=0.0, noise=0.5, max_spikes=20, seed=43
        )
        model = dn.Network(
            cells,
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for cell in cells.values():
            model.connect_one_to_one(
                stimulus[:], cell[:, 0], cell.mech.syn, weight=0.003, delay=DT
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


def demonstrate(network):
    model = make_model(network)
    options = {} if network else {"dt": DT}
    before = clone(model.state_dict_for_checkpoint())
    # region sample
    # An initialized native model supplies the current logical RNG boundary.
    draws = dn.func.sample_runtime_draws(model, steps=STEPS, **options)
    assert_equal(model.state_dict_for_checkpoint(), before)
    model.run(steps=2, draws=sliced(draws, 0, 2), **options)
    checkpoint = clone(model.state_dict_for_checkpoint())
    model.longrun_checkpointed(
        steps=STEPS - 2,
        chunklength=2,
        draws=sliced(draws, 2, STEPS),
        safe_checkpoint=True,
        **options,
    )
    # endregion sample
    final = clone(model.state_dict_for_checkpoint())
    cells = model.populations.values() if network else (model,)
    objective = sum(cell.v.square().mean() for cell in cells)
    amplitudes = tuple(
        value for name, value in model.named_parameters() if name.endswith("amp")
    )
    gradients = torch.autograd.grad(objective, amplitudes)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in gradients)
    assert_equal(model.state_dict_for_checkpoint(), final)

    receiver = make_model(network)
    receiver.restore_dict_from_checkpoint(checkpoint)
    # region continuation
    remaining = dn.func.sample_runtime_draws(receiver, steps=STEPS - 2, **options)
    assert_equal(remaining, sliced(draws, 2, STEPS))
    functional, _ = dn.func.make_functional(receiver, dt=DT)
    boundary = dn.func.capture_continuation(functional, receiver)
    expected, _ = dn.func.run(functional, boundary, steps=STEPS - 2, draws=remaining)
    receiver.longrun(steps=STEPS - 2, chunklength=1, draws=remaining, **options)
    torch.testing.assert_close(functional.extract(receiver).state, expected)
    torch.testing.assert_close(functional.extract(model).state, expected)
    # endregion continuation

    print(
        f"{'Network' if network else 'Population'}: native sampling, shared replay, gradients and checkpoints passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        for network in (False, True):
            demonstrate(network)


if __name__ == "__main__":
    main()
