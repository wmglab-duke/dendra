#!/usr/bin/env python3
"""Replay one stochastic initialization in native and functional execution.

Run from the repository root::

    python examples/shared_initialization_draws.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.distributions import Normal
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import expsyn

DT, STEPS = 0.025, 4


class Gate(State):
    State.STATE("x")
    State.GLOBALRAND("tau", mu=1.0, sigma=0.02, seed=31, rng_name="stream")
    State.RANGENOISE("forcing", mu=0.0, sigma=0.01, seed=33, rng_name="stream")
    State.DERIVATIVE("x' = (0.3 + forcing - x) / tau")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class RandomLeak(Mechanism):
    Mechanism.STATE_BUNDLE(Gate)
    Mechanism.RANGE(e=-65.0, gain=0.002)
    Mechanism.GLOBALRAND(
        "g", distribution="lognormal", mu=-7.0, sigma=0.1, seed=37, rng_name="stream"
    )
    Mechanism.GLOBALRAND("offset", mu=0.03, sigma=0.001, seed=39, rng_name="stream")
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=41, rng_name="stream")
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
            self.i_inj + self.offset + self.x
        )


def make_model(network):
    cells = {}
    stream = None
    for name in ("left", "right") if network else ("cell",):
        cell = dn.SingleCompartment(
            N=1, C=1, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(RandomLeak.rename("leak"))
        if network:
            cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        cell.train()
        cell.initialize()
        owner = cell.mech.leak
        stream = owner.stream if stream is None else stream
        owner.stream = stream
        owner.DE.Gate.stream = stream
        pulse = dn.mono_rect(amp=0.2, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=stream.rng,
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
        weights = Normal(mean=0.003, std=0.0002)
        for cell in cells.values():
            model.connect_one_to_one(
                stimulus[:], cell[:], cell.mech.syn, weight=weights, delay=DT
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
    else:
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
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
    reference = make_model(network)
    functional, _ = dn.func.make_functional(reference, dt=DT)
    tensors = functional.extract(model)
    before = clone(model.state_dict_for_checkpoint())

    # region initialization
    # Capture follows the next initialization's native default options.
    draws = dn.func.sample_initialization_draws(model)
    assert_equal(model.state_dict_for_checkpoint(), before)
    initial = dn.func.initialize(functional, tensors, draws=draws)
    if network:
        model.initialize(DT, draws=draws)
    else:
        model.initialize(draws=draws)
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
    torch.testing.assert_close(functional.extract(model).state, initial.state)
    # endregion initialization

    # region continuation
    runtime = dn.func.sample_runtime_draws(functional, initial, steps=STEPS)
    final, _ = dn.func.run(functional, initial, steps=STEPS, draws=runtime)
    options = {} if network else {"dt": DT}
    model.run(steps=2, draws=sliced(runtime, 0, 2), **options)
    checkpoint = clone(model.state_dict_for_checkpoint())
    model.longrun_checkpointed(
        steps=STEPS - 2,
        chunklength=1,
        safe_checkpoint=True,
        draws=sliced(runtime, 2, STEPS),
        **options,
    )
    torch.testing.assert_close(functional.extract(model).state, final)
    # endregion continuation

    cells = tuple(model.populations.values()) if network else (model,)
    targets = tuple(
        next(value for name, value in cell.named_parameters() if name.endswith(suffix))
        for cell in cells
        for suffix in ("offset_mu_param", "waveform.amp")
    )
    parameters = (
        initial.parameters["populations"].values() if network else (initial.parameters,)
    )
    functional_targets = tuple(
        next(value for name, value in values.items() if name.endswith(suffix))
        for values in parameters
        for suffix in ("offset_mu_param", "waveform.amp")
    )
    actual_loss = sum(cell.v.square().mean() for cell in cells)
    states = final["populations"].values() if network else (final,)
    expected_loss = sum(state["integrator"]["v"].square().mean() for state in states)
    actual_gradient = torch.autograd.grad(actual_loss, targets)
    expected_gradient = torch.autograd.grad(expected_loss, functional_targets)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert all(
        value.isfinite().all() and value.abs().sum() > 0 for value in actual_gradient
    )

    receiver = make_model(network)
    receiver.restore_dict_from_checkpoint(checkpoint)
    suffix = dn.func.sample_runtime_draws(receiver, steps=STEPS - 2, **options)
    torch.testing.assert_close(suffix, sliced(runtime, 2, STEPS))
    receiver.longrun(steps=STEPS - 2, chunklength=1, draws=suffix, **options)
    torch.testing.assert_close(functional.extract(receiver).state, final)
    print(
        f"{'Network' if network else 'Population'}: shared initialization, gradients and checkpoint continuation passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        for network in (False, True):
            demonstrate(network)


if __name__ == "__main__":
    main()
