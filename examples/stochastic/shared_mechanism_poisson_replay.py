#!/usr/bin/env python3
"""Replay mechanism-owned Poisson injections in native Population and Network runs.

Run from the repository root::

    python examples/stochastic/shared_mechanism_poisson_replay.py
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT, STEPS = 0.025, 5


class InjectedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
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
        return self.g, self.g * self.e + self.gain * self.i_inj


def make_model(network):
    generator = torch.Generator().manual_seed(73)
    populations = {}
    for name in ("left", "right") if network else ("cell",):
        cell = dn.SingleCompartment(
            N=1, C=2, v_init=-47.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(InjectedLeak.rename("owner"))
        pulse = dn.mono_rect(amp=0.2, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=generator,
        )
        # Slice.inject delegates to this mechanism's injection hook.
        # Both evaluations of the shared pulse receive their own invocation draws.
        cell[:, 0].inject(pulse + pulse)
        populations[name] = cell
    model = dn.Network(populations) if network else populations["cell"]
    model.train()
    model.initialize(DT) if network else model.initialize()
    return model


def clone(value):
    return pytree.tree_map_only(torch.Tensor, lambda leaf: leaf.detach().clone(), value)


def sliced(value, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda leaf: leaf[start:stop], value)


def demonstrate(network):
    model = make_model(network)
    # region replay
    functional, _ = dn.func.make_functional(model, dt=DT)
    boundary = dn.func.capture_continuation(functional, model)
    draws = functional.sample_runtime_draws(boundary.state, steps=STEPS)
    expected, _ = dn.func.run(functional, boundary, steps=STEPS, draws=draws)

    options = {} if network else {"dt": DT}
    model.run(steps=2, draws=sliced(draws, 0, 2), **options)
    checkpoint = clone(model.state_dict_for_checkpoint())
    model.longrun_checkpointed(
        steps=STEPS - 2,
        chunklength=2,
        draws=sliced(draws, 2, STEPS),
        safe_checkpoint=True,
        **options,
    )
    torch.testing.assert_close(functional.extract(model).state, expected)
    # endregion replay

    amplitudes = tuple(
        value for name, value in model.named_parameters() if name.endswith("amp")
    )
    cells = model.populations.values() if network else (model,)
    objective = sum(cell.v.square().mean() for cell in cells)
    final = clone(model.state_dict_for_checkpoint())
    gradients = torch.autograd.grad(objective, amplitudes)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in gradients)
    torch.testing.assert_close(functional.extract(model).state, expected)

    receiver = make_model(network)
    receiver.restore_dict_from_checkpoint(checkpoint)
    receiver.longrun(
        steps=STEPS - 2,
        chunklength=1,
        draws=sliced(draws, 2, STEPS),
        **options,
    )
    torch.testing.assert_close(functional.extract(receiver).state, expected)
    # Compare tensor state rather than the checkpoint's non-tensor metadata.
    leaves = pytree.tree_leaves(model.state_dict_for_checkpoint())
    saved = pytree.tree_leaves(final)
    for value, reference in zip(leaves, saved, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference
    print(
        f"{'Network' if network else 'Population'}: mechanism Poisson replay, gradients and checkpoint continuation passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False):
        for network in (False, True):
            demonstrate(network)


if __name__ == "__main__":
    main()
