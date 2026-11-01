#!/usr/bin/env python3
"""Replay invocation-aware Poisson draws through native Population steps.

Run from the repository root::

    python examples/shared_poisson_replay.py

The bounded CPU example covers solver-owned intra and call-local extra banks,
shared generators, repeated custom child calls, changed uniforms, gradients
and automatic continuation after native checkpoint restoration.
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas
from dendra.models.stim.waveform import Waveform
from dendra.units import nA

DT, STEPS = 0.025, 3


# region container
class OrderedWaveform(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETER(gain=0.7)

    def __init__(self, first, second):
        super().__init__()
        self.first, self.second = first, second

    def fn(self, times):
        second = self.second(times)
        first = self.first(times)
        repeated = self.first(times)
        return self.gain * (second + first + 0.25 * repeated)


# endregion container


def make_case(role):
    model = dn.Unmyelinated(
        [2.0, 2.5],
        L=4.0,
        dx=1.0,
        v_init=-65.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas, g=0.001, e=-65.0)
    generator = torch.Generator().manual_seed(17)
    pulse = dn.mono_rect(amp=0.2 * nA if role == "intra" else 0.2, pw=0.045)
    first = pulse.poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    second = pulse.poisson(
        interval=0.08,
        n=5,
        start=-0.02,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    waveform = OrderedWaveform(first, second)
    field = None
    if role == "intra":
        # The same physical child is called repeatedly in each registration.
        model[:, 1].inject(waveform)
        model[:, 2].inject(waveform)
    model.train()
    model.initialize()
    if role == "extra":
        field = torch.linspace(-1.25, 1.75, model.v.numel(), dtype=model.dtype())
        field = field.reshape(model.shape).requires_grad_()
    extra = None if field is None else (field, waveform)
    return model, waveform, field, extra


def clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def controlled_uniforms(packet, value, *, requires_grad=False):
    # Retain the sampled schema and authoritative RNG timeline while changing
    # the numerical realization used by every invocation.
    uniforms = {
        name: torch.full_like(draw, value).requires_grad_(requires_grad)
        for name, draw in packet.draws["stimulation"].items()
    }
    return packet._replace(draws={**packet.draws, "stimulation": uniforms})


def selected_sources(boundary, suffix):
    return next(
        value for name, value in boundary.parameters.items() if name.endswith(suffix)
    )


def demonstrate(role):
    model, waveform, field, extra = make_case(role)
    # region prepare
    functional, _ = dn.func.make_functional(model, dt=DT, extra=extra)
    boundary = dn.func.capture_continuation(functional, model)
    bound = functional.bind(boundary)
    expected = boundary.state
    # endregion prepare
    uniforms = []
    for index in range(STEPS):
        # region replay
        current = dn.func.capture_continuation(functional, model)
        sampled = functional.sample_runtime_draws(current.state)
        generator_before = waveform.first.generator.get_state().clone()
        packet = controlled_uniforms(sampled, 0.9, requires_grad=True)
        uniforms.extend(packet.draws["stimulation"].values())
        frozen = clone(packet)
        # Public helpers and the direct method consume the same packet.
        advance = (model.step, dn.step, dn.step_population)[index]
        if index == 0:
            advance(DT, extra=extra, draws=packet)
        else:
            advance(model, dt=DT, extra=extra, draws=packet)
        expected, _ = bound(expected, draws=packet)
        # endregion replay
        torch.testing.assert_close(model.v, expected["integrator"]["v"])
        torch.testing.assert_close(functional.extract(model).state, expected)
        torch.testing.assert_close(packet, frozen)
        assert waveform.first.generator is waveform.second.generator
        assert waveform.first.waveform is waveform.second.waveform
        if index == 0:
            # A second controlled realization proves that supplied uniforms
            # affect the trajectory rather than triggering native resampling.
            other, _ = bound(boundary.state, draws=controlled_uniforms(sampled, 0.1))
            assert not torch.allclose(
                model.v, other["integrator"]["v"], rtol=1e-12, atol=1e-12
            )
        assert not torch.equal(waveform.first.generator.get_state(), generator_before)

    # region gradients
    native_sources = (waveform.first.waveform.amp, waveform.gain)
    reference_sources = (
        selected_sources(boundary, ".waveform.amp"),
        selected_sources(boundary, ".gain"),
    )
    if field is not None:
        native_sources += (field,)
        reference_sources += (selected_sources(boundary, ".contacts.0.field"),)
    observed = torch.autograd.grad(
        model.v.square().mean(), (*native_sources, *uniforms), allow_unused=True
    )
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), reference_sources
    )
    for actual, reference in zip(observed[: len(native_sources)], wanted, strict=True):
        torch.testing.assert_close(actual, reference)
        assert actual.isfinite().all() and actual.abs().sum() > 0
    assert all(value is None for value in observed[len(native_sources) :])
    # endregion gradients

    # region checkpoint
    saved = clone(model.state_dict_for_checkpoint())
    # Call-local extra is external configuration, so save its own state too.
    external = (
        None
        if extra is None
        else {
            "waveform": clone(waveform.state_dict()),
            "rng": waveform.first.generator.get_state().clone(),
        }
    )
    current = dn.func.capture_continuation(functional, model)
    suffix_draws = functional.sample_runtime_draws(current.state)
    reference, _ = bound(expected, draws=suffix_draws)
    model.step(DT, extra=extra)  # Resume ordinary sampling from the packet suffix.
    torch.testing.assert_close(model.v, reference["integrator"]["v"])

    restored, restored_waveform, _, restored_extra = make_case(role)
    restored.restore_dict_from_checkpoint(saved)
    if external is not None:
        restored_waveform.load_state_dict(external["waveform"])
        restored_waveform.first.generator.set_state(external["rng"])
    restored.step(DT, extra=restored_extra)
    torch.testing.assert_close(restored.v, model.v)
    torch.testing.assert_close(
        restored_waveform.first.generator.get_state(),
        waveform.first.generator.get_state(),
    )
    # endregion checkpoint
    print(
        f"{role}: changed-uniform replay, shared invocations, gradients and checkpoint continuation passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for role in ("intra", "extra"):
            demonstrate(role)


if __name__ == "__main__":
    main()
