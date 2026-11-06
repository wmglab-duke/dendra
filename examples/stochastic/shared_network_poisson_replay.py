#!/usr/bin/env python3
"""Replay shared Poisson waveform invocations through native Network steps.

Run from the repository root::

    python examples/stochastic/shared_network_poisson_replay.py

The bounded CPU example covers intra/extra banks, repeated physical children
shared by two Populations, changed uniforms, gradients and automatic sampling
after native checkpoint restoration. Extra bindings use reversed Population
order and retain their caller-owned waveform and RNG state.
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas
from dendra.units import nA

DT, STEPS = 0.025, 3


def make_case(role):
    generator = torch.Generator().manual_seed(17)
    pulse = dn.mono_rect(amp=0.2 * nA if role == "intra" else 0.2, pw=0.045)
    pulse = pulse.poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    expression = pulse + pulse
    populations = {}
    for name in ("left", "right"):
        population = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(pas, g=0.001, e=-65.0)
        if role == "intra":
            population[:, 1].inject(expression)
        populations[name] = population
    model = dn.Network(populations)
    model.train()
    model.initialize(DT)
    fields, extra = {}, None
    if role == "extra":
        fields = {
            name: torch.linspace(-1.25, 1.75, cell.v.numel(), dtype=cell.dtype())
            .reshape(cell.shape)
            .requires_grad_()
            for name, cell in model.populations.items()
        }
        # Extracellular bank order is the supplied mapping's order.
        extra = {name: (fields[name], expression) for name in ("right", "left")}
    return model, pulse, expression, fields, extra


def clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def controlled_uniforms(packet, *, alternative=False):
    # Numerical realization and the authoritative sampled RNG timeline remain
    # separate: changing uniforms does not manufacture a different RNG endpoint.
    populations = {}
    for index, (name, local) in enumerate(packet.draws["populations"].items()):
        uniforms = {}
        for key, value in local["stimulation"].items():
            uniforms[key] = (
                torch.full_like(value, 0.9)
                if alternative
                else torch.linspace(
                    0.1 + 0.1 * index,
                    0.75 + 0.1 * index,
                    value.numel(),
                    dtype=value.dtype,
                    device=value.device,
                ).reshape(value.shape)
            ).requires_grad_()
        populations[name] = {**local, "stimulation": uniforms}
    return packet._replace(draws={**packet.draws, "populations": populations})


def source(parameters, suffix):
    return next(value for name, value in parameters.items() if name.endswith(suffix))


def state_loss(state):
    return sum(
        local["integrator"]["v"].square().mean()
        for local in state["populations"].values()
    )


def demonstrate(role):
    model, pulse, expression, fields, extra = make_case(role)
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
        packet = controlled_uniforms(sampled)
        frozen = clone(packet)
        uniforms.extend(
            value
            for local in packet.draws["populations"].values()
            for value in local["stimulation"].values()
        )
        advance = (model.step, dn.step, dn.step_network)[index]
        if index == 0:
            advance(extra=extra, draws=packet)
        else:
            advance(model, extra=extra, draws=packet)
        expected, _ = bound(expected, inputs={}, draws=packet)
        # endregion replay
        torch.testing.assert_close(functional.extract(model).state, expected)
        torch.testing.assert_close(packet, frozen)
        assert expression.waveforms[0] is expression.waveforms[1] is pulse
        if index == 0:
            other, _ = bound(
                boundary.state,
                inputs={},
                draws=controlled_uniforms(sampled, alternative=True),
            )
            assert any(
                not torch.allclose(
                    local["integrator"]["v"],
                    other["populations"][name]["integrator"]["v"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                for name, local in expected["populations"].items()
            )

    # region gradients
    native_sources = (pulse.waveform.amp, *fields.values())
    reference_sources = (
        source(boundary.parameters["populations"]["left"], ".waveform.amp"),
        *(
            source(boundary.parameters["populations"][name], ".contacts.0.field")
            for name in fields
        ),
    )
    native_loss = sum(cell.v.square().mean() for cell in model.populations.values())
    observed = torch.autograd.grad(
        native_loss, (*native_sources, *uniforms), allow_unused=True
    )
    wanted = torch.autograd.grad(state_loss(expected), reference_sources)
    for actual, reference in zip(observed[: len(native_sources)], wanted, strict=True):
        torch.testing.assert_close(actual, reference)
        assert actual.isfinite().all() and actual.abs().sum() > 0
    assert all(value is None for value in observed[len(native_sources) :])
    # endregion gradients

    # region checkpoint
    checkpoint = clone(model.state_dict_for_checkpoint())
    external = (
        None
        if extra is None
        else {
            "waveform": clone(expression.state_dict()),
            "rng": pulse.generator.get_state().clone(),
        }
    )
    current = dn.func.capture_continuation(functional, model)
    suffix = functional.sample_runtime_draws(current.state)
    reference, _ = bound(expected, inputs={}, draws=suffix)
    model.step(extra=extra)
    torch.testing.assert_close(functional.extract(model).state, reference)

    restored, restored_pulse, restored_expression, _, restored_extra = make_case(role)
    restored.restore_dict_from_checkpoint(checkpoint)
    if external is not None:
        restored_expression.load_state_dict(external["waveform"])
        restored_pulse.generator.set_state(external["rng"])
    restored.step(extra=restored_extra)
    for name, cell in model.populations.items():
        torch.testing.assert_close(restored.populations[name].v, cell.v)
    torch.testing.assert_close(
        restored_pulse.generator.get_state(), pulse.generator.get_state()
    )
    # endregion checkpoint
    print(
        f"Network {role}: shared invocation replay, gradients and checkpoint continuation passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for role in ("intra", "extra"):
            demonstrate(role)


if __name__ == "__main__":
    main()
