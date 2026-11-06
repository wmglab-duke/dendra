#!/usr/bin/env python3
"""Continue a functional prefix with native model/extra source sharing.

Run from the repository root::

    python examples/continuation/shared_handoff_staging.py

Population and Network cases share an amplitude and the global Poisson
generator between registered intracellular and caller-owned extracellular
waveforms. The native suffix evaluates extra before intra and resumes from
the functional boundary without resetting ambient randomness during handoff.
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas
from dendra.units import nA

DT, PREFIX = 0.025, 3


def clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def make_case(network):
    generator = torch.default_generator
    intra = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    external = dn.mono_rect(amp=0.2 * nA, pw=0.3).poisson(
        interval=0.08,
        n=5,
        start=-0.02,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    # This raw Parameter has a native owner as well as an external alias.
    external.waveform.amp = intra.waveform.amp
    populations = {}
    for name in ("left", "right") if network else ("cell",):
        population = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(pas, g=0.001, e=-65.0)
        population[:, 1].inject(intra + intra)
        populations[name] = population
    model = dn.Network(populations) if network else populations["cell"]
    model.train()
    model.initialize(DT) if network else model.initialize()
    target = populations["right"] if network else model
    field = torch.linspace(-1.25e6, 1.75e6, target.v.numel(), dtype=target.dtype())
    field = field.reshape(target.shape).requires_grad_()
    extra = {"right": (field, external)} if network else (field, external)
    return model, intra, external, field, extra


def amplitude(parameters, network):
    frame = parameters["populations"]["left"] if network else parameters
    return next(
        value for name, value in frame.items() if name.endswith(".waveform.amp")
    )


def advance(model, network, extra):
    return model.step(extra=extra) if network else model.step(DT, extra=extra)


def pure_step(bound, state, packet, network):
    return bound(state, inputs={} if network else None, draws=packet)[0]


def demonstrate(network):
    torch.manual_seed(17)
    model, intra, external, field, extra = make_case(network)
    # region prefix
    plan, _ = dn.func.make_functional(model, dt=DT, extra=extra)
    boundary = dn.func.capture_continuation(plan)
    trained_amplitude = amplitude(boundary.parameters, network)
    with torch.no_grad():
        trained_amplitude.mul_(1.1)
    bound = plan.bind(boundary)
    state = boundary.state
    for _ in range(PREFIX):
        packet = plan.sample_runtime_draws(state)
        state = pure_step(bound, state, packet, network)
    continuation = dn.func.Continuation(state, boundary.parameters, boundary.constants)
    # A detached suffix reference has the same recurrent-graph boundary as handoff.
    continuation = dn.func.detach_continuation(plan, continuation)
    bound = plan.bind(continuation)
    # endregion prefix

    # region handoff
    receiving_amplitude = intra.waveform.amp
    external_schedule = external._spike_times.detach().clone()
    torch.manual_seed(9931)
    ambient = torch.get_rng_state().clone()
    dn.func.commit_continuation_(plan, model, continuation)
    assert intra.waveform.amp is external.waveform.amp is receiving_amplitude
    assert receiving_amplitude.requires_grad
    torch.testing.assert_close(receiving_amplitude, trained_amplitude)
    torch.testing.assert_close(external._spike_times, external_schedule)
    torch.testing.assert_close(torch.get_rng_state(), ambient)
    # endregion handoff

    # region automatic
    # Extra is supplied again. It consumes the shared stream before intra.
    following = plan.sample_runtime_draws(continuation.state)
    expected = pure_step(bound, continuation.state, following, network)
    advance(model, network, extra)
    torch.testing.assert_close(plan.extract(model).state, expected)
    # endregion automatic

    # region checkpoint
    checkpoint = clone(model.state_dict_for_checkpoint())
    # External-only history stays caller-owned, even when its generator is shared.
    external_checkpoint = {
        "waveform": clone(external.state_dict()),
        "rng": external.generator.get_state().clone(),
    }
    following = plan.sample_runtime_draws(expected)
    final = pure_step(bound, expected, following, network)
    advance(model, network, extra)
    torch.testing.assert_close(plan.extract(model).state, final)

    restored, restored_intra, restored_external, _, restored_extra = make_case(network)
    restored.restore_dict_from_checkpoint(checkpoint)
    restored_external.load_state_dict(external_checkpoint["waveform"])
    restored_external.generator.set_state(external_checkpoint["rng"])
    advance(restored, network, restored_extra)
    restored_cells = restored.populations.values() if network else (restored,)
    live_cells = model.populations.values() if network else (model,)
    for actual, expected_cell in zip(restored_cells, live_cells, strict=True):
        torch.testing.assert_close(actual.v, expected_cell.v)
    assert restored_intra.generator is restored_external.generator
    torch.testing.assert_close(
        restored_external.generator.get_state(), external.generator.get_state()
    )
    # endregion checkpoint

    cells = model.populations.values() if network else (model,)
    loss = sum(cell.v.square().mean() for cell in cells)
    gradients = torch.autograd.grad(loss, (receiving_amplitude, field))
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in gradients)
    print(
        f"{'Network' if network else 'Population'}: shared-source handoff, "
        "extra-first RNG continuation, native gradients and checkpoint restoration passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            demonstrate(network)


if __name__ == "__main__":
    main()
