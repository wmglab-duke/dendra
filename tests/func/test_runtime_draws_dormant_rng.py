"""Initialization-only RNG carry does not become a runtime timeline."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.units import nA
from tests.func.test_mechanism_injection_execution import _InjectedLeak
from tests.func.test_poisson_stimulation import DT, DTYPE, STEPS, _clone, _close
from tests.func.test_stimulation_rng_sharing import _environment as _environment
from tests.func.test_stimulation_rng_sharing import _RandOnlyLeak, _slice

pytestmark = pytest.mark.cpu


def _population(mechanism=_RandOnlyLeak, *, detach=True):
    cell = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-45.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    cell.insert(mechanism.rename("probe"))
    cell.train()
    cell.initialize()
    cell.integrator._initialize(cell, cell.t.new_tensor(DT), force=True)
    if detach:
        cell.detach()
    return cell


def _network(kind, *, detach=True):
    donor = _population()
    populations = {"donor": donor}
    pulse = None
    if kind != "netstim":
        if kind == "poisson_same":
            receiver = donor
        else:
            receiver = populations["receiver"] = _population(_InjectedLeak)
        pulse = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=(
                donor.mech.probe.offset_rng.rng
                if kind == "borrowed_poisson"
                else torch.Generator().manual_seed(929)
            ),
        )
        receiver[:, 1].inject(pulse)
    stimulus = (
        dn.NetStim(N=1, interval=0.055, start=0.0, noise=0.6, max_spikes=20, seed=812)
        if kind == "netstim"
        else None
    )
    network = dn.Network(populations, netstim=stimulus)
    network.train()
    network.initialize(DT)
    if detach:
        for cell in network.populations.values():
            cell.detach()
    return network, pulse


def test_rand_only_population_keeps_rng_carry_without_a_runtime_timeline():
    source, native = _population(), _population(detach=False)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    global_rng = torch.get_rng_state().clone()
    assert tensors.state["random"]["rng"]
    packet = functional.sample_runtime_draws(tensors.state, steps=2 * STEPS)
    assert packet.draws == {"noise": None, "stimulation": None}
    assert packet.rng == {"model": {}, "stimulation": {}}
    assert not any(torch.is_tensor(value) for value in pytree.tree_leaves(packet))
    assert functional.sample_runtime_draws(tensors.state).rng["model"] == {}
    assert functional.sample_runtime_draws(tensors.state, steps=0).rng["model"] == {}
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    bound = functional.compile_rollout_chunk(3, backend="aot_eager").bind(
        tensors.parameters, prepared
    )
    first, _ = dn.func.run(
        bound, tensors.state, draws=packet, steps=STEPS, checkpoint_every=2
    )
    for _ in range(STEPS):
        native.step(DT)
    _close(first, functional.extract(native).state)
    checkpoint = _clone(native.state_dict_for_checkpoint())
    restored = _population()
    restored.restore_dict_from_checkpoint(checkpoint)
    final, _ = dn.func.run(
        bound,
        first,
        draws=functional.sample_runtime_draws(first, steps=STEPS),
        steps=STEPS,
        checkpoint_every=2,
    )
    for _ in range(STEPS):
        native.step(DT)
        restored.step(DT)
    _close(final, functional.extract(native).state)
    _close(final, functional.extract(restored).state)
    _close(final["random"]["rng"], tensors.state["random"]["rng"])
    name = tensors.parameter_name("g_param")
    (gradient,) = torch.autograd.grad(
        final["integrator"]["v"].square().mean(), tensors.parameters[name]
    )
    (expected,) = torch.autograd.grad(
        native.v.square().mean(), native.mech.probe.g_param
    )
    _close(gradient, expected)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    _close(functional.extract(), frozen)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("kind", ("netstim", "poisson_same", "borrowed_poisson"))
def test_network_runtime_rng_carry_is_independent_of_its_input_tape(kind):
    source, _ = _network(kind)
    functional, tensors = dn.func.make_functional_network(source)
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    with torch.no_grad():
        final, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=STEPS,
            draws=packet,
        )
    saved = _clone(final)
    for value in pytree.tree_leaves(packet.rng):
        if torch.is_tensor(value):
            value.zero_()
    _close(final, saved)


@pytest.mark.parametrize(
    "kind", ("netstim", "poisson_same", "poisson_peer", "borrowed_poisson")
)
def test_network_runtime_packets_include_only_consumed_model_streams(kind):
    source, pulse = _network(kind)
    native, native_pulse = _network(kind, detach=False)
    if kind == "netstim":
        native.netstim.restore_dict_from_checkpoint(
            _clone(source.netstim.state_dict_for_checkpoint())
        )
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors)
    global_rng = torch.get_rng_state().clone()
    packet = functional.sample_runtime_draws(tensors.state, steps=2 * STEPS)
    assert packet.draws["populations"]["donor"]["noise"] is None
    if kind == "borrowed_poisson":
        assert len(packet.rng["model"]) == 1
        assert not packet.rng["stimulation"]
        frame = next(iter(packet.rng["model"].values()))
        assert set(frame["cpu"]) == {"step"}
        assert packet.draws["populations"]["donor"]["stimulation"] is None
        assert packet.draws["populations"]["receiver"]["stimulation"] is not None
    else:
        assert not packet.rng["model"]
        assert tensors.state["populations"]["donor"]["random"]["rng"]
        assert len(packet.rng["stimulation"]) == (0 if kind == "netstim" else 1)
    assert (packet.draws["netstim"] is not None) == (kind == "netstim")
    bound = functional.bind(tensors.parameters, tensors.constants)
    first, _ = dn.func.run(
        bound,
        tensors.state,
        draws=_slice(packet, 0, STEPS),
        steps=STEPS,
        checkpoint_every=2,
    )
    for _ in range(STEPS):
        native.step()
    _close(first, functional.extract(native).state)
    continuation = functional.sample_runtime_draws(first, steps=STEPS)
    _close(continuation, _slice(packet, STEPS, 2 * STEPS))
    checkpoint = _clone(native.state_dict_for_checkpoint())
    restored, _ = _network(kind)
    restored.restore_dict_from_checkpoint(checkpoint)
    final, _ = dn.func.run(
        bound, first, draws=continuation, steps=STEPS, checkpoint_every=2
    )
    for _ in range(STEPS):
        native.step()
        restored.step()
    _close(final, functional.extract(native).state)
    _close(final, functional.extract(restored).state)
    if kind == "borrowed_poisson":
        actual_rng = next(iter(final["rng"].values()))["cpu"]
        _close(actual_rng, native_pulse.generator.get_state())
        assert pulse.generator is source.donor.mech.probe.offset_rng.rng
        assert not torch.equal(
            actual_rng, next(iter(tensors.state["rng"].values()))["cpu"]
        )
    else:
        _close(
            final["populations"]["donor"]["random"]["rng"],
            tensors.state["populations"]["donor"]["random"]["rng"],
        )
    donor_parameters = tensors.parameters["populations"]["donor"]
    target = next(
        value for name, value in donor_parameters.items() if name.endswith(".g_param")
    )
    (gradient,) = torch.autograd.grad(
        final["populations"]["donor"]["integrator"]["v"].square().mean(), target
    )
    (expected,) = torch.autograd.grad(
        native.donor.v.square().mean(), native.donor.mech.probe.g_param
    )
    _close(gradient, expected)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    _close(functional.extract(), frozen)
    _close(torch.get_rng_state(), global_rng)
