"""Canonical Network draw packets retain physical stream timelines."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_network import _stim_network
from tests.func.test_network_rng_sharing import PREFIX, RAW
from tests.func.test_network_rng_sharing import _network as _noise_network
from tests.func.test_stimulation_rng_sharing import (
    DT,
    STEPS,
    _clone,
    _close,
)
from tests.func.test_stimulation_rng_sharing import _environment as _environment
from tests.func.test_stimulation_rng_sharing import (
    _host_initialize,
    _lower,
    _mixed_network,
)
from tests.func.test_stimulation_rng_sharing import _network as _borrowed_network
from tests.func.test_stimulation_rng_sharing import (
    _one_stream,
    _slice,
    _step_native,
)

pytestmark = pytest.mark.cpu


def _byte_boundaries(packet):
    for frame in packet.rng["model"].values():
        for name, boundaries in frame.items():
            if name in {"seed", "negative"}:
                continue
            if boundaries and all(
                value.dtype == torch.uint8 for value in boundaries.values()
            ):
                yield boundaries


def test_runtime_network_packet_preserves_reversed_extra_intra_and_kernel_order():
    source, extra, pulses = _mixed_network()
    native, native_extra, native_pulses = _mixed_network()
    functional, tensors = _lower(source, extra)
    frozen = _clone(tensors)
    source_rng = pulses[0].generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    saved = _clone(packet)
    actual, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=STEPS, draws=packet
    )
    for _ in range(STEPS):
        _step_native(native, native_extra)
    _close(actual, _lower(native, native_extra)[1].state)
    _close(_one_stream(actual), native_pulses[0].generator.get_state())
    # A borrowed physical stream has one authoritative common timestep frame;
    # the Poisson family does not repeat those generator bytes elsewhere.
    assert not packet.rng["stimulation"]
    boundaries = tuple(_byte_boundaries(packet))
    assert boundaries and all(set(value) == {"step"} for value in boundaries)
    _close(packet, saved)
    _close(tensors, frozen)
    _close(pulses[0].generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


def test_runtime_network_ordinary_frames_keep_population_boundaries_and_continue():
    source, native = _noise_network(), _noise_network()
    functional, tensors = dn.func.make_functional_network(source)
    packet = functional.sample_runtime_draws(tensors.state, steps=6)
    boundaries = tuple(_byte_boundaries(packet))
    shared = tuple(value for value in boundaries if {"left", "right"} <= value.keys())
    assert shared
    assert any(not torch.equal(value["left"][0], value["right"][0]) for value in shared)
    bound = functional.bind(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    )
    prefix, first = dn.func.run(
        bound, tensors.state, draws=_slice(packet, 0, 3), steps=3, callbacks=callbacks
    )
    continuation = functional.sample_runtime_draws(prefix, steps=3)
    _close(continuation, _slice(packet, 3, 6))
    final, second = dn.func.run(
        bound,
        prefix,
        draws=continuation,
        steps=3,
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    traces = {name: [cell.v.clone()] for name, cell in native.populations.items()}
    for _ in range(6):
        native.step()
        for name, cell in native.populations.items():
            traces[name].append(cell.v.clone())
    _close(final, functional.extract(native).state)
    for name, expected in traces.items():
        joined = torch.cat(
            (
                first["callbacks"]["trace"][f"{name}.v"],
                second["callbacks"]["trace"][f"{name}.v"],
            )
        )
        _close(joined, torch.stack(expected))
        assert joined.shape[0] == 7
    empty = functional.sample_runtime_draws(final, steps=0)
    assert all(
        value.shape[0] == 0
        for value in pytree.tree_leaves(empty)
        if torch.is_tensor(value)
    )
    zero, _ = dn.func.run(bound, final, steps=0, draws=empty)
    _close(zero, final)


def test_runtime_network_packet_includes_netstim_draws_without_advancing_sources():
    source, native = _stim_network(), _stim_network()
    native.netstim.restore_dict_from_checkpoint(
        _clone(source.netstim.state_dict_for_checkpoint())
    )
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors)
    global_rng = torch.get_rng_state().clone()
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    saved = _clone(packet)
    assert packet.draws["netstim"] is not None
    assert packet.rng["netstim"] is not None
    chunk = functional.compile_rollout_chunk(STEPS, backend="aot_eager")
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    actual, _ = chunk(tensors.parameters, prepared, tensors.state, draws=packet)
    for _ in range(STEPS):
        native.step()
    _close(actual, functional.extract(native).state)
    _close(packet, saved)
    _close(functional.extract(), frozen)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_runtime_network_checkpoint_preserves_native_gradients_and_detached_draws(
    backend,
):
    source, native = _noise_network(), _noise_network()
    functional, tensors = dn.func.make_functional_network(source)
    sigma_name = f"{PREFIX}DE._SharedGate.sigma_param"
    names = (f"populations.left.{RAW}", f"populations.left.{sigma_name}")
    parameters = tensors.independent_parameters(trainable=names)
    targets = tuple(
        parameters["populations"]["left"][name] for name in (RAW, sigma_name)
    )
    # Build live native and functional initialization graphs before measuring
    # distribution and State parameter derivatives. Targets remain the original
    # independent leaves so the initialization lineage is part of the check.
    initialized = _host_initialize(functional, tensors._replace(parameters=parameters))
    native.initialize(functional.dt)
    _close(initialized.state, functional.extract(native).state)
    sampled = functional.sample_runtime_draws(initialized.state, steps=4)
    packet = sampled._replace(
        draws=pytree.tree_map_only(
            torch.Tensor,
            lambda value: value.detach().clone().requires_grad_(),
            sampled.draws,
        )
    )
    base = tuple(
        value
        for value in pytree.tree_leaves(packet.draws)
        if torch.is_tensor(value) and torch.is_floating_point(value)
    )
    bound = functional.compile_rollout_chunk(3, backend=backend).bind(
        initialized.parameters, initialized.constants
    )
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "right.v"))}
    )
    final, auxiliary = dn.func.run(
        bound,
        initialized.state,
        steps=4,
        checkpoint_every=2,
        draws=packet,
        callbacks=callbacks,
    )
    loss = sum(
        auxiliary["callbacks"]["trace"][f"{name}.v"].square().mean()
        for name in ("left", "right")
    )
    actual = torch.autograd.grad(loss, (*targets, *base), allow_unused=True)
    assert all(
        value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
        for value in actual[:2]
    )
    assert all(value is None or value.count_nonzero() == 0 for value in actual[2:])
    traces = {name: [cell.v.clone()] for name, cell in native.populations.items()}
    for _ in range(4):
        native.step()
        for name, cell in native.populations.items():
            traces[name].append(cell.v.clone())
    native_loss = sum(torch.stack(values).square().mean() for values in traces.values())
    expected = torch.autograd.grad(
        native_loss,
        (
            native.left.mech.probe.g_mu_param,
            native.left.mech.probe.DE._SharedGate.sigma_param,
        ),
    )
    _close(actual[:2], expected)
    _close(final, functional.extract(native).state)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_runtime_network_explicit_and_automatic_packets_resume_the_same_checkpoint(
    execution,
):
    source, _ = _borrowed_network()
    native, _ = _borrowed_network()
    inference = execution == "while"
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=inference,
        execution=execution,
        checkpoint_every=None if inference else 2,
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        backend="aot_eager",
    )
    packet = runner.functional.sample_runtime_draws(
        runner.initial_state, steps=2 * STEPS
    )
    automatic = runner()
    first, auxiliary = runner(draws=_slice(packet, 0, STEPS))
    _close(first, automatic[0])
    _close(
        dict(auxiliary["callbacks"]["trace"]), dict(automatic[1]["callbacks"]["trace"])
    )
    final, result = runner(
        first,
        draws=_slice(packet, STEPS, 2 * STEPS),
        callback_state=auxiliary["callbacks"].state,
    )
    for _ in range(2 * STEPS):
        native.step()
    _close(final, runner.functional.extract(native).state)
    committed, _ = _borrowed_network()
    runner.functional.commit_state_(committed, final)
    checkpoint = _clone(committed.state_dict_for_checkpoint())
    restored, _ = _borrowed_network()
    restored.restore_dict_from_checkpoint(checkpoint)
    _close(runner.functional.extract(restored).state, final)
    for _ in range(STEPS):
        restored.step()
    continued, _ = runner(final, callback_state=result["callbacks"].state)
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("legacy", ("population", "netstim"))
def test_runtime_network_rejects_mixed_canonical_and_legacy_draw_sources(legacy):
    source = _noise_network() if legacy == "population" else _stim_network()
    functional, tensors = dn.func.make_functional_network(source)
    packet = functional.sample_runtime_draws(tensors.state, steps=3)
    if legacy == "population":
        old = functional.sample_population_noise_tapes(tensors.state, 3)
        inputs = dn.func.NetworkRolloutInput(
            populations={"left": dn.func.RolloutInput(noise=old["left"])}
        )
    else:
        inputs = dn.func.NetworkRolloutInput(
            netstim=functional.sample_draw_tape(tensors.state, 3)
        )
    bound = functional.bind(tensors.parameters, tensors.constants)
    with pytest.raises(ValueError, match="draws|legacy|combine|mix"):
        dn.func.run(bound, tensors.state, inputs=inputs, draws=packet, steps=3)
