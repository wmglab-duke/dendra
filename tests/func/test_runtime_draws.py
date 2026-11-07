"""Canonical runtime packets preserve native stochastic execution."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_poisson_stimulation import DT, STEPS, _clone, _close
from tests.func.test_stimulation_rng_sharing import (
    _draw_input,
)
from tests.func.test_stimulation_rng_sharing import _environment as _environment
from tests.func.test_stimulation_rng_sharing import (
    _functional_gradient_leaves,
    _gradient_leaves,
    _host_initialize,
    _initialize_native,
    _lower,
    _one_stream,
    _population,
    _slice,
    _step_native,
    _TransitionGuardLeak,
)

pytestmark = pytest.mark.cpu


def _floating_draws(packet):
    return tuple(
        value
        for value in pytree.tree_leaves(packet.draws)
        if torch.is_tensor(value) and torch.is_floating_point(value)
    )


def test_runtime_draws_export_preserves_existing_packet_tuple_layouts():
    assert dn.func.RuntimeDraws._fields == ("draws", "rng")
    assert dn.func.StepInput._fields == ("ve", "intra", "noise", "stimulation")
    assert dn.func.RolloutInput._fields == ("ve", "intra", "noise", "stimulation")
    assert dn.func.NetworkRolloutInput._fields == ("populations", "netstim")


@pytest.mark.parametrize("role", ("mechanism", "extra"))
@pytest.mark.parametrize("tape", (False, True))
def test_runtime_population_packets_match_native_shared_draw_order(role, tape):
    source, extra, pulse = _population(role, repeated=True)
    native, native_extra, native_pulse = _population(role, repeated=True)
    functional, tensors = _lower(source, extra)
    original = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    packet = functional.sample_runtime_draws(
        tensors.state, steps=STEPS if tape else None
    )
    assert isinstance(packet, dn.func.RuntimeDraws)
    frozen = _clone(packet)
    _close(
        packet,
        functional.sample_runtime_draws(tensors.state, steps=STEPS if tape else None),
    )
    if tape:
        actual, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=STEPS, draws=packet
        )
        for _ in range(STEPS):
            _step_native(native, native_extra)
    else:
        actual, _ = functional.step(
            tensors.parameters, prepared, tensors.state, draws=packet
        )
        _step_native(native, native_extra)
    _close(actual, _lower(native, native_extra)[1].state)
    _close(_one_stream(actual), native_pulse.generator.get_state())
    _close(packet, frozen)
    _close(tensors, original)
    _close(functional.extract(), original)
    _close(pulse.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


def test_runtime_population_tape_slices_continue_and_zero_steps_are_empty():
    model, extra, _ = _population()
    functional, tensors = _lower(model, extra)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    tape = functional.sample_runtime_draws(tensors.state, steps=2 * STEPS)
    whole, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, draws=tape, steps=2 * STEPS
    )
    prefix, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        draws=_slice(tape, 0, STEPS),
        steps=STEPS,
    )
    continuation = functional.sample_runtime_draws(prefix, steps=STEPS)
    _close(continuation, _slice(tape, STEPS, 2 * STEPS))
    resumed, _ = functional.rollout(
        tensors.parameters, prepared, prefix, draws=continuation, steps=STEPS
    )
    _close(resumed, whole)
    empty = functional.sample_runtime_draws(prefix, steps=0)
    assert all(
        value.shape[0] == 0
        for value in pytree.tree_leaves(empty)
        if torch.is_tensor(value)
    )
    zero, _ = functional.rollout(
        tensors.parameters, prepared, prefix, draws=empty, steps=0
    )
    _close(zero, prefix)


def test_runtime_population_draws_detach_base_values_and_preserve_native_gradients():
    model, _, _ = _population()
    native, _, _ = _population()
    functional, tensors = _lower(model)
    # The fixture deliberately detaches its insertion-time coefficient frame.
    # Reinitialize both executions before measuring RAND and State parameter
    # gradients, as in the independent native fresh-initialization oracle.
    initialized = _host_initialize(functional, tensors)
    _initialize_native(native)
    _close(initialized.state, functional.extract(native).state)
    sampled = functional.sample_runtime_draws(initialized.state, steps=STEPS)
    packet = sampled._replace(
        draws=pytree.tree_map_only(
            torch.Tensor,
            lambda value: value.detach().clone().requires_grad_(),
            sampled.draws,
        )
    )
    actual, _ = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        steps=STEPS,
        draws=packet,
    )
    sigma = next(
        value
        for name, value in tensors.parameters.items()
        if name.endswith("DE._SharedStochasticGate.sigma_param")
    )
    targets = (*_functional_gradient_leaves(tensors), sigma)
    base = _floating_draws(packet)
    gradients = torch.autograd.grad(
        actual["integrator"]["v"].square().mean(), (*targets, *base), allow_unused=True
    )
    for _ in range(STEPS):
        native.step(DT)
    _close(actual, functional.extract(native).state)
    _close(_one_stream(actual), native.mech.probe.stream.rng.get_state())
    expected = torch.autograd.grad(
        native.v.square().mean(),
        (
            *_gradient_leaves(native),
            native.mech.probe.DE._SharedStochasticGate.sigma_param,
        ),
    )
    _close(gradients[: len(targets)], expected)
    assert all(
        value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
        for value in gradients[: len(targets)]
    )
    assert all(
        value is None or value.count_nonzero() == 0
        for value in gradients[len(targets) :]
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_runtime_population_explicit_automatic_checkpoint_and_recording_continuation(
    execution,
    backend,
):
    model, _, pulse = _population()
    inference = execution == "while"
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=inference,
        backend=backend,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        checkpoint_every=None if inference else 2,
    )
    source_rng = pulse.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    tape = runner.functional.sample_runtime_draws(runner.initial_state, steps=2 * STEPS)
    automatic = runner()
    explicit = runner(draws=_slice(tape, 0, STEPS))
    _close(explicit[0], automatic[0])
    _close(
        dict(explicit[1]["callbacks"]["trace"]),
        dict(automatic[1]["callbacks"]["trace"]),
    )
    resumed, auxiliary = runner(
        explicit[0],
        draws=_slice(tape, STEPS, 2 * STEPS),
        callback_state=explicit[1]["callbacks"].state,
    )
    legacy_inputs = _draw_input(runner.functional, runner.initial_state, 2 * STEPS)
    prepared = runner.functional.prepare(runner.parameters, runner.constants)
    with torch.no_grad() if inference else torch.enable_grad():
        whole, whole_aux = runner.functional.rollout(
            runner.parameters,
            prepared,
            runner.initial_state,
            legacy_inputs,
            steps=2 * STEPS,
            callbacks=runner.functional.make_callbacks(
                {"trace": dn.func.Recorder(("v", "t"))}
            ),
        )
    _close(resumed, whole)
    for name, values in whole_aux["callbacks"]["trace"].items():
        joined = torch.cat(
            (
                explicit[1]["callbacks"]["trace"][name],
                auxiliary["callbacks"]["trace"][name],
            )
        )
        _close(joined, values)
        assert joined.shape[0] == 2 * STEPS + 1
    _close(pulse.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("legacy", ("noise", "stimulation"))
def test_runtime_population_rejects_mixed_packet_sources_before_transition(
    monkeypatch, legacy
):
    model, _, _ = _population(transition_guard=True)
    functional, tensors = _lower(model)
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    old = _draw_input(functional, tensors.state, STEPS)
    inputs = dn.func.RolloutInput(**{legacy: getattr(old, legacy)})
    monkeypatch.setattr(_TransitionGuardLeak, "FAIL_ON_ADVANCE", True)
    with pytest.raises(ValueError, match="draws|legacy|combine|mix"):
        functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs,
            steps=STEPS,
            draws=packet,
        )


def test_runtime_population_rejects_malformed_rng_schema_before_transition(monkeypatch):
    model, _, _ = _population(transition_guard=True)
    functional, tensors = _lower(model)
    packet = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    malformed = packet._replace(rng={})
    monkeypatch.setattr(_TransitionGuardLeak, "FAIL_ON_ADVANCE", True)
    with pytest.raises((KeyError, ValueError, TypeError)):
        functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=STEPS,
            draws=malformed,
        )
