"""One eager Poisson frame supplies bound drives, hook mappings and publication."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.stim.waveform._poisson import POISSON_INVOCATION_POSITION
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_poisson_invocation_execution import (
    _case,
    _native_state,
    _TwiceInjectedLeak,
)
from tests.func.test_poisson_stimulation import DT, STEPS, _clone, _close, _waveform
from tests.func.test_runtime_draw_shapes import _plan, _RepeatedRuntime
from tests.test_native_runtime_draws import _environment as _environment

pytestmark = pytest.mark.cpu


def _build(network, *, role="mechanism"):
    source, extra, pulse, generator = _case(
        role=role, mechanism=_TwiceInjectedLeak, n=4
    )
    assert extra is None
    if network:
        source = dn.Network({"cell": source})
        source.train()
        source.initialize(DT)
        source.populations["cell"].detach()
        functional, tensors = dn.func.make_functional_network(source)
        local = functional.populations["cell"]
    else:
        functional, tensors = dn.func.make_functional(source, dt=DT)
        local = functional
    return source, pulse, generator, functional, local, tensors


def _callbacks(functional, network, enabled):
    if not enabled:
        return None
    return functional.make_callbacks(
        {"trace": dn.func.Recorder(("cell.v", "t") if network else ("v", "t"))}
    )


def _packet(functional, state, network, canonical, *, steps):
    if canonical:
        return None, dn.func.sample_runtime_draws(functional, state, steps=steps)
    if network:
        tapes = functional.sample_population_stimulation_tapes(state, steps)
        return dn.func.NetworkRolloutInput(
            {
                name: dn.func.RolloutInput(stimulation=tape)
                for name, tape in tapes.items()
            }
        ), None
    return dn.func.RolloutInput(
        stimulation=functional.sample_stimulation_draw_tape(state, steps)
    ), None


def _count_schedules(monkeypatch, plan):
    calls = []
    original = plan.schedules

    def observed(packet, *, initialization=False):
        calls.append(initialization)
        return original(packet, initialization=initialization)

    monkeypatch.setattr(plan, "schedules", observed)
    return calls


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("canonical", (False, True), ids=("legacy", "canonical"))
@pytest.mark.parametrize("callbacks", (False, True), ids=("plain", "callbacks"))
def test_each_accepted_eager_step_evaluates_one_schedule_frame(
    monkeypatch, network, canonical, callbacks
):
    source, pulse, generator, functional, local, tensors = _build(network)
    native, native_pulse, native_generator, _, _, _ = _build(network)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    collection = _callbacks(functional, network, callbacks)
    inputs, draws = _packet(functional, tensors.state, network, canonical, steps=STEPS)
    # Explicit uniforms remain detached even when the caller requests gradients.
    inputs, draws = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.detach().clone().requires_grad_(value.is_floating_point()),
        (inputs, draws),
    )
    frozen_inputs = _clone((inputs, draws))
    frozen = _clone(tensors)
    source_rng, global_rng = (
        generator.get_state().clone(),
        torch.get_rng_state().clone(),
    )
    calls = _count_schedules(monkeypatch, local._poisson_plan)
    actual, auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
        steps=STEPS,
        draws=draws,
        callbacks=collection,
    )
    assert calls == [False] * STEPS
    for _ in range(STEPS):
        native.step() if network else native.step(DT)
    _close(actual, _native_state(native))
    local_state = (
        functional._expand_population_state(
            "cell", actual["populations"]["cell"], actual
        )
        if network
        else actual
    )
    local_parameters = (
        tensors.parameters["populations"]["cell"] if network else tensors.parameters
    )
    amplitudes = {
        id(value): value
        for name, value in local_parameters.items()
        if name.endswith("waveform.amp")
    }
    assert len(amplitudes) == 1
    draw_leaves = tuple(
        value
        for value in pytree.tree_leaves((inputs, draws))
        if torch.is_tensor(value) and value.requires_grad
    )
    gradients = torch.autograd.grad(
        local_state["integrator"]["v"].square().mean(),
        (*amplitudes.values(), *draw_leaves),
        allow_unused=True,
    )
    native_voltage = native.populations["cell"].v if network else native.v
    wanted = torch.autograd.grad(
        native_voltage.square().mean(), native_pulse.waveform.amp
    )[0]
    assert wanted.abs().sum() > 0
    _close(gradients[0], wanted)
    assert all(
        value is None or torch.count_nonzero(value) == 0 for value in gradients[1:]
    )
    assert all(
        not value.requires_grad
        for value in local_state["stimulation"]["schedules"].values()
    )
    if callbacks:
        trace = auxiliary["callbacks"]["trace"]
        assert trace["t"].shape[0] == STEPS + 1
        _close(trace["cell.v" if network else "v"][-1], native_voltage)
    # Sampling and execution publish explicit state, never source RNG or inputs.
    _close((inputs, draws), frozen_inputs)
    _close(tensors, frozen)
    _close(generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)
    _close(
        next(iter(local_state["stimulation"]["rng"].values())),
        native_generator.get_state(),
    )
    assert pulse.n == native_pulse.n == 4


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("callbacks", (False, True), ids=("plain", "callbacks"))
def test_zero_step_rollout_does_not_evaluate_poisson_schedules(
    monkeypatch, network, callbacks
):
    _, _, generator, functional, local, tensors = _build(network)
    inputs, draws = _packet(functional, tensors.state, network, True, steps=0)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    collection = _callbacks(functional, network, callbacks)
    before_rng = generator.get_state().clone()
    calls = _count_schedules(monkeypatch, local._poisson_plan)
    state, auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
        steps=0,
        draws=draws,
        callbacks=collection,
    )
    assert calls == []
    _close(state, tensors.state)
    _close(generator.get_state(), before_rng)
    if callbacks:
        assert auxiliary["callbacks"]["trace"]["t"].shape[0] == 1


def test_parameter_adapter_and_shared_frame_preserve_bound_source_aliases(monkeypatch):
    _, _, _, functional, local, tensors = _build(False, role="intra")
    plan = local._poisson_plan
    packet = functional.sample_stimulation_draw(tensors.state)
    expected = plan.parameters(tensors.parameters, packet)
    calls = _count_schedules(monkeypatch, plan)
    bound, schedules = plan._parameters_and_schedules(tensors.parameters, packet)
    assert calls == [False]
    assert bound is not tensors.parameters
    _close(bound, expected)
    for name, value in tensors.parameters.items():
        if name not in {entry.parameter for entry in plan.entries}:
            assert bound[name] is value
    for entry in plan.entries:
        stack = schedules[entry.key]
        _close(
            bound[entry.parameter],
            stack[0].reshape(
                stack[0].shape + (1,) * entry.source._spike_time_trailing_dims
            ),
        )
        assert (
            bound[entry.parameter + ".__invocations"].untyped_storage().data_ptr()
            == stack.untyped_storage().data_ptr()
        )
    first = plan.advance(tensors.state["stimulation"], packet, schedules=schedules)
    second = plan.advance(tensors.state["stimulation"], packet)
    _close(first, second)
    assert calls == [False, False]


def test_shared_frame_gives_fresh_cursors_and_publishes_last_consumed_alias(
    monkeypatch,
):
    pulse = _waveform(n=4)
    _, _, functional, tensors = _plan(
        waveform=pulse + pulse, mechanism=_RepeatedRuntime
    )
    plan = functional._poisson_plan
    packet = functional.sample_stimulation_draw(tensors.state)
    assert any(
        plan._capacities[entry.key] > plan._runtime_counts[entry.key]
        for entry in plan.entries
    )
    draws = {
        entry.key: torch.full_like(packet.draws[entry.key], 0.2 + 0.1 * index)
        for index, entry in enumerate(plan.entries)
    }
    for entry in plan.entries:
        draws[entry.key][plan._runtime_counts[entry.key] :] = 0.9
    packet = packet._replace(draws=draws)
    _, schedules = plan._parameters_and_schedules(tensors.parameters, packet)

    def repeated_evaluation(*args, **kwargs):
        raise AssertionError("supplied schedules must be reused")

    canonical = dn.func.sample_runtime_draws(functional, tensors.state)
    canonical_stimulation = functional._runtime_draw_plan.arithmetic(canonical)[1]
    canonical_schedules = plan.schedules(canonical_stimulation)
    monkeypatch.setattr(plan, "schedules", repeated_evaluation)
    first = plan.mapping(packet, schedules=schedules)
    second = plan.mapping(packet, schedules=schedules)
    cursors = [name for name in first if name.endswith(POISSON_INVOCATION_POSITION)]
    assert cursors
    for name in cursors:
        assert first[name] is not second[name]
        assert (
            first[name].untyped_storage().data_ptr()
            != second[name].untyped_storage().data_ptr()
        )
        first[name].add_(1)
        assert torch.count_nonzero(second[name]) == 0
    advanced = plan.advance(tensors.state["stimulation"], packet, schedules=schedules)
    first_entry, last_entry = plan.entries
    assert first_entry.source is last_entry.source
    # Each of the two assigned-value calls evaluates both authored children.
    assert (
        tuple(entry.key for entry in plan._runtime_entries)
        == (
            first_entry.key,
            last_entry.key,
        )
        * 2
    )
    expected = schedules[last_entry.key][1]
    for entry in plan.entries:
        _close(advanced["schedules"][entry.key], expected)
        assert not torch.equal(
            advanced["schedules"][entry.key], schedules[entry.key][-1]
        )
    # Canonical packets have their own runtime layout and share its evaluated frame.
    schedules = canonical_schedules
    published = functional._runtime_draw_plan.publish(
        tensors.state, canonical, schedules=schedules
    )
    expected = plan.advance(
        tensors.state["stimulation"],
        canonical_stimulation,
        schedules=schedules,
        publish_rng=False,
    )
    _close(published["stimulation"]["schedules"], expected["schedules"])


def test_parameter_copy_failure_precedes_schedule_evaluation(monkeypatch):
    _, _, _, functional, local, tensors = _build(False, role="intra")
    plan = local._poisson_plan
    calls = _count_schedules(monkeypatch, plan)

    class CopyFailure(RuntimeError):
        pass

    class FailingParameters(Mapping):
        def __iter__(self):
            return iter(("broken",))

        def __len__(self):
            return 1

        def __getitem__(self, key):
            raise CopyFailure(key)

    for adapter in (plan.parameters, plan._parameters_and_schedules):
        with pytest.raises(CopyFailure, match="broken"):
            adapter(FailingParameters(), object())
    assert calls == []
    assert tensors.parameters


@pytest.mark.parametrize("canonical", (False, True), ids=("legacy", "canonical"))
def test_bound_drive_conflict_precedes_bad_packet_validation_without_schedule_evaluation(
    monkeypatch, canonical
):
    source, _, generator, functional, local, tensors = _build(False, role="intra")
    packet = (
        dn.func.sample_runtime_draws(functional, tensors.state)
        if canonical
        else functional.sample_stimulation_draw(tensors.state)
    )
    family = packet.draws["stimulation"] if canonical else packet.draws
    key = next(iter(family))
    family[key] = family[key][..., :-1]
    before_rng = generator.get_state().clone()
    before = _clone(source.state_dict_for_checkpoint())
    calls = _count_schedules(monkeypatch, local._poisson_plan)
    inputs = dn.func.StepInput(intra=torch.zeros_like(source.v))
    if not canonical:
        inputs = inputs._replace(stimulation=packet)
    with pytest.raises(
        ValueError,
        match="^explicit intra cannot be combined with registered functional intra$",
    ):
        functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs,
            **({"draws": packet} if canonical else {}),
        )
    assert calls == []
    _close(generator.get_state(), before_rng)
    _assert_packet_equal(source.state_dict_for_checkpoint(), before)


def test_private_schedule_lookup_precedes_bound_drive_conflict(monkeypatch):
    source, _, generator, functional, local, tensors = _build(False, role="intra")
    prepared = functional._prepare_values(tensors.parameters, tensors.constants)
    original = functional.sample_stimulation_draw(tensors.state)
    missing = next(iter(original.draws))
    packet = original._replace(
        draws={name: value for name, value in original.draws.items() if name != missing}
    )
    tape = pytree.tree_map_only(torch.Tensor, lambda value: value.unsqueeze(0), packet)
    drive = torch.zeros_like(source.v).unsqueeze(0)
    before_drive, version = drive.clone(), drive._version
    before_source = _clone(source.state_dict_for_checkpoint())
    before_tensors = _clone(tensors)
    before_rng = generator.get_state().clone()
    calls = _count_schedules(monkeypatch, local._poisson_plan)
    # This unchecked boundary preserves its established schedule-first order;
    # public execution rejects the conflicting drive before reaching it.
    with pytest.raises(KeyError) as error:
        functional._execute_rollout_values(
            tensors.parameters,
            prepared,
            tensors.state,
            None,
            drive,
            1,
            stimulation=tape,
        )
    assert error.value.args == (missing,)
    assert calls == [False]
    assert drive._version == version
    _close(drive, before_drive)
    _close(tensors, before_tensors)
    _close(generator.get_state(), before_rng)
    _assert_packet_equal(source.state_dict_for_checkpoint(), before_source)
