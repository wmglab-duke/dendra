"""Functional heap calendars share native consumption, renewal, and checkpoints."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

pytestmark = pytest.mark.cpu
HEAP_CONSTANTS = ("heap_times", "heap_row_offsets")
HEAP_CARRY = ("heap_cursor", "heap_head_override", "heap_override_active")
CARRY = (
    "next_stoch_time",
    "spike_counts",
    "t_last",
    "rng_state",
    "seeder_state",
    *HEAP_CARRY,
)


def _source(*, dtype=torch.float64, batch=None, mixed=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        source = dn.NetStim(
            N=3,
            interval=(0.3, 0.4, 0.5),
            noise=(0.0, 0.4, 0.8),
            start=(-0.1, 0.1, 0.4),
            max_spikes=(3, 5, 2),
            tau=0.17,
            seed=71,
            dtype=dtype,
        )
        if batch is not None:
            source.batch(batch)
        source.set_dt(0.1)
        source.initialize()
        source.schedule([0, 0, 0, 1, 1, 2], [0.05, 0.05, 0.3, 0.15, 0.5, 1.7])
        if mixed:
            source.schedule(
                [0, 1],
                torch.tensor([0.2, 0.45], dtype=dtype, requires_grad=True),
                weight=torch.tensor([-0.3, 0.7], dtype=dtype, requires_grad=True),
            )
    return source


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    left, left_spec = pytree.tree_flatten(actual)
    right, right_spec = pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, equal_nan=True)
        else:
            assert value == reference


def _carry(source):
    checkpoint = source.state_dict_for_checkpoint()
    return {name: checkpoint[name] for name in CARRY}


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("bptt", (False, True))
def test_mixed_heap_tensor_renewal_parity_at_repeated_skipped_and_decreasing_times(
    dtype, batch, bptt
):
    source = _source(dtype=dtype, batch=batch)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    original = _clone(tensors)
    for sampled in (0.0, 0.1, 0.1, 0.7, 0.2, 1.5, 2.0):
        time = torch.tensor(sampled, dtype=dtype)
        if batch is not None:
            time = torch.stack((time, time + 0.03))
        draw = plan.sample_step_draw(state)
        state, output = plan.step(
            tensors.parameters,
            tensors.constants,
            state,
            time,
            draw,
            bptt=bptt,
            schedule=tensors.schedule,
        )
        source(time, bptt=bptt)
        _close(state, _carry(source))
        torch.testing.assert_close(output.spikes, source.spikes)
        torch.testing.assert_close(output.spike_gate, source.spike_gate)
    _close(tensors, original)
    assert tensors.constants["heap_times"].dtype == torch.float64
    assert not tensors.constants["heap_times"].requires_grad
    assert torch.all(state["heap_cursor"] >= original.state["heap_cursor"])


@pytest.mark.parametrize("bptt", (False, True))
def test_heap_head_minimum_preserves_native_parameter_gradients(bptt):
    source = _source(mixed=False)
    source.noise.requires_grad_()
    source.tau = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    loss, native_loss = source.t_last.new_zeros(()), source.t_last.new_zeros(())
    for index, time in enumerate((0.0, 0.07, 0.35, 0.7, 1.0)):
        draw = plan.sample_step_draw(state)
        state, output = plan.step(
            tensors.parameters, tensors.constants, state, time, draw, bptt=bptt
        )
        source(time, bptt=bptt)
        loss = loss + (index + 1) * output.spike_gate.sum()
        native_loss = native_loss + (index + 1) * source.spike_gate.sum()
    if bptt:
        loss = loss + state["next_stoch_time"].sum()
        native_loss = native_loss + source.next_stoch_time.sum()
    actual = torch.autograd.grad(
        loss,
        tuple(tensors.parameters[name] for name in ("interval", "noise", "tau")),
        allow_unused=True,
    )
    expected = torch.autograd.grad(
        native_loss, (source.interval.rho, source.noise, source.tau), allow_unused=True
    )
    for index in (0, 1):
        if expected[index] is None:
            assert actual[index] is None
        else:
            value = actual[index]
            if index == 0:
                value = value * torch.sigmoid(source.interval.rho)
            torch.testing.assert_close(value, expected[index])
    torch.testing.assert_close(actual[2].sum(), expected[2])


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_fullgraph_heap_continuation_matches_eager_values_and_gradients(backend):
    source = _source()
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)

    def run(parameters, constants, state, packet, schedule):
        return plan.step(
            parameters,
            constants,
            state,
            torch.tensor(0.7, dtype=plan.dtype),
            packet,
            bptt=True,
            schedule=schedule,
        )

    expected = run(*tensors[:3], draw, tensors.schedule)
    with torch_compiler_warning_context():
        actual = torch.compile(run, backend=backend, fullgraph=True)(
            *tensors[:3], draw, tensors.schedule
        )
    _close(actual, expected)
    actual_grad = torch.autograd.grad(
        actual[0]["next_stoch_time"].sum() + actual[1].spike_gate.sum(),
        tensors.parameters["interval"],
        retain_graph=True,
    )[0]
    expected_grad = torch.autograd.grad(
        expected[0]["next_stoch_time"].sum() + expected[1].spike_gate.sum(),
        tensors.parameters["interval"],
    )[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("lanes", (0, 3))
def test_vmap_heap_cursors_and_empty_lanes(lanes):
    source = _source()
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    states = {
        name: value.unsqueeze(0).expand(lanes, *value.shape).clone()
        for name, value in tensors.state.items()
    }

    def run(state):
        return plan.step(
            tensors.parameters,
            tensors.constants,
            state,
            0.7,
            draw,
            bptt=True,
            schedule=tensors.schedule,
        )

    actual = torch.vmap(run)(states)
    expected = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape),
        run(tensors.state),
    )
    _close(actual, expected)


def test_cursor_commit_preserves_warm_plan_and_native_checkpoint_continuation():
    source = _source()
    reference = copy.deepcopy(source)
    plan, tensors = dn.func.make_functional_netstim(source)
    compiled = torch.compile(plan.step, backend="aot_eager", fullgraph=True)
    state = tensors.state
    for time in (0.0, 0.1, 0.4):
        draw = plan.sample_step_draw(state)
        state, _ = compiled(
            tensors.parameters,
            tensors.constants,
            state,
            time,
            draw,
            bptt=True,
            schedule=tensors.schedule,
        )
        reference(time, bptt=True)
    generation = source._renewal_config_generation
    prefix_state = state
    snapshot = _clone(state)
    plan.commit_state_(source, state)
    assert source._renewal_config_generation == generation
    _close(plan.extract().state, state)
    for time in (0.7, 1.0, 2.0):
        draw = plan.sample_step_draw(state)
        state, output = compiled(
            tensors.parameters,
            tensors.constants,
            state,
            time,
            draw,
            bptt=True,
            schedule=tensors.schedule,
        )
        source(time, bptt=True)
        reference(time, bptt=True)
        _close(state, _carry(source))
        _close(_carry(source), _carry(reference))
        torch.testing.assert_close(output.spike_gate, reference.spike_gate)
    _close(prefix_state, snapshot)


@pytest.mark.parametrize("bad_cursor", ("below", "above", "dtype"))
def test_invalid_heap_cursor_commit_is_atomic(bad_cursor):
    source = _source()
    plan, tensors = dn.func.make_functional_netstim(source)
    before = _clone(source.state_dict_for_checkpoint())
    state = _clone(tensors.state)
    if bad_cursor == "below":
        state["heap_cursor"][0] = -1
    elif bad_cursor == "above":
        state["heap_cursor"][0] = tensors.constants["heap_row_offsets"][1] + 1
    else:
        state["heap_cursor"] = state["heap_cursor"].to(torch.int32)
    with pytest.raises(ValueError):
        plan.commit_state_(source, state)
    _close(source.state_dict_for_checkpoint(), before)


def test_heap_topology_is_frozen_and_cold_transfer_rejects_another_calendar():
    source = _source()
    target = _source()
    plan, tensors = dn.func.make_functional_netstim(source)
    target.schedule(0, 4.0)
    with pytest.raises(dn.func.FunctionalizationError, match="heap calendar"):
        plan.commit_state_(target, tensors.state)


@pytest.mark.parametrize("outer_context", (torch.no_grad, torch.inference_mode))
def test_heap_extraction_owns_ordinary_non_differentiable_calendar_snapshots(
    outer_context,
):
    source = _source()
    with outer_context():
        _, tensors = dn.func.make_functional_netstim(source)
    for name in HEAP_CONSTANTS:
        value = tensors.constants[name]
        assert not value.requires_grad
        assert not value.is_inference()
        assert value.data_ptr() != getattr(source, name).data_ptr()


def test_initialization_preserves_consumed_cursors_and_resets_only_renewal_state():
    source = _source(mixed=False)
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    advanced, _ = plan.step(
        tensors.parameters, tensors.constants, tensors.state, 0.7, draw, bptt=True
    )
    plan.commit_state_(source, advanced)
    packet = plan.sample_initialization_draw(tensors.parameters, advanced)
    initialized = plan.initial_state(
        tensors.parameters, tensors.constants, advanced, packet
    )
    source.initialize()
    _close(initialized, _carry(source))
    torch.testing.assert_close(initialized["heap_cursor"], advanced["heap_cursor"])
    assert not initialized["heap_override_active"].any()


def test_non_differentiable_calendar_constants_are_validated_before_execution():
    source = _source(mixed=False)
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    constants = {
        **tensors.constants,
        "heap_times": tensors.constants["heap_times"].requires_grad_(),
    }
    with pytest.raises(ValueError, match="must not require gradients"):
        plan.step(tensors.parameters, constants, tensors.state, 0.0, draw)


@pytest.mark.parametrize("head", (0.15, float("nan"), -float("inf"), float("inf")))
def test_legacy_head_only_checkpoint_is_explicit_carry_and_initialization_clears_it(
    head,
):
    source = _source(mixed=False)
    checkpoint = _clone(source.state_dict_for_checkpoint())
    for name in (*HEAP_CONSTANTS, *HEAP_CARRY, "sched_heaps"):
        checkpoint.pop(name, None)
    checkpoint["next_sched_time"] = torch.full(source.shape, head, dtype=source.dtype())
    source.restore_dict_from_checkpoint(checkpoint)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    assert state["heap_override_active"].all()
    for time in (0.1, 0.2, 0.7):
        draw = plan.sample_step_draw(state)
        state, output = plan.step(
            tensors.parameters, tensors.constants, state, time, draw, bptt=True
        )
        source(time, bptt=True)
        _close(state, _carry(source))
        torch.testing.assert_close(output.spike_gate, source.spike_gate, equal_nan=True)
    packet = plan.sample_initialization_draw(tensors.parameters, state)
    initialized = plan.initial_state(
        tensors.parameters, tensors.constants, state, packet
    )
    source.initialize()
    _close(initialized, _carry(source))


def test_bptt_disable_clears_clock_history_with_heap_calendar_carry():
    source = _source(mixed=False)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    for index, bptt in enumerate((True, True, False, True)):
        draw = plan.sample_step_draw(state)
        state, output = plan.step(
            tensors.parameters, tensors.constants, state, index * 0.2, draw, bptt=bptt
        )
        source(index * 0.2, bptt=bptt)
        _close(state, _carry(source))
        torch.testing.assert_close(output.spike_gate, source.spike_gate)
        assert state["next_stoch_time"].requires_grad is bptt


def test_legacy_head_restore_keeps_warm_plan_independent_of_native_carry():
    source = _source(mixed=False)
    source.next_stoch_time.fill_(100.0)
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    compiled = torch.compile(plan.step, backend="aot_eager", fullgraph=True)
    expected = compiled(tensors.parameters, tensors.constants, tensors.state, 0.2, draw)
    checkpoint = _clone(source.state_dict_for_checkpoint())
    for name in (*HEAP_CONSTANTS, *HEAP_CARRY, "sched_heaps"):
        checkpoint.pop(name, None)
    checkpoint["next_sched_time"] = torch.full(source.shape, 0.9, dtype=source.dtype())
    generation = source._renewal_config_generation
    source.restore_dict_from_checkpoint(checkpoint)
    assert source._renewal_config_generation == generation
    actual = compiled(tensors.parameters, tensors.constants, tensors.state, 0.2, draw)
    _close(actual, expected)
    extracted = plan.extract()
    assert extracted.state["heap_override_active"].all()
    _, overridden = compiled(
        extracted.parameters, extracted.constants, extracted.state, 0.2, draw
    )
    assert not torch.equal(overridden.spike_gate, expected[1].spike_gate)


@pytest.mark.parametrize("parent_move", (False, True))
def test_functional_extraction_after_dtype_roundtrip_retains_original_heap_boundary(
    parent_move,
):
    source = dn.NetStim(
        N=1, start=1000.0, noise=0.0, seed=71, dtype=torch.float64
    ).initialize()
    event_time = 1.0 + 2.0**-30
    source.schedule(0, event_time)
    owner = torch.nn.ModuleDict({"source": source}) if parent_move else source
    owner.float().double()
    plan, tensors = dn.func.make_functional_netstim(source)
    assert tensors.constants["heap_times"][0].item() == event_time
    draw = plan.sample_step_draw(tensors.state)
    state, output = plan.step(
        tensors.parameters, tensors.constants, tensors.state, 1.0, draw
    )
    source(1.0, bptt=False)
    assert not output.spikes.any()
    _close(state, _carry(source))
    draw = plan.sample_step_draw(state)
    state, output = plan.step(
        tensors.parameters, tensors.constants, state, event_time, draw
    )
    source(event_time, bptt=False)
    assert output.spikes.all()
    _close(state, _carry(source))
