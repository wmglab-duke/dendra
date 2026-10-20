"""Explicit NetStim tensor schedules share native values, kernels, and replay."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

pytestmark = pytest.mark.cpu


def _native(*, kind="mixed", triangular=True, batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        source = dn.NetStim(
            N=3,
            interval=(0.3, 0.4, 0.5),
            start=(0.0, 0.15, 0.4),
            noise=(0.0, 0.4, 0.8),
            max_spikes=(2, 5, 1),
            tau=0.17,
            seed=71,
            dtype=torch.float64,
        )
        if batch is not None:
            source.batch(batch)
        source.set_dt(0.1)
        source.set_diff_config(diff_scheduled_times=triangular, sched_width=0.8)
        source.initialize()
        if kind in ("value", "mixed"):
            source.schedule(
                [0, 1, 1, 2],
                torch.tensor(
                    [0.15, 0.3, 0.3, 0.5], dtype=torch.float64, requires_grad=True
                ),
                weight=torch.tensor(
                    [0.5, 0.0, -2.0, 1.3], dtype=torch.float64, requires_grad=True
                ),
            )
        if kind in ("ref", "mixed"):
            source.bind_weight_source(
                torch.tensor([0.25, 0.8, -1.2], dtype=torch.float32, requires_grad=True)
            )
            source.schedule_ref(
                [0, 1, 2],
                torch.tensor(
                    [0.15, 0.2, 0.55], dtype=torch.float64, requires_grad=True
                ),
                [0, 2, 1],
            )
        if kind in ("time_ref", "mixed"):
            source.bind_time_source(
                torch.tensor(
                    [0.15, 0.31, 0.55], dtype=torch.float32, requires_grad=True
                )
            )
            source.schedule_time_ref(
                [0, 1, 2],
                [0, 2, 1],
                weight=torch.tensor(
                    [0.75, -1.1, 0.0], dtype=torch.float64, requires_grad=True
                ),
            )
    return source


def _close(actual, expected):
    actual_values, actual_spec = pytree.tree_flatten(actual)
    expected_values, expected_spec = pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for value, reference in zip(actual_values, expected_values, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def _carry(source):
    checkpoint = source.state_dict_for_checkpoint()
    return {
        name: checkpoint[name]
        for name in (
            "next_stoch_time",
            "spike_counts",
            "t_last",
            "rng_state",
            "seeder_state",
        )
    }


@pytest.mark.parametrize("kind", ("value", "ref", "time_ref", "mixed"))
@pytest.mark.parametrize("triangular", (False, True))
@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("bptt", (False, True))
def test_native_parity_rng_counts_and_schedule_gradients(kind, triangular, batch, bptt):
    source = _native(kind=kind, triangular=triangular, batch=batch)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    original = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors
    )
    loss = state["next_stoch_time"].new_zeros(())
    native_loss = loss.clone()
    for index in range(12):
        t = loss.new_tensor(index * 0.1)
        if batch is not None:
            t = torch.stack((t, t + 0.02))
        draw = plan.sample_step_draw(state)
        state, outputs = plan.step(
            tensors.parameters,
            tensors.constants,
            state,
            t,
            draw,
            bptt=bptt,
            schedule=tensors.schedule,
        )
        source(t, bptt=bptt)
        _close(state, _carry(source))
        torch.testing.assert_close(outputs.spikes, source.spikes)
        torch.testing.assert_close(outputs.spike_gate, source.spike_gate)
        loss = loss + (index + 1) * outputs.spike_gate.sum()
        native_loss = native_loss + (index + 1) * source.spike_gate.sum()

    fields = ("times", "amplitudes", "time_source", "amplitude_source")
    native_fields = (
        "sched_time_ms",
        "sched_weight",
        "_sched_t_source",
        "_sched_w_source",
    )
    pairs = [
        (getattr(tensors.schedule, field), getattr(source, native_field))
        for field, native_field in zip(fields, native_fields, strict=True)
        if getattr(tensors.schedule, field) is not None
        and getattr(tensors.schedule, field).requires_grad
    ]
    actual = torch.autograd.grad(loss, [pair[0] for pair in pairs], allow_unused=True)
    expected = torch.autograd.grad(
        native_loss, [pair[1] for pair in pairs], allow_unused=True
    )
    _close(actual, expected)
    _close(tensors, original)


def test_explicit_packet_is_required_and_values_are_independent_leaves():
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    assert isinstance(tensors.schedule, dn.func.TensorScheduleTensors)
    for value in tensors.schedule:
        assert value.is_leaf
        assert value.grad_fn is None
    assert tensors.schedule.amplitude_source.dtype == torch.float32
    draw = plan.sample_step_draw(tensors.state)
    with pytest.raises(TypeError, match="explicit TensorScheduleTensors"):
        plan.step(tensors.parameters, tensors.constants, tensors.state, 0.2, draw)
    malformed = tensors.schedule._replace(times=tensors.schedule.times[:-1])
    with pytest.raises(ValueError, match="schedule.times"):
        plan.step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            0.2,
            draw,
            schedule=malformed,
        )
    before = source.sched_weight.detach().clone()
    with torch.no_grad():
        tensors.schedule.amplitudes.add_(10)
    torch.testing.assert_close(source.sched_weight, before)


@pytest.mark.parametrize("outer_context", (torch.no_grad, torch.inference_mode))
def test_schedule_extraction_in_disabled_autograd_preserves_leaf_conventions(
    outer_context,
):
    source = _native()
    with outer_context():
        _, tensors = dn.func.make_functional_netstim(source)
    for value in tensors.schedule:
        assert value.requires_grad
        assert value.is_leaf
        assert not value.is_inference()


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_fullgraph_compile_explicit_schedule_and_gradients(backend):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)

    def run(schedule):
        return plan.step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            0.2,
            draw,
            bptt=True,
            schedule=schedule,
        )

    expected = run(tensors.schedule)
    with torch_compiler_warning_context():
        actual = torch.compile(run, backend=backend, fullgraph=True)(tensors.schedule)
    _close(actual, expected)
    fields = (
        tensors.schedule.times,
        tensors.schedule.amplitudes,
        tensors.schedule.time_source,
        tensors.schedule.amplitude_source,
    )
    actual_grad = torch.autograd.grad(
        actual[1].spike_gate.sum(), fields, retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected[1].spike_gate.sum(), fields)
    _close(actual_grad, expected_grad)


@pytest.mark.parametrize("lanes", (0, 3))
def test_vmap_schedule_values_and_empty_lanes(lanes):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)

    def run(schedule):
        return plan.step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            0.2,
            draw,
            schedule=schedule,
        )

    packets = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape),
        tensors.schedule,
    )
    actual = torch.vmap(run)(packets)
    expected = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape),
        run(tensors.schedule),
    )
    _close(actual, expected)


def test_scheduled_checkpoint_native_continuation_preserves_warm_plan():
    source = _native()
    reference = copy.deepcopy(source)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    compiled = torch.compile(plan.step, backend="eager", fullgraph=True)
    for index in range(4):
        draw = plan.sample_step_draw(state)
        state, outputs = compiled(
            tensors.parameters,
            tensors.constants,
            state,
            index * 0.1,
            draw,
            bptt=True,
            schedule=tensors.schedule,
        )
        reference(index * 0.1, bptt=True)
    generation = source._renewal_config_generation
    plan.commit_state_(source, state)
    assert source._renewal_config_generation == generation
    # The source-bound compiled plan remains usable after an ordinary carry restore.
    draw = plan.sample_step_draw(state)
    compiled(
        tensors.parameters,
        tensors.constants,
        state,
        0.4,
        draw,
        bptt=True,
        schedule=tensors.schedule,
    )
    for index in range(4, 10):
        source(index * 0.1, bptt=True)
        reference(index * 0.1, bptt=True)
        _close(_carry(source), _carry(reference))
        torch.testing.assert_close(source.spike_gate, reference.spike_gate)


@pytest.mark.parametrize("bound", ("empty", "nan"))
def test_unused_bound_sources_follow_native_skip_policy(bound):
    source = _native(kind="value")
    vector = (
        torch.empty(0, dtype=torch.float64)
        if bound == "empty"
        else torch.tensor([float("nan")], dtype=torch.float64)
    )
    source.bind_time_source(vector)
    source.bind_weight_source(vector)
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    state, outputs = plan.step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        0.2,
        draw,
        schedule=tensors.schedule,
    )
    source(0.2, bptt=False)
    _close(state, _carry(source))
    torch.testing.assert_close(outputs.spike_gate, source.spike_gate)
    assert torch.isfinite(outputs.spike_gate).all()


def test_schedule_values_can_change_without_changing_frozen_topology():
    source = _native(kind="value")
    plan, tensors = dn.func.make_functional_netstim(source)
    changed = tensors.schedule._replace(amplitudes=tensors.schedule.amplitudes * 3)
    draw = plan.sample_step_draw(tensors.state)
    _, baseline = plan.step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        0.2,
        draw,
        schedule=tensors.schedule,
    )
    _, modified = plan.step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        0.2,
        draw,
        schedule=changed,
    )
    assert not torch.equal(modified.spike_gate, baseline.spike_gate)
    source.sched_weight = changed.amplitudes.detach().clone()
    torch.testing.assert_close(plan.extract().schedule.amplitudes, changed.amplitudes)
    source.sched_flat_idx[0] = 2
    with pytest.raises(dn.func.FunctionalizationError, match="topology"):
        plan.extract()


@pytest.mark.parametrize("reference", ("time", "amplitude"))
def test_mixed_dtype_repeated_references_preserve_gather_then_cast_backward(reference):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        source = dn.NetStim(N=3, start=1000, noise=0, seed=71, dtype=torch.float64)
        source.set_dt(0.1)
        source.set_diff_config(diff_scheduled_times=True, sched_width=1.0)
        source.initialize()
        bound = torch.tensor(
            [0.05 if reference == "time" else 1.0],
            dtype=torch.float32,
            requires_grad=True,
        )
        if reference == "time":
            source.bind_time_source(bound)
            source.schedule_time_ref([0, 1, 2], [0, 0, 0])
            coefficients = torch.tensor([1e7, 0.1, -1e7], dtype=torch.float64)
        else:
            source.bind_weight_source(bound)
            source.schedule_ref([0, 1, 2], [0.1, 0.1, 0.1], [0, 0, 0])
            coefficients = torch.tensor([1e8, 1.0, -1e8], dtype=torch.float64)

    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    _, outputs = plan.step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        0.1,
        draw,
        bptt=True,
        schedule=tensors.schedule,
    )
    source(0.1, bptt=True)
    explicit = getattr(tensors.schedule, f"{reference}_source")
    actual = torch.autograd.grad((outputs.spike_gate * coefficients).sum(), explicit)[0]
    native = torch.autograd.grad((source.spike_gate * coefficients).sum(), bound)[0]

    # The pre-extraction native policy gathers float32 entries, casts each to
    # float64, then accumulates their backward contributions in float32. These
    # coefficients cancel to zero in that order; casting the whole source
    # before gathering instead accumulates in float64 and incorrectly gives 1.
    selected = bound.index_select(0, torch.tensor([0, 0, 0])).to(torch.float64)
    legacy_gate = (
        (1.0 - ((selected - 0.1) / 0.1).abs()).clamp(0, 1)
        if reference == "time"
        else selected
    )
    expected = torch.autograd.grad((legacy_gate * coefficients).sum(), bound)[0]
    torch.testing.assert_close(expected, torch.zeros_like(expected))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(native, expected)
