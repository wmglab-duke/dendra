"""Explicit tensor schedules preserve NetCon timing, gradients, and carry."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn

pytestmark = pytest.mark.cpu
DT = 0.1


def _network(*, mode="triangular", batch=None, scheduled=True):
    tracked = mode != "triangular"
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        pre = dn.Population(N=1, C=3, dtype=torch.float64)
        post = dn.Population(N=1, C=3, dtype=torch.float64)
        post.insert(expsyn.rename("syn"))
        network = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
            track_netcon_events=tracked,
        )
        network.connect_one_to_one(
            pre[:],
            post[:],
            post.mech.syn,
            threshold=None,
            pre_var="v",
            weight=torch.tensor([0.7, -0.4, 0.9], dtype=torch.float64),
            delay=torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64),
        )
        if batch is not None:
            network.batch(batch)
        network.initialize(dt=DT)
        native = next(iter(network.synapses.values()))
        if mode != "eval":
            native.set_diff_config(
                diff_weights=True,
                diff_delays=False,
                diff_spiking=False,
                diff_scheduled_times=mode == "triangular",
                sched_width=0.8,
                train_delay_backend="dense",
            )
            native.train()
            native.initialize(reinit_weights=False, reinit_delays=False)
        if scheduled:
            native.bind_weight_source(
                torch.tensor([0.7, -0.2], dtype=native.dtype, requires_grad=True)
            )
            native.bind_time_source(
                torch.tensor([0.125, 0.225], dtype=native.dtype, requires_grad=True)
            )
            for offset in range(0, native.pre_idx.numel(), 3):
                native.schedule(
                    con_indices=[offset, offset + 1, offset + 1, offset + 2],
                    times_ms=[0.025, 0.17, 0.17, 0.27],
                    weight=torch.tensor([1.2, 0.0, -0.4, 0.8], dtype=native.dtype),
                )
                native.schedule_ref(
                    con_indices=[offset + 2, offset],
                    times_ms=[0.075, 0.125],
                    weight_idx=[1, 0],
                )
                native.schedule_time_ref(
                    con_indices=[offset, offset + 2],
                    time_idx=[0, 1],
                    weight=torch.tensor([0.6, 0.3], dtype=native.dtype),
                )
            native.sched_time_ms = native.sched_time_ms.detach().requires_grad_()
            native.sched_weight = native.sched_weight.detach().requires_grad_()
    return network, native


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
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def _schedule_inputs(schedule):
    return tuple(value for value in schedule if value is not None)


@pytest.mark.parametrize("mode", ("eval", "hard", "triangular"))
@pytest.mark.parametrize("batch", (None, 2))
def test_mixed_value_and_indexed_schedules_match_native_queues_and_gradients(
    mode, batch
):
    _, native = _network(mode=mode, batch=batch)
    projection, tensors = dn.func.make_functional_netcon(native)
    snapshot = _clone(tensors.state)
    native_inputs = (
        native.weight.w,
        native.sched_time_ms,
        native.sched_weight,
        native._sched_t_source,
        native._sched_w_source,
    )
    functional_inputs = (tensors.weights, *_schedule_inputs(tensors.schedule))
    for value, original in zip(functional_inputs, native_inputs, strict=True):
        assert value.is_leaf
        assert value.requires_grad == original.requires_grad
        assert value.data_ptr() != original.data_ptr()
    state = tensors.state
    losses, native_losses = [], []
    values = torch.tensor([[0.5, -0.2, 0.0]], dtype=projection.source_dtype).expand(
        projection.source_shape
    )
    for step in range(6):
        state, payload = projection.step(
            tensors.weights, state, values, schedule=tensors.schedule
        )
        native.pre.v = values.clone()
        previous_g = native.syn.g.detach().clone()
        if mode == "eval":
            with torch.no_grad():
                native.advance()
        else:
            native.advance()
        _close(state, native.state_dict_for_checkpoint())
        torch.testing.assert_close(payload, native.syn.g - previous_g)
        losses.append((step + 1) * state["delivery_buffer"].sum())
        native_losses.append((step + 1) * native.delivery_buffer.sum())
    _close(tensors.state, snapshot)
    if mode != "eval":
        actual = torch.autograd.grad(sum(losses), functional_inputs, allow_unused=True)
        expected = torch.autograd.grad(
            sum(native_losses), native_inputs, allow_unused=True
        )
        for value, reference in zip(actual, expected, strict=True):
            if value is None or reference is None:
                assert value is reference
            else:
                torch.testing.assert_close(value, reference)
        assert torch.count_nonzero(actual[2])
        assert torch.count_nonzero(actual[4])
        if mode == "triangular":
            assert torch.count_nonzero(actual[1])
            assert torch.count_nonzero(actual[3])
        else:
            assert actual[1] is None and actual[3] is None


def test_repeated_and_skipped_global_steps_reevaluate_without_consuming_events():
    _, native = _network(mode="hard")
    projection, tensors = dn.func.make_functional_netcon(native)
    state = tensors.state
    values = torch.zeros(projection.source_shape, dtype=projection.source_dtype)
    for step in (0, 1, 1, 4, 2, 2):
        state = {
            **state,
            "global_step": state["global_step"].new_full(
                state["global_step"].shape, step
            ),
        }
        native.global_step = native.global_step.new_full(native.global_step.shape, step)
        state, _ = projection.step(
            tensors.weights, state, values, schedule=tensors.schedule
        )
        native.pre.v = values.clone()
        native.advance()
        _close(state, native.state_dict_for_checkpoint())
    # Two coincident events on connection 1 count separately, even though one
    # has zero amplitude and both calls at step 2 evaluate them again.
    assert native.sched_counts[1] == 2


def test_zero_and_cancelling_amplitudes_still_deliver_separate_event_counts():
    _, native = _network(mode="hard", scheduled=False)
    native.schedule(
        con_indices=[0, 1, 1],
        times_ms=[0.0] * 3,
        weight=torch.tensor([0.0, 1.0, -1.0], dtype=native.dtype),
    )
    projection, tensors = dn.func.make_functional_netcon(native)
    zeros = torch.zeros(projection.source_shape, dtype=projection.source_dtype)
    state, _ = projection.step(
        tensors.weights, tensors.state, zeros, schedule=tensors.schedule
    )
    state, payload = projection.step(
        tensors.weights, state, zeros, schedule=tensors.schedule
    )
    assert torch.count_nonzero(payload) == 0
    torch.testing.assert_close(
        state["events"], torch.tensor([1, 2, 0], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "topology", ("connection", "time_reference", "amplitude_reference")
)
def test_checkpoint_commit_continues_schedules_and_rejects_different_index_topology(
    topology,
):
    _, native = _network()
    _, resumed = _network()
    projection, tensors = dn.func.make_functional_netcon(native)
    values = torch.zeros(projection.source_shape, dtype=projection.source_dtype)
    state = tensors.state
    for _ in range(2):
        state, _ = projection.step(
            tensors.weights, state, values, schedule=tensors.schedule
        )
    assert "sched_wsum" not in state and "sched_counts" not in state
    projection.commit_state_(resumed, state)
    for _ in range(4):
        state, _ = projection.step(
            tensors.weights, state, values, schedule=tensors.schedule
        )
        resumed.pre.v = values.clone()
        resumed.advance()
        _close(state, resumed.state_dict_for_checkpoint())
    if topology == "connection":
        resumed.sched_con_idx[0] = 2
    elif topology == "time_reference":
        resumed.sched_time_idx[0] = 1
    else:
        resumed.sched_weight_idx[0] = 1
    with pytest.raises(dn.func.FunctionalizationError, match="schedule|topology"):
        projection.commit_state_(resumed, state)


def test_schedule_values_and_bound_sources_are_explicit_independent_snapshots():
    _, native = _network()
    projection, tensors = dn.func.make_functional_netcon(native)
    values = torch.zeros(projection.source_shape, dtype=projection.source_dtype)
    expected = projection.step(
        tensors.weights, tensors.state, values, schedule=tensors.schedule
    )
    with torch.no_grad():
        native.sched_time_ms.add_(0.2)
        native.sched_weight.add_(2.0)
        native._sched_t_source.add_(0.1)
        native._sched_w_source.add_(3.0)
    actual = projection.step(
        tensors.weights, tensors.state, values, schedule=tensors.schedule
    )
    _close(actual, expected)
    updated = projection.extract()
    for value, reference in zip(updated.schedule, tensors.schedule, strict=True):
        assert not torch.equal(value, reference)
    following = projection.step(
        updated.weights, updated.state, values, schedule=updated.schedule
    )
    assert not torch.equal(
        following[0]["delivery_buffer"], expected[0]["delivery_buffer"]
    )


def test_repeated_mixed_dtype_time_refs_cast_per_event_before_gradient_accumulation():
    _, native = _network(scheduled=False)
    events = 3 * 128
    indices = torch.zeros(events, dtype=torch.long)
    native.bind_time_source(
        torch.tensor([0.125], dtype=torch.float32, requires_grad=True)
    )
    amplitudes = torch.tensor([1e8, 1.0, -1e8], dtype=native.dtype).repeat(events // 3)
    native.schedule_time_ref(con_indices=indices, time_idx=indices, weight=amplitudes)
    projection, tensors = dn.func.make_functional_netcon(native)
    state = {**tensors.state, "global_step": tensors.state["global_step"].new_ones(1)}
    zeros = torch.zeros(projection.source_shape, dtype=projection.source_dtype)
    state = projection.advance(tensors.weights, state, zeros, schedule=tensors.schedule)
    actual = torch.autograd.grad(
        state["delivery_buffer"].sum(), tensors.schedule.time_source
    )[0]

    # The original native formula gathered in the bound source's dtype, then
    # cast each selected event. Moving this cast before the gather changes
    # float32 cancellation when many event gradients return to one source slot.
    baseline_source = tensors.schedule.time_source.detach().clone().requires_grad_()
    selected_times = baseline_source.index_select(0, indices).to(native.dtype)
    kernel = (
        1 - (selected_times / native.dt - 1).abs() / (native.train_flags[7] + 1e-6)
    ).clamp(0, 1)
    loss = (tensors.weights[0] * amplitudes * kernel).sum()
    expected = torch.autograd.grad(loss, baseline_source)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_schedule_projection_preserves_all_explicit_schedule_gradients(
    backend,
):
    with torch_compiler_warning_context():
        _, native = _network()
        projection, tensors = dn.func.make_functional_netcon(native)
        values = torch.zeros(projection.source_shape, dtype=projection.source_dtype)

        def evaluate(weights, schedule):
            state = tensors.state
            loss = weights.new_zeros(())
            for step in range(4):
                state, _ = projection.step(weights, state, values, schedule=schedule)
                loss = loss + (step + 1) * state["delivery_buffer"].sum()
            return loss, state

        kwargs = (
            {"options": {"cpp_wrapper": True, "compile_threads": 1}}
            if backend == "inductor"
            else {}
        )
        compiled = torch.compile(evaluate, backend=backend, fullgraph=True, **kwargs)
        actual = compiled(tensors.weights, tensors.schedule)
        expected = evaluate(tensors.weights, tensors.schedule)
        _close(actual, expected)
        inputs = (tensors.weights, *_schedule_inputs(tensors.schedule))
        actual_grads = torch.autograd.grad(actual[0], inputs)
        expected_grads = torch.autograd.grad(expected[0], inputs)
        _close(actual_grads, expected_grads)


@pytest.mark.parametrize("lanes", (0, 2))
def test_vmap_maps_schedule_values_sources_and_carry_including_empty_lanes(lanes):
    _, native = _network()
    projection, tensors = dn.func.make_functional_netcon(native)
    values = torch.zeros(projection.source_shape, dtype=projection.source_dtype)
    batched = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape).clone(),
        tensors,
    )

    def evaluate(weights, state, schedule):
        for _ in range(3):
            state, _ = projection.step(weights, state, values, schedule=schedule)
        return state

    actual = torch.vmap(evaluate)(batched.weights, batched.state, batched.schedule)
    if lanes:
        expected = [
            evaluate(
                batched.weights[index],
                pytree.tree_map(lambda value: value[index], batched.state),
                pytree.tree_map(lambda value: value[index], batched.schedule),
            )
            for index in range(lanes)
        ]
        _close(actual, pytree.tree_map(lambda *values: torch.stack(values), *expected))
    else:
        assert actual["delivery_buffer"].shape == (
            0,
            *tensors.state["delivery_buffer"].shape,
        )


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize(
    "change", ("append", "clear", "time_binding", "amplitude_binding")
)
def test_schedule_configuration_changes_invalidate_warmed_plans(compiled, change):
    _, native = _network()
    projection, tensors = dn.func.make_functional_netcon(native)
    values = torch.zeros(projection.source_shape, dtype=projection.source_dtype)

    def evaluate():
        return projection.step(
            tensors.weights, tensors.state, values, schedule=tensors.schedule
        )

    if compiled:
        evaluate = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    evaluate()
    if change == "append":
        native.schedule(con_indices=[0], times_ms=[0.0])
    elif change == "clear":
        native.clear_schedule()
    elif change == "time_binding":
        native.bind_time_source(native._sched_t_source.clone())
    else:
        native.bind_weight_source(native._sched_w_source.clone())
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        evaluate()


def test_schedule_packet_schema_is_explicit_and_validated_before_stepping():
    _, native = _network()
    projection, tensors = dn.func.make_functional_netcon(native)
    values = torch.zeros(projection.source_shape, dtype=projection.source_dtype)
    snapshot = _clone(tensors.state)
    bad_packets = (
        None,
        tensors.schedule._replace(times=tensors.schedule.times[:-1]),
        tensors.schedule._replace(time_source=None),
        tensors.schedule._replace(amplitudes=tensors.schedule.amplitudes.float()),
    )
    for schedule in bad_packets:
        with pytest.raises((TypeError, ValueError)):
            projection.step(tensors.weights, tensors.state, values, schedule=schedule)
    _close(tensors.state, snapshot)
    _, unscheduled = _network(scheduled=False)
    other, other_tensors = dn.func.make_functional_netcon(unscheduled)
    assert other_tensors.schedule is None
    with pytest.raises((TypeError, ValueError)):
        other.step(
            other_tensors.weights,
            other_tensors.state,
            values,
            schedule=tensors.schedule,
        )
