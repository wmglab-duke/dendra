"""Functional NetStim shares renewal, random streams, and native checkpoints."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

pytestmark = pytest.mark.cpu


def _native(*, seed=217, noise=(0.0, 0.6, 0.95), batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        source = dn.NetStim(
            N=3,
            interval=(0.5, 0.7, 1.2),
            start=(-0.1, 0.15, 0.4),
            noise=noise,
            max_spikes=(2, 5, 3),
            tau=0.17,
            seed=seed,
            dtype=torch.float64,
        )
        if batch is not None:
            source.batch(batch)
        source.initialize()
    return source


def _clone(value):
    return pytree.tree_map_only(torch.Tensor, lambda item: item.detach().clone(), value)


def _close(actual, expected):
    actual_values, actual_spec = pytree.tree_flatten(actual)
    expected_values, expected_spec = pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for value, reference in zip(actual_values, expected_values, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def _native_carry(source):
    checkpoint = source.state_dict_for_checkpoint()
    return {
        key: checkpoint[key]
        for key in (
            "next_stoch_time",
            "spike_counts",
            "t_last",
            "rng_state",
            "seeder_state",
        )
    }


@pytest.mark.parametrize("seed", (None, 217))
@pytest.mark.parametrize("noise", (0.0, (0.0, 0.6, 0.95)))
@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("bptt", (False, True))
def test_seeded_native_parity_and_explicit_sampling_purity(seed, noise, batch, bptt):
    source = _native(seed=seed, noise=noise, batch=batch)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    original = _clone(state)
    for step in range(24):
        before = _clone(_native_carry(source))
        draw = plan.sample_step_draw(state)
        _close(_native_carry(source), before)
        # Repeating a sample from the same explicit RNG state replays exactly.
        _close(draw, plan.sample_step_draw(state))
        t = state["next_stoch_time"].new_tensor(step * 0.15)
        if batch is not None:
            t = torch.stack((t, t + 0.03))
        state, outputs = plan.step(
            tensors.parameters, tensors.constants, state, t, draw, bptt=bptt
        )
        source(t, bptt=bptt)
        torch.testing.assert_close(outputs.spikes, source.spikes)
        torch.testing.assert_close(outputs.spike_gate, source.spike_gate)
        _close(state, _native_carry(source))
    _close(tensors.state, original)
    assert torch.all(state["spike_counts"] <= tensors.constants["max_spikes"])


@pytest.mark.parametrize("seed", (None, 217))
@pytest.mark.parametrize("noise", (0.0, (0.0, 0.6, 0.95)))
def test_initialization_matches_conditional_rng_draws_and_reseeding(seed, noise):
    source = _native(seed=seed, noise=noise)
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    advanced, _ = plan.step(
        tensors.parameters, tensors.constants, tensors.state, 0.8, draw, bptt=True
    )
    plan.commit_state_(source, advanced)
    before = _clone(_native_carry(source))
    draw = plan.sample_initialization_draw(tensors.parameters, advanced)
    state = plan.initial_state(tensors.parameters, tensors.constants, advanced, draw)
    _close(_native_carry(source), before)
    source.initialize()
    _close(state, _native_carry(source))
    assert not state["next_stoch_time"].requires_grad
    if noise == 0.0 and seed is None:
        torch.testing.assert_close(state["rng_state"], before["rng_state"])


def test_extraction_creates_independent_leaves_and_does_not_fill_eval_cache():
    source = _native().eval()
    source.noise.requires_grad_()
    source.tau = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    assert source.interval._cache is None
    before = _clone(_native_carry(source))
    plan, tensors = dn.func.make_functional_netstim(source)
    assert source.interval._cache is None
    assert tensors.parameters["interval"].is_leaf
    assert tensors.parameters["interval"].requires_grad
    for value in tensors.parameters.values():
        assert value.is_leaf
        assert value.grad_fn is None
    assert tensors.parameters["noise"].data_ptr() != source.noise.data_ptr()
    assert tensors.parameters["tau"].shape == source.shape
    with torch.no_grad():
        tensors.parameters["start"].add_(10)
    _close(_native_carry(source), before)
    assert not torch.equal(tensors.parameters["start"], source.start)
    assert plan.extract().parameters["tau"].requires_grad


@pytest.mark.parametrize("context", (torch.no_grad, torch.inference_mode))
def test_extraction_preserves_gradient_convention_inside_disabled_autograd(context):
    source = _native()
    source.noise.requires_grad_()
    with context():
        _, tensors = dn.func.make_functional_netstim(source)
    assert tensors.parameters["interval"].requires_grad
    assert tensors.parameters["noise"].requires_grad
    assert all(not value.is_inference() for value in tensors.parameters.values())


@pytest.mark.parametrize("bptt", (False, True))
def test_gradient_parity_and_source_parameter_isolation(bptt):
    source = _native()
    source.noise.requires_grad_()
    source.tau = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    functional_loss = state["next_stoch_time"].new_zeros(())
    native_loss = functional_loss.clone()
    for step in range(16):
        t = functional_loss.new_tensor(step * 0.13)
        draw = plan.sample_step_draw(state)
        state, outputs = plan.step(
            tensors.parameters, tensors.constants, state, t, draw, bptt=bptt
        )
        source(t, bptt=bptt)
        factor = step + 1.0
        functional_loss = functional_loss + factor * outputs.spike_gate.sum()
        native_loss = native_loss + factor * source.spike_gate.sum()
    if bptt:
        functional_loss = functional_loss + state["next_stoch_time"].sum()
        native_loss = native_loss + source.next_stoch_time.sum()
    assert state["next_stoch_time"].requires_grad is bptt
    names = ("interval", "noise", "tau")
    actual = torch.autograd.grad(
        functional_loss,
        tuple(tensors.parameters[name] for name in names),
        allow_unused=True,
    )
    assert all(parameter.grad is None for parameter in source.parameters())
    expected = torch.autograd.grad(
        native_loss, (source.interval.rho, source.noise, source.tau), allow_unused=True
    )
    for index, (gradient, reference) in enumerate(zip(actual, expected, strict=True)):
        if reference is None:
            assert gradient is None
        elif index == 0:
            torch.testing.assert_close(
                gradient * torch.sigmoid(source.interval.rho), reference
            )
        elif index == 2:
            torch.testing.assert_close(gradient.sum(), reference)
        else:
            torch.testing.assert_close(gradient, reference)


@pytest.mark.parametrize("outer_inference", (False, True))
def test_checkpoint_split_native_continuation_and_independent_owned_carry(
    outer_inference,
):
    source = _native(seed=None)
    reference = copy.deepcopy(source)
    plan, tensors = dn.func.make_functional_netstim(source)
    state = tensors.state
    for step in range(7):
        draw = plan.sample_step_draw(state)
        state, _ = plan.step(
            tensors.parameters, tensors.constants, state, step * 0.2, draw, bptt=True
        )
        reference(step * 0.2, bptt=True)
    frozen = _clone(state)
    with torch.inference_mode(outer_inference):
        assert plan.commit_state_(source, state) is source
    for key in ("next_stoch_time", "spike_counts", "t_last"):
        assert source._buffers[key].data_ptr() != state[key].data_ptr()
        assert not source._buffers[key].is_inference()
    for step in range(7, 14):
        source(step * 0.2, bptt=True)
        reference(step * 0.2, bptt=True)
        _close(_native_carry(source), _native_carry(reference))
        torch.testing.assert_close(source.spike_gate, reference.spike_gate)
    _close(state, frozen)
    # The saved functional checkpoint resumes the same random stream again.
    replay = state
    for step in range(7, 14):
        draw = plan.sample_step_draw(replay)
        replay, outputs = plan.step(
            tensors.parameters, tensors.constants, replay, step * 0.2, draw, bptt=True
        )
    _close(replay, _native_carry(source))
    torch.testing.assert_close(outputs.spike_gate, source.spike_gate)


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_fullgraph_compilation_and_gradients(backend):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)

    def run(parameters, constants, state, t, packet):
        next_state, outputs = plan.step(
            parameters, constants, state, t, packet, bptt=True
        )
        return (
            next_state,
            outputs,
            next_state["next_stoch_time"].sum() + outputs.spike_gate.sum(),
        )

    time = torch.tensor(0.8, dtype=torch.float64)
    expected = run(tensors.parameters, tensors.constants, tensors.state, time, draw)
    with torch_compiler_warning_context():
        compiled = torch.compile(run, backend=backend, fullgraph=True)
        actual = compiled(
            tensors.parameters, tensors.constants, tensors.state, time, draw
        )
    _close(actual, expected)
    actual_gradient = torch.autograd.grad(
        actual[2], tensors.parameters["interval"], retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(
        expected[2], tensors.parameters["interval"]
    )[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("lanes", (0, 3))
@pytest.mark.parametrize("mapped_state", (False, True))
def test_vmap_and_empty_lanes(lanes, mapped_state):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    parameters = {
        key: value.unsqueeze(0).expand(lanes, *value.shape)
        for key, value in tensors.parameters.items()
    }
    state = (
        {
            key: value.unsqueeze(0).expand(lanes, *value.shape)
            for key, value in tensors.state.items()
        }
        if mapped_state
        else tensors.state
    )
    packets = dn.func.NetStimDraw(
        draw.exp_rand.unsqueeze(0).expand(lanes, *draw.exp_rand.shape),
        draw.rng_state.unsqueeze(0).expand(lanes, *draw.rng_state.shape),
    )

    def run(params, carry, packet):
        return plan.step(params, tensors.constants, carry, 0.8, packet, bptt=True)

    actual = torch.vmap(run, in_dims=(0, 0 if mapped_state else None, 0))(
        parameters, state, packets
    )
    expected = run(tensors.parameters, tensors.state, draw)
    expanded = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape),
        expected,
    )
    _close(actual, expanded)


@pytest.mark.parametrize(
    "change",
    (
        "initialize",
        "batch",
        "eval",
        "seed",
        "schedule",
        "tensor_schedule",
        "bind",
        "diff_config",
    ),
)
def test_configuration_changes_invalidate_plan(change):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    draw = plan.sample_step_draw(tensors.state)
    if change == "initialize":
        source.initialize()
    elif change == "batch":
        source.batch(2)
    elif change == "eval":
        source.eval()
    elif change == "seed":
        source.seed = 42
    elif change == "schedule":
        source.schedule(0, 2.0)
    elif change == "tensor_schedule":
        source.schedule(
            0,
            torch.tensor(2.0, dtype=torch.float64),
            weight=torch.tensor(0.5, dtype=torch.float64),
        )
    elif change == "bind":
        source.bind_weight_source(torch.ones(1, dtype=torch.float64))
    else:
        source.set_diff_config(sched_width=2.0)
    with pytest.raises(dn.func.FunctionalizationError, match="configuration changed"):
        plan.step(tensors.parameters, tensors.constants, tensors.state, 0.0, draw)


@pytest.mark.parametrize(
    "kind", ("uninitialized", "schedule", "tensor_schedule", "bound", "subclass")
)
def test_unsupported_sources_fail_before_lowering(kind):
    source = _native()
    if kind == "uninitialized":
        source = dn.NetStim()
    elif kind == "schedule":
        source.schedule(0, 2.0)
    elif kind == "tensor_schedule":
        source.schedule(
            0,
            torch.tensor(2.0, dtype=torch.float64),
            weight=torch.tensor(0.5, dtype=torch.float64),
        )
    elif kind == "bound":
        source.bind_time_source(torch.ones(1, dtype=torch.float64))
    else:

        class CustomNetStim(dn.NetStim):
            pass

        source = CustomNetStim().initialize()
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_netstim(source)


@pytest.mark.parametrize(
    "key", ("rng_state", "seeder_state", "spike_counts", "next_stoch_time")
)
def test_invalid_state_commit_is_atomic(key):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    before = _clone(source.state_dict_for_checkpoint())
    state = _clone(tensors.state)
    if key.endswith("state"):
        state[key].fill_(255)
    else:
        state[key] = state[key].to(torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        plan.commit_state_(source, state)
    _close(source.state_dict_for_checkpoint(), before)


def test_draw_schema_and_cold_sampling_rejected_inside_compilation():
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    with pytest.raises(TypeError, match="NetStimDraw"):
        plan.step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            0.0,
            torch.ones(source.shape),
        )
    bad_draw = dn.func.NetStimDraw(
        torch.ones(1, dtype=torch.float64), tensors.state["rng_state"]
    )
    with pytest.raises(ValueError, match="draw.exp_rand"):
        plan.step(tensors.parameters, tensors.constants, tensors.state, 0.0, bad_draw)
    compiled = torch.compile(plan.sample_step_draw, backend="eager", fullgraph=True)
    with pytest.raises(Exception, match="outside torch.compile"):
        compiled(tensors.state)


@pytest.mark.parametrize("transform", (torch.func.jacrev, torch.func.jacfwd))
def test_forward_and_reverse_transforms_preserve_detached_initialization(transform):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    packet = plan.sample_initialization_draw(tensors.parameters, tensors.state)

    def initialize(start):
        parameters = {**tensors.parameters, "start": start}
        return plan.initial_state(parameters, tensors.constants, tensors.state, packet)[
            "next_stoch_time"
        ]

    derivative = transform(initialize)(tensors.parameters["start"])
    torch.testing.assert_close(derivative, torch.zeros_like(derivative))


@pytest.mark.parametrize("lanes", (0, 3))
def test_vmap_parameters_only_preserves_unmapped_rng_carry(lanes):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    packet = plan.sample_step_draw(tensors.state)

    def run(interval):
        return plan.step(
            {**tensors.parameters, "interval": interval},
            tensors.constants,
            tensors.state,
            0.8,
            packet,
            bptt=True,
        )

    interval = tensors.parameters["interval"]
    actual = torch.vmap(run)(interval.unsqueeze(0).expand(lanes, *interval.shape))
    expected = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.unsqueeze(0).expand(lanes, *value.shape),
        run(interval),
    )
    _close(actual, expected)


def test_commit_preserves_clock_history_inside_outer_inference_mode():
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    packet = plan.sample_step_draw(tensors.state)
    state, _ = plan.step(
        tensors.parameters, tensors.constants, tensors.state, 0.8, packet, bptt=True
    )
    expected = torch.autograd.grad(
        state["next_stoch_time"].sum(),
        tensors.parameters["interval"],
        retain_graph=True,
    )[0]
    with torch.inference_mode():
        plan.commit_state_(source, state)
    actual = torch.autograd.grad(
        source.next_stoch_time.sum(), tensors.parameters["interval"]
    )[0]
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual) > 0


@pytest.mark.parametrize("outer_inference", (False, True))
def test_eval_commit_detaches_incoming_clock_and_prior_native_history(outer_inference):
    source = _native()
    source(0.8, bptt=True)
    assert source.next_stoch_time.grad_fn is not None
    source.eval()
    plan, tensors = dn.func.make_functional_netstim(source)
    packet = plan.sample_step_draw(tensors.state)
    state, _ = plan.step(
        tensors.parameters, tensors.constants, tensors.state, 1.6, packet, bptt=True
    )
    assert state["next_stoch_time"].grad_fn is not None
    original_storage = source.next_stoch_time.data_ptr()

    with torch.inference_mode(outer_inference):
        plan.commit_state_(source, state)

    torch.testing.assert_close(source.next_stoch_time, state["next_stoch_time"])
    assert source.next_stoch_time.data_ptr() == original_storage
    assert source.next_stoch_time.data_ptr() != state["next_stoch_time"].data_ptr()
    assert not source.next_stoch_time.requires_grad
    assert source.next_stoch_time.grad_fn is None
    assert not source.next_stoch_time.is_inference()
    # Committing to eval must leave the independent functional graph intact.
    gradient = torch.autograd.grad(
        state["next_stoch_time"].sum(), tensors.parameters["interval"]
    )[0]
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    "change",
    (
        "initialize",
        "batch",
        "mode",
        "schedule",
        "schedule_clear",
        "tensor_schedule_clear",
        "clear",
        "clear_selected",
        "bind_weight",
        "bind_time",
        "set_dt",
        "set_diff_config",
    ),
)
def test_warm_compiled_plan_rejects_configuration_changes(change):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    packet = plan.sample_step_draw(tensors.state)
    compiled = torch.compile(plan.step, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state, 0.8, packet)
    if change == "initialize":
        source.initialize()
    elif change == "batch":
        source.batch(2)
    elif change == "mode":
        source.eval()
    elif change == "schedule":
        source.schedule(0, 2.0)
    elif change == "schedule_clear":
        source.schedule(0, 2.0)
        source.clear_schedule()
    elif change == "tensor_schedule_clear":
        source.schedule(0, 2.0, weight=0.5)
        source.clear_schedule()
    elif change == "clear":
        source.clear_schedule()
    elif change == "clear_selected":
        source.clear_schedule([0])
    elif change == "bind_weight":
        source.bind_weight_source(torch.ones(1, dtype=torch.float64))
    elif change == "bind_time":
        source.bind_time_source(torch.ones(1, dtype=torch.float64))
    elif change == "set_dt":
        source.set_dt(0.1)
    else:
        source.set_diff_config(diff_scheduled_times=False)
    with pytest.raises(Exception, match="configuration changed"):
        compiled(tensors.parameters, tensors.constants, tensors.state, 0.8, packet)


@pytest.mark.parametrize("transfer", ("native_restore", "functional_commit"))
def test_unscheduled_runtime_restore_preserves_warm_plan(transfer):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    packet = plan.sample_step_draw(tensors.state)
    compiled = torch.compile(plan.step, backend="eager", fullgraph=True)
    expected = compiled(
        tensors.parameters, tensors.constants, tensors.state, 0.8, packet
    )
    config_generation = source._renewal_config_generation
    native_snapshot = _clone(source.state_dict_for_checkpoint())
    source(100.0, bptt=False)
    if transfer == "native_restore":
        source.restore_dict_from_checkpoint(native_snapshot)
        _close(plan.extract().state, tensors.state)
    else:
        plan.commit_state_(source, expected[0])
        _close(plan.extract().state, _clone(expected[0]))
    assert source._renewal_config_generation == config_generation
    actual = compiled(tensors.parameters, tensors.constants, tensors.state, 0.8, packet)
    _close(actual, expected)


@pytest.mark.parametrize(
    "schedule", ("heap", "tensor", "legacy_head", "legacy_nan_head")
)
def test_restoring_scheduled_checkpoint_invalidates_warm_plan(schedule):
    source = _native()
    plan, tensors = dn.func.make_functional_netstim(source)
    packet = plan.sample_step_draw(tensors.state)
    compiled = torch.compile(plan.step, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state, 0.8, packet)
    if schedule in ("heap", "tensor"):
        scheduled_source = _native()
        scheduled_source.schedule(0, 2.0, weight=0.5 if schedule == "tensor" else 1.0)
        checkpoint = _clone(scheduled_source.state_dict_for_checkpoint())
    else:
        checkpoint = _clone(source.state_dict_for_checkpoint())
        del checkpoint["sched_heaps"]
        checkpoint["next_sched_time"] = torch.full(
            source.shape,
            float("nan") if schedule == "legacy_nan_head" else 2.0,
            dtype=torch.float64,
        )
    source.restore_dict_from_checkpoint(checkpoint)
    with pytest.raises(Exception, match="configuration changed"):
        compiled(tensors.parameters, tensors.constants, tensors.state, 0.8, packet)
    with pytest.raises(dn.func.FunctionalizationError, match="unscheduled renewal"):
        plan.extract()


def test_cold_extraction_still_audits_actual_schedule_contents():
    source = _native()
    plan, _ = dn.func.make_functional_netstim(source)
    # Private storage is intentionally outside the hot public lifecycle guard.
    # Cold transfer still audits every heap rather than trusting its revision.
    source._sched_heaps[0].append(2.0)
    with pytest.raises(dn.func.FunctionalizationError, match="unscheduled renewal"):
        plan.extract()


def test_compiled_hot_path_has_no_per_generator_schedule_heap_guards():
    source = dn.NetStim(N=1024, seed=217, dtype=torch.float64).initialize()
    plan, tensors = dn.func.make_functional_netstim(source)
    packet = plan.sample_step_draw(tensors.state)
    explanation = torch._dynamo.explain(plan.step)(
        tensors.parameters, tensors.constants, tensors.state, 0.8, packet
    )
    assert explanation.graph_count == 1
    assert not any("_sched_heaps" in guard.name for guard in explanation.out_guards)
    assert any(
        "_renewal_config_generation" in guard.name for guard in explanation.out_guards
    )
