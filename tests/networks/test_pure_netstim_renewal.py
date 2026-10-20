"""Shared renewal semantics, independent clocks, transforms, and RNG cadence."""

import pytest
import torch

from dendra.models.networks._netstim_renewal import (
    NetStimState,
    broadcast_netstim_time,
    netstim_initial_state,
    netstim_renewal,
)
from dendra.models.networks.netstim import NetStim

DTYPE = torch.float64


def _state(clock, counts=None):
    clock = torch.as_tensor(clock, dtype=DTYPE)
    return NetStimState(
        clock,
        torch.zeros_like(clock, dtype=torch.long) if counts is None else counts,
        torch.full_like(clock, -float("inf")),
    )


def _clone_state(state):
    return NetStimState(*(value.clone() for value in state))


def _exp_draw(generator, shape, dtype=DTYPE):
    return (
        -torch.rand(shape, generator=generator, dtype=dtype)
        .clamp_min(torch.finfo(dtype).tiny)
        .log()
    )


def test_initialization_is_detached_from_all_inputs_and_preserves_values():
    interval = torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE, requires_grad=True)
    start = torch.tensor([-1.0, 0.0, 2.0], dtype=DTYPE, requires_grad=True)
    noise = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE, requires_grad=True)
    draws = torch.tensor([0.3, 2.0, 0.5], dtype=DTYPE, requires_grad=True)
    originals = [x.detach().clone() for x in (interval, start, noise, draws)]
    state = netstim_initial_state(interval, start, noise, draws)
    torch.testing.assert_close(state.next_stoch_time, start.new_tensor([-1, 3, 4]))
    assert state.spike_counts.tolist() == [0, 0, 0]
    assert torch.isneginf(state.t_last).all()
    assert not any(x.requires_grad for x in state)
    for value, original in zip((interval, start, noise, draws), originals):
        torch.testing.assert_close(value, original)
    for transform in (torch.func.jacrev, torch.func.jacfwd):
        jac = transform(lambda x: netstim_initial_state(x, start, noise, draws)[0])(
            interval
        )
        torch.testing.assert_close(jac, torch.zeros_like(jac))


def test_deterministic_initialization_does_not_use_nonfinite_intervals():
    interval = torch.tensor([float("inf"), float("nan")], dtype=DTYPE)
    start = torch.tensor([-0.0, 2.0], dtype=DTYPE)
    state = netstim_initial_state(
        interval, start, torch.zeros_like(start), torch.zeros_like(start)
    )
    torch.testing.assert_close(state.next_stoch_time, start, rtol=0, atol=0)
    assert torch.signbit(state.next_stoch_time[0])


@pytest.mark.parametrize("noise", ([0.0, 0.0, 0.0], [0.0, 0.5, 1.0]))
def test_native_initialization_and_each_call_preserve_full_shape_rng_cadence(noise):
    shape = (2, 3)
    stim = NetStim(
        N=3,
        interval=[1.0, 2.0, 3.0],
        noise=noise,
        start=[0.0, 5.0, 10.0],
        max_spikes=[0, 1, 2],
        seed=1234,
        dtype=DTYPE,
    ).batch(2)
    clock_ptr = stim.next_stoch_time.data_ptr()
    count_ptr = stim.spike_counts.data_ptr()
    last_ptr = stim.t_last.data_ptr()
    stim.initialize()
    oracle_rng = torch.Generator().manual_seed(1234)
    expected_clock = stim.start.clone()
    if any(value > 0 for value in noise):
        expected_clock += (
            stim.noise * stim.interval().detach() * _exp_draw(oracle_rng, shape)
        )
    torch.testing.assert_close(stim.next_stoch_time, expected_clock, rtol=0, atol=0)
    assert torch.equal(stim._rng.get_state(), oracle_rng.get_state())
    for time in (-1.0, 5.0, 5.0, 100.0, 100.0):
        stim.forward(time)
        _exp_draw(oracle_rng, shape)
        assert torch.equal(stim._rng.get_state(), oracle_rng.get_state())
        assert stim.next_stoch_time.data_ptr() == clock_ptr
        assert stim.spike_counts.data_ptr() == count_ptr
        assert stim.t_last.data_ptr() == last_ptr


def test_reinitialize_detaches_clock_and_preserves_owned_storage():
    stim = (
        NetStim(interval=1.0, noise=0.5, seed=19, dtype=DTYPE).unfreeze().initialize()
    )
    stim.forward(100.0, bptt=True)
    assert stim.next_stoch_time.requires_grad
    ptr = stim.next_stoch_time.data_ptr()
    generation = stim._renewal_generation
    stim.initialize()
    assert not stim.next_stoch_time.requires_grad
    assert stim.next_stoch_time.data_ptr() == ptr
    assert stim._renewal_generation == generation + 1
    assert stim._renewal_initialized


def test_switching_off_bptt_cuts_clock_history_without_replacing_storage():
    stim = NetStim(interval=1.0, noise=0.0, dtype=DTYPE).unfreeze().initialize()
    stim.forward(0.0, bptt=True)
    assert stim.next_stoch_time.requires_grad
    pointer = stim.next_stoch_time.data_ptr()
    stim.forward(stim.next_stoch_time.detach().clone(), bptt=False)
    assert not stim.next_stoch_time.requires_grad
    assert stim.next_stoch_time.grad_fn is None
    assert stim.next_stoch_time.data_ptr() == pointer
    stim.forward(stim.next_stoch_time.detach().clone(), bptt=True)
    actual = torch.autograd.grad(
        stim.next_stoch_time.sum(), tuple(stim.interval.parameters())
    )
    expected = torch.autograd.grad(
        stim.interval().sum(), tuple(stim.interval.parameters())
    )
    for value, derivative in zip(actual, expected):
        torch.testing.assert_close(value, derivative)


def test_repeated_skipped_times_caps_and_scheduled_events_match_scalar_ledger():
    state = _state([0.0, 0.5, 0.0], torch.tensor([0, 0, 1]))
    interval = state.next_stoch_time.new_tensor([1.0, 2.0, 3.0])
    noise = state.next_stoch_time.new_tensor([0.0, 0.5, 1.0])
    caps = torch.tensor([3, 2, 1])
    clocks, counts = state.next_stoch_time.tolist(), state.spike_counts.tolist()
    retained = []
    for step, time in enumerate((0.0, 0.0, 0.25, 5.0, 5.0, 100.0)):
        draws = interval.new_tensor([0.5, 1.5, 2.5]) + step
        heap = interval.new_tensor([0.25 if step == 2 else float("inf")] * 3)
        schedule_gate = interval.new_tensor([0, 2 if step == 2 else 0, 3])
        schedule_hard = schedule_gate > 0
        before = _clone_state(state)
        transition = netstim_renewal(
            state,
            interval,
            noise,
            caps,
            interval.new_tensor(time),
            draws,
            tau=0.2,
            next_sched_time=heap,
            scheduled_gate=schedule_gate,
            scheduled_hard=schedule_hard,
        )
        for lane in range(3):
            available = counts[lane] < int(caps[lane])
            renewal = time >= clocks[lane] and available
            heap_due = time >= float(heap[lane]) and available
            scheduled = bool(schedule_hard[lane]) and available
            hard = renewal or heap_due or scheduled
            gate = float(renewal or heap_due) + (
                float(schedule_gate[lane]) if available else 0
            )
            assert bool(transition.spikes[lane]) == hard
            assert float(transition.spike_gate[lane]) == gate
            assert bool(transition.heap_fires[lane]) == heap_due
            if renewal:
                clocks[lane] += float(interval[lane]) * (
                    1 - float(noise[lane]) + float(noise[lane] * draws[lane])
                )
            counts[lane] += int(hard)
        torch.testing.assert_close(
            transition.state.next_stoch_time, interval.new_tensor(clocks)
        )
        assert transition.state.spike_counts.tolist() == counts
        assert transition.state.t_last.tolist() == [time] * 3
        for current, original in zip(state, before):
            torch.testing.assert_close(current, original)
        retained.append((state, before))
        state = transition.state
    for saved, original in retained:
        for current, expected in zip(saved, original):
            torch.testing.assert_close(current, expected)


@pytest.mark.parametrize("bptt", (False, True))
@pytest.mark.parametrize("heap_time", (0.0, 0.3, 0.5, float("inf")))
def test_renewal_gradient_uses_combined_heap_clock_and_one_ste_gate(bptt, heap_time):
    clock = torch.tensor([0.3], dtype=DTYPE, requires_grad=True)
    interval = torch.tensor([2.0], dtype=DTYPE, requires_grad=True)
    state = _state(clock)
    transition = netstim_renewal(
        state,
        interval,
        torch.zeros_like(interval),
        torch.tensor([4]),
        interval.new_tensor(0.5),
        torch.ones_like(interval),
        tau=0.25,
        bptt=bptt,
        next_sched_time=interval.new_tensor([heap_time]),
    )
    g_clock, g_interval = torch.autograd.grad(
        transition.state.next_stoch_time.sum() + transition.spike_gate.sum(),
        (clock, interval),
        allow_unused=True,
    )
    minimum = min(0.3, heap_time)
    sigmoid = torch.sigmoid(interval.new_tensor((0.5 - minimum) / 0.25))
    dgate = -sigmoid * (1 - sigmoid) / 0.25
    if bptt:
        # torch.minimum divides the derivative equally at an exact tie.
        clock_fraction = 0.0 if heap_time < 0.3 else (0.5 if heap_time == 0.3 else 1)
        torch.testing.assert_close(
            g_clock, interval.new_tensor([1.0]) + 3 * dgate * clock_fraction
        )
    else:
        assert g_clock is None
    torch.testing.assert_close(g_interval, torch.ones_like(interval))


@pytest.mark.parametrize("bptt", (False, True))
def test_native_and_pure_scheduled_renewal_gradients_and_retained_outputs(bptt):
    stim = (
        NetStim(N=2, interval=[0.5, 0.9], noise=[0.0, 0.6], seed=17, dtype=DTYPE)
        .unfreeze()
        .initialize()
    )
    stim.schedule(0, [0.1, 0.3, 0.3])
    tensor_time = torch.tensor([0.2], dtype=DTYPE, requires_grad=True)
    tensor_weight = torch.tensor([2.0], dtype=DTYPE, requires_grad=True)
    stim.schedule(1, tensor_time, weight=tensor_weight)
    generator = torch.Generator()
    generator.set_state(stim._rng.get_state())
    state = _clone_state(
        NetStimState(stim.next_stoch_time, stim.spike_counts, stim.t_last)
    )
    pure_outputs, native_outputs = [], []
    interval = stim.interval()
    for time in (0.1, 0.16, 0.3, 0.7):
        t = interval.new_tensor(time)
        scheduled_gate, scheduled_hard = stim._scheduled_tensor_gate(t, dt=0.1)
        pure = netstim_renewal(
            state,
            interval,
            stim.noise,
            stim.max_spikes,
            t,
            _exp_draw(generator, stim.shape),
            tau=stim.tau,
            bptt=bptt,
            next_sched_time=stim.next_sched_time,
            scheduled_gate=scheduled_gate,
            scheduled_hard=scheduled_hard,
        )
        stim.forward(time, bptt=bptt, dt=0.1)
        torch.testing.assert_close(pure.spikes, stim.spikes)
        torch.testing.assert_close(pure.spike_gate, stim.spike_gate)
        torch.testing.assert_close(pure.state.next_stoch_time, stim.next_stoch_time)
        torch.testing.assert_close(pure.state.spike_counts, stim.spike_counts)
        pure_outputs.append(pure.spike_gate)
        native_outputs.append(stim.spike_gate)
        state = pure.state
    leaves = (*stim.interval.parameters(), tensor_time, tensor_weight)
    pure_grad = torch.autograd.grad(
        sum(x.sum() for x in pure_outputs),
        leaves,
        allow_unused=True,
        retain_graph=True,
    )
    native_grad = torch.autograd.grad(
        sum(x.sum() for x in native_outputs), leaves, allow_unused=True
    )
    for actual, expected in zip(pure_grad, native_grad):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)
    assert torch.equal(stim._rng.get_state(), generator.get_state())


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_pure_renewal_compiles_fullgraph_with_value_and_gradient_parity(backend):
    def step(clock, interval, draws):
        return netstim_renewal(
            _state(clock),
            interval,
            interval.new_tensor([0.2, 1.0]),
            torch.tensor([3, 2]),
            interval.new_tensor(1.0),
            draws,
            tau=0.2,
        )

    clock = torch.tensor([0.1, 1.5], dtype=DTYPE, requires_grad=True)
    interval = torch.tensor([0.5, 2.0], dtype=DTYPE, requires_grad=True)
    draws = torch.tensor([1.1, 2.2], dtype=DTYPE)
    eager = step(clock, interval, draws)
    compiled = torch.compile(step, backend=backend, fullgraph=True)(
        clock, interval, draws
    )
    for value, expected in zip(compiled.state, eager.state):
        torch.testing.assert_close(value, expected)
    torch.testing.assert_close(compiled.spike_gate, eager.spike_gate)
    torch.testing.assert_close(compiled.spikes, eager.spikes)
    for transition in (eager, compiled):
        grads = torch.autograd.grad(
            transition.state.next_stoch_time.sum() + transition.spike_gate.sum(),
            (clock, interval),
        )
        if transition is eager:
            expected_grad = grads
        else:
            for value, expected in zip(grads, expected_grad):
                torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("lanes", (0, 3))
@pytest.mark.parametrize("mapped_state", (False, True))
def test_pure_renewal_vmap_with_empty_and_unmapped_clock_state(lanes, mapped_state):
    clock = torch.zeros((lanes, 2) if mapped_state else (2,), dtype=DTYPE)
    intervals = torch.ones((lanes, 2), dtype=DTYPE)
    draws = torch.full((lanes, 2), 0.5, dtype=DTYPE)

    def step(current, interval, exp_rand):
        return netstim_renewal(
            _state(current),
            interval,
            torch.zeros_like(interval),
            torch.full_like(interval, 2, dtype=torch.long),
            interval.new_tensor(0.0),
            exp_rand,
            tau=0.1,
        )

    actual = torch.vmap(step, in_dims=(0 if mapped_state else None, 0, 0))(
        clock, intervals, draws
    )
    assert actual.state.next_stoch_time.shape == (lanes, 2)
    assert actual.spike_gate.shape == (lanes, 2)
    assert actual.state.t_last.shape == (lanes, 2)
    torch.testing.assert_close(actual.state.next_stoch_time, torch.ones_like(intervals))
    torch.testing.assert_close(actual.spike_gate, torch.ones_like(intervals))


def test_shared_time_broadcast_prioritizes_batch_axes_and_rejects_invalid_shapes():
    t = torch.tensor([1.0, 2.0], dtype=DTYPE)
    result = broadcast_netstim_time(t, (2, 2), device=t.device, dtype=t.dtype)
    assert result.shape == (2, 1)
    torch.testing.assert_close(result, t[:, None])
    with pytest.raises(ValueError, match="NetStim state shape"):
        broadcast_netstim_time(torch.zeros(3), (2, 2), device=t.device, dtype=t.dtype)


def test_checkpoint_restore_preserves_lifecycle_and_initializes_restored_runtime():
    stim = NetStim(seed=123, dtype=DTYPE)
    assert not stim._renewal_initialized
    stim.initialize()
    generation = stim._renewal_generation
    checkpoint = stim.state_dict_for_checkpoint()
    stim.restore_dict_from_checkpoint(checkpoint)
    assert stim._renewal_generation == generation
    assert stim._renewal_initialized
    other = NetStim(seed=321, dtype=DTYPE)
    other.restore_dict_from_checkpoint(checkpoint)
    assert other._renewal_initialized
    other.batch(2)
    assert other._renewal_generation == 1


def test_canonical_schedule_configuration_mutations_advance_revision():
    stim = NetStim(dtype=DTYPE).initialize()
    operations = (
        lambda: stim.set_dt(0.1),
        lambda: stim.set_diff_config(diff_scheduled_times=False),
        lambda: stim.bind_weight_source(torch.ones(1, dtype=DTYPE)),
        lambda: stim.bind_time_source(torch.ones(1, dtype=DTYPE)),
        lambda: stim.schedule(0, 0.2),
        lambda: stim.schedule(0, torch.tensor(0.3, dtype=DTYPE), weight=2),
        lambda: stim.schedule_ref(0, times=0.4, weight_idx=0),
        lambda: stim.schedule_time_ref(0, time_idx=0),
        lambda: stim.clear_schedule(),
        lambda: stim.clear_schedule(),
    )
    for operation in operations:
        previous = stim._renewal_config_generation
        operation()
        assert stim._renewal_config_generation == previous + 1


def test_checkpoint_configuration_revision_changes_only_for_restored_schedules():
    stim = NetStim(dtype=DTYPE).initialize()
    checkpoint = stim.state_dict_for_checkpoint()
    revision = stim._renewal_config_generation
    stim.restore_dict_from_checkpoint(checkpoint)
    assert stim._renewal_config_generation == revision

    malformed = dict(checkpoint)
    malformed["sched_heaps"] = ()
    with pytest.raises(ValueError, match="schedule heaps"):
        stim.restore_dict_from_checkpoint(malformed)
    assert stim._renewal_config_generation == revision

    # Older checkpoints can contain a scheduled clock without Python heaps.
    legacy = dict(checkpoint)
    del legacy["sched_heaps"]
    legacy["next_sched_time"] = torch.tensor([0.25], dtype=DTYPE)
    stim.restore_dict_from_checkpoint(legacy)
    assert stim._renewal_config_generation == revision + 1
    stim.restore_dict_from_checkpoint(checkpoint)
    assert stim._renewal_config_generation == revision + 1

    other = NetStim(dtype=DTYPE).initialize()
    other.schedule(0, 0.5)
    stim.restore_dict_from_checkpoint(other.state_dict_for_checkpoint())
    assert stim._renewal_config_generation == revision + 2
