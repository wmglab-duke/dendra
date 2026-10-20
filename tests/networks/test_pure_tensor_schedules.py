"""Independent schedule ledgers, gradients, transforms, and native policies."""

import math

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn
from dendra.models.networks._tensor_schedules import (
    causal_step_index,
    netcon_tensor_schedule,
    netstim_tensor_schedule,
    resolve_schedule_values,
)
from dendra.models.networks.netstim import NetStim

DTYPE = torch.float64


def _tensor(values, **kwargs):
    return torch.tensor(values, dtype=DTYPE, **kwargs)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_causal_grid_matches_reconstructed_ledger_at_adjacent_boundaries(dtype):
    dt = torch.tensor(0.1, dtype=dtype)
    grid = torch.arange(-4, 5, dtype=dtype) * dt
    times = torch.cat(
        (
            torch.nextafter(grid, torch.full_like(grid, -math.inf)),
            grid,
            torch.nextafter(grid, torch.full_like(grid, math.inf)),
        )
    )
    expected = [
        next(index for index in range(-8, 9) if index * dt >= time) for time in times
    ]
    assert causal_step_index(times, dt).tolist() == expected


def test_reference_resolution_is_additive_and_accumulates_shared_source_gradients():
    times = _tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    weights = _tensor([1, 2, 3, 4], requires_grad=True)
    time_source = _tensor([0.05, 0.15], requires_grad=True)
    weight_source = _tensor([10, 20], requires_grad=True)
    refs = torch.tensor([1, -1, 0, 1])
    actual_times, actual_weights = resolve_schedule_values(
        times,
        weights,
        refs,
        refs,
        time_source=time_source,
        weight_source=weight_source,
    )
    torch.testing.assert_close(actual_times, _tensor([0.25, 0.2, 0.35, 0.55]))
    torch.testing.assert_close(actual_weights, _tensor([21, 2, 13, 24]))
    grads = torch.autograd.grad(
        actual_times.sum() + actual_weights.sum(),
        (times, weights, time_source, weight_source),
    )
    for actual, expected in zip(grads, ([1] * 4, [1] * 4, [1, 2], [1, 2])):
        torch.testing.assert_close(actual, _tensor(expected))


@pytest.mark.parametrize("kind", ("time", "weight"))
def test_reference_resolution_preserves_gather_then_cast_gradient_accumulation(kind):
    source = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    indices = torch.zeros(3, dtype=torch.long)
    values = torch.zeros(3, dtype=DTYPE)
    coefficients = _tensor([1e8, 1, -1e8])
    resolved = resolve_schedule_values(
        values,
        values,
        indices,
        indices,
        **{f"{kind}_source": source, f"{kind}_dtype": DTYPE},
    )[int(kind == "weight")]
    expected = source.index_select(0, indices).to(DTYPE)
    actual_grad = torch.autograd.grad((resolved * coefficients).sum(), source)[0]
    expected_grad = torch.autograd.grad((expected * coefficients).sum(), source)[0]
    # Casting the source before gathering would accumulate in float64 and give 1.
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert actual_grad.item() == 0


@pytest.mark.parametrize("differentiable", (False, True))
def test_netstim_schedule_matches_scalar_ledger_for_batched_repeated_skipped_times(
    differentiable,
):
    shape = (2, 2)
    targets = torch.tensor([0, 0, 1, 2, 3])
    times = _tensor([0.05, 0.15, 0.2, 0.3, 0.35])
    weights = _tensor([2, -0.5, 0, 3, 1])
    originals = times.clone(), weights.clone()
    dt = _tensor(0.1)
    for sampled in ([0.1, 0.3], [0.1, 0.3], [0.5, 0.5], [0.2, 0.4]):
        t = _tensor(sampled).reshape(2, 1)
        gate, hard = netstim_tensor_schedule(
            targets,
            times,
            weights,
            t,
            dt,
            shape=shape,
            differentiable_times=differentiable,
            width=1.2,
        )
        expected = [0.0] * 4
        hard_expected = [False] * 4
        for target, time, weight in zip(targets.tolist(), times, weights):
            current = t.reshape(-1)[target // 2]
            event_step = next(i for i in range(10) if i * dt >= time)
            current_step = next(i for i in range(10) if i * dt >= current)
            due = event_step == current_step
            kernel = (
                max(0.0, min(1.0, 1 - abs(float((time - current) / dt)) / 1.2))
                if differentiable
                else float(due)
            )
            expected[target] += float(weight) * kernel
            hard_expected[target] |= due
        torch.testing.assert_close(gate, _tensor(expected).reshape(shape))
        assert hard.tolist() == torch.tensor(hard_expected).reshape(shape).tolist()
    torch.testing.assert_close(times, originals[0])
    torch.testing.assert_close(weights, originals[1])


@pytest.mark.parametrize("differentiable", (False, True))
def test_netcon_schedule_counts_duplicate_zero_and_signed_events(differentiable):
    indices = torch.tensor([0, 0, 1, 2])
    times = _tensor([0.2, 0.2, 0.25, 0.2])
    weights = _tensor([2, -2, 3, 0])
    gate, counts = netcon_tensor_schedule(
        indices,
        times,
        weights,
        torch.tensor([2]),
        _tensor(0.1),
        connections=3,
        differentiable_times=differentiable,
    )
    expected_weight = 3 * (1 - 0.5 / 1.000001) if differentiable else 0
    torch.testing.assert_close(gate, _tensor([0, expected_weight, 0]))
    assert counts.tolist() == [2, int(differentiable), 1]


@pytest.mark.parametrize("owner", ("netstim", "netcon"))
def test_triangular_schedule_has_analytical_value_and_time_gradients(owner):
    times = _tensor([0.15, 0.25], requires_grad=True)
    weights = _tensor([2, -0.5], requires_grad=True)
    indices = torch.tensor([0, 0])
    dt = _tensor(0.1)
    width = 1.2

    def evaluate(times, weights):
        if owner == "netstim":
            return netstim_tensor_schedule(
                indices,
                times,
                weights,
                _tensor(0.2),
                dt,
                shape=(1,),
                differentiable_times=True,
                width=width,
            )[0]
        return netcon_tensor_schedule(
            indices,
            times,
            weights,
            torch.tensor(2),
            dt,
            connections=1,
            differentiable_times=True,
            width=width,
        )[0]

    effective_width = width + (1e-6 if owner == "netcon" else 0)
    kernel = 1 - 0.5 / effective_width
    gate = evaluate(times, weights)
    torch.testing.assert_close(gate, _tensor([1.5 * kernel]))
    time_grad, weight_grad = torch.autograd.grad(gate.sum(), (times, weights))
    torch.testing.assert_close(time_grad, _tensor([20, 5]) / effective_width)
    torch.testing.assert_close(weight_grad, _tensor([kernel, kernel]))
    for transform in (torch.func.jacrev, torch.func.jacfwd):
        actual = transform(evaluate, argnums=(0, 1))(times, weights)
        torch.testing.assert_close(actual[0], time_grad.reshape(1, 2))
        torch.testing.assert_close(actual[1], weight_grad.reshape(1, 2))


@pytest.mark.parametrize("owner", ("netstim", "netcon"))
@pytest.mark.parametrize("lanes", (0, 3))
@pytest.mark.parametrize("events", (0, 2))
def test_pure_schedules_support_vmap_with_empty_lanes_and_event_sets(
    owner, lanes, events
):
    indices = torch.zeros(events, dtype=torch.long)
    times = _tensor([0.15, 0.25][:events]).expand(lanes, events).clone()
    weights = torch.ones(lanes, events, dtype=DTYPE)

    def evaluate(times, weights):
        if owner == "netstim":
            return netstim_tensor_schedule(
                indices,
                times,
                weights,
                _tensor(0.2),
                _tensor(0.1),
                shape=(2,),
                differentiable_times=True,
            )
        return netcon_tensor_schedule(
            indices,
            times,
            weights,
            torch.tensor(2),
            _tensor(0.1),
            connections=2,
            differentiable_times=True,
        )

    gate, counts = torch.vmap(evaluate)(times, weights)
    assert gate.shape == counts.shape == (lanes, 2)
    if lanes:
        for lane in range(lanes):
            expected = evaluate(times[lane], weights[lane])
            torch.testing.assert_close(gate[lane], expected[0])
            torch.testing.assert_close(counts[lane], expected[1])


@pytest.mark.parametrize("differentiable", (False, True))
@pytest.mark.parametrize("lanes", (0, 3))
def test_netcon_vmap_accepts_mapped_step_carry_including_empty_lanes(
    differentiable, lanes
):
    indices = torch.tensor([0, 0])
    times = _tensor([0.15, 0.25]).expand(lanes, 2).clone()
    weights = torch.ones(lanes, 2, dtype=DTYPE)
    steps = torch.arange(lanes, dtype=torch.long).reshape(lanes, 1)

    def evaluate(times, weights, step):
        return netcon_tensor_schedule(
            indices,
            times,
            weights,
            step,
            _tensor(0.1),
            connections=2,
            differentiable_times=differentiable,
        )

    gate, counts = torch.vmap(evaluate)(times, weights, steps)
    assert gate.shape == counts.shape == (lanes, 2)
    for lane in range(lanes):
        expected = evaluate(times[lane], weights[lane], steps[lane])
        torch.testing.assert_close(gate[lane], expected[0])
        torch.testing.assert_close(counts[lane], expected[1])


@pytest.mark.parametrize("owner", ("netstim", "netcon"))
@pytest.mark.parametrize("differentiable", (False, True))
def test_fullgraph_schedule_compilation_retains_values_and_gradients(
    owner, differentiable
):
    indices = torch.tensor([0, 1])
    times = _tensor([0.15, 0.25], requires_grad=True)
    weights = _tensor([2, -0.5], requires_grad=True)

    def evaluate(times, weights):
        if owner == "netstim":
            return netstim_tensor_schedule(
                indices,
                times,
                weights,
                _tensor(0.2),
                _tensor(0.1),
                shape=(2,),
                differentiable_times=differentiable,
            )
        return netcon_tensor_schedule(
            indices,
            times,
            weights,
            torch.tensor(2),
            _tensor(0.1),
            connections=2,
            differentiable_times=differentiable,
        )

    expected = evaluate(times, weights)
    actual = torch.compile(evaluate, backend="aot_eager", fullgraph=True)(
        times, weights
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    actual_grad = torch.autograd.grad(actual[0].sum(), weights)
    expected_grad = torch.autograd.grad(expected[0].sum(), weights)
    torch.testing.assert_close(actual_grad, expected_grad)


def _native_netcon(dtype=DTYPE, generators=1):
    with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=dtype):
        post = dn.Population(N=1, C=generators, dtype=dtype)
        post.insert(expsyn.rename("syn"))
        network = dn.Network(
            {"post": post}, netstim=dn.NetStim(N=generators, dtype=dtype)
        )
        network.connect_one_to_one(
            network.netstim[:],
            post[:],
            post.mech.syn,
            threshold=None,
            weight=1.0,
            delay=0.1,
        )
        network.build(0.1)
        network.init_synapses()
        return next(iter(network.synapses.values()))


def test_native_adapters_keep_distinct_inactive_reference_source_policies():
    stim = NetStim(start=100, interval=100, dtype=DTYPE).initialize()
    stim.schedule(0, _tensor(0.2), weight=2)
    stim.bind_weight_source(_tensor([math.nan]))
    stim.bind_time_source(_tensor([math.nan]))
    gate, hard = stim._scheduled_tensor_gate(_tensor(0.2), dt=0.1)
    torch.testing.assert_close(gate, _tensor([2]))
    assert hard.tolist() == [True]

    connection = _native_netcon()
    connection.schedule(con_indices=[0], times_ms=[0.2], weight=2)
    connection.bind_weight_source(_tensor([math.nan]))
    gate, counts = connection._scheduled_gate_this_step(
        torch.tensor(2), use_tri_kernel=False
    )
    assert gate.isnan().all()
    assert counts.tolist() == [1]


@pytest.mark.parametrize("kind", ("time", "weight"))
def test_native_netstim_keeps_original_mixed_dtype_reference_gradient_order(kind):
    stim = NetStim(N=3, start=100, interval=100, dtype=DTYPE).initialize()
    source = torch.tensor(
        [0.15 if kind == "time" else 1.0], dtype=torch.float32, requires_grad=True
    )
    indices = torch.zeros(3, dtype=torch.long)
    if kind == "time":
        stim.bind_time_source(source)
        stim.schedule_time_ref([0, 1, 2], time_idx=indices)
    else:
        stim.bind_weight_source(source)
        stim.schedule_ref([0, 1, 2], times=[0.2] * 3, weight_idx=indices)
    gate, _ = stim._scheduled_tensor_gate(_tensor(0.2), dt=0.1)
    selected = source.index_select(0, indices).to(DTYPE)
    if kind == "time":
        expected = (1 - ((selected - 0.2) / 0.1).abs()).clamp(0, 1)
        coefficients = _tensor([1e7, 0.1, -1e7])
    else:
        expected = selected
        coefficients = _tensor([1e8, 1, -1e8])
    torch.testing.assert_close(gate, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad((gate * coefficients).sum(), source)[0]
    expected_grad = torch.autograd.grad((expected * coefficients).sum(), source)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert actual_grad.item() == 0


def test_native_netcon_keeps_original_mixed_dtype_time_reference_gradient_order():
    connection = _native_netcon(generators=3)
    connection.set_diff_config(diff_scheduled_times=True, sched_width=1.0)
    source = torch.tensor([0.15], dtype=torch.float32, requires_grad=True)
    indices = torch.zeros(3, dtype=torch.long)
    connection.bind_time_source(source)
    connection.schedule_time_ref(con_indices=[0, 1, 2], time_idx=indices)
    gate, _ = connection._scheduled_gate_this_step(torch.tensor(2), use_tri_kernel=True)
    selected = source.index_select(0, indices).to(DTYPE)
    expected = (1 - (selected / 0.1 - 2).abs() / 1.000001).clamp(0, 1)
    coefficients = _tensor([1e7, 0.1, -1e7])
    torch.testing.assert_close(gate, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad((gate * coefficients).sum(), source)[0]
    expected_grad = torch.autograd.grad((expected * coefficients).sum(), source)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert actual_grad.item() == 0


def test_netcon_reference_weights_keep_native_dtype_rejection_policy():
    connection = _native_netcon(torch.float32)
    connection.bind_weight_source(_tensor([2]))
    connection.schedule_ref(con_indices=[0], times_ms=[0.2], weight_idx=[0])
    with pytest.raises(RuntimeError, match="dtype"):
        connection._scheduled_gate_this_step(torch.tensor(2), use_tri_kernel=False)


def test_netcon_schedule_revision_tracks_public_topology_and_binding_mutations():
    connection = _native_netcon()
    # Older serialized connections can predate the schedule-specific revision.
    del connection._schedule_config_generation
    connection.initialize(reinit_weights=False, reinit_delays=False)
    assert connection._schedule_config_generation == 0
    operations = (
        lambda: connection.bind_weight_source(_tensor([2])),
        lambda: connection.bind_time_source(_tensor([0.2])),
        lambda: connection.schedule(con_indices=[0], times_ms=[0.2]),
        lambda: connection.schedule_ref(
            con_indices=[0], times_ms=[0.2], weight_idx=[0]
        ),
        lambda: connection.schedule_time_ref(con_indices=[0], time_idx=[0]),
        lambda: connection.clear_schedule(),
        lambda: connection.clear_schedule(),
    )
    for operation in operations:
        previous = connection._schedule_config_generation
        operation()
        assert connection._schedule_config_generation == previous + 1
    previous = connection._schedule_config_generation
    connection.global_step.fill_(3)
    connection.schedule(con_indices=[0], times_ms=[0.0])
    assert connection._schedule_config_generation == previous


def test_netstim_matching_tensor_restore_preserves_revision_changed_config_invalidates():
    stim = NetStim(dtype=DTYPE).set_dt(0.1).initialize()
    empty = stim.state_dict_for_checkpoint()
    stim.schedule(0, _tensor(0.2), weight=2)
    checkpoint = stim.state_dict_for_checkpoint()
    previous = stim._renewal_config_generation
    for _ in range(2):
        stim.restore_dict_from_checkpoint(checkpoint)
        assert stim._renewal_config_generation == previous
    changed = dict(checkpoint, sched_weight=_tensor([3]))
    stim.restore_dict_from_checkpoint(changed)
    assert stim._renewal_config_generation == previous + 1
    stim.restore_dict_from_checkpoint(empty)
    assert stim._renewal_config_generation == previous + 2
