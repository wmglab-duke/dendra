"""Compact projections share pending transport without losing queued semantics."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn
from tests.func.test_netcon import _clone, _close

DT = 0.1
DTYPE = torch.float64
pytestmark = pytest.mark.cpu


def _netcon(*, backend="sparse_calendar", threshold=None, batch=None, scheduled=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
        pre = dn.Population(N=1, C=3, dtype=DTYPE)
        pre.register_buffer("gate", torch.zeros(pre.shape, dtype=DTYPE))
        post = dn.Population(N=1, C=3, dtype=DTYPE)
        post.insert(expsyn.rename("syn"))
        network = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend=backend,
            track_netcon_events=backend == "sparse_calendar",
        )
        network.connect_one_to_one(
            pre[:],
            post[:],
            post.mech.syn,
            pre_var="gate",
            threshold=threshold,
            weight=torch.tensor([0.7, -0.4, 0.0], dtype=DTYPE),
            delay=torch.tensor([DT, 2 * DT, 3 * DT], dtype=DTYPE),
        )
        if batch is not None:
            network.batch(batch)
            pre.gate = torch.zeros(pre.shape, dtype=DTYPE)
        network.eval()
        network.initialize(DT)
        native = next(iter(network.synapses.values()))
        native.weight.w = torch.tensor([0.7, -0.4, 0.0], dtype=DTYPE).repeat(
            native._n_conn // 3
        )
        if scheduled:
            for start in range(0, native._n_conn, 3):
                native.schedule(
                    con_indices=[start, start, start + 1, start + 2, start],
                    times_ms=[0.0, 0.0, DT, 2 * DT, 3 * DT],
                    weight=torch.tensor([1.0, -1.0, 0.5, 1.0, 0.4], dtype=DTYPE),
                )
    assert native._checkpoint_runtime_kind() == backend
    return network, native


def _tape(shape):
    return (
        torch.tensor(
            [
                [1.0, -0.5, 0.2],
                [1.0, 0.4, -0.3],
                [-0.2, 0.6, 0.7],
                [0.3, -0.3, 0.5],
                [-0.4, -0.2, -0.1],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=DTYPE,
        )
        .reshape(8, *([1] * (len(shape) - 1)), 3)
        .expand(8, *shape)
    )


@pytest.mark.parametrize("backend", ("sparse_calendar", "bitpacked_history"))
@pytest.mark.parametrize("threshold", (None, 0.0))
@pytest.mark.parametrize("batch", (None, 2))
def test_compact_schedules_match_native_order_counts_scratch_and_enqueue_weights(
    backend, threshold, batch
):
    _, native = _netcon(backend=backend, threshold=threshold, batch=batch)
    plan, tensors = dn.func.make_functional_netcon(native)
    assert plan.pending_capacity == max(16, 2 * native._n_conn)
    assert tensors.state["pending_values"].shape == (plan.pending_capacity,)
    assert tensors.state["delivery_buffer"].shape == (1, native._syn_numel)
    frozen = _clone(tensors.state)
    state = tensors.state
    weights = tensors.weights
    for index, row in enumerate(_tape(plan.source_shape)):
        if index == 2:
            weights = weights * 3.1
            native.weight.w = weights.detach().clone()
        previous_state = state
        old = _clone(state)
        state, payload = plan.step(
            weights, previous_state, row, schedule=tensors.schedule
        )
        plan.check_overflow(state)
        native.pre.gate = row
        previous = native.syn.g.clone()
        native.advance()
        torch.testing.assert_close(payload, native.syn.g - previous)
        _close(state, plan.extract().state)
        _close(previous_state, old)
    _close(tensors.state, frozen)


@pytest.mark.parametrize("backend", ("sparse_calendar", "bitpacked_history"))
def test_native_pending_checkpoint_import_commit_and_fresh_initialization(backend):
    network, native = _netcon(backend=backend)
    tape = _tape(tuple(native.pre.v.shape))
    for row in tape[:2]:
        native.pre.gate = row
        native.advance()
    assert native._sparse_calendar
    plan, tensors = dn.func.make_functional_netcon(native)
    assert tensors.state["pending_size"].item() > 0
    with pytest.raises(ValueError, match="pending_capacity|capacity"):
        dn.func.make_functional_netcon(native, pending_capacity=0)
    state = tensors.state
    for row in tape[2:5]:
        state, _ = plan.step(tensors.weights, state, row, schedule=tensors.schedule)
    plan.commit_state_(native, state)
    _close(plan.extract().state, state)
    for row in tape[5:]:
        state, payload = plan.step(
            tensors.weights, state, row, schedule=tensors.schedule
        )
        native.pre.gate = row
        previous = native.syn.g.clone()
        native.advance()
        _close(plan.extract().state, state)
        torch.testing.assert_close(payload, native.syn.g - previous)
    reset = plan.initial_state(state, t=1.75)
    network.t.fill_(1.75)
    native.initialize(reinit_weights=False, reinit_delays=False)
    _close(plan.extract().state, reset)
    for name, value in reset.items():
        if name.startswith(("pending_", "event_pending_")):
            assert not value.any()


def test_sparse_event_counts_survive_zero_weights_and_cancelled_payloads():
    _, native = _netcon(scheduled=False)
    native.schedule(con_indices=[0, 1, 2], times_ms=[0.0] * 3, weight=-1.0)
    plan, tensors = dn.func.make_functional_netcon(native)
    row = tensors.weights.new_ones(plan.source_shape)
    state, _ = plan.step(tensors.weights, tensors.state, row, schedule=tensors.schedule)
    # Intrinsic and scheduled gates cancel; event identities remain observable.
    assert not state["pending_size"].any()
    assert state["event_pending_size"].item() == native._n_conn
    delivered = []
    for _ in range(plan._depth):
        state, payload = plan.step(
            tensors.weights, state, torch.zeros_like(row), schedule=tensors.schedule
        )
        assert not payload.any()
        delivered.append(state["events"])
    torch.testing.assert_close(
        torch.stack(delivered).sum(0), torch.full((3,), 2, dtype=torch.int64)
    )


@pytest.mark.parametrize("overflow_channel", ("payload", "events"))
def test_overflow_is_sticky_atomic_and_rejected_before_native_commit(overflow_channel):
    _, native = _netcon(scheduled=False)
    plan, tensors = dn.func.make_functional_netcon(native, pending_capacity=1)
    weights = tensors.weights if overflow_channel == "payload" else tensors.weights * 0
    source = tensors.weights.new_ones(plan.source_shape)
    initial = _clone(tensors.state)
    state, _ = plan.step(weights, tensors.state, source)
    assert state["pending_overflow"].item()
    assert state["event_pending_overflow"].item()
    assert not state["pending_size"].any()
    assert not state["event_pending_size"].any()
    with pytest.raises(RuntimeError, match="pending_capacity"):
        plan.check_overflow(state)
    checkpoint = _clone(native.state_dict_for_checkpoint())
    with pytest.raises(RuntimeError, match="pending_capacity"):
        plan.commit_state_(native, state)
    _close(native.state_dict_for_checkpoint(), checkpoint)
    following, _ = plan.step(weights, state, torch.zeros_like(source))
    assert following["pending_overflow"].item()
    _close(tensors.state, initial)


def test_inactive_infinite_weights_do_not_poison_sparse_gradients():
    _, native = _netcon(scheduled=False)
    plan, tensors = dn.func.make_functional_netcon(native)
    weights = tensors.weights.new_tensor([float("inf"), 1.0, 0.0]).requires_grad_()
    gates = weights.new_zeros(plan.source_shape).requires_grad_()
    state, payload = plan.step(weights, tensors.state, gates)
    gradients = torch.autograd.grad(state["pending_values"].sum(), (weights, gates))
    assert not payload.any()
    assert not state["pending_size"].any()
    for gradient in gradients:
        assert not gradient.any()
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("backend", ("sparse_calendar", "bitpacked_history"))
def test_indexed_schedule_bindings_freeze_enqueued_values_and_delivery_offsets(backend):
    _, native = _netcon(backend=backend, scheduled=False)
    native.bind_weight_source(torch.tensor([2.0], dtype=DTYPE, requires_grad=True))
    native.bind_time_source(torch.tensor([0.0], dtype=DTYPE, requires_grad=True))
    native.schedule_ref(con_indices=[0], times_ms=[0.0], weight_idx=[0])
    native.schedule_time_ref(con_indices=[1], time_idx=[0], weight=0.5)
    plan, tensors = dn.func.make_functional_netcon(native)
    weights = tensors.weights.detach().requires_grad_()
    source = weights.new_zeros(plan.source_shape)
    state, _ = plan.step(weights, tensors.state, source, schedule=tensors.schedule)
    native.pre.gate = source
    native.advance()
    schedule = tensors.schedule._replace(
        amplitude_source=tensors.schedule.amplitude_source * 9,
        time_source=tensors.schedule.time_source + 10,
    )
    with torch.no_grad():
        native._sched_w_source.mul_(9)
        native._sched_t_source.add_(10)
    native.weight.w = (weights * 7).detach()
    total = weights.new_zeros(())
    deliveries = []
    for _ in range(2):
        state, payload = plan.step(weights * 7, state, source, schedule=schedule)
        previous = native.syn.g.clone()
        native.advance()
        torch.testing.assert_close(payload, native.syn.g - previous)
        _close(state, plan.extract().state)
        deliveries.append(payload)
        total = total + payload.sum()
    torch.testing.assert_close(
        deliveries[0].flatten(), weights.new_tensor([1.4, 0.0, 0.0])
    )
    torch.testing.assert_close(
        deliveries[1].flatten(), weights.new_tensor([0.0, -0.2, 0.0])
    )
    weight_gradient, amplitude_gradient = torch.autograd.grad(
        total, (weights, tensors.schedule.amplitude_source)
    )
    torch.testing.assert_close(weight_gradient, weights.new_tensor([2.0, 0.5, 0.0]))
    torch.testing.assert_close(amplitude_gradient, weights.new_tensor([0.7]))


@pytest.mark.parametrize(
    "invalid", ("negative_size", "large_size", "slot", "index", "event_index")
)
def test_invalid_calendar_metadata_rejects_commit_atomically(invalid):
    _, native = _netcon(scheduled=False)
    plan, tensors = dn.func.make_functional_netcon(native)
    state = _clone(tensors.state)
    if invalid in {"negative_size", "large_size"}:
        state["pending_size"].fill_(
            -1 if invalid == "negative_size" else plan.pending_capacity + 1
        )
    elif invalid == "event_index":
        state["event_pending_size"].fill_(1)
        state["event_pending_indices"][0] = native._n_conn
    else:
        state["pending_size"].fill_(1)
        state["pending_slots"][0] = plan._depth if invalid == "slot" else 0
        state["pending_indices"][0] = native._syn_numel if invalid == "index" else 0
    snapshot = _clone(native.state_dict_for_checkpoint())
    with pytest.raises(ValueError, match="Pending calendar"):
        plan.commit_state_(native, state)
    _close(native.state_dict_for_checkpoint(), snapshot)


@pytest.mark.parametrize("backend", ("sparse_calendar", "bitpacked_history"))
@pytest.mark.parametrize(
    "helper",
    ("_append_sparse_payloads", "_pop_sparse_events", "_add_sparse_calendar_to_dense"),
)
def test_custom_pending_adapters_require_a_new_compatible_lowering(backend, helper):
    _, native = _netcon(backend=backend)
    plan, tensors = dn.func.make_functional_netcon(native)
    setattr(native, helper, lambda *args: None)
    with pytest.raises(dn.func.FunctionalizationError, match=helper):
        plan.step(
            tensors.weights, tensors.state, native.pre.gate, schedule=tensors.schedule
        )
    with pytest.raises(dn.func.FunctionalizationError, match=helper):
        dn.func.make_functional_netcon(native)


@pytest.mark.parametrize("backend", ("sparse_calendar", "bitpacked_history"))
def test_fullgraph_calendar_projection_preserves_weight_schedule_and_pending_gradients(
    backend,
):
    with torch_compiler_warning_context():
        _, native = _netcon(backend=backend)
        plan, tensors = dn.func.make_functional_netcon(native)
        initial, _ = plan.step(
            tensors.weights,
            tensors.state,
            tensors.weights.new_ones(plan.source_shape),
            schedule=tensors.schedule,
        )
        initial, _ = plan.step(
            tensors.weights,
            initial,
            tensors.weights.new_ones(plan.source_shape),
            schedule=tensors.schedule,
        )

        def evaluate(weights, amplitudes, pending):
            state = {**initial, "pending_values": pending}
            schedule = tensors.schedule._replace(amplitudes=amplitudes)
            loss = weights.new_zeros(())
            for row in _tape(plan.source_shape)[2:6]:
                state, payload = plan.step(weights, state, row, schedule=schedule)
                loss = loss + payload.square().sum()
            return state, loss

        weights = tensors.weights.detach().requires_grad_()
        amplitudes = tensors.schedule.amplitudes.detach().requires_grad_()
        pending = initial["pending_values"].detach().requires_grad_()
        compiled = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
        actual = compiled(weights, amplitudes, pending)
        expected = evaluate(weights, amplitudes, pending)
        _close(actual, expected)
        targets = (weights, amplitudes, pending)
        gradients = torch.autograd.grad(actual[1], targets)
        reference = torch.autograd.grad(expected[1], targets)
        _close(gradients, reference)
        assert all(value.abs().sum() for value in gradients)


@pytest.mark.parametrize("backend", ("sparse_calendar", "bitpacked_history"))
@pytest.mark.parametrize("lanes", (0, 2))
def test_vmap_calendar_sources_parameters_and_carry(backend, lanes):
    _, native = _netcon(backend=backend)
    plan, tensors = dn.func.make_functional_netcon(native)
    states = pytree.tree_map(
        lambda value: value.unsqueeze(0).repeat(lanes, *([1] * value.ndim)),
        tensors.state,
    )
    weights = tensors.weights.unsqueeze(0).repeat(lanes, 1)
    source = _tape(plan.source_shape)[:lanes]

    def evaluate(weights, state, source):
        state, first = plan.step(weights, state, source, schedule=tensors.schedule)
        state, second = plan.step(
            weights, state, torch.zeros_like(source), schedule=tensors.schedule
        )
        plan.check_overflow(state)
        return state, first + second

    actual = torch.vmap(evaluate)(weights, states, source)
    if lanes:
        expected = [
            evaluate(
                weights[index],
                pytree.tree_map(lambda value: value[index], states),
                source[index],
            )
            for index in range(lanes)
        ]
        _close(actual, pytree.tree_map(lambda *values: torch.stack(values), *expected))
    else:
        assert actual[1].shape == (0, *plan.shape)
        assert actual[0]["pending_values"].shape == (0, plan.pending_capacity)


@pytest.mark.parametrize("backend", ("sparse_calendar", "bitpacked_history"))
@pytest.mark.parametrize("lanes", (0, 2))
def test_host_overflow_check_accepts_vmap_outputs_and_inspects_every_lane(
    backend, lanes
):
    _, native = _netcon(backend=backend, scheduled=False)
    native.schedule(con_indices=[0, 1], times_ms=[0.0, 0.0], weight=1.0)
    plan, tensors = dn.func.make_functional_netcon(native, pending_capacity=1)
    times = tensors.weights.new_tensor([[10.0, 10.0], [0.0, 0.0]])[:lanes]
    source = tensors.weights.new_zeros(plan.source_shape)

    def evaluate(times):
        state, _ = plan.step(
            tensors.weights,
            tensors.state,
            source,
            schedule=tensors.schedule._replace(times=times),
        )
        plan.check_overflow(state)
        return state

    state = torch.vmap(evaluate)(times)
    if lanes:
        torch.testing.assert_close(
            state["pending_overflow"], torch.tensor([[False], [True]])
        )
        with pytest.raises(RuntimeError, match="pending_capacity"):
            plan.check_overflow(state)
        plan.check_overflow(pytree.tree_map(lambda value: value[:1], state))
    else:
        assert state["pending_overflow"].shape == (0, 1)
        plan.check_overflow(state)
