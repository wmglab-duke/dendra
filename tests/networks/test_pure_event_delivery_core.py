"""One fixed-delay transport core serves functional and owned NetCon state."""

from types import MethodType

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import dendra as dn
from dendra.models.mod import expsyn
from dendra.models.networks._event_delivery import (
    _clear_dense_slot,
    _enqueue_dense_values,
    _fixed_delay_indices,
    connection_event_counts,
    dense_due,
    dense_fixed_delay_transition,
    gather_source_gates,
)

DTYPE = torch.float64
DT = 0.1


def _topology():
    return {
        "post_idx": torch.tensor([1, 1, 2, 0, 1]),
        "delay_steps": torch.tensor([1, 2, 3, 1, 4]),
        "depth": 4,
        "width": 3,
    }


def _oracle(queue, slot, weights, gates, topology):
    # Independent scalar addressing, including fan-in and a full-ring delay.
    rows = []
    for row in range(topology["depth"]):
        columns = []
        for column in range(topology["width"]):
            value = queue.new_zeros(()) if row == slot else queue[row, column]
            for edge, (target, delay) in enumerate(
                zip(topology["post_idx"].tolist(), topology["delay_steps"].tolist())
            ):
                if target == column and (slot + delay) % topology["depth"] == row:
                    value = value + weights[edge] * gates[edge]
            columns.append(value)
        rows.append(torch.stack(columns))
    return torch.stack(rows)


def test_pure_transition_matches_independent_routing_and_keeps_inputs_unchanged():
    topology = _topology()
    queue = torch.arange(12, dtype=DTYPE).reshape(4, 3)
    events = torch.arange(20, dtype=torch.int32).reshape(4, 5)
    slot, global_step = torch.tensor([2]), torch.tensor([9])
    weights = torch.tensor([0.5, 1.0, -2.0, 3.0, 0.0], dtype=DTYPE)
    gates = torch.tensor([-0.2, 0.4, 0.0, 1.0, 2.0], dtype=DTYPE)
    counts = connection_event_counts(
        gates, torch.tensor([1, 0, 2, 0, 1], dtype=torch.int32)
    )
    originals = [value.clone() for value in (queue, events, slot, global_step)]
    due = dense_due(queue, slot, events)
    actual = dense_fixed_delay_transition(
        queue,
        slot,
        global_step,
        weights,
        gates,
        **topology,
        event_queue=events,
        event_counts=counts,
    )
    torch.testing.assert_close(
        actual.queue, _oracle(queue, 2, weights, gates, topology)
    )
    expected_events = events.clone()
    expected_events[2] = 0
    for edge, delay in enumerate(topology["delay_steps"].tolist()):
        expected_events[(2 + delay) % 4, edge] += counts[edge]
    torch.testing.assert_close(actual.event_queue, expected_events)
    torch.testing.assert_close(due.payload, queue[2])
    torch.testing.assert_close(due.event_counts, events[2])
    assert actual.slot.tolist() == [3]
    assert actual.global_step.tolist() == [10]
    for value, original in zip((queue, events, slot, global_step), originals):
        torch.testing.assert_close(value, original)
    assert actual.queue.data_ptr() != queue.data_ptr()
    _clear_dense_slot(queue, slot, inplace=True)
    torch.testing.assert_close(due.payload, originals[0][2])


@pytest.mark.parametrize("depth", (1, 4))
def test_pending_deliveries_retain_enqueue_weights_and_gate_gradients(depth):
    queue = torch.zeros(depth, 2, dtype=DTYPE)
    slot = global_step = torch.zeros(1, dtype=torch.long)
    post_idx = torch.tensor([0, 0, 1])
    delay_steps = torch.tensor([1, depth, 1])
    first_weights = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE, requires_grad=True)
    first_gates = torch.tensor([0.25, -0.5, 2.0], dtype=DTYPE, requires_grad=True)
    second_weights = torch.full_like(first_weights, 17.0, requires_grad=True)
    received = []
    for step in range(depth + 1):
        received.append(dense_due(queue, slot).payload)
        transition = dense_fixed_delay_transition(
            queue,
            slot,
            global_step,
            first_weights if step == 0 else second_weights,
            first_gates if step == 0 else torch.zeros_like(first_gates),
            post_idx=post_idx,
            delay_steps=delay_steps,
            depth=depth,
            width=2,
        )
        queue, slot, global_step = transition[:3]
    actual = torch.stack(received).sum()
    expected = (first_weights * first_gates).sum()
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(
        actual, (first_weights, first_gates, second_weights), materialize_grads=True
    )
    torch.testing.assert_close(gradients[0], first_gates)
    torch.testing.assert_close(gradients[1], first_weights)
    torch.testing.assert_close(gradients[2], torch.zeros_like(second_weights))


@pytest.mark.parametrize("lanes", (0, 3))
@pytest.mark.parametrize("mapped", ("weights", "gates", "state"))
def test_pure_transition_vmaps_mapped_inputs_with_unmapped_queue(lanes, mapped):
    topology = _topology()
    queue = torch.arange(12, dtype=DTYPE).reshape(4, 3)
    weights = torch.linspace(0.1, 0.5, 5, dtype=DTYPE)
    gates = torch.linspace(-0.5, 0.5, 5, dtype=DTYPE)
    slot, global_step = torch.tensor([1]), torch.tensor([7])
    if mapped == "state":
        operand = queue.expand(lanes, *queue.shape).clone()
    else:
        base = weights if mapped == "weights" else gates
        operand = base.expand(lanes, *base.shape).clone()

    def transition(value):
        result = dense_fixed_delay_transition(
            value if mapped == "state" else queue,
            slot,
            global_step,
            value if mapped == "weights" else weights,
            value if mapped == "gates" else gates,
            **topology,
        )
        return result.queue

    actual = torch.vmap(transition)(operand)
    if lanes:
        expected = torch.stack([transition(value) for value in operand])
    else:
        expected = queue.new_empty((0, *queue.shape))
    torch.testing.assert_close(actual, expected)
    reverse = torch.func.jacrev(transition)(operand.new_ones(operand.shape[1:]))
    forward = torch.func.jacfwd(transition)(operand.new_ones(operand.shape[1:]))
    torch.testing.assert_close(forward, reverse)


@pytest.mark.parametrize("lanes", (0, 3))
def test_mapped_slots_and_tracking_keep_hidden_axes(lanes):
    topology = _topology()
    queue = torch.arange(12, dtype=DTYPE).reshape(4, 3)
    events = torch.arange(20, dtype=torch.int32).reshape(4, 5)
    weights = torch.ones(5, dtype=DTYPE)
    gates = torch.tensor([-1.0, 0.0, 0.3, 1.0, 2.0], dtype=DTYPE)
    slots = torch.arange(lanes).reshape(lanes, 1)

    def transition(slot):
        due = dense_due(queue, slot, events)
        result = dense_fixed_delay_transition(
            queue,
            slot,
            slot,
            weights,
            gates,
            **topology,
            event_queue=events,
            event_counts=connection_event_counts(gates),
        )
        return due.payload, result.queue, result.slot, result.event_queue

    actual = torch.vmap(transition)(slots)
    if lanes:
        expected = tuple(
            torch.stack(items) for items in zip(*(transition(s) for s in slots))
        )
    else:
        expected = (
            queue.new_empty(0, 3),
            queue.new_empty(0, 4, 3),
            slots.new_empty(0, 1),
            events.new_empty(0, 4, 5),
        )
    for value, oracle in zip(actual, expected):
        torch.testing.assert_close(value, oracle)


def test_source_gate_gather_retains_values_and_fanout_gradients():
    source = torch.tensor([[-0.5, 0.25], [2.0, 0.0]], dtype=DTYPE, requires_grad=True)
    actual = gather_source_gates(source, torch.tensor([2, 0, 2, 1]))
    torch.testing.assert_close(actual, source.new_tensor([2.0, -0.5, 2.0, 0.25]))
    gradient = torch.autograd.grad(actual.sum(), source)[0]
    torch.testing.assert_close(gradient, source.new_tensor([[1.0, 1.0], [2.0, 0.0]]))


def test_empty_connection_set_still_delivers_and_clears_due_state():
    queue = torch.tensor([[1.0, -2.0, 0.0]], dtype=DTYPE)
    slot = torch.zeros(1, dtype=torch.long)
    empty_index = torch.empty(0, dtype=torch.long)
    empty_value = torch.empty(0, dtype=DTYPE)
    events = torch.empty(1, 0, dtype=torch.int32)
    due = dense_due(queue, slot, events)
    result = dense_fixed_delay_transition(
        queue,
        slot,
        slot,
        empty_value,
        gather_source_gates(torch.ones(3, dtype=DTYPE), empty_index),
        post_idx=empty_index,
        delay_steps=empty_index,
        depth=1,
        width=3,
        event_queue=events,
        event_counts=torch.empty(0, dtype=torch.int32),
    )
    torch.testing.assert_close(due.payload, queue[0])
    torch.testing.assert_close(result.queue, torch.zeros_like(queue))
    assert due.event_counts.shape == (0,)
    assert result.event_queue.shape == (1, 0)
    assert result.slot.tolist() == [0]
    assert result.global_step.tolist() == [1]


def test_fullgraph_compiled_transition_matches_routing_and_gradients():
    topology = _topology()
    queue = torch.arange(12, dtype=DTYPE).reshape(4, 3)
    slot = torch.tensor([2])
    weights = torch.linspace(0.1, 0.5, 5, dtype=DTYPE, requires_grad=True)
    gates = torch.linspace(-0.5, 0.5, 5, dtype=DTYPE, requires_grad=True)

    def transition(weights, gates):
        return dense_fixed_delay_transition(
            queue, slot, slot, weights, gates, **topology
        ).queue

    actual = torch.compile(transition, fullgraph=True, backend="eager")(weights, gates)
    expected = _oracle(queue, 2, weights, gates, topology)
    torch.testing.assert_close(actual, expected)
    for value, oracle in zip(
        torch.autograd.grad(actual.sum(), (weights, gates), retain_graph=True),
        torch.autograd.grad(expected.sum(), (weights, gates)),
    ):
        torch.testing.assert_close(value, oracle)


def _netcon(*, training, delays=(0.0, 0.1, 0.2)):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu"):
        pre = dn.Population(N=1, C=3, dtype=DTYPE)
        post = dn.Population(N=1, C=3, dtype=DTYPE)
        post.insert(expsyn.rename("syn"))
        net = dn.Network(
            {"pre": pre, "post": post},
            track_netcon_events=True,
            netcon_train_backend="dense",
        )
        net.connect_one_to_one(
            pre[:],
            post[:],
            post.mech.syn,
            threshold=None,
            weight=torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE),
            delay=torch.tensor(delays, dtype=DTYPE),
        )
        net.train(training)
        net.build(DT)
        connection = next(iter(net.synapses.values()))
        connection.set_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=False,
            diff_scheduled_times=False,
        )
        net.initialize(DT)
    return net, connection


@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("delays", ((0.0, 0.1, 0.2), (0.1, 0.1, 0.1)))
def test_netcon_adapters_match_shared_core_with_schedules_and_tracking(
    training, delays
):
    net, connection = _netcon(training=training, delays=delays)
    connection.schedule(con_indices=[0, 2], times_ms=[0.0, 0.1], weight=2.5)
    queue = connection.delivery_buffer.clone()
    events = connection.event_queue.clone()
    slot, global_step = (
        connection.current_time_step.clone(),
        connection.global_step.clone(),
    )
    owned_queue = connection.delivery_buffer
    for step in range(5):
        source = (
            torch.tensor([[0.25, -0.5, 1.0]], dtype=DTYPE)
            if step < 2
            else torch.zeros_like(net.pre.v)
        )
        net.pre.v = source
        due = dense_due(queue, slot, events)
        scheduled, scheduled_counts = connection._scheduled_gate_this_step(
            global_step, use_tri_kernel=False
        )
        gates = gather_source_gates(source, connection.pre_idx)
        expected = dense_fixed_delay_transition(
            queue,
            slot,
            global_step,
            connection.weight(),
            gates + scheduled,
            post_idx=connection.post_idx,
            delay_steps=connection.inference_delay_steps,
            depth=connection.max_delay_steps,
            width=connection._syn_numel,
            event_queue=events,
            event_counts=connection_event_counts(gates, scheduled_counts),
        )
        before_g = connection.syn.g
        connection.advance()
        torch.testing.assert_close(
            connection.syn.g, before_g + due.payload.reshape_as(before_g)
        )
        torch.testing.assert_close(connection.events, due.event_counts)
        torch.testing.assert_close(connection.delivery_buffer, expected.queue)
        torch.testing.assert_close(connection.event_queue, expected.event_queue)
        torch.testing.assert_close(connection.current_time_step, expected.slot)
        torch.testing.assert_close(connection.global_step, expected.global_step)
        if not training:
            assert connection.delivery_buffer is owned_queue
        queue, slot, global_step, events = expected


@pytest.mark.parametrize("training", (False, True))
def test_receiver_runs_before_detection_even_for_zero_payload(training):
    net, connection = _netcon(training=training)
    calls = []

    def receive(synapse, payload, netcon):
        calls.append(payload.clone())
        net.pre.v = torch.full_like(net.pre.v, 0.25)

    connection.syn.net_receive = MethodType(receive, connection.syn)
    connection.advance()
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], torch.zeros_like(calls[0]))
    expected = dense_fixed_delay_transition(
        torch.zeros_like(connection.delivery_buffer),
        torch.zeros_like(connection.current_time_step),
        torch.zeros_like(connection.global_step),
        connection.weight(),
        torch.full((3,), 0.25, dtype=DTYPE),
        post_idx=connection.post_idx,
        delay_steps=connection.inference_delay_steps,
        depth=connection.max_delay_steps,
        width=connection._syn_numel,
    )
    torch.testing.assert_close(connection.delivery_buffer, expected.queue)


def test_owned_buffer_primitives_preserve_fast_routes_and_ring_identity():
    topology = _topology()
    queue = torch.arange(12, dtype=DTYPE).reshape(4, 3)
    slot = torch.tensor([1])
    weights = torch.linspace(0.1, 0.5, 5, dtype=DTYPE)
    gates = torch.linspace(-0.5, 0.5, 5, dtype=DTYPE)
    expected = _oracle(queue, 1, weights, gates, topology)
    offsets = topology["delay_steps"] * 3 + topology["post_idx"]
    destination = _fixed_delay_indices(slot, **topology, flat_offsets=offsets)
    identity = queue.data_ptr()
    _clear_dense_slot(queue, slot, inplace=True)
    _enqueue_dense_values(queue, destination, weights * gates, inplace=True)
    assert queue.data_ptr() == identity
    torch.testing.assert_close(queue, expected)


def test_noncontiguous_queue_is_purely_supported_but_not_silently_donated():
    topology = _topology()
    queue = torch.arange(12, dtype=DTYPE).reshape(3, 4).transpose(0, 1)
    assert not queue.is_contiguous()
    original = queue.clone()
    slot = torch.tensor([1])
    weights, gates = torch.ones(5, dtype=DTYPE), torch.full((5,), 0.25, dtype=DTYPE)
    actual = dense_fixed_delay_transition(queue, slot, slot, weights, gates, **topology)
    torch.testing.assert_close(
        actual.queue, _oracle(queue, 1, weights, gates, topology)
    )
    destination = _fixed_delay_indices(slot, **topology)
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        _enqueue_dense_values(queue, destination, weights * gates, inplace=True)
    torch.testing.assert_close(queue, original)


@pytest.mark.parametrize("noncontiguous", (False, True))
def test_clone_once_policy_preserves_inputs_values_and_all_gradients(noncontiguous):
    topology = _topology()
    queue = (
        torch.randn(3, 4, dtype=DTYPE).transpose(0, 1)
        if noncontiguous
        else torch.randn(4, 3, dtype=DTYPE)
    ).requires_grad_()
    weights = torch.randn(5, dtype=DTYPE, requires_grad=True)
    gates = torch.randn(5, dtype=DTYPE, requires_grad=True)
    events = torch.arange(20, dtype=torch.int32)
    events = (
        events.reshape(5, 4).transpose(0, 1) if noncontiguous else events.reshape(4, 5)
    )
    originals = queue.detach().clone(), events.clone()
    slot = torch.tensor([2])
    options = {
        **topology,
        "event_queue": events,
        "event_counts": connection_event_counts(gates),
    }
    expected = dense_fixed_delay_transition(
        queue, slot, slot, weights, gates, **options
    )
    actual = dense_fixed_delay_transition(
        queue, slot, slot, weights, gates, **options, _clone_once=True
    )
    for value, oracle in zip(actual, expected):
        torch.testing.assert_close(value, oracle)
    for gradient, oracle in zip(
        torch.autograd.grad(
            actual.queue.sum(), (queue, weights, gates), retain_graph=True
        ),
        torch.autograd.grad(expected.queue.sum(), (queue, weights, gates)),
    ):
        torch.testing.assert_close(gradient, oracle)
    torch.testing.assert_close(queue, originals[0])
    torch.testing.assert_close(events, originals[1])
    assert actual.queue.data_ptr() != queue.data_ptr()
    assert actual.event_queue.data_ptr() != events.data_ptr()
    assert actual.queue.is_contiguous()
    assert actual.event_queue.is_contiguous()


def test_native_training_allocates_one_new_storage_per_ring():
    net, connection = _netcon(training=True)
    net.pre.v = torch.full_like(net.pre.v, 0.25)
    queue, events = connection.delivery_buffer, connection.event_queue
    original_queue, original_events = queue.clone(), events.clone()
    shape = tuple(queue.shape)
    known = {queue.untyped_storage().data_ptr(), events.untyped_storage().data_ptr()}
    allocations = {queue.dtype: set(), events.dtype: set()}

    class RingAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **({} if kwargs is None else kwargs))
            for value in tree_leaves(result):
                if not torch.is_tensor(value) or tuple(value.shape) != shape:
                    continue
                pointer = value.untyped_storage().data_ptr()
                if value.dtype in allocations and pointer not in known:
                    allocations[value.dtype].add(pointer)
            return result

    with RingAllocations():
        connection.advance()
    assert allocations[queue.dtype] == {
        connection.delivery_buffer.untyped_storage().data_ptr()
    }
    assert allocations[events.dtype] == {
        connection.event_queue.untyped_storage().data_ptr()
    }
    torch.testing.assert_close(queue, original_queue)
    torch.testing.assert_close(events, original_events)
