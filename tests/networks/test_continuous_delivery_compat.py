"""Freeze native continuous routing, source phases, and pending-state semantics."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import graded_syn

DT = 0.1
PARTITIONS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.2, 0.2, 0.2, 0.2),
    (0.1, 0.2, 0.3, 0.2),
    (0.0, 0.2, 0.0, 0.2),
    (0.0, 0.1, 0.0, 0.3),
)


def _network(delays, *, dtype=torch.float64, transform=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=dtype):
        pre = dn.SingleCompartment(N=2, C=1, v_init=0.0, dtype=dtype)
        post = dn.SingleCompartment(N=2, C=1, v_init=0.0, dtype=dtype)
        post.insert(graded_syn, g_scale=0.0)
        network = dn.Network({"pre": pre, "post": post})
        network.connect_continuous(
            pre[:],
            post[:],
            post.mech.graded_syn,
            conn_spec="all_to_all",
            weight=torch.tensor([0.5, -1.0, 1.5, 0.0], dtype=dtype),
            delay=torch.tensor(delays, dtype=dtype),
            transform=transform,
        )
        network.initialize(DT, max_delay_ms=0.5)
    return network, next(iter(network.continuous_synapses.values()))


def _previous_step(queue, presence, slot, clock, weights, source, connection):
    """Independent copy of the previous indexed native numerical phases.

    Python counters avoid aliasing tensor indices saved by autograd. All
    projection and delay accumulation below uses the original index_add
    operations, independently of the new shared helper implementation.
    """
    queue, presence = queue.clone(), presence.clone()
    pre_idx, post_idx = connection.pre_idx, connection.post_idx
    delays = connection.delay_steps
    immediate = delays == 0
    delayed = ~immediate
    if bool(delayed.any()):
        payload, mask = queue[slot].clone(), presence[slot].clone()
        queue[slot] = 0
        presence[slot] = False
    else:
        payload = queue.new_zeros(connection._syn_numel)
        mask = presence.new_zeros(connection._syn_numel)
    selected = source.reshape(-1).index_select(0, pre_idx)
    if connection.transform is not None:
        selected = connection.transform(selected)
    weighted = weights * selected
    if bool(immediate.any()):
        immediate_payload = queue.new_zeros(connection._syn_numel)
        immediate_payload.index_add_(0, post_idx[immediate], weighted[immediate])
        payload = payload + immediate_payload
        mask[post_idx[immediate]] = True
    if bool(delayed.any()):
        destination = ((slot + delays[delayed]) % queue.shape[0]) * queue.shape[
            1
        ] + post_idx[delayed]
        queue.view(-1).index_add_(0, destination, weighted[delayed])
        presence.view(-1).index_fill_(0, destination, True)
    next_slot = slot if not bool(delayed.any()) else (slot + 1) % queue.shape[0]
    return queue, presence, next_slot, clock + 1, payload, mask


@pytest.mark.parametrize("delays", PARTITIONS)
@pytest.mark.parametrize("transform", (None, torch.tanh))
def test_native_matches_previous_indexed_queues_values_and_all_gradients(
    delays, transform
):
    network, connection = _network(delays, transform=transform)
    weights = connection.weight().detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    connection.weight.w = weights
    gates = torch.tensor(
        [[0.0, -2.0], [1.0, 0.0], [-0.5, 3.0], [0.0, 0.0], [0.2, 0.8]],
        dtype=connection.dtype,
        requires_grad=True,
    )
    reference_gates = gates.detach().clone().requires_grad_()
    initial = torch.arange(connection.delivery_buffer.numel(), dtype=connection.dtype)
    initial = initial.reshape_as(connection.delivery_buffer) / 16
    initial.requires_grad_()
    reference_initial = initial.detach().clone().requires_grad_()
    reference_queue = reference_initial
    connection.delivery_buffer = initial
    connection.delivery_mask = initial.detach() != 0
    reference_presence = connection.delivery_mask.clone()
    connection.current_time_step = torch.tensor([2])
    slot, clock = 2, 0
    observations = []
    connection.syn.continuous_receive = (
        lambda value, *args, mask, **kwargs: observations.append(
            (value.flatten(), mask.flatten())
        )
    )
    expected_payloads = []
    for source, reference_source in zip(gates, reference_gates, strict=True):
        before = connection.delivery_buffer
        saved = before.detach().clone()
        network.pre.v = source.reshape_as(network.pre.v)
        connection.advance()
        queue, reference_presence, slot, clock, payload, mask = _previous_step(
            reference_queue,
            reference_presence,
            slot,
            clock,
            reference_weights,
            reference_source,
            connection,
        )
        reference_queue = queue
        torch.testing.assert_close(
            connection.delivery_buffer, reference_queue, rtol=0, atol=0
        )
        torch.testing.assert_close(connection.delivery_mask, reference_presence)
        torch.testing.assert_close(observations[-1][0], payload, rtol=0, atol=0)
        torch.testing.assert_close(observations[-1][1], mask)
        assert int(connection.current_time_step) == slot
        assert int(connection.global_step) == clock
        # Later training steps cannot overwrite a retained checkpoint queue.
        torch.testing.assert_close(before, saved, rtol=0, atol=0)
        expected_payloads.append(payload)
    actual_loss = (
        sum(value.square().sum() for value, _mask in observations)
        + connection.delivery_buffer.square().sum()
    )
    reference_loss = (
        sum(value.square().sum() for value in expected_payloads)
        + reference_queue.square().sum()
    )
    actual = torch.autograd.grad(actual_loss, (weights, gates, initial))
    expected = torch.autograd.grad(
        reference_loss, (reference_weights, reference_gates, reference_initial)
    )
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("delays", PARTITIONS)
def test_receive_and_source_observation_follow_specialized_native_order(delays):
    network, connection = _network(delays)
    order = []
    original_getter = connection.get_pre_var

    def sample(pre):
        order.append("sample")
        return original_getter(pre)

    def receive(value, *args, mask, **kwargs):
        order.append("receive")
        network.pre.v = network.pre.v + 10

    connection.get_pre_var = sample
    connection.syn.continuous_receive = receive
    network.pre.v = torch.ones_like(network.pre.v)
    connection.advance()
    all_delayed = all(delay != 0 for delay in delays)
    assert order == (["receive", "sample"] if all_delayed else ["sample", "receive"])
    expected_source = network.pre.v.new_full(
        network.pre.v.shape, 11 if all_delayed else 1
    )
    weighted = connection.weight() * expected_source.flatten().index_select(
        0, connection.pre_idx
    )
    if any(delays):
        expected = torch.zeros_like(connection.delivery_buffer)
        for edge, delay in enumerate(connection.delay_steps):
            if int(delay):
                expected[int(delay), int(connection.post_idx[edge])] += weighted[edge]
        torch.testing.assert_close(connection.delivery_buffer, expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("delay", (0.0, 0.2))
def test_native_low_precision_collision_accumulation_matches_previous_index_add(
    dtype, delay
):
    network, connection = _network((delay,) * 4, dtype=dtype)
    connection.post_idx.zero_()
    connection._rebuild_delay_metadata()
    connection._configure_advance_impl()
    weights = torch.tensor([2048.0, 1.0, -2048.0, 1.0], dtype=dtype)
    connection.weight.w = weights
    network.pre.v = torch.ones_like(network.pre.v)
    observations = []
    connection.syn.continuous_receive = (
        lambda value, *args, mask, **kwargs: observations.append(
            value.flatten().clone()
        )
    )
    expected = torch.zeros(connection._syn_numel, dtype=dtype)
    expected.index_add_(0, connection.post_idx, weights)
    with torch.no_grad():
        for _ in range(int(connection.delay_steps.max()) + 1):
            connection.advance()
    torch.testing.assert_close(observations[-1], expected, rtol=0, atol=0)


def test_runtime_checkpoint_preserves_pending_zero_and_weight_source_graph():
    network, connection = _network((0.0, 0.2, 0.0, 0.2))
    weights = connection.weight().detach().clone().requires_grad_()
    source = torch.tensor([[0.0], [-2.0]], dtype=connection.dtype, requires_grad=True)
    connection.weight.w = weights
    network.pre.v = source
    connection.advance()
    checkpoint = connection.state_dict_for_checkpoint()
    saved = {name: value.detach().clone() for name, value in checkpoint.items()}
    assert checkpoint["delivery_buffer"] is connection.delivery_buffer
    _, resumed = _network((0.0, 0.2, 0.0, 0.2))
    resumed.restore_dict_from_checkpoint(checkpoint)
    observations, resumed_observations = [], []
    connection.syn.continuous_receive = (
        lambda value, *args, mask, **kwargs: observations.append(
            (value.flatten(), mask.flatten())
        )
    )
    resumed.syn.continuous_receive = (
        lambda value, *args, mask, **kwargs: resumed_observations.append(
            (value.flatten(), mask.flatten())
        )
    )
    for _ in range(3):
        network.pre.v = torch.zeros_like(network.pre.v)
        resumed.pre.v = torch.zeros_like(resumed.pre.v)
        connection.advance()
        resumed.advance()
    for actual, expected in zip(observations, resumed_observations, strict=True):
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right)
    assert any(bool(mask.any()) for _value, mask in observations)
    assert any(bool(((value == 0) & mask).any()) for value, mask in observations)
    actual = torch.autograd.grad(
        sum(value.sum() for value, _ in observations),
        (weights, source),
        retain_graph=True,
    )
    expected = torch.autograd.grad(
        sum(value.sum() for value, _ in resumed_observations), (weights, source)
    )
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
    for name, value in checkpoint.items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_inference_reuses_owned_rings_and_training_rebinds_them():
    network, connection = _network((0.0, 0.1, 0.0, 0.3))
    queue, mask = connection.delivery_buffer, connection.delivery_mask
    slot, clock = connection.current_time_step, connection.global_step
    network.pre.v = torch.zeros_like(network.pre.v)
    with torch.no_grad():
        connection.advance()
    assert connection.delivery_buffer is queue and connection.delivery_mask is mask
    assert connection.current_time_step is slot and connection.global_step is clock
    connection.advance()
    assert (
        connection.delivery_buffer is not queue and connection.delivery_mask is not mask
    )
    assert (
        connection.current_time_step is not slot and connection.global_step is not clock
    )


@pytest.mark.parametrize("grad_enabled", (False, True))
def test_empty_projection_advances_only_counters_without_reading_or_receiving(
    grad_enabled,
):
    _, connection = _network((0.1, 0.2, 0.1, 0.2))
    # The network builder omits empty specifications; exercise the canonical
    # empty specialization directly after rebuilding its empty topology.
    connection.pre_idx = torch.empty(0, dtype=torch.long)
    connection.post_idx = torch.empty(0, dtype=torch.long)
    connection.delay_steps = torch.empty(0, dtype=torch.long)
    connection._rebuild_delay_metadata()
    connection._configure_advance_impl()
    connection.delivery_buffer.fill_(2)
    connection.delivery_mask.fill_(True)
    queue, mask = connection.delivery_buffer, connection.delivery_mask

    def forbidden(*args, **kwargs):
        raise AssertionError("empty projections must not sample or receive")

    connection.get_pre_var = forbidden
    connection.syn.continuous_receive = forbidden
    with torch.set_grad_enabled(grad_enabled):
        connection.advance()
    assert connection.delivery_buffer is queue and connection.delivery_mask is mask
    assert torch.all(queue == 2) and mask.all()
    assert int(connection.current_time_step) == int(connection.global_step) == 1


def test_inference_clones_restored_noncontiguous_rings_before_owned_enqueue():
    network, connection = _network((0.0, 0.1, 0.0, 0.3))
    queue = torch.ones_like(connection.delivery_buffer).T.contiguous().T
    mask = torch.ones_like(queue, dtype=torch.bool).T.contiguous().T
    checkpoint = connection.state_dict_for_checkpoint() | {
        "delivery_buffer": queue,
        "delivery_mask": mask,
    }
    connection.restore_dict_from_checkpoint(checkpoint)
    network.pre.v = torch.zeros_like(network.pre.v)
    with torch.no_grad():
        connection.advance()
    assert connection.delivery_buffer.is_contiguous()
    assert connection.delivery_mask.is_contiguous()
    assert torch.all(queue == 1) and mask.all()


def test_lifecycle_generation_changes_only_on_structure_initialization_or_moves():
    _, connection = _network((0.0, 0.1, 0.0, 0.3))
    assert connection._continuous_delivery_initialized
    generation = connection._continuous_delivery_generation
    connection.advance()
    checkpoint = connection.state_dict_for_checkpoint()
    connection.restore_dict_from_checkpoint(checkpoint)
    assert connection._continuous_delivery_generation == generation
    connection.initialize(reinit_weights=False, reinit_delays=False)
    assert connection._continuous_delivery_generation > generation
    generation = connection._continuous_delivery_generation
    connection.to(dtype=torch.float32)
    assert connection._continuous_delivery_generation > generation
