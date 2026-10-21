"""One pure interpolating delay transition serves native and functional state."""

import math
from types import MethodType

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn
from dendra.models.networks._event_delivery import (
    dense_due,
    dense_interpolated_delay_transition,
)

DTYPE = torch.float64
DT = 0.1
pytestmark = pytest.mark.cpu


def _inputs(*, noncontiguous=False, dtype=DTYPE):
    queue = torch.arange(18, dtype=dtype).reshape(3, 6).t() / 10
    if not noncontiguous:
        queue = queue.contiguous()
    return (
        queue.requires_grad_(),
        torch.tensor([4]),
        torch.tensor([17]),
        torch.tensor([0.5, 1.0, -2.0, 3.0, 0.0], dtype=dtype, requires_grad=True),
        torch.tensor([-0.2, 0.4, 0.5, 1.0, 2.0], dtype=dtype, requires_grad=True),
        torch.tensor([0.02, 0.1, 0.23, 0.2, 0.6], dtype=dtype, requires_grad=True),
        torch.tensor(DT, dtype=dtype),
    )


def _options(taps):
    return {
        "post_idx": torch.tensor([1, 1, 2, 0, 1]),
        "depth": 6,
        "width": 3,
        "taps": taps,
        "sigma": 0.5,
    }


def _oracle(queue, slot, weights, gates, delays, dt, options):
    # Scalar connections and row addressing keep this independent of vectorized
    # core routing. Retain tensor expressions so gradient comparisons are real.
    writes = []
    for edge, target in enumerate(options["post_idx"].tolist()):
        position = (delays[edge] / dt).clamp_min(1)
        lower = math.floor(position.detach().item())
        if options["taps"] == 2:
            fraction = position - lower
            offsets, fractions = (lower, lower + 1), (1 - fraction, fraction)
        else:
            offsets = (lower - 1, lower, lower + 1)
            logits = torch.stack(
                [
                    -0.5 * ((position - index) / (options["sigma"] + 1e-6)) ** 2
                    for index in offsets
                ]
            )
            fractions = logits.softmax(0)
        for offset, fraction in zip(offsets, fractions):
            writes.append(
                (
                    (slot + offset) % options["depth"],
                    target,
                    weights[edge] * gates[edge] * fraction,
                )
            )
    rows = []
    for row in range(options["depth"]):
        columns = []
        for target in range(options["width"]):
            value = queue.new_zeros(()) if row == slot else queue[row, target]
            for destination, column, contribution in writes:
                if destination == row and column == target:
                    value = value + contribution
            columns.append(value)
        rows.append(torch.stack(columns))
    return torch.stack(rows)


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("clone_once", (False, True))
@pytest.mark.parametrize("noncontiguous", (False, True))
def test_independent_routing_purity_collision_wrap_and_gradients(
    taps, clone_once, noncontiguous
):
    inputs = _inputs(noncontiguous=noncontiguous)
    queue, slot, step, weights, gates, delays, dt = inputs
    originals = [value.detach().clone() for value in inputs]
    options = _options(taps)
    expected = _oracle(queue, slot.item(), weights, gates, delays, dt, options)
    actual = dense_interpolated_delay_transition(
        *inputs, **options, _clone_once=clone_once
    )
    torch.testing.assert_close(actual.queue, expected)
    assert actual.slot.tolist() == [5]
    assert actual.global_step.tolist() == [18]
    assert actual.event_queue is None
    assert actual.queue.data_ptr() != queue.data_ptr()
    if clone_once:
        assert actual.queue.is_contiguous()
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original)
    # Unequal row coefficients expose delay derivatives; total mass alone does not.
    coefficients = torch.arange(18, dtype=DTYPE).reshape(6, 3)
    targets = queue, weights, gates, delays
    actual_grads = torch.autograd.grad(
        (actual.queue * coefficients).sum(), targets, retain_graph=True
    )
    expected_grads = torch.autograd.grad((expected * coefficients).sum(), targets)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value, reference)
    assert actual_grads[-1][0] == 0  # Physical substep delays are clamped.
    assert (
        actual_grads[-1][1] == 0
    )  # Native clamp boundary uses its saturated derivative.
    assert actual_grads[-1][3] != 0  # Larger integer boundaries retain delay gradients.


@pytest.mark.parametrize("taps", (2, 3))
def test_queued_values_retain_enqueue_time_parameters_and_checkpoint_continuation(taps):
    options = {
        "post_idx": torch.tensor([0]),
        "depth": 6,
        "width": 1,
        "taps": taps,
        "sigma": 0.5,
    }
    queue = torch.zeros(6, 1, dtype=DTYPE)
    slot = step = torch.zeros(1, dtype=torch.long)
    first_weights = torch.tensor([2.0], dtype=DTYPE, requires_grad=True)
    first_delays = torch.tensor([0.23], dtype=DTYPE, requires_grad=True)
    gates = torch.tensor([0.5], dtype=DTYPE, requires_grad=True)
    dt = torch.tensor(DT, dtype=DTYPE)
    first = dense_interpolated_delay_transition(
        queue, slot, step, first_weights, gates, first_delays, dt, **options
    )
    checkpoint = tuple(value.clone() for value in first[:3])
    later_weights = torch.tensor([17.0], dtype=DTYPE, requires_grad=True)
    later_delays = torch.tensor([0.55], dtype=DTYPE, requires_grad=True)

    def continue_from(state):
        pending, cursor, count = state
        received = []
        for _ in range(7):
            received.append(dense_due(pending, cursor).payload)
            transition = dense_interpolated_delay_transition(
                pending,
                cursor,
                count,
                later_weights,
                torch.zeros_like(gates),
                later_delays,
                dt,
                **options,
            )
            pending, cursor, count = transition[:3]
        return torch.stack(received), pending

    actual, pending = continue_from(first[:3])
    restored, restored_pending = continue_from(checkpoint)
    torch.testing.assert_close(actual, restored)
    torch.testing.assert_close(pending, restored_pending)
    torch.testing.assert_close(actual.sum(), (first_weights * gates).sum())
    weighted_loss = (actual.flatten() * torch.arange(1, 8, dtype=DTYPE)).sum()
    gradients = torch.autograd.grad(
        weighted_loss, (first_weights, first_delays, gates, later_weights, later_delays)
    )
    assert all(torch.count_nonzero(value) for value in gradients[:3])
    assert all(torch.count_nonzero(value) == 0 for value in gradients[3:])


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("mapped", ("weights", "gates", "delays", "queue", "slot"))
@pytest.mark.parametrize("lanes", (0, 3))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, DTYPE))
def test_vmap_independently_mapped_inputs_including_empty_lanes(
    taps, mapped, lanes, dtype
):
    inputs = _inputs(dtype=dtype)
    options = _options(taps)
    index = {"queue": 0, "slot": 1, "weights": 3, "gates": 4, "delays": 5}[mapped]
    base = inputs[index]
    operand = base.unsqueeze(0).expand(lanes, *base.shape).clone().detach()
    if mapped == "slot":
        operand = torch.arange(lanes).reshape(lanes, 1)
    else:
        operand.requires_grad_()

    def transition(value):
        arguments = list(inputs)
        arguments[index] = value
        result = dense_interpolated_delay_transition(*arguments, **options)
        return result.queue, result.slot, result.global_step

    actual = torch.vmap(transition)(operand)
    expected = (
        tuple(
            torch.stack(values)
            for values in zip(*(transition(value) for value in operand))
        )
        if lanes
        else (
            inputs[0].new_empty(0, 6, 3),
            inputs[1].new_empty(0, 1),
            inputs[2].new_empty(0, 1),
        )
    )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    if mapped != "slot":
        gradient = torch.autograd.grad(actual[0].sum(), operand)[0]
        assert gradient.shape == operand.shape
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("taps", (2, 3))
def test_forward_and_reverse_jacobians_agree(taps):
    queue, slot, step, weights, gates, delays, dt = _inputs()

    def evaluate(queue, weights, gates, delays):
        return dense_interpolated_delay_transition(
            queue, slot, step, weights, gates, delays, dt, **_options(taps)
        ).queue

    forward = torch.func.jacfwd(evaluate, argnums=(0, 1, 2, 3))(
        queue, weights, gates, delays
    )
    reverse = torch.func.jacrev(evaluate, argnums=(0, 1, 2, 3))(
        queue, weights, gates, delays
    )
    for value, reference in zip(forward, reverse):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_values_and_gradients_match_independent_oracle(taps, backend):
    queue, slot, step, weights, gates, delays, dt = _inputs()
    options = _options(taps)

    def evaluate(queue, weights, gates, delays):
        result = dense_interpolated_delay_transition(
            queue, slot, step, weights, gates, delays, dt, **options
        )
        return result.queue, result.slot, result.global_step

    with torch_compiler_warning_context():
        kwargs = {"options": {"compile_threads": 1}} if backend == "inductor" else {}
        compiled = torch.compile(evaluate, fullgraph=True, backend=backend, **kwargs)
        actual = compiled(queue, weights, gates, delays)
    expected = _oracle(queue, slot.item(), weights, gates, delays, dt, options)
    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual[1], torch.tensor([5]))
    torch.testing.assert_close(actual[2], torch.tensor([18]))
    coefficients = torch.arange(18, dtype=DTYPE).reshape(6, 3)
    targets = queue, weights, gates, delays
    actual_gradients = torch.autograd.grad(
        (actual[0] * coefficients).sum(), targets, retain_graph=True
    )
    expected_gradients = torch.autograd.grad((expected * coefficients).sum(), targets)
    for value, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize("taps", (2, 3))
def test_empty_connections_clear_due_row_and_advance(taps):
    queue = torch.tensor([[1.0, 2.0]], dtype=DTYPE)
    slot = torch.zeros(1, dtype=torch.long)
    empty = torch.empty(0, dtype=DTYPE)
    actual = dense_interpolated_delay_transition(
        queue,
        slot,
        slot,
        empty,
        empty,
        empty,
        torch.tensor(DT),
        post_idx=torch.empty(0, dtype=torch.long),
        depth=1,
        width=2,
        taps=taps,
        sigma=0.5,
    )
    torch.testing.assert_close(actual.queue, torch.zeros_like(queue))
    assert actual.slot.tolist() == [0]
    assert actual.global_step.tolist() == [1]


def _netcon(taps):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu"):
        pre = dn.Population(N=1, C=3, dtype=DTYPE)
        post = dn.Population(N=1, C=3, dtype=DTYPE)
        post.insert(expsyn.rename("syn"))
        network = dn.Network({"pre": pre, "post": post}, netcon_train_backend="dense")
        network.connect_one_to_one(
            pre[:],
            post[:],
            post.mech.syn,
            threshold=None,
            weight=torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE),
            delay=torch.tensor([0.02, 0.15, 0.3], dtype=DTYPE),
        )
        network.train()
        network.build(DT)
        connection = next(iter(network.synapses.values()))
        connection.set_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=False,
            diff_scheduled_times=False,
            taps=taps,
            sigma=0.7,
        )
        network.initialize(DT)
    return network, connection


@pytest.mark.parametrize("taps", (2, 3))
def test_native_receives_before_detection_and_matches_pure_state(taps):
    network, native = _netcon(taps)
    calls = []

    def receive(synapse, payload, connection):
        calls.append(payload.clone())
        network.pre.v = torch.full_like(network.pre.v, 0.25)

    native.syn.net_receive = MethodType(receive, native.syn)
    queue, slot, step = (
        native.delivery_buffer,
        native.current_time_step,
        native.global_step,
    )
    original = queue.clone()
    expected = dense_interpolated_delay_transition(
        queue,
        slot,
        step,
        native.weight(),
        torch.full((3,), 0.25, dtype=DTYPE),
        native.delay_ms(),
        native.dt,
        post_idx=native.post_idx,
        depth=native.max_delay_steps,
        width=native._syn_numel,
        taps=taps,
        sigma=0.7,
    )
    native.advance()
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], torch.zeros_like(calls[0]))
    torch.testing.assert_close(native.delivery_buffer, expected.queue)
    torch.testing.assert_close(native.current_time_step, expected.slot)
    torch.testing.assert_close(native.global_step, expected.global_step)
    torch.testing.assert_close(queue, original)
    assert native.delivery_buffer.data_ptr() != queue.data_ptr()


@pytest.mark.parametrize("taps", (2, 3))
def test_native_clone_once_allocates_one_output_ring(taps):
    network, native = _netcon(taps)
    network.pre.v = torch.full_like(network.pre.v, 0.25)
    queue = native.delivery_buffer
    original = queue.clone()
    known = queue.untyped_storage().data_ptr()
    allocations = set()

    class RingAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **({} if kwargs is None else kwargs))
            for value in tree_leaves(result):
                if not torch.is_tensor(value) or tuple(value.shape) != tuple(
                    queue.shape
                ):
                    continue
                pointer = value.untyped_storage().data_ptr()
                if pointer != known:
                    allocations.add(pointer)
            return result

    with RingAllocations():
        native.advance()
    assert allocations == {native.delivery_buffer.untyped_storage().data_ptr()}
    torch.testing.assert_close(queue, original)
