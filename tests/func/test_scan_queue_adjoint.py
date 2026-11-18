"""Validate Inductor's one-row cotangent combination and two-tap gradients."""

from collections import Counter

import pytest
import torch
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental.proxy_tensor import make_fx

from dendra.func._scan_inductor import _CopyViewLowering
from dendra.models.networks._event_delivery import (
    dense_due,
    dense_interpolated_delay_transition,
)

pytestmark = pytest.mark.cpu


@pytest.fixture
def fused_enqueue():
    from dendra.models.networks._two_tap_enqueue import get_cpu_two_tap_enqueue

    try:
        return get_cpu_two_tap_enqueue()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CPU enqueue extension unavailable: {error}")


def _profile_ring_adjoint(call):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        result = call()
    return result, Counter(event.name for event in profile.events())


def _trace(function, *inputs):
    return make_fx(function, decomposition_table=select_decomp_table())(*inputs)


def _lower(module):
    _CopyViewLowering()(module.graph)
    module.graph.eliminate_dead_code()
    module.graph.lint()
    module.recompile()


def _index_put_count(module):
    return sum(
        node.target is torch.ops.aten.index_put.default for node in module.graph.nodes
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("clone_slot", (False, True))
def test_combines_queue_cotangents_without_mutating_shared_incoming_ring(
    dtype, clone_slot
):
    slot = torch.tensor([1])
    incoming = torch.tensor(
        [
            [-0.0, float("nan"), 3.0, -float("inf")],
            [float("nan"), -0.0, 1.0, 2.0],
            [2.0, 3.0, -0.0, float("inf")],
        ],
        dtype=dtype,
    )
    payload = torch.tensor([[-0.0, 1.5, float("nan"), 0.0]], dtype=dtype)
    original = incoming.clone()
    initial_version = incoming._version

    def adjoint(ring, index, due):
        clear = ring.index_fill(0, index, 0)
        receive_index = index.clone() if clone_slot else index
        receive = torch.zeros_like(ring).index_add(0, receive_index, due)
        # An enqueue VJP also reads the received row whenever a tap wraps to it.
        combined = receive + clear if clone_slot else clear + receive
        return combined, ring.index_select(0, index)

    module = _trace(adjoint, incoming, slot, payload)
    expected = module(incoming, slot, payload)
    assert _index_put_count(module) == 2
    _lower(module)
    assert _index_put_count(module) <= 1
    actual = module(incoming, slot, payload)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    for value, reference in zip(actual, expected, strict=True):
        zeros = reference == 0
        torch.testing.assert_close(
            torch.signbit(value[zeros]), torch.signbit(reference[zeros])
        )
    assert actual[0][1, 0].item() == 0.0
    assert not torch._C._is_alias_of(actual[0], incoming)
    assert incoming._version == initial_version
    torch.testing.assert_close(incoming, original, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(torch.signbit(incoming), torch.signbit(original))


@pytest.mark.parametrize(
    "reason",
    (
        "different_slot",
        "repeated_indices",
        "nonzero_fill",
        "nonzero_base",
        "negative_zero_fill",
        "negative_zero_base",
        "scaled_add",
        "shared_clear",
        "late_mutation",
    ),
)
def test_queue_adjoint_lowering_leaves_unproven_terms_intact(reason):
    incoming = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    slot = torch.tensor([1, 1] if reason == "repeated_indices" else [1])
    due = torch.arange(4, dtype=incoming.dtype).repeat(slot.numel(), 1)
    other = torch.tensor([2])

    def adjoint(ring, index, due_row, other_index):
        fill = {"nonzero_fill": 3.0, "negative_zero_fill": -0.0}.get(reason, 0.0)
        seed = {"nonzero_base": 2.0, "negative_zero_base": -0.0}.get(reason, 0.0)
        cleared = ring.index_fill(0, index, fill)
        receive_index = other_index if reason == "different_slot" else index
        received = torch.full_like(ring, seed).index_add(0, receive_index, due_row)
        if reason == "late_mutation":
            ring.copy_(ring + 1)
        combined = cleared.add(received, alpha=2 if reason == "scaled_add" else 1)
        return (combined, cleared) if reason == "shared_clear" else combined

    module = _trace(adjoint, incoming.clone(), slot, due, other)
    expected = module(incoming.clone(), slot, due, other)
    assert _index_put_count(module) == 2
    _lower(module)
    assert _index_put_count(module) == 2
    torch.testing.assert_close(
        module(incoming.clone(), slot, due, other), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("source_indexed", "threads"),
    ((False, 1), (True, 1), (True, 2)),
    ids=("edge_gates", "source_gates", "parallel_source_fallback"),
)
def test_compiled_delay_scan_reads_original_enqueued_cotangent_when_a_tap_wraps(
    fused_enqueue, source_indexed, threads
):
    from torch._higher_order_ops.scan import scan

    from dendra._bootstrap import torch_compiler_warning_context
    from dendra.func._scan_compat import install_scan_compatibility
    from dendra.func._scan_inductor import _compile_scan

    install_scan_compatibility()
    depth, width, steps = 4, 3, 4
    # Tap destinations visit the current row in two ways. The first edge's
    # lower tap has delay = depth. The second edge's upper tap wraps from
    # delay = depth - 1. The first edge starts with a zero surrogate gate.
    offsets = torch.tensor([depth, depth - 1, 1])
    targets = torch.tensor([1, 2, 0])
    slots = torch.arange(steps, dtype=torch.long).reshape(-1, 1)
    if source_indexed:
        pre_idx = torch.tensor([0, 1, 0])
        enqueue = torch.ops.dendra_native.ordered_two_tap_enqueue_from_source.default
    else:
        pre_idx = None
        enqueue = fused_enqueue

    def recurrence(queue, weights, gates, fraction, lower, upper):
        def step(ring, sample):
            slot, gate = sample
            payload = dense_due(ring, slot).payload
            operands = (ring, slot, weights, gate, fraction, lower, upper)
            if pre_idx is not None:
                operands = (*operands, pre_idx)
            return enqueue(*operands), payload

        return scan(step, queue, (slots, gates))

    def reference(queue, weights, gates, fraction):
        outputs = []
        for slot, gate in zip(slots, gates, strict=True):
            outputs.append(dense_due(queue, slot).payload)
            if pre_idx is not None:
                gate = gate.index_select(0, pre_idx)
            queue = dense_interpolated_delay_transition(
                queue,
                slot,
                slot,
                weights,
                gate,
                None,
                None,
                post_idx=targets,
                depth=depth,
                width=width,
                taps=2,
                sigma=0.0,
                delay_taps=(offsets, fraction),
            ).queue
        return queue, torch.stack(outputs)

    def objective(final, emitted):
        coefficient = torch.arange(1, final.numel() + 1, dtype=final.dtype)
        return (final.flatten() * coefficient).sum() + emitted.square().sum()

    compiled = _compile_scan(
        recurrence,
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "compile_threads": 1,
                "cpp.threads": threads,
                "cpp_wrapper": False,
            },
        },
    )
    dtype = torch.float64
    gates = torch.tensor(
        [[0.0, -0.2, 0.3], [0.7, 0.2, 0.6], [0.1, 0.4, -0.1], [0.5, 0.4, 0.1]],
        dtype=dtype,
    )
    if source_indexed:
        gates = gates[:, :2].contiguous()
    inputs = (
        torch.linspace(0.1, 0.9, depth * width, dtype=dtype).reshape(depth, width),
        torch.tensor([0.4, 0.2, 0.8], dtype=dtype),
        gates,
        torch.tensor([0.2, 0.3, 0.1], dtype=dtype),
    )
    actual_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    expected_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    original_versions = [value._version for value in actual_inputs]
    lower = offsets.remainder(depth) * width + targets
    upper = (lower + width).remainder(depth * width)
    with torch_compiler_warning_context():
        actual = compiled(*actual_inputs, lower, upper)
        expected = reference(*expected_inputs)
        actual_grads = torch.autograd.grad(
            objective(*actual), actual_inputs, retain_graph=True
        )
        expected_grads = torch.autograd.grad(
            objective(*expected), expected_inputs, retain_graph=True
        )

        # The retained forward graph must accept a second terminal cotangent.
        # This one aliases the caller's ring, so accidental input donation
        # would also corrupt the original primal tensor.
        terminal_cotangent = actual_inputs[0].detach()
        emitted_cotangent = expected[1].detach().mul(3.0)
        reference_vjp = torch.autograd.grad(
            expected,
            expected_inputs,
            (terminal_cotangent.clone(), emitted_cotangent),
        )
        actual_vjp, calls = _profile_ring_adjoint(
            lambda: torch.autograd.grad(
                actual, actual_inputs, (terminal_cotangent, emitted_cotangent)
            )
        )

    assert calls["dendra_native::delay_ring_adjoint_update_"] == steps
    fallback = source_indexed and threads > 1
    assert calls["dendra_native::delay_ring_two_tap_vjp"] == (0 if fallback else steps)
    assert calls["dendra_native::delay_ring_gather_two_tap"] == (
        steps if fallback else 0
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual_vjp, reference_vjp, rtol=1e-13, atol=1e-13)
    assert actual_grads[2][0, 0] != 0
    assert all(gradient.abs().sum() > 0 for gradient in actual_grads)
    for value, original, version in zip(
        actual_inputs, inputs, original_versions, strict=True
    ):
        assert value._version == version
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_compiled_queue_adjoint_keeps_negative_index_and_positive_zero_semantics():
    from dendra.func._scan_inductor import _compile_scan

    aten = torch.ops.aten

    def adjoint(ring, slot, row):
        cleared = aten.index_put.default(ring, [slot], torch.zeros_like(row))
        injected = aten.index_put.default(torch.zeros_like(ring), [slot], row, True)
        return cleared + injected

    compiled = _compile_scan(
        adjoint,
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {"compile_threads": 1, "cpp.threads": 1},
        },
    )
    ring = torch.tensor([[-0.0, 2.0], [float("nan"), -0.0]], dtype=torch.float64)
    row = torch.tensor([[-0.0, -float("inf")]], dtype=ring.dtype)
    for slot in (torch.tensor([0]), torch.tensor([-1])):
        expected = adjoint(ring, slot, row)
        actual = compiled(ring, slot, row)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        zeros = expected == 0
        torch.testing.assert_close(
            torch.signbit(actual[zeros]), torch.signbit(expected[zeros])
        )


@pytest.mark.parametrize("retain_old_ring", (False, True))
def test_private_queue_adjoint_captures_old_reads_before_normalization(
    fused_enqueue, retain_old_ring
):
    from torch._higher_order_ops.scan import scan
    from torch.utils._pytree import tree_leaves

    from dendra._bootstrap import torch_compiler_warning_context
    from dendra.func._scan_compat import install_scan_compatibility
    from dendra.func._scan_inductor import _compile_scan

    install_scan_compatibility()
    aten = torch.ops.aten
    initial = torch.tensor(
        [
            [-0.0, 2.0, float("nan")],
            [float("inf"), 0.0, -0.0],
            [-0.0, -float("inf"), 1.0],
        ],
        dtype=torch.float64,
    )
    before = initial.clone()
    version = initial._version
    slots = torch.tensor([[-1], [-1], [0]])
    due_rows = initial.new_tensor(
        [[[-0.0, 4.0, 0.0]], [[2.0, 0.0, -0.0]], [[-0.0, 5.0, 0.0]]]
    )
    tap_indices = torch.tensor([[-3, 0, 0, 2], [-3, 0, 1, 2], [0, 1, 3, -1]])

    def step(ring, sample):
        slot, due, tap_index = sample
        taps = aten.index.Tensor(ring.reshape(-1), [tap_index])
        clear = aten.index_put.default(ring, [slot], torch.zeros_like(due))
        add = aten.index_put.default(torch.zeros_like(ring), [slot], due, True)
        updated = clear + add
        if retain_old_ring:
            # This view is retained after the transition. The ordinary scan
            # owns its materialized observation and observes the old ring.
            return updated, (taps, ring.T.contiguous())
        return updated, (taps,)

    def recurrence(ring, indices, rows, reads):
        return scan(step, ring, (indices, rows, reads))

    compiled = _compile_scan(
        recurrence,
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {"compile_threads": 1, "cpp.threads": 1, "cpp_wrapper": False},
        },
    )
    expected_ring = before
    frames = []
    for inputs in zip(slots, due_rows, tap_indices, strict=True):
        expected_ring, frame = step(expected_ring, inputs)
        frames.append(frame)
    expected_emissions = tuple(
        torch.stack(fields) for fields in zip(*frames, strict=True)
    )
    with torch_compiler_warning_context():
        compiled(initial, slots, due_rows, tap_indices)
        (ring, emissions), calls = _profile_ring_adjoint(
            lambda: compiled(initial, slots, due_rows, tap_indices)
        )

    assert calls["dendra_native::delay_ring_adjoint_update_"] == (
        0 if retain_old_ring else slots.shape[0]
    )
    assert calls["dendra_native::delay_ring_gather_two_tap"] == 0
    assert calls["dendra_native::delay_ring_two_tap_vjp"] == 0
    pairs = zip(
        tree_leaves((ring, emissions)),
        tree_leaves((expected_ring, expected_emissions)),
        strict=True,
    )
    for actual, expected in pairs:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        zeros = expected == 0
        assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))
    assert torch.signbit(emissions[0][0, 0])
    assert initial._version == version
    assert torch.equal(initial.view(torch.int64), before.view(torch.int64))
    assert not torch._C._is_alias_of(ring, initial)
