"""Ordered native CPU enqueue parity against the standard tensor path."""

import pytest
import torch

from dendra.models.networks._event_delivery import dense_interpolated_delay_transition
from dendra.models.networks._two_tap_enqueue import get_cpu_two_tap_enqueue
from dendra.models.networks._two_tap_enqueue_ext import load_cpu_two_tap_enqueue

pytestmark = pytest.mark.cpu


@pytest.fixture(scope="module")
def enqueue():
    try:
        load_cpu_two_tap_enqueue()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CPU enqueue extension unavailable: {error}")
    return get_cpu_two_tap_enqueue()


def _routing(queue, offsets, targets):
    depth, width = queue.shape
    lower = offsets.remainder(depth) * width + targets
    upper = (lower + width).remainder(queue.numel())
    return lower, upper


def _reference(queue, slot, weights, gates, fraction, offsets, targets):
    depth, width = queue.shape
    return dense_interpolated_delay_transition(
        queue,
        slot,
        slot,
        weights,
        gates,
        None,
        None,
        post_idx=targets,
        depth=depth,
        width=width,
        taps=2,
        sigma=0.5,
        delay_taps=(offsets, fraction),
        _clone_once=True,
    ).queue


@pytest.mark.parametrize(
    ("dtype", "bits"), ((torch.float32, torch.int32), (torch.float64, torch.int64))
)
@pytest.mark.parametrize("depth", (1, 3))
def test_direct_two_tap_gather_preserves_values_and_ownership(
    enqueue, dtype, bits, depth
):
    gather = torch.ops.dendra_native.delay_ring_gather_two_tap.default
    # A flat view of a contiguous ring can have a nonzero storage offset.
    backing = torch.tensor(
        [1.0, -0.0, 0.0, -float("nan"), float("inf"), -float("inf"), 1.5],
        dtype=dtype,
    ).repeat(2)
    width = 4
    ring = backing[1 : 1 + depth * width].reshape(depth, width)
    lower = torch.tensor([4, 0, 3, 5, 10, 1, 10]).remainder(ring.numel())
    upper = torch.tensor([0, 5, 3, 4, 3, 11, 5]).remainder(ring.numel())
    info = torch.iinfo(torch.long)
    extremes = torch.tensor([info.max, info.min, -3, 2 * ring.numel() + 1])
    before = backing.view(bits).clone()
    version = ring._version
    # Include repeated taps, empty vectors, and signed overflow in the same
    # addition that the tensor path performs before its positive remainder.
    cases = (
        ((depth - 1) * width, lower, upper),
        (0, lower[:0], upper[:0]),
        (-3, extremes, extremes.flip(0)),
        (info.min, extremes, extremes.flip(0)),
        (info.max, extremes, extremes.flip(0)),
    )
    for offset, low, high in cases:
        slot_offset = torch.tensor([offset])
        expected = tuple(
            ring.flatten().index_select(0, (tap + slot_offset).remainder(ring.numel()))
            for tap in (low, high)
        )
        positions = (slot_offset, low, high)
        initial_positions = tuple(value.clone() for value in positions)
        actual = gather(ring, *positions)
        assert len(actual) == len(expected)
        assert all(
            torch.equal(value, original)
            for value, original in zip(positions, initial_positions, strict=True)
        )
        for value, reference in zip(actual, expected, strict=True):
            assert value.shape == reference.shape
            assert value.dtype == dtype
            # Compare bit patterns to include signed zero and non-finite values.
            assert torch.equal(value.view(bits), reference.view(bits))
            assert not torch._C._is_alias_of(value, ring)
        assert not torch._C._is_alias_of(actual[0], actual[1])
    assert ring._version == version
    assert torch.equal(backing.view(bits), before)


def _reference_two_tap_vjps(
    ring, offset, weights, gates, fraction, lower, upper, pre_idx
):
    flat = ring.flatten()
    lo, hi = (
        flat.index_select(0, (addresses + offset).remainder(flat.numel()))
        for addresses in (lower, upper)
    )
    edge_gates = gates if pre_idx is None else gates[pre_idx]
    weighted_grad = hi * fraction + lo * (1 - fraction)
    weight_grad = weighted_grad * edge_gates
    edge_grad = weighted_grad * weights
    weighted = weights * edge_gates
    fraction_grad = hi * weighted - lo * weighted
    gate_grad = (
        edge_grad
        if pre_idx is None
        else torch.zeros_like(gates).index_put((pre_idx,), edge_grad, accumulate=True)
    )
    return weight_grad, gate_grad, fraction_grad


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("source_indexed", (False, True))
def test_fused_two_tap_vjps_match_tensor_order(enqueue, dtype, source_indexed):
    vjp = torch.ops.dendra_native.delay_ring_two_tap_vjp.default
    large = 4.0 / torch.finfo(dtype).eps
    weights = torch.tensor([large, 0.3, -large, -0.4, 1.0, 0.9, 1.3], dtype=dtype)
    pre_idx = torch.tensor([1, 0, 1, 2, 1, 2, 0])
    sources = torch.tensor([0.7, 0.0, -0.3, -0.0], dtype=dtype)
    frac = torch.tensor([0.0, 0.7, 0.0, 1.0, 0.0, 0.3, 0.5], dtype=dtype)
    if source_indexed:
        gates = sources
    else:
        gates = sources.index_select(0, pre_idx)
        pre_idx = None

    # The three outgoing edges of source 1 contribute large, then -large,
    # then 1. Reshuffling the updates makes the unit contribution disappear.
    backing = torch.tensor(
        [8.0, 1.0, -2.0, -0.0, 0.0, 5.0, 1.0, 2.0, -4.0, 1.0, 3.0, 0.0, -1.0],
        dtype=dtype,
    )
    ring = backing[1:].view(3, 4)
    lower = torch.tensor([4, 1, 4, 3, 4, 7, 5])
    upper = torch.tensor([0, 5, 7, 4, 10, 11, 5])
    limits = torch.iinfo(torch.long)
    extremes = torch.tensor([limits.max, limits.min, -3, 25, -26, 3, 0])
    nonfinite = ring.clone()
    nonfinite[0] = ring.new_tensor([float("inf"), -float("inf"), float("nan"), -0.0])
    nonfinite_lower = torch.tensor([3, 4, 5, 6, 3, 4, 5])
    nonfinite_upper = torch.tensor([4, 3, 6, 5, 3, 5, 4])
    empty = weights[:0]
    empty_indices = lower[:0]
    negative_pre = (
        torch.where(pre_idx % 2 == 0, pre_idx - len(gates), pre_idx)
        if source_indexed
        else None
    )
    cases = (
        (ring, 8, weights, gates, frac, lower, upper, pre_idx),
        (
            nonfinite,
            -3,
            weights,
            gates,
            frac,
            nonfinite_lower,
            nonfinite_upper,
            pre_idx,
        ),
        (
            ring,
            limits.max,
            weights,
            gates,
            frac,
            extremes,
            extremes.flip(0),
            negative_pre,
        ),
        (
            ring,
            0,
            empty,
            gates if source_indexed else empty,
            empty,
            empty_indices,
            empty_indices,
            empty_indices if source_indexed else None,
        ),
    )
    for primal, shift, w, g, fractions, lo, hi, sources in cases:
        offset = torch.tensor([shift])
        inputs = (primal, offset, w, g, fractions, lo, hi, sources)
        tensor_inputs = tuple(item for item in inputs if item is not None)
        originals = tuple(item.clone() for item in tensor_inputs)
        versions = tuple(item._version for item in tensor_inputs)
        expected = _reference_two_tap_vjps(*inputs)
        actual = vjp(*inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        for value, reference in zip(actual, expected, strict=True):
            zeros = reference == 0
            assert torch.equal(
                torch.signbit(value[zeros]), torch.signbit(reference[zeros])
            )
            assert all(not torch._C._is_alias_of(value, item) for item in tensor_inputs)
        for input_tensor, original, version in zip(
            tensor_inputs, originals, versions, strict=True
        ):
            torch.testing.assert_close(
                input_tensor, original, atol=0, rtol=0, equal_nan=True
            )
            assert input_tensor._version == version
        if source_indexed and shift == 8:
            assert actual[1][1] == 1.0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_two_tap_queue_clear_wrap_and_accumulation_order(enqueue, dtype):
    queue = torch.zeros((3, 4), dtype=dtype).t()
    queue[-1].fill_(4.0)
    original = queue.clone()
    slot = torch.tensor([3])
    offsets = torch.tensor([1, 2, 2])
    targets = torch.zeros(3, dtype=torch.long)
    fraction = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    large = 4.0 / torch.finfo(dtype).eps
    weights = torch.tensor([large, -large, 1.0], dtype=dtype)
    gates = torch.ones_like(weights)
    lower, upper = _routing(queue, offsets, targets)

    actual = enqueue(queue, slot, weights, gates, fraction, lower, upper)
    expected = _reference(queue, slot, weights, gates, fraction, offsets, targets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(queue, original, rtol=0, atol=0)
    assert actual.is_contiguous()
    assert actual.data_ptr() != queue.data_ptr()
    assert torch.count_nonzero(actual).item() == 0
    # Lower tap -large, then lower tap 1, then upper tap +large rounds to zero.
    # Combining the two taps edge by edge would produce 1 instead.


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_two_tap_adjoint_preserves_zero_gate_path(enqueue, dtype):
    queue = torch.arange(12, dtype=dtype).reshape(4, 3).requires_grad_()
    slot = torch.tensor([3])
    offsets = torch.tensor([0, 1, 2, 2, 7])
    targets = torch.tensor([0, 2, 2, 2, 0])
    weights = torch.tensor([0.3, -0.2, 0.7, 1.3, 0.9], dtype=dtype, requires_grad=True)
    gates = torch.tensor([0.7, 0.2, 0.0, -0.3, 1.1], dtype=dtype, requires_grad=True)
    fraction = torch.linspace(0.0, 1.0, 5, dtype=dtype).requires_grad_()
    lower, upper = _routing(queue, offsets, targets)
    inputs = queue, weights, gates, fraction
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = enqueue(queue, slot, weights, gates, fraction, lower, upper)
    expected = _reference(queue, slot, weights, gates, fraction, offsets, targets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(tensor.data_ptr() != queue.data_ptr() for tensor in saved)

    tangent = torch.linspace(0.3, 2.3, 12, dtype=dtype).reshape_as(queue)
    actual_gradients = torch.autograd.grad(actual, inputs, tangent, create_graph=True)
    expected_gradients = torch.autograd.grad(
        expected, inputs, tangent, create_graph=True
    )
    assert actual_gradients[2][2].item() != 0.0
    assert torch.count_nonzero(actual_gradients[0][slot]).item() == 0
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    second_actual = torch.autograd.grad(actual_gradients[2].sum(), weights)
    second_expected = torch.autograd.grad(expected_gradients[2].sum(), weights)
    torch.testing.assert_close(second_actual, second_expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_two_tap_source_gates_preserve_saved_shape_and_edge_order(enqueue, dtype):
    from_source = torch.ops.dendra_native.ordered_two_tap_enqueue_from_source.default
    # The source at index 1 has three interleaved outgoing edges. Adding
    # their cotangents out of order can lose the unit term to rounding.
    source = torch.tensor([0.7, 0.0, -0.3, -0.0], dtype=dtype, requires_grad=True)
    pre_idx = torch.tensor([1, 0, 1, 2, 1, 2, 0])
    large = 4.0 / torch.finfo(dtype).eps
    weights = torch.tensor(
        [large, 0.3, -large, -0.4, 1.0, 0.9, 1.3], dtype=dtype, requires_grad=True
    )
    fraction = torch.tensor(
        [0.0, 0.7, 0.0, 1.0, 0.0, 0.3, 0.5], dtype=dtype, requires_grad=True
    )
    queue = torch.arange(12, dtype=dtype).reshape(3, 4).t().requires_grad_()
    slot = torch.tensor([3])
    offsets = torch.tensor([0, 1, 1, 2, 2, 7, 0])
    targets = torch.tensor([0, 2, 0, 2, 0, 2, 1])
    lower, upper = _routing(queue, offsets, targets)
    before = queue.detach().clone()
    version = queue._version
    saved = []

    def pack(value):
        saved.append(value)
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        actual = from_source(
            queue, slot, weights, source, fraction, lower, upper, pre_idx
        )
    expected = enqueue(
        queue, slot, weights, source.index_select(0, pre_idx), fraction, lower, upper
    )
    assert any(value is source for value in saved)
    assert all(
        value is weights or value is fraction
        for value in saved
        if value.is_floating_point() and value.numel() == pre_idx.numel()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch._C._is_alias_of(actual, queue)
    assert queue._version == version
    torch.testing.assert_close(queue, before, rtol=0, atol=0)

    inputs = (queue, weights, source, fraction)
    tangent = torch.linspace(0.3, 2.3, queue.numel(), dtype=dtype).reshape_as(queue)
    actual_gradients = torch.autograd.grad(actual, inputs, tangent, retain_graph=True)
    expected_gradients = torch.autograd.grad(
        expected, inputs, tangent, retain_graph=True
    )
    torch.testing.assert_close(actual_gradients, expected_gradients, rtol=0, atol=0)
    assert actual_gradients[2][1] != 0.0
    assert actual_gradients[2][3] == 0.0

    unit_tangent = torch.ones_like(actual)
    (actual_source_gradient,) = torch.autograd.grad(actual, source, unit_tangent)
    (expected_source_gradient,) = torch.autograd.grad(expected, source, unit_tangent)
    torch.testing.assert_close(
        actual_source_gradient, expected_source_gradient, rtol=0, atol=0
    )
    assert actual_source_gradient[1] == 1.0


def test_two_tap_empty_projection_only_clears_current_row(enqueue):
    queue = torch.ones((2, 3), requires_grad=True)
    slot = torch.tensor([1])
    indices = torch.empty(0, dtype=torch.long)
    empty = queue.new_empty(0)
    result = enqueue(queue, slot, empty, empty, empty, indices, indices)
    expected = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(queue.grad, expected, rtol=0, atol=0)


def test_network_jit_reuses_two_tap_addresses(enqueue, monkeypatch):
    import dendra as dn
    from dendra.models.networks import netcon
    from tests.networks.test_functional_delay_network import DEMO

    addresses = []

    def inspect_enqueue(queue, slot, weights, gates, fraction, lower, upper):
        addresses.append((lower, upper))
        return enqueue(queue, slot, weights, gates, fraction, lower, upper)

    monkeypatch.setattr(
        netcon, "_cpu_two_tap_enqueue_for_policy", lambda _: inspect_enqueue
    )
    steps = DEMO["STEPS"]
    results = []
    for jit_network_ops in (0, 1):
        network = DEMO["make_network"]()
        network.pre.mech.source.gate = network.pre.v.new_tensor([[0.25], [0.5]])
        with dn.ctx(JIT=0, JIT_NETWORK_OPS=jit_network_ops):
            network.run(steps=steps)
        (projection,) = network.synapses.values()
        loss = network.post.v.square().sum() + projection.delivery_buffer.square().sum()
        gradients = torch.autograd.grad(
            loss, (projection.weight.w, projection.delay_ms.w)
        )
        results.append((network.post.v, projection.delivery_buffer, *gradients))

    assert len(addresses) == steps
    first_lower, first_upper = addresses[0]
    assert all(lo is first_lower and hi is first_upper for lo, hi in addresses)
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _functional_delay_case(taps=2):
    import dendra as dn
    from tests.networks.test_functional_delay_network import DEMO

    network = DEMO["make_network"](taps=taps)
    network.pre.mech.source.gate = network.pre.v.new_tensor([[0.0], [0.5]])
    functional, tensors = dn.func.make_functional_network(network)
    (projection_name,) = tensors.parameters["synapses"]
    parameters = tensors.parameters["synapses"][projection_name]
    weights = parameters["weight.weights.0.param.rho"]
    delays = parameters["delay_ms.weights.0.param.rho"]
    queue = tensors.state["synapses"][projection_name]["delivery_buffer"]
    gates = tensors.state["populations"]["pre"]["mechanism_buffers"]["source"]["gate"]
    with torch.no_grad():
        queue[-1].fill_(0.001)
    queue.requires_grad_()
    gates.requires_grad_()
    return (
        network,
        functional,
        tensors,
        projection_name,
        (weights, delays, queue, gates),
    )


def _delay_scan_output(state, projection_name):
    queue = state["synapses"][projection_name]["delivery_buffer"]
    voltage = state["populations"]["post"]["integrator"]["v"]
    return torch.cat((queue.flatten(), voltage.flatten()))


def _profile_enqueue(call, *, in_place_only=False):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        result = call()
    names = {"dendra_native::ordered_two_tap_enqueue_"}
    if not in_place_only:
        names.add("dendra_native::ordered_two_tap_enqueue")
    return result, sum(event.name in names for event in profile.events())


@pytest.mark.parametrize("taps, expected_calls", ((2, 5), (3, 0)))
def test_functional_scan_fuses_only_two_tap_delays(enqueue, taps, expected_calls):
    from torch.utils import _pytree as pytree

    import dendra as dn

    network, functional, tensors, projection_name, targets = _functional_delay_case(
        taps
    )
    state_before = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tensors.state
    )
    source_before = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), network.state_dict()
    )
    with dn.ctx(NATIVE_EXTENSION_POLICY="require"):
        chunk = functional.compile_rollout_chunk(
            5,
            execution="scan",
            backend="inductor",
            options={"compile_threads": 1, "cpp.threads": 1},
        )
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        args = tensors.parameters, prepared, tensors.state
        expected, _ = functional.rollout(*args, steps=chunk.steps)
        chunk(*args)
        (actual, _), count = _profile_enqueue(lambda: chunk(*args))

    assert count == expected_calls
    assert chunk.execution_report().strategy == "compiled_scan"
    actual_output = _delay_scan_output(actual, projection_name)
    expected_output = _delay_scan_output(expected, projection_name)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual_output.square().sum(), targets)
    expected_gradients = torch.autograd.grad(expected_output.square().sum(), targets)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    assert all(torch.count_nonzero(gradient) for gradient in actual_gradients)
    torch.testing.assert_close(tensors.state, state_before)
    torch.testing.assert_close(network.state_dict(), source_before)
    assert (
        actual["synapses"][projection_name]["delivery_buffer"].data_ptr()
        != targets[2].data_ptr()
    )


def test_private_ring_update_preserves_public_network_state(enqueue, monkeypatch):
    import dendra as dn
    from dendra._bootstrap import torch_compiler_warning_context
    from dendra.func import _scan_tree

    # Exercise dedicated ring carry on an independent, two-cell network.
    monkeypatch.setattr(_scan_tree, "_CPU_TRAINING_SCAN_LARGE_LEAF_BYTES", 1)
    network, functional, tensors, name, targets = _functional_delay_case()
    weights, delays, queue, gate = targets
    queue_before = queue.detach().clone()
    version = queue._version
    shifted = (queue_before + 0.002).requires_grad_()
    shifted_state = {
        **tensors.state,
        "synapses": {
            name: {
                **tensors.state["synapses"][name],
                "delivery_buffer": shifted,
            }
        },
    }
    shifted_before = shifted.detach().clone()
    shifted_version = shifted._version
    voltage_before = network.post.v.detach().clone()

    def evaluate(initial, run):
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        state, _ = run(tensors.parameters, prepared, initial)
        return state, _delay_scan_output(state, name)

    with dn.ctx(NATIVE_EXTENSION_POLICY="require"), torch_compiler_warning_context():
        chunk = functional.compile_rollout_chunk(
            5,
            execution="scan",
            backend="inductor",
            options={"compile_threads": 1, "cpp.threads": 1, "cpp_wrapper": False},
        )

        def eager(*args):
            return functional.rollout(*args, steps=chunk.steps)

        evaluate(tensors.state, chunk)
        (actual, output), count = _profile_enqueue(
            lambda: evaluate(tensors.state, chunk), in_place_only=True
        )
        second, shifted_output = evaluate(shifted_state, chunk)
        _, expected = evaluate(tensors.state, eager)
        _, expected_shifted = evaluate(shifted_state, eager)

        assert count == chunk.steps
        assert chunk.execution_report().strategy == "compiled_scan"
        torch.testing.assert_close(
            _delay_scan_output(actual, name), output, rtol=0, atol=0
        )
        for out, reference, ring in (
            (output, expected, queue),
            (shifted_output, expected_shifted, shifted),
        ):
            operands = weights, delays, ring, gate
            gradients = torch.autograd.grad(out.square().sum(), operands)
            expected_gradients = torch.autograd.grad(reference.square().sum(), operands)
            torch.testing.assert_close(out, reference)
            torch.testing.assert_close(gradients, expected_gradients)
            assert all(gradient.abs().sum() > 0 for gradient in gradients)

    assert queue._version == version and shifted._version == shifted_version
    torch.testing.assert_close(queue, queue_before, rtol=0, atol=0)
    torch.testing.assert_close(shifted, shifted_before, rtol=0, atol=0)
    torch.testing.assert_close(network.post.v, voltage_before, rtol=0, atol=0)
    final_ring = actual["synapses"][name]["delivery_buffer"]
    second_ring = second["synapses"][name]["delivery_buffer"]
    assert all(
        not torch._C._is_alias_of(final_ring, other)
        for other in (queue, shifted, second_ring)
    )


@pytest.mark.parametrize("extra_read", ("saved_primal", "old_view"))
def test_scan_ring_falls_back_when_old_ring_stays_live(
    enqueue, monkeypatch, extra_read
):
    from dendra._bootstrap import torch_compiler_warning_context
    from dendra.func import _scan_tree
    from dendra.models.networks._event_delivery import dense_due

    monkeypatch.setattr(_scan_tree, "_CPU_TRAINING_SCAN_LARGE_LEAF_BYTES", 1)
    steps = 4
    queue = torch.arange(12, dtype=torch.float64).reshape(4, 3).div(10).requires_grad_()
    weights = queue.new_tensor([0.4, -0.2, 0.8], requires_grad=True)
    fraction = queue.new_tensor([0.2, 0.3, 0.1], requires_grad=True)
    gates = queue.new_tensor(
        [[0.1, 0.2, -0.3], [0.7, 0.2, 0.6], [0.1, 0.4, -0.1], [0.0, 0.4, 0.1]],
        requires_grad=True,
    )
    slots = torch.arange(steps, dtype=torch.long).reshape(-1, 1)
    offsets, targets = torch.tensor([4, 3, 1]), torch.tensor([1, 2, 0])
    lower, upper = _routing(queue, offsets, targets)
    version = queue._version
    original = queue.detach().clone()

    def body(params, prepared, state, carries, sample):
        ring = state["ring"]
        slot, gate = sample
        due = dense_due(ring, slot).payload
        if extra_read == "saved_primal":
            # The square backward reads the whole ring, so row gathers
            # cannot justify reusing the incoming buffer here.
            audit = ring.square()
        else:
            # Clone-on-emission happens after the queue update in the
            # captured graph, so the earlier view must still see old data.
            audit = ring.view_as(ring)
        updated = enqueue(
            ring, slot, params["w"], gate, params["fraction"], lower, upper
        )
        return ({"ring": updated}, carries), {"due": due, "audit": audit}

    engine = _scan_tree._CompiledTensorTreeScan(
        body,
        steps=steps,
        compile_options={
            "backend": "inductor",
            "fullgraph": True,
            "options": {"compile_threads": 1, "cpp.threads": 1, "cpp_wrapper": False},
        },
        device="cpu",
    )
    parameters = {"w": weights, "fraction": fraction}

    def run():
        return engine(parameters, {}, {"ring": queue}, {}, (slots, gates))

    with torch_compiler_warning_context():
        run()
        (actual_state, _, actual), count = _profile_enqueue(run, in_place_only=True)
    assert count == 0
    expected_state = {"ring": queue}
    frames = []
    for slot, gate in zip(slots, gates, strict=True):
        (expected_state, _), frame = body(
            parameters, {}, expected_state, {}, (slot, gate)
        )
        frames.append(frame)
    expected = {key: torch.stack([frame[key] for frame in frames]) for key in actual}
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def objective(state, emitted):
        return state["ring"].square().sum() + sum(
            value.square().sum() for value in emitted.values()
        )

    operands = queue, weights, gates, fraction
    actual_gradients = torch.autograd.grad(objective(actual_state, actual), operands)
    expected_gradients = torch.autograd.grad(
        objective(expected_state, expected), operands
    )
    torch.testing.assert_close(
        actual_gradients, expected_gradients, rtol=1e-13, atol=1e-13
    )
    assert queue._version == version
    torch.testing.assert_close(queue, original, rtol=0, atol=0)


@pytest.mark.parametrize("scheduled", (False, True))
def test_source_sized_gates_select_only_intrinsic_fanout(
    enqueue, monkeypatch, scheduled
):
    import dendra as dn
    from dendra._bootstrap import torch_compiler_warning_context
    from dendra.func import _scan_tree
    from dendra.models.mod import expsyn, pas
    from tests.networks.test_functional_delay_network import DEMO

    monkeypatch.setattr(_scan_tree, "_CPU_TRAINING_SCAN_LARGE_LEAF_BYTES", 1)
    dtype = torch.float64
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=dtype, IMEM=False, DEVICE="cpu"):
        pre = dn.SingleCompartment(N=2, C=1)
        pre.insert(DEMO["GateInput"].rename("source"))
        post = dn.SingleCompartment(N=2, C=1)
        post.insert(pas, g=0.001, e=-65.0)
        post.insert(expsyn.rename("event"), e=0.0, tau=2.0)
        network = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect(
            pre[:],
            post[:],
            post.mech.event,
            conn_spec="all_to_all",
            pre_var="mech.source.gate",
            threshold=None,
            weight=torch.tensor([0.002, 0.003, 0.004, 0.001]),
            delay=torch.tensor([0.065, 0.135, 0.09, 0.12]),
        )
        network.train()
        network.initialize(DEMO["DT"])
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=True,
            taps=2,
            diff_scheduled_times=True,
            train_delay_backend="dense",
        )
        (projection,) = network.synapses.values()
        if scheduled:
            projection.schedule(con_indices=[0], times_ms=[0.065], weight=0.3)
        pre.mech.source.gate = pre.v.new_tensor([[0.0], [0.5]])

    functional, tensors = dn.func.make_functional_network(network)
    (name,) = tensors.parameters["synapses"]
    network_parameters = tensors.parameters["synapses"][name]
    weight = network_parameters["weight.weights.0.param.rho"]
    delay = network_parameters["delay_ms.weights.0.param.rho"]
    ring = tensors.state["synapses"][name]["delivery_buffer"].requires_grad_()

    # Exercise public carry with a strided, lazy negative source view.
    backing = torch.tensor([[0.0, 4.0], [-0.5, -3.0]], dtype=dtype, requires_grad=True)
    gate = torch._neg_view(backing[:, :1])
    source_state = tensors.state["populations"]["pre"]
    buffers = source_state["mechanism_buffers"]
    state = {
        **tensors.state,
        "populations": {
            **tensors.state["populations"],
            "pre": {
                **source_state,
                "mechanism_buffers": {
                    **buffers,
                    "source": {**buffers["source"], "gate": gate},
                },
            },
        },
    }
    ring_before, backing_before = ring.detach().clone(), backing.detach().clone()
    versions = (ring._version, backing._version)
    operands = (weight, delay, ring, backing)

    with dn.ctx(NATIVE_EXTENSION_POLICY="require"), torch_compiler_warning_context():
        chunk = functional.compile_rollout_chunk(
            5,
            execution="scan",
            backend="inductor",
            options={"compile_threads": 1, "cpp.threads": 1, "cpp_wrapper": False},
        )

        def evaluate(run):
            prepared = functional.prepare(tensors.parameters, tensors.constants)
            final, _ = run(tensors.parameters, prepared, state)
            output = _delay_scan_output(final, name)
            gradients = torch.autograd.grad(output.square().sum(), operands)
            return output, gradients

        def eager(*args):
            return functional.rollout(*args, steps=chunk.steps)

        expected, expected_gradients = evaluate(eager)
        evaluate(chunk)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            actual, actual_gradients = evaluate(chunk)

    names = [event.name for event in profile.events()]
    compact = "dendra_native::ordered_two_tap_enqueue_from_source_"
    expanded = "dendra_native::ordered_two_tap_enqueue_"
    assert names.count(compact) == (0 if scheduled else chunk.steps)
    assert names.count(expanded) == (chunk.steps if scheduled else 0)
    assert names.count("dendra_native::delay_ring_adjoint_update_") == chunk.steps
    assert chunk.execution_report().strategy == "compiled_scan"
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    assert all(gradient.count_nonzero() for gradient in actual_gradients)
    torch.testing.assert_close(ring, ring_before, rtol=0, atol=0)
    torch.testing.assert_close(backing, backing_before, rtol=0, atol=0)
    assert (ring._version, backing._version) == versions


def test_functional_scan_two_tap_eager_and_transform_fallbacks(enqueue):
    import dendra as dn

    _, functional, tensors, name, (_, delay, _, _) = _functional_delay_case()
    with dn.ctx(NATIVE_EXTENSION_POLICY="require"):
        chunk = functional.compile_rollout_chunk(
            2,
            execution="scan",
            backend="inductor",
            options={"compile_threads": 1, "cpp.threads": 1, "cpp_wrapper": False},
        )
        disabled = functional.compile_rollout_chunk(
            2, execution="scan", backend="inductor", disable=True
        )

        def output(raw_delay, execute):
            projection = {
                **tensors.parameters["synapses"][name],
                "delay_ms.weights.0.param.rho": raw_delay,
            }
            parameters = {**tensors.parameters, "synapses": {name: projection}}
            prepared = functional.prepare(parameters, tensors.constants)
            state, _ = execute(parameters, prepared, tensors.state)
            return _delay_scan_output(state, name)

        def eager(*args):
            return functional.rollout(*args, steps=chunk.steps)

        output(delay, chunk)
        with torch.no_grad():
            expected = output(delay, eager)
            chunk_output, calls = _profile_enqueue(lambda: output(delay, chunk))
            assert calls == 0
            assert chunk.execution_report().strategy == "compiled_scan"
            assert not chunk_output.requires_grad
            torch.testing.assert_close(chunk_output, expected)

        actual, calls = _profile_enqueue(lambda: output(delay, disabled))
        assert calls == 0
        assert actual.requires_grad
        assert disabled.execution_report().strategy == "eager_disabled"
        torch.testing.assert_close(actual, expected)

        primals, tangents = (delay.detach(),), (torch.ones_like(delay),)
        expected = torch.func.jvp(lambda raw: output(raw, eager), primals, tangents)
        actual, calls = _profile_enqueue(
            lambda: torch.func.jvp(lambda raw: output(raw, chunk), primals, tangents)
        )
        assert calls == 0
        assert chunk.execution_report().strategy == "eager_transform_fallback"
        torch.testing.assert_close(actual, expected)


def test_functional_scan_two_tap_extension_load_policy(monkeypatch):
    import dendra as dn
    from dendra.models.networks import _two_tap_enqueue, netcon

    _, functional, tensors, _, _ = _functional_delay_case()
    chunk = functional.compile_rollout_chunk(
        2,
        execution="scan",
        backend="inductor",
        options={"compile_threads": 1, "cpp.threads": 1},
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    args = tensors.parameters, prepared, tensors.state
    expected, _ = functional.rollout(*args, steps=chunk.steps)
    attempts = []

    def unavailable():
        attempts.append(None)
        raise OSError("test enqueue extension unavailable")

    loader = netcon._cpu_two_tap_enqueue_for_policy
    loader.cache_clear()
    monkeypatch.setattr(_two_tap_enqueue, "get_cpu_two_tap_enqueue", unavailable)
    try:
        with (
            dn.ctx(NATIVE_EXTENSION_POLICY="require"),
            pytest.raises(RuntimeError, match="test enqueue extension unavailable"),
        ):
            chunk(*args)
        with dn.ctx(NATIVE_EXTENSION_POLICY="fallback"):
            actual, _ = chunk(*args)
        torch.testing.assert_close(actual, expected)
        with (
            dn.ctx(NATIVE_EXTENSION_POLICY="warn"),
            pytest.warns(RuntimeWarning, match="test enqueue extension unavailable"),
        ):
            actual, _ = chunk(*args)
        torch.testing.assert_close(actual, expected)
        assert attempts == [None, None, None]
    finally:
        loader.cache_clear()
