"""Update owned delay-ring carries after scan's autograd partitioning."""

import operator
from collections import Counter
from typing import NamedTuple

import torch
from torch._inductor.pattern_matcher import get_mutation_region_id
from torch._subclasses.fake_tensor import FakeTensor


class _RingUpdate(NamedTuple):
    position: int
    result: torch.fx.Node
    operands: tuple
    gathers: dict
    vjp_reads: tuple
    normalization: tuple


def _gather_args(node, source, *, single_row):
    aten = torch.ops.aten
    if (
        node.op != "call_function"
        or not node.args
        or node.kwargs
        or node.args[0] is not source
    ):
        return None
    if node.target is aten.index_select.default and len(node.args) == 3:
        _, axis, index = node.args
        if axis != 0:
            return None
    elif node.target is aten.index.Tensor and len(node.args) == 2:
        _, indices = node.args
        if not isinstance(indices, (list, tuple)) or len(indices) != 1:
            return None
        (index,) = indices
    else:
        return None
    value = index.meta.get("val") if isinstance(index, torch.fx.Node) else None
    if (
        isinstance(value, FakeTensor)
        and value.ndim == 1
        and (not single_row or value.shape == (1,))
        and value.dtype == torch.int64
        and value.device.type == "cpu"
    ):
        return source, index, node.target is aten.index.Tensor
    return None


def _adjoint_update(result, ring):
    aten = torch.ops.aten
    # Only consider the normalized single-row update produced by the local
    # adjoint pass. Inductor may already have reinplaced its fresh base.
    if (
        not result.meta.get("dendra_ring_adjoint")
        or result.target not in (aten.index_put.default, aten.index_put_.default)
        or len(result.args) != 3
        or result.kwargs
    ):
        return None
    normalized, indices, row = result.args
    if (
        not isinstance(normalized, torch.fx.Node)
        or normalized.target is not aten.where.ScalarSelf
        or len(normalized.args) != 3
        or tuple(normalized.users) != (result,)
        or not isinstance(indices, (list, tuple))
        or len(indices) != 1
    ):
        return None
    eq, _, add = normalized.args
    if (
        not isinstance(eq, torch.fx.Node)
        or not isinstance(add, torch.fx.Node)
        or eq.target is not aten.eq.Scalar
        or add.target is not aten.add.Scalar
        or eq.args != (ring, 0.0)
        or add.args != (ring, 0.0)
        or normalized.args[1] != 0.0
        or tuple(eq.users) != (normalized,)
        or tuple(add.users) != (normalized,)
    ):
        return None
    return (ring, indices[0], row), (normalized, eq, add)


def _ring_gathers(ring, result, normalization, order):
    aten = torch.ops.aten
    gathers = {}
    vjp_reads = []
    fused_vjp = getattr(torch.ops.dendra_native, "delay_ring_two_tap_vjp", None)
    for user in ring.users:
        if user is result or user in normalization:
            continue
        args = _gather_args(user, ring, single_row=not normalization)
        if args is not None:
            if not normalization and order[user] >= order[result]:
                return None
            gathers[user] = args
            continue
        if (
            normalization
            and fused_vjp is not None
            and user.op == "call_function"
            and user.target is fused_vjp.default
            and user.args
            and not user.kwargs
            and user.args[0] is ring
            and ring not in user.args[1:]
        ):
            # The pure VJP reads the old ring. Keep its tensor results as
            # explicit dependencies of the in-place queue update.
            for component in user.users:
                if (
                    component.op != "call_function"
                    or component.target is not operator.getitem
                    or not isinstance(component.meta.get("val"), FakeTensor)
                ):
                    return None
                vjp_reads.append(component)
            continue
        # The enqueue adjoint reads two taps through a flattened ring.
        flat = user.meta.get("val")
        if (
            not normalization
            or user.op != "call_function"
            or user.target not in (aten.view.default, aten.reshape.default)
            or user.kwargs
            or len(user.args) != 2
            or user.args[0] is not ring
            or not isinstance(flat, FakeTensor)
            or flat.shape != (ring.meta["val"].numel(),)
            or not flat.is_contiguous()
        ):
            return None
        for reader in user.users:
            args = _gather_args(reader, user, single_row=False)
            if args is None:
                return None
            gathers[reader] = args
    return gathers, tuple(vjp_reads)


def _index_vector(node):
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    if (
        isinstance(value, FakeTensor)
        and value.device.type == "cpu"
        and value.dtype == torch.int64
        and value.ndim == 1
        and type(value.shape[0]) is int
        and value.is_contiguous()
        and not value.is_neg()
        and not torch._is_zerotensor(value)
    ):
        return value
    return None


def _tap_address(index, ring_size, width):
    aten = torch.ops.aten
    if (
        index.target is not aten.remainder.Scalar
        or len(index.args) != 2
        or index.kwargs
        or type(index.args[1]) is not int
        or index.args[1] != ring_size
    ):
        return None
    add = index.args[0]
    if (
        not isinstance(add, torch.fx.Node)
        or add.target is not aten.add.Tensor
        or len(add.args) != 2
        or add.kwargs
    ):
        return None
    base, offset = add.args
    if (
        _index_vector(base) is None
        or _index_vector(offset) is None
        or offset.meta["val"].shape != (1,)
        or offset.target not in (aten.mul.Tensor, aten.mul.Scalar)
        or len(offset.args) != 2
        or offset.kwargs
        or type(offset.args[1]) is not int
        or offset.args[1] != width
    ):
        return None
    return base, offset, add


def _fuse_two_tap_reads(graph, ring, gathers):
    native = torch.ops.dendra_native
    if not hasattr(native, "delay_ring_gather_two_tap"):
        return ()
    value = ring.meta["val"]
    if (
        value.dtype not in (torch.float32, torch.float64)
        or not all(type(size) is int and size > 0 for size in value.shape)
        or value.is_neg()
        or torch._is_zerotensor(value)
    ):
        return ()

    flat_reads = [
        (reader, args) for reader, args in gathers.items() if args[0] is not ring
    ]
    if len(flat_reads) != 2:
        return ()
    order = {node: i for i, node in enumerate(graph.nodes)}
    (left, left_args), (right, right_args) = sorted(
        flat_reads, key=lambda item: order[item[0]]
    )
    if left_args[0] is not right_args[0]:
        return ()
    left_index, right_index = left_args[1], right_args[1]
    left_address = _tap_address(left_index, value.numel(), value.shape[1])
    right_address = _tap_address(right_index, value.numel(), value.shape[1])
    if left_address is None or right_address is None:
        return ()
    left_base, offset, left_add = left_address
    right_base, right_offset, right_add = right_address
    if (
        offset is not right_offset
        or left_base.meta["val"].shape != right_base.meta["val"].shape
        or any(order[node] >= order[left] for node in (offset, left_base, right_base))
    ):
        return ()
    region = get_mutation_region_id(graph, left)
    if any(
        get_mutation_region_id(graph, node) != region
        for node in (left_add, right_add, left_index, right_index, right)
    ):
        return ()

    # Realize both reads before the private ring update. The existing outputs
    # remain separate dependencies, so gradient arithmetic and order are intact.
    with graph.inserting_before(left):
        pair = graph.call_function(
            native.delay_ring_gather_two_tap.default,
            (ring, offset, left_base, right_base),
        )
    pair.meta["val"] = (left.meta["val"], right.meta["val"])
    for position, reader in enumerate((left, right)):
        reader.target = operator.getitem
        reader.args = (pair, position)
    for node in dict.fromkeys((left_index, right_index, left_add, right_add)):
        if not node.users:
            graph.erase_node(node)
    return left, right


def _owned_ring_carries(body, count, forward_ops):
    placeholders = list(body.graph.find_nodes(op="placeholder"))
    (output,) = body.graph.find_nodes(op="output")
    returned = output.args[0]
    if (
        not isinstance(returned, (list, tuple))
        or min(len(returned), len(placeholders)) < count
    ):
        return
    return_counts = Counter(
        value for value in returned if isinstance(value, torch.fx.Node)
    )
    order = {node: i for i, node in enumerate(body.graph.nodes)}
    for position in range(count):
        ring = placeholders[position]
        result = returned[position]
        value = ring.meta.get("val")
        if (
            not isinstance(result, torch.fx.Node)
            or tuple(result.users) != (output,)
            or return_counts[result] != 1
            or not isinstance(value, FakeTensor)
            or value.device.type != "cpu"
            or value.ndim != 2
            or not value.is_contiguous()
        ):
            continue
        if result.target in forward_ops:
            if (
                len(result.args) != forward_ops[result.target][1]
                or result.kwargs
                or ring is not result.args[0]
                or ring in result.args[1:]
            ):
                continue
            operands, normalization = result.args, ()
        else:
            adjoint = _adjoint_update(result, ring)
            if adjoint is None:
                continue
            operands, normalization = adjoint
        reads = _ring_gathers(ring, result, normalization, order)
        if reads is not None:
            yield _RingUpdate(position, result, operands, *reads, normalization)


def _add_first_step(graph, scan, body, count, xs):
    if (
        not isinstance(xs, (list, tuple))
        or not xs
        or not isinstance(xs[0], torch.fx.Node)
    ):
        return None
    example = xs[0].meta.get("val")
    if (
        not isinstance(example, FakeTensor)
        or example.device.type != "cpu"
        or not example.ndim
        or type(example.shape[0]) is not int
        or example.shape[0] < 1
    ):
        return None
    steps = example.shape[0]
    aten = torch.ops.aten
    with graph.inserting_before(scan):
        indices = graph.call_function(
            aten.arange.start_step,
            (0, steps, 1),
            {"dtype": torch.int64, "device": example.device},
        )
        first = graph.call_function(aten.eq.Scalar, (indices, 0))
    with example.fake_mode:
        indices.meta["val"] = torch.arange(steps, device=example.device)
        first.meta["val"] = indices.meta["val"] == 0

    # Scan flattens body operands as carries, samples, then lifted inputs.
    position = count + len(xs)
    with body.graph.inserting_before(list(body.graph.nodes)[position]):
        sample = body.graph.placeholder("first_ring_step")
        sample.meta["val"] = first.meta["val"][0]
    return first, sample


def _supports_private_ring_mutation():
    from torch._inductor import config, ir

    if config.cpp_wrapper or config.fx_wrapper:
        return False
    create = getattr(getattr(ir, "WhileLoop", None), "create", None)
    code = getattr(create, "__code__", None)
    # This lowering needs Inductor to mark loop mutations and to return
    # the mutated operand as the corresponding final carry value.
    mutation_protocol = {"MutationOutput", "check_input_alias_and_mutation"}
    return code is not None and mutation_protocol.issubset(code.co_names)


def _lower_private_rings(graph):
    native = torch.ops.dendra_native
    owner = graph.owning_module
    if (
        owner is None
        or not hasattr(native, "ordered_two_tap_enqueue_")
        or not _supports_private_ring_mutation()
    ):
        return
    forward_ops = {
        native.ordered_two_tap_enqueue.default: (
            native.ordered_two_tap_enqueue_.default,
            7,
        ),
        native.ordered_two_tap_enqueue_from_source.default: (
            native.ordered_two_tap_enqueue_from_source_.default,
            8,
        ),
    }
    adjoint_in_place = native.delay_ring_adjoint_update_.default
    read_row = native.delay_ring_read_row.default
    attributes = Counter(node.target for node in graph.find_nodes(op="get_attr"))
    modules = dict(owner.named_modules(remove_duplicate=False))
    module_uses = Counter(id(module) for module in modules.values())
    for scan in graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.scan
    ):
        if len(scan.args) != 4 or scan.kwargs:
            continue
        ref, initial, xs, lifted = scan.args
        if (
            not isinstance(ref, torch.fx.Node)
            or ref.op != "get_attr"
            or tuple(ref.users) != (scan,)
            or attributes[ref.target] != 1
            or not isinstance(initial, (tuple, list))
        ):
            continue
        body = modules.get(ref.target)
        if not isinstance(body, torch.fx.GraphModule) or module_uses[id(body)] != 1:
            continue
        matches = list(_owned_ring_carries(body, len(initial), forward_ops))
        first = None
        if any(match.normalization for match in matches):
            first = _add_first_step(graph, scan, body, len(initial), xs)
            if first is None:
                matches = [match for match in matches if not match.normalization]
            else:
                first_tape, first = first
                xs = (*xs, first_tape)
        if not matches:
            continue
        (output,) = body.graph.find_nodes(op="output")
        returned = list(output.args[0])
        initial = list(initial)
        for position, result, operands, gathers, vjp_reads, normalization in matches:
            old = initial[position]
            with graph.inserting_before(scan):
                # The incoming pack can belong to an earlier chunk or
                # be saved for its backward. Own this loop's buffer.
                owned = graph.call_function(
                    torch.ops.aten.clone.default,
                    (old,),
                    {"memory_format": torch.contiguous_format},
                )
            owned.meta = dict(old.meta)
            owned.meta["val"] = old.meta["val"].clone(
                memory_format=torch.contiguous_format
            )
            initial[position] = owned

            fused_reads = (
                _fuse_two_tap_reads(body.graph, operands[0], gathers)
                if normalization
                else ()
            )
            for reader, args in gathers.items():
                if reader in fused_reads:
                    continue
                # An opaque read forces realization before the mutation.
                # A fused gather could otherwise be recomputed afterwards.
                reader.target = read_row
                reader.args = args
            if normalization:
                # The old ring may still be used after the functional update
                # in FX order. All gathers need to observe the incoming value.
                insertion = output
                operation = adjoint_in_place
                operands = (*operands, first)
            else:
                insertion = result
                operation, _ = forward_ops[result.target]
            with body.graph.inserting_before(insertion):
                changed = body.graph.call_function(
                    operation, (*operands, (*gathers, *vjp_reads))
                )
            changed.meta["val"] = None
            returned[position] = operands[0]
            output.args = (tuple(returned),)
            body.graph.erase_node(result)
            for retired in normalization:
                body.graph.erase_node(retired)

        scan.args = (ref, tuple(initial), xs, lifted)
        body.graph.lint()
        body.recompile()
