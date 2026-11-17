"""Keep reverse scans on their original, chronological input tapes."""

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    get_mutation_region_id,
    register_graph_pattern,
)
from torch._subclasses.fake_tensor import FakeTensor

from ._scan_executor import _ALIASES

_STORAGE_SENSITIVE_OPS = frozenset(
    (
        "aten::_has_compatible_shallow_copy_type",
        "aten::_has_same_storage_numel",
        "aten::_is_zerotensor",
        "aten::_version",
        "aten::is_pinned",
        "aten::is_set_to",
        "aten::set",
        "aten::set_",
        "aten::shallow_copy_data_",
        "aten::storage_offset",
        "aten::sym_storage_offset",
    )
)

_NATIVE_LOGICAL_READS = frozenset(
    (
        "dendra_native::delay_ring_read_row",
        "dendra_native::delay_ring_gather_two_tap",
        "dendra_native::delay_ring_two_tap_vjp",
        "dendra_native::delay_ring_adjoint_update_",
        "dendra_native::ordered_two_tap_enqueue",
        "dendra_native::ordered_two_tap_enqueue_",
        "dendra_native::ordered_two_tap_enqueue_from_source",
        "dendra_native::ordered_two_tap_enqueue_from_source_",
    )
)


def _reverse_scan_enabled():
    from torch._inductor import config

    return (
        config.use_post_grad_passes and not config.cpp_wrapper and not config.fx_wrapper
    )


def _static_tensor(value):
    return (
        isinstance(value, FakeTensor)
        and value.layout is torch.strided
        and not value.is_nested
        and value.device.type == "cpu"
        and all(type(dim) is int for dim in (*value.shape, *value.stride()))
    )


def _logical_reads(sample):
    # A physical row from the original tape may have a different storage
    # offset. Follow logical views and reject storage-sensitive or escaping uses.
    pending = [sample]
    visited = set()
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        for user in value.users:
            target = user.target
            if user.op != "call_function" or not isinstance(
                target, torch._ops.OpOverload
            ):
                return False
            schema = target._schema
            if (
                target.namespace not in ("aten", "prims")
                and schema.name not in _NATIVE_LOGICAL_READS
            ):
                return False
            if (
                schema.name in _STORAGE_SENSITIVE_OPS
                or "as_strided" in schema.name
                or target is torch.ops.aten._unsafe_view.default
            ):
                return False
            if not schema.is_mutable and not any(
                item.alias_info for item in schema.returns
            ):
                continue
            # Mutable calls can read the sample through an ordinary by-value
            # argument, e.g. a delay slot used to update a separate private ring.
            aliases = []
            for index, argument in enumerate(schema.arguments):
                bound = (
                    user.args[index]
                    if index < len(user.args)
                    else user.kwargs.get(argument.name)
                )
                if argument.alias_info and any(
                    leaf is value for leaf in torch.utils._pytree.tree_leaves(bound)
                ):
                    aliases.append(argument.alias_info)
            if any(info.is_write for info in aliases):
                return False
            if aliases and any(item.alias_info for item in schema.returns):
                if id(target) not in _ALIASES:
                    return False
                pending.append(user)
    return True


def _reversed_source(scan, operand, sample, scan_inputs):
    if (
        not isinstance(operand, torch.fx.Node)
        or operand.op != "call_function"
        or operand.target
        not in (torch.ops.prims.rev.default, torch.ops.aten.flip.default)
        or len(operand.args) != 2
        or operand.kwargs
        or not isinstance(operand.args[1], (tuple, list))
        or tuple(operand.args[1]) != (0,)
        or tuple(operand.users) != (scan,)
    ):
        return None
    source = operand.args[0]
    if not isinstance(source, torch.fx.Node):
        return None
    raw, reverse = source.meta.get("val"), operand.meta.get("val")
    if (
        not _static_tensor(raw)
        or not _static_tensor(reverse)
        or raw.ndim < 1
        or raw.shape != reverse.shape
        or raw.stride() != reverse.stride()
        or raw.dtype != reverse.dtype
        or torch._is_zerotensor(raw)
        or raw.is_neg()
        or raw.is_conj()
        or raw.storage_offset() != 0
        or reverse.storage_offset() != 0
        or get_mutation_region_id(scan.graph, operand)
        != get_mutation_region_id(scan.graph, scan)
        or not _logical_reads(sample)
    ):
        return None
    # A materialized flip owns separate storage. Keep that separation when
    # any other scan operand can share the source's storage.
    storage = raw.untyped_storage()._cdata
    if any(
        isinstance(value, FakeTensor)
        and value.device == raw.device
        and (
            value.is_nested
            or value.layout is not torch.strided
            or value.untyped_storage()._cdata == storage
        )
        for value in scan_inputs
    ):
        return None
    return source


def _scan_reversal(scan):
    if len(scan.args) != 4 or scan.kwargs:
        return None
    ref, initial, tapes, lifted = scan.args
    owner = scan.graph.owning_module
    if (
        owner is None
        or not isinstance(ref, torch.fx.Node)
        or ref.op != "get_attr"
        or not all(
            isinstance(group, (tuple, list)) for group in (initial, tapes, lifted)
        )
        or not tapes
    ):
        return None
    body = getattr(owner, ref.target, None)
    if not isinstance(body, torch.fx.GraphModule):
        return None
    placeholders = body.graph.find_nodes(op="placeholder")
    if len(placeholders) != len(initial) + len(tapes) + len(lifted):
        return None
    inputs = [tape.meta.get("val") for tape in tapes if isinstance(tape, torch.fx.Node)]
    if len(inputs) != len(tapes) or any(
        not _static_tensor(x) or not x.ndim for x in inputs
    ):
        return None
    steps = inputs[0].shape[0]
    if steps <= 1 or any(value.shape[0] != steps for value in inputs):
        return None
    results = scan.meta.get("val")
    if not isinstance(results, (list, tuple)) or len(results) < len(initial):
        return None
    emissions = results[len(initial) :]
    if any(not _static_tensor(value) for value in emissions):
        return None
    scan_inputs = [
        node.meta.get("val")
        for node in (*initial, *tapes, *lifted)
        if isinstance(node, torch.fx.Node)
    ]
    sources = tuple(
        _reversed_source(scan, operand, sample, scan_inputs)
        for operand, sample in zip(
            tapes, placeholders[len(initial) : len(initial) + len(tapes)], strict=True
        )
    )
    if any(source is not None for source in sources):
        return body, steps, emissions, sources
    return None


class _RetainChronologicalScanTapes(CustomGraphPass):
    """Keep eligible reversal nodes on the backward side of AOT partitioning."""

    def uuid(self):
        return get_hash_for_files((__file__,))

    def __call__(self, graph):
        from torch.utils.checkpoint import CheckpointPolicy

        if not _reverse_scan_enabled():
            return
        for scan in graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.scan
        ):
            if scan.meta.get("autograd_backward") is not True:
                continue
            eligible = _scan_reversal(scan)
            if eligible is None:
                continue
            for operand, source in zip(scan.args[2], eligible[-1], strict=True):
                if source is not None and operand.meta.get("recompute") is None:
                    operand.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE


def _lower_reversed_scans(graph):
    if not _reverse_scan_enabled() or graph.owning_module is None:
        return

    patterns = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.scan), pass_dict=patterns
    )
    def lower(match, *args, **kwargs):
        eligible = _scan_reversal(match.nodes[0])
        if eligible is None:
            return
        _, initial, tapes, lifted = match.nodes[0].args
        body, steps, emissions, sources = eligible
        arguments = tuple(
            source if source is not None else tape
            for tape, source in zip(tapes, sources, strict=True)
        )
        reversed_inputs = tuple(source is not None for source in sources)
        carry_count = len(initial)
        emission_count = len(emissions)
        tape_count = len(tapes)

        def reverse_index_loop(*args):
            index = torch.zeros((), dtype=torch.int64, device="cpu")
            outputs = tuple(
                torch.empty_strided(
                    value.shape,
                    value.stride(),
                    dtype=value.dtype,
                    device=value.device,
                    requires_grad=value.requires_grad,
                )
                for value in emissions
            )
            loop_state = (index, *outputs, *args[:carry_count])
            operands = args[carry_count:]
            carried_count = len(loop_state)

            def cond(tick, *_):
                return tick < steps

            def step(tick, *values):
                i = tick.item()
                torch.ops.aten._assert_scalar.default(i >= 0, "")
                torch.ops.aten._assert_scalar.default(i < steps, "")
                stacks = values[:emission_count]
                carry = values[emission_count : carried_count - 1]
                rest = values[carried_count - 1 :]
                samples = tuple(
                    tape.select(0, steps - 1 - i if reverse else i)
                    for tape, reverse in zip(
                        rest[:tape_count], reversed_inputs, strict=True
                    )
                )
                updated = body(*carry, *samples, *rest[tape_count:])
                for out, value in zip(stacks, updated[carry_count:], strict=True):
                    out.select(0, i).copy_(value)
                return tick + 1, *stacks, *updated[:carry_count]

            final = torch.ops.higher_order.while_loop(cond, step, loop_state, operands)
            return ([*final[1 + emission_count :], *final[1 : 1 + emission_count]],)

        match.replace_by_example(
            reverse_index_loop,
            (*initial, *arguments, *lifted),
            run_functional_passes=False,
        )
        for tape, source in zip(tapes, sources, strict=True):
            if source is not None and not tape.users:
                graph.erase_node(tape)

    patterns.apply(graph)
