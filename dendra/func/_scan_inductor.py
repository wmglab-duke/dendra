"""Local Inductor lowering for scan body and adjoint operators.

The scan partitioner functionalizes views into copy operators. Lower these
operators after autograd partitioning so the backward scan retains its time
axis even when its derivatives use only loop-invariant operands. Explicit
views followed by independent contiguous clones preserve copy semantics,
conditional operand layouts, and normal Inductor fusion. The graph pass and
authored-derivative decompositions belong only to this backend. The local
post-grad rewrite also combines disjoint queue-row cotangents.
"""

from __future__ import annotations

import math

import torch
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    get_custom_graph_passes,
    get_hash_for_files,
)

from dendra._bootstrap import compile_owned, isolate_compilation_callable

from . import _scan_private_rings, _scan_reverse


def _row_write(node):
    aten = torch.ops.aten
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not aten.index_put.default
        or len(node.args) not in (3, 4)
        or set(node.kwargs) - {"accumulate"}
        or (len(node.args) == 4 and node.kwargs)
    ):
        return None
    ring, indices, value = node.args[:3]
    accumulate = (
        node.args[3] if len(node.args) == 4 else node.kwargs.get("accumulate", False)
    )
    if (
        type(accumulate) is not bool
        or not isinstance(indices, (list, tuple))
        or len(indices) != 1
        or not all(
            isinstance(item, torch.fx.Node) for item in (ring, indices[0], value)
        )
    ):
        return None
    return ring, indices[0], value, accumulate


def _copied_index(node):
    while node.op == "call_function" and node.target is torch.ops.aten.clone.default:
        node = node.args[0]
    return node


def _zero_terms(node):
    # Track only owned positive zeros, optionally broadcast to the row.
    aten = torch.ops.aten
    result = []
    while node.op == "call_function":
        result.append(node)
        if node.target is aten.zeros.default:
            return result
        if node.target is aten.full.default:
            value = node.args[1]
            break
        if node.target is aten.scalar_tensor.default:
            value = node.args[0]
            break
        if node.target not in (aten.expand.default, aten.clone.default):
            return ()
        node = node.args[0]
    else:
        return ()
    return (
        result
        if type(value) in (int, float) and value == 0 and math.copysign(1, value) == 1
        else ()
    )


def _match_queue_row_adjoint(node):
    if len(node.args) != 2 or set(node.kwargs) - {"alpha"}:
        return None
    scale = node.kwargs.get("alpha", 1)
    if type(scale) not in (int, float) or scale != 1:
        return None
    clear, inject = node.args
    first, second = _row_write(clear), _row_write(inject)
    if first is None or second is None:
        return None
    if first[3]:
        clear, inject = inject, clear
        first, second = second, first
    ring, slot, zero_row, clears_accumulate = first
    zeros, due_slot, due, adds_accumulate = second
    if (
        clears_accumulate
        or not adds_accumulate
        or len(clear.users) != 1
        or len(inject.users) != 1
        or _copied_index(slot) is not _copied_index(due_slot)
    ):
        return None
    dead_zeros = _zero_terms(zero_row)
    if not dead_zeros:
        return None
    zero_base = _zero_terms(zeros)
    if not zero_base:
        return None
    operands = (ring, due, zeros, node, slot, due_slot, zero_row)
    values = [operand.meta.get("val") for operand in operands]
    if not all(torch.is_tensor(value) for value in values):
        return None
    ring_val, row_val, zero_val, out_val, slot_val, due_slot_val, fill_val = values
    shape = tuple(ring_val.shape)
    if (
        ring_val.device.type != "cpu"
        or ring_val.dtype not in (torch.float32, torch.float64)
        or len(shape) != 2
        or any(type(size) is not int or size < 1 for size in shape)
        or not ring_val.is_contiguous()
        or row_val.shape != (1, shape[1])
        or fill_val.shape not in ((), row_val.shape)
        or any(
            value.shape != shape or value.stride() != ring_val.stride()
            for value in (zero_val, out_val)
        )
        or any(
            value.dtype != ring_val.dtype or value.device != ring_val.device
            for value in (row_val, zero_val, out_val, fill_val)
        )
        or any(
            value.shape != (1,)
            or value.dtype != torch.int64
            or value.device != ring_val.device
            for value in (slot_val, due_slot_val)
        )
    ):
        return None
    from torch._inductor.pattern_matcher import get_mutation_region_id

    region = get_mutation_region_id(node.graph, node)
    reads = (clear, inject, *dead_zeros, *zero_base)
    if slot is not due_slot:
        reads += (slot, due_slot)
    if any(get_mutation_region_id(node.graph, read) != region for read in reads):
        return None
    return ring, slot, due, (node, clear, inject, *dead_zeros, *zero_base)


def _add_positive_zero(graph, value):
    aten = torch.ops.aten
    # Inductor can fold x + 0, so explicitly normalize either sign of zero.
    is_zero = graph.call_function(aten.eq.Scalar, (value, 0.0))
    summed = graph.call_function(aten.add.Scalar, (value, 0.0))
    return graph.call_function(aten.where.ScalarSelf, (is_zero, 0.0, summed))


def _fuse_queue_row_adjoint(graph):
    # A cleared carry row and an injection into otherwise-zero storage have
    # disjoint support. Use the immutable incoming cotangent for any other
    # reads (including parameter VJPs), and replace just the delivered row.
    aten = torch.ops.aten
    for node in graph.find_nodes(op="call_function", target=aten.add.Tensor):
        match = _match_queue_row_adjoint(node)
        if match is None:
            continue
        ring, slot, due, retired = match
        with graph.inserting_before(node):
            # The reference adds positive zero on both sides of the row. Keep
            # the same positive-zero result alongside ordinary/nonfinite values.
            carry = _add_positive_zero(graph, ring)
            row = _add_positive_zero(graph, due)
            output = graph.call_function(aten.index_put.default, (carry, [slot], row))
            output.meta = {**node.meta, "dendra_ring_adjoint": True}
        node.replace_all_uses_with(output)
        for previous in dict.fromkeys(retired):
            if not previous.users:
                graph.erase_node(previous)


class _CopyViewLowering(CustomGraphPass):
    """Apply the scan's local rewrites after AOT has saved the time axis."""

    def uuid(self):
        return get_hash_for_files(
            (__file__, _scan_private_rings.__file__, _scan_reverse.__file__)
        )

    def __call__(self, graph):
        targets = {
            torch.ops.aten.select_copy.int: torch.ops.aten.select.int,
            torch.ops.aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
            torch.ops.aten.as_strided_copy.default: torch.ops.aten.as_strided.default,
        }
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in targets:
                continue
            with graph.inserting_before(node):
                view = graph.call_function(targets[node.target], node.args, node.kwargs)
                result = graph.call_function(
                    torch.ops.aten.clone.default,
                    (view,),
                    {"memory_format": torch.contiguous_format},
                )
                # Only the clone has the original copy's layout. Inductor's
                # following fake-tensor update computes the new view metadata.
                result.meta = dict(node.meta)
            node.replace_all_uses_with(result)
            graph.erase_node(node)
        _fuse_queue_row_adjoint(graph)
        _scan_private_rings._lower_private_rings(graph)
        _scan_reverse._lower_reversed_scans(graph)


class _ScanInductorBackend(torch._TorchCompileInductorWrapper):
    """Keep Inductor's lifecycle with local decompositions and input protection."""

    def __eq__(self, other):
        # Do not let Dynamo equate this lowering with ordinary Inductor merely
        # because their user options match.
        return type(self) is type(other) and super().__eq__(other)

    def __call__(self, graph, inputs, *, config_patches=None):
        from torch._inductor import config
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.decomposition import select_decomp_table
        from torch._inductor.utils import parallel_num_threads

        from dendra.models.networks._two_tap_enqueue import _compile_two_tap_vjp

        patches = {**self.config, **(config_patches or {})}
        with config.patch(patches):
            fuse_vjp = (
                config.use_post_grad_passes
                and not config.cpp_wrapper
                and not config.fx_wrapper
                and not config.cpp.enable_unsafe_math_opt_flag
                and config.cpp.enable_floating_point_contract_flag == "off"
            )
            ordered_sources = (
                not config.cpp.dynamic_threads and parallel_num_threads() == 1
                if fuse_vjp
                else None
            )
            decompositions = dict(select_decomp_table())
            from ._scan_donation import _scan_donation_pass

            donation_receipts = getattr(self, "_donation_receipts", None)
            if donation_receipts is None:
                donation_receipts = self._donation_receipts = []
            patches["_pre_fusion_custom_pass"] = _scan_donation_pass(
                patches.get("_pre_fusion_custom_pass", config._pre_fusion_custom_pass),
                receipts=donation_receipts,
            )
            previous = patches.get(
                "post_grad_custom_post_pass", config.post_grad_custom_post_pass
            )
            # Keep user callbacks intact, including their inference-aware
            # protocol and cache identity. Torch applies this pass to HOP
            # subgraphs as well as their containing forward/backward graphs.
            patches["post_grad_custom_post_pass"] = (
                *get_custom_graph_passes(previous),
                _CopyViewLowering(),
            )
            if _scan_reverse._reverse_scan_enabled():
                joint = patches.get(
                    "joint_custom_post_pass", config.joint_custom_post_pass
                )
                patches["joint_custom_post_pass"] = (
                    *get_custom_graph_passes(joint),
                    _scan_reverse._RetainChronologicalScanTapes(),
                )
            # CPU outlining must remain dormant for GPU/mixed-device graphs.
            # Scalar and symbolic operands carry no model-device ownership.
            try:
                tensor_devices = tuple(
                    value.device.type
                    for value in inputs
                    if isinstance(value, torch.Tensor)
                )
            except Exception:
                tensor_devices = ()
            if tensor_devices and all(device == "cpu" for device in tensor_devices):
                from ._while_outlining import _scan_outlining_patches

                patches = _scan_outlining_patches(patches)
        from ._dufort_autograd import _inductor_decompositions

        decompositions.update(_inductor_decompositions())
        with _compile_two_tap_vjp(ordered_sources):
            return compile_fx(
                graph,
                inputs,
                config_patches=patches,
                decompositions=decompositions,
                compile_region_name=self.name,
            )

    def _donation_diagnostics(self):
        from copy import deepcopy

        return deepcopy(tuple(getattr(self, "_donation_receipts", ())))


def _uses_local_scan_inductor(options):
    selected_backend = options.get("backend")
    if selected_backend is None:
        from torch._dynamo.backends.registry import get_default_backend

        selected_backend = get_default_backend()
    return (
        selected_backend == "inductor"
        and not options.get("disable", False)
        and not (options.get("options") or {}).get("use_aoti", False)
    )


def _compile_scan(function, compile_options, *, prefer_cpp_wrapper=False):
    """Use the local lowering only when ordinary Inductor was selected.

    Custom backends, the configured non-Inductor default, disabled compilation,
    and AOTInductor keep their original routes. The incoming option dictionary
    is not modified, including its nested ``options`` dictionary.
    """
    function = isolate_compilation_callable(function)
    options = dict(compile_options)
    if options.get("options") is not None:
        options["options"] = dict(options["options"])
    if not _uses_local_scan_inductor(options):
        return compile_owned(function, **options)
    inductor_options = dict(options.get("options") or {})
    mode = options.pop("mode", None)
    if mode is not None and options.get("options") is not None:
        raise RuntimeError(
            "Either mode or options can be specified, but both can't be "
            "specified at the same time."
        )
    options.pop("options", None)
    # These options belong to torch.compile, not to Inductor's config parser.
    # Forward them through its outer boundary exactly as for the normal backend.
    outer_options = {}
    for name in ("guard_filter_fn", "use_aoti"):
        if name in inductor_options:
            outer_options[name] = inductor_options.pop(name)
    if outer_options:
        options["options"] = outer_options
    backend = _ScanInductorBackend(
        mode,
        inductor_options,
        options.get("dynamic"),
        options.get("name"),
    )
    if prefer_cpp_wrapper:
        from torch._inductor import config

        # Normalize user options first, including Torch's hyphen aliases. An
        # explicit wrapper choice wins; alternate/disabled C++ code generation
        # also keeps its original route.
        if not (
            backend.config.get("disable_cpp_codegen", config.disable_cpp_codegen)
            or backend.config.get("fx_wrapper", config.fx_wrapper)
        ):
            backend.config.setdefault("cpp_wrapper", True)
    options["backend"] = backend
    compiled = compile_owned(function, **options)
    # Torch's generic callable-backend wrapper forwards reset(), but not the
    # compiler-config provider. Preserve the standard compiled function API.
    compiled.get_compiler_config = backend.get_compiler_config
    compiled._dendra_scan_donation_diagnostics = backend._donation_diagnostics
    return compiled
