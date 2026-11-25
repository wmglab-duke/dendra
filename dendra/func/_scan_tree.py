"""Shared owned tensor-tree boundaries for fixed-horizon compiled scans.

Population and Network adapters provide only their authored tensor step. The
capture, packed carry and scan lowering here preserve one set of AD and
ownership rules across those adapters.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from .._solver_failures import _advance_solver_failure, _SolverFailureCapture
from ..helpers import MEMBRANE_ASSEMBLY
from ._callback_carry import _CallbackCarryPartition
from ._carry_packing import _PackedCarryLayout
from ._hop_inputs import (
    _dense_zero_offset,
    _input_usage,
    _isolate_dense_lifted_inputs,
    _LiftedViewIsolation,
    _own_runtime_draw_packets,
)
from ._scan_autograd import guard_scan_outputs
from ._scan_compat import (
    _capture_saved_tensor_identity,
    install_scan_compatibility,
)
from ._scan_executor import _ALIASES, _ALLOCATIONS, _executor_key
from ._scan_inductor import _compile_scan, _uses_local_scan_inductor
from ._state_projection import _plain_projection_context, _plain_state_tree
from ._tensor_contracts import _clone_contiguous_tree as _clone_tree
from ._tensor_contracts import capture_schema as _schema
from ._tree_paths import _get_path

_CPU_TRAINING_SCAN_LARGE_LEAF_BYTES = 1 << 20


def _operator_schema(node):
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
    ):
        return node.target._schema
    return None


def _fresh_result(node):
    schema = _operator_schema(node)
    return (
        schema is not None
        and not schema.is_mutable
        and len(schema.returns) == 1
        and schema.returns[0].alias_info is None
    )


def _logical_carry_reads(source):
    pending = [source]
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        for user in node.users:
            if user.op == "output":
                continue
            schema = _operator_schema(user)
            if (
                schema is None
                or schema.is_mutable
                or user.target
                in (
                    torch.ops.aten.as_strided_copy.default,
                    torch.ops.aten._unsafe_view.default,
                    torch.ops.aten.lift.default,
                )
            ):
                return False
            if any(value.alias_info is not None for value in schema.returns):
                if id(user.target) not in _ALIASES:
                    return False
                pending.append(user)
    return True


def _returns_alias(result, returned):
    # Fake tensor storage keys are unreliable for pre-dispatch view capture.
    # Follow returned view/unknown chains until a schema promises fresh storage.
    pending = [node for node in returned if isinstance(node, torch.fx.Node)]
    seen = set()
    while pending:
        node = pending.pop()
        if node is result:
            return True
        if node in seen:
            continue
        seen.add(node)
        if not (
            _fresh_result(node)
            and (
                node.target.namespace not in ("aten", "prims", "prim")
                or id(node.target) in _ALLOCATIONS
            )
        ):
            pending.extend(node.all_input_nodes)
    return False


def _direct_carry_layout(graph, layout, parameters, prepared):
    """Use shaped private carry for large, layout-preserving opaque results."""
    if not isinstance(graph, torch.fx.GraphModule):
        return layout
    out_spec = getattr(graph, "_out_spec", None)
    if out_spec is None or out_spec.num_children != 2:
        return layout
    if out_spec.children()[0] != layout.spec:
        return layout
    nodes = list(graph.graph.nodes)
    returned = nodes[-1].args[0]
    if not isinstance(returned, (list, tuple)) or len(returned) != out_spec.num_leaves:
        return layout
    outputs = returned[: len(layout.leaves)]
    placeholders = [node for node in nodes if node.op == "placeholder"]
    start = len(pytree.tree_leaves((parameters, prepared)))
    inputs = placeholders[start : start + len(layout.leaves)]
    if len(outputs) != len(inputs) or len(inputs) != len(layout.leaves):
        return layout
    emissions = returned[len(layout.leaves) :]
    groups = []
    # The dedicated pack already has offset zero and the authored logical
    # strides. Keep shared packs and their observable view layouts intact.
    for group, indices in enumerate(layout.indices):
        if len(indices) != 1:
            continue
        index = indices[0]
        source, result = inputs[index], outputs[index]
        original = source.meta.get("val")
        if (
            not torch.is_tensor(original)
            or original.device.type != "cpu"
            or original.numel() * original.element_size()
            < _CPU_TRAINING_SCAN_LARGE_LEAF_BYTES
            or not _dense_zero_offset(original)
            or not _fresh_result(result)
            or result.target.namespace in ("aten", "prims", "prim")
        ):
            continue
        value = result.meta.get("val")
        if (
            torch.is_tensor(value)
            and value.device == original.device
            and value.dtype == original.dtype
            and value.shape == original.shape
            and value.stride() == original.stride()
            and _dense_zero_offset(value)
            and _logical_carry_reads(source)
            and not _returns_alias(
                result, (*outputs[:index], *outputs[index + 1 :], *emissions)
            )
        ):
            groups.append(group)
    return replace(layout, direct_groups=frozenset(groups)) if groups else layout


def _capture_tensor_step(body, arguments, *, grad_enabled, tracer=None):
    """Capture authored detach/STE boundaries before scan differentiates them."""
    tracer = make_fx if tracer is None else tracer
    # Capture storage is created under no-grad, but authored policy such as
    # Mechanism's auto delay mode depends on the execution caller's grad mode.
    # Trace that policy in its actual context, restoring the capture boundary
    # before returning the graph.
    # Disposable tracing saves must not enter an enclosing checkpoint's tensor
    # inventory: recomputation reuses this graph instead of tracing it again.
    with (
        torch.set_grad_enabled(grad_enabled),
        torch.autograd.graph.saved_tensors_hooks(
            _capture_saved_tensor_identity, _capture_saved_tensor_identity
        ),
    ):
        return tracer(body, pre_dispatch=grad_enabled)(*arguments)


def _packed_tensor_scan(
    graph,
    layout,
    parameters,
    prepared,
    initial,
    invariant_carries,
    samples,
    initial_failure=(),
):
    """Execute the shared HOP boundary on fresh dtype/device carry groups."""
    from torch._higher_order_ops.scan import scan

    parameters, prepared, invariant_carries, samples = _isolate_dense_lifted_inputs(
        (parameters, prepared, invariant_carries, samples)
    )

    def combine(carry, sample):
        packed = carry[0] if initial_failure else carry
        state, carries = layout.unpack(packed)
        updated, emitted = graph(
            parameters, prepared, state, carries, invariant_carries, sample
        )
        # Unproven updates get a fresh pack. A qualified private carry
        # result already has the next operand's shape and independent storage.
        packed = layout.pack(updated)
        if initial_failure:
            emitted, event = emitted
            return (packed, _advance_solver_failure(carry[1], event)), emitted
        return packed, emitted

    if initial_failure:
        (packed, failure), emitted = scan(combine, (initial, initial_failure), samples)
        return packed, emitted, failure
    return scan(combine, initial, samples)


def _replace_path(tree, path, value):
    result = dict(tree)
    cursor = result
    source = tree
    for name in path[:-1]:
        source = source[name]
        child = dict(source)
        cursor[name] = child
        cursor = child
    cursor[path[-1]] = value
    return result


def _tensor_samples(tapes, steps, device):
    """Pack Tensor leaves once, retaining static None positions for the body."""
    values, spec = pytree.tree_flatten(tapes)
    tensors, positions = [], []
    for value in values:
        if value is None:
            positions.append(None)
        else:
            positions.append(len(tensors))
            tensors.append(value)
    if any(
        not torch.is_tensor(value) or value.ndim < 1 or value.shape[0] != steps
        for value in tensors
    ):
        raise ValueError("scan input tapes require a matching leading time axis")
    if not tensors:
        tensors.append(torch.arange(steps, dtype=torch.int64, device=device))
    return tuple(tensors), (spec, tuple(positions))


@dataclass(frozen=True)
class _PrivateScanCarry:
    owner: object
    layout: _PackedCarryLayout
    program: tuple
    packed: tuple
    leaves: tuple


@dataclass
class _ScanHostCarry:
    # Invocation-local storage for the next private host chunk.

    chunk: _PrivateScanCarry | None = None


class _CompiledTensorTreeScan:
    """Lazy shared scan capture for an adapter's fixed-horizon tensor step.

    ``body(parameters, prepared, state, callback_carry, sample)`` returns
    ``((next_state, next_callback_carry), emissions)``. Sampling, validation,
    callback binding and zero-step handling belong to the public adapter.
    The input tape is a Tensor/None PyTree with one leading time axis per Tensor.

    Cached entries retain graphs and schema metadata, never example tensors or
    a bound parameter/preparation graph. Public calls own fresh materialized
    carry leaves outside compilation and share Population's first-order scan
    differentiation contract.

    Raw sample views retain separate authored captures. Eligible identical
    programs share a compiled executor, with each raw entry retaining its own
    view-isolation policy and ordinary Dynamo guards.
    """

    def __init__(
        self,
        body,
        *,
        steps,
        compile_options,
        device,
        plans=None,
        static_state_paths=(),
        detached_state_paths=(),
        clone_state=None,
        capture_examples=None,
        solver_failures=False,
        derivimplicit_forward="inline",
    ):
        install_scan_compatibility()
        if type(steps) is not int or steps < 1:
            raise ValueError("compiled tensor-tree scans require a positive step count")
        self._body = body
        self._steps = steps
        self._options = dict(compile_options)
        self._device = torch.device(device)
        self._callback_carry = _CallbackCarryPartition.from_plans(plans)
        self._static_state_paths = tuple(tuple(path) for path in static_state_paths)
        self._detached_state_paths = tuple(tuple(path) for path in detached_state_paths)
        self._clone_state = _clone_tree if clone_state is None else clone_state
        self._capture_examples = capture_examples
        self._solver_failures = solver_failures
        self._derivimplicit_forward = derivimplicit_forward
        self._kernels = {}
        self._executors = {}

    def _samples(self, tapes):
        return _tensor_samples(tapes, self._steps, self._device)

    def _carry_layout(self, state, mutable, *, grad_enabled):
        # Packing large leaves together can make Inductor traverse their entire
        # combined adjoint with per-element range masks on every reverse step.
        # Separate large leaves while keeping the small loop operands grouped.
        cutoff = None
        if (
            grad_enabled
            and self._device.type == "cpu"
            and _uses_local_scan_inductor(self._options)
        ):
            cutoff = _CPU_TRAINING_SCAN_LARGE_LEAF_BYTES
        return _PackedCarryLayout.from_tree((state, mutable), large_leaf_bytes=cutoff)

    def _pack(self, state, carries, *, grad_enabled):
        # Own recurrent storage before both the cache key and Dynamo's input
        # guards. Public continuations can change offsets, aliases, leafness or
        # inference status without changing the logical recurrence. Unlike the
        # inference while boundary, these live packs must retain training history.
        with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
            state = self._clone_state(state)
            mutable, invariant = self._callback_carry.split(carries)
            layout = self._carry_layout(state, mutable, grad_enabled=grad_enabled)
            initial = layout.pack((state, mutable))
        return layout, initial, invariant

    def _resume_pack(self, state, carries, carry, *, grad_enabled):
        if carry.owner is not self:
            return None
        layout = carry.layout
        mutable, invariant = self._callback_carry.split(carries)
        values, spec = pytree.tree_flatten((state, mutable))
        if spec != layout.spec:
            return None
        retained = {
            group: carry.packed[group]
            for group, indices in enumerate(layout.indices)
            if all(values[index] is carry.leaves[index] for index in indices)
        }
        if not retained:
            return None

        # Host static/detached fields and synthetic auxiliary inputs retain
        # their ordinary boundary semantics. Rebuild only their changed packs.
        state_count = spec.children()[0].num_leaves
        with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
            for index, leaf in enumerate(layout.leaves):
                value = values[index]
                if leaf is None:
                    if value is not None:
                        return None
                    continue
                if leaf.group in retained:
                    continue
                if (
                    not torch.is_tensor(value)
                    or value.shape != leaf.shape
                    or (value.dtype, value.device) != layout.groups[leaf.group]
                ):
                    return None
                if index < state_count or leaf.group in layout.direct_groups:
                    values[index] = value.clone(memory_format=torch.contiguous_format)
            return layout.pack_leaves(values, spec, retained=retained), invariant

    def _capture(
        self,
        parameters,
        prepared,
        state,
        carries,
        samples,
        sample_layout,
        *,
        grad_enabled,
        executor_signature=None,
    ):
        body = self._body
        partition = self._callback_carry
        clone_state = self._clone_state
        # Explicit invariant operands are read-only. Retain their visible
        # metadata during capture, including requires_grad and leaf status.
        # The no-grad capture boundary owns recurrent storage separately.
        state = clone_state(state)
        if self._capture_examples is not None:
            # Adapter policy applies only to disposable tracing examples.
            # Live runtime operands keep their gradients and view metadata.
            parameters, prepared, state, carries, samples = self._capture_examples(
                (parameters, prepared, state, carries, samples)
            )
        mutable, invariant = partition.split(carries)
        layout = self._carry_layout(state, mutable, grad_enabled=grad_enabled)
        # Use the exact logical views that the scan body receives. Authored
        # programs may depend on visible strides for ordinary tensor views.
        state, mutable = layout.unpack(layout.pack((state, mutable)))
        sample = pytree.tree_map(lambda value: value[0], samples)

        def authored(parameters, prepared, state, carries, invariant_carries, sample):
            spec, positions = sample_layout
            sample = pytree.tree_unflatten(
                [None if index is None else sample[index] for index in positions], spec
            )
            updated, emitted = body(
                parameters,
                prepared,
                state,
                partition.merge(carries, invariant_carries),
                sample,
            )
            next_state, next_carries = updated
            next_carries, _ = partition.split(next_carries)
            return (next_state, next_carries), _clone_tree(emitted or {})

        failures = _SolverFailureCapture() if self._solver_failures else None
        forward_capture = nullcontext(())
        if (
            grad_enabled
            and self._derivimplicit_forward == "separate"
            and not self._options.get("disable", False)
        ):
            from ..models.mechanisms._derivimplicit_forward import (
                _capture_forward_programs,
            )

            forward_capture = _capture_forward_programs(self._options.get("backend"))
        with forward_capture as programs:
            graph = _capture_tensor_step(
                failures.wrap(authored) if failures is not None else authored,
                (parameters, prepared, state, mutable, invariant, sample),
                grad_enabled=grad_enabled,
            )
        forward_programs = tuple(programs)
        parameter_use, prepared_use, _, _, invariant_use, sample_use = _input_usage(
            graph, (parameters, prepared, state, mutable, invariant, sample)
        )
        if (
            grad_enabled
            and self._device.type == "cpu"
            and _uses_local_scan_inductor(self._options)
        ):
            layout = _direct_carry_layout(graph, layout, parameters, prepared)
            if executor_signature is not None:
                executor_signature = (layout, *executor_signature[1:])
        isolate_views = _LiftedViewIsolation.from_usage(
            (
                parameter_use,
                prepared_use,
                invariant_use,
                pytree.tree_map(lambda _value: True, sample_use),
            ),
            captured_inputs=isinstance(graph, torch.fx.GraphModule),
        )

        check_failures = failures is not None and failures.active
        device = self._device

        def execute(parameters, prepared, initial, invariant, samples):
            return _packed_tensor_scan(
                graph,
                layout,
                parameters,
                prepared,
                initial,
                invariant,
                samples,
                failures.initial(device) if check_failures else (),
            )

        # A pooled executor could refer to another capture's opaque program.
        executor_key = (
            None
            if forward_programs
            else _executor_key(graph, executor_signature, self._options)
        )
        existing = (
            self._executors.get(executor_key) if executor_key is not None else None
        )
        if existing is not None:
            compiled, retained_graph = existing
            if (
                _executor_key(retained_graph, executor_signature, self._options)
                != executor_key
            ):
                existing = None
                del self._executors[executor_key]
        if existing is None:
            compiled = _compile_scan(
                execute,
                self._options,
                prefer_cpp_wrapper=not grad_enabled and self._device.type == "cpu",
            )
            if executor_key is not None:
                self._executors[executor_key] = compiled, graph
        if check_failures:

            def execute_checked(parameters, prepared, initial, invariant, samples):
                packed, emitted, failure = compiled(
                    parameters, prepared, initial, invariant, samples
                )
                failures.check(failure)
                return packed, emitted

            return execute_checked, layout, isolate_views, forward_programs
        return compiled, layout, isolate_views, forward_programs

    @staticmethod
    def _resumed_signature(layout, schema, program):
        if not layout.direct_groups:
            return layout, schema, *program
        spec, metadata = schema
        offset = sum(child.num_leaves for child in spec.children()[:2])
        rows = list(metadata)
        for group in layout.direct_groups:
            leaf = layout.leaves[layout.indices[group][0]]
            # Use the ordinary owned pack key without materializing its
            # padded storage. The executor still receives the shaped tensor.
            row = rows[offset + group]
            rows[offset + group] = (
                (leaf.numel + 1,),
                row[1],
                row[2],
                (1,),
                0,
                row[5],
                not row[5],
                *row[7:],
            )
        return (
            replace(layout, direct_groups=frozenset()),
            (spec, tuple(rows)),
            *program,
        )

    def __call__(
        self,
        parameters,
        prepared,
        state,
        carries,
        input_tape,
        *,
        host_carry=None,
        retain_carry=False,
    ):
        input_tape = _own_runtime_draw_packets(input_tape)
        samples, sample_layout = self._samples(input_tape)
        grad_enabled = torch.is_grad_enabled()
        program = (sample_layout, MEMBRANE_ASSEMBLY.value, grad_enabled)
        entry = None
        if host_carry is not None:
            previous, host_carry.chunk = host_carry.chunk, None
            if previous is not None and previous.program == program:
                resumed = self._resume_pack(
                    state, carries, previous, grad_enabled=grad_enabled
                )
                if resumed is not None:
                    initial, invariant = resumed
                    signature = self._resumed_signature(
                        previous.layout,
                        _schema((parameters, prepared, initial, invariant, samples)),
                        program,
                    )
                    cached = self._kernels.get(signature)
                    if cached is not None and cached[1] == previous.layout:
                        entry = cached
        if entry is None:
            layout, initial, invariant = self._pack(
                state, carries, grad_enabled=grad_enabled
            )
            signature = (
                layout,
                _schema((parameters, prepared, initial, invariant, samples)),
                *program,
            )
            if signature not in self._kernels:
                pool = _plain_projection_context(
                    check_saved_tensors=False
                ) and _plain_state_tree(
                    (parameters, prepared, state, carries, samples),
                    allow_parameters=True,
                )
                with torch.inference_mode(False), torch.no_grad():
                    self._kernels[signature] = self._capture(
                        parameters,
                        prepared,
                        state,
                        carries,
                        samples,
                        sample_layout,
                        grad_enabled=grad_enabled,
                        executor_signature=signature if pool else None,
                    )
            entry = self._kernels[signature]
            initial = entry[1].scan_inputs(initial)
        compiled, layout, isolate_views, forward_programs = entry
        parameters, prepared, invariant, samples = isolate_views(
            (parameters, prepared, invariant, samples)
        )
        packed, emitted = compiled(parameters, prepared, initial, invariant, samples)
        if retain_carry:
            # Wrap the private packs themselves to preserve the first-order
            # boundary without cloning every logical leaf on this host hop.
            packed, emitted = guard_scan_outputs(
                (packed, emitted), forward_programs=forward_programs
            )

        final, updated = layout.unpack(packed)
        if retain_carry:
            host_carry.chunk = _PrivateScanCarry(
                self,
                layout,
                program,
                packed,
                tuple(pytree.tree_leaves((final, updated))),
            )
        else:
            # Public leaves may not expose the internal packed storage or share
            # aliases with their siblings.
            final = self._clone_state(final)
        updated = self._callback_carry.restore_invariants(updated, carries)
        if not retain_carry:
            updated = _clone_tree(updated)
        for path in self._static_state_paths:
            final = _replace_path(final, path, _get_path(state, path).clone())
        for path in self._detached_state_paths:
            final = _replace_path(final, path, _get_path(final, path).detach())
        result = final, updated, emitted
        return (
            result
            if retain_carry
            else guard_scan_outputs(result, forward_programs=forward_programs)
        )


__all__ = []
