"""Shared owned tensor-tree boundaries for fixed-horizon compiled scans.

Population and Network adapters provide only their authored tensor step. The
capture, packed carry and scan lowering here preserve one set of AD and
ownership rules across those adapters.
"""

from __future__ import annotations

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from ..helpers import MEMBRANE_ASSEMBLY
from ._callback_carry import _CallbackCarryPartition
from ._carry_packing import _PackedCarryLayout
from ._hop_inputs import (
    _input_usage,
    _isolate_dense_lifted_inputs,
    _LiftedViewIsolation,
)
from ._scan_autograd import guard_scan_outputs
from ._scan_compat import (
    _capture_saved_tensor_identity,
    install_scan_compatibility,
)
from ._scan_inductor import _compile_scan
from ._tensor_contracts import _clone_contiguous_tree as _clone_tree
from ._tensor_contracts import capture_schema as _schema
from ._tree_paths import _get_path


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
    graph, layout, parameters, prepared, initial, invariant_carries, samples
):
    """Execute the shared HOP boundary on fresh dtype/device carry groups."""
    from torch._higher_order_ops.scan import scan

    parameters, prepared, invariant_carries, samples = _isolate_dense_lifted_inputs(
        (parameters, prepared, invariant_carries, samples)
    )

    def combine(packed, sample):
        state, carries = layout.unpack(packed)
        updated, emitted = graph(
            parameters, prepared, state, carries, invariant_carries, sample
        )
        # The terminal pack owns storage separately from AD's saved state
        # intermediates. No private carry views become public sibling aliases.
        return layout.pack(updated), emitted

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
        self._kernels = {}

    def _samples(self, tapes):
        return _tensor_samples(tapes, self._steps, self._device)

    def _pack(self, state, carries, *, grad_enabled):
        # Own recurrent storage before both the cache key and Dynamo's input
        # guards. Public continuations can change offsets, aliases, leafness or
        # inference status without changing the logical recurrence. Unlike the
        # inference while boundary, these live packs must retain training history.
        with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
            state = self._clone_state(state)
            mutable, invariant = self._callback_carry.split(carries)
            layout = _PackedCarryLayout.from_tree((state, mutable))
            initial = layout.pack((state, mutable))
        return layout, initial, invariant

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
        layout = _PackedCarryLayout.from_tree((state, mutable))
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

        graph = _capture_tensor_step(
            authored,
            (parameters, prepared, state, mutable, invariant, sample),
            grad_enabled=grad_enabled,
        )
        parameter_use, prepared_use, _, _, invariant_use, sample_use = _input_usage(
            graph, (parameters, prepared, state, mutable, invariant, sample)
        )
        isolate_views = _LiftedViewIsolation.from_usage(
            (
                parameter_use,
                prepared_use,
                invariant_use,
                pytree.tree_map(lambda _value: True, sample_use),
            ),
            captured_inputs=isinstance(graph, torch.fx.GraphModule),
        )

        def execute(parameters, prepared, initial, invariant, samples):
            return _packed_tensor_scan(
                graph,
                layout,
                parameters,
                prepared,
                initial,
                invariant,
                samples,
            )

        compiled = _compile_scan(
            execute,
            self._options,
            prefer_cpp_wrapper=not grad_enabled and self._device.type == "cpu",
        )
        return compiled, layout, isolate_views

    def __call__(self, parameters, prepared, state, carries, input_tape):
        samples, sample_layout = self._samples(input_tape)
        grad_enabled = torch.is_grad_enabled()
        layout, initial, invariant = self._pack(
            state, carries, grad_enabled=grad_enabled
        )
        signature = (
            layout,
            _schema((parameters, prepared, initial, invariant, samples)),
            sample_layout,
            MEMBRANE_ASSEMBLY.value,
            grad_enabled,
        )
        if signature not in self._kernels:
            with torch.inference_mode(False), torch.no_grad():
                self._kernels[signature] = self._capture(
                    parameters,
                    prepared,
                    state,
                    carries,
                    samples,
                    sample_layout,
                    grad_enabled=grad_enabled,
                )
        compiled, layout, isolate_views = self._kernels[signature]
        parameters, prepared, invariant, samples = isolate_views(
            (parameters, prepared, invariant, samples)
        )
        packed, emitted = compiled(
            parameters,
            prepared,
            initial,
            invariant,
            samples,
        )
        final, updated = layout.unpack(packed)
        # Equivalent logical leaves must never escape as duplicate moved C++
        # HOP output handles. Keep these ownership clones outside compilation.
        final = self._clone_state(final)
        updated = _clone_tree(self._callback_carry.restore_invariants(updated, carries))
        for path in self._static_state_paths:
            final = _replace_path(final, path, _get_path(state, path).clone())
        for path in self._detached_state_paths:
            final = _replace_path(final, path, _get_path(final, path).detach())
        return guard_scan_outputs((final, updated, emitted))


__all__ = []
