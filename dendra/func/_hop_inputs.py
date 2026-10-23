"""Independent lifted loop operands with authored tensor-view layouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils import _pytree as pytree


def _dense_zero_offset(value):
    return (
        value.is_contiguous()
        and value.storage_offset() == 0
        and not (value.is_conj() or value.is_neg())
    )


def _copy_view(value):
    # Copy the physical span before restoring logical views. In particular,
    # clone(preserve_format) still compacts slices and expanded tensors.
    # Captured programs can use their original strides in explicit view ops.
    physical = value
    conjugate, negative = value.is_conj(), value.is_neg()
    if _dense_zero_offset(value):
        return value.clone(memory_format=torch.preserve_format)
    if conjugate:
        physical = physical.conj()
    if negative:
        physical = torch._neg_view(physical)
    if value.numel():
        size = value.storage_offset() + 1
        for length, stride in zip(value.shape, value.stride(), strict=True):
            size += (length - 1) * stride
        storage = physical.as_strided((size,), (1,), 0).clone()
    else:
        storage = physical.new_empty((0,))
    result = storage.as_strided(value.shape, value.stride(), value.storage_offset())
    if negative:
        result = torch._neg_view(result)
    if conjugate:
        result = result.conj()
    return result


class _IndependentView(torch.autograd.Function):
    """Storage isolation is the identity in logical tensor coordinates."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value):
        return _copy_view(value)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return gradient

    @staticmethod
    def jvp(ctx, tangent):
        return _copy_view(tangent)


@torch.compiler.allow_in_graph
def _isolate_lifted_tensor(value):
    # AOTAutograd keeps the logical identity derivative of overlapping views.
    # Dynamo cannot inline an autograd.Function with a custom JVP; the audited
    # operation also reads only static tensor-view metadata.
    return _IndependentView.apply(value)


def _isolate_lifted_inputs(tree):
    return pytree.tree_map_only(torch.Tensor, _isolate_lifted_tensor, tree)


def _input_usage(graph, arguments):
    """Retain only static placeholder-use metadata from an authored step."""
    values, spec = pytree.tree_flatten(arguments)
    if isinstance(graph, torch.fx.GraphModule):
        placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
        if len(placeholders) != len(values):
            raise ValueError("loop input usage changed its captured argument structure")
        used = [bool(node.users) for node in placeholders]
    else:
        # Non-FX capture adapters conservatively retain tensor arguments.
        used = [torch.is_tensor(value) for value in values]
    return pytree.tree_unflatten(used, spec)


@dataclass(frozen=True)
class _LiftedViewIsolation:
    spec: object
    indices: tuple[int, ...]

    @classmethod
    def from_usage(cls, usage):
        values, spec = pytree.tree_flatten(usage)
        return cls(spec, tuple(index for index, used in enumerate(values) if used))

    def __call__(self, tree):
        # Authored views enter the graph as owned Tensor inputs. Creating them
        # inside it can leave identity ReinterpretViews lifted into a HOP; the
        # C++ wrapper can reuse a freed temporary across condition/body calls.
        # Ignore unused inputs and keep dense copies in the compiled graph.
        values, spec = pytree.tree_flatten(tree)
        if spec != self.spec:
            # FX binds dictionary arguments by key. Reuse that captured order
            # when an equivalent invocation mapping has a different insertion
            # order, so the usage indices still identify the intended leaves.
            try:
                values = self.spec.flatten_up_to(tree)
                canonical = pytree.tree_unflatten(values, self.spec)
                if pytree.tree_structure(canonical) != self.spec:
                    raise ValueError("loop input leaf became a subtree")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "loop view isolation changed its argument structure"
                ) from error
            spec = self.spec
        copied = False
        for index in self.indices:
            value = values[index]
            if torch.is_tensor(value) and not _dense_zero_offset(value):
                values[index] = _isolate_lifted_tensor(value)
                copied = True
        return pytree.tree_unflatten(values, spec) if copied else tree


@torch.compiler.allow_in_graph
def _isolate_dense_lifted_tensor(value):
    # AOT traces static view metadata; Dynamo cannot inline storage_offset().
    return (
        value.clone(memory_format=torch.preserve_format)
        if _dense_zero_offset(value)
        else value
    )


def _isolate_dense_lifted_inputs(tree):
    # Dense copies do not need view handles. Keep these inside compilation so
    # unused copies disappear and numerical kernels can fuse with their users.
    # Non-dense operands already own storage at the invocation boundary.
    return pytree.tree_map_only(
        torch.Tensor,
        _isolate_dense_lifted_tensor,
        tree,
    )


__all__ = []
