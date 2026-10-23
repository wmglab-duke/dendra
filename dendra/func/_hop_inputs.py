"""Independent lifted loop operands with authored tensor-view layouts."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree


def _copy_view(value):
    # Copy the physical span before restoring logical views. In particular,
    # clone(preserve_format) still compacts slices and expanded tensors.
    # Captured programs can use their original strides in explicit view ops.
    physical = value
    conjugate, negative = value.is_conj(), value.is_neg()
    if (
        value.is_contiguous()
        and value.storage_offset() == 0
        and not (conjugate or negative)
    ):
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


__all__ = []
