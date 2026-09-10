"""First-order boundary for the experimental compiled scan executor."""

from typing import Any

import torch
from torch.utils import _pytree as pytree

from ._types import FunctionalizationError


class _FirstOrderScanBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *values):
        ctx.set_materialize_grads(False)
        return values

    @staticmethod
    def backward(ctx, *grad_values):
        # Reject recording the first backward, even when its cotangents are
        # constant. once_differentiable can miss this after nonlinear preparation
        # outside scan and permit an incomplete higher derivative to escape.
        if torch.is_grad_enabled():
            raise FunctionalizationError(
                "execution='scan' supports first-order reverse-mode gradients only; "
                "use fmodel.rollout for create_graph=True or higher derivatives"
            )
        return grad_values


def guard_scan_outputs(tree: Any) -> Any:
    """Wrap differentiable result leaves after the compiled callable returns.

    Keep this boundary outside torch.compile: tracing its backward would resolve
    grad mode during compilation instead of checking the caller's AD request.
    Transform-aware eager fallbacks must bypass both scan and this boundary.
    """
    if not torch.is_grad_enabled():
        return tree
    leaves, spec = pytree.tree_flatten(tree)
    indices = [
        index
        for index, leaf in enumerate(leaves)
        if isinstance(leaf, torch.Tensor)
        and leaf.requires_grad
        and (leaf.is_floating_point() or leaf.is_complex())
    ]
    if not indices:
        return tree
    if torch.compiler.is_compiling():
        raise FunctionalizationError(
            "execution='scan' already owns its compiled boundary and cannot be "
            "wrapped in another torch.compile while gradients are enabled"
        )
    guarded = _FirstOrderScanBoundary.apply(*(leaves[index] for index in indices))
    for index, value in zip(indices, guarded, strict=True):
        leaves[index] = value
    return pytree.tree_unflatten(leaves, spec)
