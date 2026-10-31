"""Alias-preserving detachment of native recurrent Tensor trees."""

import torch
from torch.utils import _pytree as pytree


def _detach_runtime_tree(value, memo):
    def detach(leaf):
        if isinstance(leaf, torch.Tensor):
            identity = id(leaf)
            if identity not in memo:
                memo[identity] = leaf.detach()
            return memo[identity]
        return leaf

    return pytree.tree_map(
        detach, value, is_leaf=lambda leaf: isinstance(leaf, torch.Tensor)
    )


def _promote_inference_runtime_tree(value, memo):
    """Own inference-produced carry without changing ordinary Tensor graphs."""

    def promote(leaf):
        if isinstance(leaf, torch.Tensor) and torch.is_inference(leaf):
            identity = id(leaf)
            if identity not in memo:
                with torch.inference_mode(False), torch.no_grad():
                    memo[identity] = leaf.clone()
            return memo[identity]
        return leaf

    return pytree.tree_map(
        promote, value, is_leaf=lambda leaf: isinstance(leaf, torch.Tensor)
    )


__all__ = []
