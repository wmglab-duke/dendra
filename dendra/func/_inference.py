"""Invocation-local representation of dynamic compiled inference operands."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils import _pytree as pytree


def _canonicalize_inference_tree(tree):
    """Match restart operands to recurrent inference outputs without caching.

    Only dynamic state and callback carry belong here. Parameters, constants,
    preparation and authored inputs retain their original versioned bindings.
    An ordinary initial state otherwise compiles separately from the inference
    state returned by its first chunk. Copies are local to this invocation and
    preserve repeated Tensor references, including aliases across state/carry.
    Distinct views are independent pure values, as with the existing state and
    callback materializers; their shared-storage relationship is not retained.
    No caller-owned tensor or tree is changed.
    """
    if (
        torch.compiler.is_compiling()
        or not torch.is_inference_mode_enabled()
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    ):
        return tree

    leaves, spec = pytree.tree_flatten(tree)

    def has_ordinary_tensor(value):
        if isinstance(value, torch.Tensor):
            return not torch.is_inference(value)
        if isinstance(value, Mapping):
            return any(
                has_ordinary_tensor(leaf)
                for child in value.values()
                for leaf in pytree.tree_leaves(child)
            )
        return False

    # Recurring chunks already have the desired representation. Avoid copies
    # and tree reconstruction in that hot path.
    if not any(has_ordinary_tensor(value) for value in leaves):
        return tree

    copied = {}

    def convert(value):
        if isinstance(value, torch.Tensor):
            if torch.is_inference(value):
                return value
            identity = id(value)
            if identity not in copied:
                copied[identity] = value.clone()
            return copied[identity]
        if isinstance(value, Mapping):
            # Public state accepts any Mapping. An unregistered Mapping is a
            # pytree leaf, but still contains dynamic tensor operands.
            return {name: map_tree(child) for name, child in value.items()}
        return value

    def map_tree(value):
        return pytree.tree_map(convert, value)

    return pytree.tree_unflatten([convert(value) for value in leaves], spec)
