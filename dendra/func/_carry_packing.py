"""Owned tensor-tree packing for inference and training loop boundaries.

Only static schema metadata is retained. Each pack is a fresh one-dimensional
Tensor for one dtype/device group. Unpacked leaves are private read-only views;
public leaves are materialized outside the compiled boundary.
Terminal training packs also separate returned carry from AD's saved
intermediate tensors, without changing values or detaching their gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils import _pytree as pytree


@dataclass(frozen=True)
class _PackedLeaf:
    group: int
    offset: int
    numel: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _PackedCarryLayout:
    spec: object
    leaves: tuple[_PackedLeaf, ...]
    groups: tuple[tuple[torch.dtype, torch.device], ...]
    indices: tuple[tuple[int, ...], ...]

    @classmethod
    def from_tree(cls, tree):
        values, spec = pytree.tree_flatten(tree)
        groups = []
        group_indices = []
        sizes = []
        leaves = []
        for index, value in enumerate(values):
            if not torch.is_tensor(value) or value.layout is not torch.strided:
                raise TypeError("packed loop carry requires strided Tensor leaves")
            key = (value.dtype, value.device)
            if key not in groups:
                groups.append(key)
                group_indices.append([])
                sizes.append(0)
            group = groups.index(key)
            numel = value.numel()
            leaves.append(_PackedLeaf(group, sizes[group], numel, tuple(value.shape)))
            group_indices[group].append(index)
            sizes[group] += numel
        return cls(
            spec,
            tuple(leaves),
            tuple(groups),
            tuple(tuple(indices) for indices in group_indices),
        )

    def pack(self, tree):
        values, spec = pytree.tree_flatten(tree)
        if spec != self.spec:
            raise ValueError("packed loop carry changed its tensor-tree structure")
        # cat owns storage even for a one-leaf group. Never use a no-copy
        # one-leaf shortcut: unchanged leaves must not alias incoming carry.
        return tuple(
            torch.cat([values[index].reshape(-1) for index in indices], dim=0)
            for indices in self.indices
        )

    def unpack(self, packs):
        if len(packs) != len(self.groups):
            raise ValueError("packed loop carry changed its dtype/device groups")
        values = [
            packs[leaf.group].narrow(0, leaf.offset, leaf.numel).view(leaf.shape)
            for leaf in self.leaves
        ]
        return pytree.tree_unflatten(values, self.spec)


def _clone_tree_like(values, templates):
    """Clone values with each invocation's original carry-format semantics.

    ``empty_like`` preserves a dense template's strides and, like ``clone``,
    makes non-dense views contiguous. The copy is outside compilation so each
    leaf has genuinely independent storage, even for duplicate logical values.
    It depends only on ``values`` under autograd; templates supply metadata.
    """
    return pytree.tree_map(
        lambda value, template: torch.empty_like(template).copy_(value),
        values,
        templates,
    )


__all__ = []
