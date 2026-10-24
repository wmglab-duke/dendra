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
    leaves: tuple[_PackedLeaf | None, ...]
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
            if value is None:
                leaves.append(None)
                continue
            if not torch.is_tensor(value) or value.layout is not torch.strided:
                raise TypeError(
                    "packed loop carry requires strided Tensor or None leaves"
                )
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
        if any(
            (leaf is None) != (value is None)
            for leaf, value in zip(self.leaves, values, strict=True)
        ):
            raise ValueError("packed loop carry changed a static None leaf")
        packs = []
        for indices in self.indices:
            pieces = [values[index].reshape(-1) for index in indices]
            if len(indices) == 1 and self.leaves[indices[0]].numel:
                # Inductor can elide a one-input cat and return a tape slice
                # with a nonzero offset as the next HOP carry. The loop's
                # aligned output contract then fails (notably for bool bytes).
                # A private padding element forces an owned full-span pack;
                # unpack exposes only the authored logical leaf.
                pieces.append(pieces[0].new_zeros((1,)))
            packs.append(torch.cat(pieces, dim=0))
        return tuple(packs)

    def unpack(self, packs):
        if len(packs) != len(self.groups):
            raise ValueError("packed loop carry changed its dtype/device groups")
        values = [
            None
            if leaf is None
            else packs[leaf.group].narrow(0, leaf.offset, leaf.numel).view(leaf.shape)
            for leaf in self.leaves
        ]
        return pytree.tree_unflatten(values, self.spec)


def _clone_tree_like(values, templates):
    """Clone values with each invocation's original carry-format semantics.

    ``empty_like`` preserves a dense template's strides and, like ``clone``,
    makes non-dense views contiguous. The copy is outside compilation so each
    leaf has genuinely independent storage, even for duplicate logical values.
    It depends only on ``values`` under autograd; templates supply metadata.
    Non-Tensor leaves pass through unchanged.
    """
    return pytree.tree_map(
        lambda value, template: (
            torch.empty_like(template).copy_(value) if torch.is_tensor(value) else value
        ),
        values,
        templates,
    )


__all__ = []
