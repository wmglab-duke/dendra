"""Owned tensor-tree packing for inference and training loop boundaries.

Only static schema metadata is retained. Leaves share an owned flat pack by
dtype and device unless they require a dedicated group. Scan can carry a
qualified dedicated group as a shaped tensor when the transition returns fresh
storage. Public leaves are materialized outside the compiled boundary.
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
    direct_groups: frozenset[int] = frozenset()

    @classmethod
    def from_tree(cls, tree, *, large_leaf_bytes=None):
        values, spec = pytree.tree_flatten(tree)
        return cls.from_leaves(values, spec, large_leaf_bytes=large_leaf_bytes)

    @classmethod
    def from_leaves(cls, values, spec, *, large_leaf_bytes=None):
        groups = []
        shared_groups = {}
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
            numel = value.numel()
            dedicated = (
                large_leaf_bytes is not None
                and numel * value.element_size() >= large_leaf_bytes
            )
            group = None if dedicated else shared_groups.get(key)
            if group is None:
                group = len(groups)
                groups.append(key)
                group_indices.append([])
                sizes.append(0)
                if not dedicated:
                    shared_groups[key] = group
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
        return self.pack_leaves(values, spec)

    def pack_leaves(self, values, spec, *, retained=None):
        if spec != self.spec:
            raise ValueError("packed loop carry changed its tensor-tree structure")
        if any(
            (leaf is None) != (value is None)
            for leaf, value in zip(self.leaves, values, strict=True)
        ):
            raise ValueError("packed loop carry changed a static None leaf")
        packs = []
        for group, indices in enumerate(self.indices):
            if retained is not None and group in retained:
                packs.append(retained[group])
                continue
            if group in self.direct_groups:
                packs.append(values[indices[0]])
                continue
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

    def scan_inputs(self, packed):
        # Each dedicated pack already owns its storage. Convert the private
        # prefix to the logical shape once at the chunk boundary.
        if not self.direct_groups:
            return packed
        values = list(packed)
        with torch.inference_mode(False):
            for group in self.direct_groups:
                leaf = self.leaves[self.indices[group][0]]
                values[group] = packed[group].narrow(0, 0, leaf.numel).view(leaf.shape)
        return tuple(values)

    def unpack(self, packs):
        if len(packs) != len(self.groups):
            raise ValueError("packed loop carry changed its dtype/device groups")
        values = [
            None
            if leaf is None
            else (
                packs[leaf.group]
                if leaf.group in self.direct_groups
                else packs[leaf.group]
                .narrow(0, leaf.offset, leaf.numel)
                .view(leaf.shape)
            )
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
