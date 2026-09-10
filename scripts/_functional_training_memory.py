"""Untimed retained-storage diagnostics for functional training benchmarks.

This is a tracked tensor-storage proxy, not total process memory or peak
activation memory. Autograd hooks alone miss nested checkpoint input trees and
Python-held closures. Callers therefore also register explicit inputs and the
prepared tensor values, and observe checkpoint arguments during the forward.

Run backward after leaving both observation contexts. All observations retain
detached tensor references until this object is discarded, so storage addresses
cannot be recycled while computing the storage union. The diagnostic must never
surround timed samples or run concurrently with another benchmark.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps

import torch
from torch.utils._pytree import tree_leaves

_StorageKey = tuple[str, int, int]


@dataclass
class _SourceStats:
    count: int = 0
    logical_bytes: int = 0
    storages: dict[_StorageKey, int] = field(default_factory=dict)


class SavedTensorStats:
    """Count observed saves/references and deduplicate their retained storage.

    ``count`` and ``logical_bytes`` count tensor occurrences, including repeated
    saves and aliases. ``unique_storage_bytes`` counts each nonempty backing
    storage once within a source, including the whole allocation behind a view.
    ``tracked_storage_bytes`` counts the union across every source, so callers
    must not add the per-source storage totals together.
    """

    def __init__(self) -> None:
        self._sources: dict[str, _SourceStats] = {
            label: _SourceStats()
            for label in ("autograd_saved", "checkpoint_inputs", "explicit_inputs")
        }
        self._retained: dict[_StorageKey, torch.Tensor] = {}
        self._storage_bytes: dict[_StorageKey, int] = {}
        self._observing_checkpoints = False

    def _retain(self, label: str, tensor: torch.Tensor) -> torch.Tensor:
        detached = tensor.detach()
        source = self._sources.setdefault(label, _SourceStats())
        source.count += 1
        source.logical_bytes += detached.numel() * detached.element_size()
        storage = detached.untyped_storage()
        size = storage.nbytes()
        if size:
            key = (str(detached.device), storage.data_ptr(), size)
            source.storages[key] = size
            self._storage_bytes[key] = size
            self._retained.setdefault(key, detached)
        return detached

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        """Autograd pack hook; detach to avoid retaining the original graph."""
        return self._retain("autograd_saved", tensor)

    @staticmethod
    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        """Return the detached saved value with unchanged content and metadata."""
        return tensor

    def retain_tree(self, label: str, value: object) -> None:
        """Register tensor leaves of a nested PyTree, ignoring other values."""
        for leaf in tree_leaves(value):
            if isinstance(leaf, torch.Tensor):
                self._retain(label, leaf)

    @contextmanager
    def observe_checkpoints(self) -> Iterator[None]:
        """Observe nested runner checkpoint arguments in an untimed forward.

        The runner imports ``checkpoint`` into its own module, so wrapping that
        exact binding also sees dictionary state inputs which PyTorch retains
        directly in Python. The original binding is restored on every exit.
        """
        from dendra.func import _runners

        if self._observing_checkpoints:
            raise RuntimeError(
                "checkpoint observation cannot be nested on one stats object"
            )
        original = _runners.checkpoint

        @wraps(original)
        def observed(function, *args, **kwargs):
            self.retain_tree("checkpoint_inputs", args)
            # Public runners currently pass only checkpoint controls by keyword.
            # Include tensor-valued keyword arguments for future runner variants.
            self.retain_tree("checkpoint_inputs", kwargs)
            return original(function, *args, **kwargs)

        self._observing_checkpoints = True
        _runners.checkpoint = observed
        try:
            yield
        finally:
            _runners.checkpoint = original
            self._observing_checkpoints = False

    def summary(self) -> dict[str, object]:
        """Return JSON-ready counts and the deduplicated observed storage union."""
        return {
            "metric": "tracked retained tensor-storage proxy",
            "sources": {
                label: {
                    "count": source.count,
                    "logical_bytes": source.logical_bytes,
                    "unique_storage_bytes": sum(source.storages.values()),
                }
                for label, source in self._sources.items()
            },
            "tracked_storage_bytes": sum(self._storage_bytes.values()),
            "limitations": (
                "Untimed forward observations retain detached storage references. "
                "Includes registered explicit/prepared inputs, autograd saves, and "
                "nested checkpoint inputs; repeated saves and aliased views are "
                "deduplicated only in storage totals. Excludes unregistered "
                "Python-held tensors, graph objects, allocator overhead, compiler "
                "memory, and transient backward/recomputation workspaces. This is "
                "not total process RSS or peak activation memory; retaining "
                "observations can include saves no longer live at forward end."
            ),
        }
