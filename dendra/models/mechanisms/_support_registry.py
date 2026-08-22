"""Runtime interning for exact mechanism-support layouts.

The registry gives one handler-wide integer identity to mechanisms that share
the same ordered physical support *and* the same local tensor layout.  It is
derived execution metadata: mechanism ``key`` buffers remain authoritative and
unchanged, while the registry is rebuilt after copying, serialization, state
loading, or device conversion.

Support IDs are authored mechanism ordinals, rather than dense group numbers.
If a previously shared group later splits (for example because packed keys are
unavailable after a meta/to-empty transition), unrelated later support IDs
therefore remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import torch

from ._support import SupportMap, SupportSpec


def _support_map(mechanism) -> SupportMap | None:
    support_map = getattr(mechanism, "support_map", None)
    return support_map if isinstance(support_map, SupportMap) else None


def _layout_signature(mechanism) -> tuple[Any, ...]:
    """Return the mechanism-visible part of exact support identity."""

    try:
        base_ndim = int(mechanism.base_ndim)
        shape_f = tuple(int(size) for size in mechanism.shape_f)
        is_composable = bool(mechanism.is_composable)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "SupportRegistry entries must expose base_ndim, shape_f, and "
            "is_composable mechanism layout metadata."
        ) from error

    support_map = _support_map(mechanism)
    if support_map is not None:
        structural = ("support_spec", support_map.spec.ordered_signature)
    else:
        key = getattr(mechanism, "key", None)
        if key is None:
            structural = ("legacy_dense",)
        elif is_composable:
            # Exact slice values are checked against candidates below.  Keeping
            # them out of the hash key also accommodates legacy selector parts
            # that are not themselves hashable.
            structural = ("legacy_composable", len(key))
        elif torch.is_tensor(key):
            structural = (
                "legacy_packed",
                tuple(key.shape),
                key.dtype,
                key.device.type,
                key.device.index,
            )
        else:
            structural = ("legacy_packed_other", type(key))

    return (base_ndim, shape_f, is_composable, structural)


def _same_legacy_selector(left, right) -> bool:
    """Mirror legacy handler equality when structural metadata is unavailable."""

    left_key = getattr(left, "key", None)
    right_key = getattr(right, "key", None)
    if left_key is None or right_key is None:
        return left_key is None and right_key is None

    if bool(left.is_composable):
        if len(left_key) != len(right_key):
            return False
        for left_part, right_part in zip(left_key, right_key):
            if isinstance(left_part, slice) and isinstance(right_part, slice):
                if (
                    left_part.start,
                    left_part.stop,
                    left_part.step,
                ) != (
                    right_part.start,
                    right_part.stop,
                    right_part.step,
                ):
                    return False
            else:
                try:
                    equal = left_part == right_part
                    if torch.is_tensor(equal):
                        equal = bool(torch.all(equal))
                except (RuntimeError, TypeError, ValueError):
                    return False
                if not equal:
                    return False
        return True

    if not torch.is_tensor(left_key) or not torch.is_tensor(right_key):
        return left_key == right_key
    if left_key.device.type == "meta" or right_key.device.type == "meta":
        return left is right
    return (
        left_key.shape == right_key.shape
        and left_key.dtype == right_key.dtype
        and left_key.device == right_key.device
        and torch.equal(left_key, right_key)
    )


def _same_support_in_bucket(mechanism, representative) -> bool:
    """Confirm exact ordered support after layout-signature bucketing."""

    support_map = _support_map(mechanism)
    representative_map = _support_map(representative)
    if support_map is None or representative_map is None:
        return support_map is representative_map and _same_legacy_selector(
            mechanism, representative
        )

    if not support_map.spec.has_compact_identity:
        if not getattr(mechanism, "_support_key_values_valid", True) or not getattr(
            representative, "_support_key_values_valid", True
        ):
            return mechanism is representative

    return support_map.same_ordered_support(
        representative_map,
        getattr(mechanism, "key", None),
        getattr(representative, "key", None),
    )


def _legacy_index_count(mechanism) -> int:
    support_map = _support_map(mechanism)
    if support_map is not None:
        return support_map.spec.legacy_index_count
    key = getattr(mechanism, "key", None)
    if torch.is_tensor(key) and not bool(getattr(mechanism, "is_composable", False)):
        return int(key.numel())
    return 0


def _compact_index_count(mechanism) -> int:
    support_map = _support_map(mechanism)
    if support_map is not None:
        return support_map.spec.compact_index_count
    return _legacy_index_count(mechanism)


@dataclass(frozen=True, slots=True)
class SupportEntry:
    """One canonical runtime support and its live representative mechanism."""

    support_id: int
    support_map: SupportMap | None
    layout_signature: tuple[Any, ...]
    legacy_index_count: int
    compact_index_count: int
    representative: Any = field(repr=False, compare=False)

    @property
    def spec(self) -> SupportSpec | None:
        """Return the canonical structural description, when available."""

        return None if self.support_map is None else self.support_map.spec

    def gather(self, field):
        """Gather ``field`` through this support's canonical runtime map.

        The representative supplies only the live compatibility key.  Runtime
        data movement is owned by the interned :class:`SupportMap`; direct
        mechanism mapping remains a fallback for legacy mechanisms that do not
        expose structural support metadata.
        """

        if self.support_map is not None:
            return self.support_map.gather(
                field,
                getattr(self.representative, "key", None),
            )
        return self.representative.get(field)

    def scatter_add_(self, destination, local):
        """Scatter-add ``local`` through the canonical map in place."""

        if self.support_map is not None:
            return self.support_map.scatter_add_(
                destination,
                local,
                getattr(self.representative, "key", None),
            )
        return self.representative.add_(destination, local)

    def scatter_add(self, destination, local):
        """Return ``destination`` with ``local`` added on this support."""

        if self.support_map is not None:
            return self.support_map.scatter_add(
                destination,
                local,
                getattr(self.representative, "key", None),
            )
        return self.representative.add(destination, local)

    def scatter_set(self, local, destination, reference, *, clone=True):
        """Replace this support in ``destination`` using its live key."""

        if self.support_map is not None:
            return self.support_map.scatter_set(
                local,
                destination,
                reference,
                getattr(self.representative, "key", None),
                clone=clone,
            )
        return self.representative.put(
            local,
            destination,
            reference,
            clone=clone,
        )

    def is_injective(self) -> bool:
        """Return whether each local slot addresses a distinct physical slot."""

        if self.support_map is not None:
            return self.support_map.is_injective(
                getattr(self.representative, "key", None)
            )
        key = getattr(self.representative, "key", None)
        if key is None or bool(self.representative.is_composable):
            return True
        if not torch.is_tensor(key) or key.device.type == "meta":
            return False
        return int(torch.unique(key).numel()) == int(key.numel())


@dataclass(frozen=True, slots=True)
class SupportStorageAccounting:
    """Integer-selector counts for the current and compact representations."""

    registered_mechanisms: int
    unique_supports: int
    compatibility_indices: int
    unique_legacy_indices: int
    compact_indices: int

    @property
    def interning_savings(self) -> int:
        """Indices avoided if each unique legacy selector were stored once."""

        return self.compatibility_indices - self.unique_legacy_indices

    @property
    def compact_savings(self) -> int:
        """Potential reduction versus today's per-mechanism key buffers."""

        return self.compatibility_indices - self.compact_indices


class SupportRegistry:
    """Intern mechanism supports into handler-wide, authored-order identities.

    The registry is deliberately not an :class:`~torch.nn.Module`, owns no
    tensors, and does not participate in a state dictionary.  Its plain strong
    references keep live ``key`` lookup safe when ``Module.to()`` replaces a
    registered buffer object.  Copying or pickling produces an empty registry;
    an owning handler must rebuild it from its authored mechanism sequence.
    """

    def __init__(self, mechanisms: Iterable[Any] = ()):
        self._entries_by_id: list[SupportEntry | None] = []
        self._active_ids: list[int] = []
        self._buckets: dict[tuple[Any, ...], list[int]] = {}
        self._support_by_object: dict[int, int] = {}
        self._objects_by_id: dict[int, Any] = {}
        self._compatibility_index_count = 0
        self.rebuild(mechanisms)

    def clear(self):
        """Discard all derived registrations and reset authored ordinals."""

        self._entries_by_id.clear()
        self._active_ids.clear()
        self._buckets.clear()
        self._support_by_object.clear()
        self._objects_by_id.clear()
        self._compatibility_index_count = 0

    def rebuild(self, mechanisms: Iterable[Any]):
        """Recreate IDs from one authored mechanism sequence."""

        self.clear()
        for mechanism in mechanisms:
            self.intern(mechanism)
        return self

    def intern(self, mechanism) -> int:
        """Register one authored occurrence and return its stable support ID.

        Repeated occurrences of the same object (for example ModuleDict
        aliases) retain the object's first support ID but still consume an
        authored ordinal.  Use :meth:`support_id` for a lookup that does not
        register another occurrence.
        """

        object_id = id(mechanism)
        registered = self._objects_by_id.get(object_id)
        if registered is mechanism:
            support_id = self._support_by_object[object_id]
            legacy_index_count = _legacy_index_count(mechanism)
            self._entries_by_id.append(None)
            self._compatibility_index_count += legacy_index_count
            return support_id

        authored_ordinal = len(self._entries_by_id)
        signature = _layout_signature(mechanism)
        legacy_index_count = _legacy_index_count(mechanism)
        compact_index_count = _compact_index_count(mechanism)
        support_id = None
        for candidate_id in self._buckets.get(signature, ()):
            candidate = self.entry(candidate_id)
            if _same_support_in_bucket(mechanism, candidate.representative):
                support_id = candidate_id
                break

        if support_id is None:
            support_id = authored_ordinal
            support_map = _support_map(mechanism)
            entry = SupportEntry(
                support_id=support_id,
                support_map=support_map,
                layout_signature=signature,
                legacy_index_count=legacy_index_count,
                compact_index_count=compact_index_count,
                representative=mechanism,
            )
            self._entries_by_id.append(entry)
            self._active_ids.append(support_id)
            self._buckets.setdefault(signature, []).append(support_id)
        else:
            # SupportMap and SupportSpec are immutable structural metadata. Make
            # the successful intern concrete while retaining every mechanism's
            # registered compatibility key buffer unchanged.
            canonical_map = self.entry(support_id).support_map
            if canonical_map is not None and mechanism.support_map is not canonical_map:
                mechanism.support_map = canonical_map
                mechanism.support_spec = canonical_map.spec
            self._entries_by_id.append(None)

        self._support_by_object[object_id] = support_id
        self._objects_by_id[object_id] = mechanism
        self._compatibility_index_count += legacy_index_count
        return support_id

    def support_id(self, mechanism) -> int:
        """Look up a registered mechanism without changing registry order."""

        object_id = id(mechanism)
        if self._objects_by_id.get(object_id) is not mechanism:
            raise KeyError("Mechanism is not registered in this SupportRegistry.")
        return self._support_by_object[object_id]

    def entry(self, support_id: int) -> SupportEntry:
        """Return one active entry, rejecting sparse or invalid IDs clearly."""

        if isinstance(support_id, bool) or not isinstance(support_id, int):
            raise TypeError("support_id must be an integer.")
        if support_id < 0 or support_id >= len(self._entries_by_id):
            raise KeyError(f"Unknown support ID {support_id}.")
        entry = self._entries_by_id[support_id]
        if entry is None:
            raise KeyError(f"Support ID {support_id} is not an active group.")
        return entry

    def partition(self, mechanisms: Iterable[Any]):
        """Build a phase plan using only previously registered global IDs.

        Returns the phase's representatives in first-use order, a mechanism/ID
        plan, and an object-id/ID mapping.  IDs remain handler-global and may be
        sparse; callers should use :meth:`entry` rather than tuple indexing.
        """

        representatives = []
        seen_supports = set()
        plan = []
        by_object = {}
        for mechanism in mechanisms:
            support_id = self.support_id(mechanism)
            if support_id not in seen_supports:
                representatives.append(self.entry(support_id).representative)
                seen_supports.add(support_id)
            plan.append((mechanism, support_id))
            by_object[id(mechanism)] = support_id
        return tuple(representatives), tuple(plan), by_object

    @property
    def active_ids(self) -> tuple[int, ...]:
        """Active support IDs in authored first-occurrence order."""

        return tuple(self._active_ids)

    @property
    def entries_by_id(self) -> tuple[SupportEntry | None, ...]:
        """Sparse authored-ordinal table used by support plans."""

        return tuple(self._entries_by_id)

    @property
    def entries(self) -> tuple[SupportEntry, ...]:
        """Active entries in authored first-occurrence order."""

        return tuple(self.entry(support_id) for support_id in self._active_ids)

    @property
    def representatives(self) -> tuple[Any, ...]:
        """Canonical live representative for each active support."""

        return tuple(entry.representative for entry in self.entries)

    @property
    def registered_count(self) -> int:
        return len(self._entries_by_id)

    @property
    def unique_support_count(self) -> int:
        return len(self._active_ids)

    @property
    def accounting(self) -> SupportStorageAccounting:
        """Return selector storage counts without allocating any selectors."""

        return SupportStorageAccounting(
            registered_mechanisms=self.registered_count,
            unique_supports=self.unique_support_count,
            compatibility_indices=self._compatibility_index_count,
            unique_legacy_indices=sum(
                entry.legacy_index_count for entry in self.entries
            ),
            compact_indices=sum(entry.compact_index_count for entry in self.entries),
        )

    def __len__(self) -> int:
        return self.unique_support_count

    def __iter__(self) -> Iterator[SupportEntry]:
        return iter(self.entries)

    def __copy__(self):
        return type(self)()

    def __deepcopy__(self, memo):
        result = type(self)()
        memo[id(self)] = result
        return result

    def __reduce__(self):
        return (type(self), ())


__all__ = ["SupportEntry", "SupportRegistry", "SupportStorageAccounting"]
