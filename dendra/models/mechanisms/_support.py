"""Canonical structural support metadata for compiled mechanisms.

This module separates *where* a mechanism is installed from the tensor layout
used to store it. :class:`SupportSpec` is the immutable structural plan;
:class:`SupportMap` executes that plan using either the legacy packed slot axis
or the opt-in population-preserving ``(N, K)`` layout.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import torch

_SUPPORT_SPEC_PICKLE_VERSION = 1


def _restore_support_spec(version, payload):
    """Restore a SupportSpec from its stable, versioned pickle payload."""

    if version != _SUPPORT_SPEC_PICKLE_VERSION:
        raise ValueError(
            f"Unsupported SupportSpec pickle version {version}; expected "
            f"{_SUPPORT_SPEC_PICKLE_VERSION}."
        )
    payload = dict(payload)
    payload["kind"] = SupportKind(payload["kind"])
    return SupportSpec(**payload)


class SupportKind(str, Enum):
    """Canonical structural layouts understood by the support compiler."""

    DENSE = "dense"
    RECTANGULAR = "rectangular"
    SHARED_COLUMNS = "shared_columns"
    # Reserved for the later row-indexed rollout. Phase 1 deliberately leaves
    # these supports packed so it does not duplicate an O(P*K) index payload.
    ROWWISE = "rowwise"
    PACKED_FLAT = "packed_flat"


def _shape_tuple(value, *, name: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(size) for size in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an iterable of integer sizes.") from error
    if any(size < 0 for size in shape):
        raise ValueError(f"{name} cannot contain negative sizes; got {shape}.")
    return shape


def _normalize_rectangular_key(key, core_shape):
    if not isinstance(key, tuple) or len(key) != len(core_shape):
        return None
    normalized = []
    for selector, size in zip(key, core_shape):
        if not isinstance(selector, slice):
            return None
        start, stop, step = selector.indices(size)
        normalized.append((start, stop, step))
    return tuple(normalized)


def _slice_size(item: tuple[int, int, int]) -> int:
    return len(range(*item))


def _flat_key_tensor(key) -> torch.Tensor:
    tensor = torch.as_tensor(key, dtype=torch.long)
    if tensor.device.type == "meta":
        return tensor.reshape(-1)
    return tensor.detach().to(device="cpu", dtype=torch.long).reshape(-1).contiguous()


def _fingerprint(flat_key: torch.Tensor) -> str | None:
    if flat_key.device.type == "meta":
        return None
    # A fingerprint is only a fast rejection/diagnostic key. Exact equality of
    # packed runtime keys remains mandatory before two supports are coalesced.
    payload = flat_key.numpy().tobytes(order="C")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _shared_column_layout(flat_key: torch.Tensor, core_shape):
    """Return ordered rows/columns when ``flat_key`` is a Cartesian support."""

    if flat_key.device.type == "meta" or flat_key.numel() == 0:
        return None
    population, compartments = core_shape
    if bool(torch.any((flat_key < 0) | (flat_key >= population * compartments))):
        return None

    rows = torch.div(flat_key, compartments, rounding_mode="floor").tolist()
    columns = torch.remainder(flat_key, compartments).tolist()
    row_order = []
    column_groups = []
    seen_rows = set()
    for row, column in zip(rows, columns):
        row = int(row)
        column = int(column)
        if not row_order or row != row_order[-1]:
            if row in seen_rows:
                # Returning to an earlier row would change legacy slot order if
                # represented as a rectangular [P,K] tensor.
                return None
            seen_rows.add(row)
            row_order.append(row)
            column_groups.append([])
        column_groups[-1].append(column)

    first_columns = tuple(column_groups[0])
    if any(tuple(group) != first_columns for group in column_groups[1:]):
        return None
    if len(set(first_columns)) != len(first_columns):
        return None
    return tuple(row_order), first_columns


@dataclass(frozen=True, slots=True)
class SupportSpec:
    """Immutable, device-independent description of mechanism placement.

    ``local_core_shape`` is the natural population-preserving shape.
    ``legacy_local_shape`` records the packed compatibility ABI, while
    :attr:`runtime_local_shape` selects between them. Packed layouts retain only
    a small fingerprint; callers must provide their registered runtime key when
    materializing exact indices.
    """

    kind: SupportKind
    source_core_shape: tuple[int, int]
    local_core_shape: tuple[int, ...]
    legacy_local_shape: tuple[int, ...]
    slot_count: int
    preserves_multiplicity: bool = False
    has_duplicates: bool = False
    normalized_slices: tuple[tuple[int, int, int], ...] = ()
    row_indices: tuple[int, ...] = ()
    column_indices: tuple[int, ...] = ()
    all_populations: bool = False
    flat_fingerprint: str | None = None
    force_packed: bool = False
    preserves_population_axis: bool = False

    @classmethod
    def from_compiled(
        cls,
        core_shape,
        key,
        is_composable: bool,
        local_shape,
        *,
        preserves_multiplicity: bool = False,
        force_packed: bool = False,
    ) -> "SupportSpec":
        """Classify one already-compiled support without changing its ABI."""

        core_shape = _shape_tuple(core_shape, name="core_shape")
        if len(core_shape) != 2:
            raise ValueError(
                "Mechanism supports require a two-axis (population, compartment) "
                f"core shape; got {core_shape}."
            )
        legacy_local_shape = _shape_tuple(local_shape, name="local_shape")
        preserves_multiplicity = bool(preserves_multiplicity)
        force_packed = bool(force_packed)
        population, compartments = core_shape

        if key is None:
            slot_count = math.prod(core_shape)
            if math.prod(legacy_local_shape) != slot_count:
                raise ValueError(
                    "Dense mechanism local shape must contain every core slot: "
                    f"expected {slot_count}, got {legacy_local_shape}."
                )
            return cls(
                kind=SupportKind.DENSE,
                source_core_shape=core_shape,
                local_core_shape=core_shape,
                legacy_local_shape=legacy_local_shape,
                slot_count=slot_count,
                all_populations=True,
            )

        if is_composable and not force_packed and not preserves_multiplicity:
            normalized = _normalize_rectangular_key(key, core_shape)
            if normalized is not None:
                local_core_shape = tuple(_slice_size(item) for item in normalized)
                slot_count = math.prod(local_core_shape)
                if math.prod(legacy_local_shape) != slot_count:
                    raise ValueError(
                        "Rectangular mechanism local shape does not match its "
                        f"selector: expected {local_core_shape}, got "
                        f"{legacy_local_shape}."
                    )
                full = normalized == tuple((0, size, 1) for size in core_shape)
                return cls(
                    kind=(SupportKind.DENSE if full else SupportKind.RECTANGULAR),
                    source_core_shape=core_shape,
                    local_core_shape=(core_shape if full else local_core_shape),
                    legacy_local_shape=legacy_local_shape,
                    slot_count=slot_count,
                    normalized_slices=(() if full else normalized),
                    all_populations=full,
                )

        flat_key = _flat_key_tensor(key)
        slot_count = int(flat_key.numel())
        if math.prod(legacy_local_shape) != slot_count:
            raise ValueError(
                "Flat mechanism local shape must match its number of support "
                f"slots: expected {slot_count}, got {legacy_local_shape}."
            )
        if flat_key.device.type != "meta" and bool(
            torch.any((flat_key < 0) | (flat_key >= population * compartments))
        ):
            raise ValueError(
                "Mechanism support key contains an index outside the source "
                f"core range [0, {population * compartments})."
            )
        if flat_key.device.type == "meta":
            has_duplicates = False
        else:
            has_duplicates = int(torch.unique(flat_key).numel()) != slot_count

        shared = None
        if not force_packed and not preserves_multiplicity and not has_duplicates:
            shared = _shared_column_layout(flat_key, core_shape)
        if shared is not None:
            rows, columns = shared
            all_populations = rows == tuple(range(population))
            return cls(
                kind=SupportKind.SHARED_COLUMNS,
                source_core_shape=core_shape,
                local_core_shape=(len(rows), len(columns)),
                legacy_local_shape=legacy_local_shape,
                slot_count=slot_count,
                row_indices=(() if all_populations else rows),
                column_indices=columns,
                all_populations=all_populations,
                flat_fingerprint=_fingerprint(flat_key),
            )

        return cls(
            kind=SupportKind.PACKED_FLAT,
            source_core_shape=core_shape,
            local_core_shape=legacy_local_shape,
            legacy_local_shape=legacy_local_shape,
            slot_count=slot_count,
            preserves_multiplicity=preserves_multiplicity,
            has_duplicates=has_duplicates,
            flat_fingerprint=_fingerprint(flat_key),
            force_packed=force_packed,
        )

    @property
    def ordered_signature(self) -> tuple[Any, ...]:
        """Return hashable support identity independent of tensor device."""

        return (
            self.kind.value,
            self.source_core_shape,
            self.local_core_shape,
            self.legacy_local_shape,
            self.slot_count,
            self.preserves_multiplicity,
            self.has_duplicates,
            self.normalized_slices,
            self.row_indices,
            self.column_indices,
            self.all_populations,
            self.flat_fingerprint,
            self.force_packed,
            self.preserves_population_axis,
        )

    def checkpoint_identity(self) -> dict[str, Any]:
        """Return named, device-independent metadata for checkpoint identity."""

        return {
            "kind": self.kind.value,
            "source_core_shape": self.source_core_shape,
            "local_core_shape": self.local_core_shape,
            "legacy_local_shape": self.legacy_local_shape,
            "runtime_local_shape": self.runtime_local_shape,
            "slot_count": self.slot_count,
            "preserves_multiplicity": self.preserves_multiplicity,
            "has_duplicates": self.has_duplicates,
            "normalized_slices": self.normalized_slices,
            "row_indices": self.row_indices,
            "column_indices": self.column_indices,
            "all_populations": self.all_populations,
            "flat_fingerprint": self.flat_fingerprint,
            "force_packed": self.force_packed,
            "preserves_population_axis": self.preserves_population_axis,
        }

    def __reduce__(self):
        """Pickle through a named schema rather than dataclass slot order."""

        payload = self.checkpoint_identity()
        payload.pop("runtime_local_shape")
        return (
            _restore_support_spec,
            (_SUPPORT_SPEC_PICKLE_VERSION, payload),
        )

    @property
    def runtime_local_shape(self) -> tuple[int, ...]:
        """Mechanism-visible support shape for the selected runtime layout."""

        if self.preserves_population_axis:
            return self.local_core_shape
        return self.legacy_local_shape

    def with_population_axis(self) -> "SupportSpec":
        """Return an equivalent shared-column spec with ``(N, K)`` storage."""

        if self.kind is not SupportKind.SHARED_COLUMNS or not self.all_populations:
            raise ValueError(
                "Population-axis storage requires shared ordered columns across "
                "every population row."
            )
        if self.source_core_shape[0] <= 1:
            raise ValueError(
                "Population-axis storage is only distinct for populations with "
                "more than one row."
            )
        return replace(self, preserves_population_axis=True)

    @property
    def has_compact_identity(self) -> bool:
        """Whether equality is exact without consulting a runtime flat key."""

        return self.kind is not SupportKind.PACKED_FLAT

    @property
    def legacy_index_count(self) -> int:
        """Number of integer indices stored by today's runtime representation."""

        if self.kind in {SupportKind.DENSE, SupportKind.RECTANGULAR}:
            return 0
        return self.slot_count

    @property
    def compact_index_count(self) -> int:
        """Integer-index count required by the planned compact representation."""

        if self.kind in {SupportKind.DENSE, SupportKind.RECTANGULAR}:
            return 0
        if self.kind is SupportKind.SHARED_COLUMNS:
            return len(self.column_indices) + (
                0 if self.all_populations else len(self.row_indices)
            )
        return self.slot_count

    def materialized_flat_indices(
        self,
        runtime_key=None,
        *,
        device=None,
    ) -> torch.Tensor:
        """Materialize exact legacy slot order for testing and compatibility."""

        population, compartments = self.source_core_shape
        if self.kind is SupportKind.DENSE:
            result = torch.arange(population * compartments, dtype=torch.long)
        elif self.kind is SupportKind.RECTANGULAR:
            row_slice, column_slice = self.normalized_slices
            rows = torch.tensor(list(range(*row_slice)), dtype=torch.long)
            columns = torch.tensor(list(range(*column_slice)), dtype=torch.long)
            result = (rows[:, None] * compartments + columns[None, :]).reshape(-1)
        elif self.kind is SupportKind.SHARED_COLUMNS:
            rows = (
                torch.arange(population, dtype=torch.long)
                if self.all_populations
                else torch.tensor(self.row_indices, dtype=torch.long)
            )
            columns = torch.tensor(self.column_indices, dtype=torch.long)
            result = (rows[:, None] * compartments + columns[None, :]).reshape(-1)
        else:
            if runtime_key is None:
                raise ValueError(
                    "PACKED_FLAT support materialization requires its runtime key."
                )
            result = _flat_key_tensor(runtime_key)
            if int(result.numel()) != self.slot_count:
                raise ValueError(
                    "Packed runtime key length does not match its SupportSpec: "
                    f"expected {self.slot_count}, got {result.numel()}."
                )
            fingerprint = _fingerprint(result)
            if (
                self.flat_fingerprint is not None
                and fingerprint is not None
                and fingerprint != self.flat_fingerprint
            ):
                raise ValueError("Packed runtime key does not match this SupportSpec.")
        return result.to(device=device) if device is not None else result

    def validate_runtime_key(self, runtime_key, *, context="Mechanism support"):
        """Validate a live flat selector against this immutable support plan."""

        if self.kind in {SupportKind.DENSE, SupportKind.RECTANGULAR}:
            return
        key = _flat_key_tensor(runtime_key)
        if key.device.type == "meta":
            # A meta move deliberately discards values; the device-independent
            # plan remains authoritative until real storage is restored.
            return
        if int(key.numel()) != self.slot_count:
            raise ValueError(
                f"{context} key length changed: expected {self.slot_count}, "
                f"got {key.numel()}. Structural selector keys are immutable."
            )
        if self.kind is SupportKind.SHARED_COLUMNS:
            expected = self.materialized_flat_indices()
            if not torch.equal(key, expected):
                raise ValueError(
                    f"{context} key no longer matches its compiled placement. "
                    "Structural selector keys are immutable; change placement "
                    "through Population.insert/delete and rebuild instead."
                )
            return
        if (
            self.flat_fingerprint is not None
            and _fingerprint(key) != self.flat_fingerprint
        ):
            raise ValueError(
                f"{context} packed key no longer matches its compiled placement. "
                "Structural selector keys are immutable; change placement "
                "through Population.insert/delete and rebuild instead."
            )


@dataclass(frozen=True, slots=True)
class SupportMap:
    """Execute reads and writes for one immutable :class:`SupportSpec`.

    The map owns no tensors. In particular, it never retains a reference to a
    mechanism's registered ``key`` buffer because ``Module.to()`` and state
    loading may replace that tensor object. Callers pass the live key to each
    operation. Shared-column supports use only one row of that compatibility
    key at runtime and return the shape selected by
    :attr:`SupportSpec.runtime_local_shape`.
    """

    spec: SupportSpec

    def __post_init__(self):
        if not isinstance(self.spec, SupportSpec):
            raise TypeError("SupportMap.spec must be a SupportSpec instance.")

    @staticmethod
    def _require_runtime_key(runtime_key) -> torch.Tensor:
        if runtime_key is None:
            raise ValueError("This support operation requires its live runtime key.")
        return torch.as_tensor(runtime_key, dtype=torch.long).reshape(-1)

    def _flat_runtime_key(self, runtime_key) -> torch.Tensor:
        key = self._require_runtime_key(runtime_key)
        if int(key.numel()) != self.spec.slot_count:
            raise ValueError(
                "Runtime key length does not match its SupportSpec: "
                f"expected {self.spec.slot_count}, got {key.numel()}."
            )
        return key

    def _shared_columns(self, runtime_key, *, device) -> torch.Tensor:
        column_count = len(self.spec.column_indices)
        if self.spec.all_populations and runtime_key is not None:
            # The first population's physical flat indices are exactly the
            # compartment columns. This is a view into the compatibility key,
            # so eager execution allocates no P*K or K-sized index tensor.
            key = self._flat_runtime_key(runtime_key)
            columns = key[:column_count]
            return columns if columns.device == device else columns.to(device=device)
        return torch.as_tensor(
            self.spec.column_indices,
            dtype=torch.long,
            device=device,
        )

    def _uses_compact_shared_columns(self, tensor: torch.Tensor) -> bool:
        return (
            self.spec.kind is SupportKind.SHARED_COLUMNS
            and self.spec.all_populations
            and tensor.ndim >= 2
        )

    def gather(self, tensor: torch.Tensor, runtime_key=None) -> torch.Tensor:
        """Gather a full field in exact row-major local-slot order."""

        if tensor.ndim == 0 or self.spec.kind is SupportKind.DENSE:
            return tensor
        if self.spec.kind is SupportKind.RECTANGULAR:
            selectors = tuple(slice(*item) for item in self.spec.normalized_slices)
            return tensor[..., *selectors]
        if self._uses_compact_shared_columns(tensor):
            columns = self._shared_columns(runtime_key, device=tensor.device)
            local = tensor.index_select(-1, columns)
            batch_shape = tuple(tensor.shape[:-2])
            return local.reshape(batch_shape + self.spec.runtime_local_shape)

        key = self._flat_runtime_key(runtime_key)
        batch_shape = tuple(tensor.shape[:-2]) if tensor.ndim >= 2 else ()
        return tensor.reshape(batch_shape + (-1,)).index_select(-1, key)

    def scatter_add_(
        self,
        destination: torch.Tensor,
        local,
        runtime_key=None,
    ) -> torch.Tensor:
        """Add local values to a full field in place."""

        if self.spec.kind is SupportKind.DENSE:
            return destination.add_(local)
        if self.spec.kind is SupportKind.RECTANGULAR:
            selectors = tuple(slice(*item) for item in self.spec.normalized_slices)
            destination[..., *selectors].add_(local)
            return destination
        if self._uses_compact_shared_columns(destination):
            columns = self._shared_columns(runtime_key, device=destination.device)
            population = self.spec.source_core_shape[0]
            column_count = len(self.spec.column_indices)
            batch_shape = tuple(destination.shape[:-2])
            structured_shape = batch_shape + (population, column_count)
            values = torch.as_tensor(
                local,
                device=destination.device,
                dtype=destination.dtype,
            ).expand(batch_shape + self.spec.runtime_local_shape)
            values = values.reshape(structured_shape)
            indices = columns.expand(structured_shape)
            destination.scatter_add_(-1, indices, values)
            return destination

        key = self._flat_runtime_key(runtime_key)
        batch_shape = tuple(destination.shape[:-2]) if destination.ndim >= 2 else ()
        flat_destination = destination.reshape(batch_shape + (-1,))
        expanded_key = key.expand(batch_shape + (key.numel(),))
        values = torch.as_tensor(
            local,
            device=destination.device,
            dtype=destination.dtype,
        ).expand_as(expanded_key)
        flat_destination.scatter_add_(-1, expanded_key, values)
        return destination

    def scatter_add(
        self,
        destination: torch.Tensor,
        local,
        runtime_key=None,
    ) -> torch.Tensor:
        """Return a full field with local values added."""

        result = destination.clone()
        return self.scatter_add_(result, local, runtime_key)

    def scatter_set(
        self,
        local,
        destination: torch.Tensor,
        reference: torch.Tensor,
        runtime_key=None,
        *,
        clone: bool = True,
    ) -> torch.Tensor:
        """Replace supported entries in ``destination`` and return the field."""

        if self.spec.kind is SupportKind.DENSE:
            return local

        destination = destination.expand_as(reference)
        if clone:
            destination = destination.clone()

        if self.spec.kind is SupportKind.RECTANGULAR:
            selectors = tuple(slice(*item) for item in self.spec.normalized_slices)
            destination[..., *selectors] = local
            return destination
        if self._uses_compact_shared_columns(destination):
            columns = self._shared_columns(runtime_key, device=destination.device)
            population = self.spec.source_core_shape[0]
            column_count = len(self.spec.column_indices)
            batch_shape = tuple(destination.shape[:-2])
            structured_shape = batch_shape + (population, column_count)
            values = torch.as_tensor(
                local,
                device=destination.device,
                dtype=destination.dtype,
            ).expand(batch_shape + self.spec.runtime_local_shape)
            values = values.reshape(structured_shape)
            destination.scatter_(-1, columns.expand(structured_shape), values)
            return destination

        key = self._flat_runtime_key(runtime_key)
        batch_shape = tuple(destination.shape[:-2]) if destination.ndim >= 2 else ()
        flat_destination = destination.reshape(batch_shape + (-1,))
        expanded_key = key.expand(batch_shape + (key.numel(),))
        flat_destination.scatter_(-1, expanded_key, local)
        return destination

    def same_ordered_support(
        self,
        other: "SupportMap",
        runtime_key=None,
        other_runtime_key=None,
    ) -> bool:
        """Return whether two maps select identical physical slots in order."""

        if not isinstance(other, SupportMap):
            return False
        if self.spec.ordered_signature != other.spec.ordered_signature:
            return False
        if self.spec.has_compact_identity and other.spec.has_compact_identity:
            return True

        left_key = self._flat_runtime_key(runtime_key)
        right_key = other._flat_runtime_key(other_runtime_key)
        if left_key.device.type == "meta" or right_key.device.type == "meta":
            return runtime_key is other_runtime_key
        return (
            left_key.shape == right_key.shape
            and left_key.dtype == right_key.dtype
            and left_key.device == right_key.device
            and torch.equal(left_key, right_key)
        )

    def is_injective(self, runtime_key=None) -> bool:
        """Return whether each local slot writes to a distinct physical slot."""

        if self.spec.kind is not SupportKind.PACKED_FLAT:
            return True
        if self.spec.flat_fingerprint is not None:
            return not self.spec.has_duplicates
        key = self._flat_runtime_key(runtime_key)
        if key.device.type == "meta":
            return False
        return int(torch.unique(key).numel()) == int(key.numel())


__all__ = ["SupportKind", "SupportMap", "SupportSpec"]
