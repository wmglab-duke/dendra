"""Retained-materialization coordinates and call-local packet operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from ._tensor_contracts import _shape_tuple
from ._types import FunctionalizationError


@dataclass(frozen=True)
class _RetainedMaterializationSlot:
    path: str
    value_key: str
    active_key: str
    baseline_key: str
    shape: tuple
    dtype: object
    device: object
    prepared_path: tuple | None


@dataclass(frozen=True)
class _RetainedMaterializationInventory:
    """Own ordered slot metadata, independently of tensor values and graphs."""

    slots: tuple = ()
    by_path: dict = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "by_path", {slot.path: slot for slot in self.slots})

    @classmethod
    def from_specs(cls, specs, prepared_paths):
        return cls(
            tuple(
                _RetainedMaterializationSlot(
                    path=path,
                    value_key=name,
                    active_key=f"materializations.active.{path}",
                    baseline_key=f"materializations.baseline.{path}",
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    prepared_path=prepared_paths.get(path),
                )
                for name, path, shape, dtype, device in specs
            )
        )


def _clear_retained_frame(inventory, constants):
    """Discard value graphs and active flags for a nonempty retained frame."""
    # Cut old graphs before inactive torch.where routes can revisit a
    # completed backward pass. Detached views retain their existing storage.
    return {
        **constants,
        **{
            slot.value_key: constants[slot.value_key].detach()
            for slot in inventory.slots
        },
        **{
            slot.active_key: torch.zeros_like(constants[slot.active_key])
            for slot in inventory.slots
        },
    }


def _capture_retained_frame(inventory, constants, values):
    """Capture a nonempty prepared frame, retaining its differentiable graphs."""
    return {
        **constants,
        **{
            slot.value_key: values[slot.path].clone(memory_format=torch.preserve_format)
            for slot in inventory.slots
        },
        **{
            slot.active_key: torch.ones_like(constants[slot.active_key])
            for slot in inventory.slots
        },
        **{
            slot.baseline_key: torch.zeros_like(constants[slot.baseline_key])
            for slot in inventory.slots
        },
    }


def _project_retained_frame(inventory, values, extras):
    """Select prepared operands without copying tensors or changing their graphs."""
    extras = {} if extras is None else extras
    retained = {}
    for slot in inventory.slots:
        if slot.prepared_path is not None:
            section, key = slot.prepared_path
            value = values[section][key]
        elif slot.path in extras and extras[slot.path] is not None:
            value = extras[slot.path]
        else:
            value = values["geometry"][slot.path]
        retained[slot.path] = value
    return retained


def _validate_retained_constants(owner, constants):
    """Validate retained constant fields in their original slot/field order."""
    # Keep owner metadata reads in the original checks and diagnostic
    # branches rather than prefetching device ahead of the fields.
    for slot in owner._retained_materializations.slots:
        for value_name in (slot.value_key, slot.baseline_key):
            value = constants[value_name]
            if (
                not torch.is_tensor(value)
                or _shape_tuple(value) != slot.shape
                or value.dtype != slot.dtype
                or value.device != slot.device
            ):
                raise ValueError(
                    f"constant {value_name!r} has incompatible retained schema"
                )
        active_name = slot.active_key
        active = constants[active_name]
        if (
            not torch.is_tensor(active)
            or _shape_tuple(active) != ()
            or active.dtype != torch.bool
            or active.device != owner.device
        ):
            raise ValueError(
                f"constant {active_name!r} must be a scalar bool on {owner.device}"
            )


def _validate_retained_prepared(owner, values):
    """Validate prepared retained keys before visiting their Tensor schemas."""
    retained = values["materializations"]
    # A non-Mapping frame fails before reading the owner's slot inventory.
    if not isinstance(retained, Mapping) or set(retained) != {
        slot.path for slot in owner._retained_materializations.slots
    }:
        raise FunctionalizationError("prepared materialization schema is invalid")
    for slot in owner._retained_materializations.slots:
        value = retained[slot.path]
        if (
            not torch.is_tensor(value)
            or _shape_tuple(value) != slot.shape
            or value.dtype != slot.dtype
            or value.device != slot.device
        ):
            raise FunctionalizationError(
                f"prepared retained materialization {slot.path!r} has invalid schema"
            )
