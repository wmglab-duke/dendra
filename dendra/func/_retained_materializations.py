"""Retained-materialization coordinates and call-local packet operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


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
