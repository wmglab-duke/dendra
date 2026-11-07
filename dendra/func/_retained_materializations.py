"""Retained-materialization coordinates and call-local packet operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from ._tensor_contracts import _clone_tensor, _shape_tuple
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


@dataclass(frozen=True)
class _RetainedMaterialization:
    cached: torch.Tensor | None
    value: torch.Tensor
    active: torch.Tensor
    baseline: torch.Tensor | None


@dataclass(frozen=True)
class _SnapshotMaterialization:
    value: torch.Tensor
    path: str
    baselines: dict[str, torch.Tensor]


def _materialization_overrides(
    owner,
    constants,
    *,
    prefix="",
    use_snapshot_offsets=True,
    snapshot_baseline_capture=None,
):
    """Construct call-local overrides in the original prefix and lookup order."""
    if not owner._cache_value_specs and not owner._retained_materializations.slots:
        return None, None
    values = {
        path.removeprefix(prefix): constants[name]
        for name, path, _shape, _dtype, _device in owner._cache_value_specs
        if path.startswith(prefix)
    }
    for slot in owner._retained_materializations.slots:
        if slot.path.startswith(prefix):
            relative = slot.path.removeprefix(prefix)
            values[relative] = (
                _SnapshotMaterialization(
                    constants[slot.value_key], slot.path, snapshot_baseline_capture
                )
                if snapshot_baseline_capture is not None
                else _RetainedMaterialization(
                    values.get(relative),
                    constants[slot.value_key],
                    constants[slot.active_key],
                    constants[slot.baseline_key] if use_snapshot_offsets else None,
                )
            )
    return values, constants.get("cache.active")


def _cache_materialization(value, name, cache_values, cache_active):
    """Select an explicit native materialization before its numerical consumers."""
    if cache_values is None or name not in cache_values:
        return value
    selected = cache_values[name]
    if isinstance(selected, _SnapshotMaterialization):
        # Extraction records each operation's baseline before supplying the
        # native result to downstream preparation. Integrator baselines must
        # be measured after native physical buffers are selected.
        selected.baselines[selected.path] = _clone_tensor(value)
        return selected.value
    if isinstance(selected, _RetainedMaterialization):
        if selected.baseline is not None:
            # Subtract the computed baseline before adding the native value:
            # a large raw write must not cancel a small retained native operand.
            value = (
                selected.value ^ (value ^ selected.baseline)
                if value.dtype == torch.bool
                else selected.value + (value - selected.baseline)
            )
        if selected.cached is not None:
            value = torch.where(cache_active, selected.cached.detach(), value)
        return torch.where(selected.active, selected.value, value)
    return torch.where(cache_active, selected.detach(), value)


def _retained_workspace_builder(module_path, owner, phase, cache_values, cache_active):
    """Identify wholly discarded preparation before authored getter arithmetic."""
    from dendra.models.mechanisms._noise_workspaces import noise_workspace_phases

    if cache_values is None or phase not in noise_workspace_phases(owner):
        return None
    active = []
    for name in getattr(owner, f"_{phase}_buffers"):
        retained = cache_values.get(f"{module_path}.{name}")
        if isinstance(retained, _RetainedMaterialization):
            selected = retained.active
            if retained.cached is not None and cache_active is not None:
                selected = torch.logical_or(selected, cache_active)
        elif torch.is_tensor(retained) and cache_active is not None:
            selected = cache_active
        else:
            return None
        active.append(selected)
    return torch.stack(active).all() if active else None


def _seed_snapshot_frame(owner, population, constants):
    """Seed a retained snapshot in separate value, active and baseline phases."""
    constants.update(
        {
            slot.value_key: _clone_tensor(population.get_buffer(slot.path))
            for slot in owner._retained_materializations.slots
        }
    )
    constants.update(
        {
            slot.active_key: torch.zeros((), device=owner.device, dtype=torch.bool)
            for slot in owner._retained_materializations.slots
        }
    )
    constants.update(
        {
            slot.baseline_key: torch.zeros_like(constants[slot.value_key])
            for slot in owner._retained_materializations.slots
        }
    )


def _complete_snapshot_baselines(owner, constants, baselines):
    """Validate captured snapshot coverage before publishing its baselines."""
    expected_baselines = {
        slot.path
        for slot in owner._retained_materializations.slots
        if slot.path != "integrator.dt"
    }
    if set(baselines) != expected_baselines:
        raise FunctionalizationError(
            "Snapshot preparation did not visit every retained operand: "
            f"missing={sorted(expected_baselines - set(baselines))}, "
            f"unexpected={sorted(set(baselines) - expected_baselines)}"
        )
    constants.update(
        {
            owner._retained_materializations.by_path[path].baseline_key: baseline
            for path, baseline in baselines.items()
        }
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
