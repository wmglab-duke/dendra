"""Pure reconstruction of declared workspaces from retained NOISE frames.

The native lifecycle owns the provenance schema and builder transactions.
Functional execution supplies those registered tensors as carry, then uses
the same builders with live deterministic parameters and detached held NOISE.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from dendra.models.mechanisms._noise_workspaces import (
    noise_workspace_noise_bindings,
    noise_workspace_phases,
    noise_workspace_slots,
    reset_noise_workspace_timestep,
    temporary_noise_workspace,
)


def noise_workspace_owners(population):
    """Yield canonical owners in native parameter-population order."""
    from ._population import _parameterized_owner_paths

    return tuple(
        (path, owner)
        for path, owner in _parameterized_owner_paths(population)
        if noise_workspace_phases(owner)
    )


def noise_workspace_bindings(population):
    """Map registered native slots to explicit functional carry paths."""
    return {
        f"{path + '.' if path else ''}{buffer_name}": (
            "noise_workspaces",
            path,
            suffix,
        )
        for path, owner in noise_workspace_owners(population)
        for suffix, _slot_owner, buffer_name in noise_workspace_slots(owner)
    }


def _live_noise_bindings(population):
    """Deduplicate the canonical live NOISE operands reachable by builders."""
    from ._population import _parameterized_owner_paths

    required = {
        (id(noise_owner), name)
        for _path, owner in noise_workspace_owners(population)
        for phase in noise_workspace_phases(owner)
        for _relative, noise_owner, name, _held in noise_workspace_noise_bindings(
            owner, phase
        )
    }
    return tuple(
        (f"{path + '.' if path else ''}{name}", owner, name)
        for path, owner in _parameterized_owner_paths(population)
        for name in owner.runtime_noises
        if (id(owner), name) in required
    )


def _noise_workspace_constant_references(population):
    result = {
        f"noise_workspaces.{path + '.' if path else ''}{buffer_name}": slot_owner._buffers[
            buffer_name
        ]
        for path, owner in noise_workspace_owners(population)
        for suffix, slot_owner, buffer_name in noise_workspace_slots(owner)
        if ".buffer." not in suffix
    }
    result.update(
        {
            f"noise_workspaces.{path}": owner._buffers[name]
            for path, owner, name in _live_noise_bindings(population)
        }
    )
    return result


def noise_workspace_constant_values(population):
    """Extract retained frames and live fallback NOISE preparation operands."""
    return {
        name: value.detach().clone()
        for name, value in _noise_workspace_constant_references(population).items()
    }


def noise_workspace_constants_from_state(population, constants, state):
    """Carry an initialized frame into preparation without numerical copies."""
    result = dict(constants)
    frames = state.get("noise_workspaces", {})
    for path, owner in noise_workspace_owners(population):
        for suffix, _slot_owner, buffer_name in noise_workspace_slots(owner):
            if ".buffer." in suffix:
                continue
            result[f"noise_workspaces.{path + '.' if path else ''}{buffer_name}"] = (
                frames[path][suffix].detach().clone()
            )
    for path, _owner, _name in _live_noise_bindings(population):
        result[f"noise_workspaces.{path}"] = (
            state["runtime_noise"][path].detach().clone()
        )
    return result


def noise_workspace_preparation_mapping(constants):
    """Bind explicit metadata constants to their registered private slots."""
    prefix = "noise_workspaces."
    return {
        f"population.{name.removeprefix(prefix)}": value
        for name, value in constants.items()
        if name.startswith(prefix)
    }


def noise_workspace_preparation_frames(population, constants, dt):
    """Describe the timestep NOISE realization used by explicit preparation."""
    live_paths = {
        (id(owner), name): path
        for path, owner, name in _live_noise_bindings(population)
    }
    result = {}
    for path, owner in noise_workspace_owners(population):
        if "timestep" not in noise_workspace_phases(owner):
            continue
        prefix = f"noise_workspaces.{path + '.' if path else ''}"
        bindings = noise_workspace_noise_bindings(owner, "timestep")
        result[path] = _choose_noise_values(
            constants[f"{prefix}_noise_workspace_timestep_active"],
            {
                relative: constants[f"{prefix}{held_name}"]
                for relative, _noise_owner, _name, held_name in bindings
            },
            {
                relative: constants[
                    f"noise_workspaces.{live_paths[id(noise_owner), name]}"
                ]
                for relative, noise_owner, name, _held_name in bindings
            },
            configured_dt=constants[f"{prefix}_noise_workspace_timestep_dt"],
            dt=dt,
        )
    return result


def validate_noise_workspace_constants(population, constants):
    """Validate phase metadata against the established native tensor schema."""
    references = _noise_workspace_constant_references(population)
    actual = {
        name: value
        for name, value in constants.items()
        if name.startswith("noise_workspaces.")
    }
    if set(actual) != set(references):
        raise ValueError(
            "NOISE workspace constants must match the registered phase metadata names"
        )
    for name, reference in references.items():
        value = actual[name]
        if not torch.is_tensor(value) or (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise ValueError(
                f"NOISE workspace constant {name!r} must match "
                f"{reference.device}/{reference.dtype}/{tuple(reference.shape)}"
            )


def _choose_noise_values(active, held, live, *, configured_dt=None, dt=None):
    if configured_dt is not None and dt is not None:
        selected_dt = torch.as_tensor(
            dt, device=configured_dt.device, dtype=configured_dt.dtype
        )
        active = torch.logical_and(active, configured_dt == selected_dt.detach())
    return {
        path: torch.where(
            active, value.detach(), live[path].detach().reshape(value.shape)
        )
        for path, value in held.items()
    }


def _retained_noise_values(owner, phase, *, dt=None):
    """Choose a phase frame with tensor control flow, including hidden lanes."""
    bindings = noise_workspace_noise_bindings(owner, phase)
    return _choose_noise_values(
        owner._buffers[f"_noise_workspace_{phase}_active"],
        {
            path: owner._buffers[held_name]
            for path, _noise_owner, _name, held_name in bindings
        },
        {
            path: noise_owner._buffers[noise_name]
            for path, noise_owner, noise_name, _held_name in bindings
        },
        configured_dt=(
            owner._buffers["_noise_workspace_timestep_dt"]
            if phase == "timestep"
            else None
        ),
        dt=dt,
    )


@contextmanager
def retained_noise_workspace_frame(owner, phase, *, dt=None):
    """Evaluate one private builder in its native retained NOISE frame.

    Each owner retains the frame consumed by its own builder, including any
    nested State NOISE reads. A global context could merge independently
    configured owner frames.
    """
    if phase not in noise_workspace_phases(owner):
        yield
        return
    with temporary_noise_workspace(
        owner, phase, _retained_noise_values(owner, phase, dt=dt)
    ):
        yield


@contextmanager
def audit_noise_workspace_values(population):
    """Diagnose authored writes after framework workspace refresh completes."""
    from ._population import _audit_tensor_binding
    from ._types import FunctionalizationError

    snapshots = tuple(
        (
            f"{path}.{suffix}",
            slot_owner,
            name,
            _audit_tensor_binding(slot_owner._buffers[name]),
        )
        for path, owner in noise_workspace_owners(population)
        for suffix, slot_owner, name in noise_workspace_slots(owner)
    )
    try:
        yield
    finally:
        changed = tuple(
            label
            for label, owner, name, before in snapshots
            if not torch.is_tensor(owner._buffers.get(name))
            or _audit_tensor_binding(owner._buffers[name]) != before
        )
        if changed:
            raise FunctionalizationError(
                "Authored execution mutated or rebound read-only NOISE workspace "
                f"values: {list(changed)}"
            )


@contextmanager
def _masked_workspace_builder_inputs(owner, matching, dt):
    """Cut a discarded builder route before potentially singular arithmetic."""
    if matching is None:
        yield dt
        return

    aliases = []

    def masked(value):
        for original, replacement in aliases:
            if value is original:
                return replacement
        replacement = torch.where(matching, value.detach(), value)
        aliases.append((value, replacement))
        return replacement

    originals = tuple(
        (collection, name, value)
        for _path, module in owner.named_modules()
        for collection in (module._parameters, module._buffers)
        for name, value in collection.items()
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
    )
    try:
        for collection, name, value in originals:
            collection[name] = masked(value)
        yield masked(dt)
    finally:
        for collection, name, value in reversed(originals):
            collection[name] = value


def refresh_noise_workspaces(
    population,
    dt,
    *,
    initialization=False,
    phases=("derived", "timestep"),
    prepared_frames=None,
):
    """Reconstruct live parameter graphs without drawing or refreshing NOISE.

    Call before binding a step's new runtime NOISE. A timestep change consumes
    the previous accepted runtime realization, matching native ``set_dt``.
    Fresh initialization materializes derived workspaces from its supplied
    initialization realization and leaves timestep workspaces at zero.
    """
    from dendra.models.mechanisms._state import (
        _install_timestep_buffers,
        _materialize_derived_buffers,
        _stage_timestep_buffers,
    )

    owners = noise_workspace_owners(population)
    for _path, owner in owners:
        if "derived" not in phases or "derived" not in noise_workspace_phases(owner):
            continue
        if initialization:
            _materialize_derived_buffers(owner, values=owner.derive_buffers())
        else:
            with retained_noise_workspace_frame(owner, "derived"):
                _materialize_derived_buffers(owner, values=owner.derive_buffers())

    for path, owner in owners:
        if "timestep" not in phases or "timestep" not in noise_workspace_phases(owner):
            continue
        if initialization:
            for name in owner._timestep_buffers:
                owner._buffers[name] = torch.zeros_like(owner._buffers[name]).detach()
            reset_noise_workspace_timestep(owner)
            continue
        values = _retained_noise_values(owner, "timestep", dt=dt)
        reference = None if prepared_frames is None else prepared_frames.get(path)
        matching = None
        if reference is not None:
            matching = owner._buffers["_noise_workspace_timestep_active"].new_ones(())
            for name, value in values.items():
                matching = torch.logical_and(
                    matching, torch.all(value == reference[name])
                )
        with temporary_noise_workspace(owner, "timestep", values):
            with _masked_workspace_builder_inputs(owner, matching, dt) as builder_dt:
                staged = _stage_timestep_buffers(
                    owner,
                    builder_dt,
                    values=owner.derive_timestep_buffers(builder_dt),
                )
            if matching is not None:
                staged = {
                    name: torch.where(matching, owner._buffers[name], value)
                    for name, value in staged.items()
                }
            _install_timestep_buffers(owner, staged, dt=dt, update_schema=False)
