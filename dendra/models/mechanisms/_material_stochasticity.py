"""Shared checkpoint ownership for stochastic material workspaces."""

from __future__ import annotations

import copy
import math

import torch


def retain_stochastic_material_workspace_once(process) -> None:
    """Keep a restored cold workspace through its first solver setup."""
    operator = process._spatial_operators[process._diffusion_material_plan[0][0]]
    process._material_restored_workspace_dt = float(operator._buffers["dt"].item())


def consume_stochastic_material_workspace_restore(process, dt) -> bool:
    """Consume the cold-restore handoff only for its original timestep."""
    restored = process.__dict__.pop("_material_restored_workspace_dt", None)
    if restored is None or not process._spatial_configured:
        return False
    operator = process._spatial_operators[process._diffusion_material_plan[0][0]]
    return float(operator._buffers["dt"].item()) == float(dt)


def stage_stochastic_material_workspace(process, state_dict, *, prefix):
    """Build a cold target's tensor schema without changing any live owner.

    The held realization configures only the staged numerical plan. Current
    runtime NOISE and sampled RAND values are validated independently by the
    handler's ordinary stochastic checkpoint preflight.
    """
    if process._spatial_configured:
        return None
    workspace_prefix = f"{prefix}material_workspace."
    if not any(
        isinstance(key, str) and key.startswith(workspace_prefix) for key in state_dict
    ):
        return None
    dt_keys = tuple(
        f"{workspace_prefix}{entry[0]}.dt" for entry in process._diffusion_material_plan
    )
    timesteps = []
    for key in dt_keys:
        if key not in state_dict:
            raise KeyError(
                f"MechanismHandler material workspace state is missing {key!r}."
            )
        value = state_dict[key]
        if not torch.is_tensor(value) or value.ndim != 0:
            raise ValueError("Material workspace timestep must be a scalar tensor.")
        value = float(value.item())
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Material workspace timestep must be finite and positive.")
        timesteps.append(value)
    if len(set(timesteps)) != 1:
        raise ValueError("Material workspace timesteps must agree within a process.")

    staged = copy.copy(process)
    staged._buffers = dict(process._buffers)
    staged._modules = dict(process._modules)
    staged._parameters = dict(process._parameters)
    staged._non_persistent_buffers_set = set(process._non_persistent_buffers_set)
    for name in process.runtime_noises:
        key = f"{workspace_prefix}noise.{name}"
        if key not in state_dict:
            raise KeyError(
                f"MechanismHandler material workspace state is missing {key!r}."
            )
        value, reference = state_dict[key], process._buffers[name]
        if not torch.is_tensor(value) or value.shape != reference.shape:
            raise ValueError(
                f"Material workspace held NOISE {key!r} has incompatible shape."
            )
        staged._buffers[name] = value.to(reference).detach().clone()
    for name in process.random_parameters:
        key = f"{prefix}random_parameter.{name}"
        value, reference = state_dict.get(key), process._buffers[name]
        if not torch.is_tensor(value) or value.shape != reference.shape:
            raise ValueError(f"Material workspace RAND {key!r} has incompatible shape.")
        staged._buffers[name] = value.to(reference).clone()
    staged.__dict__.pop("_material_restored_workspace_dt", None)
    staged.set_dt(timesteps[0])
    return staged


def capture_stochastic_material_noise(process) -> None:
    """Retain detached NOISE provenance only after operator configuration."""
    process.__dict__.pop("_material_restored_workspace_dt", None)
    for noise_name in process.runtime_noises:
        buffer_name = f"_material_held_noise_{noise_name}"
        value = process._buffers[noise_name].detach().clone()
        if buffer_name in process._buffers:
            process._buffers[buffer_name] = value
        else:
            process.register_buffer(buffer_name, value, persistent=False)


def stochastic_material_workspace_slots(process):
    """Yield the held numerical diffusion workspace behind explicit names.

    Runtime noise is sampled on every accepted step, whereas native diffusion
    holds the operator configured at the timestep boundary.  Those coefficients
    therefore belong to a stochastic runtime checkpoint independently of the
    most recently sampled noise.  Functional execution keeps the same tensors
    in registered prepared slots and uses the same checkpoint suffixes.
    """
    from ._material_process import DiffusionProcess

    if not isinstance(process, DiffusionProcess) or not process.runtime_noises:
        return ()
    private = getattr(process, "_functional_material_workspace_slots", None)
    if private is not None:
        return tuple((suffix, process, name) for suffix, name in private)
    if not process._spatial_configured:
        return ()
    slots = []
    for entry in process._diffusion_material_plan:
        operator_name = entry[0]
        operator = process._spatial_operators[operator_name]
        # This canonical export excludes Tree topology and supplies exactly the
        # numerical leaves consumed by the shared pure preparation protocol.
        for buffer_name in operator._prepared_buffer_names():
            slots.append((f"{operator_name}.{buffer_name}", operator, buffer_name))
    for noise_name in process.runtime_noises:
        buffer_name = f"_material_held_noise_{noise_name}"
        if buffer_name in process._buffers:
            slots.append((f"noise.{noise_name}", process, buffer_name))
    return tuple(slots)
