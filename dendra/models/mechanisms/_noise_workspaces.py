"""Shared ownership of NOISE frames retained by declared workspaces."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager

import torch


def _workspace_noise_bindings(module):
    """Describe reachable NOISE without inspecting an authored builder."""
    bindings = []
    for path, owner in module.named_modules():
        for name in getattr(owner, "runtime_noises", {}):
            bindings.append((f"{path}.{name}" if path else name, owner, name))
    return tuple(bindings)


def noise_workspace_phases(module):
    if not _workspace_noise_bindings(module):
        return ()
    return tuple(
        phase
        for phase, declaration in (
            ("derived", "_derived_buffers"),
            ("timestep", "_timestep_buffers"),
        )
        if getattr(module, declaration, ())
    )


def ensure_noise_workspace_slots(module):
    """Install fixed provenance slots with persistent initialization frames."""
    bindings = _workspace_noise_bindings(module)
    for phase in noise_workspace_phases(module):
        active_name = f"_noise_workspace_{phase}_active"
        if active_name not in module._buffers:
            module.register_buffer(
                active_name,
                torch.zeros((), device=module.diam.device, dtype=torch.bool),
                persistent=phase == "derived",
            )
        for index, (_path, owner, name) in enumerate(bindings):
            buffer_name = f"_noise_workspace_{phase}_noise_{index}"
            if buffer_name not in module._buffers:
                module.register_buffer(
                    buffer_name,
                    owner._buffers[name].detach().clone(),
                    persistent=phase == "derived",
                )
        if (
            phase == "timestep"
            and "_noise_workspace_timestep_dt" not in module._buffers
        ):
            module.register_buffer(
                "_noise_workspace_timestep_dt",
                module.diam.new_zeros(()),
                persistent=False,
            )
        if phase == "timestep" and not any(
            hook is _invalidate_noise_workspace_after_load
            for hook in module._load_state_dict_post_hooks.values()
        ):
            module.register_load_state_dict_post_hook(
                _invalidate_noise_workspace_after_load
            )


def complete_legacy_noise_workspace_state(module, state_dict, prefix=""):
    """Fill an entirely absent legacy initialization-frame namespace.

    Ordinary state dictionaries persist the initialization NOISE frame with
    derived numerical buffers. Older dictionaries have neither frame field;
    retain the destination's existing frame in that case. Partial or unknown
    frame namespaces remain untouched so strict loading rejects them.
    """
    if "derived" not in noise_workspace_phases(module):
        return state_dict
    namespace = f"{prefix}_noise_workspace_derived_"
    if any(isinstance(key, str) and key.startswith(namespace) for key in state_dict):
        return state_dict
    completed = OrderedDict(state_dict)
    if hasattr(state_dict, "_metadata"):
        completed._metadata = state_dict._metadata
    for name, value in module._buffers.items():
        if name.startswith("_noise_workspace_derived_"):
            completed[f"{prefix}{name}"] = value.detach().clone()
    return completed


def invalidate_noise_workspace_timestep(module):
    """Mark configuration pending after ordinary state-dict restoration."""
    module.__dict__.pop("_noise_workspace_restored_dt", None)
    active_name = "_noise_workspace_timestep_active"
    if active_name in module._buffers:
        module._buffers[active_name] = module._buffers[active_name].new_zeros(())


def _invalidate_noise_workspace_after_load(module, incompatible_keys):
    del incompatible_keys
    invalidate_noise_workspace_timestep(module)


def noise_workspace_noise_bindings(module, phase):
    """Return explicit live-to-held NOISE bindings for one builder phase."""
    if phase not in noise_workspace_phases(module):
        return ()
    return tuple(
        (path, owner, name, f"_noise_workspace_{phase}_noise_{index}")
        for index, (path, owner, name) in enumerate(_workspace_noise_bindings(module))
    )


def noise_workspace_noise_values(module, phase):
    return {
        path: module._buffers[held_name]
        for path, _owner, _name, held_name in noise_workspace_noise_bindings(
            module, phase
        )
    }


def current_noise_workspace_values(module, phase):
    """Expose the current NOISE frame as explicit builder inputs."""
    return {
        path: noise_owner._buffers[noise_name]
        for path, noise_owner, noise_name, _held_name in noise_workspace_noise_bindings(
            module, phase
        )
    }


@contextmanager
def temporary_noise_workspace(module, phase, values=None):
    """Evaluate a private builder with an explicit, detached NOISE frame."""
    values = noise_workspace_noise_values(module, phase) if values is None else values
    originals = []
    try:
        for path, owner, name, _held_name in noise_workspace_noise_bindings(
            module, phase
        ):
            originals.append((owner, name, owner._buffers[name]))
            owner._buffers[name] = values[path].detach().clone()
        yield
    finally:
        for owner, name, value in reversed(originals):
            owner._buffers[name] = value


def _noise_workspace_metadata(module, phase, *, dt=None):
    values = {
        f"_noise_workspace_{phase}_active": module._buffers[
            f"_noise_workspace_{phase}_active"
        ].new_ones(())
    }
    for _path, owner, name, held_name in noise_workspace_noise_bindings(module, phase):
        reference = module._buffers[held_name]
        # Functional local frames pad GLOBAL/BATCH values to the support rank.
        # Provenance keeps its declaration-owned shape; reshape retains hidden
        # vmap lanes while removing only that visible singleton padding.
        values[held_name] = (
            owner._buffers[name].detach().reshape(reference.shape).clone()
        )
    if phase == "timestep":
        reference = module._buffers["_noise_workspace_timestep_dt"]
        values["_noise_workspace_timestep_dt"] = (
            torch.as_tensor(dt).to(reference).detach().clone()
        )
    return values


def capture_noise_workspace(module, phase, *, dt=None):
    """Publish provenance after a complete successful builder transaction."""
    if phase not in noise_workspace_phases(module):
        return
    module._buffers.update(_noise_workspace_metadata(module, phase, dt=dt))


def restored_noise_workspace_values(module, dt):
    """Keep a restored realization through its first same-dt solver setup."""
    restored = module.__dict__.get("_noise_workspace_restored_dt")
    if restored is None or float(dt) != restored:
        return None
    return {name: module._buffers[name] for name in module._timestep_buffers}


def commit_noise_workspace(module, *, dt=None):
    restored = module.__dict__.get("_noise_workspace_restored_dt")
    if dt is not None and (restored is None or float(dt) != restored):
        capture_noise_workspace(module, "timestep", dt=dt)
    module.__dict__.pop("_noise_workspace_restored_dt", None)


def reset_noise_workspace_timestep(module):
    module.__dict__.pop("_noise_workspace_restored_dt", None)
    if "timestep" not in noise_workspace_phases(module):
        return
    module._buffers["_noise_workspace_timestep_active"] = module._buffers[
        "_noise_workspace_timestep_active"
    ].new_zeros(())
    module._buffers["_noise_workspace_timestep_dt"] = module._buffers[
        "_noise_workspace_timestep_dt"
    ].new_zeros(())


def retain_noise_workspace_once(module):
    if "timestep" not in noise_workspace_phases(module):
        return
    if bool(module._buffers["_noise_workspace_timestep_active"].item()):
        module._noise_workspace_restored_dt = float(
            module._buffers["_noise_workspace_timestep_dt"].item()
        )


def noise_workspace_slots(module):
    """Yield canonical numerical/provenance checkpoint fields."""
    slots = []
    for phase in noise_workspace_phases(module):
        names = getattr(module, f"_{phase}_buffers")
        slots.extend((f"{phase}.buffer.{name}", module, name) for name in sorted(names))
        slots.append((f"{phase}.active", module, f"_noise_workspace_{phase}_active"))
        slots.extend(
            (f"{phase}.noise.{path}", module, held_name)
            for path, _owner, _name, held_name in noise_workspace_noise_bindings(
                module, phase
            )
        )
        if phase == "timestep":
            slots.append(("timestep.dt", module, "_noise_workspace_timestep_dt"))
    return tuple(slots)
