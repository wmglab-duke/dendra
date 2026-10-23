"""Registered workspace ownership for stochastic material operators.

Diffusion samples its runtime noise with the other material processes, but its
native numerical operator is configured by ``set_dt``.  The retained values
therefore name the actual native operator buffers, rather than a second private
checkpoint representation.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.utils import _pytree

from dendra.models.mechanisms import DiffusionProcess

from ._types import FunctionalizationError


def stochastic_diffusion_timestep_matches(process, dt) -> bool:
    """Compare a native configured timestep at the cold lowering boundary."""
    if not process._spatial_configured:
        return False
    operator = process._spatial_operators[process._diffusion_material_plan[0][0]]
    configured_dt = operator._buffers["dt"]
    return torch.equal(configured_dt, torch.full_like(configured_dt, dt))


@contextmanager
def retained_stochastic_diffusion_noise(population, source, dt):
    """Configure an isolated clone with the source's retained realization."""
    originals = []
    for name, process in population.mech.material_processes.items():
        native = source.mech.material_processes[name]
        if (
            not isinstance(native, DiffusionProcess)
            or not native.runtime_noises
            or not stochastic_diffusion_timestep_matches(native, dt)
        ):
            continue
        for noise_name in native.runtime_noises:
            originals.append((process, noise_name, process._buffers[noise_name]))
            process._buffers[noise_name] = (
                native._buffers[f"_material_held_noise_{noise_name}"].detach().clone()
            )
    try:
        yield
    finally:
        for process, name, value in originals:
            process._buffers[name] = value


def stochastic_diffusion_workspace_bindings(population) -> dict[str, str]:
    """Map pure prepared slots to the corresponding native diffusion buffers.

    This is a cold layout operation.  Both native and functional preparation
    export the same ``prepared_buffers`` PyTree from each already configured
    spatial operator.  Its leaf paths provide the shared storage/checkpoint
    names, including Boolean workspaces and the topology-free Tree export.
    """
    result = {}
    for process_name, process in population.mech.material_processes.items():
        if not isinstance(process, DiffusionProcess) or not process.runtime_noises:
            continue
        prefix = f"integrator.mech.material_processes.{process_name}"
        entries = tuple(process._diffusion_material_plan)
        workspaces = tuple(
            process._spatial_operators[entry[0]].prepared_buffers() for entry in entries
        )
        leaves, tree_spec = _pytree.tree_flatten_with_path(workspaces)
        slots = getattr(process, "_functional_material_slots", None)
        if slots is None:
            slots = tuple(
                f"_functional_material_prepared_{index}" for index in range(len(leaves))
            )
        elif len(slots) != len(leaves) or (
            process._functional_material_tree_spec != tree_spec
        ):
            raise FunctionalizationError(
                "Stochastic diffusion preparation does not match its native "
                "operator workspace layout."
            )
        for slot, (leaf_path, value) in zip(slots, leaves, strict=True):
            if (
                len(leaf_path) != 2
                or not isinstance(leaf_path[0], _pytree.SequenceKey)
                or not isinstance(leaf_path[1], _pytree.MappingKey)
                or not torch.is_tensor(value)
            ):
                raise FunctionalizationError(
                    "Stochastic diffusion requires the canonical tuple of "
                    "registered operator-buffer dictionaries."
                )
            entry_index, buffer_name = leaf_path[0].idx, leaf_path[1].key
            operator_name = entries[entry_index][0]
            operator = process._spatial_operators[operator_name]
            native = operator._buffers.get(buffer_name)
            if (
                native is None
                or native.shape != value.shape
                or native.dtype != value.dtype
                or native.device != value.device
            ):
                raise FunctionalizationError(
                    "Stochastic diffusion preparation must retain each native "
                    "registered workspace buffer's tensor schema."
                )
            if slot in process._buffers:
                prepared = process._buffers[slot]
                if (
                    prepared.shape != native.shape
                    or prepared.dtype != native.dtype
                    or prepared.device != native.device
                ):
                    raise FunctionalizationError(
                        "Stochastic diffusion prepared slots have incompatible "
                        "native operator-buffer schemas."
                    )
            result[f"{prefix}.{slot}"] = (
                f"{prefix}._spatial_operators.{operator_name}.{buffer_name}"
            )
        for noise_name in process.runtime_noises:
            name = f"_material_held_noise_{noise_name}"
            if name not in process._buffers:
                raise FunctionalizationError(
                    "Stochastic diffusion has no retained timestep NOISE "
                    "provenance; configure its timestep before lowering."
                )
            result[f"{prefix}.{name}"] = f"{prefix}.{name}"
    return result


def bind_stochastic_diffusion_workspaces(population, bindings) -> None:
    """Bind native checkpoint names to a private transition's prepared slots."""
    for process_name, process in population.mech.material_processes.items():
        prefix = f"integrator.mech.material_processes.{process_name}."
        slots = tuple(
            (
                "noise." + native.removeprefix(f"{prefix}_material_held_noise_")
                if native.startswith(f"{prefix}_material_held_noise_")
                else native.removeprefix(f"{prefix}_spatial_operators."),
                prepared.removeprefix(prefix),
            )
            for prepared, native in bindings.items()
            if prepared.startswith(prefix)
        )
        if slots:
            process._functional_material_workspace_slots = slots
