"""Cold writable-source correspondence for native Population handoff.

The existing plan inventories supply public packet keys. Reading registered
slots keeps authored parameter getters and workspace builders out of source
discovery; synthetic and retained execution operands are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._module_traversal import _raw_module_namespace
from ._raw_source_discovery import raw_registered_parameters
from ._types import FunctionalizationError


@dataclass(frozen=True)
class _NativeSourceGroup:
    """Several packed native slots represented by one explicit constant key."""

    tensors: tuple


def _registered_owner(root, path):
    owner = root
    for name in path.split(".") if path else ():
        owner = _raw_module_namespace(owner)["_modules"].get(name)
        if owner is None:
            raise FunctionalizationError(
                f"Native handoff lost registered module {path!r}"
            )
    return owner


def _registered_tensor(root, path, *, registry=None):
    module_path, _separator, name = path.rpartition(".")
    namespace = _raw_module_namespace(_registered_owner(root, module_path))
    registries = (registry,) if registry is not None else ("_parameters", "_buffers")
    for collection in registries:
        value = namespace[collection].get(name)
        if torch.is_tensor(value):
            return value
    raise FunctionalizationError(
        f"Native handoff lost registered Tensor source {path!r}"
    )


def _canonical_parameters(root):
    """Mirror canonical named_parameters without invoking subclass readers."""
    return raw_registered_parameters(root, deduplicate=True)


def _record_source(result, key, value):
    previous = result.setdefault(key, value)
    if previous is not value:
        raise FunctionalizationError(
            f"Native handoff source aliases for {key!r} do not match the plan"
        )


def _geometry_source(target, name):
    if name in {"canonical_area_cm2", "canonical_edge_resistance_ohm"}:
        return None  # canonical morphology geometry is immutable provenance
    if name == "diam_original":
        return _registered_tensor(target, "parametrizations.diam.original")
    namespace = _raw_module_namespace(target)
    for registry in ("_parameters", "_buffers"):
        value = namespace[registry].get(name)
        if torch.is_tensor(value):
            return value
    # A property-only effective geometry value is not a writable raw source.
    return None


def _constant_sources(target, geometry_names, entries):
    result = {}
    for name in geometry_names:
        value = _geometry_source(target, name)
        if value is not None:
            _record_source(result, name, value)
    for entry in entries:
        if entry.key in {"canonical_area_cm2", "canonical_edge_resistance_ohm"}:
            continue
        path = (
            f"{entry.module_path + '.' if entry.module_path else ''}{entry.buffer_name}"
        )
        value = _registered_tensor(target, path, registry="_buffers")
        for alias in entry.aliases:
            if _registered_tensor(target, alias, registry="_buffers") is not value:
                raise FunctionalizationError(
                    f"Native handoff constant aliases for {entry.key!r} "
                    "do not match the plan"
                )
        _record_source(result, entry.key, value)
    return result


def _intra_sources(plan, target):
    from dendra.models.stim.waveform.core import _randomized_poisson

    from ._stimuli import _solver_injection_specs

    bank = plan._intra._waveforms
    sources = tuple(
        waveform for waveform, _shape, _index in _solver_injection_specs(target)
    )
    if len(sources) != len(bank._sources):
        raise FunctionalizationError(
            "Native handoff intracellular Waveform layout does not match the plan"
        )
    result = {}
    for leaves, registry in (
        (bank._parameter_leaves, "_parameters"),
        (bank._constant_leaves, "_buffers"),
    ):
        for leaf in leaves:
            waveform = sources[leaf.waveform_index]
            path, _separator, name = leaf.source_name.rpartition(".")
            owner = _registered_owner(waveform, path)
            if name == "_spike_times" and type(owner) is _randomized_poisson:
                continue  # runtime history is restored by the Poisson state adapter
            value = _registered_tensor(waveform, leaf.source_name, registry=registry)
            _record_source(result, leaf.public_name, value)
    return result


def population_source_frames(plan, target):
    """Read ordered writable packet sources from an already compatible target.

    Return native Tensor objects without cloning, casting, rebinding or calling
    authored getters. Parameter names follow the plan's canonical registered
    inventory, including registered mechanism injections and initial sources.
    Plan-bound external extra and randomized waveform history are excluded.
    Constants contain only raw geometry and recorded registered dependencies;
    timestep, canonical geometry, RAND/NOISE frames and retained/cache operands
    remain the orchestration boundary's equality/rejection responsibility.
    """
    registered = _canonical_parameters(target)
    parameters = {}
    for name in plan._population_parameter_names:
        if name not in registered:
            raise FunctionalizationError(
                f"Native handoff lost canonical Parameter source {name!r}"
            )
        parameters[name] = registered[name]
    parameters.update(_intra_sources(plan, target))

    constants = {}
    records = plan._multi_component_plans
    if records:
        populations = _registered_owner(target, "populations")
        for record in records:
            component = _registered_owner(populations, record.name)
            local = _constant_sources(
                component,
                record.geometry_constant_names,
                record.parametrization_constants,
            )
            for relative, public in record.constant_sources:
                if relative in local:
                    value = local[relative]
                    previous = constants.get(public)
                    if previous is None or previous is value:
                        constants[public] = value
                    else:
                        values = (
                            previous.tensors
                            if isinstance(previous, _NativeSourceGroup)
                            else (previous,)
                        )
                        if all(item is not value for item in values):
                            constants[public] = _NativeSourceGroup((*values, value))
        local = _constant_sources(target, (), plan._parametrization_constants)
    else:
        local = _constant_sources(
            target, plan._geometry_constant_names, plan._parametrization_constants
        )
    for name, value in local.items():
        _record_source(constants, name, value)
    return parameters, constants


__all__ = []
