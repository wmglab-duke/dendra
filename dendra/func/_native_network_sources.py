"""Live native Network source slots for detached continuation handoff.

These projections retain the public packet paths without materializing values
or copying tensors. Synthetic metadata and effective-only connection storage
have no writable raw source correspondence.
"""

from __future__ import annotations

import torch

from ._connection_parameters import _ROLES
from ._continuous_transform import _named_tensors
from ._native_handoff_sources import population_source_frames
from ._schedules import TensorScheduleTensors, _TensorSchedulePlan


def _raw_source_frames(plan, target):
    """Project registered raw paths while omitting synthetic scalar slots."""
    parameters = plan.parameter_sources(target)
    constants = plan.constant_sources(target)
    return (
        {
            key: parameters[key]
            for key, binding in plan._raw_sources.parameters.items()
            if binding.path is not None
        },
        {
            key: constants[key]
            for key, binding in plan._raw_sources.constants.items()
            if binding.path is not None
        },
    )


def _schedule_sources(plan, target):
    if not plan._schedule.active:
        return None
    return TensorScheduleTensors(**_TensorSchedulePlan._values(target))


def network_source_frames(plan, target):
    """Return parameter and partial constant trees of writable native tensors.

    Raw source inventories retain their existing classification and encounter
    order. This excludes connection materialization metadata, effective-only
    samples, scalar NetStim attributes, and external extracellular inputs.
    The caller resolves parameter-owned constant readers before using these
    paths to construct a handoff transaction.
    """
    parameters = {
        "populations": {},
        "synapses": {},
        "continuous_synapses": {},
        "netstim": None,
    }
    constants = {
        "populations": {},
        "synapses": {},
        "continuous_synapses": {},
        "netstim": None,
    }
    for name, population_plan in plan.populations.items():
        population_parameters, population_constants = population_source_frames(
            population_plan, target.populations[name]
        )
        parameters["populations"][name] = population_parameters
        constants["populations"][name] = population_constants
    for (collection, name), source_plan in plan._connection_parameter_plans.items():
        connection = getattr(target, collection)[name]
        source_parameters, source_constants = _raw_source_frames(
            source_plan, connection
        )
        effective_only = {
            f"{native}.w" for role, native in _ROLES if not source_plan._raw[role]
        }
        source_parameters = {
            key: value
            for key, value in source_parameters.items()
            if key not in effective_only
        }
        if collection == "synapses":
            source_parameters["schedule"] = _schedule_sources(
                plan.synapses[name], connection
            )
        else:
            source_parameters["transform_parameters"] = (
                _named_tensors(connection.transform)
                if isinstance(connection.transform, torch.nn.Module)
                else None
            )
        parameters[collection][name] = source_parameters
        constants[collection][name] = source_constants
    if plan.netstim is not None:
        source = target.netstim
        source_parameters, source_constants = _raw_source_frames(
            plan._netstim_parameter_plan, source
        )
        parameters["netstim"] = {
            "parameters": source_parameters,
            "schedule": _schedule_sources(plan.netstim, source),
        }
        constants["netstim"] = {
            "max_spikes": source.max_spikes,
            "sources": source_constants,
        }
    return parameters, constants


def scalar_source_writes(plan, target, parameters, constants=None):
    """Stage scalar NetStim attributes absent from registered Tensor sources."""
    if plan.netstim is None:
        return []
    source = target.netstim
    inventory = plan._netstim_parameter_plan._raw_sources
    result = []
    for bindings, values in (
        (inventory.parameters, parameters["netstim"]["parameters"]),
        (
            inventory.constants,
            None if constants is None else constants["netstim"]["sources"],
        ),
    ):
        for key, binding in bindings.items():
            if binding.path is not None:
                continue
            # NetStim's raw inventory represents a Python-valued tau as an
            # explicit scalar Tensor. Preserve its native attribute type.
            value = values[key].detach().item()
            result.append((source, key, type(getattr(source, key))(value)))
    return result
