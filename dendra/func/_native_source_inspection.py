"""Cold native source frames for authored Populations and Networks.

Native registration paths remain visible under every alias. Configuration
selection uses declared source scopes rather than runtime Tensor discovery;
no model build, lowering, initialization or numerical getter is involved.
"""

from __future__ import annotations

import torch

from dendra.models.core import Population
from dendra.models.networks.net import WeightExpander
from dendra.models.networks.netcon import ContinuousCon, NetCon
from dendra.models.networks.netstim import NetStim

from ._module_traversal import _raw_module_namespace, _registered_modules
from ._raw_source_discovery import (
    raw_configuration_sources,
    raw_registered_parameters,
)


def _join(prefix, name):
    return f"{prefix}.{name}" if prefix else name


def _merge(result, prefix, sources):
    for name, value in sources.items():
        path = _join(prefix, name)
        previous = result.setdefault(path, value)
        if previous is not value:
            raise ValueError(f"Native source paths collide at {path!r}")


def _direct_constants(owner, names):
    """Read registered or ordinary source slots without invoking descriptors."""
    namespace = _raw_module_namespace(owner)
    parameters, buffers = namespace["_parameters"], namespace["_buffers"]
    result = {}
    for name in names:
        if name in parameters:
            continue  # this exact registration already belongs to parameters
        value = buffers[name] if name in buffers else namespace.get(name)
        if issubclass(type(value), torch.Tensor):
            result[name] = value
    return result


def _connection_constants(owner):
    """Select raw wrapper sources, excluding each expander's effective sample."""
    namespace = _raw_module_namespace(owner)
    result = {}
    for name in ("weight", "delay_ms"):
        expander = namespace["_modules"].get(name)
        if not issubclass(type(expander), WeightExpander):
            continue
        bank = _raw_module_namespace(expander)["_modules"].get("weights")
        if issubclass(type(bank), torch.nn.Module):
            _merge(
                result,
                f"{name}.weights",
                raw_configuration_sources(bank, source_roots=(bank,)),
            )
    return result


def _netstim_constants(owner):
    namespace = _raw_module_namespace(owner)
    result = _direct_constants(
        owner, ("start", "noise", "tau", "max_spikes", "heap_times")
    )
    for name in ("interval", "parametrizations"):
        source = namespace["_modules"].get(name)
        if issubclass(type(source), torch.nn.Module):
            _merge(
                result,
                name,
                raw_configuration_sources(source, source_roots=(source,)),
            )
    return result


def _owned_injection_sources(owner, prefix, parameters, constants):
    """Expose Waveforms retained by the two native injection declaration lists."""
    namespace = _raw_module_namespace(owner)
    for name in ("injections", "mechanism_injections"):
        declarations = namespace.get(name)
        if type(declarations) not in (list, tuple):
            continue
        for index, declaration in enumerate(declarations):
            if type(declaration) not in (list, tuple) or not declaration:
                continue
            waveform = declaration[0]
            if not issubclass(type(waveform), torch.nn.Module):
                continue
            route = _join(prefix, f"{name}.{index}.waveform")
            _merge(
                parameters,
                route,
                raw_registered_parameters(waveform, deduplicate=False),
            )
            _merge(constants, route, raw_configuration_sources(waveform))


def native_source_frames(subject):
    """Return flat native parameter and raw-configuration Tensor dictionaries.

    Every registered Parameter alias retains its native registration spelling.
    Population-owned injection declarations add ``injections.<index>.waveform``
    and ``mechanism_injections.<index>.waveform`` routes, including
    aliases to already registered Waveforms. Constant sources exclude effective
    connection samples, solver workspaces, runtime state and RNG data.

    The result reflects objects already owned at inspection time. Pending
    mechanism or connection declarations are not built or materialized. Native
    construction, built models and initialized models use the same readers;
    inspection neither validates execution eligibility nor evaluates sources.
    """
    parameters = raw_registered_parameters(subject, deduplicate=False)
    constants = raw_configuration_sources(subject)
    for prefix, owner in _registered_modules(
        subject, duplicates=True, namespace=_raw_module_namespace
    ):
        if issubclass(type(owner), Population):
            _owned_injection_sources(owner, prefix, parameters, constants)
        if issubclass(type(owner), (NetCon, ContinuousCon)):
            _merge(constants, prefix, _connection_constants(owner))
        if issubclass(type(owner), NetStim):
            _merge(constants, prefix, _netstim_constants(owner))
        if issubclass(type(owner), (NetCon, NetStim)):
            _merge(
                constants,
                prefix,
                _direct_constants(
                    owner,
                    (
                        "sched_time_ms",
                        "sched_weight",
                        "_sched_t_source",
                        "_sched_w_source",
                    ),
                ),
            )
        if issubclass(type(owner), ContinuousCon):
            transform = _raw_module_namespace(owner)["_modules"].get("transform")
            if issubclass(type(transform), torch.nn.Module):
                _merge(
                    constants,
                    _join(prefix, "transform"),
                    raw_configuration_sources(transform, source_roots=(transform,)),
                )
    return parameters, constants


__all__ = []
