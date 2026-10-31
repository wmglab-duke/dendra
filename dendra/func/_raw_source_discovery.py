"""Registered raw sources, without materialization or execution qualification."""

from __future__ import annotations

from ._module_traversal import _raw_module_namespace, _registered_modules

_PARAMETER_DECLARATIONS = (
    "_global",
    "_range",
    "_batch",
    "_global_p",
    "_range_p",
    "_batch_p",
    "_global_n",
    "_range_n",
    "_batch_n",
)
_SIMPLE_DECLARATIONS = ("_params", "_params_p", "_params_n")
_RUNTIME_DECLARATIONS = (
    "_state",
    "_all_states",
    "_carry",
    "_assigned",
    "_inputs",
    "_derived_buffers",
    "_timestep_buffers",
    "_material_fields",
    "_functional_material_slots",
    "random_parameters",
    "runtime_noises",
)
_POPULATION_GEOMETRY = (
    "diam",
    "diameters",
    "dx",
    "xraxial",
    "xc",
    "xg",
)
_POISSON_INVOCATION_BUFFERS = frozenset(
    ("_poisson_invocation_stack", "_poisson_invocation_position")
)
CUSTOM_MATERIAL_CONTEXT_NAMES = (
    "celsius",
    "diam",
    "diameters",
    "diam_original",
    "dx",
    "area",
    "canonical_area_cm2",
    "component_area",
    "dt",
)


def registered_source_slots(
    modules,
    *,
    collections=("_parameters", "_buffers"),
    namespace=_raw_module_namespace,
):
    """Read supplied modules in module-major, then registry encounter order.

    Traversal, qualification, tensor admission and duplicate policy belong to
    callers. Only ``None`` registrations are omitted; malformed values remain
    visible to each consumer's existing validation boundary.
    """
    for path, module in modules:
        for registry in collections:
            for name, value in namespace(module)[registry].items():
                if value is not None:
                    yield f"{path + '.' if path else ''}{name}", value, registry


def raw_registered_parameters(root, *, deduplicate=False):
    """Return live registered parameters by native path, retaining aliases."""
    modules = _registered_modules(
        root, duplicates=not deduplicate, namespace=_raw_module_namespace
    )
    result, seen = {}, set()
    for path, value, _registry in registered_source_slots(
        modules, collections=("_parameters",)
    ):
        if not deduplicate or id(value) not in seen:
            result[path] = value
            seen.add(id(value))
    return result


def _metadata(module, name, default=()):
    """Read declaration storage without executing instance or class getters."""
    namespace = _raw_module_namespace(module)
    if name in namespace:
        return namespace[name]
    for cls in type.__getattribute__(type(module), "__mro__"):
        namespace = type.__getattribute__(cls, "__dict__")
        if name in namespace:
            return namespace[name]
    return default


def _names(value):
    if type(value) in (dict, list, tuple, set, frozenset):
        return tuple(name for name in value if issubclass(type(name), str))
    return ()


def raw_configuration_sources(root, *, source_roots=()):
    """Read registered configuration selected by native declaration metadata.

    Source-module buffers belong to declared raw parameters, initializers,
    Waveforms, TABLEs and parameter transforms. Direct configuration slots come
    from geometry, fixed mechanism delays and injection specifications. Evolving
    state and reconstructed workspace slots are excluded without hiding raw
    source aliases to their tensors. Arbitrary sibling buffers have no source
    declaration and are omitted. ``source_roots`` lets a caller mark known raw
    source subtrees while retaining these runtime exclusions.

    Results retain native registration paths, aliases and live references. This
    discovery neither constructs a functional plan nor invokes authored readers,
    numerical operations, lifecycle hooks or random-number generators.
    """
    from dendra.models.core import Population
    from dendra.models.integrators.core import Integrator
    from dendra.models.mechanisms._handler import MechanismHandler
    from dendra.models.mechanisms._ions import Ion
    from dendra.models.mechanisms._material_process import (
        ClampProcess,
        ClearanceProcess,
        DiffusionProcess,
        ExchangeProcess,
        MaterialProcess,
    )
    from dendra.models.mechanisms._materials import Material
    from dendra.models.multi import MultiPopulation
    from dendra.models.rng import RNGMixin
    from dendra.models.stim.waveform.core import Waveform, _randomized_poisson
    from dendra.utils import PreparedInterp1d

    modules = tuple(
        _registered_modules(root, duplicates=True, namespace=_raw_module_namespace)
    )
    source_root_ids = {id(module) for module in source_roots}
    selected_slots, runtime_slots = set(), set()

    def source_subtree(module):
        source_root_ids.add(id(module))

    def source_slot(module, name):
        namespace = _raw_module_namespace(module)
        selected_slots.add((id(module), name))
        child = namespace["_modules"].get(name)
        if child is not None:
            source_subtree(child)

    seen = set()
    for _path, module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        namespace = _raw_module_namespace(module)
        buffers, children = namespace["_buffers"], namespace["_modules"]
        runtime_names = set()
        for declaration in _RUNTIME_DECLARATIONS:
            runtime_names.update(_names(_metadata(module, declaration)))
        runtime_names.update(f"{name}_" for name in _names(_metadata(module, "_save")))
        for declaration in _PARAMETER_DECLARATIONS:
            for name in _names(_metadata(module, declaration)):
                source_slot(module, f"{name}_param")
                runtime_names.add(name)
        for declaration in _SIMPLE_DECLARATIONS:
            for name in _names(_metadata(module, declaration)):
                source_slot(module, name)
        additional = namespace.get("additional_parameters", {})
        if type(additional) is dict:
            for entries in additional.values():
                if type(entries) not in (tuple, list):
                    continue
                for entry in entries:
                    if type(entry) in (tuple, list) and len(entry) == 2:
                        if issubclass(type(entry[1]), str):
                            source_slot(module, entry[1])
        if issubclass(type(module), MultiPopulation):
            runtime_names.update(_POPULATION_GEOMETRY)
        elif issubclass(type(module), Population):
            for name in _POPULATION_GEOMETRY:
                if name not in runtime_names:
                    source_slot(module, name)
        if issubclass(type(module), Waveform):
            source_subtree(module)
        if issubclass(type(module), Ion):
            for name in ("e_init", "i_init", "o_init"):
                source_slot(module, name)
        if issubclass(type(module), Material):
            source_slot(module, "_initial_sources")
            for name in _names(_metadata(module, "_material_min_fields")):
                source_slot(module, f"_min_{name}")
        if issubclass(type(module), MaterialProcess) and not issubclass(
            type(module),
            (ClampProcess, ClearanceProcess, DiffusionProcess, ExchangeProcess),
        ):
            source_subtree(module)
            runtime_names.update(CUSTOM_MATERIAL_CONTEXT_NAMES)
            runtime_names.update(_names(additional))
            registry = children.get("parametrizations")
            if registry is not None:
                runtime_names.update(_raw_module_namespace(registry)["_modules"])
            runtime_names.update(name for name in buffers if name.startswith("_mp_"))
        if "parametrizations" in children:
            source_subtree(children["parametrizations"])
        transforms = namespace.get("in_graph_parametrizations", {})
        if type(transforms) is dict:
            for entries in transforms.values():
                if type(entries) not in (tuple, list):
                    continue
                for entry in entries:
                    if type(entry) not in (tuple, list) or len(entry) != 2:
                        continue
                    transform, arguments = entry
                    if any(child is transform for child in children.values()):
                        source_subtree(transform)
                    for name in _names(arguments):
                        if name not in runtime_names:
                            source_slot(module, name)
        if issubclass(type(module), PreparedInterp1d):
            # The retained sample axis and ordinates describe a TABLE; slopes,
            # spacing and endpoint views are reconstructed interpolation data.
            for name in ("_x_search", "_y"):
                source_slot(module, name)
            runtime_names.update(
                name for name in buffers if name not in {"_x_search", "_y"}
            )
        injections = namespace.get("_injection_specs", ())
        if type(injections) in (tuple, list):
            for spec in injections:
                if type(spec) is dict:
                    for field in ("mask", "scale", "index"):
                        name = spec.get(field)
                        if issubclass(type(name), str):
                            source_slot(module, name)
        delays = namespace.get("_delayed_state_specs", {})
        if type(delays) is dict:
            for spec in delays.values():
                if type(spec) is not dict:
                    continue
                name = spec.get("steps_buffer")
                if issubclass(type(name), str):
                    source_slot(module, name)
                runtime_names.update(
                    spec[field]
                    for field in ("buffer", "pointer")
                    if issubclass(type(spec.get(field)), str)
                )
        if issubclass(type(module), _randomized_poisson):
            runtime_names.add("_spike_times")
        if issubclass(type(module), Waveform):
            runtime_names.update(_POISSON_INVOCATION_BUFFERS)
        if issubclass(
            type(module), (Population, Integrator, MechanismHandler, RNGMixin)
        ):
            runtime_names.update(
                name for name in buffers if (id(module), name) not in selected_slots
            )
        runtime_slots.update((id(module), name) for name in runtime_names)

    result, source_paths = {}, set()
    for path, module in modules:
        parent, _separator, _name = path.rpartition(".")
        if id(module) in source_root_ids or (path and parent in source_paths):
            source_paths.add(path)
        for name, value in _raw_module_namespace(module)["_buffers"].items():
            if (
                value is not None
                and (id(module), name) not in runtime_slots
                and (path in source_paths or (id(module), name) in selected_slots)
            ):
                result[f"{path + '.' if path else ''}{name}"] = value
    return result


__all__ = []
