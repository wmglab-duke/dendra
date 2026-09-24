"""Experimental, fail-closed functional Population lowering."""

from __future__ import annotations

import builtins
import copy
import dis
import hashlib
import inspect
import math
import random
import types
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from functools import cache

import numpy as np
import torch
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

from dendra.models.core import (
    Cable,
    Myelinated,
    Population,
    SingleCompartment,
    Unmyelinated,
    _PassiveEndDiameter,
)
from dendra.models.extcell import ExtCellAxon, ExtCellTree
from dendra.models.initialization import _InitializationTransformHook
from dendra.models.integrators.cable import (
    _canonical_edge_conductance,
    _cylindrical_edge_conductance,
    _cylindrical_membrane_area,
    _layered_edge_conductance,
)
from dendra.models.integrators.core import _FunctionalIntegratorSpec
from dendra.models.integrators.implicit import _bwd_euler_sc, _bwd_euler_ub
from dendra.models.integrators.tree_bt import _tree_layered_edge_conductance
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion
from dendra.models.mechanisms._materials import Material
from dendra.models.mechanisms._mechanism import Mechanism, PointProcess
from dendra.models.mechanisms._numerical import (
    validate_declared_numerical_current as _validate_declared_numerical_current,
)
from dendra.models.mechanisms._state import State
from dendra.models.mechanisms._support import SupportKind
from dendra.models.mechanisms._support_registry import SupportEntry
from dendra.models.mechanisms._symbolic import build_numerical_equation
from dendra.models.multi import MultiPopulation
from dendra.models.parametric import Functional as ParameterFunctional
from dendra.models.parametric import (
    Parameterized,
    _ReplicaRangeExpander,
    cacheable,
    staticproperty,
)
from dendra.models.tree import Tree
from dendra.utils.tensor_ops import _logical_tensor_bytes

from ._stimuli import (
    FunctionalExtra,
    FunctionalIntra,
    _is_internal_module_state,
    _numpy_rng_state_equal,
)
from ._types import (
    FunctionalizationError,
    InitializationInput,
    PopulationTensors,
    RolloutInput,
    StepInput,
)

_COMMON_POPULATION_PARAMETER_NAMES = {
    "celsius_param",
    "rhoa_scale_param.rho",
    "cm_scale_param.rho",
    "area_scale_param.rho",
}
_UNMYELINATED_POPULATION_PARAMETER_NAMES = {
    *_COMMON_POPULATION_PARAMETER_NAMES,
    "cm_param",
    "rhoa_param",
}
_SINGLE_COMPARTMENT_POPULATION_PARAMETER_NAMES = {
    *_COMMON_POPULATION_PARAMETER_NAMES,
    "cm_param.rho",
    "rhoa_param.rho",
}
_PARAMETER_CATEGORIES = (
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
_STRUCTURED_ROLLOUT_CHUNK = 4
_CLASS_STATE_METADATA_NAMES = frozenset(
    {
        "__annotations__",
        "__classcell__",
        "__dict__",
        "__doc__",
        "__module__",
        "__qualname__",
        "__weakref__",
    }
)
_RUNTIME_SCHEMA_CACHE_STATE = frozenset(
    {
        "_assigned_schema_validated",
        "_advance_schema_validated",
        "_advance_return_names",
    }
)
_FUNCTIONAL_INTEGRATOR_HOOKS = (
    "_functional_spec",
    "_functional_solver",
    "_voltage_update",
    "_prepare_workspace",
    "_prepared_workspace_schema",
    "_derive_prepared_workspace",
    "_install_prepared_workspace",
    "_step",
    "_select_solver",
    "initialize",
    "step",
)
_FUNCTIONAL_HANDLER_HOOKS = ("_evaluate_current_frame",)
_STANDARD_FUNCTIONAL_HANDLER_HOOKS = {
    name: inspect.getattr_static(MechanismHandler, name)
    for name in _FUNCTIONAL_HANDLER_HOOKS
}
_HANDLER_INITIALIZATION_HOOKS = (
    "initialize",
    "_sync_celsius",
    "make_maps",
    "init_rng",
    "populate",
    "_reset_initialization_timestep",
    "_initialize_tensor_transaction",
    "_reset_shared_initial_fields",
    "ion_init",
    "material_init",
    "set_buffers",
    "_bind_state_geometry",
    "_reset_shared_local_buffers",
    "init_i_g_bufs",
    "_initialize_state_plan",
    "_initialization_current_frame",
    "i",
    "capture_ion_current_frame",
    "_publish_ion_current_frame",
    "write_to_ions",
    "write_material_replacements",
    "_advance_shared_fields",
    "read_from_ions",
    "read_from_materials",
    "_evaluate_current_frame",
)
_STANDARD_HANDLER_INITIALIZATION = {
    name: inspect.getattr_static(MechanismHandler, name)
    for name in _HANDLER_INITIALIZATION_HOOKS
}
_HANDLER_SYNC_WRAPPER_HOOKS = frozenset(MechanismHandler._SYNC_METHODS)
_POPULATION_FRESH_INITIALIZATION_HOOKS = (
    "initialize",
    "build",
    "build_intra",
    "_restore_steady_state",
    "_clear_duration_remainder",
    "populate_parameter_buffers",
    "_refresh_parameter_views_for_initialization",
    "pre_initialize",
    "post_initialize",
)
_STANDARD_POPULATION_FRESH_INITIALIZATION = {
    population_type: {
        name: inspect.getattr_static(population_type, name)
        for name in _POPULATION_FRESH_INITIALIZATION_HOOKS
    }
    for population_type in (SingleCompartment, Unmyelinated)
}
_INTEGRATOR_FRESH_INITIALIZATION_HOOKS = ("init_v",)
_STANDARD_INTEGRATOR_FRESH_INITIALIZATION = {
    implementation: {
        name: inspect.getattr_static(implementation, name)
        for name in _INTEGRATOR_FRESH_INITIALIZATION_HOOKS
    }
    for implementation in (_bwd_euler_sc, _bwd_euler_ub)
}


def _functional_initialization_profile(population):
    """Return the audited scalar initializer family for ``population``.

    SingleCompartment lowering remains exact-type because general subclasses are
    not admitted by the transition contract. Unmyelinated subclasses may add
    mechanisms and construction-time parameterization while retaining the same
    scalar-path voltage and lifecycle semantics; their inherited hooks are
    checked against the canonical base separately.
    """

    if type(population) is SingleCompartment:
        return SingleCompartment, "scalar_point", _bwd_euler_sc
    if isinstance(population, Unmyelinated):
        return Unmyelinated, "scalar_path", _bwd_euler_ub
    return None


_STANDARD_MYELINATED_RHOA_FORWARD = inspect.getattr_static(
    Myelinated.myelinated_rhoa,
    "forward",
)
_STANDARD_MYELINATED_DIAMETER_FORWARD = inspect.getattr_static(
    Myelinated.myelinated_node_d,
    "forward",
)
_STANDARD_PASSIVE_END_DIAMETER_FORWARD = inspect.getattr_static(
    _PassiveEndDiameter,
    "forward",
)
_STANDARD_MECHANISM_INITIALIZATION = {
    name: inspect.getattr_static(Mechanism, name)
    for name in (
        "_init_buffers_s",
        "_initialize_declared_values",
        "_derive_initial_state_values",
        "_initial_value_frame",
        "_derive_initial_values",
        "_install_initial_values",
        "populate",
        "populate_parameter_buffers",
        "init_rng",
    )
}
_STANDARD_MECHANISM_INITIAL_VALUES = inspect.getattr_static(
    Mechanism,
    "initial_values",
)
_STANDARD_MECHANISM_ASSIGNED_VALUES = inspect.getattr_static(
    Mechanism,
    "assigned_values",
)
_STANDARD_MECHANISM_ADVANCE = inspect.getattr_static(Mechanism, "advance")
_STATE_IMPERATIVE_INITIALIZATION_HOOKS = (
    "populate_parameter_buffers",
    "init_rng",
)
_STATE_FRESH_INITIALIZATION_HOOKS = (
    "_derive_initial_state_values",
    "_derive_initial_values",
    *_STATE_IMPERATIVE_INITIALIZATION_HOOKS,
)
_STANDARD_STATE_FRESH_INITIALIZATION = {
    name: inspect.getattr_static(State, name)
    for name in _STATE_FRESH_INITIALIZATION_HOOKS
}
_STANDARD_STATE_DEFAULTS = inspect.getattr_static(State, "state_defaults")
_STANDARD_STATE_INITIAL_VALUES = inspect.getattr_static(State, "initial_values")
_STANDARD_STATE_ASSIGNED_VALUES = inspect.getattr_static(State, "assigned_values")
_STANDARD_STATE_ADVANCE = inspect.getattr_static(State, "advance")
_ION_INITIALIZATION_HOOKS = (
    "initialize",
    "advance",
    "detach",
    "_derive_initial_field_values",
    "_derive_advanced_field_updates",
    "_install_field_values",
    "_install_field_updates",
    "_expand_init_like",
    "_nernst",
)
_MATERIAL_INITIALIZATION_HOOKS = (
    "initialize",
    "advance",
    "detach",
    "initial_source",
    "_derive_initial_field_values",
    "_derive_advanced_field_updates",
    "_install_field_values",
    "_install_field_updates",
)
_STANDARD_ION_INITIALIZATION = {
    name: inspect.getattr_static(Ion, name) for name in _ION_INITIALIZATION_HOOKS
}
_STANDARD_MATERIAL_INITIALIZATION = {
    name: inspect.getattr_static(Material, name)
    for name in _MATERIAL_INITIALIZATION_HOOKS
}
_ION_MATERIAL_BASE_INITIALIZATION_HOOKS = ("_derive_advanced_field_updates",)
_STANDARD_ION_MATERIAL_BASE_INITIALIZATION = {
    name: inspect.getattr_static(Material, name)
    for name in _ION_MATERIAL_BASE_INITIALIZATION_HOOKS
}


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.preserve_format)


def _audit_tensor_digest(value: torch.Tensor) -> str:
    """Return a stable value digest for unregistered Python tensor state."""
    detached = value.detach()
    if detached.layout != torch.strided:
        detached = detached.to_dense()
    if detached.numel() == 0:
        # Empty expanded tensors can retain a zero final stride even after
        # ``contiguous()``; metadata is already represented by the surrounding
        # signature, and every empty logical value has the same byte payload.
        return hashlib.sha256(b"").hexdigest()
    payload = _logical_tensor_bytes(detached).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _audit_tensor_binding(value: torch.Tensor) -> tuple:
    """Capture identity, schema, version, and bitwise logical value."""
    try:
        version = value._version
    except RuntimeError:
        version = None
    return (
        id(value),
        version,
        _shape_tuple(value),
        tuple(int(item) for item in value.stride()),
        value.storage_offset(),
        value.dtype,
        value.device,
        value.requires_grad,
        _audit_tensor_digest(value),
    )


def _audit_tensor_value(value: torch.Tensor) -> tuple:
    """Capture exact logical output metadata/value without object identity."""
    return (
        _shape_tuple(value),
        tuple(int(item) for item in value.stride()),
        value.dtype,
        value.device,
        value.requires_grad,
        _audit_tensor_digest(value),
    )


def _audit_tensor_numerical_value(value: torch.Tensor) -> tuple:
    """Capture logical schema/value while ignoring transform-owned metadata."""

    return (
        _shape_tuple(value),
        value.dtype,
        value.device,
        _audit_tensor_digest(value),
    )


def _registered_tensor_bindings(module: torch.nn.Module) -> dict[tuple, tuple]:
    """Snapshot every registered parameter/buffer slot, including aliases."""
    bindings = {}
    for module_path, owner in module.named_modules(remove_duplicate=False):
        for collection_name in ("_parameters", "_buffers"):
            for name, value in getattr(owner, collection_name).items():
                if value is None:
                    continue
                key = (module_path, collection_name, name)
                bindings[key] = _audit_tensor_binding(value)
    return bindings


def _audit_python_value_signature(value, seen: set[int]):
    """Fingerprint transition Python state without rejecting framework objects."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, complex)):
        # ``repr`` gives deterministic paired-NaN and signed-zero semantics.
        return (type(value).__name__, repr(value))
    if isinstance(value, (torch.dtype, torch.device)):
        return (type(value).__name__, str(value))
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if torch.is_tensor(value):
        try:
            version = value._version
        except RuntimeError:  # pragma: no cover - private audit owns normal tensors
            version = None
        return (
            "Tensor",
            id(value),
            version,
            _shape_tuple(value),
            value.dtype,
            value.device,
            value.layout,
            (
                tuple(int(item) for item in value.stride())
                if value.layout == torch.strided
                else None
            ),
            value.storage_offset() if value.layout == torch.strided else None,
            value.requires_grad,
            _audit_tensor_digest(value),
        )
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return (
            type(value).__name__,
            str(scalar.dtype),
            hashlib.sha256(scalar.tobytes()).hexdigest(),
        )
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "ndarray",
            id(value),
            tuple(int(size) for size in value.shape),
            str(value.dtype),
            hashlib.sha256(contiguous.tobytes()).hexdigest(),
        )
    if isinstance(value, torch.Generator):
        state = value.get_state().detach().cpu().contiguous().numpy().tobytes()
        return (
            "torch.Generator",
            id(value),
            str(value.device),
            hashlib.sha256(state).hexdigest(),
        )
    if inspect.isdatadescriptor(value) or inspect.ismethoddescriptor(value):
        return (
            "descriptor",
            id(value),
            type(value).__module__,
            type(value).__qualname__,
        )
    if isinstance(value, torch.nn.Module):
        return (
            "Module",
            id(value),
            type(value).__module__,
            type(value).__qualname__,
        )
    if callable(value):
        return (
            "callable",
            id(value),
            type(value).__module__,
            type(value).__qualname__,
        )

    identity = id(value)
    if identity in seen:
        return ("reference", identity)
    seen.add(identity)
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            identity,
            tuple(_audit_python_value_signature(item, seen) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        items = [_audit_python_value_signature(item, seen) for item in value]
        return (type(value).__name__, identity, tuple(sorted(items, key=repr)))
    if isinstance(value, dict):
        items = [
            (
                _audit_python_value_signature(key, seen),
                _audit_python_value_signature(item, seen),
            )
            for key, item in value.items()
        ]
        return ("dict", identity, tuple(sorted(items, key=repr)))
    if hasattr(value, "__dict__"):
        attributes = tuple(
            (
                name,
                _audit_python_value_signature(item, seen),
            )
            for name, item in sorted(vars(value).items())
        )
        return (
            "object",
            identity,
            type(value).__module__,
            type(value).__qualname__,
            attributes,
        )
    # Extension and immutable helper objects are retained by identity. Their
    # representation catches the common observable-state mutation case without
    # imposing a serialization protocol on legitimate framework internals.
    return (
        "opaque",
        identity,
        type(value).__module__,
        type(value).__qualname__,
        repr(value),
    )


def _transition_python_state_snapshot(module: torch.nn.Module) -> dict[str, object]:
    """Snapshot unregistered instance state throughout a transition tree."""
    snapshot = {
        "<module-tree>": tuple(
            (
                path,
                id(owner),
                type(owner).__module__,
                type(owner).__qualname__,
            )
            for path, owner in module.named_modules()
        )
    }
    for path, owner in module.named_modules():
        label = path or "<root>"
        for name, value in sorted(vars(owner).items()):
            if _is_internal_module_state(owner, name):
                continue
            if isinstance(owner, (Mechanism, State)) and (
                name in _RUNTIME_SCHEMA_CACHE_STATE
            ):
                continue
            if name == "_cache" and isinstance(owner, cacheable):
                continue
            snapshot[f"{label}.{name}"] = _audit_python_value_signature(value, set())
    return snapshot


def _authored_workspace_classes(module: torch.nn.Module) -> tuple[type, ...]:
    """Return authored classes whose mutable class state deepcopy shares."""
    population = getattr(module, "population", module)
    transform_module_ids = set()
    physical_populations = [population] if isinstance(population, Population) else []
    if type(population) is MultiPopulation:
        physical_populations.extend(population.populations.values())
    component_adapter_type = globals().get("_ComponentPhysicalPreparation")
    if component_adapter_type is not None:
        physical_populations.extend(
            child.population
            for child in module.modules()
            if isinstance(child, component_adapter_type)
        )
    for physical_population in physical_populations:
        initialization_transforms = getattr(
            physical_population,
            "_initialization_transforms",
            (),
        )
        for transform_action in initialization_transforms:
            action = initialization_transforms[transform_action]
            transform_module_ids.update(
                id(child) for child in action.transform.modules()
            )
        owners = (
            _parameterized_owners(physical_population)
            if hasattr(physical_population, "mech")
            else (physical_population,)
        )
        for owner in owners:
            for transforms in owner.in_graph_parametrizations.values():
                for transform, _args in transforms:
                    transform_module_ids.update(
                        id(child) for child in transform.modules()
                    )
            parametrizations = getattr(owner, "parametrizations", None)
            if parametrizations is not None:
                transform_module_ids.update(
                    id(child)
                    for transforms in parametrizations.values()
                    for transform in transforms
                    for child in transform.modules()
                )

    classes = set()
    for owner in module.modules():
        is_transform_module = id(owner) in transform_module_ids
        if not (
            isinstance(owner, (Mechanism, State, cacheable)) or is_transform_module
        ):
            continue
        if is_transform_module:
            # Multiple inheritance may place an authored configuration mixin
            # after nn.Module. Traverse the complete MRO, retaining authored
            # classes while excluding PyTorch's framework implementation.
            for cls in type(owner).__mro__:
                if cls in {torch.nn.Module, object} or cls.__module__.startswith(
                    "torch."
                ):
                    continue
                classes.add(cls)
        else:
            for cls in type(owner).__mro__:
                if cls in {Mechanism, State, cacheable, torch.nn.Module}:
                    break
                classes.add(cls)
    return tuple(sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__)))


@dataclass(frozen=True)
class _TensorClassRestore:
    storage: object
    storage_value: torch.Tensor
    storage_nbytes: int
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    requires_grad: bool


@dataclass(frozen=True)
class _NumpyClassRestore:
    value: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype


def _snapshot_tensor_class_restore(value: torch.Tensor) -> _TensorClassRestore:
    """Save the exact raw storage and view metadata of a shared class Tensor."""
    if value.layout != torch.strided:
        raise FunctionalizationError(
            "Functional lowering cannot safely audit mutable class Tensor state "
            f"with non-strided layout {value.layout}; move it to declared tensor "
            "state."
        )
    storage = value.untyped_storage()
    storage_nbytes = storage.nbytes()
    if storage_nbytes:
        storage_view = torch.empty(
            0,
            device=value.device,
            dtype=torch.uint8,
        ).set_(storage, 0, (storage_nbytes,), (1,))
        storage_value = storage_view.clone()
    else:
        storage_value = torch.empty(0, device=value.device, dtype=torch.uint8)
    return _TensorClassRestore(
        storage=storage,
        storage_value=storage_value,
        storage_nbytes=storage_nbytes,
        storage_offset=value.storage_offset(),
        size=_shape_tuple(value),
        stride=tuple(int(item) for item in value.stride()),
        dtype=value.dtype,
        device=value.device,
        layout=value.layout,
        requires_grad=value.requires_grad,
    )


def _safe_class_restore_copy(value, *, label: str, seen: set[int]):
    """Copy only class state whose snapshot cannot execute authored code."""
    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            torch.dtype,
            torch.device,
            torch.layout,
            np.generic,
        ),
    ):
        return value
    if value is Ellipsis or value is NotImplemented:
        return value
    if isinstance(value, (slice, range)):
        return value
    if isinstance(value, staticproperty):
        return staticproperty(value.func)
    if isinstance(value, torch.nn.Module):
        raise FunctionalizationError(
            f"Functional lowering found an unregistered class-level Module at "
            f"{label}; assign it to the instance so its state becomes explicit."
        )
    if (
        isinstance(value, (classmethod, staticmethod, property))
        or inspect.isroutine(value)
        or inspect.isclass(value)
    ):
        # Functions and standard descriptors are immutable class behavior.
        return value
    if callable(value):
        raise FunctionalizationError(
            f"Functional lowering found a stateful class-level callable object at "
            f"{label}; express it as class behavior or an instance child Module."
        )
    if isinstance(value, (types.MemberDescriptorType, types.GetSetDescriptorType)):
        return value
    if inspect.isdatadescriptor(value) or inspect.ismethoddescriptor(value):
        raise FunctionalizationError(
            f"Functional lowering cannot safely audit mutable class descriptor "
            f"{label}; use property, staticproperty, or declared tensor state."
        )
    if isinstance(value, torch.Generator):
        restored = torch.Generator(device=value.device)
        restored.set_state(value.get_state())
        return restored

    identity = id(value)
    if identity in seen:
        raise FunctionalizationError(
            f"Functional lowering cannot safely audit cyclic class state {label}; "
            "move it to declared tensor state or immutable configuration."
        )
    seen.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            restored = object.__new__(type(value))
            for field in dataclass_fields(value):
                object.__setattr__(
                    restored,
                    field.name,
                    _safe_class_restore_copy(
                        object.__getattribute__(value, field.name),
                        label=f"{label}.{field.name}",
                        seen=seen,
                    ),
                )
            return restored
        if type(value) is tuple:
            return tuple(
                _safe_class_restore_copy(
                    item,
                    label=f"{label}[{index}]",
                    seen=seen,
                )
                for index, item in enumerate(value)
            )
        if type(value) is frozenset:
            return frozenset(
                _safe_class_restore_copy(item, label=f"{label}[]", seen=seen)
                for item in value
            )
        if type(value) is list:
            return [
                _safe_class_restore_copy(
                    item,
                    label=f"{label}[{index}]",
                    seen=seen,
                )
                for index, item in enumerate(value)
            ]
        if type(value) is set:
            return {
                _safe_class_restore_copy(item, label=f"{label}[]", seen=seen)
                for item in value
            }
        if type(value) is dict:
            return {
                _safe_class_restore_copy(key, label=f"{label}.key", seen=seen): (
                    _safe_class_restore_copy(
                        item,
                        label=f"{label}[{key!r}]",
                        seen=seen,
                    )
                )
                for key, item in value.items()
            }
    finally:
        seen.remove(identity)
    raise FunctionalizationError(
        "Functional lowering cannot safely audit mutable class state "
        f"{label}; move it to declared tensor state or an immutable literal."
    )


def _class_state_snapshot(classes: tuple[type, ...]):
    """Capture restorable authored class attributes shared by every deepcopy."""
    snapshot = {}
    for cls in classes:
        attributes = {}
        for name, value in vars(cls).items():
            if name in _CLASS_STATE_METADATA_NAMES:
                continue
            if torch.is_tensor(value):
                restored = _snapshot_tensor_class_restore(value)
            elif isinstance(value, np.ndarray):
                restored = _NumpyClassRestore(
                    value=value.copy(),
                    shape=tuple(int(item) for item in value.shape),
                    dtype=value.dtype,
                )
            else:
                restored = _safe_class_restore_copy(
                    value,
                    label=f"{cls.__module__}.{cls.__qualname__}.{name}",
                    seen=set(),
                )
            signature = _audit_python_value_signature(value, set())
            attributes[name] = (signature, value, restored)
        snapshot[cls] = attributes
    return snapshot


def _restore_class_state(snapshot) -> set[str]:
    """Restore changed class attributes and return their qualified names."""
    changed = set()
    for cls, attributes in snapshot.items():
        current_names = {
            name for name in vars(cls) if name not in _CLASS_STATE_METADATA_NAMES
        }
        expected_names = set(attributes)
        for name in current_names - expected_names:
            changed.add(f"{cls.__module__}.{cls.__qualname__}.{name}")
            delattr(cls, name)
        for name, (signature, original, restored) in attributes.items():
            current = vars(cls).get(name, None)
            current_signature = (
                None
                if name not in vars(cls)
                else _audit_python_value_signature(current, set())
            )
            if current_signature != signature:
                changed.add(f"{cls.__module__}.{cls.__qualname__}.{name}")
                if isinstance(original, dict):
                    original.clear()
                    original.update(copy.deepcopy(restored))
                    restored_value = original
                elif isinstance(original, list):
                    original[:] = copy.deepcopy(restored)
                    restored_value = original
                elif isinstance(original, set):
                    original.clear()
                    original.update(copy.deepcopy(restored))
                    restored_value = original
                elif isinstance(original, np.ndarray):
                    if original.dtype != restored.dtype:
                        original.dtype = restored.dtype
                    original.resize(restored.shape, refcheck=False)
                    np.copyto(original, restored.value)
                    restored_value = original
                elif torch.is_tensor(original):
                    with torch.no_grad():
                        # Restore raw storage before the original view metadata.
                        # This preserves aliases and handles overlapping or
                        # expanded tensors, where logical ``copy_`` is invalid.
                        if restored.storage.nbytes() != restored.storage_nbytes:
                            restored.storage.resize_(restored.storage_nbytes)
                        if restored.storage_nbytes:
                            storage_view = torch.empty(
                                0,
                                device=restored.device,
                                dtype=torch.uint8,
                            ).set_(
                                restored.storage,
                                0,
                                (restored.storage_nbytes,),
                                (1,),
                            )
                            storage_view.copy_(restored.storage_value)
                        restored_view = torch.empty(
                            0,
                            device=restored.device,
                            dtype=restored.dtype,
                            layout=restored.layout,
                        ).set_(
                            restored.storage,
                            restored.storage_offset,
                            restored.size,
                            restored.stride,
                        )
                        # ``set_`` alone cannot restore dtype/device metadata
                        # changed through ``tensor.data = ...``. Reattach the
                        # exact typed view while preserving the original Python
                        # identity and captured shared raw storage.
                        original.data = restored_view
                    if original.requires_grad != restored.requires_grad:
                        original.requires_grad_(restored.requires_grad)
                    restored_value = original
                elif isinstance(original, torch.Generator):
                    original.set_state(restored.get_state())
                    restored_value = original
                else:
                    restored_value = original if current is not original else restored
                setattr(cls, name, restored_value)
    return changed


def _global_rng_snapshot(device: torch.device) -> dict[str, object]:
    snapshot = {
        "torch": torch.random.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if device.type == "cuda":  # pragma: no cover - future device slice
        snapshot["cuda"] = torch.cuda.get_rng_state(device)
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        snapshot["mps"] = torch.mps.get_rng_state()
    return snapshot


def _global_rng_changes(snapshot, device: torch.device) -> set[str]:
    changed = set()
    if not torch.equal(torch.random.get_rng_state(), snapshot["torch"]):
        changed.add("global PyTorch")
    if random.getstate() != snapshot["python"]:
        changed.add("global Python")
    if not _numpy_rng_state_equal(np.random.get_state(), snapshot["numpy"]):
        changed.add("global NumPy")
    if "cuda" in snapshot and not torch.equal(
        torch.cuda.get_rng_state(device), snapshot["cuda"]
    ):
        changed.add("CUDA")
    if "mps" in snapshot and not torch.equal(
        torch.mps.get_rng_state(), snapshot["mps"]
    ):
        changed.add("MPS")
    return changed


def _restore_global_rng(snapshot, device: torch.device) -> None:
    torch.random.set_rng_state(snapshot["torch"])
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    if "cuda" in snapshot:  # pragma: no cover - future device slice
        torch.cuda.set_rng_state(snapshot["cuda"], device)
    if "mps" in snapshot:
        torch.mps.set_rng_state(snapshot["mps"])


def _deepcopy_execution_module(module: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a module while detaching authored non-leaf tensor caches."""
    memo = {}
    for _name, value in module.named_buffers(remove_duplicate=False):
        memo.setdefault(id(value), _clone_tensor(value))
    handler = getattr(module, "mech", None)
    if handler is not None:
        # A child of an initialized MultiPopulation is configuration-only:
        # the packed parent owns the live current scratch, so these optional
        # child caches may never have been materialized.  They are execution
        # aliases when present, not a prerequisite for cloning the component.
        for values in (
            getattr(handler, "_buf_i", ()),
            getattr(handler, "_buf_g", ()),
        ):
            for value in values:
                if torch.is_tensor(value):
                    memo.setdefault(id(value), _clone_tensor(value))

    # Parameterized modules also retain a few ergonomic Python-side mappings
    # (for example ``range_p``) whose values can be differentiable views of
    # registered parameters.  MultiPopulation refreshes its packed physical
    # fields through exactly those mappings.  PyTorch deliberately refuses to
    # deepcopy non-leaf tensors, so detach such execution caches explicitly,
    # preserving identity aliases through the common memo. Canonical
    # Parameters and registered buffers are handled by deepcopy/the loops
    # above and must not be replaced here.
    visited_containers = set()

    def memoize_unregistered_tensors(value):
        if torch.is_tensor(value):
            if not isinstance(value, torch.nn.Parameter):
                memo.setdefault(id(value), _clone_tensor(value))
            return
        if isinstance(value, Mapping):
            if id(value) in visited_containers:
                return
            visited_containers.add(id(value))
            for item in value.values():
                memoize_unregistered_tensors(item)
            return
        if isinstance(value, (tuple, list)):
            if id(value) in visited_containers:
                return
            visited_containers.add(id(value))
            for item in value:
                memoize_unregistered_tensors(item)

    for owner in module.modules():
        for name, value in vars(owner).items():
            if name not in {"_parameters", "_buffers", "_modules"}:
                memoize_unregistered_tensors(value)
    return copy.deepcopy(module, memo)


def _unregistered_module_alias_paths(module: torch.nn.Module) -> tuple[str, ...]:
    """Return plain-Python paths that retain modules outside ``module``'s tree."""
    registered_ids = {id(owner) for owner in module.modules()}
    seen = set()
    aliases = []

    def walk(value, path):
        if isinstance(value, torch.nn.Module):
            if id(value) not in registered_ids:
                aliases.append(path)
            return
        if value is None or isinstance(
            value,
            (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                torch.Tensor,
                np.ndarray,
                torch.Generator,
                type,
            ),
        ):
            return
        if callable(value) or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Mapping):
            for name, item in value.items():
                walk(item, f"{path}[{name!r}]")
        elif isinstance(value, (tuple, list, set, frozenset)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif is_dataclass(value) and not isinstance(value, type):
            for field in dataclass_fields(value):
                walk(getattr(value, field.name), f"{path}.{field.name}")
        elif hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                walk(item, f"{path}.{name}")

    for module_path, owner in module.named_modules():
        prefix = module_path or "<root>"
        for name, value in vars(owner).items():
            if name not in {"_parameters", "_buffers", "_modules"}:
                walk(value, f"{prefix}.{name}")
    return tuple(sorted(set(aliases)))


def _rebind_population_time_references(population: Population) -> None:
    """Bind every instance-proxy clock reference to ``population`` itself.

    ``Referency.setreference`` installs a per-instance proxy class whose
    descriptor closes over the owning Population. ``deepcopy`` preserves that
    class object, so every disposable clone must shadow the descriptor before
    authored code runs against it. Otherwise a purity probe can read or mutate
    the retained execution plan through the copied mechanism's stale closure.
    """
    handler = getattr(population, "mech", None)
    if handler is None:
        return
    for mechanism in handler.mechanisms.values():
        mechanism.setreference("t", lambda population=population: population.t)
    for process in handler.material_processes.values():
        process.setreference("t", lambda population=population: population.t)


def _clone_execution_population(population: Population) -> Population:
    """Copy a model without retaining live autograd scratch or buffer aliases."""
    cloned = _deepcopy_execution_module(population)
    if cloned.integrator is not None:
        cloned.integrator.clear_jit_cache()
    # Imperative execution installs conditional Dynamo barriers around shared
    # ion/material synchronization.  This private program is explicitly driven
    # by an outer functional_call/torch.compile boundary, so retain the original
    # class methods instead of those per-instance wrappers.
    if cloned.mech is not None:
        _rebind_population_time_references(cloned)
        for method_name in cloned.mech._SYNC_METHODS:
            cloned.mech.__dict__.pop(method_name, None)
    return cloned


def _strip_packed_component_runtime_aliases(population: MultiPopulation) -> None:
    """Retain only the packed owner after its component plan is compiled."""
    if "populations" not in population._modules:
        raise FunctionalizationError(
            "The cloned MultiPopulation lost its registered components before "
            "packed runtime lowering."
        )
    # MultiPopulation exposes each component as a labelled Slice, and those
    # Slices recursively retain their source component labels/models through
    # ordinary Python attributes.  Remove the complete ergonomic label tree
    # while its registry is still intact, before severing the registered
    # component owners below.
    population.clear_labels()
    population._modules["populations"] = torch.nn.ModuleDict()
    population.injections = []
    population.mechanism_injections = []
    population.mechanism_injection_accepted = []
    population.intra = None
    aliases = _unregistered_module_alias_paths(population)
    if aliases:
        raise FunctionalizationError(
            "Packed functional runtime retained unregistered component module "
            f"aliases at {list(aliases)}."
        )


def _module_tensor_slots(module: torch.nn.Module):
    """Yield each canonical registered tensor slot, including shared buffers.

    ``named_buffers()`` deduplicates by tensor identity. Dendra intentionally
    shares a few synchronization buffers (notably temperature) across distinct
    module owners, and each owner can be rebound during a step. Supplying every
    canonical owner/slot lets functional_call restore all of them while still
    avoiding the ergonomic module aliases exposed by Population.
    """
    for module_path, owner in module.named_modules():
        prefix = f"{module_path}." if module_path else ""
        for name, value in owner._parameters.items():
            if value is not None:
                yield f"{prefix}{name}", value
        for name, value in owner._buffers.items():
            if value is not None:
                yield f"{prefix}{name}", value


def _parameter_alias_paths(module: torch.nn.Module) -> dict[str, tuple[str, ...]]:
    """Map each canonical Parameter to every Parameter/buffer alias slot."""
    canonical_by_identity = {
        id(parameter): name for name, parameter in module.named_parameters()
    }
    aliases = {name: [] for name in canonical_by_identity.values()}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases[canonical_by_identity[id(parameter)]].append(name)
    # PyTorch permits one Tensor object to be registered in both categories.
    # ``functional_call`` does not infer that cross-category alias when
    # ``tie_weights=False``, so bind every buffer slot from the same public
    # Parameter leaf explicitly.
    for module_path, owner in module.named_modules():
        prefix = f"{module_path}." if module_path else ""
        for buffer_name, value in owner._buffers.items():
            if value is not None and id(value) in canonical_by_identity:
                aliases[canonical_by_identity[id(value)]].append(
                    f"{prefix}{buffer_name}"
                )
    return {name: tuple(paths) for name, paths in aliases.items()}


def _unpack_forward_ad(value: torch.Tensor):
    """Expose a dual tangent even when the caller has inference mode enabled."""
    if torch.is_inference_mode_enabled():
        # Inference mode normally hides forward-AD tangents from unpack_dual().
        # Freshness metadata must describe the tensor itself rather than the
        # ambient execution mode, or a plan could silently lose its tangent
        # binding when prepared/consumed under inference mode.
        with torch.inference_mode(False):
            return torch.autograd.forward_ad.unpack_dual(value)
    return torch.autograd.forward_ad.unpack_dual(value)


def _has_inference_tensor_component(value: torch.Tensor) -> bool:
    tangent = _unpack_forward_ad(value).tangent
    return torch.is_inference(value) or (
        tangent is not None and torch.is_inference(tangent)
    )


def _tensor_leaf_binding(value: torch.Tensor) -> tuple:
    """Bind one primal (or ordinary Tensor) and its optional dual tangent."""
    unpacked = _unpack_forward_ad(value)
    tangent = unpacked.tangent
    if torch.is_inference(value) or (
        tangent is not None and torch.is_inference(tangent)
    ):
        raise TypeError(
            "tensor bindings require versioned primal and forward-AD tangent values"
        )
    tangent_binding = (
        None
        if tangent is None
        else (id(tangent), tangent._version, tangent.requires_grad)
    )
    # ``value`` is the stable owner for a dual's primal storage. The primal
    # view returned by unpack_dual() is recreated on demand, so its Python
    # identity is deliberately not part of the freshness contract.
    return (id(value), value._version, value.requires_grad, tangent_binding)


def _tensor_binding(values: Mapping[str, torch.Tensor]) -> tuple:
    """Bind a derived workspace to the exact tensor versions that created it."""
    return tuple((name, *_tensor_leaf_binding(values[name])) for name in sorted(values))


def _tensor_tree_binding(values) -> tuple:
    """Bind every tensor leaf in a prepared workspace tree."""
    leaves, tree_spec = torch.utils._pytree.tree_flatten(values)
    if any(not torch.is_tensor(value) for value in leaves):
        raise TypeError("prepared workspaces must contain only Tensor leaves")
    return (
        tree_spec,
        tuple(_tensor_leaf_binding(value) for value in leaves),
    )


class _PreparedPopulation:
    """Opaque plan-bound tensor bundle returned by :meth:`prepare`."""

    __slots__ = (
        "__weakref__",
        "token",
        "parameter_binding",
        "constant_binding",
        "constant_sources",
        "grad_enabled_at_prepare",
        "value_binding",
        "values",
    )

    def __init__(
        self,
        token,
        parameter_binding,
        constant_binding,
        constant_sources,
        grad_enabled_at_prepare,
        values,
    ):
        self.token = token
        self.parameter_binding = parameter_binding
        self.constant_binding = constant_binding
        self.constant_sources = constant_sources
        self.grad_enabled_at_prepare = grad_enabled_at_prepare
        self.value_binding = _tensor_tree_binding(values)
        self.values = values


def _shape_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _module_at(population: Population, path: str) -> torch.nn.Module:
    return population if not path else population.get_submodule(path)


def _buffer_at(population: Population, module_path: str, name: str) -> torch.Tensor:
    owner = _module_at(population, module_path)
    if name in owner._buffers:
        return owner._buffers[name]
    return getattr(owner, name)


def _set_buffer(
    population: Population,
    module_path: str,
    name: str,
    value: torch.Tensor,
) -> None:
    owner = _module_at(population, module_path)
    if name in owner._buffers:
        owner._buffers[name] = value
    else:
        setattr(owner, name, value)


def _put_path(tree: dict, path: tuple[str, ...], value) -> None:
    cursor = tree
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def _get_path(tree: Mapping, path: tuple[str, ...]):
    value = tree
    for part in path:
        value = value[part]
    return value


@dataclass(frozen=True)
class _StateLeaf:
    public_path: tuple[str, ...]
    module_path: str
    buffer_name: str
    mapping_slots: tuple[tuple[str, str], ...]
    checkpoint_key: str | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _InitializationStateLeaf:
    """One insertion-time ``ic`` tensor consumed by fresh initialization."""

    name: str
    public_path: tuple[str, ...]
    mechanism_name: str
    state_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _InitializationTransformInput:
    """One explicit tensor consumed by a registered pure init transform."""

    key: str
    phase: str
    action_name: str
    input_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _PreparedBuffer:
    key: str
    module_path: str
    buffer_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    kind: str


@dataclass(frozen=True)
class _GeometryBinding:
    """One full or support-local geometry value and its registered aliases."""

    constant_name: str
    buffer_name: str
    module_paths: tuple[str, ...]
    support_entry: SupportEntry | None = None


@dataclass(frozen=True)
class _PointAreaFactor:
    """One framework-owned PointProcess density-conversion workspace."""

    key: str
    module_path: str
    buffer_name: str
    shape: tuple[int, ...]
    support_entry: SupportEntry


@dataclass(frozen=True)
class _ParametrizationConstant:
    """One explicit registered buffer read by an in-graph transform."""

    key: str
    module_path: str
    buffer_name: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _MultiComponentPlan:
    """One immutable component-preparation adapter for a packed population.

    Component plans are used only to derive differentiable geometry and
    electrical coefficients.  Runtime carry, mechanisms, stimulation, and the
    voltage solve remain owned by the packed ``MultiPopulation`` transition.
    """

    name: str
    parameter_sources: tuple[tuple[str, str], ...]
    parameter_slots: tuple[tuple[str, tuple[str, ...]], ...]
    constant_sources: tuple[tuple[str, str], ...]
    geometry_constant_names: tuple[str, ...]
    parametrization_constants: tuple[_ParametrizationConstant, ...]
    topology_kind: str
    core_shape: tuple[int, ...]


def _derived_buffer_names(module) -> tuple[str, ...]:
    """Return the statically declared initialization-derived buffer names."""
    return tuple(sorted(getattr(module, "_derived_buffers", ())))


def _declared_buffer_layout_signature(
    module,
    names,
    *,
    specs_attribute: str,
    resolved_attribute: str,
) -> tuple[tuple[str, object, object], ...]:
    """Return authored and once-resolved storage layouts for freshness checks."""

    specs = getattr(module, specs_attribute, {})
    resolved = getattr(module, resolved_attribute, {})
    return tuple(
        (name, specs.get(name, (None, "local")), resolved.get(name)) for name in names
    )


def _unresolved_deferred_layouts(module) -> tuple[str, ...]:
    unresolved = []
    for attribute in ("_carry_resolved_shapes", "_derived_resolved_shapes"):
        unresolved.extend(
            name
            for name, shape in getattr(module, attribute, {}).items()
            if shape is None
        )
    return tuple(unresolved)


def _timestep_buffer_names(module) -> tuple[str, ...]:
    """Return the statically declared timestep-derived buffer names."""
    return tuple(sorted(getattr(module, "_timestep_buffers", ())))


def _timestep_buffer_shape_specs(module) -> tuple[tuple[str, object], ...]:
    """Return the authored canonical-shape contract for each timestep buffer."""

    shapes = getattr(module, "_timestep_buffer_shapes", {})
    return tuple(
        (name, shapes.get(name, "local")) for name in _timestep_buffer_names(module)
    )


def _derived_handler_keys(handler) -> set[str]:
    """Return checkpoint keys reconstructed from prepared workspace tensors."""
    keys = set()
    for mechanism_name, mechanism in handler.mechanisms.items():
        keys.update(
            f"{mechanism_name}.{buffer_name}"
            for buffer_name in (
                *_derived_buffer_names(mechanism),
                *_timestep_buffer_names(mechanism),
            )
        )
        for state_name, state in mechanism.DE.items():
            keys.update(
                f"{mechanism_name}.DE.{state_name}.{buffer_name}"
                for buffer_name in (
                    *_derived_buffer_names(state),
                    *_timestep_buffer_names(state),
                )
            )
    return keys


def _canonical_handler_keys(handler) -> dict[str, tuple[str, ...]]:
    """Map checkpoint inventory keys to the public functional state tree."""
    keys = {}
    for mechanism_name, mechanism in handler.mechanisms.items():
        for state in mechanism.DE.values():
            for state_name in sorted(state._state):
                keys[f"{mechanism_name}.{state_name}"] = (
                    "mechanisms",
                    mechanism_name,
                    state_name,
                )
        for buffer_name in sorted(mechanism._carry):
            keys[f"{mechanism_name}.{buffer_name}"] = (
                "mechanism_buffers",
                mechanism_name,
                buffer_name,
            )
        for saved_name in sorted(mechanism._save):
            buffer_name = f"{saved_name}_"
            keys[f"{mechanism_name}.{buffer_name}"] = (
                "mechanism_buffers",
                mechanism_name,
                buffer_name,
            )
        for state_name, state in mechanism.DE.items():
            for buffer_name in sorted(state._carry):
                keys[f"{mechanism_name}.DE.{state_name}.{buffer_name}"] = (
                    "state_buffers",
                    mechanism_name,
                    state_name,
                    buffer_name,
                )

    for process_name, process in handler.material_processes.items():
        for state in process.DE.values():
            for state_name in sorted(state._state):
                keys[f"{process_name}.{state_name}"] = (
                    "material_processes",
                    process_name,
                    state_name,
                )
        for buffer_name in sorted(process._carry):
            keys[f"{process_name}.{buffer_name}"] = (
                "material_process_buffers",
                process_name,
                buffer_name,
            )
        for saved_name in sorted(process._save):
            buffer_name = f"{saved_name}_"
            keys[f"{process_name}.{buffer_name}"] = (
                "material_process_buffers",
                process_name,
                buffer_name,
            )
        for state_name, state in process.DE.items():
            for buffer_name in sorted(state._carry):
                keys[f"{process_name}.DE.{state_name}.{buffer_name}"] = (
                    "material_process_state_buffers",
                    process_name,
                    state_name,
                    buffer_name,
                )

    for ion_name, ion in handler.ions.items():
        for field in ion.fields:
            keys[f"{ion_name}_ion.{field}"] = ("ions", ion_name, field)
    for material_name, material in handler.materials.items():
        for field in material.fields:
            keys[f"{material_name}_material.{field}"] = (
                "materials",
                material_name,
                field,
            )
    for checkpoint_key, public_path in _persistent_shared_local_keys(handler).items():
        keys.setdefault(checkpoint_key, public_path)
    return keys


def _local_synchronization_keys(handler) -> set[str]:
    """Return transient local mirrors reconstructed from canonical carry."""
    keys = set()
    for ion_map in handler.read_ion.values():
        for mechanism_name, fields in ion_map.items():
            keys.update(f"{mechanism_name}.{field}" for field in fields)
    for material_map in handler.read_material.values():
        for mechanism_name, fields in material_map.items():
            keys.update(f"{mechanism_name}.{field}" for field in fields)
    return keys


def _persistent_shared_local_keys(handler) -> dict[str, tuple[str, ...]]:
    """Classify write/source locals that are not rebound from shared fields."""
    transient = _local_synchronization_keys(handler)
    keys = {}
    for ion_map in handler.write_ion_c.values():
        for mechanism_name, fields in ion_map.items():
            for field in fields:
                checkpoint_key = f"{mechanism_name}.{field}"
                if checkpoint_key not in transient:
                    keys.setdefault(
                        checkpoint_key,
                        ("ion_write_buffers", mechanism_name, field),
                    )
    for material_map in handler.write_material.values():
        for mechanism_name, fields in material_map.items():
            for field in fields:
                checkpoint_key = f"{mechanism_name}.{field}"
                if checkpoint_key not in transient:
                    keys.setdefault(
                        checkpoint_key,
                        ("material_write_buffers", mechanism_name, field),
                    )
    for material_map in handler.source_material.values():
        for mechanism_name, field_map in material_map.items():
            for local_name in field_map.values():
                checkpoint_key = f"{mechanism_name}.{local_name}"
                if checkpoint_key not in transient:
                    keys.setdefault(
                        checkpoint_key,
                        ("material_source_buffers", mechanism_name, local_name),
                    )
    return keys


def _state_layout(population: Population) -> tuple[_StateLeaf, ...]:
    """Discover canonical carry leaves through the checkpoint state manifest."""
    handler = population.mech
    expected = _canonical_handler_keys(handler)
    bindings = {}
    for checkpoint_key, owner, buffer_name in handler._mutable_state_bindings():
        bindings.setdefault(checkpoint_key, []).append((owner, buffer_name))

    unclassified = (
        set(bindings)
        - set(expected)
        - _local_synchronization_keys(handler)
        - _derived_handler_keys(handler)
    )
    if unclassified:
        raise FunctionalizationError(
            "The MechanismHandler mutable-state manifest contains unclassified "
            f"carry fields: {sorted(unclassified)}."
        )

    module_paths = {}
    for name, module in population.named_modules(remove_duplicate=False):
        # Population exposes ergonomic aliases (for example ``mech.hh``) for
        # modules that are canonically owned below the integrator. Keep the
        # first registered path so functional_call receives the same canonical
        # names as named_parameters()/named_buffers().
        module_paths.setdefault(id(module), name)
    integrator_state_names = tuple(population.integrator.v_vars)
    if not integrator_state_names or len(integrator_state_names) != len(
        set(integrator_state_names)
    ):
        raise FunctionalizationError(
            "Integrator v_vars must declare unique Population tensor-state names."
        )
    leaves = []
    for state_name in integrator_state_names:
        value = getattr(population, state_name, None)
        if not torch.is_tensor(value) or state_name not in population._buffers:
            raise FunctionalizationError(
                f"Integrator state {state_name!r} must be a registered Population "
                "Tensor buffer."
            )
        leaves.append(
            _StateLeaf(
                public_path=("integrator", state_name),
                module_path="",
                buffer_name=state_name,
                mapping_slots=(("", state_name),),
                checkpoint_key=None,
                shape=_shape_tuple(value),
                dtype=value.dtype,
                device=value.device,
            )
        )
    for checkpoint_key, public_path in expected.items():
        try:
            owner, buffer_name = bindings[checkpoint_key][0]
        except KeyError as exc:
            raise FunctionalizationError(
                "The MechanismHandler mutable-state manifest is missing "
                f"{checkpoint_key!r}."
            ) from exc
        try:
            module_path = module_paths[id(owner)]
        except KeyError as exc:
            raise FunctionalizationError(
                f"Mutable state owner for {checkpoint_key!r} is not registered."
            ) from exc
        value = owner._buffers[buffer_name]
        mapping_slots = []
        for alias_owner, alias_name in bindings[checkpoint_key]:
            try:
                alias_path = module_paths[id(alias_owner)]
            except KeyError as exc:
                raise FunctionalizationError(
                    f"Mutable state alias for {checkpoint_key!r} is not registered."
                ) from exc
            mapping_slots.append((alias_path, alias_name))
        leaves.append(
            _StateLeaf(
                public_path=public_path,
                module_path=module_path,
                buffer_name=buffer_name,
                mapping_slots=tuple(mapping_slots),
                checkpoint_key=checkpoint_key,
                shape=_shape_tuple(value),
                dtype=value.dtype,
                device=value.device,
            )
        )

    leaves.extend(
        (
            _StateLeaf(
                public_path=("clock", "t"),
                module_path="",
                buffer_name="t",
                mapping_slots=(("", "t"),),
                checkpoint_key=None,
                shape=_shape_tuple(population.t),
                dtype=population.t.dtype,
                device=population.t.device,
            ),
            _StateLeaf(
                public_path=("control", "duration_remainder"),
                module_path="",
                buffer_name="_duration_remainder",
                mapping_slots=(("", "_duration_remainder"),),
                checkpoint_key=None,
                shape=_shape_tuple(population._duration_remainder),
                dtype=population._duration_remainder.dtype,
                device=population._duration_remainder.device,
            ),
        )
    )
    return tuple(leaves)


def _state_layout_signature(layout: tuple[_StateLeaf, ...]) -> tuple:
    """Return the immutable public/runtime binding contract for carry state."""
    return tuple(
        (
            leaf.public_path,
            leaf.module_path,
            leaf.buffer_name,
            leaf.mapping_slots,
            leaf.checkpoint_key,
            leaf.shape,
            leaf.dtype,
            leaf.device,
        )
        for leaf in layout
    )


def _initialization_state_layout(
    population: Population,
    state_layout: tuple[_StateLeaf, ...],
) -> tuple[_InitializationStateLeaf, ...]:
    """Expose insertion-time ``ic`` values as canonical tensor inputs."""

    entries = []
    for leaf in state_layout:
        if len(leaf.public_path) != 3 or leaf.public_path[0] != "mechanisms":
            continue
        _category, mechanism_name, state_name = leaf.public_path
        mechanism = population.mech.mechanisms[mechanism_name]
        if state_name not in mechanism._init_params:
            continue
        entries.append(
            _InitializationStateLeaf(
                name=".".join(leaf.public_path),
                public_path=leaf.public_path,
                mechanism_name=mechanism_name,
                state_name=state_name,
                shape=leaf.shape,
                dtype=leaf.dtype,
                device=leaf.device,
            )
        )
    return tuple(entries)


def _initialization_transform_hooks(
    population: Population,
    phase: str,
) -> tuple[_InitializationTransformHook, ...]:
    """Return registered pure actions in their imperative hook order."""

    hooks = (
        population.pre_initialize_hooks
        if phase == "pre"
        else population.post_initialize_hooks
    )
    return tuple(
        hook for hook in hooks if isinstance(hook, _InitializationTransformHook)
    )


def _initialization_transform_input_layout(
    population: Population,
) -> tuple[_InitializationTransformInput, ...]:
    """Expose action-owned defaults as explicit initialization-only leaves."""

    entries = []
    for phase in ("pre", "post"):
        for hook in _initialization_transform_hooks(population, phase):
            action = population._initialization_transforms[hook.action_name]
            values = action.input_values()
            if len(values) != len(action.input_names):
                raise FunctionalizationError(
                    f"initialization transform {action.name!r} has an invalid "
                    "explicit-input registration"
                )
            for input_name, value in zip(
                action.input_names,
                values,
                strict=True,
            ):
                if not torch.is_tensor(value):
                    raise FunctionalizationError(
                        f"initialization transform input {action.name}.{input_name} "
                        "must be a Tensor"
                    )
                entries.append(
                    _InitializationTransformInput(
                        key=f"{phase}.{action.name}.{input_name}",
                        phase=phase,
                        action_name=action.name,
                        input_name=input_name,
                        shape=_shape_tuple(value),
                        dtype=value.dtype,
                        device=value.device,
                    )
                )
    return tuple(entries)


def _initialization_transform_signature(population: Population) -> tuple:
    """Fingerprint the ordered, pure Population initialization program."""

    phases = []
    for phase in ("pre", "post"):
        actions = []
        for hook in _initialization_transform_hooks(population, phase):
            action = population._initialization_transforms[hook.action_name]
            actions.append(
                (
                    phase,
                    action.name,
                    tuple(action.reads),
                    tuple(action.writes),
                    tuple(action.input_names),
                    tuple(
                        (
                            entry.input_name,
                            entry.shape,
                            entry.dtype,
                            entry.device,
                        )
                        for entry in _initialization_transform_input_layout(population)
                        if entry.phase == phase and entry.action_name == action.name
                    ),
                    type(action.transform).__module__,
                    type(action.transform).__qualname__,
                    _initialization_callable_dependency_signature(
                        action.transform.forward,
                        label=f"{phase} initialization transform {action.name}",
                        owner=action.transform,
                        strict=False,
                        include_identity=False,
                    ),
                )
            )
        phases.append(tuple(actions))
    return tuple(phases)


def _parameterized_module_paths(population: Population) -> tuple[str, ...]:
    paths = []
    handler = population.mech
    if handler is None:
        return ()
    for collection_name in ("mechanisms", "material_processes"):
        collection = getattr(handler, collection_name)
        for name, module in collection.items():
            prefix = f"integrator.mech.{collection_name}.{name}"
            paths.append(prefix)
            paths.extend(f"{prefix}.DE.{state_name}" for state_name in module.DE)
    return tuple(paths)


def _declared_parameter_names(module) -> tuple[str, ...]:
    names = []
    seen = set()
    for category in _PARAMETER_CATEGORIES:
        for name in getattr(module.__class__, category, {}):
            if name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def _prepared_buffer_layout(population: Population) -> tuple[_PreparedBuffer, ...]:
    effective = []
    derived = []
    timestep = []
    for module_path in _parameterized_module_paths(population):
        module = _module_at(population, module_path)
        for name in _declared_parameter_names(module):
            if name not in module._buffers or not hasattr(module, f"{name}_param"):
                continue
            buffer = module._buffers[name]
            effective.append(
                _PreparedBuffer(
                    key=f"{module_path}.{name}",
                    module_path=module_path,
                    buffer_name=name,
                    shape=_shape_tuple(buffer),
                    dtype=buffer.dtype,
                    device=buffer.device,
                    kind="effective_parameter",
                )
            )
        for name in _derived_buffer_names(module):
            buffer = module._buffers[name]
            derived.append(
                _PreparedBuffer(
                    key=f"{module_path}.{name}",
                    module_path=module_path,
                    buffer_name=name,
                    shape=_shape_tuple(buffer),
                    dtype=buffer.dtype,
                    device=buffer.device,
                    kind="derived",
                )
            )
        for name in _timestep_buffer_names(module):
            buffer = module._buffers[name]
            timestep.append(
                _PreparedBuffer(
                    key=f"{module_path}.{name}",
                    module_path=module_path,
                    buffer_name=name,
                    shape=_shape_tuple(buffer),
                    dtype=buffer.dtype,
                    device=buffer.device,
                    kind="timestep",
                )
            )
    return tuple((*effective, *derived, *timestep))


def _geometry_binding_layout(population: Population) -> tuple[_GeometryBinding, ...]:
    """Plan full and support-local registered geometry aliases once.

    A Mechanism and every nested State share one local diameter tensor in the
    imperative handler. Preserve that identity while making the gather an
    explicit differentiable operation in both preparation and transition
    mapping. MaterialProcess geometry is deliberately rejected by functional
    eligibility and therefore cannot silently fall back to dense geometry here.
    """

    bindings = [_GeometryBinding("dx", "dx", ("",))]
    if "diam" in population._buffers:
        bindings.append(_GeometryBinding("diam", "diam", ("",)))

    parameterized_paths = set(_parameterized_module_paths(population))
    module_paths = {}
    for module_path, owner in population.named_modules(remove_duplicate=False):
        module_paths.setdefault(id(owner), module_path)

    covered_paths = set()
    binding_index_by_support = {}
    handler = population.mech
    for mechanism, support_entry in handler._mechanism_support_entry_plan:
        local_paths = []
        for owner in (mechanism, *mechanism.DE.values()):
            module_path = module_paths.get(id(owner))
            if module_path in parameterized_paths and "diam" in owner._buffers:
                local_paths.append(module_path)
                covered_paths.add(module_path)
        if local_paths:
            binding_index = binding_index_by_support.get(support_entry.support_id)
            if binding_index is None:
                binding_index_by_support[support_entry.support_id] = len(bindings)
                bindings.append(
                    _GeometryBinding(
                        "diam",
                        "diam",
                        tuple(local_paths),
                        support_entry,
                    )
                )
            else:
                binding = bindings[binding_index]
                bindings[binding_index] = _GeometryBinding(
                    binding.constant_name,
                    binding.buffer_name,
                    (*binding.module_paths, *local_paths),
                    binding.support_entry,
                )

    uncovered_paths = sorted(
        module_path
        for module_path in parameterized_paths - covered_paths
        if "diam" in _module_at(population, module_path)._buffers
    )
    if uncovered_paths:
        raise FunctionalizationError(
            "Functional geometry owners are missing canonical Mechanism support: "
            f"{uncovered_paths}."
        )
    return tuple(bindings)


def _point_area_factor_layout(population: Population) -> tuple[_PointAreaFactor, ...]:
    """Plan each current-producing PointProcess's explicit area workspace."""

    handler = population.mech
    module_paths = {}
    for module_path, owner in population.named_modules(remove_duplicate=False):
        module_paths.setdefault(id(owner), module_path)
    support_entries = {
        id(mechanism): support_entry
        for mechanism, support_entry in handler._mechanism_support_entry_plan
    }

    layout = []
    seen = set()
    for _current_index, mechanism, _fn, _scaler, _factorable in handler._map:
        if not isinstance(mechanism, PointProcess) or id(mechanism) in seen:
            continue
        seen.add(id(mechanism))
        module_path = module_paths.get(id(mechanism))
        support_entry = support_entries.get(id(mechanism))
        buffer_name = PointProcess._AREA_FACTOR_BUFFER
        value = mechanism._buffers.get(buffer_name)
        if module_path is None or support_entry is None or value is None:
            raise FunctionalizationError(
                f"PointProcess {mechanism.name!r} is missing its canonical "
                "module path, support, or area workspace."
            )
        layout.append(
            _PointAreaFactor(
                key=f"{module_path}.{buffer_name}",
                module_path=module_path,
                buffer_name=buffer_name,
                shape=_shape_tuple(value),
                support_entry=support_entry,
            )
        )
    return tuple(layout)


def _geometry_binding_value(
    binding: _GeometryBinding,
    geometry: Mapping[str, torch.Tensor],
):
    """Resolve one full or support-local geometry tensor out of place."""

    value = geometry[binding.constant_name]
    if binding.support_entry is not None:
        value = binding.support_entry.gather(value)
    return value


def _cache_local_geometry(binding: _GeometryBinding) -> bool:
    """Whether this binding needs a prepared support-local tensor leaf."""

    if binding.support_entry is None:
        return False
    spec = binding.support_entry.spec
    return spec is None or spec.kind is not SupportKind.DENSE


def _geometry_binding_signature(bindings) -> tuple:
    """Describe positional geometry plans without target-local representatives."""

    return tuple(
        (
            binding.constant_name,
            binding.buffer_name,
            binding.module_paths,
            (
                None
                if binding.support_entry is None or binding.support_entry.spec is None
                else binding.support_entry.spec.ordered_signature
            ),
        )
        for binding in bindings
    )


def _mechanism_dt_slots(population: Population) -> tuple[str, ...]:
    """Return registered Mechanism timestep slots rebound during transition."""
    return tuple(
        f"population.{module_path}.dt"
        for module_path in _parameterized_module_paths(population)
        if "dt" in _module_at(population, module_path)._buffers
    )


def _local_synchronization_slots(population: Population) -> tuple[str, ...]:
    """Return framework-owned local mirrors rebound from shared carry fields."""
    handler = population.mech
    module_paths = {}
    for module_path, owner in population.named_modules():
        module_paths.setdefault(id(owner), module_path)

    slots = []
    for checkpoint_key in _local_synchronization_keys(handler):
        owner_name, separator, buffer_name = checkpoint_key.partition(".")
        if not separator:
            continue  # pragma: no cover - handler keys are always qualified
        owner = (
            handler.mechanisms[owner_name] if owner_name in handler.mechanisms else None
        )
        if owner is None and owner_name in handler.material_processes:
            owner = handler.material_processes[owner_name]
        if owner is None:
            continue  # pragma: no cover - guarded by the handler manifest
        owners = (owner, *getattr(owner, "DE", {}).values())
        for local_owner in owners:
            if id(local_owner) in module_paths and buffer_name in local_owner._buffers:
                slots.append(
                    f"population.{module_paths[id(local_owner)]}.{buffer_name}"
                )
    return tuple(sorted(slots))


def _ephemeral_assigned_slots(population: Population) -> tuple[str, ...]:
    """Return registered Mechanism ASSIGNED slots rebuilt during evaluation."""

    module_paths = {
        id(owner): module_path for module_path, owner in population.named_modules()
    }
    slots = []
    handler = population.mech
    for collection_name in ("mechanisms", "material_processes"):
        for owner in getattr(handler, collection_name).values():
            module_path = module_paths.get(id(owner))
            if module_path is None:  # pragma: no cover - handler owns every module
                continue
            slots.extend(
                f"population.{module_path}.{name}"
                for name in owner._assigned
                if name in owner._buffers
            )
    return tuple(sorted(slots))


def _resolve_parameter_value(owner, parameter_name: str) -> torch.Tensor:
    parameter = getattr(owner, parameter_name)
    if isinstance(parameter, torch.nn.Parameter):
        return parameter
    if isinstance(parameter, cacheable):
        return parameter._compute()
    if isinstance(parameter, torch.nn.Module):
        return parameter()
    if torch.is_tensor(parameter):
        return parameter
    raise FunctionalizationError(
        f"Parameter source {parameter_name!r} is not tensor-valued."
    )


def _evaluate_derived_buffers(module) -> Mapping[str, torch.Tensor]:
    """Evaluate one pure, statically keyed derived-buffer builder."""
    expected = set(_derived_buffer_names(module))
    values = module.derive_buffers()
    if not isinstance(values, Mapping):
        raise FunctionalizationError(
            f"{type(module).__qualname__}.derive_buffers() must return a mapping"
        )
    if set(values) != expected:
        missing = sorted(expected - set(values))
        unexpected = sorted(set(values) - expected)
        raise FunctionalizationError(
            f"{type(module).__qualname__}.derive_buffers() returned the wrong "
            f"keys; missing={missing}, unexpected={unexpected}"
        )
    normalized = {}
    for name, value in values.items():
        if not torch.is_tensor(value):
            raise FunctionalizationError(
                f"{type(module).__qualname__}.derive_buffers()[{name!r}] "
                "must be a Tensor"
            )
        reference = module._buffers[name]
        if value.device != reference.device or value.dtype != reference.dtype:
            raise FunctionalizationError(
                f"{type(module).__qualname__}.derive_buffers()[{name!r}] must "
                f"use {reference.device}/{reference.dtype}; got "
                f"{value.device}/{value.dtype}"
            )
        try:
            value = torch.broadcast_to(value, reference.shape)
        except RuntimeError as exc:
            raise FunctionalizationError(
                f"{type(module).__qualname__}.derive_buffers()[{name!r}] with "
                f"shape {_shape_tuple(value)} is not broadcastable to its "
                f"resolved shape {_shape_tuple(reference)}"
            ) from exc
        normalized[name] = value
    return normalized


def _evaluate_timestep_buffers(module, dt) -> Mapping[str, torch.Tensor]:
    """Evaluate one pure, statically keyed timestep-buffer builder."""
    expected = set(_timestep_buffer_names(module))
    values = module.derive_timestep_buffers(dt)
    if not isinstance(values, Mapping):
        raise FunctionalizationError(
            f"{type(module).__qualname__}.derive_timestep_buffers() must return "
            "a mapping"
        )
    if set(values) != expected:
        missing = sorted(expected - set(values))
        unexpected = sorted(set(values) - expected)
        raise FunctionalizationError(
            f"{type(module).__qualname__}.derive_timestep_buffers() returned the "
            f"wrong keys; missing={missing}, unexpected={unexpected}"
        )
    for name, value in values.items():
        if not torch.is_tensor(value):
            raise FunctionalizationError(
                f"{type(module).__qualname__}.derive_timestep_buffers()[{name!r}] "
                "must be a Tensor"
            )
    return values


class _PopulationPreparation(torch.nn.Module):
    """Pure materialization of effective parameters and static workspaces."""

    def __init__(self, population: Population, dt):
        super().__init__()
        self.population = population
        # Constructor-time materialization is part of the purity boundary too.
        # Audit it immediately so a transform that mutates only on its first
        # call cannot normalize that write into the retained execution clone.
        self.audit_builder_inputs = True
        self.dt_value = float(dt)
        bindings = self._registered_input_bindings()
        with torch.no_grad():
            population_values = population._derive_parameter_buffers()
        self._assert_registered_inputs_unchanged(
            bindings,
            f"{type(population).__qualname__}._derive_parameter_buffers()",
        )
        self.population_parameter_names = tuple(population_values)
        self.population_parameter_shapes = {
            name: _shape_tuple(value) for name, value in population_values.items()
        }
        self.uses_diameter_parametrization = (
            hasattr(population, "parametrizations")
            and "diam" in population.parametrizations
        )
        self.layout = _prepared_buffer_layout(population)
        self.module_paths = _parameterized_module_paths(population)
        self.geometry_bindings = _geometry_binding_layout(population)
        self.local_geometry_bindings = tuple(
            binding
            for binding in self.geometry_bindings
            if _cache_local_geometry(binding)
        )
        self.derived_module_paths = tuple(
            module_path
            for module_path in self.module_paths
            if _derived_buffer_names(_module_at(population, module_path))
        )
        self.timestep_module_paths = tuple(
            module_path
            for module_path in self.module_paths
            if _timestep_buffer_names(_module_at(population, module_path))
        )
        offset_names = []
        # Preserve the exact effective-buffer baseline at the extraction
        # boundary. A post-initialize hook is allowed to update a raw parameter
        # after normal population has occurred (Tigerholm's balancing hook is a
        # real example), so raw evaluation and the live effective buffer need
        # not initially agree. Express subsequent parameter changes as a smooth
        # delta from that actual baseline instead of changing the model merely
        # by lowering it.
        with torch.no_grad():
            # Evaluate authored/raw preparation against a disposable clone so
            # an invalid one-shot source cannot taint the retained skeleton.
            validation_population = _clone_execution_population(population)
            owner_parameter_values = {}
            for index, entry in enumerate(self.layout):
                if entry.kind in {"derived", "timestep"}:
                    offset_names.append(None)
                    continue
                owner = _module_at(population, entry.module_path)
                evaluation_owner = _module_at(
                    validation_population,
                    entry.module_path,
                )
                # Use the same deterministic raw-to-effective boundary as
                # imperative initialization. In particular, regional
                # additional_parameters must be reconstructed from their
                # explicit raw leaves rather than frozen into this offset.
                if entry.module_path not in owner_parameter_values:
                    bindings = self._registered_input_bindings(validation_population)
                    owner_parameter_values[entry.module_path] = (
                        evaluation_owner._derive_parameter_buffers()
                    )
                    self._assert_registered_inputs_unchanged(
                        bindings,
                        f"{type(evaluation_owner).__qualname__}"
                        "._derive_parameter_buffers()",
                        module=validation_population,
                    )
                evaluated = owner_parameter_values[entry.module_path][entry.buffer_name]
                evaluated = torch.broadcast_to(evaluated, entry.shape)
                offset = owner._buffers[entry.buffer_name] - evaluated
                if bool(torch.count_nonzero(offset).item()):
                    offset_name = f"_effective_offset_{index}"
                    self.register_buffer(offset_name, _clone_tensor(offset))
                else:
                    offset_name = None
                offset_names.append(offset_name)

            # A DERIVED_BUFFER declaration promises that the builder exactly
            # reconstructs the already-initialized live workspace. Failing here
            # prevents lowering a post-initialization mutation or incomplete
            # migration into a silently different transition.
            for module_path in self.derived_module_paths:
                owner = _module_at(validation_population, module_path)
                live_owner = _module_at(population, module_path)
                bindings = self._registered_input_bindings(validation_population)
                values = _evaluate_derived_buffers(owner)
                self._assert_registered_inputs_unchanged(
                    bindings,
                    f"{type(owner).__qualname__}.derive_buffers()",
                    module=validation_population,
                )
                for name in _derived_buffer_names(owner):
                    value = values[name]
                    live = live_owner._buffers[name]
                    if (
                        _shape_tuple(value) != _shape_tuple(live)
                        or value.dtype != live.dtype
                        or value.device != live.device
                        or not torch.equal(value, live)
                    ):
                        raise FunctionalizationError(
                            f"{type(owner).__qualname__}.derive_buffers()[{name!r}] "
                            "does not exactly reconstruct the initialized buffer"
                        )
            dt_tensor = torch.as_tensor(
                self.dt_value,
                device=population.device(),
                dtype=population.dtype(),
            )
            for module_path in self.timestep_module_paths:
                owner = _module_at(validation_population, module_path)
                live_owner = _module_at(population, module_path)
                bindings = self._registered_input_bindings(validation_population)
                values = _evaluate_timestep_buffers(owner, dt_tensor)
                self._assert_registered_inputs_unchanged(
                    bindings,
                    f"{type(owner).__qualname__}.derive_timestep_buffers()",
                    module=validation_population,
                )
                for name in _timestep_buffer_names(owner):
                    value = values[name]
                    live = live_owner._buffers[name]
                    if value.dtype != live.dtype or value.device != live.device:
                        raise FunctionalizationError(
                            f"{type(owner).__qualname__}.derive_timestep_buffers()"
                            f"[{name!r}] must use {live.device}/{live.dtype}"
                        )
                    try:
                        value = torch.broadcast_to(value, _shape_tuple(live))
                    except RuntimeError as exc:
                        raise FunctionalizationError(
                            f"{type(owner).__qualname__}.derive_timestep_buffers()"
                            f"[{name!r}] with shape {_shape_tuple(value)} is not "
                            "broadcastable to the initialized timestep-buffer "
                            f"shape {_shape_tuple(live)}"
                        ) from exc
                    if not torch.equal(value, live):
                        raise FunctionalizationError(
                            f"{type(owner).__qualname__}.derive_timestep_buffers()"
                            f"[{name!r}] does not exactly reconstruct the "
                            "initialized timestep buffer"
                        )
        self.offset_names = tuple(offset_names)
        self.audit_builder_inputs = False

    def _builder_input_bindings(self, celsius, diam, dt):
        if not self.audit_builder_inputs:
            return None
        registered = self._registered_input_bindings()
        positional = {
            "celsius": _audit_tensor_binding(celsius),
            "diam": _audit_tensor_binding(diam),
            "dt": _audit_tensor_binding(dt),
        }
        return registered, positional

    def _registered_input_bindings(self, module=None):
        if not self.audit_builder_inputs:
            return None
        module = self if module is None else module
        return {
            name: _audit_tensor_binding(value)
            for name, value in _module_tensor_slots(module)
        }

    def _assert_registered_inputs_unchanged(self, bindings, hook, *, module=None):
        if bindings is None:
            return
        module = self if module is None else module
        current = dict(_module_tensor_slots(module))
        changed = [
            name
            for name, binding in bindings.items()
            if name not in current or _audit_tensor_binding(current[name]) != binding
        ]
        if set(current) != set(bindings):
            changed.extend(sorted(set(current) ^ set(bindings)))
        if changed:
            raise FunctionalizationError(
                f"{hook} mutated or rebound registered preparation inputs "
                f"{sorted(set(changed))}. Parameter builders must return new "
                "tensors without modifying parameters, constants, or buffers."
            )

    def _assert_builder_inputs_unchanged(self, bindings, celsius, diam, dt, hook):
        if bindings is None:
            return
        registered, positional = bindings
        current = dict(_module_tensor_slots(self))
        changed = [
            name
            for name, binding in registered.items()
            if name not in current or _audit_tensor_binding(current[name]) != binding
        ]
        changed.extend(
            f"positional:{name}"
            for name, value in {"celsius": celsius, "diam": diam, "dt": dt}.items()
            if _audit_tensor_binding(value) != positional[name]
        )
        if set(current) != set(registered):
            changed.extend(sorted(set(current) ^ set(registered)))
        if changed:
            raise FunctionalizationError(
                f"{hook} mutated or rebound registered preparation inputs "
                f"{sorted(set(changed))}. Workspace builders must return new "
                "tensors without modifying parameters, constants, or buffers."
            )

    def forward(self, diam, dx, diameters, diam_original, dt):
        self.population._buffers["dx"] = dx
        if self.uses_diameter_parametrization:
            if diameters is None or diam_original is None:
                raise FunctionalizationError(
                    "A parametrized Population diameter requires explicit "
                    "canonical diameter and parametrization-source inputs."
                )
            self.population._buffers["diameters"] = diameters
            self.population.parametrizations.diam.original = diam_original
        else:
            self.population._buffers["diam"] = diam

        bindings = self._registered_input_bindings()
        population_values = self.population._derive_parameter_buffers()
        self._assert_registered_inputs_unchanged(
            bindings,
            f"{type(self.population).__qualname__}._derive_parameter_buffers()",
        )
        for name in self.population_parameter_names:
            if name in self.population._buffers:
                self.population._buffers[name] = population_values[name]
        celsius = population_values["celsius"]
        effective_diam = population_values.get("diam", diam)

        handler = self.population.mech
        handler._sync_celsius(celsius)
        geometry = {"diam": effective_diam, "dx": dx}
        local_geometry = []
        for binding in self.geometry_bindings:
            value = _geometry_binding_value(binding, geometry)
            for module_path in binding.module_paths:
                _module_at(self.population, module_path)._buffers[
                    binding.buffer_name
                ] = value
            if _cache_local_geometry(binding):
                local_geometry.append(value)

        outputs = []
        owner_parameter_values = {}
        for index, entry in enumerate(self.layout):
            if entry.kind in {"derived", "timestep"}:
                continue
            owner = _module_at(self.population, entry.module_path)
            if entry.module_path not in owner_parameter_values:
                bindings = self._registered_input_bindings()
                owner_parameter_values[entry.module_path] = (
                    owner._derive_parameter_buffers()
                )
                self._assert_registered_inputs_unchanged(
                    bindings,
                    f"{type(owner).__qualname__}._derive_parameter_buffers()",
                )
            value = owner_parameter_values[entry.module_path][entry.buffer_name]
            value = torch.broadcast_to(value, entry.shape)
            offset_name = self.offset_names[index]
            if offset_name is not None:
                value = value + getattr(self, offset_name)
            owner._buffers[entry.buffer_name] = value
            outputs.append((entry.key, value))

        for module_path in self.derived_module_paths:
            owner = _module_at(self.population, module_path)
            bindings = self._builder_input_bindings(celsius, effective_diam, dt)
            values = _evaluate_derived_buffers(owner)
            self._assert_builder_inputs_unchanged(
                bindings,
                celsius,
                effective_diam,
                dt,
                f"{type(owner).__qualname__}.derive_buffers()",
            )
            for entry in self.layout:
                if entry.kind != "derived" or entry.module_path != module_path:
                    continue
                # Builders may return views of explicit dependencies. Keep the
                # prepared workspace independently owned while retaining its
                # autograd connection, matching imperative buffer semantics.
                value = values[entry.buffer_name].clone(
                    memory_format=torch.preserve_format
                )
                owner._buffers[entry.buffer_name] = value
                outputs.append((entry.key, value))

        for module_path in self.timestep_module_paths:
            owner = _module_at(self.population, module_path)
            bindings = self._builder_input_bindings(celsius, effective_diam, dt)
            values = _evaluate_timestep_buffers(owner, dt)
            self._assert_builder_inputs_unchanged(
                bindings,
                celsius,
                effective_diam,
                dt,
                f"{type(owner).__qualname__}.derive_timestep_buffers()",
            )
            for entry in self.layout:
                if entry.kind != "timestep" or entry.module_path != module_path:
                    continue
                value = values[entry.buffer_name]
                if value.device != owner.diam.device or value.dtype != owner.diam.dtype:
                    raise FunctionalizationError(
                        f"{type(owner).__qualname__}.derive_timestep_buffers()"
                        f"[{entry.buffer_name!r}] must use "
                        f"{owner.diam.device}/{owner.diam.dtype}"
                    )
                try:
                    value = torch.broadcast_to(value, entry.shape)
                except RuntimeError as exc:
                    raise FunctionalizationError(
                        f"{type(owner).__qualname__}.derive_timestep_buffers()"
                        f"[{entry.buffer_name!r}] with shape "
                        f"{_shape_tuple(value)} is not broadcastable to canonical "
                        f"timestep-buffer shape {entry.shape}"
                    ) from exc
                value = value.clone(memory_format=torch.preserve_format)
                owner._buffers[entry.buffer_name] = value
                outputs.append((entry.key, value))

        by_key = dict(outputs)
        result = (
            tuple(population_values[name] for name in self.population_parameter_names),
            tuple(by_key[entry.key] for entry in self.layout),
        )
        if self.local_geometry_bindings:
            return (*result, tuple(local_geometry))
        return result


class _ComponentPhysicalPreparation(_PopulationPreparation):
    """Narrow pure physical preparation for one packed scalar component.

    The packed parent owns mechanisms, state, stimulation, and numerical
    execution.  Retaining any of those child runtime trees here would make
    every ``prepare`` clone tensors whose outputs are immediately discarded.
    This adapter therefore registers only the component's root physical
    parameter graph and immutable geometry, plus the tiny Tree ordering maps
    needed to assemble the packed DHS workspace.
    """

    def __init__(
        self,
        population: Population,
        *,
        edge_child_orig: torch.Tensor | None = None,
        solver_order: torch.Tensor | None = None,
    ):
        torch.nn.Module.__init__(self)
        topology_kind = _functional_topology_kind(population)
        if topology_kind not in {
            "single_compartment",
            "unmyelinated",
            "myelinated",
            "native_cable",
            "scalar_tree",
        }:
            raise FunctionalizationError(
                "A packed component physical adapter requires an admitted "
                "scalar Population topology."
            )

        self.topology_kind = topology_kind
        self.audit_builder_inputs = True
        device = population.device()
        authored_classes = _authored_workspace_classes(population)

        if topology_kind == "scalar_tree":
            if edge_child_orig is None or solver_order is None:
                raise FunctionalizationError(
                    "Scalar Tree component preparation requires the packed "
                    "integrator's immutable edge and solver ordering."
                )
            self.register_buffer(
                "edge_child_orig",
                _clone_tensor(edge_child_orig),
                persistent=False,
            )
            self.register_buffer(
                "solver_order",
                _clone_tensor(solver_order),
                persistent=False,
            )

        # The candidate is already an isolated, initialized clone admitted by
        # the scalar functional contract.  Remove the runtime-owned subtrees
        # before registering it here so functional_call can never traverse or
        # clone child mechanism/state/ion/integrator tensors.
        population._modules.pop("integrator", None)
        population._modules.pop("mech", None)
        physical_buffer_names = {
            "diam",
            "diameters",
            "dx",
            *_declared_parameter_names(population),
            *population.in_graph_parametrizations,
        }
        for transforms in population.in_graph_parametrizations.values():
            for _transform, args in transforms:
                physical_buffer_names.update(args)
        for name in tuple(population._buffers):
            if name not in physical_buffer_names:
                population._buffers.pop(name)
        # Population keeps an imperative hot-path alias list outside the
        # registered module tree. Sever it too: otherwise the supposedly
        # physical-only adapter still retains complete Mechanism modules and
        # their live autograd buffers through plain Python references.
        population._m_list = []
        population.injections = []
        population.mechanism_injections = []
        population.mechanism_injection_accepted = []
        population.intra = None
        population.__dict__.pop("_duration_remainder", None)
        population.i_membrane = None
        unregistered_module_paths = _unregistered_module_alias_paths(population)
        if unregistered_module_paths:
            raise FunctionalizationError(
                "Packed component physical preparation retained unregistered "
                "runtime module aliases at "
                f"{list(unregistered_module_paths)}."
            )
        self.population = population
        leaked_runtime_parameters = tuple(
            name
            for name, _parameter in population.named_parameters()
            if name.startswith(("integrator.", "mech."))
        )
        if leaked_runtime_parameters:  # pragma: no cover - defensive invariant
            raise FunctionalizationError(
                "Packed component physical preparation retained runtime "
                f"parameters {leaked_runtime_parameters}."
            )

        self.uses_diameter_parametrization = (
            hasattr(population, "parametrizations")
            and "diam" in population.parametrizations
        )
        self.parametrization_constants = _parametrization_constant_layout_for_owners(
            population, (population,)
        )

        # Constructor-time authored evaluation is audited on the disposable
        # component clone.  If it is impure, failure cannot contaminate the
        # user's source or the packed execution skeleton.
        registered = self._registered_input_bindings()
        python_state = _transition_python_state_snapshot(self)
        rng_state = _global_rng_snapshot(device)
        class_state = _class_state_snapshot(authored_classes)
        mutated_class_state = set()
        mutated_python_state = set()
        try:
            with torch.no_grad():
                population_values = population._derive_parameter_buffers()
            self._assert_registered_inputs_unchanged(
                registered,
                f"{type(population).__qualname__}._derive_parameter_buffers()",
            )
            current_python_state = _transition_python_state_snapshot(self)
            mutated_python_state = {
                name
                for name in set(python_state) | set(current_python_state)
                if python_state.get(name) != current_python_state.get(name)
            }
        finally:
            try:
                mutated_class_state.update(_restore_class_state(class_state))
            finally:
                consumed_rng = _global_rng_changes(rng_state, device)
                _restore_global_rng(rng_state, device)
        if mutated_python_state:
            raise FunctionalizationError(
                "Packed component physical preparation observed authored "
                "mutation of unregistered Python instance state "
                f"{sorted(mutated_python_state)}."
            )
        if mutated_class_state:
            raise FunctionalizationError(
                "Packed component physical preparation observed authored "
                "mutation of shared Python class state "
                f"{sorted(mutated_class_state)}."
            )
        if consumed_rng:
            raise FunctionalizationError(
                "Packed component physical preparation observed authored "
                f"consumption of {sorted(consumed_rng)} RNG state."
            )
        if not isinstance(population_values, Mapping) or any(
            not torch.is_tensor(value) for value in population_values.values()
        ):
            raise FunctionalizationError(
                f"{type(population).__qualname__}._derive_parameter_buffers() "
                "must return a tensor mapping."
            )
        self.population_parameter_names = tuple(population_values)
        self.audit_builder_inputs = False

    def forward(self, diam, dx, diameters, diam_original):
        """Materialize only effective component-level physical parameters."""
        self.population._buffers["dx"] = dx
        if self.uses_diameter_parametrization:
            if diameters is None or diam_original is None:
                raise FunctionalizationError(
                    "A parametrized packed component diameter requires explicit "
                    "canonical diameter and parametrization-source inputs."
                )
            self.population._buffers["diameters"] = diameters
            self.population.parametrizations.diam.original = diam_original
        else:
            self.population._buffers["diam"] = diam

        bindings = self._registered_input_bindings()
        population_values = self.population._derive_parameter_buffers()
        self._assert_registered_inputs_unchanged(
            bindings,
            f"{type(self.population).__qualname__}._derive_parameter_buffers()",
        )
        return tuple(
            population_values[name] for name in self.population_parameter_names
        )


_TRUSTED_INITIALIZATION_MODULE_PREFIXES = (
    "torch",
    "operator",
    "_operator",
    "dendra.models.mechanisms.ops",
)
_FORBIDDEN_INITIALIZATION_PYTHON_ESCAPES = frozenset(
    {
        "bool",
        "complex",
        "float",
        "getattr",
        "hasattr",
        "int",
        "item",
        "numpy",
        "tolist",
        "type",
    }
)
_FORBIDDEN_INITIALIZATION_CONTEXT_READS = frozenset(
    {
        "_are_functorch_transforms_active",
        "__class__",
        "__delattr__",
        "__dict__",
        "__setattr__",
        "_current_level",
        "get_autocast_dtype",
        "is_anomaly_check_nan_enabled",
        "is_anomaly_enabled",
        "is_autocast_cache_enabled",
        "is_autocast_enabled",
        "is_compiling",
        "is_deterministic_algorithms_warn_only_enabled",
        "are_deterministic_algorithms_enabled",
        "is_grad_enabled",
        "is_inference_mode_enabled",
        "is_scripting",
        "is_tracing",
        "requires_grad",
        "tangent",
        "unpack_dual",
    }
)
_FORBIDDEN_INITIALIZATION_INPLACE_METHODS = frozenset(
    {
        "add_",
        "copy_",
        "div_",
        "fill_",
        "index_add_",
        "masked_fill_",
        "mul_",
        "resize_",
        "scatter_add_",
        "set_",
        "sub_",
        "zero_",
    }
)
_FORBIDDEN_INITIALIZATION_EFFECTFUL_ROUTINES = frozenset(
    {
        "autocast",
        "compile",
        "delitem",
        "enable_grad",
        "export",
        "inference_mode",
        "load",
        "manual_seed",
        "no_grad",
        "record_function",
        "save",
        "seed",
        "set_grad_enabled",
        "setitem",
        "use_deterministic_algorithms",
    }
)
_SAFE_INITIALIZATION_BUILTIN_ROUTINES = frozenset(
    {
        "abs",
        "all",
        "any",
        "divmod",
        "isinstance",
        "issubclass",
        "len",
        "max",
        "min",
        "pow",
        "repr",
        "round",
        "sorted",
        "sum",
    }
)
_TRUSTED_INITIALIZATION_FUNCTION_ATTRIBUTES = frozenset(
    {
        "__generated_filename__",
        "__source__",
        "_dendra_conductance_fallback_reason",
        "_dendra_conductance_mode",
        "_dendra_monomorphic_advance_states",
        "_dendra_monomorphic_advance_signature",
        "_dendra_monomorphic_advance_source",
        "factorable",
    }
)


def _trusted_initialization_module(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in _TRUSTED_INITIALIZATION_MODULE_PREFIXES
    )


def _forbidden_initialization_callable_name(value) -> str | None:
    """Resolve context-query aliases that bytecode name checks cannot see."""

    target = getattr(value, "__func__", value)
    names = {
        getattr(target, "__name__", None),
        getattr(target, "__qualname__", None),
    }
    names.update(
        name.rsplit(".", 1)[-1] for name in tuple(names) if isinstance(name, str)
    )
    forbidden = (
        _FORBIDDEN_INITIALIZATION_CONTEXT_READS
        | _FORBIDDEN_INITIALIZATION_EFFECTFUL_ROUTINES
        | _FORBIDDEN_INITIALIZATION_PYTHON_ESCAPES
        | {"item", "numpy", "tolist"}
    )
    return next(
        (
            name
            for name in sorted(names - {None})
            if name in forbidden or name.lstrip("_").startswith("set_")
        ),
        None,
    )


def _unsupported_initialization_dependency(
    value,
    *,
    label: str,
    reason: str,
    strict: bool,
):
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if strict:
        raise FunctionalizationError(
            f"Pure functional-initialization dependency {label!r} is {reason} "
            f"({kind}). "
            "Move numerical dependencies to declared parameters/buffers and "
            "keep external configuration immutable."
        )
    return ("unsupported", reason, kind)


def _initialization_dependency_value_signature(
    value,
    *,
    label: str,
    owner: State | Mechanism,
    strict: bool,
    seen: set[int],
    forbidden_receiver_attributes: frozenset[str] = frozenset(),
):
    """Fingerprint one resolved initialization dependency or fail closed."""

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return (type(value).__qualname__, repr(value))
    if value is Ellipsis or value is NotImplemented:
        return (type(value).__qualname__, repr(value))
    if isinstance(value, (torch.dtype, torch.device)):
        return (type(value).__qualname__, str(value))
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, range):
        return ("range", value.start, value.stop, value.step)
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return (
            type(value).__qualname__,
            str(scalar.dtype),
            hashlib.sha256(scalar.tobytes()).hexdigest(),
        )
    if isinstance(value, tuple):
        return (
            type(value).__qualname__,
            tuple(
                _initialization_dependency_value_signature(
                    item,
                    label=f"{label}[{index}]",
                    owner=owner,
                    strict=strict,
                    seen=seen,
                    forbidden_receiver_attributes=forbidden_receiver_attributes,
                )
                for index, item in enumerate(value)
            ),
        )
    if isinstance(value, frozenset):
        items = [
            _initialization_dependency_value_signature(
                item,
                label=f"{label}[]",
                owner=owner,
                strict=strict,
                seen=seen,
                forbidden_receiver_attributes=forbidden_receiver_attributes,
            )
            for item in value
        ]
        return ("frozenset", tuple(sorted(items, key=repr)))
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return _unsupported_initialization_dependency(
            value,
            label=label,
            reason="hidden mutable numerical state",
            strict=strict,
        )
    if isinstance(value, (list, set, dict, torch.Generator)):
        return _unsupported_initialization_dependency(
            value,
            label=label,
            reason="mutable external Python state",
            strict=strict,
        )
    if isinstance(value, types.ModuleType):
        module_name = value.__name__
        if not _trusted_initialization_module(module_name):
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason="an untrusted mutable module namespace",
                strict=strict,
            )
        return ("module", module_name, id(value))
    if inspect.isclass(value):
        forbidden_name = _forbidden_initialization_callable_name(value)
        if forbidden_name is not None:
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason=(
                    "an aliased tensor-to-Python or dynamic-dispatch routine "
                    f"({forbidden_name})"
                ),
                strict=strict,
            )
        if value.__module__ != "builtins" and not _trusted_initialization_module(
            value.__module__
        ):
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason="an authored class with hidden external state",
                strict=strict,
            )
        return ("class", value.__module__, value.__qualname__, id(value))
    if inspect.isfunction(value) or inspect.ismethod(value):
        bound_receiver = value.__self__ if inspect.ismethod(value) else None
        if (
            bound_receiver is not None
            and bound_receiver is not owner
            and bound_receiver is not type(owner)
        ):
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason="a bound method with hidden external receiver state",
                strict=strict,
            )
        forbidden_name = _forbidden_initialization_callable_name(value)
        if forbidden_name is not None:
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason=(
                    "an aliased tensor-to-Python or execution-context routine "
                    f"({forbidden_name})"
                ),
                strict=strict,
            )
        return _initialization_callable_dependency_signature(
            value,
            label=label,
            owner=owner,
            strict=strict,
            seen=seen,
            forbidden_receiver_attributes=forbidden_receiver_attributes,
        )
    if inspect.isbuiltin(value) or inspect.ismethoddescriptor(value):
        module_name = getattr(value, "__module__", None)
        routine_name = getattr(
            value,
            "__name__",
            getattr(value, "__qualname__", None),
        )
        forbidden_name = _forbidden_initialization_callable_name(value)
        if forbidden_name is not None:
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason=(
                    "an aliased tensor-to-Python or execution-context routine "
                    f"({forbidden_name})"
                ),
                strict=strict,
            )
        bound_receiver = getattr(value, "__self__", None)
        receiver_is_trusted_module = isinstance(
            bound_receiver,
            types.ModuleType,
        ) and (
            bound_receiver.__name__ == "builtins"
            or _trusted_initialization_module(bound_receiver.__name__)
        )
        if (
            bound_receiver is not None
            and bound_receiver is not owner
            and bound_receiver is not type(owner)
            and not receiver_is_trusted_module
        ):
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason="a bound routine with hidden external receiver state",
                strict=strict,
            )
        if (
            module_name == "builtins"
            and routine_name not in _SAFE_INITIALIZATION_BUILTIN_ROUTINES
        ):
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason="an effectful or unsupported builtin routine",
                strict=strict,
            )
        if module_name not in {None, "builtins"} and not _trusted_initialization_module(
            module_name
        ):
            return _unsupported_initialization_dependency(
                value,
                label=label,
                reason="an untrusted opaque routine",
                strict=strict,
            )
        return (
            "builtin-routine",
            module_name,
            getattr(value, "__qualname__", routine_name),
            id(value),
        )
    if callable(value):
        return _unsupported_initialization_dependency(
            value,
            label=label,
            reason="a callable object with hidden state",
            strict=strict,
        )
    return _unsupported_initialization_dependency(
        value,
        label=label,
        reason="unsupported opaque external state",
        strict=strict,
    )


def _initialization_receiver_dependency_signature(
    owner: State | Mechanism,
    name: str,
    *,
    label: str,
    strict: bool,
    seen: set[int],
    forbidden_receiver_attributes: frozenset[str] = frozenset(),
):
    """Fingerprint one attribute loaded directly from an authored ``self``."""

    if name in owner._parameters:
        return ("registered-parameter", name)
    if name in owner._buffers:
        return ("registered-buffer", name)
    if name in owner._modules:
        module = owner._modules[name]
        return _unsupported_initialization_dependency(
            module,
            label=f"{label}.{name}",
            reason=(
                "a registered child module whose nested callable dependencies "
                "cannot yet be proven"
            ),
            strict=strict,
        )
    if name == "t" and isinstance(owner, Mechanism):
        # Population.build installs this framework-owned live clock reference;
        # functional_call rebinds it to the explicit clock state.
        return ("explicit-population-clock", name)

    if name in owner.__dict__:
        value = owner.__dict__[name]
    else:
        sentinel = object()
        value = sentinel
        for cls in type(owner).__mro__:
            if name in cls.__dict__:
                value = cls.__dict__[name]
                break
        if value is sentinel:
            return _unsupported_initialization_dependency(
                owner,
                label=f"{label}.self.{name}",
                reason="an unresolved receiver attribute",
                strict=strict,
            )

    if isinstance(value, staticmethod):
        value = value.__get__(owner, type(owner))
    elif isinstance(value, classmethod):
        value = value.__get__(owner, type(owner))
    elif isinstance(value, property):
        value = value.fget.__get__(owner, type(owner))
    elif inspect.isfunction(value):
        value = value.__get__(owner, type(owner))
    return _initialization_dependency_value_signature(
        value,
        label=f"{label}.self.{name}",
        owner=owner,
        strict=strict,
        seen=seen,
        forbidden_receiver_attributes=forbidden_receiver_attributes,
    )


def _direct_receiver_attribute_names(
    target,
    receiver_name: str | None,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return direct receiver attributes and any opaque receiver escapes."""

    if receiver_name is None:
        return frozenset(), ()
    names = set()
    escapes = set()
    instructions = tuple(dis.get_instructions(target))
    for index, instruction in enumerate(instructions[:-1]):
        if instruction.opname != "LOAD_FAST" or instruction.argval != receiver_name:
            continue
        following = instructions[index + 1]
        if following.opname in {"LOAD_ATTR", "LOAD_METHOD"}:
            names.add(following.argval)
        elif following.opname != "STORE_ATTR":
            escapes.add(following.opname)
    return frozenset(names), tuple(sorted(escapes))


def _unsafe_initialization_operations(
    target,
    receiver_name: str | None,
) -> tuple[str, ...]:
    """Find explicit Python escapes, context reads, and in-place method calls."""

    unsafe = set()
    instructions = tuple(dis.get_instructions(target))
    for index, instruction in enumerate(instructions):
        if instruction.opname == "STORE_ATTR":
            previous = instructions[index - 1] if index else None
            direct_receiver_store = receiver_name is not None and (
                (previous.opname == "LOAD_FAST" and previous.argval == receiver_name)
                or (
                    previous.opname == "LOAD_FAST_LOAD_FAST"
                    and isinstance(previous.argval, tuple)
                    and previous.argval[-1] == receiver_name
                )
            )
            if not direct_receiver_store:
                unsafe.add(f"attribute write {instruction.argval}")
            continue
        if instruction.opname in {
            "DELETE_ATTR",
            "DELETE_DEREF",
            "DELETE_GLOBAL",
            "DELETE_SUBSCR",
            "STORE_DEREF",
            "STORE_GLOBAL",
            "STORE_SUBSCR",
        }:
            unsafe.add(instruction.opname.lower().replace("_", " "))
            continue
        if instruction.opname == "MAKE_FUNCTION":
            unsafe.add("nested function or comprehension")
            continue
        if (
            instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
            and instruction.argval in _FORBIDDEN_INITIALIZATION_PYTHON_ESCAPES
        ):
            unsafe.add(instruction.argval)
        if instruction.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
            continue
        name = instruction.argval
        if name in (
            _FORBIDDEN_INITIALIZATION_CONTEXT_READS
            | _FORBIDDEN_INITIALIZATION_EFFECTFUL_ROUTINES
            | {"item", "numpy", "tolist"}
            | _FORBIDDEN_INITIALIZATION_INPLACE_METHODS
        ) or (isinstance(name, str) and name.endswith("_") and not name.endswith("__")):
            unsafe.add(name)
        if isinstance(name, str) and name.lstrip("_").startswith("set_"):
            unsafe.add(name)
    return tuple(sorted(unsafe))


@cache
def _canonical_numerical_current_adapter(
    current_name: str,
    assign: bool,
) -> tuple[str, str, tuple]:
    """Return the exact framework-generated numerical-current provenance."""

    source = build_numerical_equation(current_name, assign)
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    filename = f"<dendra.mechanisms.numerical:{digest}>"
    module_code = compile(source, filename, "exec", dont_inherit=True)
    function_codes = tuple(
        value
        for value in module_code.co_consts
        if inspect.iscode(value) and value.co_name == current_name
    )
    if len(function_codes) != 1:  # pragma: no cover - fixed framework template
        raise RuntimeError(
            "Dendra's numerical-current template did not produce exactly one "
            f"function named {current_name!r}"
        )
    return (
        source,
        filename,
        _transform_code_structure_signature(
            function_codes[0],
            label=f"canonical numerical current {current_name}.__code__",
        ),
    )


def _initialization_callable_dependency_signature(
    value,
    *,
    label: str,
    owner: State | Mechanism,
    strict: bool,
    seen: set[int] | None = None,
    include_identity: bool = True,
    forbidden_receiver_attributes: frozenset[str] = frozenset(),
):
    """Fingerprint live globals, closures, and helper methods used by a hook."""

    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    bound_receiver = value.__self__ if inspect.ismethod(value) else None
    target = getattr(value, "__func__", value)

    generated_filename = getattr(target, "__generated_filename__", None)
    conductance_mode = getattr(target, "_dendra_conductance_mode", None)
    if (
        bound_receiver is owner
        and target.__module__ == "dendra.models.mechanisms._symbolic"
        and isinstance(generated_filename, str)
        and generated_filename.startswith("<dendra.mechanisms.numerical:")
        and conductance_mode == "numerical-declared"
        and getattr(target, "_dendra_conductance_fallback_reason", None) is None
    ):
        current_name = target.__name__
        wrapper = getattr(owner, f"{current_name}_with_g", None)
        authored_current = getattr(owner, current_name, None)
        if (
            getattr(wrapper, "__func__", wrapper) is target
            and callable(authored_current)
            and getattr(authored_current, "__func__", authored_current) is not target
        ):
            expected_source, expected_filename, expected_code = (
                _canonical_numerical_current_adapter(
                    current_name,
                    current_name in owner._save,
                )
            )
            actual_code = _transform_code_structure_signature(
                target.__code__,
                label=f"{label}.__code__",
            )
            globals_are_canonical = (
                target.__globals__.get("torch") is torch
                and target.__globals__.get("validate_declared_numerical_current")
                is _validate_declared_numerical_current
                and target.__globals__.get("__builtins__") is builtins.__dict__
            )
            if (
                getattr(target, "__source__", None) != expected_source
                or generated_filename != expected_filename
                or actual_code != expected_code
                or not globals_are_canonical
            ):
                return _unsupported_initialization_dependency(
                    target,
                    label=label,
                    reason=(
                        "a generated numerical-current adapter whose source, "
                        "code, or framework validator bindings are no longer "
                        "canonical"
                    ),
                    strict=strict,
                )
            # The generated adapter's context branch controls only Dendra's
            # eager numerical-current validator; its tensor result is the fixed
            # centred-difference expression captured in the generated code.
            # Fingerprint that code and recursively audit the authored current
            # rather than treating framework-owned ``is_compiling``/``type``
            # diagnostics as authored transform-dependent behavior.
            return (
                "dendra-generated-numerical-current",
                generated_filename,
                actual_code,
                expected_source,
                _initialization_callable_dependency_signature(
                    authored_current,
                    label=f"{label}.authored_current",
                    owner=owner,
                    strict=strict,
                    seen=seen,
                    forbidden_receiver_attributes=forbidden_receiver_attributes,
                ),
            )
    forbidden_name = _forbidden_initialization_callable_name(target)
    if forbidden_name is not None:
        return _unsupported_initialization_dependency(
            target,
            label=label,
            reason=(
                "an aliased tensor-to-Python or execution-context routine "
                f"({forbidden_name})"
            ),
            strict=strict,
        )
    if not inspect.isfunction(target):
        return _initialization_dependency_value_signature(
            target,
            label=label,
            owner=owner,
            strict=strict,
            seen=set() if seen is None else seen,
            forbidden_receiver_attributes=forbidden_receiver_attributes,
        )
    receiver_name = None
    if bound_receiver is owner or bound_receiver is type(owner):
        if target.__code__.co_argcount:
            receiver_name = target.__code__.co_varnames[0]

    seen = set() if seen is None else seen
    identity = id(target)
    if identity in seen:
        return (
            "recursive-routine",
            target.__module__,
            target.__qualname__,
            identity,
        )
    seen.add(identity)
    try:
        unsafe_operations = _unsafe_initialization_operations(
            target,
            receiver_name,
        )
        if unsafe_operations and strict:
            raise FunctionalizationError(
                f"Pure functional-initialization dependency {label!r} uses "
                "tensor-to-Python, dynamic-dispatch, execution-context, or "
                "in-place operations "
                f"{list(unsafe_operations)}. Functional "
                "initialization must remain tensor-native and read-only under "
                "torch.func transforms."
            )
        defaults = _initialization_dependency_value_signature(
            target.__defaults__,
            label=f"{label}.__defaults__",
            owner=owner,
            strict=strict,
            seen=seen,
            forbidden_receiver_attributes=forbidden_receiver_attributes,
        )
        kwdefaults = target.__kwdefaults__
        kwdefault_signature = (
            None
            if kwdefaults is None
            else tuple(
                (
                    name,
                    _initialization_dependency_value_signature(
                        item,
                        label=f"{label}.__kwdefaults__.{name}",
                        owner=owner,
                        strict=strict,
                        seen=seen,
                        forbidden_receiver_attributes=forbidden_receiver_attributes,
                    ),
                )
                for name, item in sorted(kwdefaults.items())
            )
        )
        function_attributes = tuple(
            (
                name,
                (
                    _initialization_dependency_value_signature(
                        item,
                        label=f"{label}.__dict__.{name}",
                        owner=owner,
                        strict=strict,
                        seen=seen,
                        forbidden_receiver_attributes=forbidden_receiver_attributes,
                    )
                    if name in _TRUSTED_INITIALIZATION_FUNCTION_ATTRIBUTES
                    else _unsupported_initialization_dependency(
                        target,
                        label=f"{label}.__dict__.{name}",
                        reason="mutable authored function-object state",
                        strict=strict,
                    )
                ),
            )
            for name, item in sorted(target.__dict__.items())
        )
        try:
            closure = inspect.getclosurevars(target)
        except (TypeError, ValueError) as exc:
            if strict:
                raise FunctionalizationError(
                    f"Pure functional-initialization dependency {label!r} could "
                    "not be "
                    f"inspected safely: {exc}"
                ) from exc
            closure_signature = (("unsupported-inspection", type(exc).__name__),)
        else:
            closure_signature = tuple(
                (
                    scope,
                    name,
                    _initialization_dependency_value_signature(
                        item,
                        label=f"{label}.{scope}.{name}",
                        owner=owner,
                        strict=strict,
                        seen=seen,
                        forbidden_receiver_attributes=forbidden_receiver_attributes,
                    ),
                )
                for scope, values in (
                    ("nonlocal", closure.nonlocals),
                    ("global", closure.globals),
                    ("builtin", closure.builtins),
                )
                for name, item in sorted(values.items())
            )

        receiver_signatures = []
        self_attributes, receiver_escapes = _direct_receiver_attribute_names(
            target,
            receiver_name,
        )
        forbidden_receiver_reads = tuple(
            sorted(self_attributes & forbidden_receiver_attributes)
        )
        if forbidden_receiver_reads and strict:
            raise FunctionalizationError(
                f"Pure functional-initialization dependency {label!r} reads "
                "receiver attributes that are not transform-safe in this "
                f"context {list(forbidden_receiver_reads)}. Read their explicit "
                "support-visible values from the hook's explicit values mapping."
            )
        if receiver_escapes and strict:
            raise FunctionalizationError(
                f"Pure functional-initialization dependency {label!r} lets its "
                "bound receiver escape direct attribute access through "
                f"bytecode operations {list(receiver_escapes)}. Keep receiver "
                "dependencies directly resolvable by the initializer audit."
            )
        for name in sorted(self_attributes):
            receiver_signatures.append(
                _initialization_receiver_dependency_signature(
                    owner,
                    name,
                    label=label,
                    strict=strict,
                    seen=seen,
                    forbidden_receiver_attributes=forbidden_receiver_attributes,
                )
            )
        return (
            "python-routine",
            target.__module__,
            target.__qualname__,
            identity if include_identity else None,
            _transform_code_structure_signature(
                target.__code__,
                label=f"{label}.__code__",
            ),
            defaults,
            kwdefault_signature,
            function_attributes,
            closure_signature,
            tuple(zip(sorted(self_attributes), receiver_signatures, strict=True)),
            receiver_escapes,
            forbidden_receiver_reads,
            unsafe_operations,
        )
    finally:
        seen.remove(identity)


def _initialization_final_frame_callable_plan(
    population: Population,
) -> tuple[tuple[str, State | Mechanism, object], ...]:
    """Return authored callables reached by the final initialization frame."""

    handler = population.mech
    assigned_mechanisms = {
        id(mechanism) for mechanism, _support_index in handler._current_assigned_plan
    }
    assigned_mechanisms.update(
        id(mechanism) for mechanism, _support_index in handler._post_state_advance_plan
    )
    current_callables = {}
    for _category, _support_index, _support_entry, entries in handler._map_grouped:
        for mechanism, function_name, _scale, _factorable in entries:
            current_callables.setdefault(id(mechanism), {})[function_name] = getattr(
                mechanism,
                function_name,
            )

    plan = []
    for mechanism_name, mechanism in handler.mechanisms.items():
        if id(mechanism) in assigned_mechanisms and (
            "assigned_values" in mechanism.__dict__
            or inspect.getattr_static(type(mechanism), "assigned_values")
            is not _STANDARD_MECHANISM_ASSIGNED_VALUES
        ):
            plan.append(
                (
                    f"{mechanism_name}.assigned_values",
                    mechanism,
                    getattr(mechanism, "assigned_values"),
                )
            )
        for function_name, callable_value in sorted(
            current_callables.get(id(mechanism), {}).items()
        ):
            plan.append(
                (
                    f"{mechanism_name}.{function_name}",
                    mechanism,
                    callable_value,
                )
            )
    return tuple(plan)


def _initialization_final_frame_dependency_signature(
    population: Population,
    *,
    strict: bool,
) -> tuple:
    return tuple(
        (
            label,
            _initialization_callable_dependency_signature(
                callable_value,
                label=label,
                owner=owner,
                strict=strict,
                include_identity=not bool(
                    getattr(
                        getattr(callable_value, "__func__", callable_value),
                        "__generated_filename__",
                        None,
                    )
                ),
            ),
        )
        for label, owner, callable_value in _initialization_final_frame_callable_plan(
            population
        )
    )


def _functional_initialization_transform_failures(
    population: Population,
) -> list[str]:
    """Validate the strict tensor-only Population transform contract."""

    failures = []
    legacy_pre = [
        hook
        for hook in population.pre_initialize_hooks
        if not isinstance(hook, _InitializationTransformHook)
    ]
    legacy_post = [
        hook
        for hook in population.post_initialize_hooks
        if not isinstance(hook, _InitializationTransformHook)
    ]
    if legacy_pre or legacy_post:
        failures.append(
            "arbitrary Population pre/post-initialize hooks remain imperative-only; "
            "register a pure initialization transform with explicit tensor "
            "reads, writes, and inputs"
        )

    parameter_refs = {
        f"parameters.{name}" for name, _value in population.named_parameters()
    }
    try:
        state_layout = _state_layout(population)
        explicit_layout = _initialization_state_layout(population, state_layout)
    except FunctionalizationError as exc:
        failures.append(str(exc))
        return failures
    all_state_refs = {f"state.{'.'.join(leaf.public_path)}" for leaf in state_layout}
    pre_state_refs = {
        "state.integrator.v",
        *(f"state.{'.'.join(entry.public_path)}" for entry in explicit_layout),
    }
    allowed_by_phase = {
        "pre": parameter_refs | pre_state_refs,
        "post": parameter_refs | all_state_refs,
    }

    seen_actions = set()
    for phase in ("pre", "post"):
        for hook in _initialization_transform_hooks(population, phase):
            action_name = hook.action_name
            key = (phase, action_name)
            if key in seen_actions:
                failures.append(
                    f"{phase} initialization transform {action_name!r} is "
                    "registered more than once"
                )
                continue
            seen_actions.add(key)
            if action_name not in population._initialization_transforms:
                failures.append(
                    f"{phase} initialization transform {action_name!r} lost its "
                    "registered Module"
                )
                continue
            action = population._initialization_transforms[action_name]
            if action.phase != phase or action.name != action_name:
                failures.append(
                    f"initialization transform {action_name!r} phase/name metadata "
                    "does not match its hook registration"
                )
            if len(set(action.writes)) != len(action.writes):
                failures.append(
                    f"initialization transform {action_name!r} declares duplicate "
                    "output paths"
                )
            for role, refs in (("read", action.reads), ("write", action.writes)):
                invalid = [ref for ref in refs if ref not in allowed_by_phase[phase]]
                if invalid:
                    failures.append(
                        f"{phase} initialization transform {action_name!r} has "
                        f"unsupported {role} paths {invalid}; allowed paths are "
                        "canonical raw parameters and phase-available explicit state"
                    )
            transform = action.transform
            hidden_tensors = tuple(transform.named_parameters()) + tuple(
                transform.named_buffers()
            )
            hidden_modules = tuple(transform.named_children())
            if hidden_tensors or hidden_modules:
                failures.append(
                    f"initialization transform {action_name!r} must be a stateless "
                    "Module; pass every tensor through its registered explicit "
                    "inputs and compose helper operations in forward()"
                )
            try:
                _initialization_callable_dependency_signature(
                    transform.forward,
                    label=f"{phase} initialization transform {action_name}",
                    owner=transform,
                    strict=True,
                    include_identity=False,
                )
            except FunctionalizationError as exc:
                failures.append(str(exc))

    registered_names = set(getattr(population, "_initialization_transforms", {}))
    hooked_names = {name for _phase, name in seen_actions}
    orphaned = sorted(registered_names - hooked_names)
    if orphaned:
        failures.append(
            "registered initialization transform Modules are not present exactly "
            f"once in a pre/post hook sequence: {orphaned}"
        )
    try:
        _initialization_transform_input_layout(population)
    except FunctionalizationError as exc:
        failures.append(str(exc))
    return failures


def _registered_child_value_signature(owner, name: str) -> tuple:
    """Describe one registered child slot without authored attribute dispatch."""
    value = owner._parameters.get(name)
    category = "parameter"
    if value is None:
        value = owner._modules.get(name)
        category = "module"
    if value is None:
        value = owner._buffers.get(name)
        category = "buffer"
    if value is None:
        return ("missing",)
    return (category, type(value).__module__, type(value).__qualname__)


def _handler_initialization_hook_is_standard(handler, name: str) -> bool:
    """Recognize canonical handler hooks, including framework Dynamo wrappers."""

    standard = _STANDARD_HANDLER_INITIALIZATION[name]
    if inspect.getattr_static(type(handler), name) is not standard:
        return False
    if name not in handler.__dict__:
        return True
    if name not in _HANDLER_SYNC_WRAPPER_HOOKS:
        return False
    wrapper = handler.__dict__[name]
    wrapped = getattr(wrapper, "__wrapped__", None)
    return bool(
        getattr(wrapper, "_torchdynamo_disable", False)
        and isinstance(wrapped, types.MethodType)
        and wrapped.__self__ is handler
        and wrapped.__func__ is standard
    )


def _handler_initialization_hook_signature(handler, name: str) -> tuple:
    """Fingerprint canonical wrappers without retaining wrapper object identity."""

    if _handler_initialization_hook_is_standard(handler, name):
        return (
            "canonical",
            MechanismHandler.__module__,
            MechanismHandler.__qualname__,
            name,
            "dynamo-disabled" if name in handler.__dict__ else "class",
        )
    return _effective_hook_identity(handler, name)


def _initialization_structure_signature(population: Population) -> tuple:
    """Fingerprint only structure that affects fresh initialization."""

    population_hooks = (
        type(population).__module__,
        type(population).__qualname__,
        tuple(
            (name, _effective_hook_identity(population, name))
            for name in _POPULATION_FRESH_INITIALIZATION_HOOKS
        ),
    )
    integrator = population.integrator
    integrator_hooks = (
        type(integrator).__module__,
        type(integrator).__qualname__,
        tuple(
            (name, _effective_hook_identity(integrator, name))
            for name in _INTEGRATOR_FRESH_INITIALIZATION_HOOKS
        ),
    )
    handler = population.mech
    handler_hooks = (
        type(handler).__module__,
        type(handler).__qualname__,
        tuple(
            (name, _handler_initialization_hook_signature(handler, name))
            for name in _HANDLER_INITIALIZATION_HOOKS
        ),
    )
    shared_local_names = {
        id(mechanism): set(mechanism._runtime_shared_local_names())
        for mechanism in handler.mechanisms.values()
    }
    mechanisms = []
    for mechanism_name, mechanism in population.mech.mechanisms.items():
        declared_names = {
            name for state in mechanism.DE.values() for name in state._state
        }
        states = []
        for state_name, state in mechanism.DE.items():
            missing = set(state._state) - set(mechanism._init_params)
            default_dependencies = (
                None
                if (
                    not missing
                    or (
                        inspect.getattr_static(type(state), "state_defaults")
                        is _STANDARD_STATE_DEFAULTS
                        and "state_defaults" not in state.__dict__
                    )
                )
                else _initialization_callable_dependency_signature(
                    getattr(state, "state_defaults"),
                    label=f"{mechanism_name}.{state_name}.state_defaults",
                    owner=state,
                    strict=False,
                )
            )
            state_initial_values = getattr(state, "initial_values")
            state_initial_value_dependencies = (
                None
                if inspect.getattr_static(type(state), "initial_values")
                is _STANDARD_STATE_INITIAL_VALUES
                and "initial_values" not in state.__dict__
                else _initialization_callable_dependency_signature(
                    state_initial_values,
                    label=f"{mechanism_name}.{state_name}.initial_values",
                    owner=state,
                    strict=False,
                    forbidden_receiver_attributes=frozenset(
                        {"celsius"} | shared_local_names.get(id(mechanism), set())
                    ),
                )
            )
            states.append(
                (
                    state_name,
                    type(state).__module__,
                    type(state).__qualname__,
                    tuple(state._state),
                    tuple(
                        (name, _effective_hook_identity(state, name))
                        for name in (
                            "state_defaults",
                            "initial_values",
                            *_STATE_FRESH_INITIALIZATION_HOOKS,
                        )
                    ),
                    default_dependencies,
                    state_initial_value_dependencies,
                )
            )
        mechanisms.append(
            (
                mechanism_name,
                type(mechanism).__module__,
                type(mechanism).__qualname__,
                tuple(
                    (name, _effective_hook_identity(mechanism, name))
                    for name in _STANDARD_MECHANISM_INITIALIZATION
                ),
                (
                    (
                        "initial_values",
                        _effective_hook_identity(mechanism, "initial_values"),
                    ),
                ),
                (
                    None
                    if inspect.getattr_static(type(mechanism), "initial_values")
                    is _STANDARD_MECHANISM_INITIAL_VALUES
                    and "initial_values" not in mechanism.__dict__
                    else _initialization_callable_dependency_signature(
                        getattr(mechanism, "initial_values"),
                        label=f"{mechanism_name}.initial_values",
                        owner=mechanism,
                        strict=False,
                        forbidden_receiver_attributes=frozenset(
                            {"celsius"} | shared_local_names.get(id(mechanism), set())
                        ),
                    )
                ),
                tuple(
                    sorted(
                        name
                        for name in mechanism._init_params
                        if name in declared_names
                    )
                ),
                tuple(states),
            )
        )
    shared_fields = (
        tuple(
            (
                name,
                type(ion).__module__,
                type(ion).__qualname__,
                tuple(ion.fields),
                tuple(
                    (hook, _effective_hook_identity(ion, hook))
                    for hook in _ION_INITIALIZATION_HOOKS
                )
                + tuple(
                    (
                        f"Material.{hook}",
                        (
                            Material.__module__,
                            Material.__qualname__,
                            id(inspect.getattr_static(Material, hook)),
                        ),
                    )
                    for hook in _ION_MATERIAL_BASE_INITIALIZATION_HOOKS
                ),
                tuple(
                    (source, _registered_child_value_signature(ion, source))
                    for source in ("e_init", "i_init", "o_init")
                ),
            )
            for name, ion in population.mech.ions.items()
        ),
        tuple(
            (
                name,
                type(material).__module__,
                type(material).__qualname__,
                tuple(material.fields),
                tuple(
                    (hook, _effective_hook_identity(material, hook))
                    for hook in _MATERIAL_INITIALIZATION_HOOKS
                ),
                tuple(
                    (
                        field,
                        (
                            ("missing-initial-source-module",)
                            if material._modules.get("_initial_sources") is None
                            else _registered_child_value_signature(
                                material._modules["_initial_sources"],
                                field,
                            )
                        ),
                    )
                    for field in material.fields
                ),
            )
            for name, material in population.mech.materials.items()
        ),
    )
    inspect_final_frame = _functional_initialization_profile(
        population
    ) is not None and not any(
        bool(collection)
        for collection in (
            population.mech.material_processes,
            population.mech.voltage_processes,
        )
    )
    final_frame = (
        _initialization_final_frame_dependency_signature(population, strict=False)
        if inspect_final_frame
        else None
    )
    return (
        population_hooks,
        integrator_hooks,
        handler_hooks,
        tuple(mechanisms),
        shared_fields,
        final_frame,
        _initialization_transform_signature(population),
    )


def _functional_initialization_failures(population: Population) -> list[str]:
    """Return reasons fresh initialization cannot yet be lowered exactly.

    Functional transition admission is intentionally broader. This initializer
    supports declared State plus canonical Ion/Material tensor transactions on
    the exact single-compartment and structurally canonical unmyelinated
    implicit-Euler topologies.
    Authored shared-field hooks and process initialization remain imperative and
    fail closed until their outputs have an explicit pure contract.
    """

    failures = []
    profile = _functional_initialization_profile(population)
    if profile is None:
        failures.append(
            "functional initialization currently supports exact "
            "SingleCompartment and structurally canonical Unmyelinated "
            "Populations"
        )
        population_family = None
        operator_kind = None
        implementation = None
    else:
        population_family, operator_kind, implementation = profile

    if operator_kind is not None:
        spec = _resolve_functional_integrator_spec(
            population.integrator,
            operator_kind,
        )
        if spec is None or spec.implementation is not implementation:
            failures.append(
                "functional initialization currently supports bwd_euler_sc "
                "for SingleCompartment and bwd_euler_ub for Unmyelinated"
            )

    population_standards = _STANDARD_POPULATION_FRESH_INITIALIZATION.get(
        population_family
    )
    if population_standards is not None:
        replaced_population_hooks = [
            name
            for name, standard in population_standards.items()
            if name in population.__dict__
            or inspect.getattr_static(type(population), name) is not standard
        ]
        if replaced_population_hooks:
            failures.append(
                "Population replaces canonical fresh-initialization hooks "
                f"{replaced_population_hooks}; authored outer initialization "
                "needs an explicit pure output contract"
            )

    if implementation is not None:
        integrator_standards = _STANDARD_INTEGRATOR_FRESH_INITIALIZATION[implementation]
        replaced_integrator_hooks = [
            name
            for name, standard in integrator_standards.items()
            if name in population.integrator.__dict__
            or inspect.getattr_static(type(population.integrator), name) is not standard
        ]
        if replaced_integrator_hooks:
            failures.append(
                "Integrator replaces canonical fresh-initialization hooks "
                f"{replaced_integrator_hooks}; authored voltage initialization "
                "needs an explicit pure output contract"
            )

    failures.extend(_functional_initialization_transform_failures(population))
    if "_steady_state" in population._caches or bool(
        population.initializing_from_state_cache
    ):
        failures.append(
            "steady-state cache restoration is not a fresh deterministic "
            "functional initialization"
        )

    handler = population.mech
    if type(handler) is not MechanismHandler:
        failures.append(
            "fresh functional initialization requires the canonical "
            f"MechanismHandler type, not {type(handler).__qualname__}"
        )
    replaced_handler_hooks = [
        name
        for name in _HANDLER_INITIALIZATION_HOOKS
        if not _handler_initialization_hook_is_standard(handler, name)
    ]
    if replaced_handler_hooks:
        failures.append(
            "MechanismHandler replaces canonical initialization hooks "
            f"{replaced_handler_hooks}; authored handler initialization needs "
            "an explicit pure output contract"
        )
    if handler.material_processes:
        failures.append(
            "MaterialProcess initialization is not included in the pure "
            "Ion/Material initializer transaction"
        )
    if handler.voltage_processes:
        failures.append(
            "VoltageProcess initialization is not included in the pure "
            "Ion/Material initializer transaction"
        )

    for ion_name, ion in handler.ions.items():
        if type(ion) is not Ion:
            failures.append(
                f"Ion {ion_name!r} has custom type {type(ion).__qualname__}; "
                "fresh functional initialization currently admits the canonical "
                "Ion tensor transaction only"
            )
            continue
        replaced = [
            name
            for name, standard in _STANDARD_ION_INITIALIZATION.items()
            if name in ion.__dict__
            or inspect.getattr_static(type(ion), name) is not standard
        ]
        replaced.extend(
            f"Material.{name}"
            for name, standard in _STANDARD_ION_MATERIAL_BASE_INITIALIZATION.items()
            if inspect.getattr_static(Material, name) is not standard
        )
        if replaced:
            failures.append(
                f"Ion {ion_name!r} replaces canonical initialization hooks "
                f"{replaced} on the source object or type"
            )
        unsupported_sources = [
            name
            for name in ("e_init", "i_init", "o_init")
            if not isinstance(getattr(ion, name), torch.nn.Parameter)
        ]
        if unsupported_sources:
            failures.append(
                f"Ion {ion_name!r} initial sources {unsupported_sources} are "
                "callable Modules; this initializer slice requires direct "
                "registered Tensor parameters"
            )

    for material_name, material in handler.materials.items():
        if type(material) is not Material:
            failures.append(
                f"Material {material_name!r} has custom type "
                f"{type(material).__qualname__}; authored initialization/advance "
                "hooks need an explicit pure output contract"
            )
            continue
        replaced = [
            name
            for name, standard in _STANDARD_MATERIAL_INITIALIZATION.items()
            if name in material.__dict__
            or inspect.getattr_static(type(material), name) is not standard
        ]
        if replaced:
            failures.append(
                f"Material {material_name!r} replaces canonical initialization "
                "hooks "
                f"{replaced} on the source object or type"
            )
        unsupported_sources = [
            field
            for field in material.fields
            if not isinstance(material.initial_source(field), torch.nn.Parameter)
        ]
        if unsupported_sources:
            failures.append(
                f"Material {material_name!r} initial sources "
                f"{unsupported_sources} are callable Modules; this initializer "
                "slice requires direct registered Tensor parameters"
            )

    if (
        profile is not None
        and not handler.material_processes
        and not handler.voltage_processes
    ):
        try:
            _initialization_final_frame_dependency_signature(
                population,
                strict=True,
            )
        except FunctionalizationError as exc:
            failures.append(str(exc))

    shared_local_names = {
        id(mechanism): set(mechanism._runtime_shared_local_names())
        for mechanism in handler.mechanisms.values()
    }
    absolute_state_seed_names = {}
    for _ion, field, mechanism, _support in handler._ion_write_support_plan:
        absolute_state_seed_names.setdefault(id(mechanism), set()).add(field)
    for _material, field, mechanism, _support in handler._material_write_support_plan:
        absolute_state_seed_names.setdefault(id(mechanism), set()).add(field)

    for mechanism_name, mechanism in handler.mechanisms.items():
        class_has_initial_values = (
            inspect.getattr_static(type(mechanism), "initial_values")
            is not _STANDARD_MECHANISM_INITIAL_VALUES
        )
        if bool(mechanism._has_authored_initial_values) != class_has_initial_values:
            failures.append(
                f"mechanism {mechanism_name!r} changed initial_values after "
                "construction; rebuild the Population so its immutable "
                "initialization plan matches the class"
            )
        changed = [
            name
            for name, implementation in _STANDARD_MECHANISM_INITIALIZATION.items()
            if name in mechanism.__dict__
            or inspect.getattr_static(type(mechanism), name) is not implementation
        ]
        if changed:
            failures.append(
                f"mechanism {mechanism_name!r} overrides initialization hooks "
                f"{changed}; authored hooks need an explicit pure output contract"
            )

        if "initial_values" in mechanism.__dict__:
            failures.append(
                f"mechanism {mechanism_name!r} replaces initial_values on the "
                "instance; define the pure hook on its Mechanism subclass"
            )
        elif class_has_initial_values:
            try:
                _initialization_callable_dependency_signature(
                    getattr(mechanism, "initial_values"),
                    label=f"{mechanism_name}.initial_values",
                    owner=mechanism,
                    strict=True,
                    forbidden_receiver_attributes=frozenset(
                        {"celsius"} | shared_local_names.get(id(mechanism), set())
                    ),
                )
            except FunctionalizationError as exc:
                failures.append(str(exc))

        if _timestep_buffer_names(mechanism):
            failures.append(
                f"mechanism {mechanism_name!r} has TIMESTEP_BUFFER workspaces; "
                "imperative initialization does not install timestep workspaces "
                "before State.state_defaults"
            )

        declared_names = {
            name for state in mechanism.DE.values() for name in state._state
        }
        for state_name, state in mechanism.DE.items():
            class_has_state_initial_values = (
                inspect.getattr_static(type(state), "initial_values")
                is not _STANDARD_STATE_INITIAL_VALUES
            )
            if (
                bool(state._has_authored_initial_values)
                != class_has_state_initial_values
            ):
                failures.append(
                    f"State {mechanism_name}.{state_name} changed "
                    "initial_values after construction; rebuild the Population "
                    "so its immutable initialization plan matches the class"
                )
            changed = [
                name
                for name, implementation in _STANDARD_STATE_FRESH_INITIALIZATION.items()
                if name in state.__dict__
                or inspect.getattr_static(type(state), name) is not implementation
            ]
            if changed:
                failures.append(
                    f"State {mechanism_name}.{state_name} overrides initialization "
                    f"hooks {changed}; authored hooks need an explicit pure output "
                    "contract"
                )

            if "initial_values" in state.__dict__:
                failures.append(
                    f"State {mechanism_name}.{state_name} replaces "
                    "initial_values on the instance; define the pure hook on its "
                    "State subclass"
                )
            elif class_has_state_initial_values:
                try:
                    _initialization_callable_dependency_signature(
                        getattr(state, "initial_values"),
                        label=f"{mechanism_name}.{state_name}.initial_values",
                        owner=state,
                        strict=True,
                        forbidden_receiver_attributes=frozenset(
                            {"celsius"} | shared_local_names.get(id(mechanism), set())
                        ),
                    )
                except FunctionalizationError as exc:
                    failures.append(str(exc))

            shared_state_buffer_overlap = set(state._carry) & shared_local_names.get(
                id(mechanism), set()
            )
            if shared_state_buffer_overlap:
                failures.append(
                    f"State {mechanism_name}.{state_name} CARRY names "
                    "overlap handler-owned shared locals "
                    f"{sorted(shared_state_buffer_overlap)}; use the canonical "
                    "Mechanism shared field through the values mapping"
                )

            if _timestep_buffer_names(state):
                failures.append(
                    f"State {mechanism_name}.{state_name} has TIMESTEP_BUFFER "
                    "workspaces; imperative initialization does not install them "
                    "before State.state_defaults"
                )

            missing = set(state._state) - set(mechanism._init_params)
            seeded = missing & absolute_state_seed_names.get(id(mechanism), set())
            class_has_state_defaults = (
                inspect.getattr_static(type(state), "state_defaults")
                is not _STANDARD_STATE_DEFAULTS
            )
            if "state_defaults" in state.__dict__:
                failures.append(
                    f"State {mechanism_name}.{state_name} replaces state_defaults "
                    "on the instance; define the pure hook on its State subclass"
                )
            if missing and (
                class_has_state_defaults or "state_defaults" in state.__dict__
            ):
                defaults_callable = getattr(state, "state_defaults")
                try:
                    _initialization_callable_dependency_signature(
                        defaults_callable,
                        label=f"{mechanism_name}.{state_name}.state_defaults",
                        owner=state,
                        strict=True,
                    )
                except FunctionalizationError as exc:
                    failures.append(str(exc))
                defaults_target = getattr(
                    defaults_callable,
                    "__func__",
                    defaults_callable,
                )
                try:
                    receiver_name = next(
                        iter(inspect.signature(defaults_target).parameters)
                    )
                except (StopIteration, TypeError, ValueError):
                    receiver_name = None
                receiver_attributes, _receiver_escapes = (
                    _direct_receiver_attribute_names(
                        defaults_target,
                        receiver_name,
                    )
                )
                hidden_shared_aliases = sorted(
                    receiver_attributes & shared_local_names.get(id(mechanism), set())
                )
                if hidden_shared_aliases:
                    failures.append(
                        f"State {mechanism_name}.{state_name}.state_defaults reads "
                        "support-local shared fields through hidden receiver "
                        f"aliases {hidden_shared_aliases}; move shared-dependent "
                        "initialization to initial_values(v, values)"
                    )
            authored_state_values = class_has_state_initial_values or (
                "initial_values" in state.__dict__
            )
            authored_mechanism_values = class_has_initial_values or (
                "initial_values" in mechanism.__dict__
            )
            if (
                (missing - seeded)
                and not (authored_state_values or authored_mechanism_values)
                and not class_has_state_defaults
            ):
                failures.append(
                    f"State {mechanism_name}.{state_name} has no state_defaults "
                    f"for declared states {sorted(missing - seeded)}"
                )

        unknown_init = set(mechanism._init_params) - declared_names
        if unknown_init:
            failures.append(
                f"mechanism {mechanism_name!r} declares insertion-time ic values "
                "for unknown "
                f"states {sorted(unknown_init)}"
            )

    try:
        layout = _state_layout(population)
    except FunctionalizationError as exc:
        failures.append(str(exc))
    else:
        unsupported = []
        for leaf in layout:
            path = leaf.public_path
            if path in {
                ("integrator", "v"),
                ("clock", "t"),
                ("control", "duration_remainder"),
            }:
                continue
            if len(path) == 3 and path[0] == "mechanisms":
                continue
            if len(path) == 3 and path[0] in {
                "ions",
                "materials",
                "ion_write_buffers",
                "material_write_buffers",
                "material_source_buffers",
            }:
                continue
            if (
                len(path) == 3
                and path[0] == "mechanism_buffers"
                and path[1] in handler.mechanisms
            ):
                continue
            if (
                len(path) == 4
                and path[0] == "state_buffers"
                and path[1] in handler.mechanisms
                and path[2] in handler.mechanisms[path[1]].DE
            ):
                continue
            unsupported.append(".".join(path))
        if unsupported:
            failures.append(
                "fresh functional initialization cannot yet reconstruct carry "
                "outside voltage, declared Mechanism state, and the canonical "
                f"Ion/Material transaction: {unsupported}"
            )
    return failures


def _initialization_transform_value(parameters, state, reference: str):
    if reference.startswith("parameters."):
        return parameters[reference.removeprefix("parameters.")]
    if reference.startswith("state."):
        return _get_path(state, tuple(reference.removeprefix("state.").split(".")))
    raise FunctionalizationError(
        f"unsupported initialization transform tensor path {reference!r}"
    )


def _set_initialization_transform_value(
    parameters,
    state,
    reference: str,
    value: torch.Tensor,
) -> None:
    if reference.startswith("parameters."):
        parameters[reference.removeprefix("parameters.")] = value
        return
    if reference.startswith("state."):
        _put_path(
            state,
            tuple(reference.removeprefix("state.").split(".")),
            value,
        )
        return
    raise FunctionalizationError(
        f"unsupported initialization transform tensor path {reference!r}"
    )


def _apply_initialization_transforms(
    initialization,
    phase: str,
    parameters,
    state,
    transform_inputs,
):
    """Apply one ordered pure transform phase to explicit tensor mappings."""

    parameters = dict(parameters)
    state = torch.utils._pytree.tree_map(lambda value: value, state)
    population = initialization.population
    for hook in _initialization_transform_hooks(population, phase):
        action = population._initialization_transforms[hook.action_name]
        read_values = tuple(
            _initialization_transform_value(parameters, state, reference)
            for reference in action.reads
        )
        explicit_values = tuple(
            transform_inputs[f"{phase}.{action.name}.{input_name}"]
            for input_name in action.input_names
        )
        write_targets = tuple(
            _initialization_transform_value(parameters, state, reference)
            for reference in action.writes
        )
        # A scalar BatchedTensor cannot currently broadcast against an
        # unbatched matrix when vmap has zero lanes. Give explicit inputs the
        # visible rank of this action, then remove only this framework-created
        # leading singleton padding from lower-rank outputs below.
        visible_rank = 0
        for value in (*read_values, *write_targets):
            if value.ndim > visible_rank:
                visible_rank = value.ndim
        padded_explicit_inputs = any(
            value.ndim < visible_rank for value in explicit_values
        )
        explicit_values = tuple(
            (
                value.reshape((1,) * (visible_rank - value.ndim) + tuple(value.shape))
                if value.ndim < visible_rank
                else value
            )
            for value in explicit_values
        )
        arguments = tuple(
            value.clone(memory_format=torch.preserve_format)
            for value in (*read_values, *explicit_values)
        )
        versions = (
            None
            if torch.compiler.is_compiling() or torch.is_inference_mode_enabled()
            else tuple(value._version for value in arguments)
        )
        outputs = action(*arguments)
        mutated_arguments = (
            ()
            if versions is None
            else tuple(
                index
                for index, (value, version) in enumerate(
                    zip(arguments, versions, strict=True)
                )
                if value._version != version
            )
        )
        if mutated_arguments:
            raise FunctionalizationError(
                f"initialization transform {action.name!r} mutated tensor "
                f"arguments {mutated_arguments}; return new output tensors"
            )
        if type(outputs) is not tuple:
            raise FunctionalizationError(
                f"initialization transform {action.name!r} must return an exact "
                "tuple of declared output tensors"
            )
        if len(outputs) != len(action.writes):
            raise FunctionalizationError(
                f"initialization transform {action.name!r} returned {len(outputs)} "
                f"outputs; expected {len(action.writes)}"
            )

        # Validate every result before publishing any update from this action.
        # Sequential actions compose intentionally, but one malformed action is
        # atomic and cannot leave a partially updated tensor frame.
        staged = []
        for reference, value, target in zip(
            action.writes,
            outputs,
            write_targets,
            strict=True,
        ):
            if not torch.is_tensor(value):
                raise FunctionalizationError(
                    f"initialization transform {action.name!r} output "
                    f"{reference!r} must be a Tensor"
                )
            if padded_explicit_inputs:
                while value.ndim > target.ndim and value.shape[0] == 1:
                    value = value.squeeze(0)
            value = value.to(device=target.device, dtype=target.dtype)
            try:
                value = torch.broadcast_to(value, _shape_tuple(target))
            except RuntimeError as exc:
                raise FunctionalizationError(
                    f"initialization transform {action.name!r} output "
                    f"{reference!r} with shape {_shape_tuple(value)} is not "
                    f"broadcastable to {_shape_tuple(target)}"
                ) from exc
            staged.append(
                (
                    reference,
                    value.clone(memory_format=torch.preserve_format),
                )
            )
        for reference, value in staged:
            _set_initialization_transform_value(
                parameters,
                state,
                reference,
                value,
            )
    return parameters, state


def _reset_functional_initialization_clock(state):
    """Apply Population.initialize()'s terminal clock/control reset."""

    for path in (
        ("clock", "t"),
        ("control", "duration_remainder"),
    ):
        value = _get_path(state, path)
        _put_path(state, path, torch.zeros_like(value))
    return state


def _pad_to_visible_rank(value, reference):
    """Right-align a low-rank tensor with a functional shared-field frame."""

    if value.ndim >= reference.ndim:
        return value
    return value.reshape((1,) * (reference.ndim - value.ndim) + tuple(value.shape))


class _PopulationInitialization(torch.nn.Module):
    """Pure facade for the admitted declaration-driven initialization."""

    def __init__(
        self,
        population: Population,
        layout: tuple[_StateLeaf, ...],
        explicit_layout: tuple[_InitializationStateLeaf, ...],
    ):
        super().__init__()
        self.population = population
        self.layout = layout
        self.explicit_layout = explicit_layout
        self.mechanism_plan = tuple(
            (
                mechanism_name,
                tuple(
                    (index, entry.state_name)
                    for index, entry in enumerate(explicit_layout)
                    if entry.mechanism_name == mechanism_name
                ),
            )
            for mechanism_name, mechanism in population.mech.mechanisms.items()
            if mechanism.DE
        )

    def forward(self, v_init, *explicit_values):
        if len(explicit_values) != len(self.explicit_layout):
            raise FunctionalizationError(
                "functional initialization received the wrong number of explicit "
                "state leaves"
            )

        # Imperative ``Integrator.init_v`` owns one detached, contiguous
        # voltage copy before any Mechanism/State initialization runs. Keep
        # the ownership/layout semantics while retaining the functional graph.
        initial_voltage = v_init.clone(memory_format=torch.contiguous_format)
        state_overrides = {}
        for mechanism_name, override_plan in self.mechanism_plan:
            mechanism = self.population.mech.mechanisms[mechanism_name]
            state_overrides[mechanism.name] = {
                state_name: explicit_values[index]
                for index, state_name in override_plan
            }

        # Run the same reset/provisional/commit/guard/accepted-frame transaction
        # as imperative initialization. Geometry and parameter workspaces are
        # already explicit functional-call bindings; current scratch is not.
        handler = self.population.mech
        shared_celsius = _pad_to_visible_rank(handler.celsius, initial_voltage)
        # Keep authored ``celsius`` aliases bound to the canonical scalar (or
        # user-provided tensor) exactly as imperative initialization does. The
        # rank-padded view exists only to make the shared Ion/Material formulas
        # preserve hidden ``vmap`` lanes, including a zero-sized lane.
        handler._sync_celsius(handler.celsius)
        handler._initialize_tensor_transaction(
            initial_voltage,
            shared_celsius,
            state_overrides=state_overrides,
            call_local_currents=True,
        )

        outputs = []
        for leaf in self.layout:
            if leaf.public_path == ("integrator", "v"):
                value = initial_voltage
            elif leaf.public_path == ("clock", "t") or leaf.public_path == (
                "control",
                "duration_remainder",
            ):
                value = torch.zeros(
                    (),
                    device=leaf.device,
                    dtype=leaf.dtype,
                )
            else:
                value = _buffer_at(
                    self.population,
                    leaf.module_path,
                    leaf.buffer_name,
                )
            if not torch.is_tensor(value):
                raise FunctionalizationError(
                    "fresh functional initialization returned non-Tensor state "
                    f"leaf {'.'.join(leaf.public_path)!r}"
                )
            if (
                _shape_tuple(value) != leaf.shape
                or value.dtype != leaf.dtype
                or value.device != leaf.device
            ):
                raise FunctionalizationError(
                    "fresh functional initialization changed the schema of state "
                    f"leaf {'.'.join(leaf.public_path)!r}: expected "
                    f"shape/device/dtype {leaf.shape}/{leaf.device}/{leaf.dtype}, "
                    f"got {_shape_tuple(value)}/{value.device}/{value.dtype}"
                )
            outputs.append(value)
        return tuple(outputs)


class _PopulationTransition(torch.nn.Module):
    """Functional facade over a supported deterministic step implementation."""

    def __init__(self, population: Population, solver, layout):
        super().__init__()
        self.population = population
        self.solver = solver
        self.layout = layout
        self.block_voltage_state = tuple(population.integrator.v_vars) == ("v", "vc")
        object.__setattr__(
            self,
            "_state_owners",
            tuple(_module_at(population, entry.module_path) for entry in layout),
        )

    def forward(self, v, t, duration_remainder, dt, celsius, ve, intra, steps: int):
        handler = self.population.mech
        shared_celsius = _pad_to_visible_rank(celsius, v)
        handler._sync_celsius(celsius)
        handler.read_from_ions()
        handler.read_from_materials()

        integrator = self.population.integrator
        vc = self.population._buffers.get("vc") if self.block_voltage_state else None
        for index in range(steps):
            ve_step = None if ve is None else ve[index]
            intra_step = None if intra is None else intra[index]
            # Mechanism ``t`` references resolve through the Population buffer.
            # Publish the explicit carry immediately before each authored step,
            # matching imperative execution where the clock advances afterward.
            self.population._buffers["t"] = t
            if self.block_voltage_state:
                vc, v, _i_membrane = integrator._step(
                    vc.reshape(integrator.B, integrator.K, integrator.M),
                    v,
                    dt,
                    shared_celsius,
                    ve_step,
                    intra_step,
                    solver=self.solver,
                    call_local_currents=True,
                )
            else:
                v, _i_membrane = integrator._step(
                    v,
                    dt,
                    shared_celsius,
                    ve_step,
                    intra_step,
                    solver=self.solver,
                    call_local_currents=True,
                )
            # Multiplication by a scalar-shaped tensor preserves hidden vmap
            # lanes (including zero lanes) while keeping dt an explicit
            # differentiable input to both dynamics and the model clock.
            t = (t.reshape(1) + dt.reshape(1)).squeeze(0)

        outputs = []
        for entry, owner in zip(self.layout, self._state_owners, strict=True):
            if entry.public_path == ("integrator", "v"):
                value = v
            elif entry.public_path == ("integrator", "vc") and self.block_voltage_state:
                value = vc
            elif entry.public_path == ("clock", "t"):
                value = t
            elif entry.public_path == ("control", "duration_remainder"):
                value = duration_remainder
            else:
                value = owner._buffers[entry.buffer_name]
            outputs.append(value)
        return tuple(outputs)


def _uses_standard_unmyelinated_geometry(population) -> bool:
    for name in ("area", "edge_resistance_ohm"):
        if inspect.getattr_static(type(population), name) is not inspect.getattr_static(
            Unmyelinated, name
        ):
            return False
    return True


def _uses_standard_myelinated_geometry(population) -> bool:
    for name in ("area", "edge_resistance_ohm"):
        if inspect.getattr_static(type(population), name) is not inspect.getattr_static(
            Myelinated,
            name,
        ):
            return False
    return True


def _uses_standard_extcell_axon_geometry(population) -> bool:
    """Return whether an ExtCell path retains the standard cylindrical area."""
    return inspect.getattr_static(type(population), "area") is inspect.getattr_static(
        ExtCellAxon,
        "area",
    )


def _module_autograd_hook_signature(module: torch.nn.Module) -> tuple:
    """Fingerprint registered hooks that can change forward or backward semantics."""
    signatures = []
    for registry_name in (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    ):
        registry = getattr(module, registry_name, None)
        if registry:
            signatures.append(
                (
                    registry_name,
                    tuple((key, id(hook)) for key, hook in registry.items()),
                )
            )
    return tuple(signatures)


def _uses_standard_myelinated_parametrizations(population) -> bool:
    in_graph = population.in_graph_parametrizations
    if set(in_graph) != {"rhoa"} or len(in_graph["rhoa"]) != 1:
        return False
    rhoa_transform, args = in_graph["rhoa"][0]
    if (
        type(rhoa_transform) is not Myelinated.myelinated_rhoa
        or "forward" in rhoa_transform.__dict__
        or inspect.getattr_static(type(rhoa_transform), "forward")
        is not _STANDARD_MYELINATED_RHOA_FORWARD
        or tuple(args) != ("dx", "diameters", "diam")
        or _module_autograd_hook_signature(rhoa_transform)
    ):
        return False
    if not any(module is rhoa_transform for module in population.modules()):
        return False

    parametrizations = getattr(population, "parametrizations", None)
    if parametrizations is None or set(parametrizations) != {"diam"}:
        return False
    diameter_transforms = parametrizations["diam"]
    if not len(diameter_transforms):
        return False
    diameter_transform = diameter_transforms[0]
    if not (
        type(diameter_transform) is Myelinated.myelinated_node_d
        and "forward" not in diameter_transform.__dict__
        and inspect.getattr_static(type(diameter_transform), "forward")
        is _STANDARD_MYELINATED_DIAMETER_FORWARD
        and not _module_autograd_hook_signature(diameter_transforms)
        and not _module_autograd_hook_signature(diameter_transform)
    ):
        return False
    return all(
        _uses_passive_end_diameter_override(transform, population)
        for index, transform in enumerate(diameter_transforms)
        if index > 0
    )


def _uses_passive_end_diameter_override(transform, population) -> bool:
    """Admit only the framework's fixed sparse geometry override."""
    if (
        type(transform) is not _PassiveEndDiameter
        or "forward" in transform.__dict__
        or inspect.getattr_static(type(transform), "forward")
        is not _STANDARD_PASSIVE_END_DIAMETER_FORWARD
        or _module_autograd_hook_signature(transform)
        or transform._parameters
        or transform._modules
        or set(transform._buffers) != {"mask", "value"}
    ):
        return False
    mask = transform._buffers["mask"]
    value = transform._buffers["value"]
    return (
        torch.is_tensor(mask)
        and mask.dtype == torch.bool
        and _shape_tuple(mask) == tuple(population.core_shape())
        and mask.device == population.device()
        and torch.is_tensor(value)
        and _shape_tuple(value) == ()
        and value.dtype == population.dtype()
        and value.device == population.device()
    )


def _geometry_constant_value(population, name):
    """Resolve one explicit functional geometry source from a Population."""
    if name == "diam_original":
        parametrizations = getattr(population, "parametrizations", None)
        if parametrizations is None or "diam" not in parametrizations:
            return None
        return parametrizations.diam.original
    if name == "canonical_area_cm2":
        if _is_canonical_tree(population):
            return population.area
        return getattr(population, "_canonical_area_cm2", None)
    if name == "canonical_edge_resistance_ohm":
        if _is_canonical_tree(population):
            return population.edge_resistance_ohm
        return getattr(population, "_canonical_edge_resistance_ohm", None)
    return getattr(population, name, None)


def _uses_standard_single_compartment_geometry(population) -> bool:
    """Return whether scalar-compartment area follows the base Population rule."""
    return inspect.getattr_static(type(population), "area") is inspect.getattr_static(
        Population,
        "area",
    )


def _is_native_cable(population) -> bool:
    """Return whether ``population`` is an exact canonical generic Cable."""
    return type(population) is Cable and population.compartment_graph is not None


def _is_scalar_tree(population) -> bool:
    """Return whether ``population`` owns one canonical scalar Tree topology."""
    return (
        isinstance(population, Tree)
        and not isinstance(population, ExtCellTree)
        and population.compartment_graph is not None
    )


def _is_extcell_tree(population) -> bool:
    """Return whether ``population`` owns one canonical block Tree topology."""
    return (
        isinstance(population, ExtCellTree) and population.compartment_graph is not None
    )


def _is_canonical_tree(population) -> bool:
    """Return whether a supported Tree family has an immutable topology snapshot."""
    return _is_scalar_tree(population) or _is_extcell_tree(population)


def _is_scalar_multi(population) -> bool:
    """Return whether ``population`` is the exact packed scalar container."""
    return type(population) is MultiPopulation


def _functional_topology_kind(population) -> str | None:
    """Return the exact audited functional topology implemented by a model."""
    if _is_scalar_multi(population):
        operator_kind = getattr(
            population.integrator,
            "_FUNCTIONAL_OPERATOR_KIND",
            None,
        )
        return (
            "multi_point" if operator_kind == "scalar_multi_point" else "multi_scalar"
        )
    if isinstance(population, ExtCellAxon):
        return "extcell_axon"
    if _is_extcell_tree(population):
        return "extcell_tree"
    if type(population) is SingleCompartment:
        return "single_compartment"
    if _is_scalar_tree(population):
        return "scalar_tree"
    if _is_native_cable(population):
        return "native_cable"
    if isinstance(population, Myelinated):
        return "myelinated"
    if isinstance(population, Unmyelinated):
        return "unmyelinated"
    return None


def _functional_geometry_constant_names(topology_kind: str) -> tuple[str, ...]:
    """Return the explicit immutable geometry leaves for one topology."""
    if topology_kind in {"multi_point", "multi_scalar"}:
        names = ["diam", "dx"]
    elif topology_kind == "native_cable":
        names = ["diam", "dx"]
    elif topology_kind == "single_compartment":
        names = ["diam", "dx"]
    elif topology_kind == "scalar_tree":
        names = ["diam", "dx"]
    elif topology_kind in {"extcell_axon", "extcell_tree"}:
        names = ["diam", "dx", "xraxial", "xc", "xg"]
    elif topology_kind == "myelinated":
        names = ["diameters", "diam_original", "dx"]
    elif topology_kind == "unmyelinated":
        names = ["diam", "dx"]
    else:  # pragma: no cover - callers admit exact supported topologies first
        raise FunctionalizationError(
            f"Unknown functional topology kind {topology_kind!r}."
        )
    if topology_kind in {"native_cable", "scalar_tree", "extcell_tree"}:
        names.extend(("canonical_area_cm2", "canonical_edge_resistance_ohm"))
    return tuple(names)


def _multi_component_topology_signature(population: MultiPopulation) -> tuple:
    """Fingerprint ordered component structure without consulting child state.

    Child runtime state is intentionally absent: after concatenation it is not
    authoritative.  Canonical morphology identity and configuration-module
    structure remain part of the lowering contract, however.
    """
    result = []
    for name, component in population.populations.items():
        canonical = None
        if _is_native_cable(component):
            canonical = (
                "native_cable",
                component._canonical_morphology_fingerprint_reference,
                _shape_tuple(component._canonical_area_cm2),
                _shape_tuple(component._canonical_edge_resistance_ohm),
            )
        elif _is_canonical_tree(component):
            canonical = (
                "tree",
                tuple(
                    int(parent)
                    for parent in component.compartment_graph.topology.parent_index
                ),
                _shape_tuple(component.area),
                _shape_tuple(component.edge_resistance_ohm),
            )
        result.append(
            (
                name,
                type(component).__module__,
                type(component).__qualname__,
                tuple(component.shape),
                tuple(component.core_shape()),
                component.device().type,
                component.dtype(),
                canonical,
                tuple(
                    (parameter_name, _shape_tuple(parameter))
                    for parameter_name, parameter in component.named_parameters()
                    if not parameter_name.startswith("integrator.")
                ),
            )
        )
    return tuple(result)


def _multi_canonical_source_binding(population: MultiPopulation) -> tuple:
    """Bind immutable canonical component geometry for source freshness."""
    bindings = []
    for name, component in population.populations.items():
        if _is_native_cable(component):
            binding = _native_cable_source_binding(component)
        elif _is_canonical_tree(component):
            binding = ("immutable_compartment_graph", id(component.compartment_graph))
        else:
            binding = None
        bindings.append((name, binding))
    return tuple(bindings)


def _multi_canonical_value_signature(population: MultiPopulation) -> tuple:
    """Return an identity-free exact canonical-geometry compatibility record."""
    result = []
    for name, component in population.populations.items():
        if _is_native_cable(component) or _is_canonical_tree(component):
            values = tuple(
                (
                    field,
                    _audit_tensor_value(_geometry_constant_value(component, field)),
                )
                for field in (
                    "canonical_area_cm2",
                    "canonical_edge_resistance_ohm",
                )
            )
        else:
            values = None
        result.append((name, values))
    return tuple(result)


def _validate_canonical_tree_source(population, *, label: str) -> str | None:
    """Validate registered Tree topology and exact geometry provenance."""
    graph = population.compartment_graph
    if graph is None:
        return f"{label} is missing its canonical CompartmentGraph"
    expected_parent = torch.as_tensor(
        graph.topology.parent_index,
        dtype=torch.long,
        device=population.device(),
    )
    parent = getattr(population, "diff_parent_index", None)
    if (
        not torch.is_tensor(parent)
        or parent.dtype != torch.long
        or _shape_tuple(parent) != _shape_tuple(expected_parent)
        or not torch.equal(parent, expected_parent)
    ):
        return f"{label} registered topology disagrees with its CompartmentGraph"
    if inspect.getattr_static(type(population), "area") is not inspect.getattr_static(
        Tree, "area"
    ):
        return f"{label} functionalization requires the standard exact area"
    if inspect.getattr_static(
        type(population), "edge_resistance_ohm"
    ) is not inspect.getattr_static(Tree, "edge_resistance_ohm"):
        return f"{label} functionalization requires the standard exact axial resistance"
    return None


def _validate_scalar_tree_source(population) -> str | None:
    return _validate_canonical_tree_source(population, label="scalar Tree")


def _validate_extcell_tree_source(population) -> str | None:
    return _validate_canonical_tree_source(population, label="ExtCellTree")


def _validate_native_cable_source(population) -> str | None:
    """Audit canonical Cable provenance without retaining validator cache writes."""
    cached_signature = population._validated_runtime_contract_signature
    try:
        population._validate_canonical_geometry()
    except (TypeError, ValueError, RuntimeError) as exc:
        return f"native Cable canonical geometry is invalid: {exc}"
    finally:
        population._validated_runtime_contract_signature = cached_signature
    return None


_NATIVE_CABLE_FIXED_SOURCE_TENSORS = (
    "diam",
    "dx",
    "rhoa",
    "volume",
    "volume_um3",
    "volume_i",
    "volume_o",
    "diff_geom_um",
    "diff_parent_index",
    "_canonical_area_cm2",
    "_canonical_edge_resistance_ohm",
    "_canonical_morphology_fingerprint",
    "_canonical_geometry_reference",
)


def _immutable_source_tensor_binding(value: torch.Tensor) -> tuple:
    """Return a cheap mutation signature, with value fallback for inference tensors."""
    try:
        version = value._version
    except RuntimeError:
        version = None
    return (
        id(value),
        version,
        _shape_tuple(value),
        tuple(int(item) for item in value.stride()),
        value.storage_offset(),
        value.dtype,
        value.device,
        value.layout,
        value.data_ptr(),
        _audit_tensor_digest(value) if version is None else None,
    )


def _additional_parameter_key_binding(population) -> tuple:
    """Capture cheap identity/version metadata for indexed-override keys."""
    return tuple(
        (
            module_path,
            name,
            _immutable_source_tensor_binding(key),
        )
        for module_path, owner in _parameterized_owner_paths(population)
        for name in owner.additional_parameters
        for key in (owner.keys[name],)
    )


def _additional_parameter_layout_signature(population) -> tuple:
    """Fingerprint exact alias-to-local-slot assignments.

    A Mechanism's SupportSpec identifies its total physical support, but two
    compatible-looking Populations can partition that union differently among
    named parameter overrides. Record each ordered alias's exact local key
    slice so compatible extraction cannot silently exchange those meanings.
    """
    layout = []
    with torch.no_grad():
        for module_path, owner in _parameterized_owner_paths(population):
            for name, entries in owner.additional_parameters.items():
                key = owner.keys[name].detach().to(device="cpu", dtype=torch.long)
                key = key.reshape(-1)
                cursor = 0
                records = []
                for fill, source in entries:
                    source_name = _additional_parameter_source_name(owner, source)
                    value = _resolve_parameter_value(owner, source_name)
                    if not torch.is_tensor(value):
                        raise FunctionalizationError(
                            f"Regional parameter source {source_name!r} on "
                            f"{type(owner).__qualname__} is not tensor-valued."
                        )
                    # Replica overrides share physical keys across their
                    # leading value axes; only the final slot axis consumes
                    # keys. Retain that distinction in the layout fingerprint.
                    replica_value = isinstance(fill, _ReplicaRangeExpander)
                    count = (
                        fill.slot_count if replica_value else int(fill(value).numel())
                    )
                    record_key = key[cursor : cursor + count]
                    if record_key.numel() != count:
                        raise FunctionalizationError(
                            f"Regional parameter layout for {name!r} on "
                            f"{type(owner).__qualname__} is inconsistent."
                        )
                    records.append(
                        (
                            source_name,
                            _shape_tuple(value),
                            count,
                            replica_value,
                            _audit_tensor_digest(record_key),
                        )
                    )
                    cursor += count
                if cursor != key.numel():
                    raise FunctionalizationError(
                        f"Regional parameter layout for {name!r} on "
                        f"{type(owner).__qualname__} leaves unmatched keys."
                    )
                layout.append((module_path, name, tuple(records)))
    return tuple(layout)


def _native_cable_source_binding(population) -> tuple:
    """Capture immutable canonical inputs without updating Cable audit caches."""
    return (
        population._canonical_morphology_fingerprint_reference,
        tuple(
            (name, _immutable_source_tensor_binding(getattr(population, name)))
            for name in _NATIVE_CABLE_FIXED_SOURCE_TENSORS
        ),
    )


def _effective_population_parameter_failure(population) -> str | None:
    """Reject live overrides not represented by the functional raw tree."""
    try:
        # Eligibility checks must never execute authored parameter modules on
        # the caller's live Population. Stateful or ultimately unsupported
        # transforms are confined to an isolated clone.
        candidate = _clone_execution_population(population)
        with torch.no_grad():
            expected = candidate._derive_parameter_buffers()
    except (TypeError, ValueError, RuntimeError) as exc:
        return str(exc)
    mismatched = [
        name
        for name, value in expected.items()
        if not torch.is_tensor(getattr(population, name, None))
        or _shape_tuple(getattr(population, name)) != _shape_tuple(value)
        or not torch.equal(getattr(population, name), value)
    ]
    if mismatched:
        return (
            "Population live effective values no longer match their raw "
            f"parameter sources: {sorted(mismatched)}; rebuild/populate the "
            "model or express variation through functional tensor inputs"
        )
    return None


def _parameterized_owner_paths(population):
    yield "", population
    for module_path in _parameterized_module_paths(population):
        yield module_path, _module_at(population, module_path)


def _parameterized_owners(population):
    for _module_path, owner in _parameterized_owner_paths(population):
        yield owner


def _parameter_materialization_output_names(owner) -> set[str]:
    """Return names seeded before authored in-graph transforms execute.

    An in-graph transform target is not automatically reconstructed: when the
    target is absent from this initial mapping, ``_derive_parameter_buffers``
    reads its resident owner attribute as the transform's base value. That
    dependency must therefore be routed just like an explicit transform arg.
    """
    names = {
        name
        for name in _declared_parameter_names(owner)
        if name in owner._buffers and torch.is_tensor(owner._buffers[name])
    }
    parametrizations = getattr(owner, "parametrizations", None)
    if parametrizations is not None:
        names.update(parametrizations)
    names.update(owner.additional_parameters)
    return names


def _parametrization_constant_layout_for_owners(
    population,
    owners,
    *,
    public_constant_by_identity=None,
) -> tuple[_ParametrizationConstant, ...]:
    """Discover registered buffer dependencies for selected parameter owners."""
    owners = tuple(owners)
    public_constant_by_identity = (
        {} if public_constant_by_identity is None else public_constant_by_identity
    )
    modules = tuple(population.named_modules(remove_duplicate=False))
    module_paths = {}
    for path, module in modules:
        module_paths.setdefault(id(module), path)

    grouped = {}

    def record(value, paths):
        record = grouped.setdefault(
            id(value),
            {
                "value": value,
                "paths": [],
            },
        )
        record["paths"].extend(paths)

    def is_registered_parameter(value):
        return any(
            parameter is value
            for _path, module in modules
            for parameter in module._parameters.values()
        )

    def registered_buffer_aliases(value):
        aliases = []
        for module_path, module in modules:
            for buffer_name, buffer in module._buffers.items():
                if buffer is value:
                    aliases.append(
                        f"{module_path + '.' if module_path else ''}{buffer_name}"
                    )
        return aliases

    def route_resident_dependency(owner, dependency_name, *, role):
        if dependency_name in owner._parameters:
            value = owner._parameters[dependency_name]
        elif dependency_name in owner._buffers:
            value = owner._buffers[dependency_name]
        else:
            raise FunctionalizationError(
                f"In-graph parameter transform {role} {dependency_name!r} is "
                f"not a directly registered Tensor on "
                f"{type(owner).__qualname__}; arguments must be a registered "
                "buffer or Parameter."
            )
        if is_registered_parameter(value):
            # Arbitrary registered Parameters are already explicit public
            # leaves; the alias-complete parameter route replaces every slot.
            return
        if not torch.is_tensor(value):
            raise FunctionalizationError(
                f"In-graph parameter transform {role} {dependency_name!r} on "
                f"{type(owner).__qualname__} must resolve to a reconstructed "
                "value or a registered Tensor."
            )
        aliases = registered_buffer_aliases(value)
        if not aliases:
            raise FunctionalizationError(
                f"In-graph parameter transform tensor {role} "
                f"{dependency_name!r} on {type(owner).__qualname__} must be a "
                "registered buffer or Parameter."
            )
        record(value, aliases)

    for owner in owners:
        # Myelinated geometry may append framework-owned passive-end overrides
        # to its standard torch parametrization. Their mask/value buffers must
        # remain explicit inputs, just like authored in-graph dependencies.
        parametrizations = getattr(owner, "parametrizations", None)
        if isinstance(owner, Myelinated) and parametrizations is not None:
            for index, transform in enumerate(parametrizations["diam"]):
                if index == 0 or type(transform) is not _PassiveEndDiameter:
                    continue
                transform_path = module_paths[id(transform)]
                for buffer_name, value in transform.named_buffers(recurse=False):
                    record(value, (f"{transform_path}.{buffer_name}",))
        available_names = _parameter_materialization_output_names(owner)
        available_names.update({"diam", "diameters", "dx", "celsius"})
        for target_name, transforms in owner.in_graph_parametrizations.items():
            if target_name not in available_names:
                raise FunctionalizationError(
                    "In-graph parameter transform target "
                    f"{target_name!r} on {type(owner).__qualname__} has no "
                    "reconstructed parameter base. Declare it as a supported "
                    "parameter field instead of transforming resident state."
                )
            for transform, args in transforms:
                try:
                    transform_path = module_paths[id(transform)]
                except KeyError as exc:
                    raise FunctionalizationError(
                        "In-graph parameter transforms must be registered modules."
                    ) from exc
                for local_module_path, module in transform.named_modules(
                    remove_duplicate=False
                ):
                    for local_name, value in module._buffers.items():
                        if value is None:
                            continue
                        if is_registered_parameter(value):
                            # Cross-category aliases are routed from the one
                            # canonical public Parameter leaf, including this
                            # transform-buffer slot.
                            continue
                        relative_path = (
                            f"{local_module_path + '.' if local_module_path else ''}"
                            f"{local_name}"
                        )
                        full_path = f"{transform_path}.{relative_path}"
                        record(value, (full_path,))

                for argument_name in args:
                    if not isinstance(argument_name, str):
                        raise FunctionalizationError(
                            "In-graph parameter transform arguments must be "
                            "registered tensor names."
                        )
                    # ``_derive_parameter_buffers`` resolves these through its
                    # freshly reconstructed values mapping. Geometry and
                    # framework-shared fields are likewise rebound under their
                    # canonical names before owner materialization.
                    if argument_name in available_names:
                        continue
                    route_resident_dependency(
                        owner,
                        argument_name,
                        role="argument",
                    )
            available_names.add(target_name)

    layout = []
    for record in grouped.values():
        value = record["value"]
        aliases = tuple(dict.fromkeys(record["paths"]))
        canonical = aliases[0]
        module_path, _separator, buffer_name = canonical.rpartition(".")
        if not canonical:  # pragma: no cover - named buffers are never empty
            raise FunctionalizationError(
                "In-graph transform buffer path cannot be empty."
            )
        layout.append(
            _ParametrizationConstant(
                key=public_constant_by_identity.get(
                    id(value),
                    f"parametrizations.{canonical}",
                ),
                module_path=module_path,
                buffer_name=buffer_name,
                aliases=aliases,
                shape=_shape_tuple(value),
                dtype=value.dtype,
                device=value.device,
            )
        )
    return tuple(layout)


def _parametrization_constant_layout(
    population,
) -> tuple[_ParametrizationConstant, ...]:
    """Discover registered buffer dependencies of authored in-graph transforms."""
    topology_kind = _functional_topology_kind(population)
    public_constant_by_identity = {}
    if topology_kind is not None:
        for name in _functional_geometry_constant_names(topology_kind):
            value = _geometry_constant_value(population, name)
            if torch.is_tensor(value):
                public_constant_by_identity.setdefault(id(value), name)
    return _parametrization_constant_layout_for_owners(
        population,
        _parameterized_owners(population),
        public_constant_by_identity=public_constant_by_identity,
    )


def _parametrization_constant_layout_signature(population) -> tuple:
    return tuple(
        (
            entry.key,
            entry.aliases,
            entry.shape,
            entry.dtype,
            entry.device,
        )
        for entry in _parametrization_constant_layout(population)
    )


def _additional_parameter_source_name(owner, source) -> str:
    """Return the stable registered slot backing one indexed override."""
    if isinstance(source, str):
        return source
    for name, parameter in owner.named_parameters(remove_duplicate=False):
        if parameter is source:
            return name
    for name, module in owner.named_modules(remove_duplicate=False):
        if name and module is source:
            return name
    raise FunctionalizationError(
        f"{type(owner).__qualname__} retains an unregistered regional parameter "
        "source; rebuild the Population."
    )


def _additional_parameter_schema(owner) -> tuple:
    """Describe ordered indexed-override sources without reading tensor values."""
    return tuple(
        (
            name,
            tuple(
                _additional_parameter_source_name(owner, source)
                for _fill, source in entries
            ),
        )
        for name, entries in owner.additional_parameters.items()
    )


def _additional_parameter_raw_names(population) -> set[str]:
    """Return canonical raw leaves owned by indexed regional overrides."""
    canonical_by_identity = {
        id(parameter): name for name, parameter in population.named_parameters()
    }
    names = set()
    for owner in _parameterized_owners(population):
        for entries in owner.additional_parameters.values():
            for _fill, source in entries:
                source_name = _additional_parameter_source_name(owner, source)
                registered = getattr(owner, source_name)
                if isinstance(registered, torch.nn.Parameter):
                    parameter_names = (canonical_by_identity.get(id(registered)),)
                elif isinstance(registered, torch.nn.Module):
                    parameter_names = tuple(
                        canonical_by_identity.get(id(parameter))
                        for parameter in registered.parameters()
                    )
                else:
                    parameter_names = ()
                names.update(name for name in parameter_names if name is not None)
    return names


def _in_graph_transform_parameter_names(population) -> set[str]:
    """Return public Parameters explicitly consumed by authored transforms."""
    canonical_by_identity = {
        id(parameter): name for name, parameter in population.named_parameters()
    }
    names = set()
    for owner in _parameterized_owners(population):
        for transforms in owner.in_graph_parametrizations.values():
            for transform, args in transforms:
                names.update(
                    canonical_by_identity[id(parameter)]
                    for parameter in transform.parameters()
                    if id(parameter) in canonical_by_identity
                )
                for argument_name in args:
                    if not isinstance(argument_name, str):
                        continue
                    if argument_name in owner._parameters:
                        value = owner._parameters[argument_name]
                    elif argument_name in owner._buffers:
                        value = owner._buffers[argument_name]
                    else:
                        continue
                    if id(value) in canonical_by_identity:
                        names.add(canonical_by_identity[id(value)])
    return names


def _shared_initial_parameter_target_shapes(
    population,
    parameter_name_by_identity: Mapping[int, str],
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Map direct Ion/Material initial Parameters to their runtime frames.

    These sources are allowed to carry any visible shape that broadcasts
    exactly to the corresponding shared field. Other raw Parameters retain the
    deliberately narrower scalar-or-explicit-regional lowering contract.
    """

    target_shapes: dict[str, set[tuple[int, ...]]] = {}

    def record(source, target) -> None:
        if not isinstance(source, torch.nn.Parameter):
            return
        parameter_name = parameter_name_by_identity.get(id(source))
        if parameter_name is None:
            return
        target_shapes.setdefault(parameter_name, set()).add(_shape_tuple(target))

    handler = getattr(population, "mech", None)
    if handler is None:
        return {}

    for ion in handler.ions.values():
        ion_name = ion.name
        record(ion.e_init, ion._buffers[f"e{ion_name}"])
        record(ion.i_init, ion._buffers[f"{ion_name}i"])
        record(ion.o_init, ion._buffers[f"{ion_name}o"])

    for material in handler.materials.values():
        sources = material._modules.get("_initial_sources")
        if sources is None:
            continue
        for field in material._material_fields:
            record(sources._parameters.get(field), material._buffers[field])

    return {
        parameter_name: tuple(sorted(shapes))
        for parameter_name, shapes in target_shapes.items()
    }


def _shape_broadcasts_to_targets(
    source_shape: tuple[int, ...],
    target_shapes: tuple[tuple[int, ...], ...],
) -> bool:
    """Return whether one source broadcasts to every declared target exactly."""

    for target_shape in target_shapes:
        try:
            broadcast_shape = tuple(torch.broadcast_shapes(source_shape, target_shape))
        except RuntimeError:
            return False
        if broadcast_shape != target_shape:
            return False
    return True


def _parameter_buffer_alias_conflicts(population) -> tuple[tuple, ...]:
    """Find Parameter/buffer aliases that cross functional ownership domains."""
    modules = tuple(population.named_modules(remove_duplicate=False))
    parameter_name_by_identity = {
        id(parameter): name for name, parameter in population.named_parameters()
    }
    buffer_paths_by_identity = {}
    for module_path, module in modules:
        if isinstance(population, MultiPopulation) and module_path.startswith(
            "populations."
        ):
            # Each component is independently admitted below; the packed
            # parent never owns these child runtime/configuration slots.
            continue
        for buffer_name, value in module._buffers.items():
            if value is None or id(value) not in parameter_name_by_identity:
                continue
            buffer_paths_by_identity.setdefault(id(value), []).append(
                f"{module_path + '.' if module_path else ''}{buffer_name}"
            )

    consumed_parameter_ids = set()
    for _owner_path, owner in _parameterized_owner_paths(population):
        for transforms in owner.in_graph_parametrizations.values():
            for transform, args in transforms:
                consumed_parameter_ids.update(
                    id(parameter)
                    for parameter in transform.parameters()
                    if id(parameter) in parameter_name_by_identity
                )
                for argument_name in args:
                    if not isinstance(argument_name, str):
                        continue
                    if argument_name in owner._parameters:
                        value = owner._parameters[argument_name]
                    elif argument_name in owner._buffers:
                        value = owner._buffers[argument_name]
                    else:
                        continue
                    if id(value) in parameter_name_by_identity:
                        consumed_parameter_ids.add(id(value))

    semantic_paths = set()
    for leaf in _state_layout(population):
        semantic_paths.update(
            f"{module_path + '.' if module_path else ''}{buffer_name}"
            for module_path, buffer_name in leaf.mapping_slots
        )
    semantic_paths.update(
        f"{entry.module_path + '.' if entry.module_path else ''}{entry.buffer_name}"
        for entry in _prepared_buffer_layout(population)
    )
    try:
        geometry_bindings = _geometry_binding_layout(population)
    except FunctionalizationError:
        # Eligibility reports unsupported MaterialProcess/custom support
        # geometry through its dedicated diagnostics later in the same pass.
        geometry_bindings = ()
    for binding in geometry_bindings:
        semantic_paths.update(
            f"{module_path + '.' if module_path else ''}{binding.buffer_name}"
            for module_path in binding.module_paths
        )
    topology_kind = _functional_topology_kind(population)
    if topology_kind is not None:
        geometry_ids = {
            id(value)
            for name in _functional_geometry_constant_names(topology_kind)
            for value in (_geometry_constant_value(population, name),)
            if torch.is_tensor(value)
        }
        semantic_paths.update(
            path
            for identity in geometry_ids
            for path in buffer_paths_by_identity.get(identity, ())
        )
    for owner_path, owner in _parameterized_owner_paths(population):
        for name in _parameter_materialization_output_names(owner):
            if name in owner._buffers:
                semantic_paths.add(f"{owner_path + '.' if owner_path else ''}{name}")
    semantic_paths.update(
        path
        for _identity, paths in buffer_paths_by_identity.items()
        for path in paths
        if path.rsplit(".", 1)[-1] in {"celsius", "dt"}
    )
    integrator = getattr(population, "integrator", None)
    if integrator is not None:
        semantic_paths.update(
            f"integrator.{name}"
            for name, value in integrator._buffers.items()
            if value is not None
        )

    conflicts = []
    for identity, paths in buffer_paths_by_identity.items():
        semantic_aliases = tuple(sorted(set(paths) & semantic_paths))
        if identity not in consumed_parameter_ids or semantic_aliases:
            conflicts.append(
                (
                    parameter_name_by_identity[identity],
                    tuple(sorted(set(paths))),
                    semantic_aliases,
                )
            )
    return tuple(conflicts)


def _resolve_functional_integrator_spec(integrator, operator_kind):
    """Resolve an audited exact implementation or configuration-only wrapper."""
    if integrator is None:
        return None
    forbidden_instance_overrides = {
        "_PREPARED_WORKSPACE_SCHEMA",
        "_FUNCTIONAL_OPERATOR_KIND",
        "_FUNCTIONAL_CRITICAL_METHODS",
        *_FUNCTIONAL_INTEGRATOR_HOOKS,
    }
    if forbidden_instance_overrides & set(integrator.__dict__):
        return None
    try:
        spec = integrator._functional_spec()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if (
        not isinstance(spec, _FunctionalIntegratorSpec)
        or spec.operator_kind != operator_kind
        or set(spec.critical_methods) & set(integrator.__dict__)
    ):
        return None

    integrator_type = type(integrator)
    implementation = spec.implementation
    if integrator_type is implementation:
        recognized_type = True
    else:
        # Public factories use ``partial_class`` to create one direct subclass
        # which changes only __init__ argument defaults. Arbitrary subclasses
        # remain outside the audited numerical contract.
        recognized_type = (
            integrator_type.__module__ == "dendra.models.integrators"
            and integrator_type.__qualname__ == implementation.__qualname__
            and integrator_type.__bases__ == (implementation,)
            and set(integrator_type.__dict__) <= {"__module__", "__init__", "__doc__"}
        )
    if not recognized_type:
        return None

    try:
        methods_match = all(
            inspect.getattr_static(integrator_type, name)
            is inspect.getattr_static(implementation, name)
            for name in spec.critical_methods
        )
    except AttributeError:
        return None
    if not methods_match:
        return None
    return spec


def _functionalization_failures(
    population,
    dt: float,
    *,
    multi_component_candidates: dict[str, Population] | None = None,
) -> list[str]:
    failures = []
    if not isinstance(population, Population):
        return ["expected an initialized dendra Population"]
    if not population.initialized:
        failures.append("Population must be initialized before functionalization")
    is_multi = _is_scalar_multi(population)
    is_single_compartment = type(population) is SingleCompartment
    is_extcell_axon = isinstance(population, ExtCellAxon)
    is_extcell_tree = _is_extcell_tree(population)
    is_unmyelinated = isinstance(population, Unmyelinated)
    is_myelinated = isinstance(population, Myelinated)
    is_native_cable = _is_native_cable(population)
    is_scalar_tree = _is_scalar_tree(population)
    if is_multi:
        if not population.populations:
            failures.append("MultiPopulation must contain at least one component")
        if len({id(component) for component in population.populations.values()}) != len(
            population.populations
        ):
            failures.append(
                "MultiPopulation components must be distinct Population objects"
            )
        expected_width = sum(
            math.prod(component.core_shape())
            for component in population.populations.values()
        )
        if tuple(population.shape[-2:]) != (1, expected_width):
            failures.append(
                "MultiPopulation packed shape does not match its ordered component "
                "core widths"
            )
        batch_shape = tuple(population.shape[:-2])
        for name, component in population.populations.items():
            if isinstance(component, MultiPopulation):
                failures.append(
                    f"component {name!r} is a nested MultiPopulation, which is "
                    "not supported"
                )
                continue
            if tuple(component.shape[:-2]) != batch_shape:
                failures.append(
                    f"component {name!r} batch shape {tuple(component.shape[:-2])} "
                    f"does not match packed batch shape {batch_shape}"
                )
            try:
                candidate = _clone_execution_population(component)
                candidate.initialize(populate_parameter_buffers=False)
            except (TypeError, ValueError, RuntimeError) as exc:
                failures.append(
                    f"component {name!r} could not be prepared independently: {exc}"
                )
                continue
            component_failures = _functionalization_failures(candidate, dt)
            failures.extend(
                f"component {name!r}: {failure}" for failure in component_failures
            )
            if not component_failures and multi_component_candidates is not None:
                # Admission and lowering are one synchronous transaction. Keep
                # the already-isolated, initialized candidate so construction
                # does not repeat the expensive clone/initialize boundary.
                multi_component_candidates[name] = candidate
    elif is_extcell_axon:
        if not _uses_standard_extcell_axon_geometry(population):
            failures.append(
                "ExtCellAxon functionalization requires the standard cylindrical "
                "membrane-area geometry"
            )
    elif is_extcell_tree:
        canonical_failure = _validate_extcell_tree_source(population)
        if canonical_failure is not None:
            failures.append(canonical_failure)
    elif is_single_compartment:
        if not _uses_standard_single_compartment_geometry(population):
            failures.append(
                "SingleCompartment area overrides are not supported; standard "
                "Population cylindrical area semantics are required"
            )
    elif is_unmyelinated:
        if not _uses_standard_unmyelinated_geometry(population):
            failures.append(
                "subclasses overriding Unmyelinated area or axial geometry are not "
                "supported; exactly Unmyelinated geometry semantics are required"
            )
    elif is_myelinated:
        if not _uses_standard_myelinated_geometry(population):
            failures.append(
                "subclasses overriding Myelinated area or axial geometry are not "
                "supported; standard Myelinated geometry semantics are required"
            )
        if not _uses_standard_myelinated_parametrizations(population):
            failures.append(
                "Myelinated functionalization requires the standard registered "
                "diameter and axial-resistivity parametrizations"
            )
    elif is_native_cable:
        canonical_failure = _validate_native_cable_source(population)
        if canonical_failure is not None:
            failures.append(canonical_failure)
        for name in (
            "_canonical_area_cm2",
            "_canonical_edge_resistance_ohm",
        ):
            if not torch.is_tensor(getattr(population, name, None)):
                failures.append(
                    f"native Cable is missing canonical geometry tensor {name!r}"
                )
    elif is_scalar_tree:
        canonical_failure = _validate_scalar_tree_source(population)
        if canonical_failure is not None:
            failures.append(canonical_failure)
    else:
        failures.append(
            "the current supported topologies are exact SingleCompartment and "
            "Unmyelinated or Myelinated with standard geometry, plus exact "
            "native Cable, scalar Tree, or ExtCellTree instances built from a "
            "canonical morphology"
        )

    if is_extcell_axon or is_extcell_tree:
        topology = "ExtCellAxon" if is_extcell_axon else "ExtCellTree"
        for name in ("xraxial", "xc", "xg"):
            value = getattr(population, name, None)
            expected_shape = tuple(population.core_shape()) + (population.n_layers,)
            if not torch.is_tensor(value) or _shape_tuple(value) != expected_shape:
                failures.append(
                    f"{topology} {name} must use shared geometry shape {expected_shape}"
                )

    if not is_multi and (
        is_extcell_axon
        or is_extcell_tree
        or is_single_compartment
        or is_unmyelinated
        or is_myelinated
        or is_native_cable
        or is_scalar_tree
    ):
        geometry_shape = tuple(int(size) for size in population.core_shape())
        geometry_names = ["dx"]
        if is_extcell_axon or is_extcell_tree:
            geometry_names.extend(("diam", "xraxial", "xc", "xg"))
        elif is_myelinated:
            geometry_names.extend(("diameters", "diam_original"))
        else:
            geometry_names.append("diam")
        if is_native_cable or is_scalar_tree or is_extcell_tree:
            geometry_names.extend(
                ("canonical_area_cm2", "canonical_edge_resistance_ohm")
            )
        for name in geometry_names:
            value = _geometry_constant_value(population, name)
            expected_shape = (
                _shape_tuple(population.diameters)
                if name == "diameters"
                else (
                    geometry_shape + (population.n_layers,)
                    if (is_extcell_axon or is_extcell_tree)
                    and name in {"xraxial", "xc", "xg"}
                    else geometry_shape
                )
            )
            if not torch.is_tensor(value) or _shape_tuple(value) != expected_shape:
                failures.append(
                    f"{name} must use shared geometry shape {expected_shape}; "
                    "lane-specific geometry across explicit Population.batch() "
                    "replicas is not supported yet"
                )
    if population.device().type != "cpu":
        failures.append("the current supported device is CPU")
    if population.dtype() not in (torch.float32, torch.float64):
        failures.append("dtype must be float32 or float64")
    if is_multi:
        point_spec = _resolve_functional_integrator_spec(
            population.integrator,
            "scalar_multi_point",
        )
        packed_spec = _resolve_functional_integrator_spec(
            population.integrator,
            "scalar_multi",
        )
        if point_spec is None and packed_spec is None:
            failures.append(
                "MultiPopulation functionalization requires exactly "
                "bwd_euler_sc_multi or dhs_multi"
            )
        elif population.integrator.imem:
            failures.append("i_membrane recording is not supported yet")
        if tuple(population.integrator.v_vars) != ("v",):
            failures.append(
                "functional MultiPopulation integrators must declare exactly "
                "the ('v',) voltage-state contract"
            )
    elif is_extcell_axon:
        if (
            _resolve_functional_integrator_spec(
                population.integrator,
                "block_path",
            )
            is None
        ):
            failures.append(
                "ExtCellAxon functionalization requires exactly bwd_euler_bt"
            )
        elif population.integrator.imem:
            failures.append("i_membrane recording is not supported yet")
        elif population.integrator.method == "spd":
            failures.append(
                "functional bwd_euler_bt currently supports the Thomas method only"
            )
        if tuple(population.integrator.v_vars) != ("v", "vc"):
            failures.append(
                "functional bwd_euler_bt must declare exactly the ('v', 'vc') "
                "integrator state contract"
            )
    elif is_extcell_tree:
        if (
            _resolve_functional_integrator_spec(
                population.integrator,
                "block_tree",
            )
            is None
        ):
            failures.append("ExtCellTree functionalization requires exactly dhs_bt")
        elif population.integrator.imem:
            failures.append("i_membrane recording is not supported yet")
        if tuple(population.integrator.v_vars) != ("v", "vc"):
            failures.append(
                "functional dhs_bt must declare exactly the ('v', 'vc') "
                "integrator state contract"
            )
    elif is_single_compartment:
        if (
            _resolve_functional_integrator_spec(
                population.integrator,
                "scalar_point",
            )
            is None
        ):
            failures.append(
                "SingleCompartment functionalization requires exactly bwd_euler_sc"
            )
        elif population.integrator.imem:
            failures.append("i_membrane recording is not supported yet")
    elif is_unmyelinated or is_myelinated or is_native_cable:
        if (
            _resolve_functional_integrator_spec(
                population.integrator,
                "scalar_path",
            )
            is None
        ):
            topology = (
                "native Cable"
                if is_native_cable
                else ("Myelinated" if is_myelinated else "Unmyelinated")
            )
            failures.append(f"{topology} functionalization requires bwd_euler_ub")
        elif population.integrator.imem:
            failures.append("i_membrane recording is not supported yet")
        elif population.integrator.use_gc_variant:
            failures.append("the gradient-clipped Thomas variant is not supported yet")
    elif is_scalar_tree:
        if (
            _resolve_functional_integrator_spec(
                population.integrator,
                "scalar_tree",
            )
            is None
        ):
            failures.append("scalar Tree functionalization requires exactly dhs")
        elif population.integrator.imem:
            failures.append("i_membrane recording is not supported yet")

    handler = getattr(population, "mech", None)
    if handler is not None:
        handler_contract_matches = (
            type(handler) is MechanismHandler
            and not (set(_FUNCTIONAL_HANDLER_HOOKS) & set(handler.__dict__))
            and all(
                inspect.getattr_static(type(handler), name)
                is _STANDARD_FUNCTIONAL_HANDLER_HOOKS[name]
                for name in _FUNCTIONAL_HANDLER_HOOKS
            )
        )
        if not handler_contract_matches:
            failures.append(
                "functionalization requires the standard MechanismHandler "
                "current-frame evaluator"
            )
    mechanisms = () if handler is None else tuple(handler.mechanisms.values())
    if not mechanisms:
        failures.append("at least one membrane mechanism must be inserted")
    stateful_modules = []
    if handler is not None:
        stateful_modules.extend(handler.mechanisms.values())
        stateful_modules.extend(handler.material_processes.values())
        if handler.material_processes:
            failures.append(
                "MaterialProcesses have parameter-dependent spatial workspaces "
                "that are not lowered yet"
            )
    for mechanism in stateful_modules:
        mechanism_label = getattr(mechanism, "name", type(mechanism).__qualname__)
        class_has_assigned_values = (
            inspect.getattr_static(type(mechanism), "assigned_values")
            is not _STANDARD_MECHANISM_ASSIGNED_VALUES
        )
        class_has_advance = (
            inspect.getattr_static(type(mechanism), "advance")
            is not _STANDARD_MECHANISM_ADVANCE
        )
        if (
            bool(mechanism._has_authored_assigned_values) != class_has_assigned_values
            or bool(mechanism._has_authored_advance) != class_has_advance
        ):
            failures.append(
                f"{mechanism_label} changed its role-based transition hooks after "
                "construction; rebuild the Population before functionalization"
            )
        if class_has_assigned_values and not mechanism._assigned:
            failures.append(
                f"{mechanism_label} authors assigned_values but declares no "
                "ASSIGNED outputs"
            )
        for hook in ("assigned_values", "advance"):
            if hook in mechanism.__dict__:
                failures.append(
                    f"{mechanism_label} replaces {hook} on the instance; define "
                    "the pure returned-output hook on its Mechanism subclass"
                )

        support_map = getattr(mechanism, "support_map", None)
        if support_map is None:
            failures.append(f"{mechanism.name} must expose a structural SupportMap")
        elif support_map.spec.kind is SupportKind.ROWWISE:
            failures.append(
                f"{mechanism.name} uses reserved rowwise support, which is not "
                "lowered yet"
            )
        elif support_map.spec.kind is not SupportKind.DENSE:
            # Shared-field reads, ionic-current production, and additive
            # material sources already use the same pure gather/scatter-add
            # support plans as ordinary regional mechanism execution. Absolute
            # concentration/material replacement writes additionally need one
            # local slot per physical destination. PyTorch scatter has
            # inconsistent forward/reverse AD semantics for repeated
            # replacement indices, so keep that exact non-injective boundary
            # fail-closed.
            replaces_shared_fields = any(
                bool(values)
                for values in (
                    getattr(mechanism, "write_ion_c", {}),
                    getattr(mechanism, "write_material", {}),
                )
            )
            if replaces_shared_fields and not support_map.is_injective(
                getattr(mechanism, "key", None)
            ):
                failures.append(
                    f"{mechanism.name} uses non-injective regional shared-field "
                    "replacement writes, which are not lowered yet"
                )
        if not getattr(mechanism, "_support_key_values_valid", True):
            failures.append(
                f"{mechanism.name} support-key values are unavailable or invalid"
            )
        if mechanism._delayed_state_specs:
            failures.append(
                f"{mechanism.name} uses delayed state, which is not supported yet"
            )
        if mechanism._injection_specs:
            failures.append(f"{mechanism.name} uses a registered waveform injection")
        for state_name, state in mechanism.DE.items():
            state_label = f"{mechanism_label}.{state_name}"
            class_has_state_assigned_values = (
                inspect.getattr_static(type(state), "assigned_values")
                is not _STANDARD_STATE_ASSIGNED_VALUES
            )
            class_has_state_advance = (
                inspect.getattr_static(type(state), "advance")
                is not _STANDARD_STATE_ADVANCE
            )
            if (
                bool(state._has_authored_assigned_values)
                != class_has_state_assigned_values
                or bool(state._has_authored_advance) != class_has_state_advance
            ):
                failures.append(
                    f"{state_label} changed its role-based transition hooks after "
                    "construction; rebuild the Population before functionalization"
                )
            if class_has_state_assigned_values and not state._assigned:
                failures.append(
                    f"{state_label} authors assigned_values but declares no "
                    "ASSIGNED outputs"
                )
            for hook in ("assigned_values", "advance"):
                if hook in state.__dict__:
                    failures.append(
                        f"{state_label} replaces {hook} on the instance; define "
                        "the hook on its State subclass"
                    )

        for module in (mechanism, *mechanism.DE.values()):
            unresolved_layouts = _unresolved_deferred_layouts(module)
            if unresolved_layouts:
                failures.append(
                    f"{mechanism.name} has unresolved deferred buffer layouts "
                    f"{list(unresolved_layouts)}; initialize the Population before "
                    "functionalization"
                )
            if getattr(module.__class__, "_table", None):
                failures.append(
                    f"{mechanism.name} uses TABLE lookup workspaces, which are "
                    "not lowered yet"
                )
            if module.random_parameters:
                failures.append(
                    f"{mechanism.name} uses random parameters, which are not supported yet"
                )
            if module.runtime_noises or getattr(module, "_diffusion", None):
                failures.append(
                    f"{mechanism.name} uses stochastic state, which is not supported yet"
                )

    if handler is not None and handler.voltage_processes:
        failures.append("VoltageProcesses are not supported yet")
    if any(getattr(population, "mechanism_injection_accepted", ())):
        failures.append(
            "mechanism-owned waveform injections are not supported yet; only "
            "solver-owned model[slice].inject(waveform) registrations can be lowered"
        )

    parameter_contract_supported = True
    for owner in _parameterized_owners(population):
        if "_derive_parameter_buffers" in owner.__dict__ or inspect.getattr_static(
            type(owner), "_derive_parameter_buffers"
        ) is not inspect.getattr_static(Parameterized, "_derive_parameter_buffers"):
            failures.append(
                "custom parameter-buffer materializers are not supported yet"
            )
            parameter_contract_supported = False
            break
        if owner is population and is_myelinated:
            if not _uses_standard_myelinated_parametrizations(population):
                failures.append(
                    "custom Myelinated parameterizations are not supported yet"
                )
                parameter_contract_supported = False
            continue
        for transforms in owner.in_graph_parametrizations.values():
            for transform, _args in transforms:
                if not any(module is transform for module in owner.modules()):
                    failures.append(
                        "in-graph parameter transforms must be registered modules"
                    )
                    parameter_contract_supported = False
                    break
                if any(
                    _module_autograd_hook_signature(module)
                    for module in transform.modules()
                ):
                    failures.append(
                        "in-graph parameter transforms with registered hooks are "
                        "not supported"
                    )
                    parameter_contract_supported = False
                    break
                try:
                    _transform_structure_signature(transform, strict=True)
                except FunctionalizationError as exc:
                    failures.append(str(exc))
                    parameter_contract_supported = False
                    break
                registered_tensors = {
                    id(value)
                    for _name, value in (
                        *transform.named_parameters(remove_duplicate=False),
                        *transform.named_buffers(remove_duplicate=False),
                    )
                }
                if any(
                    torch.is_tensor(value) and id(value) not in registered_tensors
                    for value in vars(transform).values()
                ):
                    failures.append(
                        "in-graph parameter transform tensor dependencies must be "
                        "registered parameters or buffers"
                    )
                    parameter_contract_supported = False
                    break
            if not parameter_contract_supported:
                break
        if not parameter_contract_supported:
            break
        parametrizations = getattr(owner, "parametrizations", None)
        if parametrizations is not None and len(parametrizations):
            failures.append("torch parameterizations are not supported yet")
            parameter_contract_supported = False
            break

    if (
        population.initialized
        and parameter_contract_supported
        and (
            is_extcell_axon
            or is_extcell_tree
            or is_single_compartment
            or is_unmyelinated
            or is_myelinated
            or is_native_cable
            or is_scalar_tree
        )
    ):
        effective_failure = _effective_population_parameter_failure(population)
        if effective_failure is not None:
            failures.append(effective_failure)

    named_parameters = dict(population.named_parameters())
    if population.initialized:
        parameter_buffer_conflicts = _parameter_buffer_alias_conflicts(population)
        if parameter_buffer_conflicts:
            failures.append(
                "Parameter/buffer aliases may only back read-only in-graph "
                "transform dependencies; conflicting ownership="
                f"{parameter_buffer_conflicts}"
            )
    parameter_name_by_identity = {
        id(parameter): name for name, parameter in named_parameters.items()
    }
    topology_kind = _functional_topology_kind(population)
    if topology_kind is not None:
        geometry_parameter_aliases = []
        for geometry_name in _functional_geometry_constant_names(topology_kind):
            geometry_value = _geometry_constant_value(population, geometry_name)
            if (
                torch.is_tensor(geometry_value)
                and id(geometry_value) in parameter_name_by_identity
            ):
                geometry_parameter_aliases.append(
                    (geometry_name, parameter_name_by_identity[id(geometry_value)])
                )
        if geometry_parameter_aliases:
            failures.append(
                "canonical geometry tensors cannot also be registered Parameters; "
                f"cross-namespace aliases={geometry_parameter_aliases}"
            )
    if is_multi:
        required_population_parameters = {"celsius_param"}
    else:
        required_population_parameters = (
            _SINGLE_COMPARTMENT_POPULATION_PARAMETER_NAMES
            if is_extcell_axon
            or is_extcell_tree
            or is_single_compartment
            or is_native_cable
            or is_scalar_tree
            else _UNMYELINATED_POPULATION_PARAMETER_NAMES
        )
    missing_population_parameters = required_population_parameters - set(
        named_parameters
    )
    if missing_population_parameters:
        failures.append(
            "required Population parameters are missing: "
            f"{sorted(missing_population_parameters)}"
        )
    allowed_shapes = {}
    if is_native_cable or is_scalar_tree or is_extcell_tree:
        compartment_shape = (
            (1, int(population.core_shape()[-1]))
            if is_scalar_tree or is_extcell_tree
            else (int(population.core_shape()[-1]),)
        )
        allowed_shapes = {
            "cm_param.rho": compartment_shape,
            "rhoa_param.rho": compartment_shape,
        }
    regional_parameter_names = _additional_parameter_raw_names(population)
    transform_parameter_names = _in_graph_transform_parameter_names(population)
    shared_initial_targets = _shared_initial_parameter_target_shapes(
        population,
        parameter_name_by_identity,
    )
    shape_checked_parameters = (
        {
            name: parameter
            for name, parameter in named_parameters.items()
            if name == "celsius_param" or name.startswith("integrator.")
        }
        if is_multi
        else named_parameters
    )
    invalid_parameter_shapes = {}
    for name, parameter in shape_checked_parameters.items():
        if name in regional_parameter_names or name in transform_parameter_names:
            continue
        actual_shape = _shape_tuple(parameter)
        if name in shared_initial_targets:
            target_shapes = shared_initial_targets[name]
            if _shape_broadcasts_to_targets(actual_shape, target_shapes):
                continue
            expected = f"broadcastable to {target_shapes}"
        else:
            expected_shape = allowed_shapes.get(name, ())
            if actual_shape == expected_shape:
                continue
            expected = str(expected_shape)
        invalid_parameter_shapes[name] = (actual_shape, expected)
    if invalid_parameter_shapes:
        details = ", ".join(
            f"{name} has {actual}, expected {expected}"
            for name, (actual, expected) in sorted(invalid_parameter_shapes.items())
        )
        failures.append(f"unsupported raw parameter shapes: {details}")

    if not math.isfinite(dt) or dt <= 0.0:
        failures.append("dt must be a finite positive scalar")
    return failures


def _support_signature(module):
    support_map = getattr(module, "support_map", None)
    return None if support_map is None else support_map.spec.ordered_signature


def _support_key_modules(population):
    """Yield canonical parent-module paths whose selectors define placement."""

    handler = getattr(population, "mech", None)
    if handler is None:
        return
    for collection_name in ("mechanisms", "material_processes"):
        collection = getattr(handler, collection_name)
        for name, module in collection.items():
            yield f"integrator.mech.{collection_name}.{name}", module


def _support_key_runtime_binding(population) -> tuple:
    """Capture a cheap trace-time guard for every live tensor selector."""

    bindings = []
    for module_path, module in _support_key_modules(population):
        key = getattr(module, "key", None)
        if torch.is_tensor(key):
            try:
                version = key._version
            except RuntimeError:
                version = None
            key_binding = (
                id(key),
                version,
                _shape_tuple(key),
                key.dtype,
                key.device,
            )
        else:
            key_binding = None
        bindings.append(
            (
                module_path,
                bool(getattr(module, "_support_key_values_valid", True)),
                key_binding,
            )
        )
    return tuple(bindings)


def _compiled_support_keys(population) -> dict:
    """Return independent exact placement keys retained by Population.build()."""

    names = tuple(getattr(population, "_m_name", ()))
    keys = tuple(getattr(population, "_m_keys", ()))
    if len(names) != len(keys) or len(set(names)) != len(names):
        raise FunctionalizationError(
            "Population compiled mechanism-placement metadata is inconsistent; "
            "rebuild the Population before functionalizing it."
        )
    return dict(zip(names, keys, strict=True))


def _validate_support_keys(
    population,
    *,
    reference=None,
    context="Population",
) -> None:
    """Validate exact ordered support against immutable lowering-time keys."""

    compiled_keys = _compiled_support_keys(population)
    reference_modules = (
        None if reference is None else dict(_support_key_modules(reference))
    )
    for module_path, module in _support_key_modules(population):
        label = f"{context} mechanism {module.name!r}"
        support_map = getattr(module, "support_map", None)
        if support_map is None:
            raise FunctionalizationError(
                f"{label} has no structural SupportMap; rebuild the Population."
            )
        if not getattr(module, "_support_key_values_valid", True):
            raise FunctionalizationError(
                f"{label} selector values are unavailable or invalid; rebuild "
                "the Population from materialized placement data."
            )

        key = getattr(module, "key", None)
        if torch.is_tensor(key) and key.device.type == "meta":
            raise FunctionalizationError(
                f"{label} selector is on the meta device and has no values; "
                "rebuild the Population after materializing placement data."
            )
        try:
            support_map.spec.validate_runtime_key(key, context=f"{label} support")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise FunctionalizationError(str(exc)) from exc

        if support_map.spec.kind in {
            SupportKind.DENSE,
            SupportKind.RECTANGULAR,
        }:
            continue
        compiled_key = compiled_keys.get(module.name)
        if compiled_key is None:
            raise FunctionalizationError(
                f"{label} is missing its compiled placement key; rebuild the "
                "Population before functionalizing it."
            )
        if not torch.is_tensor(key):
            raise FunctionalizationError(
                f"{label} requires a materialized tensor selector."
            )
        expected_compiled_key = torch.as_tensor(
            compiled_key,
            dtype=key.dtype,
            device=key.device,
        ).reshape(-1)
        if _shape_tuple(key) != _shape_tuple(expected_compiled_key) or not torch.equal(
            key,
            expected_compiled_key,
        ):
            raise FunctionalizationError(
                f"{label} selector no longer matches the exact placement retained "
                "by Population.build(); rebuild before functionalizing it."
            )

        if reference_modules is None:
            continue
        reference_module = reference_modules.get(module_path)
        reference_key = (
            None if reference_module is None else getattr(reference_module, "key", None)
        )
        if not torch.is_tensor(key) or not torch.is_tensor(reference_key):
            raise FunctionalizationError(
                f"{label} requires a materialized tensor selector."
            )
        if (
            _shape_tuple(key) != _shape_tuple(reference_key)
            or key.dtype != reference_key.dtype
            or key.device != reference_key.device
            or not torch.equal(key, reference_key)
        ):
            raise FunctionalizationError(
                f"{label} selector changed after make_functional(); support keys "
                "are immutable, so lower the rebuilt Population again."
            )


def _parameter_module_signatures(population) -> tuple:
    bounded_names = (
        "min_val",
        "max_val",
        "beta",
        "threshold",
        "lower_mode",
        "lower_alpha",
        "cap_mode",
        "cap_beta",
        "_auto_promoted_hard_cap",
    )
    signatures = []
    for name, module in population.named_modules():
        is_parameter_source = isinstance(module, cacheable) or name.rsplit(".", 1)[
            -1
        ].endswith("_param")
        if not is_parameter_source:
            continue
        config = []
        for config_name in bounded_names:
            if hasattr(module, config_name):
                config.append((config_name, getattr(module, config_name)))
        for config_name, value in vars(module).items():
            if (
                config_name.startswith("_")
                or config_name == "training"
                or any(existing == config_name for existing, _ in config)
            ):
                continue
            if isinstance(value, (bool, int, float, str, type(None))):
                config.append((config_name, value))
        signatures.append(
            (
                name,
                type(module).__module__,
                type(module).__qualname__,
                tuple(sorted(config)),
            )
        )
    return tuple(signatures)


def _immutable_transform_config_signature(value, *, label: str, strict: bool):
    """Fingerprint immutable Python configuration or reject hidden state."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return (type(value).__qualname__, repr(value))
    if isinstance(value, (torch.dtype, torch.device)):
        return (type(value).__qualname__, str(value))
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, range):
        return ("range", value.start, value.stop, value.step)
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return (
            type(value).__qualname__,
            str(scalar.dtype),
            hashlib.sha256(scalar.tobytes()).hexdigest(),
        )
    if isinstance(value, tuple):
        return (
            type(value).__qualname__,
            tuple(
                _immutable_transform_config_signature(
                    item,
                    label=f"{label}[{index}]",
                    strict=strict,
                )
                for index, item in enumerate(value)
            ),
        )
    if isinstance(value, frozenset):
        items = [
            _immutable_transform_config_signature(
                item,
                label=f"{label}[]",
                strict=strict,
            )
            for item in value
        ]
        return ("frozenset", tuple(sorted(items, key=repr)))
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        if strict:
            raise FunctionalizationError(
                f"In-graph parameter transform state {label!r} is an unregistered "
                "Tensor or array; register numerical dependencies as parameters or "
                "buffers so they become explicit functional inputs."
            )
        return (
            "unsupported-numerical",
            type(value).__module__,
            type(value).__qualname__,
        )
    if isinstance(value, (list, set, dict, torch.Generator)):
        if strict:
            raise FunctionalizationError(
                f"In-graph parameter transform state {label!r} is mutable Python "
                f"configuration ({type(value).__qualname__}); use immutable literal "
                "configuration or registered tensor state."
            )
        return ("unsupported-mutable", type(value).__module__, type(value).__qualname__)
    if callable(value):
        if strict:
            raise FunctionalizationError(
                f"In-graph parameter transform state {label!r} is callable; express "
                "the operation in a registered torch.nn.Module class."
            )
        return (
            "unsupported-callable",
            type(value).__module__,
            type(value).__qualname__,
        )
    if strict:
        raise FunctionalizationError(
            f"In-graph parameter transform state {label!r} has unsupported type "
            f"{type(value).__module__}.{type(value).__qualname__}; use immutable "
            "literal configuration or registered tensor state."
        )
    return ("unsupported-opaque", type(value).__module__, type(value).__qualname__)


def _transform_code_structure_signature(code, *, label: str) -> tuple:
    """Return a stable, address-free fingerprint for Python function code."""
    constants = []
    for index, value in enumerate(code.co_consts):
        if inspect.iscode(value):
            signature = _transform_code_structure_signature(
                value,
                label=f"{label}.co_consts[{index}]",
            )
        else:
            signature = _immutable_transform_config_signature(
                value,
                label=f"{label}.co_consts[{index}]",
                strict=False,
            )
        constants.append(signature)
    return (
        code.co_name,
        getattr(code, "co_qualname", code.co_name),
        code.co_filename,
        code.co_firstlineno,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_code,
        tuple(constants),
        tuple(code.co_names),
        tuple(code.co_freevars),
    )


def _transform_callable_structure_signature(value, *, label: str, strict: bool):
    """Fingerprint immutable authored class behavior without using object ids."""
    descriptor_kind = type(value).__qualname__
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if isinstance(value, property):
        return (
            "property",
            *(
                (
                    None
                    if function is None
                    else _transform_callable_structure_signature(
                        function,
                        label=f"{label}.{accessor}",
                        strict=strict,
                    )
                )
                for accessor, function in (
                    ("fget", value.fget),
                    ("fset", value.fset),
                    ("fdel", value.fdel),
                )
            ),
        )
    if inspect.isclass(value):
        return ("class", value.__module__, value.__qualname__)
    if not (inspect.isroutine(value) or inspect.ismethoddescriptor(value)):
        if strict:
            raise FunctionalizationError(
                f"In-graph parameter transform class state {label!r} is a "
                "callable object; register it as an instance child Module or "
                "express the operation as class behavior without mutable state."
            )
        return (
            "unsupported-callable-object",
            type(value).__module__,
            type(value).__qualname__,
        )

    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    defaults = _immutable_transform_config_signature(
        getattr(target, "__defaults__", None),
        label=f"{label}.__defaults__",
        strict=strict,
    )
    kwdefaults = getattr(target, "__kwdefaults__", None)
    kwdefault_signature = (
        None
        if kwdefaults is None
        else tuple(
            (
                name,
                _immutable_transform_config_signature(
                    item,
                    label=f"{label}.__kwdefaults__.{name}",
                    strict=strict,
                ),
            )
            for name, item in sorted(kwdefaults.items())
        )
    )
    closure = getattr(target, "__closure__", None)
    closure_signature = (
        None
        if closure is None
        else tuple(
            (
                (
                    "class",
                    cell.cell_contents.__module__,
                    cell.cell_contents.__qualname__,
                )
                if inspect.isclass(cell.cell_contents)
                else _immutable_transform_config_signature(
                    cell.cell_contents,
                    label=f"{label}.__closure__[{index}]",
                    strict=strict,
                )
            )
            for index, cell in enumerate(closure)
        )
    )
    return (
        descriptor_kind,
        getattr(target, "__module__", None),
        getattr(target, "__qualname__", None),
        (
            None
            if code is None
            else _transform_code_structure_signature(code, label=f"{label}.__code__")
        ),
        defaults,
        kwdefault_signature,
        closure_signature,
    )


def _transform_class_structure_signature(
    module: torch.nn.Module,
    *,
    strict: bool,
) -> tuple:
    """Fingerprint authored transform class behavior and static configuration."""
    signatures = []
    for owner in type(module).__mro__:
        if owner in (torch.nn.Module, object) or owner.__module__.startswith("torch."):
            continue
        members = []
        for name, value in vars(owner).items():
            if name in _CLASS_STATE_METADATA_NAMES:
                continue
            label = f"{owner.__module__}.{owner.__qualname__}.{name}"
            if isinstance(value, torch.nn.Module):
                if strict:
                    raise FunctionalizationError(
                        f"In-graph parameter transform class state {label!r} is an "
                        "unregistered class-level Module; assign it to the transform "
                        "instance so its parameters and buffers become explicit."
                    )
                signature = (
                    "unsupported-class-module",
                    type(value).__module__,
                    type(value).__qualname__,
                )
            elif (
                inspect.isroutine(value)
                or inspect.isclass(value)
                or callable(value)
                or isinstance(value, (staticmethod, classmethod, property))
                or inspect.ismethoddescriptor(value)
            ):
                signature = _transform_callable_structure_signature(
                    value,
                    label=label,
                    strict=strict,
                )
            elif inspect.isdatadescriptor(value):
                signature = (
                    "descriptor",
                    type(value).__module__,
                    type(value).__qualname__,
                    name,
                )
            else:
                signature = _immutable_transform_config_signature(
                    value,
                    label=label,
                    strict=strict,
                )
            members.append((name, signature))
        signatures.append((owner.__module__, owner.__qualname__, tuple(members)))
    return tuple(signatures)


def _transform_structure_signature(
    transform: torch.nn.Module,
    *,
    strict: bool = False,
) -> tuple:
    """Fingerprint one transform tree, including immutable Python config."""
    modules = []
    for module_path, module in transform.named_modules(remove_duplicate=False):
        hooks = _module_autograd_hook_signature(module)
        configuration = []
        # Module attribute insertion order is structural and deterministic for
        # compatible constructors. Avoid sorting here: this signature is also
        # evaluated while Dynamo traces public atomic calls, where Python sort
        # over symbolic tuple variables is not fullgraph-compatible.
        for name, value in vars(module).items():
            if _is_internal_module_state(module, name) or name == "training":
                continue
            if name == "_cache" and isinstance(module, cacheable):
                continue
            # Framework-generated regional transforms retain a fill closure;
            # its exact placement/shape contract is fingerprinted separately
            # by _additional_parameter_layout_signature().
            if isinstance(module, ParameterFunctional) and name == "fill":
                continue
            if isinstance(value, torch.nn.Module):
                continue
            configuration.append(
                (
                    name,
                    _immutable_transform_config_signature(
                        value,
                        label=f"{module_path or '<root>'}.{name}",
                        strict=strict,
                    ),
                )
            )
        modules.append(
            (
                module_path,
                type(module).__module__,
                type(module).__qualname__,
                _effective_hook_identity(module, "forward"),
                hooks,
                _transform_class_structure_signature(module, strict=strict),
                tuple(configuration),
            )
        )
    return tuple(modules)


def _parametrization_signatures(population) -> tuple:
    """Fingerprint transform ordering, arguments, types, and effective hooks."""
    in_graph = tuple(
        (
            owner_path,
            tuple(
                (
                    name,
                    tuple(
                        (
                            tuple(args),
                            _transform_structure_signature(transform),
                        )
                        for transform, args in transforms
                    ),
                )
                for name, transforms in owner.in_graph_parametrizations.items()
            ),
        )
        for owner_path, owner in _parameterized_owner_paths(population)
        if owner.in_graph_parametrizations
    )
    parametrizations = getattr(population, "parametrizations", None)
    torch_registered = (
        ()
        if parametrizations is None
        else tuple(
            (
                name,
                _shape_tuple(transforms.original),
                _module_autograd_hook_signature(transforms),
                tuple(
                    _transform_structure_signature(transform)
                    for transform in transforms
                ),
            )
            for name, transforms in parametrizations.items()
        )
    )
    return in_graph, torch_registered


def _configuration_signature(value):
    """Convert small execution configuration values into comparable structure."""
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (str(key), _configuration_signature(item))
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_configuration_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_configuration_signature(item) for item in value))
    if isinstance(value, (bool, int, float, str, type(None))):
        return (type(value).__qualname__, value)
    if isinstance(value, (torch.dtype, torch.device)):
        return (type(value).__qualname__, str(value))
    return (
        type(value).__module__,
        type(value).__qualname__,
        repr(value),
    )


def _flag_signature(module) -> tuple:
    return tuple(
        sorted(
            (name, _configuration_signature(getattr(module, name)))
            for name in getattr(module, "flags", {})
        )
    )


def _static_hook_identity(module, name: str) -> tuple:
    """Identify the effective authored hook without descriptor evaluation."""
    for owner in type(module).__mro__:
        if name in owner.__dict__:
            value = owner.__dict__[name]
            return (owner.__module__, owner.__qualname__, id(value))
    raise AttributeError(f"{type(module).__qualname__} has no hook {name!r}")


def _effective_hook_identity(module, name: str) -> tuple:
    """Identify a class hook plus any instance-level replacement."""

    if name in module.__dict__:
        value = module.__dict__[name]
        function = getattr(value, "__func__", value)
        return (
            "instance",
            type(value).__module__,
            type(value).__qualname__,
            id(value),
            id(function),
        )
    return ("class", *_static_hook_identity(module, name))


def _material_signature(name, material) -> tuple:
    field_specs = tuple(
        (
            field,
            material.field_spec(field).min_value,
            material.field_spec(field).conserved,
            material.field_spec(field).domain,
            material.field_spec(field).units,
        )
        for field in material.fields
    )
    return (
        name,
        type(material).__module__,
        type(material).__qualname__,
        field_specs,
        getattr(material, "rzf", None),
        getattr(material, "min_concentration", None),
        getattr(material, "init_e_reversal", None),
        getattr(material, "advance_e", None),
    )


def _structure_signature(population) -> tuple:
    handler = population.mech
    multi_components = (
        _multi_component_topology_signature(population)
        if _is_scalar_multi(population)
        else None
    )
    canonical_cable_identity = None
    if _is_native_cable(population):
        fingerprint = population._canonical_morphology_fingerprint_reference
        canonical_cable_identity = (
            fingerprint,
            _shape_tuple(population._canonical_area_cm2),
            _shape_tuple(population._canonical_edge_resistance_ohm),
        )
    canonical_tree_topology = None
    if _is_canonical_tree(population):
        canonical_tree_topology = tuple(
            int(parent) for parent in population.compartment_graph.topology.parent_index
        )

    def module_signature(name, module):
        return (
            name,
            type(module).__module__,
            type(module).__qualname__,
            _support_signature(module),
            _flag_signature(module),
            _additional_parameter_schema(module),
            tuple(
                (
                    state_name,
                    type(state).__module__,
                    type(state).__qualname__,
                    tuple(state._state),
                    _declared_buffer_layout_signature(
                        state,
                        state._carry,
                        specs_attribute="_carry_specs",
                        resolved_attribute="_carry_resolved_shapes",
                    ),
                    tuple(state._assigned),
                    _declared_buffer_layout_signature(
                        state,
                        _derived_buffer_names(state),
                        specs_attribute="_derived_buffer_specs",
                        resolved_attribute="_derived_resolved_shapes",
                    ),
                    tuple(_timestep_buffer_names(state)),
                    _timestep_buffer_shape_specs(state),
                    tuple(
                        (hook, _effective_hook_identity(state, hook))
                        for hook in (
                            "assigned_values",
                            "advance",
                        )
                    ),
                    _transform_callable_structure_signature(
                        state._solve,
                        label=(
                            f"{type(state).__module__}."
                            f"{type(state).__qualname__}._solve"
                        ),
                        strict=False,
                    ),
                    _effective_hook_identity(state, "derive_buffers"),
                    _effective_hook_identity(state, "derive_timestep_buffers"),
                    _flag_signature(state),
                    _additional_parameter_schema(state),
                    state.method,
                    _configuration_signature(state.method_kwargs),
                )
                for state_name, state in module.DE.items()
            ),
            _declared_buffer_layout_signature(
                module,
                module._carry,
                specs_attribute="_carry_specs",
                resolved_attribute="_carry_resolved_shapes",
            ),
            tuple(sorted(module._assigned)),
            _declared_buffer_layout_signature(
                module,
                _derived_buffer_names(module),
                specs_attribute="_derived_buffer_specs",
                resolved_attribute="_derived_resolved_shapes",
            ),
            tuple(_timestep_buffer_names(module)),
            _timestep_buffer_shape_specs(module),
            _effective_hook_identity(module, "derive_buffers"),
            _effective_hook_identity(module, "derive_timestep_buffers"),
            tuple(
                (hook, _effective_hook_identity(module, hook))
                for hook in (
                    "assigned_values",
                    "advance",
                    "_advance_states",
                    "_evaluate_assigned",
                )
            ),
            tuple(sorted(module._save)),
        )

    return (
        (type(population).__module__, type(population).__qualname__),
        _additional_parameter_schema(population),
        multi_components,
        canonical_cable_identity,
        canonical_tree_topology,
        tuple(population.shape),
        tuple(
            (name, _shape_tuple(getattr(population, name))) for name in ("diam", "dx")
        ),
        population.device().type,
        population.dtype(),
        (
            type(population.integrator).__module__,
            type(population.integrator).__qualname__,
            getattr(population.integrator, "method", None),
            getattr(population.integrator, "threads", None),
            getattr(population.integrator, "write_back", None),
            bool(population.integrator.imem),
            bool(getattr(population.integrator, "use_gc_variant", False)),
            tuple(population.integrator.v_vars),
            tuple(population.integrator._PREPARED_WORKSPACE_SCHEMA),
            getattr(population.integrator, "_FUNCTIONAL_OPERATOR_KIND", None),
            tuple(
                getattr(
                    population.integrator,
                    "_FUNCTIONAL_CRITICAL_METHODS",
                    (),
                )
            ),
            tuple(
                (
                    name,
                    _effective_hook_identity(population.integrator, name),
                )
                for name in dict.fromkeys(
                    (
                        *_FUNCTIONAL_INTEGRATOR_HOOKS,
                        *getattr(
                            population.integrator,
                            "_FUNCTIONAL_CRITICAL_METHODS",
                            (),
                        ),
                    )
                )
                if hasattr(population.integrator, name)
            ),
        ),
        (
            type(handler).__module__,
            type(handler).__qualname__,
            tuple(
                (name, _effective_hook_identity(handler, name))
                for name in _FUNCTIONAL_HANDLER_HOOKS
            ),
        ),
        tuple(
            module_signature(name, module)
            for name, module in handler.mechanisms.items()
        ),
        tuple(
            module_signature(name, module)
            for name, module in handler.material_processes.items()
        ),
        tuple(_material_signature(name, ion) for name, ion in handler.ions.items()),
        tuple(
            _material_signature(name, material)
            for name, material in handler.materials.items()
        ),
        tuple(handler.voltage_processes),
        _parametrization_signatures(population),
        _parameter_module_signatures(population),
        tuple(
            (name, _shape_tuple(parameter))
            for name, parameter in population.named_parameters()
        ),
        tuple(
            (name, bool(module.training)) for name, module in population.named_modules()
        ),
    )


def _structure_ids(population) -> tuple[int, ...]:
    return tuple(id(module) for _name, module in population.named_modules())


@torch.compiler.assume_constant_result
def _compiled_structure_signature_matches(population, fingerprint) -> bool:
    """Evaluate the Python-only source fingerprint once per compiled graph."""
    return _structure_signature(population) == fingerprint


@torch.compiler.assume_constant_result
def _compiled_initialization_failures(population) -> tuple[str, ...]:
    """Evaluate Python-only initializer admission once per compiled graph."""
    return tuple(_functional_initialization_failures(population))


@torch.compiler.assume_constant_result
def _compiled_initialization_structure_matches(population, fingerprint) -> bool:
    """Evaluate initializer-only source freshness once per compiled graph."""
    return _initialization_structure_signature(population) == fingerprint


class FunctionalPopulation:
    """Immutable execution plan for supported deterministic scalar models.

    Construct instances with :func:`make_functional`. Existing Population
    execution is not redirected through this reference path.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The immutable state payload shape accepted by this plan."""
        return self._shape

    @property
    def dt(self) -> float:
        """The default timestep used when this plan was lowered."""
        return self._dt

    @property
    def dtype(self) -> torch.dtype:
        """The immutable tensor dtype accepted by this plan."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """The immutable tensor device accepted by this plan."""
        return self._device

    @property
    def intra(self) -> FunctionalIntra:
        """The immutable plan for registered intracellular Waveforms."""
        return self._intra

    @property
    def extra(self) -> FunctionalExtra:
        """The immutable plan for an ``extra=`` spec bound while lowering."""
        return self._extra

    def __init__(
        self,
        population: Population,
        *,
        dt: float,
        extra=None,
        _multi_component_candidates: Mapping[str, Population] | None = None,
    ):
        if not getattr(population, "is_built", False) or getattr(
            population, "_flag_rebuild", False
        ):
            raise FunctionalizationError(
                "Population must be fully built with no pending structural "
                "changes before functionalization."
            )
        _validate_support_keys(population, context="Source Population")
        self._source_support_key_binding = _support_key_runtime_binding(population)
        self._source_additional_parameter_key_binding = (
            _additional_parameter_key_binding(population)
        )
        self._additional_parameter_layout = _additional_parameter_layout_signature(
            population
        )
        self._shape = tuple(population.shape)
        # ``Population.batch()`` prepends explicit state axes while retaining
        # one shared morphology over the final population/compartment axes.
        # The implicit cable solver has a third representation: every leading
        # state axis is flattened into its matrix-batch dimension.  Keep these
        # contracts distinct instead of expanding shared geometry into state
        # storage or passing public state shapes to the tridiagonal solver.
        self._geometry_shape = tuple(int(size) for size in population.core_shape())
        self._parameter_shape = tuple(int(size) for size in population._calc_shape_p())
        self._solve_shape = tuple(int(size) for size in population.batched_shape())
        self._topology_kind = _functional_topology_kind(population)
        if self._topology_kind is None:  # pragma: no cover - admission precedes init
            raise FunctionalizationError(
                "The Population no longer satisfies an exact functional topology."
            )
        self._geometry_constant_names = _functional_geometry_constant_names(
            self._topology_kind
        )
        self._geometry_constant_shapes = {
            name: (
                self._shape
                if self._topology_kind in {"multi_point", "multi_scalar"}
                else (
                    _shape_tuple(population.diameters)
                    if name == "diameters"
                    else (
                        self._geometry_shape + (population.n_layers,)
                        if self._topology_kind in {"extcell_axon", "extcell_tree"}
                        and name in {"xraxial", "xc", "xg"}
                        else self._geometry_shape
                    )
                )
            )
            for name in self._geometry_constant_names
        }
        self._parametrization_constants = _parametrization_constant_layout(population)
        self._parametrization_constant_layout_signature = (
            _parametrization_constant_layout_signature(population)
        )
        self._dt = float(dt)
        self._dtype = population.dtype()
        self._device = population.device()
        self._population = population
        self._source_ids = _structure_ids(population)
        self._fingerprint = _structure_signature(population)
        self._initialization_fingerprint = _initialization_structure_signature(
            population
        )
        self._canonical_source_binding = (
            _multi_canonical_source_binding(population)
            if self._topology_kind in {"multi_point", "multi_scalar"}
            else (
                _native_cable_source_binding(population)
                if self._topology_kind == "native_cable"
                else None
            )
        )
        self._multi_canonical_values = (
            _multi_canonical_value_signature(population)
            if self._topology_kind in {"multi_point", "multi_scalar"}
            else None
        )
        self._prepared_token = object()
        self._intra = FunctionalIntra(population)
        self._extra = FunctionalExtra(
            population,
            extra,
            namespace="stimulation.extra",
        )
        if (
            self._intra._waveform_source_identities
            & self._extra._waveform_source_identities
        ):
            raise FunctionalizationError(
                "The same Waveform object cannot be shared between registered "
                "intracellular stimulation and extra bound through "
                "make_functional(..., extra=...). The two plans expose "
                "independent tensor namespaces; use distinct Waveform instances "
                "so their explicit leaves cannot silently diverge."
            )
        if self._extra.enabled and not self._extra.uses_waveforms:
            raise FunctionalizationError(
                "make_functional(..., extra=...) requires Waveform temporal "
                "specifications. Tensor-temporal extra has a call-local time "
                "horizon; pass it at runtime through step(), rollout(), "
                "dn.func.run(), longrun(), or longrun_checkpointed() instead."
            )

        self._multi_component_plans = ()
        self._multi_public_constant_shapes = {}
        self._multi_public_constant_dtypes = {}
        self._multi_public_constant_devices = {}
        self._multi_canonical_parameter_names = None
        self._multi_component_physical_parameter_names = frozenset()
        self._initialization_failures = tuple(
            _functional_initialization_failures(population)
        )

        execution_population = _clone_execution_population(population)
        dt_tensor = torch.as_tensor(self.dt, device=self.device, dtype=self.dtype)
        has_timestep_buffers = any(
            _timestep_buffer_names(_module_at(execution_population, module_path))
            for module_path in _parameterized_module_paths(execution_population)
        )
        try:
            execution_population.integrator._initialize(
                execution_population,
                dt_tensor,
                force=execution_population.force_integrator_reinit(),
                compile_scope="population",
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            if not has_timestep_buffers:
                raise
            raise FunctionalizationError(
                "timestep-buffer materialization failed while lowering the "
                f"Population: {exc}"
            ) from exc
        if self._topology_kind in {"multi_point", "multi_scalar"}:
            # The packed integrator and consolidated MechanismHandler now own
            # the complete transition. Component Populations are source-side
            # configuration/commit objects only; retaining their registered
            # runtime trees would make every functional mapping clone unused
            # voltage, mechanism, ion, and child-integrator tensors.
            _strip_packed_component_runtime_aliases(execution_population)
        operator_kind = {
            "single_compartment": "scalar_point",
            "extcell_axon": "block_path",
            "extcell_tree": "block_tree",
            "scalar_tree": "scalar_tree",
            "multi_point": "scalar_multi_point",
            "multi_scalar": "scalar_multi",
        }.get(self._topology_kind, "scalar_path")
        integrator_spec = _resolve_functional_integrator_spec(
            execution_population.integrator,
            operator_kind,
        )
        if integrator_spec is None:  # pragma: no cover - source was prevalidated
            raise FunctionalizationError(
                "The cloned integrator no longer satisfies its functional contract."
            )
        try:
            solver = execution_population.integrator._functional_solver()
        except RuntimeError as exc:
            raise FunctionalizationError(str(exc)) from exc
        self._integrator_workspace_schema = integrator_spec.workspace_schema
        self._integrator_workspace_names = tuple(
            name for name, _shape_role in self._integrator_workspace_schema
        )
        edge_shape = (*self._solve_shape[:-1], self._solve_shape[-1] - 1)
        block_size = int(getattr(execution_population.integrator, "M", 1))
        workspace_shapes = {
            "node": self._solve_shape,
            "edge": edge_shape,
            "parameter": self._parameter_shape,
            "geometry": self._geometry_shape,
            "block_node": self._solve_shape + (block_size,),
            "block_edge": edge_shape + (block_size,),
            "block_matrix": self._solve_shape + (block_size, block_size),
            "shell_node": self._solve_shape + (block_size - 1,),
        }
        if self._topology_kind == "multi_point":
            workspace_shapes["parameter"] = self.shape
            workspace_shapes["geometry"] = self.shape
        elif self._topology_kind == "multi_scalar":
            planes = math.prod(self.shape[:-2]) or 1
            workspace_shapes.update(
                {
                    "multi_plane": (
                        planes,
                        int(execution_population.integrator.B_total),
                        int(execution_population.integrator.K_stride),
                    ),
                    "multi_node": (planes, int(self.shape[-1])),
                    "multi_edge": (
                        planes,
                        int(execution_population.integrator.EDGE_GAX_FLAT.shape[-1]),
                    ),
                }
            )
        unknown_workspace_roles = {
            shape_role
            for _name, shape_role in self._integrator_workspace_schema
            if shape_role not in workspace_shapes
        }
        if unknown_workspace_roles:
            raise FunctionalizationError(
                f"{type(execution_population.integrator).__qualname__} declares "
                "unsupported prepared-workspace shape roles "
                f"{sorted(unknown_workspace_roles)}"
            )
        self._integrator_workspace_shapes = {
            name: workspace_shapes[shape_role]
            for name, shape_role in self._integrator_workspace_schema
        }
        self._state_layout = _state_layout(execution_population)
        self._state_layout_signature = _state_layout_signature(self._state_layout)
        self._initialization_state_layout = _initialization_state_layout(
            execution_population,
            self._state_layout,
        )
        self._initialization_transform_inputs = _initialization_transform_input_layout(
            execution_population
        )
        self._transition = _PopulationTransition(
            execution_population,
            solver,
            self._state_layout,
        )
        _validate_support_keys(
            population,
            reference=execution_population,
            context="Source Population",
        )
        self._base_mapping = {
            name: _clone_tensor(value)
            for name, value in _module_tensor_slots(self._transition)
        }
        self._point_area_factors = _point_area_factor_layout(execution_population)
        missing_point_factor_slots = sorted(
            f"population.{entry.module_path}.{entry.buffer_name}"
            for entry in self._point_area_factors
            if f"population.{entry.module_path}.{entry.buffer_name}"
            not in self._base_mapping
        )
        if missing_point_factor_slots:
            raise FunctionalizationError(
                "Functional PointProcess area workspaces lost registered slots: "
                f"{missing_point_factor_slots}"
            )
        self._geometry_bindings = _geometry_binding_layout(execution_population)
        self._local_geometry_bindings = tuple(
            binding
            for binding in self._geometry_bindings
            if _cache_local_geometry(binding)
        )
        self._local_geometry_shapes = tuple(
            _shape_tuple(_module_at(execution_population, binding.module_paths[0]).diam)
            for binding in self._local_geometry_bindings
        )
        self._mechanism_dt_slots = _mechanism_dt_slots(execution_population)
        self._mechanism_celsius_slots = tuple(
            name
            for name in self._base_mapping
            if name == "population.celsius" or name.endswith(".celsius")
        )
        self._mutable_mapping_slots = frozenset(
            f"population.{module_path + '.' if module_path else ''}{buffer_name}"
            for leaf in self._state_layout
            for module_path, buffer_name in leaf.mapping_slots
        ) | frozenset(
            (
                *_local_synchronization_slots(execution_population),
                *_ephemeral_assigned_slots(execution_population),
            )
        )
        # Direct construction remains supported for internal tests and
        # extensions. Until the one-time purity audit succeeds, transition
        # mappings defensively isolate every read-only input from the caller.
        self._lowering_audit_complete = False

        # Multi component runtime trees have already been removed above, so
        # this clone carries only the packed owner.  This avoids deep-copying
        # every discarded child mechanism/integrator/state tree a second time
        # merely to construct the parent preparation skeleton.
        preparation_population = _clone_execution_population(execution_population)
        self._preparation = _PopulationPreparation(preparation_population, dt_tensor)
        if _geometry_binding_signature(self._preparation.geometry_bindings) != (
            _geometry_binding_signature(self._geometry_bindings)
        ):
            raise FunctionalizationError(
                "Preparation and transition geometry plans do not match."
            )
        self._prepared_buffers = self._preparation.layout
        self._prepared_buffer_visible_ranks = {
            entry.key: (
                len(_module_at(execution_population, entry.module_path).shape_f)
                if entry.kind == "effective_parameter"
                else 0
            )
            for entry in self._prepared_buffers
        }
        self._prepared_workspace_slots = tuple(
            f"population.{entry.module_path}.{entry.buffer_name}"
            for entry in self._prepared_buffers
            if entry.kind in {"derived", "timestep"}
        )
        if any(name is not None for name in self._preparation.offset_names):
            self._initialization_failures = (
                *self._initialization_failures,
                "live effective parameters differ from their raw deterministic "
                "materialization; initialization-time parameter transforms need "
                "an explicit pure output contract",
            )
        if self._topology_kind in {"multi_point", "multi_scalar"}:
            self._build_multi_component_plans(
                population,
                candidates=_multi_component_candidates,
            )
        self._preparation_base_mapping = {
            name: _clone_tensor(value)
            for name, value in _module_tensor_slots(self._preparation)
            if not name.startswith("component_physical_preparations.")
        }
        if self._topology_kind in {"multi_point", "multi_scalar"}:
            leaked_component_runtime = sorted(
                name
                for mapping in (self._base_mapping, self._preparation_base_mapping)
                for name in mapping
                if name.startswith("population.populations.")
            )
            if leaked_component_runtime:  # pragma: no cover - lowering invariant
                raise FunctionalizationError(
                    "Packed functional runtime retained component Population "
                    f"tensor slots {leaked_component_runtime}."
                )
        missing_parametrization_constants = sorted(
            f"population.{alias}"
            for entry in self._parametrization_constants
            for alias in entry.aliases
            if f"population.{alias}" not in self._preparation_base_mapping
        )
        if missing_parametrization_constants:
            raise FunctionalizationError(
                "Functional preparation lost registered in-graph parametrization "
                f"buffers: {missing_parametrization_constants}"
            )

        all_population_parameters = dict(population.named_parameters())
        population_parameters = (
            {
                name: all_population_parameters[name]
                for name in self._multi_canonical_parameter_names
            }
            if self._multi_canonical_parameter_names is not None
            else all_population_parameters
        )
        self._population_parameter_names = tuple(population_parameters)
        parameter_aliases = _parameter_alias_paths(population)
        self._transition_parameter_slots = {
            name: tuple(
                slot
                for path in parameter_aliases[name]
                if (slot := f"population.{path}") in self._base_mapping
            )
            for name in self._population_parameter_names
        }
        self._preparation_parameter_slots = {
            name: tuple(
                slot
                for path in parameter_aliases[name]
                if (slot := f"population.{path}") in self._preparation_base_mapping
            )
            for name in self._population_parameter_names
        }
        missing_transition_parameters = sorted(
            name
            for name, slots in self._transition_parameter_slots.items()
            if not slots and name not in self._multi_component_physical_parameter_names
        )
        missing_preparation_parameters = sorted(
            name
            for name, slots in self._preparation_parameter_slots.items()
            if not slots and name not in self._multi_component_physical_parameter_names
        )
        if missing_transition_parameters or missing_preparation_parameters:
            raise FunctionalizationError(
                "Functional Population clones lost canonical parameter slots; "
                f"transition missing={missing_transition_parameters}, "
                f"preparation missing={missing_preparation_parameters}"
            )
        stimulation_schema = {}
        for plan in (self._intra, self._extra):
            stimulation_schema.update(plan.parameter_schema)
            stimulation_schema.update(plan.constant_schema)
        overlap = set(population_parameters) & set(stimulation_schema)
        if overlap:  # pragma: no cover - namespaces make this defensive only
            raise FunctionalizationError(
                f"stimulation tensor names collide with Population parameters: "
                f"{sorted(overlap)}"
            )
        self._stimulation_schema = stimulation_schema
        self._parameter_names = (
            *self._population_parameter_names,
            *tuple(stimulation_schema),
        )
        self._parameter_shapes = {
            name: _shape_tuple(parameter)
            for name, parameter in population_parameters.items()
        }
        self._parameter_shapes.update(
            {name: contract.shape for name, contract in stimulation_schema.items()}
        )
        self._parameter_dtypes = {
            name: parameter.dtype for name, parameter in population_parameters.items()
        }
        self._parameter_dtypes.update(
            {name: contract.dtype for name, contract in stimulation_schema.items()}
        )
        self._parameter_devices = {
            name: parameter.device for name, parameter in population_parameters.items()
        }
        self._parameter_devices.update(
            {name: contract.device for name, contract in stimulation_schema.items()}
        )
        self._state_schema = self._make_state_schema(self._state_layout)
        self._initialization = None
        self._initialization_state_template = None
        if not self._initialization_failures:
            initialization_population = _clone_execution_population(
                execution_population
            )
            initialization_layout = _state_layout(initialization_population)
            if _state_layout_signature(initialization_layout) != (
                self._state_layout_signature
            ):
                raise FunctionalizationError(
                    "Functional initialization and transition state layouts do "
                    "not match."
                )
            initialization = _PopulationInitialization(
                initialization_population,
                initialization_layout,
                _initialization_state_layout(
                    initialization_population,
                    initialization_layout,
                ),
            )
            initialization_slots = {
                name for name, _value in _module_tensor_slots(initialization)
            }
            transition_slots = set(self._base_mapping)
            if initialization_slots != transition_slots:
                raise FunctionalizationError(
                    "Functional initialization and transition tensor bindings do "
                    "not match."
                )
            state_template = {}
            for leaf in self._state_layout:
                value = _buffer_at(
                    execution_population,
                    leaf.module_path,
                    leaf.buffer_name,
                )
                if leaf.public_path in {
                    ("clock", "t"),
                    ("control", "duration_remainder"),
                }:
                    value = torch.zeros_like(value)
                _put_path(
                    state_template,
                    leaf.public_path,
                    _clone_tensor(value),
                )
            self._initialization = initialization
            self._initialization_state_template = state_template
        self._structured_step_graphs = {}
        self._structured_capture_errors = {}
        self._structured_capture_error = None

    def _build_multi_component_plans(
        self,
        population: MultiPopulation,
        *,
        candidates: Mapping[str, Population] | None,
    ) -> None:
        """Build narrow physical adapters and canonical public leaves."""
        parent_parameters = dict(population.named_parameters())
        canonical_by_identity = {
            id(parameter): name for name, parameter in parent_parameters.items()
        }
        canonical_parameter_names = {
            name
            for name in parent_parameters
            if name == "celsius_param" or name.startswith("integrator.")
        }

        public_constant_shapes = {}
        public_constant_dtypes = {}
        public_constant_devices = {}
        # Retain each source object while deduplicating. Some canonical
        # geometry properties return fresh Tensor views; retaining the object
        # prevents Python id reuse from being mistaken for a true alias later
        # in the component walk.
        constant_by_identity = {}
        records = []
        physical_preparations = []
        packed_integrator = self._transition.population.integrator
        solver_cursor = 0

        for component_index, (
            component_name,
            source_component,
        ) in enumerate(population.populations.items()):
            component = None if candidates is None else candidates.get(component_name)
            if component is None:
                component = _clone_execution_population(source_component)
                component.initialize(populate_parameter_buffers=False)
            topology_kind = _functional_topology_kind(component)
            group_k = (
                int(packed_integrator.group_K[component_index])
                if self._topology_kind == "multi_scalar"
                else math.prod(source_component.core_shape())
            )
            if topology_kind == "scalar_tree":
                parent_index = torch.as_tensor(
                    source_component.compartment_graph.topology.parent_index,
                    device=source_component.device(),
                    dtype=torch.int64,
                )
                edge_child_orig = torch.nonzero(
                    parent_index >= 0,
                    as_tuple=False,
                ).flatten()
                solver_order = packed_integrator.SOLVER_cat[
                    solver_cursor : solver_cursor + group_k
                ]
            else:
                edge_child_orig = None
                solver_order = None
            solver_cursor += group_k
            component_preparation = _ComponentPhysicalPreparation(
                component,
                edge_child_orig=edge_child_orig,
                solver_order=solver_order,
            )
            physical_preparations.append(component_preparation)
            topology_kind = component_preparation.topology_kind
            geometry_constant_names = _functional_geometry_constant_names(topology_kind)

            source_parameters = dict(source_component.named_parameters())
            parameter_sources = []
            component_parameters = dict(
                component_preparation.population.named_parameters()
            )
            component_parameter_aliases = _parameter_alias_paths(
                component_preparation.population
            )
            for relative_name in component_parameters:
                if relative_name == "celsius_param":
                    # Concatenation has one shared runtime temperature. Bind a
                    # physical transform that explicitly consumes celsius to
                    # that same public leaf instead of hiding a stale child
                    # temperature snapshot in the adapter.
                    public_name = "celsius_param"
                else:
                    try:
                        source_parameter = source_parameters[relative_name]
                        public_name = canonical_by_identity[id(source_parameter)]
                    except KeyError as exc:
                        raise FunctionalizationError(
                            f"MultiPopulation component {component_name!r} lost "
                            f"canonical physical parameter {relative_name!r}."
                        ) from exc
                if public_name not in parent_parameters:
                    raise FunctionalizationError(
                        f"MultiPopulation component {component_name!r} physical "
                        f"parameter {relative_name!r} routes to missing public "
                        f"parameter {public_name!r}."
                    )
                canonical_parameter_names.add(public_name)
                parameter_sources.append((relative_name, public_name))

            routed_parameter_names = [
                relative_name for relative_name, _public_name in parameter_sources
            ]
            component_parameter_names = set(component_parameters)
            if (
                len(routed_parameter_names) != len(set(routed_parameter_names))
                or set(routed_parameter_names) != component_parameter_names
                or set(component_parameter_aliases) != component_parameter_names
            ):
                raise FunctionalizationError(
                    f"MultiPopulation component {component_name!r} physical "
                    "parameter routing does not cover each canonical adapter "
                    "parameter exactly once."
                )
            parameter_slots = tuple(
                (
                    relative_name,
                    tuple(
                        f"population.{path}"
                        for path in component_parameter_aliases[relative_name]
                    ),
                )
                for relative_name in component_parameters
            )
            component_parameter_ids = {
                id(parameter) for parameter in component_parameters.values()
            }
            registered_parameter_slots = {
                name
                for name, value in _module_tensor_slots(component_preparation)
                if id(value) in component_parameter_ids
            }
            routed_parameter_slots = {
                slot for _relative_name, slots in parameter_slots for slot in slots
            }
            if routed_parameter_slots != registered_parameter_slots:
                raise FunctionalizationError(
                    f"MultiPopulation component {component_name!r} physical "
                    "parameter routing does not cover every registered adapter "
                    "parameter slot."
                )

            constant_sources = []
            component_constants = []
            for relative_name in geometry_constant_names:
                component_constants.append(
                    (
                        relative_name,
                        _geometry_constant_value(source_component, relative_name),
                    )
                )
            for entry in component_preparation.parametrization_constants:
                component_constants.append(
                    (
                        entry.key,
                        _buffer_at(
                            source_component,
                            entry.module_path,
                            entry.buffer_name,
                        ),
                    )
                )

            for relative_name, source_value in component_constants:
                if not torch.is_tensor(source_value):
                    raise FunctionalizationError(
                        f"MultiPopulation component {component_name!r} constant "
                        f"{relative_name!r} is not tensor-valued."
                    )
                proposed_name = f"populations.{component_name}.{relative_name}"
                existing_constant = constant_by_identity.get(id(source_value))
                if existing_constant is None:
                    public_name = proposed_name
                    constant_by_identity[id(source_value)] = (
                        source_value,
                        public_name,
                    )
                else:
                    existing_value, public_name = existing_constant
                    if existing_value is not source_value:  # pragma: no cover
                        raise FunctionalizationError(
                            "MultiPopulation constant identity bookkeeping failed."
                        )
                shape = _shape_tuple(source_value)
                previous_shape = public_constant_shapes.setdefault(public_name, shape)
                previous_dtype = public_constant_dtypes.setdefault(
                    public_name,
                    source_value.dtype,
                )
                previous_device = public_constant_devices.setdefault(
                    public_name,
                    source_value.device,
                )
                if (
                    previous_shape != shape
                    or previous_dtype != source_value.dtype
                    or previous_device != source_value.device
                ):
                    raise FunctionalizationError(
                        "Aliased MultiPopulation component constants must have one "
                        "shape, dtype, and device contract."
                    )
                constant_sources.append((relative_name, public_name))

            records.append(
                _MultiComponentPlan(
                    name=component_name,
                    parameter_sources=tuple(parameter_sources),
                    parameter_slots=parameter_slots,
                    constant_sources=tuple(constant_sources),
                    geometry_constant_names=geometry_constant_names,
                    parametrization_constants=(
                        component_preparation.parametrization_constants
                    ),
                    topology_kind=topology_kind,
                    core_shape=tuple(source_component.core_shape()),
                )
            )

        # Register adapters below the already-audited parent preparation. The
        # parent purity probe then deep-copies and fingerprints their complete
        # physical module trees alongside its own workspace builder.
        self._preparation.component_physical_preparations = torch.nn.ModuleList(
            physical_preparations
        )
        self._multi_component_plans = tuple(records)
        self._multi_public_constant_shapes = public_constant_shapes
        self._multi_public_constant_dtypes = public_constant_dtypes
        self._multi_public_constant_devices = public_constant_devices
        self._multi_canonical_parameter_names = tuple(
            name for name in parent_parameters if name in canonical_parameter_names
        )
        self._multi_component_physical_parameter_names = frozenset(
            public_name
            for record in records
            for relative_name, public_name in record.parameter_sources
            if relative_name != "celsius_param"
        )

    @staticmethod
    def _make_state_schema(layout):
        schema = {}
        for leaf in layout:
            _put_path(schema, leaf.public_path, leaf)
        return schema

    def _capture_structured_step_graph(self, ve_present, intra_present):
        """Capture one pure step graph for an optional-drive signature."""

        def authored_step(parameters, prepared_values, state, *drive_values):
            drive_index = 0
            ve = None
            intra = None
            if ve_present:
                ve = drive_values[drive_index]
                drive_index += 1
            if intra_present:
                intra = drive_values[drive_index]
            return self._transition_rollout_values(
                parameters,
                prepared_values,
                state,
                ve,
                intra,
                1,
            )[0]

        with torch.inference_mode(False), torch.no_grad():
            tensors = self.extract()
            prepared = self._prepare_values(tensors.parameters, tensors.constants)
            zero_drive = torch.zeros(
                (1, *self.shape),
                device=self.device,
                dtype=self.dtype,
            )
            example_drives = []
            if ve_present:
                example_drives.append(zero_drive.clone())
            if intra_present:
                example_drives.append(zero_drive.clone())
            return make_fx(authored_step)(
                tensors.parameters,
                prepared,
                tensors.state,
                *example_drives,
            )

    def _ensure_structured_step_graph(self, ve_present, intra_present):
        """Lazily capture one inference graph outside a Dynamo boundary."""
        signature = (ve_present, intra_present)
        if signature in self._structured_step_graphs:
            return True
        if signature in self._structured_capture_errors:
            return False
        if torch.compiler.is_compiling():
            # Nested make_fx capture is neither safe nor fullgraph-compatible.
            # A raw user-authored torch.compile boundary therefore retains the
            # established unrolled fallback unless it was explicitly prewarmed.
            return False
        try:
            graph = self._capture_structured_step_graph(*signature)
        except Exception as exc:
            # Structured control flow is an inference compilation optimization,
            # not part of the functional semantics. Cache failure per signature
            # so an unsupported drive form cannot disable the other variants.
            details = f"{type(exc).__name__}: {exc}"
            self._structured_capture_errors[signature] = details
            self._structured_capture_error = details
            return False
        self._structured_step_graphs[signature] = graph
        return True

    def prewarm_structured_rollout(self, *, ve: bool = False, intra: bool = False):
        """Prewarm bounded-graph inference for one optional-drive signature.

        This is only needed when wrapping :meth:`prepare_and_rollout` directly
        in a user-authored ``torch.compile`` boundary. The higher-level
        :meth:`compile_rollout_chunk` wrapper prewarms its signature lazily.
        Capture failure is non-fatal and returns ``False``; compilation then
        uses the semantically equivalent unrolled rollout.
        """
        if not isinstance(ve, bool) or not isinstance(intra, bool):
            raise TypeError("ve and intra must be bool values")
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "prewarm_structured_rollout() must be called before entering "
                "torch.compile"
            )
        self._validate_source()
        return self._ensure_structured_step_graph(
            ve or self._extra.enabled,
            intra or self._intra.enabled,
        )

    def make_extra(self, extra) -> FunctionalExtra:
        """Lower the ordinary Population ``extra=`` form without mutating it."""
        self._validate_source()
        return FunctionalExtra(self._population, extra)

    def make_callbacks(self, callbacks):
        """Bind named native functional callbacks into an immutable plan.

        Callbacks execute in functional rollouts and host runners. A bound
        compiled chunk fuses each callback update with its model transition.
        Values must implement :class:`dendra.func.FunctionalCallback`; built-ins
        such as :class:`dendra.func.Recorder` provide familiar common behavior.
        """
        from ._callbacks import FunctionalCallbacks

        self._validate_source()
        return FunctionalCallbacks(self, callbacks)

    def _validate_extra_plan(self, extra: FunctionalExtra) -> None:
        if not isinstance(extra, FunctionalExtra):
            raise TypeError("extra plan must be a FunctionalExtra")
        if (
            extra.shape != self.shape
            or extra.device != self.device
            or extra.dtype != self.dtype
        ):
            raise FunctionalizationError(
                "FunctionalExtra shape, device, or dtype does not match this "
                "FunctionalPopulation plan"
            )

    def _validate_source(self) -> None:
        if not torch.compiler.is_compiling():
            current_support_binding = _support_key_runtime_binding(self._population)
            if current_support_binding != self._source_support_key_binding:
                if (
                    torch._C._are_functorch_transforms_active()
                    or torch.autograd.forward_ad._current_level >= 0
                ):
                    raise FunctionalizationError(
                        "The source Population support-key binding changed before "
                        "an active torch.func transform; validate eagerly or lower "
                        "the Population again before applying the transform."
                    )
                # Identity/version/schema changes are unusual and therefore
                # pay the exact ordered comparison only on this slow path.
                # Same-value state loading or rebinding is accepted and makes
                # subsequent entry checks cheap again.
                _validate_support_keys(
                    self._population,
                    reference=self._transition.population,
                    context="Source Population",
                )
                self._source_support_key_binding = current_support_binding
            current_override_binding = _additional_parameter_key_binding(
                self._population
            )
            if (
                current_override_binding
                != self._source_additional_parameter_key_binding
            ):
                if (
                    torch._C._are_functorch_transforms_active()
                    or torch.autograd.forward_ad._current_level >= 0
                ):
                    raise FunctionalizationError(
                        "The source Population regional-parameter layout changed "
                        "before an active torch.func transform; validate eagerly "
                        "or lower the Population again before applying the transform."
                    )
                if (
                    _additional_parameter_layout_signature(self._population)
                    != self._additional_parameter_layout
                ):
                    raise FunctionalizationError(
                        "The source Population regional-parameter placement "
                        "changed after make_functional(); rebuild and lower it again."
                    )
                self._source_additional_parameter_key_binding = current_override_binding
        if getattr(self._population, "_flag_rebuild", False):
            raise FunctionalizationError(
                "The source Population has pending structural changes; rebuild "
                "and lower it again."
            )
        if not getattr(self._population, "is_built", False):
            raise FunctionalizationError(
                "The source Population is not built; build and lower it again."
            )
        if not torch.compiler.is_compiling():
            if (
                _parametrization_constant_layout_signature(self._population)
                != self._parametrization_constant_layout_signature
            ):
                raise FunctionalizationError(
                    "The source Population in-graph parametrization buffer layout "
                    "changed after make_functional(); lower it again."
                )
        if (
            self._topology_kind in {"multi_point", "multi_scalar"}
            and not torch.compiler.is_compiling()
        ):
            if (
                _multi_canonical_source_binding(self._population)
                != self._canonical_source_binding
            ):
                raise FunctionalizationError(
                    "A canonical MultiPopulation component geometry changed after "
                    "make_functional(); construct and lower a new packed model."
                )
        elif (
            self._topology_kind == "native_cable" and not torch.compiler.is_compiling()
        ):
            if (
                _native_cable_source_binding(self._population)
                != self._canonical_source_binding
            ):
                raise FunctionalizationError(
                    "The source native Cable canonical geometry changed after "
                    "make_functional(); construct and lower a new Cable."
                )
        structure_matches = (
            _compiled_structure_signature_matches(self._population, self._fingerprint)
            if torch.compiler.is_compiling()
            else _structure_signature(self._population) == self._fingerprint
        )
        if (
            _structure_ids(self._population) != self._source_ids
            or not structure_matches
        ):
            raise FunctionalizationError(
                "The source Population structure changed after make_functional(); "
                "lower it again after insert(), batch(), rebuild(), .to(), or dtype changes."
            )
        # The immutable plan and explicit tensor leaves are the compiled
        # execution contract. Inspecting arbitrary authored Waveform Python
        # objects inside Dynamo is neither tensor work nor fullgraph-safe; eager
        # entry points (including prewarm and compiled-wrapper construction)
        # perform this source-topology check before tracing begins.
        if not torch.compiler.is_compiling():
            self._intra._validate_source()

    def _validate_compatible_population(self, population) -> None:
        if not isinstance(population, Population):
            raise TypeError("population must be a dendra Population")
        if not getattr(population, "is_built", False) or getattr(
            population, "_flag_rebuild", False
        ):
            raise FunctionalizationError(
                "Population has pending structural changes; rebuild it before "
                "using this functional execution plan."
            )
        _validate_support_keys(
            population,
            reference=self._transition.population,
            context="Compatible Population",
        )
        if (
            _additional_parameter_layout_signature(population)
            != self._additional_parameter_layout
        ):
            raise FunctionalizationError(
                "Population regional-parameter placement does not match this "
                "functional execution plan."
            )
        if (
            _parametrization_constant_layout_signature(population)
            != self._parametrization_constant_layout_signature
        ):
            raise FunctionalizationError(
                "Population in-graph parametrization buffer layout does not match "
                "this functional execution plan."
            )
        if self._topology_kind in {"multi_point", "multi_scalar"}:
            if not _is_scalar_multi(population):
                raise FunctionalizationError(
                    "Population is not an exact scalar MultiPopulation."
                )
            if _multi_component_topology_signature(
                population
            ) != _multi_component_topology_signature(self._population):
                raise FunctionalizationError(
                    "MultiPopulation ordered component structure does not match "
                    "this functional execution plan."
                )
            if (
                _multi_canonical_value_signature(population)
                != self._multi_canonical_values
            ):
                raise FunctionalizationError(
                    "MultiPopulation canonical component geometry values do not "
                    "match this functional execution plan."
                )
        elif self._topology_kind == "native_cable":
            if not _is_native_cable(population):
                raise FunctionalizationError(
                    "Population is not an exact canonical native Cable."
                )
            failure = _validate_native_cable_source(population)
            if failure is not None:
                raise FunctionalizationError(failure)
            failure = _effective_population_parameter_failure(population)
            if failure is not None:
                raise FunctionalizationError(failure)
        elif self._topology_kind == "scalar_tree":
            if not _is_scalar_tree(population):
                raise FunctionalizationError(
                    "Population is not a canonical scalar Tree."
                )
            failure = _validate_scalar_tree_source(population)
            if failure is not None:
                raise FunctionalizationError(failure)
            failure = _effective_population_parameter_failure(population)
            if failure is not None:
                raise FunctionalizationError(failure)
        elif self._topology_kind == "extcell_tree":
            if not _is_extcell_tree(population):
                raise FunctionalizationError(
                    "Population is not a canonical ExtCellTree."
                )
            failure = _validate_extcell_tree_source(population)
            if failure is not None:
                raise FunctionalizationError(failure)
            failure = _effective_population_parameter_failure(population)
            if failure is not None:
                raise FunctionalizationError(failure)
        if _structure_signature(population) != self._fingerprint:
            raise FunctionalizationError(
                "Population structure does not match this functional execution plan."
            )
        if (
            _state_layout_signature(_state_layout(population))
            != self._state_layout_signature
        ):
            raise FunctionalizationError(
                "Population runtime-state layout does not match this functional "
                "execution plan. Rebuild and lower the model again after changing "
                "integrator state buffers or their shape, dtype, or device."
            )

    def _validate_parameters(self, parameters) -> None:
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping of named parameter tensors")
        if set(parameters) != set(self._parameter_names):
            missing = sorted(set(self._parameter_names) - set(parameters))
            unexpected = sorted(set(parameters) - set(self._parameter_names))
            raise KeyError(
                f"parameter tree mismatch; missing={missing}, unexpected={unexpected}"
            )
        for name in self._parameter_names:
            value = parameters[name]
            if not torch.is_tensor(value):
                raise TypeError(f"parameter {name!r} must be a Tensor")
            if _shape_tuple(value) != self._parameter_shapes[name]:
                raise ValueError(
                    f"parameter {name!r} has shape {_shape_tuple(value)}; "
                    f"expected {self._parameter_shapes[name]}"
                )
            expected_device = self._parameter_devices[name]
            expected_dtype = self._parameter_dtypes[name]
            if value.device != expected_device or value.dtype != expected_dtype:
                raise ValueError(
                    f"parameter {name!r} must use {expected_device}/{expected_dtype}"
                )

    def _validate_constants(self, constants) -> None:
        if not isinstance(constants, Mapping):
            raise TypeError("constants must be a mapping of named tensors")
        expected_names = (
            {
                *self._multi_public_constant_shapes,
                *(entry.key for entry in self._parametrization_constants),
                "dt",
            }
            if self._topology_kind in {"multi_point", "multi_scalar"}
            else {
                *self._geometry_constant_names,
                *(entry.key for entry in self._parametrization_constants),
                "dt",
            }
        )
        if set(constants) != expected_names:
            missing = sorted(expected_names - set(constants))
            unexpected = sorted(set(constants) - expected_names)
            raise KeyError(
                f"constant tree mismatch; missing={missing}, unexpected={unexpected}"
            )
        geometry_contract = (
            (
                (
                    name,
                    self._multi_public_constant_shapes[name],
                    self._multi_public_constant_devices[name],
                    self._multi_public_constant_dtypes[name],
                )
                for name in self._multi_public_constant_shapes
            )
            if self._topology_kind in {"multi_point", "multi_scalar"}
            else (
                (
                    name,
                    self._geometry_constant_shapes[name],
                    self.device,
                    self.dtype,
                )
                for name in self._geometry_constant_names
            )
        )
        for name, expected_shape, expected_device, expected_dtype in geometry_contract:
            value = constants[name]
            if not torch.is_tensor(value) or _shape_tuple(value) != expected_shape:
                raise ValueError(
                    f"constant {name!r} must have shared geometry shape "
                    f"{expected_shape}"
                )
            if value.device != expected_device or value.dtype != expected_dtype:
                raise ValueError(
                    f"constant {name!r} must use {expected_device}/{expected_dtype}"
                )
        for entry in self._parametrization_constants:
            value = constants[entry.key]
            if not torch.is_tensor(value) or _shape_tuple(value) != entry.shape:
                raise ValueError(
                    f"constant {entry.key!r} must have parametrization shape "
                    f"{entry.shape}"
                )
            if value.device != entry.device or value.dtype != entry.dtype:
                raise ValueError(
                    f"constant {entry.key!r} must use {entry.device}/{entry.dtype}"
                )
        dt = constants["dt"]
        if not torch.is_tensor(dt) or _shape_tuple(dt) != ():
            raise ValueError("constant 'dt' must be a scalar Tensor")
        if dt.device != self.device or dt.dtype != self.dtype:
            raise ValueError(f"constant 'dt' must use {self.device}/{self.dtype}")
        if (
            not torch.compiler.is_compiling()
            and not torch._C._are_functorch_transforms_active()
            and torch.autograd.forward_ad._current_level < 0
        ):
            dt_value = float(dt.detach())
            if not math.isfinite(dt_value) or dt_value <= 0.0:
                raise ValueError("constant 'dt' must be finite and positive")

    def _validate_initialization(self, inputs) -> None:
        if not isinstance(inputs, InitializationInput):
            raise TypeError("inputs must be an InitializationInput")
        value = inputs.v_init
        if not torch.is_tensor(value):
            raise TypeError("initialization v_init must be a Tensor")
        if _shape_tuple(value) != self.shape:
            raise ValueError(
                "initialization v_init has shape "
                f"{_shape_tuple(value)}; expected {self.shape}"
            )
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError(
                f"initialization v_init must use {self.device}/{self.dtype}"
            )

        states = {} if inputs.states is None else inputs.states
        if not isinstance(states, Mapping):
            raise TypeError(
                "initialization states must be a mapping of canonical state names"
            )
        expected = {entry.name for entry in self._initialization_state_layout}
        if set(states) != expected:
            missing = sorted(expected - set(states))
            unexpected = sorted(set(states) - expected)
            raise KeyError(
                "initial state input mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for entry in self._initialization_state_layout:
            value = states[entry.name]
            if not torch.is_tensor(value):
                raise TypeError(f"initial state {entry.name!r} must be a Tensor")
            if _shape_tuple(value) != entry.shape:
                raise ValueError(
                    f"initial state {entry.name!r} has shape {_shape_tuple(value)}; "
                    f"expected {entry.shape}"
                )
            if value.device != entry.device or value.dtype != entry.dtype:
                raise ValueError(
                    f"initial state {entry.name!r} must use "
                    f"{entry.device}/{entry.dtype}"
                )

        transforms = {} if inputs.transforms is None else inputs.transforms
        if not isinstance(transforms, Mapping):
            raise TypeError(
                "initialization transforms must be a mapping of canonical input names"
            )
        expected_transforms = {
            entry.key for entry in self._initialization_transform_inputs
        }
        if set(transforms) != expected_transforms:
            missing = sorted(expected_transforms - set(transforms))
            unexpected = sorted(set(transforms) - expected_transforms)
            raise KeyError(
                "initialization transform input mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for entry in self._initialization_transform_inputs:
            value = transforms[entry.key]
            if not torch.is_tensor(value):
                raise TypeError(
                    f"initialization transform input {entry.key!r} must be a Tensor"
                )
            if _shape_tuple(value) != entry.shape:
                raise ValueError(
                    f"initialization transform input {entry.key!r} has shape "
                    f"{_shape_tuple(value)}; expected {entry.shape}"
                )
            if value.device != entry.device or value.dtype != entry.dtype:
                raise ValueError(
                    f"initialization transform input {entry.key!r} must use "
                    f"{entry.device}/{entry.dtype}"
                )

    def _validate_state_node(self, value, schema, path=()) -> None:
        if isinstance(schema, _StateLeaf):
            label = ".".join(path)
            if not torch.is_tensor(value):
                raise TypeError(f"state leaf {label!r} must be a Tensor")
            if _shape_tuple(value) != schema.shape:
                raise ValueError(
                    f"state leaf {label!r} has shape {_shape_tuple(value)}; "
                    f"expected {schema.shape}"
                )
            if value.device != schema.device or value.dtype != schema.dtype:
                raise ValueError(
                    f"state leaf {label!r} must use {schema.device}/{schema.dtype}"
                )
            return
        if not isinstance(value, Mapping):
            label = "state" + "".join(f"[{part!r}]" for part in path)
            raise TypeError(f"{label} must be a mapping")
        if set(value) != set(schema):
            label = "state" + "".join(f"[{part!r}]" for part in path)
            raise KeyError(f"{label} must contain exactly {sorted(schema)}")
        for name, child_schema in schema.items():
            self._validate_state_node(value[name], child_schema, (*path, name))

    def _validate_state(self, state) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        self._validate_state_node(state, self._state_schema)

    def extract(self, population: Population | None = None) -> PopulationTensors:
        """Extract explicit initialization, constants/state, and raw parameters."""
        population = self._population if population is None else population
        self._validate_compatible_population(population)
        if (
            _initialization_structure_signature(population)
            != self._initialization_fingerprint
        ):
            raise FunctionalizationError(
                "Population initialization structure does not match this "
                "functional execution plan."
            )
        state = {}
        for leaf in _state_layout(population):
            value = _buffer_at(population, leaf.module_path, leaf.buffer_name)
            _put_path(state, leaf.public_path, _clone_tensor(value))
        self._validate_state(state)
        all_parameters = dict(population.named_parameters())
        parameters = (
            {
                name: all_parameters[name]
                for name in self._multi_canonical_parameter_names
            }
            if self._multi_canonical_parameter_names is not None
            else all_parameters
        )
        if population is self._population:
            intra = self._intra
        else:
            intra = FunctionalIntra(population)
            if intra.layout_signature != self._intra.layout_signature:
                raise FunctionalizationError(
                    "Population intracellular stimulation topology does not "
                    "match this functional execution plan."
                )
        for plan in (intra, self._extra):
            stimulation = plan.extract()
            parameters.update(stimulation.parameters)
            # Waveform buffers, fields, and precomputed temporal values are
            # execution-time tensor inputs. Keep them beside raw parameters so
            # changing stimulation never invalidates the prepared dynamics.
            parameters.update(stimulation.constants)
        if self._topology_kind in {"multi_point", "multi_scalar"}:
            constants = {}
            for component_record in self._multi_component_plans:
                component = population.populations[component_record.name]
                parametrization_entries = {
                    entry.key: entry
                    for entry in component_record.parametrization_constants
                }
                for relative_name, public_name in component_record.constant_sources:
                    if relative_name in component_record.geometry_constant_names:
                        value = _geometry_constant_value(component, relative_name)
                    else:
                        entry = parametrization_entries[relative_name]
                        value = _buffer_at(
                            component,
                            entry.module_path,
                            entry.buffer_name,
                        )
                    if public_name in constants:
                        selected = constants[public_name]
                        if (
                            _shape_tuple(value) != _shape_tuple(selected)
                            or value.dtype != selected.dtype
                            or value.device != selected.device
                            or not torch.equal(value, selected)
                        ):
                            raise FunctionalizationError(
                                "Compatible MultiPopulation components represented "
                                f"by aliased constant {public_name!r} must have "
                                "exactly equal values."
                            )
                        continue
                    constants[public_name] = _clone_tensor(value)
            constants.update(
                {
                    entry.key: _clone_tensor(
                        _buffer_at(population, entry.module_path, entry.buffer_name)
                    )
                    for entry in self._parametrization_constants
                }
            )
            constants["dt"] = torch.as_tensor(
                self.dt,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            constants = {
                **{
                    name: _clone_tensor(_geometry_constant_value(population, name))
                    for name in self._geometry_constant_names
                    if name
                    not in {"canonical_area_cm2", "canonical_edge_resistance_ohm"}
                },
                **(
                    {
                        "canonical_area_cm2": _clone_tensor(
                            _geometry_constant_value(
                                population,
                                "canonical_area_cm2",
                            )
                        ),
                        "canonical_edge_resistance_ohm": _clone_tensor(
                            _geometry_constant_value(
                                population,
                                "canonical_edge_resistance_ohm",
                            )
                        ),
                    }
                    if self._topology_kind
                    in {"native_cable", "scalar_tree", "extcell_tree"}
                    else {}
                ),
                **{
                    entry.key: _clone_tensor(
                        _buffer_at(population, entry.module_path, entry.buffer_name)
                    )
                    for entry in self._parametrization_constants
                },
                "dt": torch.as_tensor(
                    self.dt,
                    device=self.device,
                    dtype=self.dtype,
                ),
            }
        self._validate_parameters(parameters)
        self._validate_constants(constants)
        initialization_states = {}
        for entry in self._initialization_state_layout:
            mechanism = population.mech.mechanisms[entry.mechanism_name]
            raw_value = mechanism._init_params[entry.state_name]
            if torch.is_tensor(raw_value):
                value = raw_value.to(device=entry.device, dtype=entry.dtype)
            else:
                value = torch.as_tensor(
                    raw_value,
                    device=entry.device,
                    dtype=entry.dtype,
                )
            try:
                value = value.expand(entry.shape)
            except RuntimeError as exc:
                raise FunctionalizationError(
                    f"explicit initial state {entry.name!r} with shape "
                    f"{_shape_tuple(value)} is not broadcastable to {entry.shape}"
                ) from exc
            initialization_states[entry.name] = _clone_tensor(value)
        initialization_transform_values = {}
        for entry in self._initialization_transform_inputs:
            action = population._initialization_transforms[entry.action_name]
            input_index = action.input_names.index(entry.input_name)
            initialization_transform_values[entry.key] = _clone_tensor(
                action.input_values()[input_index]
            )
        initialization = InitializationInput(
            v_init=_clone_tensor(population.expanded_v_init(self.shape)),
            states=initialization_states,
            transforms=initialization_transform_values,
        )
        return PopulationTensors(
            parameters=parameters,
            constants=constants,
            state=state,
            initialization=initialization,
        )

    def _population_parameters(self, parameters):
        return {name: parameters[name] for name in self._population_parameter_names}

    def _prepare_parameter_buffers(
        self,
        parameters,
        diam,
        dx,
        diameters,
        diam_original,
        dt,
        parametrization_constants,
        *,
        preparation=None,
        base_mapping=None,
    ):
        preparation = self._preparation if preparation is None else preparation
        base_mapping = (
            self._preparation_base_mapping if base_mapping is None else base_mapping
        )
        # Authored workspace builders execute against differentiable private
        # copies. Invalid writes cannot escape into caller tensors or poison
        # the retained preparation skeleton.
        mapping = {name: value.clone() for name, value in base_mapping.items()}
        for name in self._population_parameter_names:
            value = parameters[name].clone()
            for slot in self._preparation_parameter_slots[name]:
                mapping[slot] = value
        for entry in self._parametrization_constants:
            value = parametrization_constants[entry.key].clone()
            for alias in entry.aliases:
                mapping[f"population.{alias}"] = value
        outputs = functional_call(
            preparation,
            mapping,
            (
                None if diam is None else diam.clone(),
                dx.clone(),
                None if diameters is None else diameters.clone(),
                None if diam_original is None else diam_original.clone(),
                dt.clone(),
            ),
            tie_weights=False,
        )
        if preparation.local_geometry_bindings:
            population_outputs, mechanism_outputs, local_geometry_outputs = outputs
        else:
            population_outputs, mechanism_outputs = outputs
            local_geometry_outputs = ()
        population_values = dict(
            zip(
                preparation.population_parameter_names,
                population_outputs,
                strict=True,
            )
        )
        mechanism_values = {
            entry.key: value
            for entry, value in zip(
                self._prepared_buffers,
                mechanism_outputs,
                strict=True,
            )
        }
        return population_values, mechanism_values, tuple(local_geometry_outputs)

    def prepare(self, parameters, constants):
        """Materialize an eagerly reusable, freshness-checked tensor plan."""
        self._validate_source()
        self._validate_parameters(parameters)
        self._validate_constants(constants)
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "Opaque prepared plans cannot cross torch.compile graph "
                "boundaries; compile prepare_and_step() or "
                "prepare_and_rollout() instead"
            )

        population_parameters = self._population_parameters(parameters)
        sources = (*population_parameters.values(), *constants.values())
        if any(_has_inference_tensor_component(value) for value in sources):
            raise FunctionalizationError(
                "prepare() cannot freshness-check inference-tensor parameters or "
                "constants, including forward-AD tangents; clone them outside "
                "inference mode first"
            )

        inference_mode_enabled = torch.is_inference_mode_enabled()
        grad_enabled_at_prepare = torch.is_grad_enabled() and not inference_mode_enabled
        if inference_mode_enabled:
            # Opaque prepared plans rely on Tensor version counters for their
            # freshness contract. Temporarily disable inference mode so derived
            # workspaces remain ordinary versioned tensors; no gradient graph
            # is recorded because the caller is still preparing for inference.
            with torch.inference_mode(False), torch.no_grad():
                values = self._prepare_values(parameters, constants)
        else:
            values = self._prepare_values(parameters, constants)
        return _PreparedPopulation(
            self._prepared_token,
            _tensor_binding(population_parameters),
            _tensor_binding(constants),
            constants,
            grad_enabled_at_prepare,
            values,
        )

    def initialize(self, parameters, constants, inputs: InitializationInput):
        """Purely construct fresh state and initialization-derived parameters.

        Registered pure pre/post transforms consume their explicit
        ``inputs.transforms`` leaves in imperative registration order. Their
        raw-parameter and state replacements are returned in the complete
        bundle; constants pass through unchanged.
        """

        self._validate_source()
        self._validate_parameters(parameters)
        self._validate_constants(constants)
        self._validate_initialization(inputs)
        current_failures = (
            _compiled_initialization_failures(self._population)
            if torch.compiler.is_compiling()
            else tuple(_functional_initialization_failures(self._population))
        )
        initialization_structure_matches = (
            _compiled_initialization_structure_matches(
                self._population,
                self._initialization_fingerprint,
            )
            if torch.compiler.is_compiling()
            else _initialization_structure_signature(self._population)
            == self._initialization_fingerprint
        )
        initialization_failures = (
            *self._initialization_failures,
            *current_failures,
            *(
                ()
                if initialization_structure_matches
                else (
                    "the source Population initialization structure changed after "
                    "make_functional(); lower it again after changing State "
                    "initialization hooks or insertion-time ic keys",
                )
            ),
        )
        if initialization_failures:
            details = "\n".join(
                f"- {failure}" for failure in dict.fromkeys(initialization_failures)
            )
            raise FunctionalizationError(
                "Population transition is functional, but fresh initialization "
                "cannot yet be represented by the current pure initializer "
                f"slice:\n{details}"
            )
        if self._initialization is None or self._initialization_state_template is None:
            raise FunctionalizationError(
                "The functional initialization plan is unavailable."
            )

        initialization_states = {} if inputs.states is None else inputs.states
        initialization_transform_values = (
            {} if inputs.transforms is None else inputs.transforms
        )
        pre_state = torch.utils._pytree.tree_map(
            lambda value: value,
            self._initialization_state_template,
        )
        _put_path(pre_state, ("integrator", "v"), inputs.v_init)
        for entry in self._initialization_state_layout:
            _put_path(
                pre_state,
                entry.public_path,
                initialization_states[entry.name],
            )
        initialized_parameters, pre_state = _apply_initialization_transforms(
            self._initialization,
            "pre",
            parameters,
            pre_state,
            initialization_transform_values,
        )
        prepared = self._prepare_values(initialized_parameters, constants)
        # Imperative Mechanism/State initialization and its final current frame
        # run before the integrator configures a timestep. Preserve that exact
        # zero-dt view even though transition preparation already owns runtime dt.
        initialization_dt = torch.zeros_like(prepared["integrator"]["dt"])
        mapping, _dt, _celsius, _v, _t, _duration_remainder = self._mapping(
            initialized_parameters,
            prepared,
            self._initialization_state_template,
            isolate_read_only=True,
            mechanism_dt=initialization_dt,
        )
        outputs = functional_call(
            self._initialization,
            mapping,
            (
                _get_path(pre_state, ("integrator", "v")),
                *(
                    _get_path(pre_state, entry.public_path)
                    for entry in self._initialization_state_layout
                ),
            ),
            tie_weights=False,
        )
        state = {}
        for leaf, value in zip(self._state_layout, outputs, strict=True):
            if not torch.is_tensor(value):
                raise FunctionalizationError(
                    "functional initialization returned non-Tensor state leaf "
                    f"{'.'.join(leaf.public_path)!r}"
                )
            _put_path(state, leaf.public_path, value)
        initialized_parameters, state = _apply_initialization_transforms(
            self._initialization,
            "post",
            initialized_parameters,
            state,
            initialization_transform_values,
        )
        state = _reset_functional_initialization_clock(state)
        self._validate_parameters(initialized_parameters)
        self._validate_state(state)
        return PopulationTensors(
            parameters={
                name: initialized_parameters[name] for name in self._parameter_names
            },
            constants={name: constants[name] for name in constants},
            state=state,
            initialization=InitializationInput(
                v_init=inputs.v_init,
                states=initialization_states,
                transforms=initialization_transform_values,
            ),
        )

    def _prepare_values(
        self,
        parameters,
        constants,
        *,
        preparation=None,
        preparation_base_mapping=None,
    ):
        if self._topology_kind in {"multi_point", "multi_scalar"}:
            return self._prepare_multi_values(
                parameters,
                constants,
                preparation=preparation,
                preparation_base_mapping=preparation_base_mapping,
            )

        shape = self.shape
        geometry_shape = self._geometry_shape
        solve_shape = self._solve_shape
        batch_rank = len(shape) - len(geometry_shape)
        diam = constants.get("diam")
        diameters = constants.get("diameters")
        diam_original = constants.get("diam_original")
        xraxial = constants.get("xraxial")
        xc = constants.get("xc")
        xg = constants.get("xg")
        dx = constants["dx"]
        dt = constants["dt"]

        population_values, mechanism_values, local_geometry_values = (
            self._prepare_parameter_buffers(
                parameters,
                diam,
                dx,
                diameters,
                diam_original,
                dt,
                {
                    entry.key: constants[entry.key]
                    for entry in self._parametrization_constants
                },
                preparation=preparation,
                base_mapping=preparation_base_mapping,
            )
        )
        diam = population_values.get("diam", diam)
        cm = population_values["cm"]
        rhoa = population_values["rhoa"]
        rhoa_scale = population_values["rhoa_scale"]
        cm_scale = population_values["cm_scale"]
        area_scale = population_values["area_scale"]

        integrator = self._transition.population.integrator
        if self._topology_kind == "single_compartment":
            # The scalar integrator deliberately retains two different
            # broadcastable layouts: capacitance follows RANGE parameter shape
            # (including singleton explicit-batch prefixes), while membrane
            # area follows the shared core geometry. Do not expand either to
            # public state shape merely to make their layouts identical.
            area = _cylindrical_membrane_area(diam, dx)
            point_area = area
            integrator_workspace = integrator._derive_prepared_workspace(
                dt,
                cm=cm * cm_scale,
                area=area * area_scale,
            )
        elif self._topology_kind == "extcell_axon":
            point_area = _cylindrical_membrane_area(diam, dx)
            block_size = self._transition.population.integrator.M
            shell_size = block_size - 1
            if batch_rank:
                diam_solve = torch.broadcast_to(diam, shape).reshape(solve_shape)
                dx_solve = torch.broadcast_to(dx, shape).reshape(solve_shape)
                cm_solve = cm.expand(shape).reshape(solve_shape)
                rhoa_solve = rhoa.expand(shape).reshape(solve_shape)
                shell_shape = shape + (shell_size,)
                xraxial_solve = torch.broadcast_to(xraxial, shell_shape).reshape(
                    solve_shape + (shell_size,)
                )
                xc_solve = torch.broadcast_to(xc, shell_shape).reshape(
                    solve_shape + (shell_size,)
                )
                xg_solve = torch.broadcast_to(xg, shell_shape).reshape(
                    solve_shape + (shell_size,)
                )
            else:
                diam_solve = diam
                dx_solve = dx
                cm_solve = cm
                rhoa_solve = rhoa
                xraxial_solve = xraxial
                xc_solve = xc
                xg_solve = xg

            area = _cylindrical_membrane_area(diam_solve, dx_solve)
            integrator_workspace = integrator._derive_prepared_workspace(
                dt,
                cm=cm_solve * cm_scale,
                area=area * area_scale,
                intracellular_edge_conductance=_cylindrical_edge_conductance(
                    diam_solve,
                    dx_solve,
                    rhoa_solve * rhoa_scale,
                ),
                extracellular_edge_conductance=_layered_edge_conductance(
                    xraxial_solve,
                    dx_solve,
                ),
                xc=xc_solve,
                xg=xg_solve,
            )
        elif self._topology_kind == "extcell_tree":
            # ExtCellTree combines the exact compiled intracellular resistor
            # tree with two independently authored extracellular axial layers.
            # Topology/order is immutable plan structure; all numerical planes
            # remain explicit differentiable preparation inputs.
            canonical_area = constants["canonical_area_cm2"]
            canonical_resistance = constants["canonical_edge_resistance_ohm"]
            point_area = canonical_area
            block_size = integrator.M
            shell_size = block_size - 1
            if batch_rank:
                area_solve = torch.broadcast_to(canonical_area, shape).reshape(
                    solve_shape
                )
                resistance_solve = torch.broadcast_to(
                    canonical_resistance,
                    shape,
                ).reshape(solve_shape)
                dx_solve = torch.broadcast_to(dx, shape).reshape(solve_shape)
                cm_solve = cm.expand(shape).reshape(solve_shape)
                rhoa_scale_solve = rhoa_scale.expand(shape).reshape(solve_shape)
                shell_shape = shape + (shell_size,)
                xraxial_solve = torch.broadcast_to(xraxial, shell_shape).reshape(
                    solve_shape + (shell_size,)
                )
                xc_solve = torch.broadcast_to(xc, shell_shape).reshape(
                    solve_shape + (shell_size,)
                )
                xg_solve = torch.broadcast_to(xg, shell_shape).reshape(
                    solve_shape + (shell_size,)
                )
            else:
                area_solve = canonical_area
                resistance_solve = canonical_resistance
                dx_solve = dx
                cm_solve = cm
                rhoa_scale_solve = rhoa_scale.expand_as(resistance_solve)
                xraxial_solve = xraxial
                xc_solve = xc
                xg_solve = xg

            edge_child = integrator.edge_child_orig
            edge_parent = integrator.edge_parent_orig
            intracellular_edge_conductance = (
                resistance_solve.index_select(-1, edge_child)
                * rhoa_scale_solve.index_select(-1, edge_child)
            ).reciprocal()
            extracellular_edge_conductance = _tree_layered_edge_conductance(
                xraxial_solve,
                dx_solve,
                edge_child,
                edge_parent,
            )
            integrator_workspace = integrator._derive_prepared_workspace(
                dt,
                cm=cm_solve * cm_scale,
                area=area_solve * area_scale,
                intracellular_edge_conductance=intracellular_edge_conductance,
                extracellular_edge_conductance=extracellular_edge_conductance,
                xc=xc_solve,
                xg=xg_solve,
                edge_child_orig=edge_child,
                solver_order=integrator.solver_order,
            )
        elif self._topology_kind == "scalar_tree":
            # Scalar Tree geometry is an exact compiled resistor graph.  The
            # child-indexed resistance already contains the authored axial
            # resistivity, so only the explicit runtime ``rhoa_scale`` is
            # applied here.  Solver ordering is structural metadata owned by
            # the lowered integrator; all differentiable values remain
            # explicit preparation inputs.
            canonical_area = constants["canonical_area_cm2"]
            canonical_resistance = constants["canonical_edge_resistance_ohm"]
            point_area = canonical_area
            if batch_rank:
                area_solve = torch.broadcast_to(canonical_area, shape).reshape(
                    solve_shape
                )
                resistance_solve = torch.broadcast_to(
                    canonical_resistance,
                    shape,
                ).reshape(solve_shape)
                cm_solve = cm.expand(shape).reshape(solve_shape)
                rhoa_scale_solve = rhoa_scale.expand(shape).reshape(solve_shape)
            else:
                area_solve = canonical_area
                resistance_solve = canonical_resistance
                cm_solve = cm
                rhoa_scale_solve = rhoa_scale.expand_as(resistance_solve)

            edge_child = integrator.edge_child_orig
            edge_resistance = resistance_solve.index_select(-1, edge_child)
            edge_rhoa_scale = rhoa_scale_solve.index_select(-1, edge_child)
            edge_conductance = (edge_resistance * edge_rhoa_scale).reciprocal()
            edge_index = edge_child.reshape(
                (1,) * (edge_conductance.ndim - 1) + (-1,)
            ).expand_as(edge_conductance)
            node_conductance = torch.zeros_like(resistance_solve).scatter(
                -1,
                edge_index,
                edge_conductance,
            )
            axial_conductance = node_conductance.index_select(
                -1,
                integrator.solver_order,
            )
            integrator_workspace = integrator._derive_prepared_workspace(
                dt,
                cm=cm_solve * cm_scale,
                area=area_solve * area_scale,
                axial_conductance=axial_conductance,
                edge_conductance=edge_conductance,
            )
        elif self._topology_kind in {"unmyelinated", "myelinated"}:
            # Geometry and RANGE values are shared across explicit Population
            # batches. Expand them only as call-local views, then flatten exactly
            # as the imperative unbranched integrator does. Preserve the original
            # unbatched expression graph: redundant reshape operations on captured
            # constants currently expose a PyTorch compile(jacrev(...)) tracing bug.
            point_area = _cylindrical_membrane_area(diam, dx)
            if batch_rank:
                diam_solve = torch.broadcast_to(diam, shape).reshape(solve_shape)
                dx_solve = torch.broadcast_to(dx, shape).reshape(solve_shape)
                cm_solve = cm.expand(shape).reshape(solve_shape)
                rhoa_solve = rhoa.expand(shape).reshape(solve_shape)
            else:
                diam_solve = diam
                dx_solve = dx
                cm_solve = cm
                rhoa_solve = rhoa

            area = (
                _cylindrical_membrane_area(diam_solve, dx_solve)
                if batch_rank
                else point_area
            )
            area_scaled = area * area_scale
            edge_conductance = _cylindrical_edge_conductance(
                diam_solve,
                dx_solve,
                rhoa_solve * rhoa_scale,
            )
            integrator_workspace = integrator._derive_prepared_workspace(
                dt,
                cm=cm_solve * cm_scale,
                area=area_scaled,
                edge_conductance=edge_conductance,
            )
        else:
            # Native Cable geometry is the compiled runtime source of truth.
            # The child-indexed resistance already includes authored Section
            # rhoa, so raw ``rhoa`` must not enter this topology adapter again.
            canonical_area = constants["canonical_area_cm2"]
            canonical_resistance = constants["canonical_edge_resistance_ohm"]
            point_area = canonical_area
            if batch_rank:
                area_solve = torch.broadcast_to(canonical_area, shape).reshape(
                    solve_shape
                )
                resistance_solve = torch.broadcast_to(
                    canonical_resistance,
                    shape,
                ).reshape(solve_shape)
                cm_solve = cm.expand(shape).reshape(solve_shape)
            else:
                area_solve = canonical_area
                resistance_solve = canonical_resistance
                cm_solve = cm
            edge_conductance = _canonical_edge_conductance(
                resistance_solve,
                rhoa_scale,
            )
            integrator_workspace = integrator._derive_prepared_workspace(
                dt,
                cm=cm_solve * cm_scale,
                area=area_solve * area_scale,
                edge_conductance=edge_conductance,
            )

        # PointProcess currents are authored in nA/µS and normalized by the
        # unscaled physical membrane area. Materialize that exact support-local
        # divisor beside other mechanism workspaces so diameter/canonical-area
        # changes remain explicit and differentiable. ``area_scale`` is an
        # integrator-only capacitance/current-area modifier and intentionally
        # does not alter the established PointProcess unit conversion.
        for entry in self._point_area_factors:
            mechanism_values[entry.key] = 1e6 * entry.support_entry.gather(point_area)

        values = {
            "geometry": {
                name: constants[name] for name in self._geometry_constant_names
            },
            "population": population_values,
            "mechanisms": mechanism_values,
            "integrator": {
                "dt": dt,
                **integrator_workspace,
            },
        }
        if self._local_geometry_bindings:
            values["local_geometry"] = local_geometry_values
        return values

    def _prepare_multi_values(
        self,
        parameters,
        constants,
        *,
        preparation=None,
        preparation_base_mapping=None,
    ):
        """Derive one differentiable packed workspace from component sources."""
        dt = constants["dt"]
        batch_shape = tuple(self.shape[:-2])
        planes = math.prod(batch_shape) or 1
        packed_integrator = self._transition.population.integrator
        physical_preparations = (
            self._preparation.component_physical_preparations
            if preparation is None
            else preparation.component_physical_preparations
        )

        packed_fields = {
            name: []
            for name in (
                "diam",
                "dx",
                "cm",
                "rhoa",
                "cm_scale",
                "rhoa_scale",
                "area_scale",
                "point_area",
                "area",
            )
        }
        axial_rows = []
        compact_edges = []

        for index, component_record in enumerate(self._multi_component_plans):
            component_preparation = physical_preparations[index]
            component_shape = (*batch_shape, *component_record.core_shape)
            component_parameters = {
                relative_name: parameters[public_name]
                for relative_name, public_name in component_record.parameter_sources
            }

            component_constants = {
                relative_name: constants[public_name]
                for relative_name, public_name in component_record.constant_sources
            }
            parameter_slots = dict(component_record.parameter_slots)
            routed_parameter_slots = {
                slot for slots in parameter_slots.values() for slot in slots
            }
            component_mapping = {
                name: value.clone()
                for name, value in _module_tensor_slots(component_preparation)
                if name not in routed_parameter_slots
            }
            for relative_name, value in component_parameters.items():
                value = value.clone()
                for slot in parameter_slots[relative_name]:
                    component_mapping[slot] = value
            for entry in component_record.parametrization_constants:
                value = component_constants[entry.key].clone()
                for alias in entry.aliases:
                    component_mapping[f"population.{alias}"] = value
            component_diam = component_constants.get("diam")
            component_diameters = component_constants.get("diameters")
            component_diam_original = component_constants.get("diam_original")
            component_outputs = functional_call(
                component_preparation,
                component_mapping,
                (
                    None if component_diam is None else component_diam.clone(),
                    component_constants["dx"].clone(),
                    (
                        None
                        if component_diameters is None
                        else component_diameters.clone()
                    ),
                    (
                        None
                        if component_diam_original is None
                        else component_diam_original.clone()
                    ),
                ),
                tie_weights=False,
            )
            population_values = dict(
                zip(
                    component_preparation.population_parameter_names,
                    component_outputs,
                    strict=True,
                )
            )
            geometry_values = component_constants

            def expanded(value):
                return torch.broadcast_to(value, component_shape)

            diam = expanded(population_values.get("diam", geometry_values.get("diam")))
            dx = expanded(geometry_values["dx"])
            cm = expanded(population_values["cm"])
            rhoa = expanded(population_values["rhoa"])
            cm_scale = expanded(population_values["cm_scale"])
            rhoa_scale = expanded(population_values["rhoa_scale"])
            area_scale = expanded(population_values["area_scale"])

            if component_record.topology_kind in {
                "native_cable",
                "scalar_tree",
            }:
                point_area = expanded(geometry_values["canonical_area_cm2"])
            else:
                point_area = _cylindrical_membrane_area(diam, dx)
            area = point_area * area_scale

            for name, value in (
                ("diam", diam),
                ("dx", dx),
                ("cm", cm),
                ("rhoa", rhoa),
                ("cm_scale", cm_scale),
                ("rhoa_scale", rhoa_scale),
                ("area_scale", area_scale),
                ("point_area", point_area),
                ("area", area),
            ):
                packed_fields[name].append(
                    value.reshape(
                        *batch_shape,
                        math.prod(component_record.core_shape),
                    )
                )

            if self._topology_kind == "multi_point":
                continue

            group_b = int(packed_integrator.group_B[index])
            group_k = int(packed_integrator.group_K[index])
            if group_b * group_k != math.prod(component_record.core_shape):
                raise FunctionalizationError(
                    f"Packed component {component_record.name!r} solver shape "
                    "does not match its structural core."
                )

            if component_record.topology_kind == "single_compartment":
                axial = torch.zeros_like(area.reshape(planes, group_b, group_k))
                edge = area.reshape(planes, group_b * group_k)[:, :0]
            elif component_record.topology_kind == "scalar_tree":
                resistance = expanded(
                    geometry_values["canonical_edge_resistance_ohm"]
                ).reshape(planes, group_b, group_k)
                rhoa_scale_solve = rhoa_scale.reshape(planes, group_b, group_k)
                edge_child = component_preparation.edge_child_orig
                edge = (
                    resistance.index_select(-1, edge_child)
                    * rhoa_scale_solve.index_select(-1, edge_child)
                ).reciprocal()
                edge_index = edge_child.reshape(
                    (1,) * (edge.ndim - 1) + (-1,)
                ).expand_as(edge)
                node_conductance = torch.zeros_like(resistance).scatter(
                    -1,
                    edge_index,
                    edge,
                )
                axial = node_conductance.index_select(
                    -1,
                    component_preparation.solver_order,
                )
                edge = edge.reshape(
                    planes,
                    group_b * max(group_k - 1, 0),
                )
            else:
                diam_solve = diam.reshape(planes, group_b, group_k)
                dx_solve = dx.reshape(planes, group_b, group_k)
                rhoa_scale_solve = rhoa_scale.reshape(planes, group_b, group_k)
                if component_record.topology_kind == "native_cable":
                    resistance = expanded(
                        geometry_values["canonical_edge_resistance_ohm"]
                    ).reshape(planes, group_b, group_k)
                    edge_by_row = _canonical_edge_conductance(
                        resistance,
                        rhoa_scale_solve[..., 1:],
                    )
                else:
                    rhoa_solve = rhoa.reshape(planes, group_b, group_k)
                    edge_by_row = _cylindrical_edge_conductance(
                        diam_solve,
                        dx_solve,
                        rhoa_solve * rhoa_scale_solve,
                    )
                axial = torch.cat(
                    (
                        torch.zeros_like(
                            area.reshape(planes, group_b, group_k)[..., :1]
                        ),
                        edge_by_row,
                    ),
                    dim=-1,
                )
                edge = edge_by_row.reshape(
                    planes,
                    group_b * max(group_k - 1, 0),
                )

            padding = int(packed_integrator.K_stride) - group_k
            axial_rows.append(torch.nn.functional.pad(axial, (0, padding)))
            compact_edges.append(edge)

        full_fields = {
            name: torch.cat(parts, dim=-1).unsqueeze(-2)
            for name, parts in packed_fields.items()
        }
        packed_diam = full_fields["diam"]
        packed_dx = full_fields["dx"]

        population_values, mechanism_values, local_geometry_values = (
            self._prepare_parameter_buffers(
                parameters,
                packed_diam,
                packed_dx,
                None,
                None,
                dt,
                {
                    entry.key: constants[entry.key]
                    for entry in self._parametrization_constants
                },
                preparation=preparation,
                base_mapping=preparation_base_mapping,
            )
        )
        for name in (
            "cm",
            "rhoa",
            "cm_scale",
            "rhoa_scale",
            "area_scale",
        ):
            population_values[name] = full_fields[name]

        if self._topology_kind == "multi_point":
            integrator_workspace = packed_integrator._derive_prepared_workspace(
                dt,
                cm=full_fields["cm"] * full_fields["cm_scale"],
                area=full_fields["area"],
            )
        else:
            a_geom_flat = torch.cat(axial_rows, dim=-2)
            edge_conductance = torch.cat(compact_edges, dim=-1)
            integrator_workspace = packed_integrator._derive_prepared_workspace(
                dt,
                a_geom_flat=a_geom_flat,
                cm=full_fields["cm"] * full_fields["cm_scale"],
                area=full_fields["area"],
                edge_conductance=edge_conductance,
            )

        for entry in self._point_area_factors:
            mechanism_values[entry.key] = 1e6 * entry.support_entry.gather(
                full_fields["point_area"]
            )

        values = {
            "geometry": {"diam": packed_diam, "dx": packed_dx},
            "population": population_values,
            "mechanisms": mechanism_values,
            "integrator": {"dt": dt, **integrator_workspace},
        }
        if self._local_geometry_bindings:
            values["local_geometry"] = local_geometry_values
        return values

    def _validate_prepared_freshness(self, prepared, parameters) -> Mapping:
        """Perform cheap identity/version checks on an opaque prepared plan."""
        if not isinstance(prepared, _PreparedPopulation):
            raise TypeError("prepared must be the opaque value returned by prepare()")
        if prepared.token is not self._prepared_token:
            raise FunctionalizationError(
                "prepared belongs to a different functional Population plan"
            )
        if not isinstance(parameters, Mapping) or any(
            not torch.is_tensor(value) for value in parameters.values()
        ):
            raise TypeError("parameters must be a mapping of named parameter tensors")
        population_parameters = self._population_parameters(parameters)
        if prepared.parameter_binding != _tensor_binding(population_parameters):
            raise FunctionalizationError(
                "parameters changed after prepare(); call prepare() again"
            )
        if not isinstance(prepared.constant_sources, Mapping) or any(
            not torch.is_tensor(value) for value in prepared.constant_sources.values()
        ):
            raise FunctionalizationError("prepared constant binding is invalid")
        if prepared.constant_binding != _tensor_binding(prepared.constant_sources):
            raise FunctionalizationError(
                "constants changed after prepare(); call prepare() again"
            )
        try:
            current_value_binding = _tensor_tree_binding(prepared.values)
        except TypeError as exc:
            raise FunctionalizationError("prepared tensor schema is invalid") from exc
        if prepared.value_binding != current_value_binding:
            raise FunctionalizationError(
                "prepared workspaces changed after prepare(); call prepare() again"
            )
        if (
            torch.is_grad_enabled()
            and not prepared.grad_enabled_at_prepare
            and any(
                value.requires_grad
                for value in (
                    *population_parameters.values(),
                    *prepared.constant_sources.values(),
                )
            )
        ):
            raise FunctionalizationError(
                "prepared was created with gradient recording disabled but its "
                "parameters or constants require gradients; call prepare() again "
                "with gradient recording enabled"
            )
        return prepared.values

    def _validate_prepared(self, prepared, parameters) -> Mapping:
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "Opaque prepared plans cannot cross torch.compile graph "
                "boundaries; compile prepare_and_step(), "
                "prepare_and_rollout(), or use compile_rollout_chunk() instead"
            )

        values = self._validate_prepared_freshness(prepared, parameters)
        expected_sections = {
            "geometry",
            "population",
            "mechanisms",
            "integrator",
        }
        if self._local_geometry_bindings:
            expected_sections.add("local_geometry")
        if not isinstance(values, Mapping) or set(values) != expected_sections:
            raise FunctionalizationError("prepared tensor schema is invalid")
        if not isinstance(values["geometry"], Mapping) or set(
            values["geometry"]
        ) != set(self._geometry_constant_names):
            raise FunctionalizationError("prepared geometry tensor schema is invalid")
        if self._local_geometry_bindings:
            if not isinstance(values["local_geometry"], tuple) or len(
                values["local_geometry"]
            ) != len(self._local_geometry_shapes):
                raise FunctionalizationError(
                    "prepared local-geometry tensor schema is invalid"
                )
        expected_population = set(self._preparation.population_parameter_names)
        if (
            not isinstance(values["population"], Mapping)
            or set(values["population"]) != expected_population
        ):
            raise FunctionalizationError("prepared population tensor schema is invalid")
        expected_mechanisms = {entry.key for entry in self._prepared_buffers}
        expected_mechanisms.update(entry.key for entry in self._point_area_factors)
        if (
            not isinstance(values["mechanisms"], Mapping)
            or set(values["mechanisms"]) != expected_mechanisms
        ):
            raise FunctionalizationError("prepared mechanism tensor schema is invalid")
        expected_integrator = {"dt", *self._integrator_workspace_names}
        if (
            not isinstance(values["integrator"], Mapping)
            or set(values["integrator"]) != expected_integrator
        ):
            raise FunctionalizationError("prepared integrator tensor schema is invalid")

        tensor_shapes = {
            **{
                ("geometry", name): self._geometry_constant_shapes[name]
                for name in self._geometry_constant_names
            },
            **{
                ("population", name): shape
                for name, shape in self._preparation.population_parameter_shapes.items()
            },
            ("integrator", "dt"): (),
        }
        tensor_shapes.update(
            {
                ("integrator", name): shape
                for name, shape in self._integrator_workspace_shapes.items()
            }
        )
        for (section, name), expected_shape in tensor_shapes.items():
            value = values[section][name]
            if not torch.is_tensor(value) or _shape_tuple(value) != expected_shape:
                raise FunctionalizationError(
                    f"prepared {section}.{name} must have shape {expected_shape}"
                )
            if value.device != self.device or value.dtype != self.dtype:
                raise FunctionalizationError(
                    f"prepared {section}.{name} must use {self.device}/{self.dtype}"
                )
        for index, (value, expected_shape) in enumerate(
            zip(
                values.get("local_geometry", ()),
                self._local_geometry_shapes,
                strict=True,
            )
        ):
            if not torch.is_tensor(value) or _shape_tuple(value) != expected_shape:
                raise FunctionalizationError(
                    f"prepared local_geometry[{index}] must have shape {expected_shape}"
                )
            if value.device != self.device or value.dtype != self.dtype:
                raise FunctionalizationError(
                    f"prepared local_geometry[{index}] must use "
                    f"{self.device}/{self.dtype}"
                )
        for entry in self._prepared_buffers:
            value = values["mechanisms"][entry.key]
            if not torch.is_tensor(value) or _shape_tuple(value) != entry.shape:
                raise FunctionalizationError(
                    f"prepared mechanism {entry.key} must have shape {entry.shape}"
                )
            if value.device != entry.device or value.dtype != entry.dtype:
                raise FunctionalizationError(
                    f"prepared mechanism {entry.key} must use "
                    f"{entry.device}/{entry.dtype}"
                )
        for entry in self._point_area_factors:
            value = values["mechanisms"][entry.key]
            if not torch.is_tensor(value) or _shape_tuple(value) != entry.shape:
                raise FunctionalizationError(
                    f"prepared PointProcess area factor {entry.key} must have "
                    f"shape {entry.shape}"
                )
            if value.device != self.device or value.dtype != self.dtype:
                raise FunctionalizationError(
                    f"prepared PointProcess area factor {entry.key} must use "
                    f"{self.device}/{self.dtype}"
                )
        return values

    def _mapping(
        self,
        parameters,
        prepared,
        state,
        *,
        isolate_read_only: bool | None = None,
        mechanism_dt=None,
    ):
        if isolate_read_only is None:
            isolate_read_only = not self._lowering_audit_complete

        def read_only(value):
            return value.clone() if isolate_read_only else value

        mapping = {name: read_only(value) for name, value in self._base_mapping.items()}
        prefix = "population."
        for name in self._population_parameter_names:
            value = read_only(parameters[name])
            for slot in self._transition_parameter_slots[name]:
                mapping[slot] = value

        for name, value in prepared["population"].items():
            slot = f"{prefix}{name}"
            if slot in self._base_mapping:
                mapping[slot] = read_only(value)
        geometry = {
            name: read_only(value) for name, value in prepared["geometry"].items()
        }
        if "diam" in prepared["population"]:
            geometry["diam"] = read_only(prepared["population"]["diam"])
        local_geometry = tuple(
            read_only(value) for value in prepared.get("local_geometry", ())
        )
        local_index = 0
        for binding in self._geometry_bindings:
            if not _cache_local_geometry(binding):
                value = geometry[binding.constant_name]
            else:
                value = local_geometry[local_index]
                local_index += 1
            for module_path in binding.module_paths:
                slot = (
                    f"{prefix}{module_path + '.' if module_path else ''}"
                    f"{binding.buffer_name}"
                )
                mapping[slot] = value

        # The handler intentionally synchronizes every local temperature view
        # at transition entry. Give each registered alias the same explicit
        # object up front so that synchronization itself is identity-stable;
        # authored writes then remain visible to the read-only audit.
        celsius = mapping[f"{prefix}celsius"]
        for slot in self._mechanism_celsius_slots:
            mapping[slot] = celsius

        dt = read_only(prepared["integrator"]["dt"])
        mechanism_dt = dt if mechanism_dt is None else read_only(mechanism_dt)
        for slot in self._mechanism_dt_slots:
            mapping[slot] = mechanism_dt
        for entry in self._prepared_buffers:
            value = read_only(prepared["mechanisms"][entry.key])
            visible_rank = self._prepared_buffer_visible_ranks[entry.key]
            if value.ndim < visible_rank:
                # A hidden zero-sized vmap lane on a logical scalar effective
                # parameter is selected at lane zero when authored code first
                # combines it with an unbatched spatial field. Preserve the
                # public scalar/prepared schema, but expose singleton local
                # axes in the execution binding so ordinary broadcasting
                # carries that hidden lane. Derived/timestep workspaces retain
                # their declared ranks: authored code may intentionally reduce
                # and combine them as logical scalars before spatial expansion.
                value = value.reshape(
                    (1,) * (visible_rank - value.ndim) + tuple(value.shape)
                )
            mapping[f"{prefix}{entry.module_path}.{entry.buffer_name}"] = value

        for entry in self._point_area_factors:
            mapping[f"{prefix}{entry.module_path}.{entry.buffer_name}"] = read_only(
                prepared["mechanisms"][entry.key]
            )

        for name in self._integrator_workspace_names:
            mapping[f"{prefix}integrator.{name}"] = read_only(
                prepared["integrator"][name]
            )

        state_positionals = {}
        for leaf in self._state_layout:
            # Authored accepted-step code may legally update declared carry in
            # place. Give the private execution skeleton one
            # differentiable working copy, shared by every registered alias, so
            # those updates become output carry without modifying (or
            # leaf-in-place invalidating) the caller's explicit state tensor.
            value = _get_path(state, leaf.public_path).clone()
            state_positionals[leaf.public_path] = value
            for module_path, buffer_name in leaf.mapping_slots:
                path = f"{module_path}." if module_path else ""
                key = f"{prefix}{path}{buffer_name}"
                # The duration carry deliberately is a plain host-side
                # attribute, not a registered module buffer. It is already an
                # explicit transition input.
                if key in self._base_mapping:
                    mapping[key] = value
        return (
            mapping,
            dt,
            celsius,
            state_positionals[("integrator", "v")],
            state_positionals[("clock", "t")],
            state_positionals[("control", "duration_remainder")],
        )

    def _audit_read_only_workspaces(self, tensors: PopulationTensors) -> None:
        """Prove preparation/transition purity before exposing a hot-path plan."""
        # A failed re-audit leaves even directly constructed/internal plans on
        # the defensive mapping path rather than trusting an earlier result.
        self._lowering_audit_complete = False
        mutated_read_only = set()
        mutated_registered = set()
        mutated_python_state = set()
        mutated_class_state = set()
        consumed_rng = set()
        repeated_outputs = []
        rng_state = _global_rng_snapshot(self.device)
        class_state = _class_state_snapshot(
            _authored_workspace_classes(self._preparation)
        )

        def record_rng_consumption():
            consumed_rng.update(_global_rng_changes(rng_state, self.device))

        try:
            with torch.inference_mode(False), torch.no_grad():
                # Run preparation and transition probes on second private
                # clones. Rejected authored Python mutation can therefore
                # neither contaminate the accepted execution plan nor escape
                # to the source Population.
                audit_preparation = _deepcopy_execution_module(self._preparation)
                _rebind_population_time_references(audit_preparation.population)
                audit_preparation.audit_builder_inputs = True
                for component_preparation in getattr(
                    audit_preparation,
                    "component_physical_preparations",
                    (),
                ):
                    component_preparation.audit_builder_inputs = True
                audit_transition = copy.deepcopy(self._transition)
                _rebind_population_time_references(audit_transition.population)
                preparation_python_state = _transition_python_state_snapshot(
                    audit_preparation
                )
                transition_python_state = _transition_python_state_snapshot(
                    audit_transition
                )
                preparation_registered = _registered_tensor_bindings(audit_preparation)
                transition_registered = _registered_tensor_bindings(audit_transition)

                # The lowering audit must be unable to mutate the explicit
                # tensors returned to the caller, even if authored code
                # illegally writes another input while we diagnose a workspace
                # write. Disabling inference mode also gives every probe tensor
                # an ordinary version counter.
                parameters = {
                    name: value.detach().clone()
                    for name, value in tensors.parameters.items()
                }
                constants = {
                    name: value.detach().clone()
                    for name, value in tensors.constants.items()
                }
                state = torch.utils._pytree.tree_map(
                    lambda value: value.detach().clone(),
                    tensors.state,
                )
                audit_preparation_base_mapping = {
                    name: _clone_tensor(value)
                    for name, value in _module_tensor_slots(audit_preparation)
                    if not name.startswith("component_physical_preparations.")
                }
                prepared = self._prepare_values(
                    parameters,
                    constants,
                    preparation=audit_preparation,
                    preparation_base_mapping=audit_preparation_base_mapping,
                )
                current_preparation_registered = _registered_tensor_bindings(
                    audit_preparation
                )
                mutated_registered.update(
                    f"preparation:{key}"
                    for key in set(preparation_registered)
                    | set(current_preparation_registered)
                    if preparation_registered.get(key)
                    != current_preparation_registered.get(key)
                )
                current_preparation_state = _transition_python_state_snapshot(
                    audit_preparation
                )
                mutated_python_state.update(
                    f"preparation:{name}"
                    for name in set(preparation_python_state)
                    | set(current_preparation_state)
                    if preparation_python_state.get(name)
                    != current_preparation_state.get(name)
                )
                record_rng_consumption()
                # Repeating a one-step call from identical explicit carry
                # catches hidden registered state even when its mutation is
                # obscured by functional_call's normal slot restoration. The
                # two-step probe additionally exercises authored loop bodies.
                for steps in (1, 1, 2):
                    mapping, dt, celsius, v, t, duration_remainder = self._mapping(
                        parameters,
                        prepared,
                        state,
                        isolate_read_only=True,
                    )
                    read_only_bindings = {
                        slot: _audit_tensor_binding(value)
                        for slot, value in mapping.items()
                        if slot not in self._mutable_mapping_slots
                    }
                    read_only_positional = {
                        "dt": _audit_tensor_binding(dt),
                        "celsius": _audit_tensor_binding(celsius),
                        "v": _audit_tensor_binding(v),
                        "t": _audit_tensor_binding(t),
                        "duration_remainder": _audit_tensor_binding(duration_remainder),
                    }
                    ve, intra = self._assemble_bound_inputs(
                        parameters,
                        state,
                        None,
                        None,
                        steps,
                        dt,
                    )
                    outputs = functional_call(
                        audit_transition,
                        mapping,
                        (
                            v,
                            t,
                            duration_remainder,
                            dt,
                            celsius,
                            ve,
                            intra,
                            steps,
                        ),
                        tie_weights=False,
                    )
                    mutated_read_only.update(
                        slot
                        for slot, binding in read_only_bindings.items()
                        if slot not in mapping
                        or _audit_tensor_binding(mapping[slot]) != binding
                    )
                    mutated_read_only.update(
                        (set(mapping) - set(self._mutable_mapping_slots))
                        ^ set(read_only_bindings)
                    )
                    mutated_read_only.update(
                        f"positional:{name}"
                        for name, value in {
                            "dt": dt,
                            "celsius": celsius,
                            "v": v,
                            "t": t,
                            "duration_remainder": duration_remainder,
                        }.items()
                        if _audit_tensor_binding(value) != read_only_positional[name]
                    )

                    current_transition_registered = _registered_tensor_bindings(
                        audit_transition
                    )
                    mutated_registered.update(
                        f"transition:{key}"
                        for key in set(transition_registered)
                        | set(current_transition_registered)
                        if transition_registered.get(key)
                        != current_transition_registered.get(key)
                    )

                    if steps == 1:
                        repeated_outputs.append(
                            tuple(_audit_tensor_value(value) for value in outputs)
                        )

                    current_python_state = _transition_python_state_snapshot(
                        audit_transition
                    )
                    mutated_python_state.update(
                        name
                        for name in set(transition_python_state)
                        | set(current_python_state)
                        if transition_python_state.get(name)
                        != current_python_state.get(name)
                    )
                    record_rng_consumption()
        finally:
            # A failed audit is observational only: process-global random state
            # and shared authored class state are restored even when authored
            # code raises partway through.
            try:
                mutated_class_state.update(_restore_class_state(class_state))
            finally:
                _restore_global_rng(rng_state, self.device)

        mutated_workspaces = mutated_read_only & set(self._prepared_workspace_slots)
        if mutated_workspaces:
            public_names = [
                slot.removeprefix("population.") for slot in sorted(mutated_workspaces)
            ]
            raise FunctionalizationError(
                "Population lowering observed authored mutation or rebinding "
                "of read-only DERIVED_BUFFER or TIMESTEP_BUFFER workspaces "
                f"{public_names}. Return evolving values through declared "
                "state or declared CARRY instead."
            )
        if mutated_registered:
            raise FunctionalizationError(
                "Population lowering purity audit observed authored addition, "
                "removal, mutation, or rebinding of registered parameter/buffer "
                f"state {sorted(mutated_registered)}. Register every evolving "
                "value as declared state or CARRY before lowering."
            )
        if mutated_read_only:
            public_names = [
                slot.removeprefix("population.") for slot in sorted(mutated_read_only)
            ]
            raise FunctionalizationError(
                "Population lowering observed authored mutation or rebinding "
                f"of read-only transition inputs {public_names}. Parameters, "
                "constants, and prepared values must remain immutable; return "
                "evolving values through declared state or CARRY."
            )
        if mutated_python_state:
            raise FunctionalizationError(
                "Population lowering purity audit observed authored mutation "
                "of unregistered Python instance state "
                f"{sorted(mutated_python_state)}. Functional transitions must "
                "be stateless outside declared state and CARRY."
            )
        if mutated_class_state:
            raise FunctionalizationError(
                "Population lowering purity audit observed authored mutation "
                "of shared Python class state "
                f"{sorted(mutated_class_state)}. Functional transitions and "
                "workspace builders must not mutate class attributes."
            )
        if consumed_rng:
            raise FunctionalizationError(
                "Population lowering purity audit observed authored consumption "
                f"of {sorted(consumed_rng)} RNG state. Functional transitions "
                "must be deterministic or make RNG state explicit carry."
            )
        if len(repeated_outputs) == 2 and repeated_outputs[0] != repeated_outputs[1]:
            raise FunctionalizationError(
                "Population lowering purity audit observed a non-deterministic "
                "transition: repeated one-step calls from the same explicit "
                "inputs returned different values. Make every evolving value "
                "explicit state or CARRY."
            )
        self._lowering_audit_complete = True

    def _audit_initialization(self, tensors: PopulationTensors) -> None:
        """Prove admitted initialization and final-current-frame purity."""

        if self._initialization is None:
            return

        rng_state = _global_rng_snapshot(self.device)
        class_state = _class_state_snapshot(
            _authored_workspace_classes(self._initialization)
        )
        mutated_registered = set()
        mutated_read_only = set()
        mutated_python_state = set()
        mutated_class_state = set()
        consumed_rng = set()
        repeated_outputs = []
        repeated_numerical_outputs = []
        transform_outputs = {}
        caught_error = None
        audit_initialization = None
        registered = {}
        python_state = {}
        try:
            with torch.inference_mode(False):
                audit_initialization = _deepcopy_execution_module(self._initialization)
                _rebind_population_time_references(audit_initialization.population)
                registered = _registered_tensor_bindings(audit_initialization)
                python_state = _transition_python_state_snapshot(audit_initialization)
                parameter_names = tuple(tensors.parameters)
                constant_names = tuple(tensors.constants)
                explicit_names = tuple(
                    entry.name for entry in self._initialization_state_layout
                )
                transform_input_names = tuple(
                    entry.key for entry in self._initialization_transform_inputs
                )
                voltage_state_index = next(
                    index
                    for index, leaf in enumerate(self._state_layout)
                    if leaf.public_path == ("integrator", "v")
                )
                transform_output_indices = tuple(range(len(parameter_names))) + tuple(
                    len(parameter_names) + index
                    for index, leaf in enumerate(self._state_layout)
                    if leaf.public_path
                    not in {
                        ("clock", "t"),
                        ("control", "duration_remainder"),
                    }
                )
                transform_output_labels = (
                    *(f"parameters.{name}" for name in parameter_names),
                    *(
                        f"state.{'.'.join(leaf.public_path)}"
                        for leaf in self._state_layout
                        if leaf.public_path
                        not in {
                            ("clock", "t"),
                            ("control", "duration_remainder"),
                        }
                    ),
                )

                def audit_input(value, *, requires_grad: bool):
                    cloned = value.detach().clone()
                    if requires_grad and (
                        cloned.is_floating_point() or cloned.is_complex()
                    ):
                        cloned.requires_grad_(True)
                    return cloned

                def record_module_side_effects():
                    current_registered = _registered_tensor_bindings(
                        audit_initialization
                    )
                    mutated_registered.update(
                        key
                        for key in set(registered) | set(current_registered)
                        if registered.get(key) != current_registered.get(key)
                    )
                    current_python_state = _transition_python_state_snapshot(
                        audit_initialization
                    )
                    mutated_python_state.update(
                        name
                        for name in set(python_state) | set(current_python_state)
                        if python_state.get(name) != current_python_state.get(name)
                    )
                    consumed_rng.update(_global_rng_changes(rng_state, self.device))

                def evaluate(
                    parameters,
                    constants,
                    v_init,
                    explicit_values,
                    transform_values,
                    *,
                    track_read_only: bool,
                ):
                    direct_inputs = {
                        **{
                            f"parameter:{name}": value
                            for name, value in parameters.items()
                        },
                        **{
                            f"constant:{name}": value
                            for name, value in constants.items()
                        },
                        "v_init": v_init,
                        **{
                            f"state:{name}": value
                            for name, value in zip(
                                explicit_names,
                                explicit_values,
                                strict=True,
                            )
                        },
                        **{
                            f"transform:{name}": value
                            for name, value in zip(
                                transform_input_names,
                                transform_values,
                                strict=True,
                            )
                        },
                    }
                    direct_bindings = (
                        {
                            name: _audit_tensor_binding(value)
                            for name, value in direct_inputs.items()
                        }
                        if track_read_only
                        else None
                    )
                    transform_inputs = dict(
                        zip(
                            transform_input_names,
                            transform_values,
                            strict=True,
                        )
                    )
                    pre_state = torch.utils._pytree.tree_map(
                        lambda value: value,
                        self._initialization_state_template,
                    )
                    _put_path(pre_state, ("integrator", "v"), v_init)
                    for entry, value in zip(
                        self._initialization_state_layout,
                        explicit_values,
                        strict=True,
                    ):
                        _put_path(pre_state, entry.public_path, value)
                    initialized_parameters, pre_state = (
                        _apply_initialization_transforms(
                            audit_initialization,
                            "pre",
                            parameters,
                            pre_state,
                            transform_inputs,
                        )
                    )
                    initialized_v = _get_path(pre_state, ("integrator", "v"))
                    initialized_explicit = tuple(
                        _get_path(pre_state, entry.public_path)
                        for entry in self._initialization_state_layout
                    )
                    prepared = self._prepare_values(
                        initialized_parameters,
                        constants,
                    )
                    initialization_dt = torch.zeros_like(prepared["integrator"]["dt"])
                    mapping, _dt, _celsius, _v, _t, _remainder = self._mapping(
                        initialized_parameters,
                        prepared,
                        self._initialization_state_template,
                        isolate_read_only=True,
                        mechanism_dt=initialization_dt,
                    )
                    if track_read_only:
                        read_only_bindings = {
                            slot: _audit_tensor_binding(value)
                            for slot, value in mapping.items()
                            if slot not in self._mutable_mapping_slots
                        }
                    state_outputs = functional_call(
                        audit_initialization,
                        mapping,
                        (initialized_v, *initialized_explicit),
                        tie_weights=False,
                    )
                    state = {}
                    for leaf, value in zip(
                        self._state_layout,
                        state_outputs,
                        strict=True,
                    ):
                        _put_path(state, leaf.public_path, value)
                    initialized_parameters, state = _apply_initialization_transforms(
                        audit_initialization,
                        "post",
                        initialized_parameters,
                        state,
                        transform_inputs,
                    )
                    state = _reset_functional_initialization_clock(state)
                    outputs = (
                        *(initialized_parameters[name] for name in parameter_names),
                        *(
                            _get_path(state, leaf.public_path)
                            for leaf in self._state_layout
                        ),
                    )

                    if track_read_only:
                        mutated_read_only.update(
                            slot
                            for slot, binding in read_only_bindings.items()
                            if slot not in mapping
                            or _audit_tensor_binding(mapping[slot]) != binding
                        )
                        mutated_read_only.update(
                            name
                            for name, binding in direct_bindings.items()
                            if _audit_tensor_binding(direct_inputs[name]) != binding
                        )
                        if _audit_tensor_numerical_value(
                            state_outputs[voltage_state_index].detach()
                        ) != _audit_tensor_numerical_value(initialized_v.detach()):
                            mutated_read_only.add("returned:state.integrator.v")
                    return outputs

                def make_inputs(*, requires_grad: bool):
                    parameters = {
                        name: audit_input(
                            tensors.parameters[name],
                            requires_grad=requires_grad,
                        )
                        for name in parameter_names
                    }
                    constants = {
                        name: audit_input(
                            tensors.constants[name],
                            requires_grad=requires_grad,
                        )
                        for name in constant_names
                    }
                    explicit = {
                        name: audit_input(
                            (tensors.initialization.states or {})[name],
                            requires_grad=requires_grad,
                        )
                        for name in explicit_names
                    }
                    v_init = audit_input(
                        tensors.initialization.v_init,
                        requires_grad=requires_grad,
                    )
                    transforms = {
                        name: audit_input(
                            (tensors.initialization.transforms or {})[name],
                            requires_grad=requires_grad,
                        )
                        for name in transform_input_names
                    }
                    return (
                        parameters,
                        constants,
                        v_init,
                        tuple(explicit.values()),
                        tuple(transforms.values()),
                    )

                # Two ordinary calls establish determinism. A third call gives
                # every differentiable explicit input autograd metadata and
                # proves authored initialization does not branch on execution
                # context while retaining the same numerical result.
                for grad_enabled in (False, False, True):
                    with torch.set_grad_enabled(grad_enabled):
                        (
                            parameters,
                            constants,
                            v_init,
                            explicit_values,
                            transform_values,
                        ) = make_inputs(requires_grad=grad_enabled)
                        outputs = evaluate(
                            parameters,
                            constants,
                            v_init,
                            explicit_values,
                            transform_values,
                            track_read_only=True,
                        )
                        record_module_side_effects()
                        repeated_outputs.append(
                            tuple(
                                _audit_tensor_value(value.detach()) for value in outputs
                            )
                        )
                        repeated_numerical_outputs.append(
                            tuple(value.detach().clone() for value in outputs)
                        )

                # Identical lanes must remain valid and numerically identical
                # inside vmap. This catches tensor-dependent Python control flow,
                # scalar extraction, and branches on active torch.func transforms.
                (
                    parameters,
                    constants,
                    v_init,
                    explicit_values,
                    transform_values,
                ) = make_inputs(requires_grad=False)
                flat_inputs = (
                    *(parameters[name] for name in parameter_names),
                    *(constants[name] for name in constant_names),
                    v_init,
                    *explicit_values,
                    *transform_values,
                )
                parameter_count = len(parameter_names)
                constant_count = len(constant_names)
                explicit_count = len(explicit_names)

                def evaluate_flat(*values):
                    parameter_values = values[:parameter_count]
                    constant_stop = parameter_count + constant_count
                    constant_values = values[parameter_count:constant_stop]
                    local_v_init = values[constant_stop]
                    explicit_stop = constant_stop + 1 + explicit_count
                    local_explicit = values[constant_stop + 1 : explicit_stop]
                    local_transforms = values[explicit_stop:]
                    outputs = evaluate(
                        dict(zip(parameter_names, parameter_values, strict=True)),
                        dict(zip(constant_names, constant_values, strict=True)),
                        local_v_init,
                        local_explicit,
                        local_transforms,
                        track_read_only=False,
                    )
                    # Some state leaves are legitimately independent of initial
                    # voltage (notably insertion-time ic values). A numerical-zero
                    # lane anchor lets vmap carry those outputs without changing
                    # their values or their meaningful derivatives.
                    lane_anchor = local_v_init.sum() * 0.0
                    return tuple(
                        outputs[index] + lane_anchor
                        for index in transform_output_indices
                    )

                with torch.no_grad():
                    vmapped_outputs = torch.vmap(evaluate_flat)(
                        *(torch.stack((value, value)) for value in flat_inputs)
                    )
                record_module_side_effects()
                transform_outputs["vmap"] = tuple(
                    value[0].detach().clone() for value in vmapped_outputs
                )

                # A joint forward-mode probe through every explicit tensor input
                # establishes that initialization has the same primal result
                # inside JVP. The tangent may legitimately be zero or nonzero.
                with torch.set_grad_enabled(True):
                    jvp_outputs, _tangents = torch.func.jvp(
                        evaluate_flat,
                        flat_inputs,
                        tuple(torch.ones_like(value) for value in flat_inputs),
                    )
                record_module_side_effects()
                transform_outputs["jvp"] = tuple(
                    value.detach().clone() for value in jvp_outputs
                )
        except Exception as exc:
            caught_error = exc
        finally:
            try:
                if audit_initialization is not None:
                    current_registered = _registered_tensor_bindings(
                        audit_initialization
                    )
                    mutated_registered.update(
                        key
                        for key in set(registered) | set(current_registered)
                        if registered.get(key) != current_registered.get(key)
                    )
                    current_python_state = _transition_python_state_snapshot(
                        audit_initialization
                    )
                    mutated_python_state.update(
                        name
                        for name in set(python_state) | set(current_python_state)
                        if python_state.get(name) != current_python_state.get(name)
                    )
                mutated_class_state.update(_restore_class_state(class_state))
            finally:
                consumed_rng.update(_global_rng_changes(rng_state, self.device))
                _restore_global_rng(rng_state, self.device)

        if caught_error is not None:
            raise FunctionalizationError(
                "pure declared-State initialization could not be evaluated: "
                f"{caught_error}"
            ) from caught_error
        if mutated_registered:
            raise FunctionalizationError(
                "initialization purity audit observed mutation or rebinding of "
                f"registered tensor state {sorted(mutated_registered)}"
            )
        if mutated_read_only:
            raise FunctionalizationError(
                "initialization purity audit observed mutation of read-only "
                f"inputs {sorted(mutated_read_only)}"
            )
        if mutated_python_state:
            raise FunctionalizationError(
                "initialization purity audit observed mutation of unregistered "
                f"Python instance state {sorted(mutated_python_state)}"
            )
        if mutated_class_state:
            raise FunctionalizationError(
                "initialization purity audit observed mutation of shared Python "
                f"class state {sorted(mutated_class_state)}"
            )
        if consumed_rng:
            raise FunctionalizationError(
                "initialization purity audit observed consumption of "
                f"{sorted(consumed_rng)} RNG state"
            )
        if repeated_outputs[0] != repeated_outputs[1]:
            raise FunctionalizationError(
                "initialization purity audit observed non-deterministic outputs "
                "from identical explicit inputs"
            )
        if repeated_outputs[0] != repeated_outputs[2]:
            raise FunctionalizationError(
                "initialization purity audit observed values that depend on "
                "autograd metadata or grad-enabled execution context"
            )
        baseline = tuple(
            repeated_numerical_outputs[0][index] for index in transform_output_indices
        )
        for transform_name, values in transform_outputs.items():
            mismatches = []
            for output_label, actual, expected in zip(
                transform_output_labels,
                values,
                baseline,
                strict=True,
            ):
                numerically_equal = False
                if (
                    actual.shape == expected.shape
                    and actual.dtype == expected.dtype
                    and actual.device == expected.device
                ):
                    if actual.is_floating_point() or actual.is_complex():
                        tolerance = 8.0 * torch.finfo(actual.dtype).eps
                        numerically_equal = torch.allclose(
                            actual,
                            expected,
                            rtol=tolerance,
                            atol=tolerance,
                            equal_nan=True,
                        )
                    else:
                        numerically_equal = torch.equal(actual, expected)
                if numerically_equal:
                    continue
                difference = None
                if (
                    actual.shape == expected.shape
                    and actual.dtype == expected.dtype
                    and actual.device == expected.device
                    and (actual.is_floating_point() or actual.is_complex())
                    and actual.numel()
                ):
                    difference = torch.max(torch.abs(actual - expected)).item()
                mismatches.append(
                    (
                        output_label,
                        difference,
                    )
                )
            if mismatches:
                raise FunctionalizationError(
                    "initialization purity audit observed values that depend on "
                    f"active torch.func {transform_name} execution context; "
                    f"mismatched outputs={mismatches}"
                )

    @staticmethod
    def _stimulus_values(plan, parameters):
        stimulus_parameters = {name: parameters[name] for name in plan.parameter_names}
        stimulus_constants = {name: parameters[name] for name in plan.constant_names}
        return stimulus_parameters, stimulus_constants

    def _stimulation_times(self, state, steps: int, dt=None) -> torch.Tensor:
        """Return endpoint-exclusive pre-step absolute times without host reads."""
        start = state["clock"]["t"]
        if dt is None:
            dt = start.new_tensor(self.dt)
        offsets = torch.arange(
            steps,
            device=self.device,
            dtype=torch.float64,
        )
        return (start.to(torch.float64) + offsets * dt.to(torch.float64)).to(self.dtype)

    def _assemble_bound_inputs(
        self,
        parameters,
        state,
        ve,
        intra,
        steps: int,
        dt,
    ):
        if self._intra.enabled and intra is not None:
            raise ValueError(
                "explicit intra cannot be combined with registered functional intra"
            )
        if self._extra.enabled and ve is not None:
            raise ValueError("explicit ve cannot be combined with bound extra")
        if not self._intra.enabled and not self._extra.enabled:
            return ve, intra

        times = self._stimulation_times(state, steps, dt)
        if self._intra.enabled:
            stimulus_parameters, stimulus_constants = self._stimulus_values(
                self._intra,
                parameters,
            )
            intra = self._intra.assemble_values(
                stimulus_parameters,
                stimulus_constants,
                times,
            )
        if self._extra.enabled:
            stimulus_parameters, stimulus_constants = self._stimulus_values(
                self._extra,
                parameters,
            )
            ve = self._extra.assemble_values(
                stimulus_parameters,
                stimulus_constants,
                times,
            )
        return ve, intra

    def _normalize_rollout_value(self, value, *, name: str, steps: int | None):
        if value is None:
            return None, steps
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a Tensor or None")
        if value.ndim == 0:
            raise ValueError(f"{name} needs a leading time axis; got a scalar tensor")
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError(f"{name} must use {self.device}/{self.dtype}")
        value_steps = int(value.shape[0])
        if steps is not None and value_steps != steps:
            raise ValueError(
                f"{name} has {value_steps} steps, but the rollout expects {steps}"
            )

        # Match Population's direct-drive contract. Ordinary trailing
        # broadcasting is authoritative, so a payload whose size happens to
        # equal an explicit batch axis retains its established spatial suffix
        # meaning. Only when that fails may a low-rank payload be interpreted
        # as batch-only and padded with singleton population/compartment axes.
        payload_shape = tuple(value.shape[1:])
        target_shape = (value_steps, *self.shape)
        trailing_error = None
        if len(payload_shape) <= len(self.shape):
            aligned_shape = (
                value_steps,
                *((1,) * (len(self.shape) - len(payload_shape))),
                *payload_shape,
            )
            try:
                normalized = torch.broadcast_to(
                    value.reshape(aligned_shape),
                    target_shape,
                )
                return normalized.contiguous(), value_steps
            except RuntimeError as exc:
                trailing_error = exc

        batch_shape = self.shape[:-2]
        if batch_shape and 0 < len(payload_shape) <= len(batch_shape):
            batch_aligned_shape = (
                value_steps,
                *((1,) * (len(batch_shape) - len(payload_shape))),
                *payload_shape,
            )
            try:
                batch_value = torch.broadcast_to(
                    value.reshape(batch_aligned_shape),
                    (value_steps, *batch_shape),
                )
                normalized = torch.broadcast_to(
                    batch_value.reshape(value_steps, *batch_shape, 1, 1),
                    target_shape,
                )
                return normalized.contiguous(), value_steps
            except RuntimeError:
                pass

        raise ValueError(
            f"{name} per-step shape {payload_shape} is not broadcastable to "
            f"Population shape {self.shape}. Ordinary trailing PyTorch "
            "broadcasting is tried first; a batch-only fallback accepts "
            f"payloads broadcastable to explicit batch shape {batch_shape}. "
            "Use singleton axes to disambiguate batch and spatial intent."
        ) from trailing_error

    def _transition_rollout_values(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        steps,
    ):
        """Execute the established authored transition with a Python loop."""
        mapping, dt, celsius, v, t, duration_remainder = self._mapping(
            parameters,
            prepared,
            state,
        )
        outputs = functional_call(
            self._transition,
            mapping,
            (
                v,
                t,
                duration_remainder,
                dt,
                celsius,
                ve,
                intra,
                steps,
            ),
            tie_weights=False,
        )
        next_state = {}
        for leaf, value in zip(self._state_layout, outputs, strict=True):
            if not torch.is_tensor(value):
                raise FunctionalizationError(
                    f"functional transition returned non-Tensor state leaf "
                    f"{'.'.join(leaf.public_path)!r}"
                )
            if (
                _shape_tuple(value) != leaf.shape
                or value.dtype != leaf.dtype
                or value.device != leaf.device
            ):
                raise FunctionalizationError(
                    "functional transition changed the schema of state leaf "
                    f"{'.'.join(leaf.public_path)!r}: expected "
                    f"shape/device/dtype {leaf.shape}/{leaf.device}/{leaf.dtype}, "
                    f"got {_shape_tuple(value)}/{value.device}/{value.dtype}. "
                    "Mutable STATE and CARRY assignments must preserve their "
                    "initialized tensor schema across every timestep."
                )
            _put_path(next_state, leaf.public_path, value)
        return next_state, {"v": next_state["integrator"]["v"]}

    def _clone_state_tree(self, state):
        # while_loop requires the carry's pytree order and every tensor's
        # metadata, including strides, to match the body output exactly.
        # Public state is key-addressed and may arrive with a different Mapping
        # insertion order or valid non-contiguous views. Rebuild it from the
        # immutable layout so initial and returned carry have one owned,
        # canonical representation.
        cloned = {}
        for leaf in self._state_layout:
            value = _get_path(state, leaf.public_path).clone(
                memory_format=torch.contiguous_format
            )
            _put_path(cloned, leaf.public_path, value)
        return cloned

    def _structured_rollout_values(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        steps,
    ):
        """Run compiled inference with graph size bounded by a small step block."""
        step_graph = self._structured_step_graphs[(ve is not None, intra is not None)]
        carry = self._clone_state_tree(state)
        chunk = _STRUCTURED_ROLLOUT_CHUNK
        full_chunks = steps // chunk
        full_steps = full_chunks * chunk

        ve_chunks = None
        if ve is not None and full_chunks:
            ve_chunks = ve[:full_steps].reshape(
                full_chunks,
                chunk,
                *self.shape,
            )
        intra_chunks = None
        if intra is not None and full_chunks:
            intra_chunks = intra[:full_steps].reshape(
                full_chunks,
                chunk,
                *self.shape,
            )

        def apply_step(current, ve_step=None, intra_step=None):
            graph_inputs = [parameters, prepared, current]
            if ve is not None:
                graph_inputs.append(ve_step)
            if intra is not None:
                graph_inputs.append(intra_step)
            return self._clone_state_tree(step_graph(*graph_inputs))

        if full_chunks:
            chunk_index = torch.zeros((), device=self.device, dtype=torch.int64)

            def cond(index, _current):
                return index < full_chunks

            def body(index, current):
                selector = index.reshape(1)
                ve_chunk = (
                    None
                    if ve_chunks is None
                    else torch.index_select(ve_chunks, 0, selector).squeeze(0)
                )
                intra_chunk = (
                    None
                    if intra_chunks is None
                    else torch.index_select(intra_chunks, 0, selector).squeeze(0)
                )
                for offset in range(chunk):
                    current = apply_step(
                        current,
                        None if ve_chunk is None else ve_chunk[offset : offset + 1],
                        (
                            None
                            if intra_chunk is None
                            else intra_chunk[offset : offset + 1]
                        ),
                    )
                return index + 1, current

            _chunk_index, carry = torch.while_loop(
                cond,
                body,
                (chunk_index, carry),
            )

        for index in range(full_steps, steps):
            carry = apply_step(
                carry,
                None if ve is None else ve[index : index + 1],
                None if intra is None else intra[index : index + 1],
            )
        return carry, {"v": carry["integrator"]["v"]}

    def _resolve_rollout_inputs(self, inputs, steps):
        """Validate public rollout inputs and resolve their common length."""
        if inputs is None:
            inputs = RolloutInput()
        if not isinstance(inputs, RolloutInput):
            raise TypeError("inputs must be a RolloutInput or None")
        if steps is not None:
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise ValueError("steps must be a non-negative integer")
        if self._extra.enabled and inputs.ve is not None:
            raise ValueError("explicit ve cannot be combined with bound extra")
        if self._intra.enabled and inputs.intra is not None:
            raise ValueError(
                "explicit intra cannot be combined with registered functional intra"
            )

        ve, resolved_steps = self._normalize_rollout_value(
            inputs.ve,
            name="ve",
            steps=steps,
        )
        intra, resolved_steps = self._normalize_rollout_value(
            inputs.intra,
            name="intra",
            steps=resolved_steps,
        )
        if resolved_steps is None:
            raise ValueError("steps is required when both rollout inputs are None")
        return ve, intra, resolved_steps

    def _execute_rollout_values(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        steps,
    ):
        """Execute already validated tensor inputs through the selected path."""
        ve, intra = self._assemble_bound_inputs(
            parameters,
            state,
            ve,
            intra,
            steps,
            prepared["integrator"]["dt"],
        )
        structured_signature = (ve is not None, intra is not None)
        if (
            torch.compiler.is_compiling()
            and not torch.is_grad_enabled()
            and torch.autograd.forward_ad._current_level < 0
            and not torch._C._are_functorch_transforms_active()
            and structured_signature in self._structured_step_graphs
        ):
            return self._structured_rollout_values(
                parameters,
                prepared,
                state,
                ve,
                intra,
                steps,
            )
        return self._transition_rollout_values(
            parameters,
            prepared,
            state,
            ve,
            intra,
            steps,
        )

    def _execute_rollout_values_with_callbacks(
        self,
        parameters,
        prepared,
        state,
        ve,
        intra,
        steps,
        callbacks,
        callback_state,
    ):
        """Execute a fixed-step rollout while threading pure callback carry."""

        from ._callbacks import FunctionalCallbackResults

        callbacks._validate_owner(self)
        callbacks._validate_runtime_dt_tensor(prepared["integrator"]["dt"])
        parts = []
        if callback_state is None:
            callback_state, initial_emission = callbacks._initialize(state)
            initial_part = callbacks._stack([initial_emission])
            if callbacks._has_emissions(initial_part):
                parts.append(initial_part)
        else:
            callback_state = callbacks._validate_state(callback_state)

        # Bound stimulation is evaluated once for the complete time horizon,
        # then sliced beside explicit drives. The no-callback rollout remains
        # on its established fused transition path.
        ve, intra = self._assemble_bound_inputs(
            parameters,
            state,
            ve,
            intra,
            steps,
            prepared["integrator"]["dt"],
        )
        emissions = []
        auxiliary = {"v": state["integrator"]["v"]}
        if steps == 0:
            state, auxiliary = self._transition_rollout_values(
                parameters,
                prepared,
                state,
                ve,
                intra,
                0,
            )
        else:
            for index in range(steps):
                state, auxiliary = self._transition_rollout_values(
                    parameters,
                    prepared,
                    state,
                    None if ve is None else ve[index : index + 1],
                    None if intra is None else intra[index : index + 1],
                    1,
                )
                callback_state, emitted = callbacks._update(
                    callback_state,
                    state,
                    auxiliary,
                )
                if emitted is not None:
                    emissions.append(emitted)
        emitted_part = callbacks._stack(emissions)
        if callbacks._has_emissions(emitted_part):
            parts.append(emitted_part)

        if auxiliary is None:
            result = {}
        elif isinstance(auxiliary, Mapping):
            result = dict(auxiliary)
        else:  # pragma: no cover - the owned transition returns a Mapping
            raise TypeError(
                "callback-enabled rollouts require a Mapping or None auxiliary"
            )
        if "callbacks" in result:  # pragma: no cover - reserved by this method
            raise FunctionalizationError(
                "rollout auxiliary output uses the reserved 'callbacks' key"
            )
        stacked = callbacks._concatenate(parts)
        callback_state = callbacks._capture_emission_schemas(
            callback_state,
            stacked,
        )
        outputs = callbacks._finalize(callback_state, stacked)
        callback_state = callbacks._materialize_state(callback_state)
        result["callbacks"] = FunctionalCallbackResults(
            state=callback_state,
            outputs=outputs,
        )
        return state, result

    def _step_values(self, parameters, prepared, state, inputs: StepInput):
        """Advance one already-validated host step using tensor values only.

        Host runners use this private boundary after validating the source,
        parameter tree, opaque prepared plan, initial state, and complete
        time-first drives.  Keeping the unchecked transition here avoids
        repeating those Python freshness/schema walks at every timestep while
        preserving :meth:`step` as the fully checked public entrypoint.
        """
        ve = None if inputs.ve is None else inputs.ve.unsqueeze(0)
        intra = None if inputs.intra is None else inputs.intra.unsqueeze(0)
        return self._execute_rollout_values(
            parameters,
            prepared,
            state,
            ve,
            intra,
            1,
        )

    def _rollout_values(
        self,
        parameters,
        prepared,
        state,
        inputs: RolloutInput | None = None,
        *,
        steps: int | None = None,
        extra=None,
        callbacks=None,
        callback_state=None,
    ):
        ve, intra, resolved_steps = self._resolve_rollout_inputs(inputs, steps)
        if extra is not None:
            if self._extra.enabled:
                raise ValueError(
                    "runtime extra cannot be combined with extra bound by "
                    "make_functional"
                )
            if ve is not None:
                raise ValueError("explicit ve cannot be combined with extra")
            extra_plan = (
                extra if isinstance(extra, FunctionalExtra) else self.make_extra(extra)
            )
            self._validate_extra_plan(extra_plan)
            if extra_plan.enabled:
                extra_tensors = extra_plan.extract()
                ve = extra_plan.assemble_tensors(
                    extra_tensors,
                    self._stimulation_times(
                        state,
                        resolved_steps,
                        prepared["integrator"]["dt"],
                    ),
                )
        if callbacks is None:
            if callback_state is not None:
                raise ValueError("callback_state requires callbacks")
            return self._execute_rollout_values(
                parameters,
                prepared,
                state,
                ve,
                intra,
                resolved_steps,
            )
        from ._callbacks import FunctionalCallbacks

        if not isinstance(callbacks, FunctionalCallbacks):
            raise TypeError(
                "callbacks must be a FunctionalCallbacks plan created by "
                "functional.make_callbacks(...)"
            )
        return self._execute_rollout_values_with_callbacks(
            parameters,
            prepared,
            state,
            ve,
            intra,
            resolved_steps,
            callbacks,
            callback_state,
        )

    def rollout(
        self,
        parameters,
        prepared,
        state,
        inputs: RolloutInput | None = None,
        *,
        steps: int | None = None,
        extra=None,
        callbacks=None,
        callback_state=None,
    ):
        """Run an eager/torch.func rollout from an opaque prepared plan.

        A plan from :meth:`make_callbacks` returns finalized callback values
        under ``auxiliary["callbacks"]``. Its explicit result state may be
        supplied as ``callback_state`` when continuing without reinitializing.
        """
        self._validate_source()
        self._validate_parameters(parameters)
        self._validate_state(state)
        prepared = self._validate_prepared(prepared, parameters)
        return self._rollout_values(
            parameters,
            prepared,
            state,
            inputs,
            steps=steps,
            extra=extra,
            callbacks=callbacks,
            callback_state=callback_state,
        )

    def prepare_and_rollout(
        self,
        parameters,
        constants,
        state,
        inputs: RolloutInput | None = None,
        *,
        steps: int | None = None,
        extra=None,
        callbacks=None,
        callback_state=None,
    ):
        """Derive and consume coefficients atomically, including under compile."""
        self._validate_source()
        self._validate_parameters(parameters)
        self._validate_constants(constants)
        self._validate_state(state)
        prepared = self._prepare_values(parameters, constants)
        return self._rollout_values(
            parameters,
            prepared,
            state,
            inputs,
            steps=steps,
            extra=extra,
            callbacks=callbacks,
            callback_state=callback_state,
        )

    def step(
        self,
        parameters,
        prepared,
        state,
        inputs: StepInput | None = None,
        *,
        extra=None,
    ):
        """Advance one functional timestep without mutating the Population."""
        if inputs is None:
            inputs = StepInput()
        if not isinstance(inputs, StepInput):
            raise TypeError("inputs must be a StepInput or None")
        for name, value in zip(("ve", "intra"), inputs):
            if value is not None and not torch.is_tensor(value):
                raise TypeError(f"{name} must be a Tensor or None")
        ve = None if inputs.ve is None else inputs.ve.unsqueeze(0)
        intra = None if inputs.intra is None else inputs.intra.unsqueeze(0)
        return self.rollout(
            parameters,
            prepared,
            state,
            RolloutInput(ve=ve, intra=intra),
            steps=1,
            extra=extra,
        )

    def prepare_and_step(
        self,
        parameters,
        constants,
        state,
        inputs: StepInput | None = None,
        *,
        extra=None,
    ):
        """Derive coefficients and advance one atomic, compile-safe step."""
        if inputs is None:
            inputs = StepInput()
        if not isinstance(inputs, StepInput):
            raise TypeError("inputs must be a StepInput or None")
        for name, value in zip(("ve", "intra"), inputs):
            if value is not None and not torch.is_tensor(value):
                raise TypeError(f"{name} must be a Tensor or None")
        ve = None if inputs.ve is None else inputs.ve.unsqueeze(0)
        intra = None if inputs.intra is None else inputs.intra.unsqueeze(0)
        return self.prepare_and_rollout(
            parameters,
            constants,
            state,
            RolloutInput(ve=ve, intra=intra),
            steps=1,
            extra=extra,
        )

    def bind(self, parameters, prepared):
        """Bind one eager step to explicit parameters and reusable preparation.

        The binding snapshots the parameter mapping, retains its tensor leaves,
        and carries this functional plan into the host runners. Prepare and bind
        again for each new forward/backward graph when training.
        """
        from ._binding import BoundPopulation

        return BoundPopulation(self, parameters, prepared)

    def compile_rollout_chunk(
        self, steps: int, *, execution: str = "default", **compile_options
    ):
        """Create a freshness-checked fixed-length compiled rollout.

        Preparation remains outside the compiled boundary and can therefore be
        reused across calls. Grad-enabled calls compose through ordinary
        autograd, allowing a Python loop over this callable to build exact BPTT
        without statically compiling the complete time horizon.

        ``execution="scan"`` opts into a horizon-independent loop graph on
        supported PyTorch 2.14 installations. Dendra installs a narrowly scoped
        process-local compatibility fix when this mode is selected. Ordinary
        backward is supported; ``create_graph=True`` raises explicitly. Calls
        inside ``torch.func`` or forward AD use the ordinary eager recurrence.
        The default preserves the existing compiled execution strategy.
        """
        from ._compiled import CompiledPopulationChunk

        return CompiledPopulationChunk(
            self,
            steps=steps,
            compile_options=compile_options,
            execution=execution,
        )

    def commit_state_(self, population: Population, state) -> Population:
        """Explicitly commit validated functional carry into a compatible model."""
        self._validate_compatible_population(population)
        self._validate_state(state)
        remainder = state["control"]["duration_remainder"]
        remainder_value = remainder.item()
        if not math.isfinite(remainder_value) or remainder_value < 0.0:
            raise ValueError("duration_remainder must be finite and non-negative")

        staged_population = _clone_execution_population(population)
        staged_layout = _state_layout(staged_population)
        for leaf in staged_layout:
            _set_buffer(
                staged_population,
                leaf.module_path,
                leaf.buffer_name,
                _get_path(state, leaf.public_path).clone(),
            )
        staged_population.mech.read_from_ions()
        staged_population.mech.read_from_materials()
        checkpoint = staged_population.state_dict_for_checkpoint()
        if self._topology_kind not in {"multi_point", "multi_scalar"}:
            population.restore_dict_from_checkpoint(checkpoint)
            return population

        write_back = bool(population.integrator.write_back)
        component_voltages = []
        if write_back:
            packed_voltage = _get_path(state, ("integrator", "v"))
            offset = 0
            for name, component in population.populations.items():
                width = math.prod(component.core_shape())
                value = packed_voltage[..., 0, offset : offset + width].reshape(
                    component.shape
                )
                if (
                    _shape_tuple(value) != tuple(component.shape)
                    or value.device != component.device()
                    or value.dtype != component.dtype()
                ):
                    raise FunctionalizationError(
                        f"Committed voltage for component {name!r} does not match "
                        "its shape, device, or dtype."
                    )
                component_voltages.append((component, value))
                offset += width
            if offset != self.shape[-1]:  # pragma: no cover - structure preflight
                raise FunctionalizationError(
                    "Committed packed voltage does not cover every component."
                )

        # A checkpoint restore rebinds every mutable tensor to a validated
        # clone.  If a later write-back step fails, restoring another cloned
        # checkpoint would recover values but not the caller's exact tensor
        # objects, autograd graph, or component views into the packed voltage.
        # Snapshot the admitted functional carry bindings and framework-owned
        # local synchronization mirrors by reference instead.  Rebinding those
        # exact objects is sufficient because the standard checkpoint restore
        # is itself out-of-place for tensor state.
        previous_bindings = {}

        def remember_binding(owner, name):
            key = (id(owner), name)
            if key in previous_bindings:
                return
            registered = name in owner._buffers
            value = owner._buffers[name] if registered else getattr(owner, name)
            previous_bindings[key] = (owner, name, registered, value)

        target_layout = _state_layout(population)
        for leaf in target_layout:
            for module_path, buffer_name in leaf.mapping_slots:
                remember_binding(_module_at(population, module_path), buffer_name)
        for slot in _local_synchronization_slots(population):
            relative_slot = slot.removeprefix("population.")
            module_path, _separator, buffer_name = relative_slot.rpartition(".")
            remember_binding(_module_at(population, module_path), buffer_name)
        for component in population.populations.values():
            remember_binding(component, "v")

        def restore_previous_bindings():
            for owner, name, registered, value in previous_bindings.values():
                if registered:
                    owner._buffers[name] = value
                else:
                    setattr(owner, name, value)

        try:
            population.restore_dict_from_checkpoint(checkpoint)
            if write_back:
                offset = 0
                for component, _value in component_voltages:
                    width = math.prod(component.core_shape())
                    component.v = population.v[..., 0, offset : offset + width].reshape(
                        component.shape
                    )
                    offset += width
        except Exception:
            try:
                restore_previous_bindings()
            except Exception as rollback_error:  # pragma: no cover - catastrophic
                raise RuntimeError(
                    "MultiPopulation functional commit failed and rollback could "
                    "not recover the previous packed/component state."
                ) from rollback_error
            raise
        return population


def make_functional(
    population: Population,
    *,
    dt: float,
    extra=None,
) -> tuple[FunctionalPopulation, PopulationTensors]:
    """Lower an initialized deterministic Population into a pure transition."""
    if isinstance(dt, bool):
        raise TypeError("dt must be a finite positive scalar")
    try:
        dt_value = float(dt)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt must be a finite positive scalar") from exc
    if not isinstance(population, Population):
        raise FunctionalizationError(
            "Population cannot be functionalized by the current capability slice:\n"
            "- expected an initialized dendra Population"
        )
    device = population.device()
    rng_state = _global_rng_snapshot(device)
    class_state = {}
    try:
        class_state = _class_state_snapshot(_authored_workspace_classes(population))
        multi_component_candidates = {} if _is_scalar_multi(population) else None
        failures = _functionalization_failures(
            population,
            dt_value,
            multi_component_candidates=multi_component_candidates,
        )
        if failures:
            details = "\n".join(f"- {failure}" for failure in dict.fromkeys(failures))
            raise FunctionalizationError(
                "Population cannot be functionalized by the current capability "
                f"slice:\n{details}"
            )
        # Lowering owns ordinary, versioned execution-plan tensors even when
        # the caller is already in inference mode. The extracted raw parameter
        # mapping still refers to the caller's source tensors; prepare() will
        # reject those explicitly if they themselves are inference tensors.
        with torch.inference_mode(False):
            functional = FunctionalPopulation(
                population,
                dt=dt_value,
                extra=extra,
                _multi_component_candidates=multi_component_candidates,
            )
            construction_rng = _global_rng_changes(rng_state, device)
            construction_class_state = _restore_class_state(class_state)
            if construction_class_state or construction_rng:
                details = []
                if construction_class_state:
                    details.append(
                        f"class state mutation {sorted(construction_class_state)}"
                    )
                if construction_rng:
                    details.append(f"RNG state consumption {sorted(construction_rng)}")
                raise FunctionalizationError(
                    "Population lowering observed authored "
                    + " and ".join(details)
                    + "."
                )
            tensors = functional.extract(population)
        functional._audit_read_only_workspaces(tensors)
        try:
            functional._audit_initialization(tensors)
        except FunctionalizationError as exc:
            # Initialization is an optional capability layered over the broader
            # pure transition. Preserve that separation while retaining the
            # exact audit failure for initialize().
            functional._initialization_failures = (
                *functional._initialization_failures,
                str(exc),
            )
            functional._initialization = None
            functional._initialization_state_template = None
    except Exception:
        try:
            _restore_class_state(class_state)
        finally:
            _restore_global_rng(rng_state, device)
        raise

    consumed_rng = _global_rng_changes(rng_state, device)
    try:
        mutated_class_state = _restore_class_state(class_state)
    finally:
        _restore_global_rng(rng_state, device)
    if mutated_class_state:
        raise FunctionalizationError(
            "Population lowering observed authored mutation of shared Python "
            f"class state {sorted(mutated_class_state)}. Workspace builders and "
            "functional transitions must not mutate class attributes."
        )
    if consumed_rng:
        raise FunctionalizationError(
            "Population lowering observed authored consumption of "
            f"{sorted(consumed_rng)} RNG state while constructing or auditing "
            "the functional plan. Workspace builders and transitions must be "
            "deterministic or make RNG state explicit carry."
        )
    return functional, tensors


__all__ = ["FunctionalPopulation", "make_functional"]
