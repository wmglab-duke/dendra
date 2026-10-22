"""Audit custom full-field operators without replacing their native arithmetic.

The shared ``MaterialProcess.advance_materials`` adapter owns publication.
Optional diagnostics freeze authored pure-hook dependencies and check that preparation
uses field metadata rather than the changing concentrations themselves.
"""

from __future__ import annotations

import dis
import inspect
import types

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from dendra.helpers import current_check_purity
from dendra.models.mechanisms import (
    ClampProcess,
    ClearanceProcess,
    DiffusionProcess,
    ExchangeProcess,
    MaterialProcess,
)

from ._continuous_transform import (
    _CALL_METHODS,
    _COPY_METHODS,
    _codes,
    _global_values,
    _qualify_module,
    _reject_rng_object,
    _reject_rng_reads,
    _safe_modules,
)
from ._initialization_sources import (
    _class_member_guards,
    _member_binding,
)
from ._mechanism_injections import _binding, _literal
from ._types import FunctionalizationError

_PURE_HOOKS = ("material_fields", "prepare_materials", "material_updates")
_HELPERS = {
    name: inspect.getattr_static(MaterialProcess, name)
    for name in (
        "_resolve_quantity",
        "_expand_material_operand",
        "_material_support_mask",
        "_mask_material_updates",
    )
}
_LIFECYCLE = {
    name: inspect.getattr_static(MaterialProcess, name)
    for name in (
        "advance_materials",
        "configure_process",
        "set_dt",
        "_reset_initialization_timestep",
        "_bind_materials",
        "_configure_material_support",
        "_get_material",
        "_install_monomorphic_advance",
        "populate",
        "populate_parameter_buffers",
        "init_rng",
        "_configure_timestep",
        "__setattr__",
        "__delattr__",
    )
}
_CANONICAL_FUNCTIONS = {
    function: _binding(function)
    for function in (*_HELPERS.values(), *_LIFECYCLE.values())
}
_CANONICAL_BINDINGS = tuple(
    (function.__globals__, name, function.__globals__[name])
    for function in _HELPERS.values()
    for name in function.__code__.co_names
    if name in function.__globals__
)
_METADATA_FACTORIES = frozenset(
    (
        "aten.empty_like.default",
        "aten.zeros_like.default",
        "aten.ones_like.default",
        "aten.full_like.default",
        "aten.new_empty.default",
        "aten.new_empty_strided.default",
        "aten.new_zeros.default",
        "aten.new_ones.default",
        "aten.new_full.default",
    )
)


class _RawMaterialClassMember:
    """Raw descriptor identity plus value comparison for immutable literals.

    Dynamo reconstructs tuple-valued metadata during capture. Identity of that
    temporary tuple is not a change to the authored class configuration.
    """

    def __init__(self, name, present, expected):
        self.name = name
        self.present = present
        self.expected = expected

    def __eq__(self, namespace):
        value = namespace.get(self.name)
        if (self.name in namespace) is not self.present:
            return False
        if type(self.expected) in (str, bool, int, float, complex, type(None), tuple):
            return _literal(value) == _literal(self.expected)
        return value is self.expected


def _helpers():
    from . import _population

    return _population


def custom_material_process_helper(owner, method):
    """Recognize exact shared numerical helpers during a broader hook audit."""
    function = getattr(method, "__func__", None)
    if getattr(method, "__self__", None) is not owner:
        return None
    for name, canonical in _HELPERS.items():
        if (
            function is canonical
            and _binding(canonical) == _CANONICAL_FUNCTIONS[canonical]
            and name not in vars(owner)
            and inspect.getattr_static(type(owner), name) is canonical
        ):
            return name
    return None


def _ordinary_method(owner, name):
    """Resolve an authored method only after excluding executing descriptors."""
    if name in vars(owner):
        raise FunctionalizationError(
            f"Custom MaterialProcess hook {name!r} must be defined on its class."
        )
    descriptor = inspect.getattr_static(type(owner), name, None)
    if type(descriptor) is not types.FunctionType:
        raise FunctionalizationError(
            f"Custom MaterialProcess hook {name!r} requires an ordinary pure method."
        )
    return descriptor.__get__(owner, type(owner))


def _receiver_member(owner, name):
    if name in owner._parameters:
        return "parameter", None
    if name in owner._buffers:
        return "buffer", None
    if name in owner._modules:
        return "child", owner._modules[name]
    value = inspect.getattr_static(owner, name)
    if type(value) is property:
        if value.fget is None or value.fset is not None or value.fdel is not None:
            raise FunctionalizationError(
                "Custom MaterialProcess properties must be read-only pure getters."
            )
        return "method", value.fget.__get__(owner, type(owner))
    if type(value) in (types.FunctionType, staticmethod, classmethod):
        return "method", value.__get__(owner, type(owner))
    if inspect.getattr_static(type(value), "__get__", None) is not None:
        raise FunctionalizationError(
            f"Custom MaterialProcess configuration {name!r} uses an executing descriptor."
        )
    return "config", value


def _hook_signature(method, owner, *, strict, seen=None):
    """Reuse dependency primitives, with narrow shared material-helper leaves."""
    p = _helpers()
    seen = set() if seen is None else seen
    target = getattr(method, "__func__", method)
    trusted = (
        custom_material_process_helper(owner, method) if owner is not None else None
    )
    if trusted is not None:
        return "shared-material-helper", trusted, _binding(target)
    _reject_rng_object(target)
    _reject_rng_reads(target)
    forbidden = p._forbidden_initialization_callable_name(target)
    if forbidden is not None:
        raise FunctionalizationError(
            f"Custom MaterialProcess uses effectful routine {forbidden!r}."
        )
    if id(target) in seen:
        return "recursive", id(target)
    if not inspect.isfunction(target):
        return p._initialization_dependency_value_signature(
            target, label="custom material hook", owner=owner, strict=strict, seen=set()
        )
    module_name = target.__module__ or ""
    if module_name == "torch" or module_name.startswith("torch."):
        return (
            "torch-routine",
            id(target),
            p._transform_callable_structure_signature(
                target, label="custom material hook", strict=False
            ),
        )
    seen.add(id(target))
    try:
        receiver = getattr(method, "__self__", None)
        receiver_name = (
            target.__code__.co_varnames[0]
            if receiver is owner and target.__code__.co_argcount
            else None
        )
        unsafe = tuple(
            operation
            for code in _codes(target.__code__)
            for operation in p._unsafe_initialization_operations(
                code, receiver_name if code is target.__code__ else None
            )
            if operation != "nested function or comprehension"
        )
        writes = tuple(
            instruction.argval
            for code in _codes(target.__code__)
            for instruction in dis.get_instructions(code)
            if instruction.opname in {"STORE_ATTR", "IMPORT_NAME", "IMPORT_FROM"}
        )
        if strict and (unsafe or writes or target.__dict__):
            raise FunctionalizationError(
                "Custom MaterialProcess hooks must be pure and read-only; "
                f"effectful operations {sorted(set(unsafe))}, attribute writes {writes}."
            )

        def dependency(value):
            if (
                inspect.isfunction(value)
                or inspect.ismethod(value)
                or inspect.isbuiltin(value)
            ):
                bound = getattr(value, "__self__", None)
                if (
                    bound is not None
                    and bound is not owner
                    and not isinstance(bound, types.ModuleType)
                ):
                    raise FunctionalizationError(
                        "Custom MaterialProcess captures an external bound receiver."
                    )
                return _hook_signature(value, owner, strict=strict, seen=seen)
            return p._initialization_dependency_value_signature(
                value,
                label="custom material hook",
                owner=owner,
                strict=strict,
                seen=set(),
            )

        closure = inspect.getclosurevars(target)
        dependencies = tuple(
            (scope, name, dependency(value))
            for scope, entries in (
                ("closure", closure.nonlocals),
                ("global", _global_values(target)),
                ("builtin", closure.builtins),
            )
            for name, value in sorted(entries.items())
        )
        names, escapes = p._direct_receiver_attribute_names(target, receiver_name)
        if strict and escapes:
            raise FunctionalizationError(
                f"Custom MaterialProcess receiver escapes its explicit frame: {escapes}."
            )
        members = []
        for name in sorted(names):
            kind, value = _receiver_member(owner, name)
            signature = (
                _qualify_module(value, strict=strict)
                if kind == "child"
                else dependency(value)
                if kind in ("method", "config")
                else None
            )
            members.append((name, kind, signature))
        return (
            "authored",
            _binding(target),
            p._transform_callable_structure_signature(
                target, label="custom material hook", strict=strict
            ),
            dependencies,
            tuple(members),
        )
    finally:
        seen.remove(id(target))


def _qualify(process, *, strict, diagnostic=None):
    diagnostic = current_check_purity() if diagnostic is None else diagnostic
    if not isinstance(process, MaterialProcess):
        raise FunctionalizationError(
            "Custom material operators require a MaterialProcess."
        )
    if diagnostic:
        _safe_modules(process)
    for name, canonical in {**_CALL_METHODS, **_LIFECYCLE}.items():
        if (
            name in vars(process)
            or inspect.getattr_static(type(process), name) is not canonical
        ):
            raise FunctionalizationError(
                f"Custom MaterialProcess replaces canonical lifecycle adapter {name!r}; "
                "implement material_fields, prepare_materials, and material_updates."
            )
    for name in _COPY_METHODS:
        if inspect.getattr_static(
            type(process), name, None
        ) is not inspect.getattr_static(MaterialProcess, name, None):
            raise FunctionalizationError(
                "Custom MaterialProcess has custom serialization hooks."
            )
    if _helpers()._module_autograd_hook_signature(process):
        raise FunctionalizationError(
            "Custom MaterialProcess cannot retain module hooks."
        )
    for name in (*process._parameters, *process._buffers, *process._modules):
        if inspect.getattr_static(type(process), name, None) is not None:
            raise FunctionalizationError(
                f"Custom MaterialProcess registered field {name!r} is shadowed by a descriptor."
            )
    descriptor = inspect.getattr_static(type(process), "material_fields", None)
    if type(descriptor) is tuple and "material_fields" not in vars(process):
        field_signature = _literal(descriptor)
    elif (
        "material_fields" in vars(process)
        or type(descriptor) is not property
        or descriptor.fget is None
        or descriptor.fset is not None
        or descriptor.fdel is not None
        or descriptor is MaterialProcess.material_fields
    ):
        raise FunctionalizationError(
            "Custom MaterialProcess must define a pure material_fields property."
        )
    else:
        field_signature = (
            _hook_signature(
                descriptor.fget.__get__(process, type(process)), process, strict=strict
            )
            if diagnostic
            else _binding(descriptor.fget)
        )
    signatures = [("material_fields", field_signature)]
    for name in _PURE_HOOKS[1:]:
        method = _ordinary_method(process, name)
        if method.__func__ is inspect.getattr_static(MaterialProcess, name):
            raise FunctionalizationError(
                f"Custom MaterialProcess must implement {name}."
            )
        signatures.append(
            (
                name,
                _hook_signature(method, process, strict=strict)
                if diagnostic
                else _binding(method),
            )
        )
    # Trusted helpers dispatch to each other on the receiver. Freeze or audit
    # every override, including branches not exercised by representative fields.
    for name in _HELPERS:
        method = _ordinary_method(process, name)
        signatures.append(
            (
                name,
                _hook_signature(method, process, strict=strict)
                if diagnostic
                else _binding(method),
            )
        )
    if any(
        _binding(function) != binding
        for function, binding in _CANONICAL_FUNCTIONS.items()
    ) or any(
        namespace.get(name) is not value
        for namespace, name, value in _CANONICAL_BINDINGS
    ):
        raise FunctionalizationError(
            "Shared MaterialProcess helpers changed; lower again."
        )
    return tuple(signatures)


def _field_keys(process):
    keys = process.material_fields
    if (
        type(keys) is not tuple
        or not keys
        or any(
            type(key) is not tuple
            or len(key) != 2
            or any(type(name) is not str or not name for name in key)
            for key in keys
        )
        or len(set(keys)) != len(keys)
    ):
        raise FunctionalizationError(
            "Custom material_fields must be a nonempty tuple of unique (material, field) string pairs."
        )
    return keys


def custom_material_process_failures(process, population=None):
    """Return cold qualification failures without executing unsafe descriptors."""
    try:
        _qualify(process, strict=True)
        keys = _field_keys(process)
        if population is not None:
            handler = population.mech
            registry = {**handler.ions, **handler.materials}
            for material, field in keys:
                if material not in registry:
                    raise FunctionalizationError(
                        f"Custom material_fields names unknown canonical material {material!r}."
                    )
                if field not in registry[material]._material_fields:
                    raise FunctionalizationError(
                        f"Custom material_fields names unknown field {material!r}.{field!r}."
                    )
    except (
        FunctionalizationError,
        TypeError,
        ValueError,
        RuntimeError,
        AttributeError,
    ) as exc:
        return (str(exc),)
    return ()


def custom_material_process_signature(process):
    """Cold compatible-copy recipe, independent of registered tensor values."""
    # The outer structural contract is independent of the caller's diagnostic
    # context. A Plan retains its own opt-in dependency frame separately.
    return _qualify(process, strict=False, diagnostic=False), _field_keys(process)


def custom_material_constant_slots(process):
    """Yield explicit config buffers, leaving effective/runtime/geometry owners alone."""
    if not isinstance(process, MaterialProcess) or isinstance(
        process, (ClampProcess, ClearanceProcess, DiffusionProcess, ExchangeProcess)
    ):
        return
    p = _helpers()
    excluded = {
        "celsius",
        "diam",
        "diameters",
        "diam_original",
        "dx",
        "area",
        "canonical_area_cm2",
        "component_area",
        "dt",
        *p._parameter_materialization_output_names(process),
        *getattr(process, "_functional_material_slots", ()),
    }
    for path, child in _safe_modules(process):
        for name, value in child._buffers.items():
            if value is not None and (
                path or (name not in excluded and not name.startswith("_mp_"))
            ):
                yield f"{path}.{name}" if path else name, value


class _MetadataFields(TorchDispatchMode):
    def __init__(self, fields):
        super().__init__()
        self.fields = {id(value) for value in fields.values()}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        reads_fields = any(
            id(value) in self.fields for value in pytree.tree_leaves((args, kwargs))
        )
        if reads_fields and str(func) not in _METADATA_FACTORIES:
            raise FunctionalizationError(
                "Custom prepare_materials must not read evolving field values; "
                "use field shape/dtype/device for preparation and evaluate field arithmetic in material_updates."
            )
        return func(*args, **kwargs)


def _prepared_schema(value):
    if value is None:
        raise FunctionalizationError(
            "Custom material preparation cannot return None; use an empty tuple for no workspace."
        )
    leaves, spec = pytree.tree_flatten(value)
    if not all(type(leaf) in (torch.Tensor, torch.nn.Parameter) for leaf in leaves):
        raise FunctionalizationError(
            "Custom material preparation must be a tensor-only PyTree."
        )
    return spec, tuple((tuple(leaf.shape), leaf.dtype, leaf.device) for leaf in leaves)


def _updates_schema(fields, updates):
    if type(updates) is not dict or any(key not in fields for key in updates):
        raise FunctionalizationError(
            "Custom material updates must be a dict replacing declared material_fields only."
        )
    result = []
    for key, value in updates.items():
        reference = fields[key]
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != tuple(reference.shape)
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise FunctionalizationError(
                "Custom material replacements must preserve each full field's shape, dtype, and device."
            )
        result.append(key)
    return tuple(result)


def _checked_hook(process, name, hook, *, metadata=False):
    def evaluate(*args, **kwargs):
        p = _helpers()
        registered = p._registered_tensor_bindings(process)
        python = p._transition_python_state_snapshot(process)
        arguments = (args, kwargs)
        before = p._audit_python_value_signature(arguments, set())
        try:
            if metadata:
                with _MetadataFields(args[0]):
                    return hook(*args, **kwargs)
            return hook(*args, **kwargs)
        finally:
            if (
                registered != p._registered_tensor_bindings(process)
                or python != p._transition_python_state_snapshot(process)
                or before != p._audit_python_value_signature(arguments, set())
            ):
                raise FunctionalizationError(
                    f"Custom MaterialProcess {name} mutated its module or read-only field/prepared inputs."
                )

    return evaluate


def install_custom_material_audits(population):
    """Observe both pure hooks on disposable preparation/transition audit copies."""
    if not current_check_purity():
        return
    for process in population.mech.material_processes.values():
        if isinstance(
            process, (ClampProcess, ClearanceProcess, DiffusionProcess, ExchangeProcess)
        ):
            continue
        for name in _PURE_HOOKS[1:]:
            setattr(
                process,
                name,
                _checked_hook(
                    process,
                    name,
                    getattr(process, name),
                    metadata=name == "prepare_materials",
                ),
            )


def _custom_material_updates(self, fields, prepared):
    result = self._functional_material_update_hook(fields, prepared)
    if _updates_schema(fields, result) != self._functional_material_output_keys:
        raise FunctionalizationError(
            "Custom MaterialProcess returned-output keys changed; lower again."
        )
    return result


class CustomMaterialProcessPlan:
    """Frozen pure-hook recipe and tensor-only material operator schemas."""

    def __init__(self, process, fields, dt):
        self.process = process
        self.diagnostic = current_check_purity()
        self._contract = _qualify(process, strict=True, diagnostic=self.diagnostic)
        self.field_keys = _field_keys(process)
        if tuple(fields) != self.field_keys:
            raise FunctionalizationError(
                "Custom material field mapping disagrees with material_fields."
            )
        self.field_schema = tuple(
            (key, tuple(value.shape), value.dtype, value.device)
            for key, value in fields.items()
        )
        self._functions = []
        self._dependencies = []
        self._configs = []
        self._class_members = []
        self._class_guards = []
        self._modules = []
        if self.diagnostic:
            self._collect(process)
        self._bindings = self.source_binding_signature()
        if self.diagnostic:
            self._audit(process, fields, dt)
        else:
            self._infer_schema(process, fields, dt)

    def _collect(self, process):
        seen = set()

        def visit(method, owner):
            function = getattr(method, "__func__", method)
            if not inspect.isfunction(function) or id(function) in seen:
                return
            seen.add(id(function))
            self._functions.append(function)
            # Canonical helper globals are frozen separately. Its registered
            # receiver frame is supplied by ordinary Population bindings.
            if custom_material_process_helper(owner, method) is not None:
                return
            for name, value in _global_values(function).items():
                self._dependencies.append(("dict", function.__globals__, name))
                if isinstance(value, types.ModuleType):
                    for member in function.__code__.co_names:
                        if member in vars(value):
                            self._dependencies.append(("dict", vars(value), member))
                elif inspect.isfunction(value):
                    visit(value, None)
            for cell in function.__closure__ or ():
                self._dependencies.append(("cell", cell, "cell_contents"))
                if inspect.isfunction(cell.cell_contents):
                    visit(cell.cell_contents, None)
            if getattr(method, "__self__", None) is not owner or owner is None:
                return
            names, _ = _helpers()._direct_receiver_attribute_names(
                function, function.__code__.co_varnames[0]
            )
            for name in names:
                kind, value = _receiver_member(owner, name)
                if kind == "method":
                    visit(value, owner)
                elif kind == "config":
                    self._configs.append((owner, name))

        for path, owner in _safe_modules(process):
            self._modules.append(owner)
            if path:
                visit(owner.forward, owner)
            else:
                fields = inspect.getattr_static(type(owner), "material_fields")
                if type(fields) is property:
                    visit(fields.fget.__get__(owner, type(owner)), owner)
                for name in (*_PURE_HOOKS[1:], *_HELPERS):
                    visit(getattr(owner, name), owner)
            for cls in type(owner).__mro__:
                if cls is object or cls.__module__.startswith("torch."):
                    continue
                names = tuple(
                    dict.fromkeys(
                        (
                            *(
                                name
                                for name in vars(cls)
                                if name not in _helpers()._CLASS_STATE_METADATA_NAMES
                                and name not in ("__init__", "__new__")
                            ),
                            *_CALL_METHODS,
                            *_PURE_HOOKS,
                            *_HELPERS,
                            *_LIFECYCLE,
                            *owner._parameters,
                            *owner._buffers,
                            *owner._modules,
                        )
                    )
                )
                for name in names:
                    raw = inspect.getattr_static(cls, name, None)
                    self._class_members.append(raw)
                    self._class_guards.extend(_class_member_guards(cls, name))
        self._functions = tuple(self._functions)
        self._dependencies = tuple(self._dependencies)
        self._configs = tuple(self._configs)
        self._class_members = tuple(self._class_members)
        unique = {}
        for record in self._class_guards:
            unique.setdefault((record[0], record[1]), record)
        self._class_guards = tuple(unique.values())
        self._modules = tuple(self._modules)

    def source_binding_signature(self):
        if not self.diagnostic:
            return ()
        return (
            tuple(_binding(function) for function in self._functions),
            tuple(_binding(function) for function in _CANONICAL_FUNCTIONS),
            tuple(
                _member_binding(owner[name] if kind == "dict" else getattr(owner, name))
                for kind, owner, name in self._dependencies
            ),
            tuple(_literal(getattr(owner, name)) for owner, name in self._configs),
            tuple(_member_binding(value) for value in self._class_members),
            tuple(
                cls.__dict__ == _RawMaterialClassMember(name, present, expected)
                for cls, name, present, expected in self._class_guards
            ),
            tuple(
                (
                    owner.training,
                    tuple(owner._parameters),
                    tuple(owner._buffers),
                    tuple((name, id(child)) for name, child in owner._modules.items()),
                    _helpers()._module_autograd_hook_signature(owner),
                )
                for owner in self._modules
            ),
            tuple(
                namespace.get(name) is expected
                for namespace, name, expected in _CANONICAL_BINDINGS
            ),
        )

    def validate_source(self, process=None):
        if process is not None and process is not self.process:
            raise FunctionalizationError(
                "Custom MaterialProcess source owner changed; lower again."
            )
        if self.source_binding_signature() != self._bindings:
            raise FunctionalizationError(
                "Custom MaterialProcess behavior or dependencies changed; lower again."
            )

    def validate_compatible(self, process):
        if (
            _qualify(process, strict=True, diagnostic=self.diagnostic) != self._contract
            or _field_keys(process) != self.field_keys
        ):
            raise FunctionalizationError(
                "Custom MaterialProcess does not match its lowered pure-hook recipe."
            )

    def configure_execution(self, process):
        """Validate returned replacements before the shared atomic publication."""
        process._functional_material_update_hook = process.material_updates
        process._functional_material_output_keys = self.output_keys
        process.material_updates = types.MethodType(_custom_material_updates, process)

    @staticmethod
    def _execution_process(process):
        p = _helpers()
        population = getattr(process, "_mp_population", None)
        if population is not None:
            cloned = p._clone_execution_population(population)
            name = next(
                name
                for name, candidate in population.mech.material_processes.items()
                if candidate is process
            )
            return cloned.mech.material_processes[name]
        return p._deepcopy_execution_module(process)

    def _infer_schema(self, process, fields, dt):
        execution = self._execution_process(process)
        inputs = {key: _helpers()._clone_tensor(value) for key, value in fields.items()}
        prepared = execution.prepare_materials(inputs, dt)
        self.prepared_schema = _prepared_schema(prepared)
        self.output_keys = _updates_schema(
            inputs, execution.material_updates(inputs, prepared)
        )

    def _audit(self, process, fields, dt):
        p = _helpers()
        classes = tuple(
            dict.fromkeys(
                cls
                for _, owner in _safe_modules(process)
                for cls in type(owner).__mro__
                if cls is not object and not cls.__module__.startswith("torch.")
            )
        )
        class_state = p._class_state_snapshot(classes)
        device = next(iter(fields.values())).device
        rng = p._global_rng_snapshot(device)
        changed_classes = changed_rng = ()
        try:
            execution = self._execution_process(process)
            authored_updates = execution.material_updates
            for name in _PURE_HOOKS[1:]:
                setattr(
                    execution,
                    name,
                    _checked_hook(
                        execution,
                        name,
                        getattr(execution, name),
                        metadata=name == "prepare_materials",
                    ),
                )
            expected_prepared = expected_updates = None
            with torch.inference_mode(False):
                for enabled in (False, True):
                    with torch.set_grad_enabled(enabled):
                        for fill in (0.0, -0.75, 1.25):
                            inputs = {
                                key: torch.full_like(value, fill)
                                for key, value in fields.items()
                            }
                            timestep = dt.clone() if torch.is_tensor(dt) else dt
                            prepared = execution.prepare_materials(inputs, timestep)
                            if any(
                                id(value) in {id(field) for field in inputs.values()}
                                for value in pytree.tree_leaves(prepared)
                            ):
                                raise FunctionalizationError(
                                    "Custom preparation retained an evolving field value."
                                )
                            schema = _prepared_schema(prepared)
                            repeated_prepared = execution.prepare_materials(
                                inputs, timestep
                            )
                            if _prepared_schema(repeated_prepared) != schema or any(
                                p._audit_tensor_numerical_value(value)
                                != p._audit_tensor_numerical_value(again)
                                for value, again in zip(
                                    pytree.tree_leaves(prepared),
                                    pytree.tree_leaves(repeated_prepared),
                                    strict=True,
                                )
                            ):
                                raise FunctionalizationError(
                                    "Custom material preparation is not deterministic from explicit inputs."
                                )
                            updates = execution.material_updates(inputs, prepared)
                            outputs = _updates_schema(inputs, updates)
                            repeated = execution.material_updates(inputs, prepared)
                            if any(
                                p._audit_tensor_numerical_value(updates[key])
                                != p._audit_tensor_numerical_value(repeated[key])
                                for key in updates
                            ):
                                raise FunctionalizationError(
                                    "Custom material updates are not deterministic from explicit inputs."
                                )
                            if expected_prepared is None:
                                expected_prepared, expected_updates = schema, outputs
                            elif (
                                schema != expected_prepared
                                or outputs != expected_updates
                            ):
                                raise FunctionalizationError(
                                    "Custom material prepared/output schema changes with field values or grad mode."
                                )
            self.prepared_schema = expected_prepared
            self.output_keys = expected_updates
            # Tensor-dependent Python branches and implicit random operators
            # may be silent on an individual eager sample. vmap exercises the
            # ordinary authored implementation under the transform contract.
            torch.vmap(
                lambda *values: authored_updates(
                    dict(zip(self.field_keys, values, strict=True)), prepared
                )
            )(*(torch.stack((value, -value)) for value in inputs.values()))
        except FunctionalizationError:
            raise
        except Exception as exc:
            raise FunctionalizationError(
                f"Custom MaterialProcess pure-hook audit failed: {exc}"
            ) from exc
        finally:
            changed_classes = p._restore_class_state(class_state)
            changed_rng = p._global_rng_changes(rng, device)
            p._restore_global_rng(rng, device)
        if changed_classes or changed_rng:
            raise FunctionalizationError(
                "Custom MaterialProcess consumed implicit RNG or mutated shared class state."
            )
