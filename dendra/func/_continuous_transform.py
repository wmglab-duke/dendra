"""Memoryless callables for the ordinary continuous transform API.

The native callable remains the numerical implementation. Modules execute on
an isolated copy with registered tensors rebound from an explicit packet;
optional diagnostics inspect authored dependencies. There is no model-type
dispatch or alternate implementation of their arithmetic.
"""

from __future__ import annotations

import dis
import inspect
import types
from collections.abc import Mapping

import torch
from torch.func import functional_call

from dendra.helpers import current_check_purity

from ._module_traversal import _registered_modules
from ._netcon import _snapshot
from ._types import FunctionalizationError

_CALL_METHODS = {
    name: inspect.getattr_static(torch.nn.Module, name)
    for name in ("__call__", "_call_impl", "_wrapped_call_impl", "__getattr__")
}
_CALL_METHODS["__getattribute__"] = object.__getattribute__
_COPY_METHODS = (
    "__deepcopy__",
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",
)
_METADATA_METHODS = (
    "named_modules",
    "modules",
    "named_parameters",
    "named_buffers",
    "named_children",
    "parameters",
    "buffers",
)
_INTERNAL_FIELDS = ("__dict__", *vars(torch.nn.Module()))
_RNG_READ_NAMES = frozenset(
    {
        "initial_seed",
        "get_rng_state",
        "get_rng_state_all",
        "get_state",
        "graphsafe_get_state",
        "default_generator",
        "default_generators",
        "Generator",
        "fork_rng",
    }
)


def _safe_modules(module):
    """Traverse raw registrations only after qualifying attribute dispatch."""
    modules = []
    for path, child in _registered_modules(module):
        kind = type(child)
        for cls in kind.__mro__:
            if cls in (torch.nn.Module, object):
                continue
            namespace = vars(cls)
            for name in _INTERNAL_FIELDS:
                if name in namespace:
                    value = namespace[name]
                    standard = inspect.getattr_static(torch.nn.Module, name, None)
                    if value is not standard and not (
                        name == "__dict__" and inspect.isgetsetdescriptor(value)
                    ):
                        raise FunctionalizationError(
                            "Continuous transform shadows framework storage with a descriptor or class field."
                        )
        for name, standard in _CALL_METHODS.items():
            if inspect.getattr_static(kind, name) is not standard:
                raise FunctionalizationError(
                    "Continuous transform needs canonical module call/attribute dispatch."
                )
        for name in _METADATA_METHODS:
            if name in vars(child) or inspect.getattr_static(
                kind, name
            ) is not inspect.getattr_static(torch.nn.Module, name):
                raise FunctionalizationError(
                    "Continuous transform overrides canonical module metadata traversal."
                )
        modules.append((path, child))
    return tuple(modules)


def _helpers():
    # func imports this adapter before FunctionalPopulation. Resolve its shared
    # qualification/audit helpers only when a projection is actually lowered.
    from . import _population

    return _population


def _named_tensors(module):
    # Canonical deduplicated keys preserve tied aliases through functional_call.
    result, seen = {}, set()
    for collection in ("_parameters", "_buffers"):
        for path, child in _safe_modules(module):
            for name, value in vars(child)[collection].items():
                if value is not None and id(value) not in seen:
                    if type(value) not in (torch.Tensor, torch.nn.Parameter):
                        raise FunctionalizationError(
                            "Continuous transform tensor subclasses need an explicit lowering."
                        )
                    seen.add(id(value))
                    result[f"{path}.{name}" if path else name] = value
    return result


def _tensor_layout(module):
    aliases, layout = {}, []
    for path, child in _safe_modules(module):
        for collection in ("_parameters", "_buffers"):
            for name, value in vars(child)[collection].items():
                key = f"{path}.{name}" if path else name
                schema = None
                if value is not None:
                    alias = aliases.setdefault(id(value), key)
                    schema = (tuple(value.shape), value.dtype, value.device, alias)
                layout.append((path, collection, name, schema))
    return tuple(layout)


def _routine(value):
    return getattr(value, "__func__", value)


def _callable_binding(function):
    target = _routine(function)
    return (target, getattr(target, "__code__", None))


def _static_configuration(module):
    p = _helpers()
    return tuple(
        (
            name,
            value
            if isinstance(
                value,
                (
                    type(None),
                    bool,
                    int,
                    float,
                    complex,
                    str,
                    bytes,
                    tuple,
                    frozenset,
                    torch.dtype,
                    torch.device,
                ),
            )
            else (type(value), id(value)),
        )
        for name, value in vars(module).items()
        if not p._is_internal_module_state(module, name)
        and name != "training"
        and not isinstance(value, torch.nn.Module)
    )


def _codes(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _codes(constant)


def _global_values(function):
    """Include globals used by pure local helpers/comprehensions as well."""
    names = {
        instruction.argval
        for code in _codes(function.__code__)
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
    }
    return {
        name: function.__globals__[name]
        for name in names
        if name in function.__globals__
    }


def _reject_rng_object(value):
    target = _routine(value)
    module = getattr(target, "__module__", "") or ""
    if (
        isinstance(value, torch.Generator)
        or value is torch.Generator
        or isinstance(getattr(target, "__self__", None), torch.Generator)
        or (
            (module == "torch" or module.startswith("torch."))
            and getattr(target, "__name__", None) in _RNG_READ_NAMES
        )
    ):
        raise FunctionalizationError(
            "Continuous transforms cannot read implicit RNG state; "
            "supply stochastic values as explicit inputs."
        )


def _reject_rng_reads(function):
    """Resolve static Torch namespace accesses before any probe is executed."""
    if not inspect.isfunction(function):
        return
    closure = inspect.getclosurevars(function)
    dependencies = {**closure.builtins, **_global_values(function), **closure.nonlocals}
    for value in dependencies.values():
        _reject_rng_object(value)
    for code in _codes(function.__code__):
        instructions = tuple(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"}:
                continue
            value = dependencies.get(instruction.argval)
            for following in instructions[index + 1 :]:
                if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                    break
                if not isinstance(value, types.ModuleType):
                    break
                name = following.argval
                namespace = vars(value)
                if name not in namespace:
                    break
                module_name = value.__name__
                if (
                    module_name == "torch" or module_name.startswith("torch.")
                ) and name in _RNG_READ_NAMES:
                    raise FunctionalizationError(
                        "Continuous transforms cannot read implicit RNG state; "
                        "supply stochastic values as explicit inputs."
                    )
                value = namespace[name]
                _reject_rng_object(value)


def _function_signature(function, owner, *, strict, seen=None):
    """Audit authored dependencies, treating framework numerical helpers opaquely.

    PyTorch helpers can contain inactive mutation branches (F.relu's inplace
    option). Inspecting every such branch as authored code would reject pure
    configurations. Framework calls are instead exercised by the same mutation,
    RNG, and transform probes as the surrounding authored operation.
    """
    p = _helpers()
    seen = set() if seen is None else seen
    target = _routine(function)
    _reject_rng_object(target)
    _reject_rng_reads(target)
    forbidden = p._forbidden_initialization_callable_name(target)
    if forbidden is not None:
        raise FunctionalizationError(
            f"Continuous transform uses effectful routine {forbidden!r}."
        )
    if id(target) in seen:
        return ("recursive", id(target))
    seen.add(id(target))
    try:
        module_name = getattr(target, "__module__", "") or ""
        if module_name == "torch" or module_name.startswith("torch."):
            return (
                "torch-routine",
                id(target),
                p._transform_callable_structure_signature(
                    target, label="continuous transform", strict=False
                ),
            )
        if not inspect.isfunction(target):
            return p._initialization_dependency_value_signature(
                target,
                label="continuous transform",
                owner=owner,
                strict=strict,
                seen=set(),
            )
        if strict and target.__dict__:
            raise FunctionalizationError(
                "Continuous transform functions cannot retain authored function-object state; "
                "use registered module parameters/buffers for numerical dependencies."
            )
        receiver = function.__self__ if inspect.ismethod(function) else None
        receiver_name = (
            target.__code__.co_varnames[0]
            if receiver is not None
            and receiver is owner
            and target.__code__.co_argcount
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
        stores = tuple(
            instruction.argval
            for code in _codes(target.__code__)
            for instruction in dis.get_instructions(code)
            if instruction.opname in {"STORE_ATTR", "IMPORT_NAME", "IMPORT_FROM"}
        )
        if strict and (unsafe or stores):
            raise FunctionalizationError(
                "Continuous transforms must be pure and read-only; "
                f"effectful operations {sorted(set(unsafe))}, attribute writes {stores}."
            )

        def dependency(value, label):
            if (
                inspect.isfunction(value)
                or inspect.ismethod(value)
                or inspect.isbuiltin(value)
            ):
                bound = value.__self__ if inspect.ismethod(value) else None
                if bound is not None and bound is not owner:
                    raise FunctionalizationError(
                        "Continuous transform captures an external bound receiver."
                    )
                return _function_signature(value, owner, strict=strict, seen=seen)
            return p._initialization_dependency_value_signature(
                value,
                label=label,
                owner=owner,
                strict=strict,
                seen=set(),
            )

        closure = inspect.getclosurevars(target)
        dependencies = tuple(
            (scope, name, dependency(value, f"continuous transform {scope}.{name}"))
            for scope, entries in (
                ("closure", closure.nonlocals),
                ("global", _global_values(target)),
                ("builtin", closure.builtins),
            )
            for name, value in sorted(entries.items())
        )
        attributes, escapes = p._direct_receiver_attribute_names(target, receiver_name)
        if strict and escapes:
            raise FunctionalizationError(
                f"Continuous transform receiver escapes its explicit dependency frame: {escapes}."
            )
        receivers = []
        for name in sorted(attributes):
            if name in owner._parameters:
                receivers.append((name, "parameter"))
            elif name in owner._buffers:
                receivers.append((name, "buffer"))
            elif name in owner._modules:
                receivers.append((name, "child-module"))
            else:
                descriptor = inspect.getattr_static(owner, name)
                if isinstance(descriptor, property):
                    value = descriptor.fget.__get__(owner, type(owner))
                elif isinstance(descriptor, (staticmethod, classmethod)):
                    value = descriptor.__get__(owner, type(owner))
                elif inspect.isfunction(descriptor):
                    value = descriptor.__get__(owner, type(owner))
                else:
                    value = descriptor
                receivers.append(
                    (name, dependency(value, f"continuous transform self.{name}"))
                )
        return (
            "authored",
            id(target),
            p._transform_callable_structure_signature(
                target, label="continuous transform", strict=strict
            ),
            dependencies,
            tuple(receivers),
            tuple(unsafe),
            stores,
        )
    finally:
        seen.remove(id(target))


def _qualify_module(module, *, strict, check_purity=True):
    p = _helpers()
    signatures = []
    for path, child in _safe_modules(module):
        kind = type(child)
        for name, standard in _CALL_METHODS.items():
            if (
                name in vars(child)
                or inspect.getattr_static(kind, name) is not standard
            ):
                raise FunctionalizationError(
                    "Continuous transform needs canonical module call/attribute dispatch."
                )
        if getattr(child, "_compiled_call_impl", None) is not None:
            raise FunctionalizationError(
                "Continuous transform cannot retain compiled custom call dispatch."
            )
        if "forward" in vars(child):
            raise FunctionalizationError(
                "Define continuous transform forward on its Module class."
            )
        for name in _COPY_METHODS:
            actual = inspect.getattr_static(kind, name, None)
            standard = inspect.getattr_static(torch.nn.Module, name, None)
            if actual is not standard:
                raise FunctionalizationError(
                    "Continuous transform has custom deepcopy/serialization hooks."
                )
        if any(
            (
                child._forward_hooks,
                child._forward_pre_hooks,
                child._backward_hooks,
                child._backward_pre_hooks,
            )
        ):
            raise FunctionalizationError(
                "Continuous transform hooks are not a pure forward contract."
            )
        # Registered tensors must resolve through Module's ordinary storage,
        # rather than an authored property that functional_call cannot rebind.
        for name in (*child._parameters, *child._buffers, *child._modules):
            if inspect.getattr_static(kind, name, None) is not None:
                raise FunctionalizationError(
                    f"Continuous transform registered field {name!r} is shadowed by a descriptor."
                )
        signatures.append(
            (
                path,
                bool(child.training),
                (
                    _function_signature(child.forward, child, strict=strict)
                    if check_purity
                    else (
                        _callable_binding(child.forward),
                        _static_configuration(child),
                    )
                ),
            )
        )
    if not check_purity:
        return (tuple(signatures), _tensor_layout(module))
    return (
        p._transform_structure_signature(
            module, strict=strict if check_purity else False
        ),
        tuple(signatures),
        _tensor_layout(module),
    )


@torch.compiler.assume_constant_result
def _compiled_signature_matches(plan):
    return plan._signature(plan._source, strict=False) == plan._fingerprint


class ContinuousTransformPlan:
    """A frozen callable plus an explicit registered-tensor schema."""

    def __init__(self, transform, *, size, device, dtype):
        self._source = transform
        self._check_purity = current_check_purity()
        self.size = size
        # Resolve aliases such as cpu:0 and unindexed mps to the actual tensor
        # device. Native connection metadata remains separately frozen.
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.is_module = isinstance(transform, torch.nn.Module)
        if (
            transform is not None
            and not self.is_module
            and not (inspect.isfunction(transform) or inspect.isbuiltin(transform))
        ):
            raise FunctionalizationError(
                "Continuous transforms must be a pure function or torch.nn.Module."
            )
        self._fingerprint = self._signature(transform, strict=True)
        self._schema = {
            name: (tuple(value.shape), value.dtype, value.device)
            for name, value in (
                _named_tensors(transform).items() if self.is_module else ()
            )
        }
        if self.is_module:
            self._execution = _helpers()._deepcopy_execution_module(transform)
            if self._execution is transform:
                raise FunctionalizationError(
                    "Continuous transform deepcopy returned the live source module."
                )
        else:
            self._execution = transform
        self._guard_attributes = []
        self._guard_classes = []
        self._guard_functions = []
        self._guard_function_ids = set()
        self._guard_modules = []
        self._guard_tensor_slots = []
        self._guard_aliases = {}
        self._guard_instances = []
        if self._check_purity:
            self._collect_guards(transform)
            self._probe()
        else:
            self._collect_binding_guards(transform)

    def _signature(self, transform, *, strict):
        if transform is None:
            return ("identity",)
        if isinstance(transform, torch.nn.Module):
            return _qualify_module(
                transform, strict=strict, check_purity=self._check_purity
            )
        if self._check_purity:
            return _function_signature(transform, None, strict=strict)
        return _callable_binding(transform)

    def _collect_module_binding_guards(self, child, p):
        """Capture explicit module bindings before class and function guards."""
        self._guard_instances.append((vars(child), tuple(vars(child))))
        self._guard_modules.append(
            (
                child,
                tuple(child._parameters),
                tuple(child._buffers),
                tuple(child._modules.items()),
            )
        )
        for collection in ("_parameters", "_buffers"):
            for name, value in vars(child)[collection].items():
                schema = (
                    None
                    if value is None
                    else (tuple(value.shape), value.dtype, value.device)
                )
                alias = (
                    None
                    if value is None
                    else self._guard_aliases.setdefault(
                        id(value), len(self._guard_aliases)
                    )
                )
                self._guard_tensor_slots.append(
                    (child, collection, name, schema, alias)
                )
        self._guard_attributes.append((child, "training", child.training, "config"))
        for name, value in vars(child).items():
            if not p._is_internal_module_state(child, name) and name != "training":
                self._guard_attributes.append((child, name, value, "config"))

    def _collect_binding_guards(self, transform):
        """Freeze explicit registrations and dispatch without dependency audits."""
        p = _helpers()
        if not self.is_module:
            target = _routine(transform)
            if inspect.isfunction(target):
                self._guard_functions.append(
                    (
                        target,
                        target.__code__,
                        target.__defaults__,
                        target.__kwdefaults__,
                    )
                )
            return
        for _path, child in _safe_modules(transform):
            self._collect_module_binding_guards(child, p)
            for cls in type(child).__mro__:
                if cls in (torch.nn.Module, object):
                    continue
                for name in (
                    *_CALL_METHODS,
                    *_METADATA_METHODS,
                    *_COPY_METHODS,
                    "forward",
                    *child._parameters,
                    *child._buffers,
                    *child._modules,
                    *_INTERNAL_FIELDS,
                ):
                    self._guard_classes.append(
                        (cls, name, name in vars(cls), vars(cls).get(name))
                    )
            target = _routine(child.forward)
            if inspect.isfunction(target):
                self._guard_functions.append(
                    (
                        target,
                        target.__code__,
                        target.__defaults__,
                        target.__kwdefaults__,
                    )
                )

    def _collect_function_guards(self, function, owner):
        target = _routine(function)
        if id(target) in self._guard_function_ids:
            return
        self._guard_function_ids.add(id(target))
        if not inspect.isfunction(target):
            return
        self._guard_functions.append(
            (target, target.__code__, target.__defaults__, target.__kwdefaults__)
        )
        if not self._check_purity:
            return
        globals_used = _global_values(target)
        for name, value in globals_used.items():
            self._guard_attributes.append((target.__globals__, name, value, "global"))
            if inspect.isfunction(value):
                self._collect_function_guards(value, owner)
        # Captured immutable values must remain unchanged after graph warming.
        for index, cell in enumerate(target.__closure__ or ()):
            self._guard_attributes.append(
                (cell, "cell_contents", cell.cell_contents, "cell")
            )
        # A framework namespace is mutable even though importing it is safe.
        # Guard referenced routines as well as the namespace itself.
        for namespace in globals_used.values():
            if isinstance(namespace, types.ModuleType):
                for name in target.__code__.co_names:
                    if name in vars(namespace):
                        value = vars(namespace)[name]
                        self._guard_attributes.append(
                            (namespace, name, value, "namespace")
                        )
                        if inspect.isfunction(value):
                            self._collect_function_guards(value, owner)

    def _collect_guards(self, transform):
        if self.is_module:
            p = _helpers()
            for path, child in _safe_modules(transform):
                self._collect_module_binding_guards(child, p)
                for cls in type(child).__mro__:
                    if cls in (torch.nn.Module, object):
                        continue
                    for name, descriptor in vars(cls).items():
                        if name not in p._CLASS_STATE_METADATA_NAMES:
                            self._guard_classes.append((cls, name, True, descriptor))
                    for name in (
                        *_CALL_METHODS,
                        *_METADATA_METHODS,
                        *_COPY_METHODS,
                        "forward",
                        *child._parameters,
                        *child._buffers,
                        *child._modules,
                        *_INTERNAL_FIELDS,
                    ):
                        self._guard_classes.append(
                            (cls, name, name in vars(cls), vars(cls).get(name))
                        )
                self._collect_function_guards(child.forward, child)
        elif transform is not None:
            self._collect_function_guards(transform, None)

    def validate_source(self, transform):
        if transform is not self._source:
            raise FunctionalizationError(
                "The ContinuousCon transform changed; lower again."
            )
        p = _helpers()
        compiling = torch.compiler.is_compiling()
        for cls, name, present, expected in self._guard_classes:
            matches = (
                (cls.__dict__ == p._RawClassHookIdentity(name, expected))
                if compiling
                else cls.__dict__.get(name) is expected
            )
            if (name in cls.__dict__) != present or not matches:
                raise FunctionalizationError(
                    "Continuous transform class behavior changed; lower again."
                )
        for storage, keys in self._guard_instances:
            if tuple(storage) != keys:
                raise FunctionalizationError(
                    "Continuous transform instance configuration changed; lower again."
                )
        for child, parameter_names, buffer_names, modules in self._guard_modules:
            if (
                tuple(child._parameters) != parameter_names
                or tuple(child._buffers) != buffer_names
                or tuple(child._modules) != tuple(name for name, _ in modules)
            ):
                raise FunctionalizationError(
                    "Continuous transform registered schema changed; lower again."
                )
            for name, expected in modules:
                if child._modules[name] is not expected:
                    raise FunctionalizationError(
                        "Continuous transform child modules changed; lower again."
                    )
            if (
                "forward" in child.__dict__
                or any(name in child.__dict__ for name in _CALL_METHODS)
                or getattr(child, "_compiled_call_impl", None) is not None
                or child._forward_hooks
                or child._forward_pre_hooks
                or child._backward_hooks
                or child._backward_pre_hooks
            ):
                raise FunctionalizationError(
                    "Continuous transform canonical hooks changed; lower again."
                )
        actual_tensors = []
        for child, collection, name, schema, alias in self._guard_tensor_slots:
            value = getattr(child, collection)[name]
            if schema is None:
                if value is not None:
                    raise FunctionalizationError(
                        "Continuous transform registered schema changed; lower again."
                    )
            elif (
                not torch.is_tensor(value)
                or tuple(value.shape) != schema[0]
                or value.dtype != schema[1]
                or value.device != schema[2]
            ):
                raise FunctionalizationError(
                    "Continuous transform registered schema changed; lower again."
                )
            actual_tensors.append(value)
        for index, value in enumerate(actual_tensors):
            for previous in range(index):
                tied = (
                    self._guard_tensor_slots[index][4]
                    == self._guard_tensor_slots[previous][4]
                )
                if (value is actual_tensors[previous]) != tied:
                    raise FunctionalizationError(
                        "Continuous transform registered aliases changed; lower again."
                    )
        for owner, name, expected, kind in self._guard_attributes:
            value = owner[name] if kind == "global" else getattr(owner, name)
            if isinstance(
                expected,
                (
                    type(None),
                    bool,
                    int,
                    float,
                    complex,
                    str,
                    bytes,
                    tuple,
                    frozenset,
                    torch.dtype,
                    torch.device,
                ),
            ):
                changed = value is not expected and value != expected
            else:
                changed = value is not expected
            if changed:
                raise FunctionalizationError(
                    "Continuous transform configuration or dependencies changed; lower again."
                )
        for function, code, defaults, kwdefaults in self._guard_functions:
            if (
                function.__code__ is not code
                or function.__defaults__ != defaults
                or function.__kwdefaults__ != kwdefaults
            ):
                raise FunctionalizationError(
                    "Continuous transform callable changed; lower again."
                )
        matches = (
            _compiled_signature_matches(self)
            if compiling
            else self._signature(transform, strict=False) == self._fingerprint
        )
        if not matches:
            raise FunctionalizationError(
                "Continuous transform configuration changed; lower again."
            )

    def validate_compatible(self, transform):
        if self._signature(transform, strict=True) != self._fingerprint:
            raise FunctionalizationError(
                "Continuous transform configuration does not match this plan."
            )
        values = _named_tensors(transform) if self.is_module else {}
        self.validate(values or None)

    def extract(self, transform):
        self.validate_compatible(transform)
        return (
            {
                name: _snapshot(value, parameter=True)
                for name, value in _named_tensors(transform).items()
            }
            if self._schema
            else None
        )

    def validate(self, parameters):
        if not self._schema:
            if parameters is not None:
                raise ValueError("This transform requires transform_parameters=None")
            return
        if not isinstance(parameters, Mapping) or set(parameters) != set(self._schema):
            raise ValueError(
                "transform_parameters must match the registered transform tensor schema"
            )
        for name, (shape, dtype, device) in self._schema.items():
            value = parameters[name]
            if not torch.is_tensor(value):
                raise TypeError(f"transform_parameters.{name} must be a Tensor")
            if (
                tuple(value.shape) != shape
                or value.dtype != dtype
                or value.device != device
            ):
                raise ValueError(
                    f"transform_parameters.{name} must match {device}/{dtype}/{shape}"
                )

    def apply(self, values, parameters):
        if self._execution is None:
            return values
        if self.is_module:
            result = functional_call(
                self._execution,
                {} if parameters is None else dict(parameters),
                (values,),
                strict=True,
                tie_weights=True,
            )
        else:
            result = self._execution(values)
        if not torch.is_tensor(result):
            raise TypeError("Continuous transform must return a Tensor")
        if result.device != self.device or result.dtype != self.dtype:
            raise ValueError(
                "Continuous transform output must preserve target dtype and device"
            )
        if torch.broadcast_shapes(tuple(result.shape), (self.size,)) != (self.size,):
            raise ValueError(
                "Continuous transform output must broadcast to its edge vector"
            )
        return result

    def _probe(self):
        if self._execution is None:
            return
        p = _helpers()
        parameters = self.extract(self._source)
        classes = (
            tuple(
                sorted(
                    {
                        cls
                        for child in self._execution.modules()
                        for cls in type(child).__mro__
                        if cls not in (torch.nn.Module, object)
                        and not cls.__module__.startswith("torch.")
                    },
                    key=lambda cls: (cls.__module__, cls.__qualname__),
                )
            )
            if self.is_module
            else ()
        )
        class_state = p._class_state_snapshot(classes)
        rng_state = p._global_rng_snapshot(self.device)
        changed_classes, changed_rng = set(), set()
        try:
            with torch.inference_mode(False), torch.no_grad():
                for fill in (0.0, -0.75, 1.25):
                    argument = torch.full(
                        (self.size,), fill, device=self.device, dtype=self.dtype
                    )
                    arguments = (argument, parameters)
                    before = p._audit_python_value_signature(arguments, set())
                    registered = (
                        p._registered_tensor_bindings(self._execution)
                        if self.is_module
                        else None
                    )
                    python_state = (
                        p._transition_python_state_snapshot(self._execution)
                        if self.is_module
                        else None
                    )
                    output = self.apply(argument, parameters)
                    first = output.detach().clone()
                    again = self.apply(argument, parameters)
                    if p._audit_tensor_numerical_value(
                        first
                    ) != p._audit_tensor_numerical_value(again):
                        raise FunctionalizationError(
                            "Continuous transform is not deterministic from its explicit inputs."
                        )
                    if before != p._audit_python_value_signature(arguments, set()) or (
                        self.is_module
                        and (
                            registered != p._registered_tensor_bindings(self._execution)
                            or python_state
                            != p._transition_python_state_snapshot(self._execution)
                        )
                    ):
                        raise FunctionalizationError(
                            "Continuous transform mutated inputs, registered tensors, or Python state."
                        )
                # vmap rejects hidden randomness and data-dependent Python
                # branches even when the ordinary probe would take a quiet arm.
                batched = torch.stack((argument, -argument))
                torch.vmap(lambda item: self.apply(item, parameters))(batched)
        except FunctionalizationError:
            raise
        except Exception as exc:
            raise FunctionalizationError(
                f"Continuous transform pure forward audit failed: {exc}"
            ) from exc
        finally:
            changed_classes.update(p._restore_class_state(class_state))
            changed_rng.update(p._global_rng_changes(rng_state, self.device))
            p._restore_global_rng(rng_state, self.device)
        if changed_classes or changed_rng:
            raise FunctionalizationError(
                "Continuous transform consumed implicit RNG or mutated shared class state."
            )
