"""Audited zero-argument sources for shared Ion and Material initialization.

Source arithmetic remains in the native module and shared material resolver.
This adapter freezes its dependency frame and qualifies an isolated execution
copy; it does not introduce a second initializer implementation.
"""

from __future__ import annotations

import inspect
import types

import torch
from torch.func import functional_call

from dendra.models import parametric
from dendra.models.mechanisms import _materials
from dendra.models.multi import _PackedMaterialInitial
from dendra.models.parametric import Bounded, NegativeParam, cacheable

from ._continuous_transform import (
    _CALL_METHODS,
    _COPY_METHODS,
    _function_signature,
    _named_tensors,
    _safe_modules,
    _tensor_layout,
)
from ._mechanism_injections import _binding, _literal
from ._stimuli import _is_internal_module_state
from ._types import FunctionalizationError

_CANONICAL_CACHE_FORWARD = cacheable.forward
_CANONICAL_CACHE_STATE = cacheable.__getstate__
_CANONICAL_COMPUTES = (Bounded._compute, NegativeParam._compute)
_CANONICAL_PACKED_FORWARD = _PackedMaterialInitial.forward
_CANONICAL_BOUNDED_HELPERS = (Bounded._apply_upper, Bounded._upper_mode)
_CANONICAL_MEMBERS = (
    (Bounded, "_compute", Bounded._compute),
    (Bounded, "_apply_upper", Bounded._apply_upper),
    (Bounded, "_upper_mode", Bounded._upper_mode),
)
_CANONICAL_FUNCTIONS = {
    function: function.__code__
    for function in (
        _CANONICAL_CACHE_FORWARD,
        _CANONICAL_CACHE_STATE,
        *_CANONICAL_COMPUTES,
        _CANONICAL_PACKED_FORWARD,
        Bounded._upper_mode,
        Bounded._apply_upper,
        parametric._softplus_transform,
        parametric._softcap_transform,
        _materials._initial_source_cache_free_forward,
        _materials._resolve_initial_source,
        _materials._configure_initial_source_execution,
    )
}
_CANONICAL_BINDINGS = tuple(
    (function.__globals__, name, function.__globals__[name])
    for function in _CANONICAL_FUNCTIONS
    for name in function.__code__.co_names
    if name in function.__globals__
)


def _helpers():
    from . import _population

    return _population


def _registered_modules(module, *, duplicates=False):
    """Traverse framework registration dictionaries without authored methods."""
    seen = set()

    def visit(path, child, ancestors):
        identity = id(child)
        if identity in ancestors or (not duplicates and identity in seen):
            return
        seen.add(identity)
        yield path, child
        for name, nested in vars(child)["_modules"].items():
            if nested is not None:
                yield from visit(
                    f"{path}.{name}" if path else name,
                    nested,
                    ancestors | {identity},
                )

    return visit("", module, set())


def _registered_tensor_slots(module, *, duplicates=False):
    for path, child in _registered_modules(module, duplicates=duplicates):
        for collection in ("_parameters", "_buffers"):
            for name, value in vars(child)[collection].items():
                if value is not None:
                    yield f"{path}.{name}" if path else name, value


def _initial_source_entries(population, *, include_tensors=False):
    """Yield source slots, preserving aliases and native field order.

    Ordinary recipe discovery omits direct Tensor sources to keep its empty
    path cheap. Cold parameter ownership classification may include them.
    """
    handler = getattr(population, "mech", None)
    if handler is None:
        return ()
    entries = []
    for ion in handler.ions.values():
        for name in ("e_init", "i_init", "o_init"):
            source = getattr(ion, name)
            if include_tensors or isinstance(source, torch.nn.Module):
                entries.append((ion, name, source))
    for material in handler.materials.values():
        if not material._material_has_initial_sources:
            continue
        for field in material._material_fields:
            source = material.initial_source(field)
            if include_tensors or isinstance(source, torch.nn.Module):
                entries.append((material, f"_initial_sources.{field}", source))
    # Tensor-only models need no full module/alias inventory.
    if not entries:
        return ()
    paths = {id(module): path for path, module in _registered_modules(population)}
    return tuple(
        (f"{paths[id(owner)]}.{name}", source) for owner, name, source in entries
    )


def initialization_source_roots(population):
    """Return registered Module sources in canonical initialization order."""
    return tuple(
        (path, source)
        for path, source in _initial_source_entries(population)
        if isinstance(source, torch.nn.Module)
    )


def initialization_source_constant_slots(source):
    """Yield registered source buffers, including tied alias paths."""
    for path, child in _registered_modules(source, duplicates=True):
        for name, value in vars(child)["_buffers"].items():
            if value is not None:
                yield f"{path}.{name}" if path else name, value


def initialization_source_runtime_buffer_ids(population):
    """Identify reconstructed/runtime ownership that cannot be source config.

    Registered raw Parameters may deliberately have additional buffer aliases.
    Their identity remains an explicit trainable leaf and takes precedence over
    the buffer classification. Ordinary source config and other static module
    buffers are not classified as simulation state.
    """
    from dendra.models.core import Population
    from dendra.models.mechanisms._materials import Material
    from dendra.models.parametric import Parameterized

    p = _helpers()
    if getattr(population, "mech", None) is None and hasattr(population, "populations"):
        identities = set()
        for child in population.populations.values():
            identities.update(initialization_source_runtime_buffer_ids(child))
        return identities - {
            id(value)
            for _name, value in _registered_tensor_slots(population)
            if isinstance(value, torch.nn.Parameter)
        }
    source_ids = {
        id(child)
        for _path, source in initialization_source_roots(population)
        for _nested_path, child in _registered_modules(source)
    }
    raw_ids = {
        id(value)
        for _name, value in _registered_tensor_slots(population)
        if isinstance(value, torch.nn.Parameter)
    }
    handler = getattr(population, "mech", None)
    integrator = getattr(population, "integrator", None)
    identities = set()
    for _module_path, child in _registered_modules(population):
        if id(child) in source_ids:
            continue
        names = set()
        if isinstance(child, Population) or child is handler or child is integrator:
            names.update(child._buffers)
        if isinstance(child, Material):
            names.update(child._material_fields)
        if isinstance(child, Parameterized):
            names.update(p._parameter_materialization_output_names(child))
            names.update(p._derived_buffer_names(child))
            names.update(p._timestep_buffer_names(child))
            names.update(getattr(child, "random_parameters", ()))
            names.update(getattr(child, "runtime_noises", ()))
            names.update(getattr(child, "_functional_material_slots", ()))
            for category in ("_state", "_carry", "_assigned"):
                names.update(getattr(child, category, ()))
        identities.update(
            id(value)
            for name in names
            if (value := child._buffers.get(name)) is not None
        )
    return identities - raw_ids


def initialization_source_ownership_failures(population, *, runtime_buffer_ids=None):
    """Report source config buffers that cross simulation ownership domains."""
    if runtime_buffer_ids is None:
        runtime_buffer_ids = initialization_source_runtime_buffer_ids(population)
    if getattr(population, "mech", None) is None and hasattr(population, "populations"):
        return tuple(
            f"Population {name!r}: {failure}"
            for name, child in population.populations.items()
            for failure in initialization_source_ownership_failures(
                child, runtime_buffer_ids=runtime_buffer_ids
            )
        )
    failures = []
    for path, source in initialization_source_roots(population):
        aliases = tuple(
            name
            for name, value in initialization_source_constant_slots(source)
            if id(value) in runtime_buffer_ids
        )
        if aliases:
            failures.append(
                f"Initial-value source {path!r} config buffers {aliases} alias "
                "runtime state, reconstructed parameters, or geometry; use "
                "registered raw Parameters or independent config buffers"
            )
    return tuple(failures)


def configure_initialization_source_execution(population):
    """Prepare private source copies without changing authored training modes."""
    from dendra.models.mechanisms._materials import _configure_initial_source_execution

    for _path, source in initialization_source_roots(population):
        _configure_initial_source_execution(source)


def _structural_source_view(source):
    """Erase canonical private adapters from a read-only structural view.

    Internal packed-component plans can be constructed from execution copies.
    Their source contracts equal ordinary authored recipes; no live module,
    tensor, training flag, or registration is modified to compare them.
    """
    memo = {}

    def visit(module):
        if id(module) in memo:
            return memo[id(module)]
        view = object.__new__(type(module))
        object.__setattr__(view, "__dict__", dict(vars(module)))
        memo[id(module)] = view
        forward = vars(module).get("forward")
        if (
            isinstance(module, cacheable)
            and getattr(forward, "__func__", None)
            is _materials._initial_source_cache_free_forward
            and getattr(forward, "__self__", None) is module
        ):
            view.__dict__.pop("forward", None)
        if vars(module).get("_initialization_source_cache_free") is True:
            view.__dict__.pop("_initialization_source_cache_free")
        object.__setattr__(
            view,
            "_modules",
            {
                name: None if child is None else visit(child)
                for name, child in module._modules.items()
            },
        )
        return view

    return visit(source)


def _require_static_method(source, name):
    """Qualify dispatch without invoking an authored descriptor getter."""
    descriptor = inspect.getattr_static(source, name, None)
    if type(descriptor) not in (
        types.FunctionType,
        types.MethodType,
        staticmethod,
        classmethod,
    ):
        raise FunctionalizationError(
            f"Initial-value source {name!r} requires ordinary method dispatch."
        )


def _trusted_receiver_signatures(source, *, strict):
    """Audit authored overrides reached by otherwise canonical arithmetic.

    Bounded's numerical body calls receiver helpers and reads configuration.
    Trusting that body must not also trust subclass helper overrides or execute
    configuration descriptors before the isolated evaluation audit.
    """
    p = _helpers()
    functions = (
        (_CANONICAL_COMPUTES[0], *_CANONICAL_BOUNDED_HELPERS)
        if isinstance(source, Bounded)
        else (_CANONICAL_PACKED_FORWARD,)
    )
    signatures = []
    names = set()
    for function in functions:
        attributes, _escapes = p._direct_receiver_attribute_names(
            function, function.__code__.co_varnames[0]
        )
        names.update(attributes)
    for name in sorted(names):
        if (
            name in source._parameters
            or name in source._buffers
            or name in source._modules
        ):
            continue
        descriptor = inspect.getattr_static(source, name)
        if (
            type(descriptor)
            not in (
                types.FunctionType,
                staticmethod,
                classmethod,
            )
            and inspect.getattr_static(type(descriptor), "__get__", None) is not None
        ):
            raise FunctionalizationError(
                f"Canonical initial-value source configuration {name!r} "
                "cannot use an authored descriptor."
            )
        if inspect.isfunction(descriptor) and descriptor in _CANONICAL_BOUNDED_HELPERS:
            continue
        if type(descriptor) in (types.FunctionType, staticmethod, classmethod):
            method = descriptor.__get__(source, type(source))
            signatures.append(
                (name, _function_signature(method, source, strict=strict))
            )
    return tuple(signatures)


def _qualify(source, *, strict):
    """Reuse the continuous-callable proof with narrow native recipe leaves."""
    p = _helpers()
    modules = _safe_modules(source)
    for _path, child in modules:
        _require_static_method(child, "forward")
        if isinstance(child, cacheable):
            _require_static_method(child, "_compute")
    if any(
        inspect.getattr_static(type(child), "_initialization_source_cache_free", None)
        is not None
        for _path, child in modules
    ):
        raise FunctionalizationError(
            "Initial-value source uses reserved functional execution metadata on its class."
        )
    marked = vars(source).get("_initialization_source_cache_free") is True
    if marked:
        for _path, child in modules:
            if isinstance(child, cacheable) and (
                getattr(child.forward, "__func__", None)
                is not _materials._initial_source_cache_free_forward
                or getattr(child.forward, "__self__", None) is not child
            ):
                raise FunctionalizationError(
                    "Initial-value source uses reserved functional execution metadata without canonical adapters."
                )
    for _path, child in modules:
        for name in ("__setattr__", "__delattr__"):
            if inspect.getattr_static(type(child), name) is not inspect.getattr_static(
                torch.nn.Module, name
            ):
                raise FunctionalizationError(
                    "Initial-value sources require canonical registration attribute dispatch."
                )
    source = _structural_source_view(source)
    signatures = []
    for path, child in _safe_modules(source):
        if "_initialization_source_cache_free" in vars(child):
            raise FunctionalizationError(
                "Initial-value source uses reserved functional execution metadata."
            )
        if getattr(child, "_compiled_call_impl", None) is not None:
            raise FunctionalizationError(
                "Initial-value sources require canonical module call dispatch."
            )
        if "forward" in vars(child) or any(
            name in vars(child) for name in _CALL_METHODS
        ):
            raise FunctionalizationError(
                "Initial-value source call/attribute methods must be defined on its class."
            )
        for name in _COPY_METHODS:
            actual = inspect.getattr_static(type(child), name, None)
            standard = inspect.getattr_static(torch.nn.Module, name, None)
            if name == "__getstate__" and isinstance(child, cacheable):
                standard = _CANONICAL_CACHE_STATE
            if actual is not standard:
                raise FunctionalizationError(
                    "Initial-value sources cannot override deepcopy/serialization hooks."
                )
        if p._module_autograd_hook_signature(child):
            raise FunctionalizationError(
                "Initial-value source hooks are not a pure initialization contract."
            )
        for name in (*child._parameters, *child._buffers, *child._modules):
            if inspect.getattr_static(type(child), name, None) is not None:
                raise FunctionalizationError(
                    f"Initial-value source registered field {name!r} is shadowed by a descriptor."
                )
        for value in child.parameters(recurse=False):
            if getattr(value, "_backward_hooks", None):
                raise FunctionalizationError(
                    "Initial-value source parameters cannot retain autograd hooks."
                )
        method = child.forward
        trusted = False
        if isinstance(child, cacheable):
            if getattr(
                child.forward, "__func__", None
            ) is not _CANONICAL_CACHE_FORWARD or "_compute" in vars(child):
                raise FunctionalizationError(
                    "Cached initial-value sources require canonical cache dispatch."
                )
            method = child._compute
            trusted = getattr(method, "__func__", None) in _CANONICAL_COMPUTES
        elif type(child) is _PackedMaterialInitial:
            if getattr(method, "__func__", None) is not _CANONICAL_PACKED_FORWARD:
                raise FunctionalizationError(
                    "Packed initial-value sources require canonical composition."
                )
            trusted = True
        if (
            any(
                function.__code__ is not code
                for function, code in _CANONICAL_FUNCTIONS.items()
            )
            or any(
                namespace.get(name) is not expected
                for namespace, name, expected in _CANONICAL_BINDINGS
            )
            or any(
                inspect.getattr_static(owner, name) is not expected
                for owner, name, expected in _CANONICAL_MEMBERS
            )
        ):
            raise FunctionalizationError(
                "Shared initial-value source helpers changed; lower again."
            )
        signatures.append(
            (
                path,
                child.training,
                p._transform_callable_structure_signature(
                    method, label="initial-value source", strict=False
                )
                if trusted
                else _function_signature(method, child, strict=strict),
                _trusted_receiver_signatures(child, strict=strict) if trusted else (),
            )
        )
    return (
        p._transform_structure_signature(source, strict=strict),
        tuple(signatures),
        _tensor_layout(source),
    )


class _SourceCall(torch.nn.Module):
    """Call the shared native resolver with an explicit registered recipe."""

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.audit = True

    def forward(self):
        from dendra.models.mechanisms._materials import _resolve_initial_source

        if not self.audit:
            return _resolve_initial_source(self.source)
        p = _helpers()
        registered = p._registered_tensor_bindings(self.source)
        python = p._transition_python_state_snapshot(self.source)
        output = _resolve_initial_source(self.source)
        if registered != p._registered_tensor_bindings(
            self.source
        ) or python != p._transition_python_state_snapshot(self.source):
            raise FunctionalizationError(
                "Initial-value source mutated registered tensors or Python state."
            )
        return output


def _audit_source(source):
    """Probe once on an isolated copy, restoring class state and every RNG."""
    p = _helpers()
    modules = _safe_modules(source)
    classes = tuple(
        sorted(
            {
                cls
                for _path, child in modules
                for cls in type(child).__mro__
                if cls not in (torch.nn.Module, object)
                and not cls.__module__.startswith("torch.")
            },
            key=lambda cls: (cls.__module__, cls.__qualname__),
        )
    )
    tensors = _named_tensors(source)
    first = next(iter(tensors.values()), None)
    device = torch.device("cpu") if first is None else first.device
    class_state = p._class_state_snapshot(classes)
    rng_state = p._global_rng_snapshot(device)
    changed = ()
    try:
        execution = p._deepcopy_execution_module(source)
        if execution is source:
            raise FunctionalizationError(
                "Initial-value source deepcopy retained its live owner."
            )
        from dendra.models.mechanisms._materials import (
            _configure_initial_source_execution,
        )

        _configure_initial_source_execution(execution)
        call = _SourceCall(execution)
        arguments = {
            f"source.{name}": p._clone_tensor(value) for name, value in tensors.items()
        }

        def apply(values):
            output = functional_call(call, values, (), strict=True, tie_weights=True)
            if not torch.is_tensor(output):
                raise FunctionalizationError(
                    "Initial-value sources must return a Tensor."
                )
            return output

        with torch.inference_mode(False), torch.no_grad():
            python = p._transition_python_state_snapshot(call)
            before = p._audit_python_value_signature(arguments, set())
            result = apply(arguments).detach().clone()
            again = apply(arguments)
            if p._audit_tensor_numerical_value(
                result
            ) != p._audit_tensor_numerical_value(again):
                raise FunctionalizationError(
                    "Initial-value source is not deterministic from its explicit leaves."
                )
            if (
                python != p._transition_python_state_snapshot(call)
                or before != p._audit_python_value_signature(arguments, set())
                or any(
                    isinstance(child, cacheable) and child._cache is not None
                    for _module_path, child in _registered_modules(execution)
                )
            ):
                raise FunctionalizationError(
                    "Initial-value source mutated registered tensors, inputs, or Python state."
                )
            # Batched leaves exercise hidden RNG and data-dependent dispatch.
            call.audit = False
            if arguments:
                batched = {
                    name: torch.stack((value, value))
                    for name, value in arguments.items()
                }
                torch.vmap(apply)(batched)
            else:
                torch.vmap(lambda _lane: apply(arguments))(
                    torch.arange(2, device=device)
                )
    except FunctionalizationError:
        raise
    except Exception as exc:
        raise FunctionalizationError(
            f"Initial-value source pure evaluation audit failed: {exc}"
        ) from exc
    finally:
        changed = p._restore_class_state(class_state)
        rng_changes = p._global_rng_changes(rng_state, device)
        p._restore_global_rng(rng_state, device)
        if changed or rng_changes:
            raise FunctionalizationError(
                "Initial-value source consumed implicit RNG or mutated shared class state."
            )


def _member_binding(value):
    if type(value) in (str, bool, int, float, complex, type(None), tuple, list, dict):
        return _literal(value)
    if type(value) is property:
        return "property", tuple(
            None if method is None else _binding(method)
            for method in (value.fget, value.fset, value.fdel)
        )
    if type(value) in (classmethod, staticmethod):
        return _binding(value.__func__)
    if type(value) in (types.FunctionType, types.MethodType):
        return _binding(value)
    return _literal(value)


def _static_class_member(owner, name):
    """Look up guard metadata without invoking authored class descriptors."""
    for parent in owner.__mro__:
        if name in parent.__dict__:
            return parent.__dict__[name]
    return None


class _RawClassMemberIdentity:
    """Compare one raw class slot after functional_call dictionary writes."""

    def __init__(self, name, present, expected):
        self.name = name
        self.present = present
        self.expected = expected

    def __eq__(self, namespace):
        return (self.name in namespace) is self.present and namespace.get(
            self.name
        ) is self.expected


def _class_member_guards(owner, name):
    """Freeze the raw lookup chain once, without descriptor resolution."""
    for parent in owner.__mro__:
        # Dynamo treats Module.__setstate__ as a special deserialization entry;
        # guard authored shadowing without resolving that framework descriptor.
        if name == "__setstate__" and parent.__module__.startswith("torch."):
            break
        namespace = vars(parent)
        present = name in namespace
        yield parent, name, present, namespace.get(name)
        if present:
            break


def _collect_dependencies(roots):
    """Resolve the callable frame once; warm checks read stored containers."""
    from ._continuous_transform import _global_values

    dependencies, functions = [], []
    seen = set()
    seen_namespaces = set()

    def visit_namespace(namespace, names):
        key = id(namespace), tuple(names)
        if key in seen_namespaces:
            return
        seen_namespaces.add(key)
        for name in names:
            if name not in vars(namespace):
                continue
            value = vars(namespace)[name]
            dependencies.append(
                (
                    "namespace",
                    namespace,
                    name,
                    inspect.isfunction(value) or inspect.ismethod(value),
                )
            )
            if isinstance(value, types.ModuleType):
                visit_namespace(value, names)
            elif inspect.isfunction(value) and not value.__module__.startswith(
                "torch."
            ):
                visit(value, None)

    def visit(method, owner):
        function = getattr(method, "__func__", method)
        if not inspect.isfunction(function) or id(function) in seen:
            return
        seen.add(id(function))
        functions.append(function)
        for name, value in _global_values(function).items():
            dependencies.append(
                (
                    "global",
                    function.__globals__,
                    name,
                    inspect.isfunction(value) or inspect.ismethod(value),
                )
            )
            if isinstance(value, types.ModuleType):
                visit_namespace(value, function.__code__.co_names)
            elif inspect.isfunction(value):
                visit(value, None)
        for cell in function.__closure__ or ():
            dependencies.append(
                (
                    "cell",
                    cell,
                    "cell_contents",
                    inspect.isfunction(cell.cell_contents)
                    or inspect.ismethod(cell.cell_contents),
                )
            )
            if inspect.isfunction(cell.cell_contents):
                visit(cell.cell_contents, None)
        receiver = getattr(method, "__self__", None)
        if owner is None or receiver is not owner:
            return
        names, _escapes = _helpers()._direct_receiver_attribute_names(
            function,
            function.__code__.co_varnames[0] if function.__code__.co_argcount else None,
        )
        for name in names:
            if (
                name in owner._parameters
                or name in owner._buffers
                or name in owner._modules
            ):
                continue
            descriptor = inspect.getattr_static(owner, name, None)
            if isinstance(descriptor, property):
                visit(descriptor.fget.__get__(owner, type(owner)), owner)
            elif isinstance(descriptor, (staticmethod, classmethod)):
                visit(descriptor.__get__(owner, type(owner)), owner)
            elif inspect.isfunction(descriptor):
                visit(descriptor.__get__(owner, type(owner)), owner)

    for _path, source in roots:
        for _name, child in _safe_modules(source):
            if isinstance(child, cacheable):
                _require_static_method(child, "_compute")
                visit(child._compute, child)
            else:
                _require_static_method(child, "forward")
                visit(child.forward, child)
    for function in _CANONICAL_FUNCTIONS:
        if function not in (
            _CANONICAL_CACHE_FORWARD,
            _CANONICAL_CACHE_STATE,
            _materials._configure_initial_source_execution,
        ):
            visit(function, None)
    return tuple(functions), tuple(dependencies)


def _dependency_binding(record):
    kind, owner, name, function = record
    value = owner[name] if kind == "global" else getattr(owner, name)
    return _binding(value) if function else _literal(value)


def _module_bindings(source, class_members):
    """Read the explicit module frame, including canonical private adapters."""
    p = _helpers()
    aliases = {}
    records = []
    # Source admission has already qualified canonical metadata traversal.
    for path, child in _registered_modules(source):
        records.append(
            (
                path,
                id(child),
                id(type(child)),
                tuple(
                    (
                        collection,
                        name,
                        None
                        if value is None
                        else (
                            tuple(value.shape),
                            value.dtype,
                            value.device,
                            aliases.setdefault(id(value), (path, name)),
                            tuple(
                                (key, id(hook))
                                for key, hook in (
                                    getattr(value, "_backward_hooks", None) or {}
                                ).items()
                            ),
                        ),
                    )
                    for collection in ("_parameters", "_buffers")
                    for name, value in vars(child)[collection].items()
                ),
                tuple((name, id(module)) for name, module in child._modules.items()),
                p._module_autograd_hook_signature(child),
                tuple(
                    (name, _literal(value))
                    for name, value in vars(child).items()
                    if not _is_internal_module_state(child, name)
                    and not (name == "_cache" and isinstance(child, cacheable))
                    and not (
                        name == "forward"
                        and getattr(value, "__func__", None)
                        is _materials._initial_source_cache_free_forward
                    )
                ),
                tuple(
                    (
                        cls.__module__,
                        cls.__qualname__,
                        tuple(
                            (name, _member_binding(value))
                            for name, value in class_members.get(cls, ())
                        ),
                    )
                    for cls in type(child).__mro__
                    if cls is not object and not cls.__module__.startswith("torch.")
                ),
            )
        )
    return tuple(records)


def _alias_signature(population, roots):
    if not roots:
        return ()
    aliases = {}
    for path, value in _registered_tensor_slots(population):
        aliases.setdefault(id(value), []).append(path)
    return tuple(
        (path, name, tuple(aliases.get(id(value), ())))
        for path, source in roots
        for name, value in _registered_tensor_slots(source)
    )


def network_initialization_source_alias_signature(network):
    """Freeze sharing of initialization recipes and their registered leaves."""
    modules, tensors = {}, {}
    records = []
    for population_name, population in network.populations.items():
        roots = initialization_source_roots(population)
        for path, source in roots:
            module_alias = modules.setdefault(id(source), (population_name, path))
            records.append((population_name, path, "module", module_alias))
            for name, value in _registered_tensor_slots(source):
                alias = tensors.setdefault(id(value), (population_name, path, name))
                records.append((population_name, path, name, alias))
    return tuple(records)


class InitializationSourcePlan:
    """Frozen source recipes with lazy fresh-initialization admission.

    Unsupported source Modules do not prevent snapshot execution, which never
    evaluates them. Their diagnostics are exposed through ``failures`` and
    enforced by ``validate_initialization`` at the fresh boundary.
    """

    def __init__(self, population):
        self.population = population
        self.failures = ()
        self._contracts = []
        roots = initialization_source_roots(population)
        if not roots:
            self._aliases = ()
            self._class_binding_names = {}
            self._class_members = {}
            self._functions = self._dependencies = self._raw_class_guards = ()
            self._bindings = ()
            return
        audited = set()
        failures = list(initialization_source_ownership_failures(population))
        for path, source in initialization_source_roots(population):
            try:
                contract = _qualify(source, strict=True)
                if id(source) not in audited:
                    _audit_source(source)
                    audited.add(id(source))
                self._contracts.append((path, contract))
            except (
                FunctionalizationError,
                TypeError,
                ValueError,
                RuntimeError,
                AttributeError,
            ) as exc:
                failures.append(f"Initial-value source {path!r}: {exc}")
        self.failures = tuple(failures)
        self._aliases = _alias_signature(
            population, initialization_source_roots(population)
        )
        self._class_binding_names = {
            cls: tuple(
                dict.fromkeys(
                    (
                        *(
                            name
                            for name in vars(cls)
                            if name not in _helpers()._CLASS_STATE_METADATA_NAMES
                            and name not in ("__init__", "__new__")
                        ),
                        *_CALL_METHODS,
                        *(name for name in _COPY_METHODS if name != "__setstate__"),
                        "forward",
                        "_compute",
                    )
                )
            )
            for _path, source in initialization_source_roots(population)
            for _nested_path, child in _registered_modules(source)
            for cls in type(child).__mro__
            if cls is not object and not cls.__module__.startswith("torch.")
        }
        self._class_members = {
            cls: tuple((name, _static_class_member(cls, name)) for name in names)
            for cls, names in self._class_binding_names.items()
        }
        try:
            self._functions, self._dependencies = _collect_dependencies(
                initialization_source_roots(population)
            )
        except (FunctionalizationError, AttributeError, TypeError, ValueError):
            self._functions, self._dependencies = (), ()
        self._raw_class_guards = tuple(
            record
            for cls, names in self._class_binding_names.items()
            for name in (*names, "__setstate__")
            for record in _class_member_guards(cls, name)
        )
        self._bindings = self.source_binding_signature()

    def source_binding_signature(self, population=None):
        population = self.population if population is None else population
        roots = initialization_source_roots(population)
        if not roots:
            return ()
        try:
            return (
                tuple(
                    (path, _module_bindings(source, self._class_members))
                    for path, source in roots
                ),
                _alias_signature(population, roots),
                tuple(_binding(function) for function in self._functions),
                tuple(_binding(function) for function in _CANONICAL_FUNCTIONS),
                tuple(_dependency_binding(record) for record in self._dependencies),
                tuple(
                    cls.__dict__ == _RawClassMemberIdentity(name, present, expected)
                    for cls, name, present, expected in self._raw_class_guards
                ),
            )
        except (
            FunctionalizationError,
            TypeError,
            ValueError,
            RuntimeError,
            AttributeError,
        ):
            # A rejected source remains a valid snapshot-only dependency.
            return tuple((path, id(source), id(type(source))) for path, source in roots)

    def validate_population(self, population, *, source=False):
        roots = initialization_source_roots(population)
        if tuple(path for path, _recipe in roots) != tuple(
            path for path, _recipe in initialization_source_roots(self.population)
        ):
            raise FunctionalizationError(
                "Initial-value source registrations changed; lower again."
            )
        if not self.failures:
            contracts = tuple(
                (path, _qualify(recipe, strict=True)) for path, recipe in roots
            )
            if (
                contracts != tuple(self._contracts)
                or _alias_signature(population, roots) != self._aliases
            ):
                raise FunctionalizationError(
                    "Initial-value source behavior, layout, or aliases changed; lower again."
                )
        if source and self.source_binding_signature(population) != self._bindings:
            raise FunctionalizationError(
                "Initial-value source dependencies changed; lower again."
            )

    def validate_source(self):
        self.validate_population(self.population, source=True)

    def validate_initialization(self):
        if self.failures:
            raise FunctionalizationError("; ".join(self.failures))
        self.validate_source()

    def configure_execution(self, population):
        configure_initialization_source_execution(population)
