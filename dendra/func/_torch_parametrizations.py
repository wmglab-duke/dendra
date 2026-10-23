"""Bindings for native PyTorch parametrizations; their getters own the math."""

from __future__ import annotations

import inspect
from types import CodeType, FunctionType, MethodType

import torch
from torch.nn.utils import parametrize

from dendra.helpers import current_check_purity

from ._types import FunctionalizationError

_NATIVE_LIST_FORWARD = parametrize.ParametrizationList.forward
_NATIVE_LIST_FORWARD_CODE = _NATIVE_LIST_FORWARD.__code__


def _nested_code(function, name):
    return next(
        (
            value
            for value in function.__code__.co_consts
            if isinstance(value, CodeType) and value.co_name == name
        ),
        None,
    )


_NATIVE_GETTER_CODE = _nested_code(parametrize._inject_property, "get_parametrized")
_NATIVE_SETTER_CODE = _nested_code(parametrize._inject_property, "set_original")
_NATIVE_CACHED_GETTER_CODE = _nested_code(
    parametrize._inject_property, "get_cached_parametrization"
)


def _closure_values(function):
    try:
        return {
            name: cell.cell_contents
            for name, cell in zip(
                function.__code__.co_freevars, function.__closure__ or (), strict=True
            )
        }
    except (AttributeError, ValueError):
        return {}


def _matches_function(function, code, namespace):
    return (
        isinstance(function, FunctionType)
        and code is not None
        and function.__code__ is code
        and function.__globals__ is namespace
        and function.__defaults__ is None
        and not function.__kwdefaults__
    )


def _canonical_property(owner, name):
    """Recognize installed PyTorch accessors and our private uncached getter.

    Deepcopy retains PyTorch's generated class and its getter closure, including
    the source owner used only as a process-global cache key. Its tensor-name
    binding and nested cache function remain canonical; its captured owner need
    not be the copied owner. The private getter bypasses that cache entirely.
    """
    descriptor = inspect.getattr_static(type(owner), name, None)
    if type(descriptor) is not property or descriptor.fdel is not None:
        return False
    setter = descriptor.fset
    if not _matches_function(setter, _NATIVE_SETTER_CODE, vars(parametrize)):
        return False
    if _closure_values(setter) != {"tensor_name": name}:
        return False
    getter = descriptor.fget
    if _matches_function(getter, _UNCACHED_GETTER_CODE, globals()):
        return _closure_values(getter) == {"name": name}
    if not _matches_function(getter, _NATIVE_GETTER_CODE, vars(parametrize)):
        return False
    closure = _closure_values(getter)
    if (
        set(closure) != {"tensor_name", "get_cached_parametrization"}
        or closure["tensor_name"] != name
    ):
        return False
    cached = closure["get_cached_parametrization"]
    if not _matches_function(cached, _NATIVE_CACHED_GETTER_CODE, vars(parametrize)):
        return False
    cached_closure = _closure_values(cached)
    return (
        set(cached_closure) == {"module", "tensor_name"}
        and cached_closure["tensor_name"] == name
    )


def _callable_binding(value):
    target = value.__func__ if isinstance(value, MethodType) else value
    return id(target), id(getattr(target, "__code__", None))


def source_binding_signature(population):
    """Return only registered layout and execution bindings, without purity scans.

    Keep the source paths, descriptors, list dispatch, and transform dispatch
    visible to Dynamo. A warmed graph must not keep executing private copies
    after their corresponding source bindings change.
    """
    from ._stimuli import _is_internal_module_state, _trusted_state_signature

    entries = []
    for path, owner, registry in registries(population):
        for name, transforms in registry.items():
            descriptor = getattr(type(owner), name, None)
            entries.append(
                (
                    path,
                    name,
                    id(owner),
                    id(type(owner)),
                    id(registry),
                    id(transforms),
                    id(type(transforms)),
                    id(descriptor),
                    _callable_binding(getattr(descriptor, "fget", None)),
                    _callable_binding(getattr(descriptor, "fset", None)),
                    _callable_binding(transforms.forward),
                    transforms.is_tensor,
                    transforms.ntensors,
                    transforms.unsafe,
                    tuple(
                        (
                            child_path,
                            id(child),
                            id(type(child)),
                            _callable_binding(child.forward),
                            tuple(
                                (key, _trusted_state_signature(value))
                                for key, value in vars(child).items()
                                if not _is_internal_module_state(child, key)
                                and not isinstance(value, torch.nn.Module)
                            ),
                            tuple(
                                (
                                    hook_name,
                                    tuple(
                                        (key, id(hook))
                                        for key, hook in getattr(
                                            child, hook_name
                                        ).items()
                                    ),
                                )
                                for hook_name in (
                                    "_forward_pre_hooks",
                                    "_forward_hooks",
                                    "_backward_pre_hooks",
                                    "_backward_hooks",
                                )
                            ),
                            tuple(
                                (
                                    kind,
                                    key,
                                    id(value),
                                    None if value is None else tuple(value.shape),
                                    None if value is None else value.dtype,
                                    None if value is None else value.device,
                                )
                                for kind, values in (
                                    ("parameter", child._parameters),
                                    ("buffer", child._buffers),
                                )
                                for key, value in values.items()
                            ),
                        )
                        for child_path, child in transforms.named_modules()
                    ),
                )
            )
    return tuple(entries)


def canonical_type(module):
    """Remove only PyTorch's generated parametrization wrapper."""
    if type(
        module
    ).__module__ == "torch.nn.utils.parametrize" and parametrize.is_parametrized(
        module
    ):
        return parametrize.type_before_parametrizations(module)
    return type(module)


def registries(population):
    """Yield each registered owner once, including nested parameter sources."""
    for path, owner in population.named_modules():
        registry = owner._modules.get("parametrizations")
        if registry is not None and parametrize.is_parametrized(owner):
            yield path, owner, registry


def buffer_slots(population):
    """Expose originals and registered transform buffers by native paths."""
    for path, _owner, registry in registries(population):
        prefix = f"{path + '.' if path else ''}parametrizations"
        for relative, value in registry.named_buffers(remove_duplicate=False):
            yield f"{prefix}.{relative}", value


def parameter_names(population):
    identities = {
        id(value)
        for _path, _owner, registry in registries(population)
        for value in registry.parameters()
    }
    identities.update(id(value) for _path, value in buffer_slots(population))
    return {
        name for name, value in population.named_parameters() if id(value) in identities
    }


def original_parameter_paths(population, name):
    """Resolve a required raw leaf through a parametrized tensor attribute."""
    prefix, _sep, local = name.rpartition(".")
    try:
        owner = population.get_submodule(prefix)
    except AttributeError:
        return ()
    registry = owner._modules.get("parametrizations")
    if registry is None or local not in registry:
        return ()
    target = registry[local]
    return tuple(
        f"{prefix + '.' if prefix else ''}parametrizations.{local}.{key}"
        for key, value in target._parameters.items()
        if key.startswith("original") and value is not None
    )


def failures(population, *, diagnostic=None):
    """Validate fixed tensor ownership; authored purity remains opt-in."""
    registered = tuple(registries(population))
    if not registered:
        return ()
    from dendra.models.multi import MultiPopulation

    from . import _population as p

    diagnostic = current_check_purity() if diagnostic is None else diagnostic
    errors = []
    from ._initialization_sources import initialization_source_runtime_buffer_ids

    runtime_ids = initialization_source_runtime_buffer_ids(population)
    conflicts = tuple(
        name for name, value in buffer_slots(population) if id(value) in runtime_ids
    )
    if conflicts:
        errors.append(
            f"Torch parametrization buffers {conflicts} alias runtime state, geometry, "
            "or reconstructed parameters; use registered raw Parameters or "
            "independent config buffers"
        )
    for path, owner, registry in registered:
        if isinstance(owner, MultiPopulation):
            errors.append(
                "Torch parametrizations on a packed parent are currently unsupported; "
                "register the parametrization on its component Populations instead. "
                "Native packed initialization assigns assembled fields through "
                "right_inverse, which cannot use the pure forward-only adapter."
            )
            continue
        for name, transforms in registry.items():
            label = f"{path + '.' if path else ''}{name}"
            if not _canonical_property(owner, name):
                errors.append(
                    f"Torch parametrization {label!r} needs its canonical generated "
                    "property binding; custom getters or setters are currently unsupported"
                )
            if type(transforms) is not parametrize.ParametrizationList or (
                "forward" in vars(transforms)
                or inspect.getattr_static(type(transforms), "forward")
                is not _NATIVE_LIST_FORWARD
                or _NATIVE_LIST_FORWARD.__code__ is not _NATIVE_LIST_FORWARD_CODE
            ):
                errors.append(
                    f"Torch parametrization {label!r} needs its canonical list adapter"
                )
                continue
            if name in getattr(
                owner, "in_graph_parametrizations", {}
            ) or name in getattr(owner, "additional_parameters", {}):
                errors.append(
                    f"Torch parametrization {label!r} also has regional or in-graph "
                    "overrides; combined target writes are currently unsupported"
                )
            mutable = (
                set(getattr(owner, "_state", {}))
                | set(getattr(owner, "_carry", {}))
                | set(getattr(owner, "_assigned", {}))
            )
            if owner is population:
                mutable.update(("v", "vc", "t", "i_membrane", "_duration_remainder"))
            mutable.update(p._derived_buffer_names(owner))
            mutable.update(p._timestep_buffer_names(owner))
            mutable.update(getattr(owner, "random_parameters", ()))
            mutable.update(getattr(owner, "runtime_noises", ()))
            mutable.update(getattr(owner, "_functional_material_slots", ()))
            if isinstance(owner, p.Material):
                mutable.update(owner._material_fields)
            if owner is getattr(population, "integrator", None):
                # Solver buffers are rebuilt or advanced by the integrator,
                # rather than owned by a fixed parameter-source recipe.
                mutable.update(registry)
            if name in mutable:
                errors.append(
                    f"Torch parametrization {label!r} targets mutable state or a "
                    "derived workspace; parametrize a raw parameter or fixed buffer"
                )
            for transform in transforms.modules():
                if p._module_autograd_hook_signature(transform):
                    errors.append(
                        f"Torch parametrization {label!r} has registered hooks"
                    )
            if diagnostic:
                for transform in transforms:
                    if type(transform).__module__ == "dendra.models.core" and type(
                        transform
                    ).__qualname__.startswith("Myelinated."):
                        continue  # canonical borrowed geometry is bound by its adapter
                    try:
                        p._transform_structure_signature(transform, strict=True)
                    except FunctionalizationError as exc:
                        errors.append(str(exc))
    return tuple(errors)


def property_signature(owner, name, *, label, strict):
    """Treat the canonical getter as a binding to its explicit pure transforms."""
    registry = owner._modules.get("parametrizations")
    if registry is None or name not in registry:
        return None
    if not _canonical_property(owner, name):
        raise FunctionalizationError(
            f"Torch parametrization {label}.{name} needs its canonical generated property binding"
        )
    from . import _population as p

    transforms = registry[name]
    return (
        "torch-parametrization",
        name,
        tuple(
            p._initialization_callable_contract_signature(
                transform.forward,
                label=f"{label}.{name}.{index}",
                owner=transform,
                strict=strict,
                diagnostic=True,
            )
            for index, transform in enumerate(transforms)
        ),
    )


def _getter(name):
    def get(self):
        # Use the same native list evaluation without the optional process-wide
        # parametrization cache: explicit packets can change between calls.
        return _checked_forward(self.parametrizations[name])

    return get


_UNCACHED_GETTER_CODE = _nested_code(_getter, "get")


def _checked_forward(self, *originals):
    result = _NATIVE_LIST_FORWARD(self, *originals)
    expected = self._functional_output_schema
    if (
        not torch.is_tensor(result)
        or (tuple(result.shape), result.dtype, result.device) != expected
    ):
        raise FunctionalizationError(
            "Torch parametrization output changed shape, dtype, or device; lower again"
        )
    return result


def install_output_schemas(population):
    """Validate native getters on execution copies without altering arithmetic."""
    for _path, owner, registry in registries(population):
        cls = type(owner)
        # Let Dendra's declaration metaclasses rebuild their own inherited
        # metadata. Copying their generated class dictionaries would redeclare
        # State solver fields and other normalized framework attributes.
        namespace = {
            name: vars(cls)[name]
            for name in ("__module__", "__getstate__", "__deepcopy__")
            if name in vars(cls)
        }
        for name in registry:
            if not _canonical_property(owner, name):
                raise FunctionalizationError(
                    f"Torch parametrization {name!r} needs its canonical generated property binding"
                )
            descriptor = inspect.getattr_static(cls, name)
            namespace[name] = property(_getter(name), descriptor.fset, descriptor.fdel)
        owner.__class__ = type(cls.__name__, cls.__bases__, namespace)
        for transforms in registry.values():
            with torch.no_grad():
                value = transforms()
            if not torch.is_tensor(value):
                raise FunctionalizationError(
                    "Torch parametrization must return a Tensor"
                )
            transforms._functional_output_schema = (
                tuple(value.shape),
                value.dtype,
                value.device,
            )
