"""Frozen waveform ownership over shared pure Mechanism current evaluation."""

from __future__ import annotations

import copy
import dis
import inspect
import sys
from types import MethodType

import numpy as np
import torch

from dendra.helpers import current_check_purity
from dendra.models.mechanisms import Mechanism, _injections
from dendra.models.parametric import cacheable
from dendra.models.stim.waveform import Waveform

from ._stimuli import (
    _WAVEFORM_MODULE_DISPATCH,
    _audit_waveform_purity,
    _strict_state_signature,
    _tensor_content_signature,
    _validate_waveform_contract,
    _waveform_layout_signature,
)
from ._types import FunctionalizationError

_HELPER_NAMES = (
    "injection_values",
    "evaluate_injections",
    "_expand_injection_value",
    "register_waveform_injection",
    "clear_injections",
)
_CANONICAL_HELPERS = {
    name: inspect.getattr_static(Mechanism, name) for name in _HELPER_NAMES
}
_CANONICAL_CODES = {
    name: function.__code__ for name, function in _CANONICAL_HELPERS.items()
}
_CANONICAL_DEFAULTS = {
    name: (function.__defaults__, tuple((function.__kwdefaults__ or {}).items()))
    for name, function in _CANONICAL_HELPERS.items()
}
_SPEC_FIELDS = frozenset(("mask", "scale", "index", "n_selected", "current_name"))
_EXPAND = _injections.expand_injection_value
_EXPAND_CODE = _EXPAND.__code__
_MATH = _injections.math
_MATH_PROD = _MATH.prod


def _literal(value):
    if type(value) in (str, bool, int, float, type(None)):
        return value
    if type(value) in (tuple, list):
        return tuple(_literal(item) for item in value)
    if type(value) is dict:
        return "dict", tuple(
            (_literal(key), _literal(item)) for key, item in value.items()
        )
    return "nonliteral", id(type(value)), id(value)


def _layout(value):
    if not torch.is_tensor(value):
        return "not-tensor", id(type(value)), id(value)
    return tuple(value.shape), value.dtype, value.device, value.layout


def _binding(value):
    value = value.__func__ if isinstance(value, MethodType) else value
    return (
        id(value),
        id(getattr(value, "__code__", None)),
        _literal(getattr(value, "__defaults__", None)),
        tuple(
            (key, _literal(item))
            for key, item in (getattr(value, "__kwdefaults__", None) or {}).items()
        ),
    )


def _index(value, *, contents=False):
    if torch.is_tensor(value):
        return _tensor_content_signature(value) if contents else _layout(value)
    if isinstance(value, slice):
        return "slice", value.start, value.stop, value.step
    if type(value) in (list, tuple):
        return type(value).__name__, tuple(
            _index(item, contents=contents) for item in value
        )
    return _literal(value)


def _index_tensors(value, path=()):
    if torch.is_tensor(value):
        yield path, value
    elif type(value) in (tuple, list):
        for position, item in enumerate(value):
            yield from _index_tensors(item, (*path, position))


def configure_mechanism_injection_execution(population):
    """Disable derived parameter caches only on private execution copies."""
    for _path, owner in mechanism_injection_owners(population):
        for waveform in owner.injected_waveforms:
            from ._poisson_stimulation import configure_execution

            configure_execution(waveform)
            for module in waveform.modules():
                if isinstance(module, cacheable) and not isinstance(module, Waveform):
                    module.clear_cache()
                    module.training = True


def mechanism_injection_owners(population, owners=None):
    """Return injection owners in canonical handler order."""
    handler = getattr(population, "mech", None)
    if handler is None:
        return ()
    selected = None if owners is None else {id(owner) for owner in owners}
    registered = tuple(
        owner
        for collection in (handler.mechanisms, handler.material_processes)
        for owner in collection.values()
        if (selected is None or id(owner) in selected)
        and (
            type(getattr(owner, "_injection_specs", None)) is not list
            or owner._injection_specs
            or len(owner.injected_waveforms)
        )
    )
    if not registered:
        return ()
    paths = {id(owner): path for path, owner in population.named_modules()}
    return tuple((paths[id(owner)], owner) for owner in registered)


def mechanism_injection_constant_slots(owner):
    """Yield local registration and nested waveform buffers, with aliases."""
    seen = set()
    for spec in getattr(owner, "_injection_specs", ()):
        for field in ("mask", "scale", "index"):
            name = spec[field]
            if name not in seen:
                seen.add(name)
                yield name, getattr(owner, name)
    waveforms = getattr(owner, "injected_waveforms", None)
    if waveforms is not None:
        for name, value in waveforms.named_buffers(remove_duplicate=False):
            parent, _separator, field = name.rpartition(".")
            from dendra.models.stim.waveform._poisson import (
                POISSON_INVOCATION_POSITION,
                POISSON_INVOCATION_STACK,
            )
            from dendra.models.stim.waveform.core import _randomized_poisson

            if getattr(
                waveforms.get_submodule(parent), "_functional_poisson_recipe", False
            ) and field in (POISSON_INVOCATION_STACK, POISSON_INVOCATION_POSITION):
                continue  # Per-call operands are not authored model constants.
            if field == "_spike_times" and (
                type(waveforms.get_submodule(parent)) is _randomized_poisson
                or getattr(
                    waveforms.get_submodule(parent), "_functional_poisson_recipe", False
                )
            ):
                continue  # Runtime schedules live in explicit stimulation carry.
            yield f"injected_waveforms.{name}", value


def mechanism_injection_ephemeral_slots(owner):
    """Yield legacy current scratch slots that have no declared state contract."""
    declared = set(owner._assigned) | set(owner._carry)
    seen = set()
    for spec in getattr(owner, "_injection_specs", ()):
        name = spec["current_name"]
        if name not in declared and name not in seen and name in owner._buffers:
            seen.add(name)
            yield name, owner._buffers[name]


def mechanism_injection_helper(owner, method):
    """Recognize the framework pure helper during authored-hook inspection."""
    function = getattr(method, "__func__", None)
    if (
        getattr(method, "__self__", None) is owner
        and function is _CANONICAL_HELPERS["injection_values"]
        and function.__code__ is _CANONICAL_CODES["injection_values"]
        and "injection_values" not in vars(owner)
        and inspect.getattr_static(type(owner), "injection_values") is function
    ):
        return "injection_values"
    return None


def _metadata_failures(owner):
    specs = owner._injection_specs
    if type(specs) is not list:
        return ("Mechanism injection registrations require a plain list",)
    if not isinstance(owner.injected_waveforms, torch.nn.ModuleList):
        return ("Mechanism injected waveforms require a registered ModuleList",)
    if len(specs) != len(owner.injected_waveforms):
        return ("Mechanism injection registrations and waveforms disagree",)
    for spec in specs:
        if type(spec) is not dict or any(type(name) is not str for name in spec):
            return ("Mechanism injection registration requires string-keyed metadata",)
        if set(spec) != _SPEC_FIELDS:
            return ("Mechanism injection registration requires canonical metadata",)
        if any(
            type(spec[name]) is not str or not spec[name]
            for name in ("mask", "scale", "index", "current_name")
        ):
            return ("Mechanism injection metadata requires nonempty string names",)
        if type(spec["n_selected"]) is not int or spec["n_selected"] <= 0:
            return ("Mechanism injection requires a positive selected count",)
    return ()


def mechanism_injection_failures(owner):
    """Validate canonical registration helpers and explicit routing schemas."""
    if (
        type(owner._injection_specs) is list
        and not owner._injection_specs
        and type(owner.injected_waveforms) is torch.nn.ModuleList
        and not len(owner.injected_waveforms)
    ):
        return ()
    failures = list(_metadata_failures(owner))
    if failures:
        return tuple(failures)
    for name, canonical in _CANONICAL_HELPERS.items():
        if (
            inspect.getattr_static(type(owner), name) is not canonical
            or name in vars(owner)
            or canonical.__code__ is not _CANONICAL_CODES[name]
            or (canonical.__defaults__, tuple((canonical.__kwdefaults__ or {}).items()))
            != _CANONICAL_DEFAULTS[name]
        ):
            failures.append(
                "Mechanism injections require canonical shared current helpers"
            )
            break
    if (
        _injections.expand_injection_value is not _EXPAND
        or _EXPAND.__code__ is not _EXPAND_CODE
        or _CANONICAL_HELPERS["injection_values"].__globals__.get(
            "expand_injection_value"
        )
        is not _EXPAND
        or _EXPAND.__globals__.get("torch") is not torch
        or _CANONICAL_HELPERS["injection_values"].__globals__.get("torch") is not torch
        or _EXPAND.__globals__.get("math") is not _MATH
        or _MATH.prod is not _MATH_PROD
    ):
        failures.append(
            "Mechanism injections require canonical shared padding evaluation"
        )
    for k, spec in enumerate(owner._injection_specs):
        label = f"Mechanism injection {k}"
        if any(name not in owner._buffers for name in (spec["mask"], spec["index"])):
            failures.append(f"{label} requires registered routing buffers")
            continue
        mask = owner._buffers[spec["mask"]]
        index = owner._buffers[spec["index"]]
        scale = getattr(owner, spec["scale"], None)
        if (
            not torch.is_tensor(mask)
            or mask.dtype is not torch.bool
            or mask.layout is not torch.strided
            or mask.requires_grad
            or tuple(mask.shape) not in (tuple(owner.diam.shape), tuple(owner.shape_f))
        ):
            failures.append(
                f"{label} requires a boolean mask shaped like its local geometry"
            )
            continue
        if (
            not torch.is_tensor(index)
            or index.dtype is not torch.long
            or index.layout is not torch.strided
            or tuple(index.shape) != (spec["n_selected"],)
            or index.device != mask.device
            or index.requires_grad
            or not torch.equal(
                index, torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
            )
        ):
            failures.append(f"{label} selected index disagrees with its mask")
        if (
            not torch.is_tensor(scale)
            or scale.layout is not torch.strided
            or not torch.is_floating_point(scale)
            or scale.device != mask.device
            or spec["scale"] not in owner._buffers
            and spec["scale"] not in owner._parameters
        ):
            failures.append(f"{label} requires a registered floating-point scale")
        if not isinstance(owner.injected_waveforms[k], Waveform):
            failures.append(f"{label} requires a Waveform")
    return tuple(failures)


def _waveform_behavior(waveform, *, check_purity=False):
    from ._population import (
        _initialization_callable_contract_signature,
        _transform_structure_signature,
    )

    return (
        _waveform_layout_signature(waveform, check_purity=check_purity),
        _transform_structure_signature(waveform, strict=False) if check_purity else (),
        tuple(
            (
                path,
                name,
                _initialization_callable_contract_signature(
                    getattr(module, name),
                    label=f"injected_waveform.{path}.{name}",
                    owner=module,
                    strict=False,
                    diagnostic=check_purity,
                ),
            )
            for path, module in waveform.named_modules()
            for name in ("forward", "fn", "_compute")
            if callable(getattr(module, name, None))
        ),
    )


def _waveform_bindings(waveform, *, check_purity=False):
    """Expose code/global identities without inspection or tensor extraction."""
    from ._population import _module_autograd_hook_signature

    if not check_purity:
        return (
            id(waveform),
            _waveform_layout_signature(waveform, check_purity=False),
            tuple(
                (
                    path,
                    id(module),
                    tuple(
                        (
                            name,
                            _binding(getattr(type(module), name, None)),
                            _binding(vars(module).get(name)),
                        )
                        for name in (
                            *_WAVEFORM_MODULE_DISPATCH,
                            "forward",
                            "fn",
                            "_compute",
                        )
                    ),
                    _module_autograd_hook_signature(module),
                )
                for path, module in waveform.named_modules()
            ),
        )

    def routine(value, seen):
        target = getattr(value, "__func__", value)
        code = getattr(target, "__code__", None)
        identity = id(target)
        if code is None or identity in seen:
            return _binding(target)
        seen = seen | {identity}
        globals_ = target.__globals__
        return (
            _binding(target),
            tuple(
                (name, dependency(globals_[name], seen))
                for name in code.co_names
                if name in globals_
            ),
            tuple(
                dependency(cell.cell_contents, seen)
                for cell in (target.__closure__ or ())
            ),
        )

    def dependency(value, seen):
        if hasattr(value, "__code__") or isinstance(value, MethodType):
            return routine(value, seen)
        if type(value) in (str, bool, int, float, type(None), tuple, list):
            return _literal(value)
        return id(value), id(type(value))

    def member(value):
        if isinstance(value, property):
            return "property", tuple(
                None if function is None else routine(function, set())
                for function in (value.fget, value.fset, value.fdel)
            )
        if hasattr(getattr(value, "__func__", value), "__code__"):
            return routine(getattr(value, "__func__", value), set())
        return _literal(value)

    return (
        id(waveform),
        _waveform_layout_signature(waveform, check_purity=check_purity),
        tuple(
            (
                path,
                id(module),
                tuple(
                    (
                        name,
                        _binding(getattr(type(module), name)),
                        _binding(vars(module).get(name)),
                    )
                    for name in _WAVEFORM_MODULE_DISPATCH
                ),
                _module_autograd_hook_signature(module),
                tuple(
                    (
                        name,
                        tuple(
                            (key, id(hook))
                            for key, hook in (
                                getattr(value, "_backward_hooks", None) or {}
                            ).items()
                        ),
                    )
                    for name, value in module.named_parameters(
                        recurse=False, remove_duplicate=False
                    )
                ),
                tuple(
                    (
                        cls.__module__,
                        cls.__qualname__,
                        tuple(
                            (name, member(value))
                            for name, value in vars(cls).items()
                            if not name.startswith("__")
                        ),
                    )
                    for cls in type(module).__mro__
                    if cls is not object and not cls.__module__.startswith("torch.")
                ),
            )
            for path, module in waveform.named_modules()
        ),
    )


def _validate_waveform_dependencies(waveform):
    """Reuse the pure-hook audit for externally captured numerical recipes."""
    from ._population import (
        _direct_receiver_attribute_names,
        _initialization_dependency_value_signature,
    )

    seen = set()
    missing = object()

    def visit(module, value, label):
        receiver = True
        if isinstance(value, property):
            value = value.fget
        elif isinstance(value, staticmethod):
            value = value.__func__
            receiver = False
        elif isinstance(value, classmethod):
            value = value.__func__
        function = getattr(value, "__func__", value)
        if not inspect.isfunction(function):
            return
        key = id(module), id(function), receiver
        if key in seen:
            return
        seen.add(key)
        closure = inspect.getclosurevars(function)
        dependencies = (
            ("global", closure.globals),
            ("nonlocal", closure.nonlocals),
            (
                "default",
                {
                    "positional": function.__defaults__,
                    **(function.__kwdefaults__ or {}),
                },
            ),
        )
        for scope, values in dependencies:
            for name, dependency in values.items():
                if inspect.isfunction(dependency) and dependency.__module__.startswith(
                    "dendra."
                ):
                    # Shared numerical leaves include custom AD wrappers. Their
                    # code/global bindings remain frozen by this plan.
                    continue
                _initialization_dependency_value_signature(
                    dependency,
                    label=f"{label}.{scope}.{name}",
                    owner=module,
                    strict=True,
                    seen=set(),
                )
        receiver_name = (
            function.__code__.co_varnames[0]
            if receiver and function.__code__.co_argcount
            else None
        )
        _names, escapes = _direct_receiver_attribute_names(function, receiver_name)
        if escapes and not function.__module__.startswith("dendra."):
            raise FunctionalizationError(
                f"Waveform helper {label!r} purity cannot be proven because of opaque receiver dependencies {list(escapes)}; use direct registered attributes"
            )
        instructions = tuple(dis.get_instructions(function))
        for position, instruction in enumerate(instructions):
            if instruction.opname != "LOAD_FAST" or instruction.argval != receiver_name:
                continue
            current = module
            cursor = position + 1
            path = []
            while cursor < len(instructions):
                following = instructions[cursor]
                indexed = (
                    type(current)
                    in (
                        torch.nn.ModuleList,
                        torch.nn.ModuleDict,
                        torch.nn.Sequential,
                    )
                    and following.opname == "LOAD_CONST"
                    and cursor + 1 < len(instructions)
                    and instructions[cursor + 1].opname == "BINARY_SUBSCR"
                )
                if indexed:
                    index = following.argval
                    if type(current) is torch.nn.ModuleDict and type(index) is str:
                        child = current._modules.get(index, missing)
                    elif (
                        type(current)
                        in (
                            torch.nn.ModuleList,
                            torch.nn.Sequential,
                        )
                        and type(index) is int
                    ):
                        children = tuple(current._modules.values())
                        child = (
                            children[index]
                            if -len(children) <= index < len(children)
                            else missing
                        )
                    else:
                        child = missing
                    if child is missing:
                        raise FunctionalizationError(
                            f"Waveform helper {label!r} purity cannot be proven for registered child index {index!r}"
                        )
                    current = child
                    name = repr(index)
                    path.append(f"[{name}]")
                    cursor += 2
                elif following.opname in {"LOAD_ATTR", "LOAD_METHOD"}:
                    name = following.argval
                    path.append(f".{name}")
                    cursor += 1
                    if name not in current._modules:
                        if name in current._parameters or name in current._buffers:
                            break
                        from ._torch_parametrizations import property_signature

                        if (
                            property_signature(current, name, label=label, strict=True)
                            is not None
                        ):
                            # Canonical getters dispatch registered pure transforms.
                            break
                        target = inspect.getattr_static(current, name, missing)
                        if target is not missing:
                            visit(current, target, label + "".join(path))
                        break
                    current = current._modules[name]
                else:
                    break
                if current is None:
                    break
                if cursor < len(instructions) and (
                    instructions[cursor].opname in {"LOAD_ATTR", "LOAD_METHOD"}
                    or (
                        type(current)
                        in (
                            torch.nn.ModuleList,
                            torch.nn.ModuleDict,
                            torch.nn.Sequential,
                        )
                        and instructions[cursor].opname == "LOAD_CONST"
                        and cursor + 1 < len(instructions)
                        and instructions[cursor + 1].opname == "BINARY_SUBSCR"
                    )
                ):
                    continue
                direct_call = following.opname == "LOAD_METHOD" or (
                    sys.version_info >= (3, 12)
                    and following.opname == "LOAD_ATTR"
                    and bool(following.arg & 1)
                )
                if indexed:
                    # An indexed module call has a null call sentinel before
                    # its receiver. A module passed to another helper or stored
                    # in a local has no such sentinel and remains opaque.
                    direct_call = (
                        position > 0
                        and instructions[position - 1].opname == "PUSH_NULL"
                        and cursor < len(instructions)
                        and instructions[cursor].opname
                        in {
                            "LOAD_FAST",
                            "LOAD_CONST",
                            "LOAD_DEREF",
                            "LOAD_GLOBAL",
                            "PUSH_NULL",
                            "KW_NAMES",
                            "PRECALL",
                            "CALL",
                        }
                    )
                if not direct_call and not function.__module__.startswith("dendra."):
                    raise FunctionalizationError(
                        f"Waveform helper {label!r} purity cannot be proven when registered child module {name!r} escapes a direct call; call its helper directly"
                    )
                if type(current) in (
                    torch.nn.ModuleList,
                    torch.nn.ModuleDict,
                    torch.nn.Sequential,
                ):
                    for child_name, child in current.named_children():
                        visit(
                            child,
                            inspect.getattr_static(child, "forward", missing),
                            f"{label}{''.join(path)}.{child_name}.forward",
                        )
                else:
                    visit(
                        current,
                        inspect.getattr_static(current, "forward", missing),
                        f"{label}{''.join(path)}.forward",
                    )
                break

    visit(
        waveform,
        inspect.getattr_static(waveform, "forward", missing),
        "injected_waveform.forward",
    )


def _validate_waveform_class_configuration(waveform):
    """Keep class configuration auditable, with numerical leaves registered."""

    def reject_hidden_numerical_state(value, label):
        if isinstance(value, (np.ndarray, torch.Generator)):
            raise FunctionalizationError(
                f"Waveform class state {label!r} contains hidden mutable numerical "
                "state; use registered tensor parameters/buffers and explicit RNG."
            )
        if isinstance(value, set):
            raise FunctionalizationError(
                f"Waveform class state {label!r} contains a mutable set; use "
                "immutable tuple/frozenset configuration."
            )
        if isinstance(value, dict):
            for key, item in value.items():
                reject_hidden_numerical_state(key, f"{label}.key")
                reject_hidden_numerical_state(item, f"{label}[{key!r}]")
        elif isinstance(value, (tuple, list, set, frozenset)):
            for index, item in enumerate(value):
                reject_hidden_numerical_state(item, f"{label}[{index}]")

    for path, module in waveform.named_modules():
        for cls in type(module).__mro__:
            if cls is object or cls.__module__.startswith("torch."):
                continue
            for name, value in vars(cls).items():
                if (
                    name.startswith("__")
                    or name.startswith("_params")
                    or callable(value)
                    or isinstance(value, (staticmethod, classmethod, property))
                    or inspect.isdatadescriptor(value)
                ):
                    continue
                label = f"injected_waveform.{path}.{cls.__qualname__}.{name}"
                reject_hidden_numerical_state(value, label)
                _strict_state_signature(value, label=label)


def _tensor_alias_signature(module, owners):
    if not owners:
        return ()
    from ._population import _module_tensor_slots

    aliases = {}
    for path, value in _module_tensor_slots(module):
        aliases.setdefault(id(value), []).append(path)
    return tuple(
        (path, name, tuple(aliases.get(id(value), ())))
        for path, owner in owners
        for name, value in (
            *mechanism_injection_constant_slots(owner),
            *(
                (f"injected_waveforms.{name}", value)
                for name, value in owner.injected_waveforms.named_parameters(
                    remove_duplicate=False
                )
            ),
        )
    )


def network_mechanism_injection_alias_signature(network):
    """Include waveform and registration aliases crossing population boundaries."""
    owners = tuple(
        (f"populations.{name}.{path}", owner)
        for name, population in network.populations.items()
        for path, owner in mechanism_injection_owners(population)
    )
    return _tensor_alias_signature(network, owners)


def _population_ownership(population, *, validate):
    injections = getattr(population, "injections", ())
    pending = getattr(population, "mechanism_injections", ())
    accepted = getattr(population, "mechanism_injection_accepted", ())
    owners = mechanism_injection_owners(population)
    if (
        not owners
        and type(accepted) is list
        and all(type(flag) is bool and not flag for flag in accepted)
    ):
        # Solver-owned layouts are validated by their existing stimulus plan.
        return ()
    if (
        type(injections) is not list
        or type(pending) is not list
        or type(accepted) is not list
    ):
        raise FunctionalizationError(
            "Mechanism injection ownership requires canonical registration lists"
        )
    if len(pending) != len(accepted) or len(injections) != len(pending):
        raise FunctionalizationError(
            "Mechanism injection ownership registrations disagree; lower again"
        )
    if any(type(flag) is not bool for flag in accepted):
        raise FunctionalizationError(
            "Mechanism injection ownership requires boolean acceptance flags"
        )
    records = []
    available = [
        (path, owner, k, waveform)
        for path, owner in owners
        for k, waveform in enumerate(owner.injected_waveforms)
    ]
    consumed = set()
    module_paths = {id(module): path for path, module in population.named_modules()}

    for i, (registration, offered, claimed) in enumerate(
        zip(injections, pending, accepted)
    ):
        if (
            type(registration) is not tuple
            or len(registration) != 3
            or type(offered) is not tuple
            or len(offered) != 3
        ):
            raise FunctionalizationError(
                "Mechanism injection ownership requires canonical waveform registrations"
            )
        waveform, shape, index = registration
        if (
            waveform is not offered[0]
            or tuple(shape) != tuple(offered[1])
            or _index(index, contents=validate) != _index(offered[2], contents=validate)
        ):
            raise FunctionalizationError(
                "Mechanism injection solver and ownership registrations disagree"
            )
        chosen = None
        if claimed:
            for slot, (path, owner, k, candidate) in enumerate(available):
                if slot in consumed or candidate is not waveform:
                    continue
                if validate:
                    full_mask = torch.zeros_like(population.v, dtype=torch.bool)
                    full_mask[index] = True
                    local_mask = owner.get(full_mask)
                    registered_mask = getattr(owner, owner._injection_specs[k]["mask"])
                    try:
                        registered_mask = registered_mask.expand_as(local_mask)
                    except RuntimeError:
                        continue
                    if not torch.equal(local_mask, registered_mask):
                        continue
                chosen = (path, k)
                consumed.add(slot)
                break
            if chosen is None:
                raise FunctionalizationError(
                    "Accepted mechanism injection has no matching registered waveform owner"
                )
        records.append(
            (
                module_paths.get(id(waveform), ("solver-waveform", i)),
                tuple(shape),
                _index(index, contents=validate),
                claimed,
                chosen,
                None if validate else id(waveform),
            )
        )
    # Direct registrations are allowed; they are not solver-owned population specs.
    population_waveforms = {
        id(registration[0])
        for registration, claimed in zip(injections, accepted)
        if not claimed
    }
    claimed_waveforms = {
        id(registration[0])
        for registration, claimed in zip(injections, accepted)
        if claimed
    }
    if any(
        slot not in consumed
        and id(waveform) in population_waveforms - claimed_waveforms
        for slot, (_path, _owner, _k, waveform) in enumerate(available)
    ):
        raise FunctionalizationError(
            "Mechanism injection has stale or duplicate ownership of a population waveform"
        )
    return tuple(records)


def _contract(population, *, check_purity=False):
    records = []
    owners = mechanism_injection_owners(population)
    for path, owner in owners:
        failures = mechanism_injection_failures(owner)
        if failures:
            raise FunctionalizationError("; ".join(failures))
        for waveform in owner.injected_waveforms:
            _validate_waveform_contract(
                waveform, "mechanism injection", check_purity=check_purity
            )
            if check_purity:
                _validate_waveform_class_configuration(waveform)
        records.append(
            (
                path,
                tuple(
                    tuple((name, _literal(value)) for name, value in spec.items())
                    for spec in owner._injection_specs
                ),
                tuple(
                    (name, _layout(value))
                    for name, value in mechanism_injection_constant_slots(owner)
                ),
                tuple(
                    _waveform_behavior(waveform, check_purity=check_purity)
                    for waveform in owner.injected_waveforms
                ),
            )
        )
    return (
        tuple(records),
        _population_ownership(population, validate=True),
        _tensor_alias_signature(population, owners),
    )


class MechanismInjectionPlan:
    """Freeze ownership and routing, with optional authored purity diagnostics."""

    def __init__(self, population):
        self.population = population
        self.check_purity = current_check_purity()
        self.contract = _contract(population, check_purity=self.check_purity)
        self.constant_entries = tuple(
            (path, name, value.detach().clone())
            for path, owner in mechanism_injection_owners(population)
            for name, value in mechanism_injection_constant_slots(owner)
            if name
            in {
                spec[field]
                for spec in owner._injection_specs
                for field in ("mask", "index")
            }
        )
        self._source_index_entries = tuple(
            (attribute, position, path, value.detach().clone())
            for attribute in (
                ("injections", "mechanism_injections")
                if mechanism_injection_owners(population)
                else ()
            )
            for position, (_waveform, _shape, index) in enumerate(
                getattr(population, attribute)
            )
            for path, value in _index_tensors(index)
        )
        audited = set()
        from ._population import (
            _authored_workspace_classes,
            _class_state_snapshot,
            _restore_class_state,
        )

        for _path, owner in (
            mechanism_injection_owners(population) if self.check_purity else ()
        ):
            for waveform in owner.injected_waveforms:
                if id(waveform) in audited:
                    continue
                audited.add(id(waveform))
                _validate_waveform_contract(
                    waveform, "mechanism injection", check_purity=self.check_purity
                )
                class_state = _class_state_snapshot(
                    _authored_workspace_classes(waveform)
                )
                try:
                    try:
                        clone = copy.deepcopy(waveform)
                    except Exception as exc:
                        raise FunctionalizationError(
                            "Mechanism injection Waveform could not be isolated by deepcopy"
                        ) from exc
                    self._configure_waveform(clone)
                    _validate_waveform_dependencies(clone)
                    _audit_waveform_purity(
                        clone,
                        label="mechanism injection",
                        device=owner.diam.device,
                        dtype=owner.diam.dtype,
                    )
                finally:
                    changed = _restore_class_state(class_state)
                    if changed:
                        raise FunctionalizationError(
                            "Mechanism injection Waveform mutated shared class state "
                            f"{sorted(changed)} during its functional purity audit"
                        )
        self._source_bindings = self.source_binding_signature()

    @staticmethod
    def _configure_waveform(waveform):
        from ._poisson_stimulation import configure_execution

        configure_execution(waveform)
        for module in waveform.modules():
            if isinstance(module, cacheable) and not isinstance(module, Waveform):
                module.clear_cache()
                module.training = True

    def configure_execution(self, population):
        """Keep explicit leaves authoritative in isolated execution copies."""
        configure_mechanism_injection_execution(population)

    def validate_population(self, population, *, source=False):
        if _contract(population, check_purity=self.check_purity) != self.contract:
            raise FunctionalizationError(
                "Mechanism injection ownership, layout, waveform behavior, or aliases changed; lower again"
            )
        if (
            source
            and self.source_binding_signature(population) != self._source_bindings
        ):
            raise FunctionalizationError(
                "Mechanism injection helper bindings changed; lower again"
            )
        modules = dict(population.named_modules())
        for path, name, expected in self.constant_entries:
            if expected is not None and not torch.equal(
                getattr(modules[path], name), expected
            ):
                raise FunctionalizationError(
                    "Mechanism injection routing changed; lower again"
                )

    def validate_source(self):
        self.validate_population(self.population, source=True)

    def source_binding_signature(self, population=None):
        population = self.population if population is None else population
        owners = mechanism_injection_owners(population)
        return (
            tuple(
                (
                    path,
                    tuple(
                        tuple(
                            (_literal(name), _literal(value))
                            for name, value in spec.items()
                        )
                        for spec in owner._injection_specs
                    )
                    if not _metadata_failures(owner)
                    else ("invalid", _literal(owner._injection_specs)),
                    tuple(
                        (
                            name,
                            _binding(getattr(type(owner), name)),
                            _binding(getattr(owner, name)),
                        )
                        for name in _HELPER_NAMES
                    ),
                    tuple(
                        (name, _layout(value))
                        for name, value in mechanism_injection_constant_slots(owner)
                    )
                    if not _metadata_failures(owner)
                    else (),
                    tuple(
                        _waveform_bindings(waveform, check_purity=self.check_purity)
                        for waveform in owner.injected_waveforms
                    ),
                )
                for path, owner in owners
            ),
            _population_ownership(population, validate=False),
            _tensor_alias_signature(population, owners),
            _binding(_injections.expand_injection_value),
            _binding(
                _CANONICAL_HELPERS["injection_values"].__globals__.get(
                    "expand_injection_value"
                )
            ),
            id(_EXPAND.__globals__.get("torch")),
            id(_CANONICAL_HELPERS["injection_values"].__globals__.get("torch")),
            id(_EXPAND.__globals__.get("math")),
            id(_MATH.prod),
        )

    def validate_source_tensor_values(self):
        modules = dict(self.population.named_modules())
        for path, name, expected in self.constant_entries:
            if expected is not None:
                torch._check_tensor_all(
                    getattr(modules[path], name) == expected,
                    "Mechanism injection routing changed; lower again",
                )
        for attribute, position, path, expected in self._source_index_entries:
            value = getattr(self.population, attribute)[position][2]
            for index in path:
                value = value[index]
            torch._check_tensor_all(
                value == expected,
                "Mechanism injection population index changed; lower again",
            )

    def validate_constant_values(self, values):
        for path, name, expected in self.constant_entries:
            if expected is not None:
                torch._check_tensor_all(
                    values[f"{path}.{name}" if path else name] == expected,
                    "Mechanism injection routing must preserve its registered mask and index",
                )


__all__ = []
