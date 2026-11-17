"""Find mechanism consumers of raw writes in a native initialization transform."""

import dis
import inspect
from collections import defaultdict
from contextvars import ContextVar

import torch

from dendra.models.parametric import (
    NegativeParam,
    Parameterized,
    PositiveParam,
    staticproperty,
)

# Only fresh native initialize() may use the graph. A direct hook call, state
# restoration, and the gradient/handoff rebuilds use the complete transaction.
_scoped_initialization_population = ContextVar(
    "scoped_initialization_population", default=None
)
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
_PARAMETER_METHODS = (
    "_derive_parameter_buffers",
    "_install_parameter_buffers",
    "make_contiguous",
    "populate_parameter_buffers",
)


def _storage_span(value):
    if (
        type(value) not in (torch.Tensor, torch.nn.Parameter)
        or value.layout != torch.strided
    ):
        raise TypeError("No exact storage span for this parameter dependency.")
    storage = value.untyped_storage()
    start, size = storage.data_ptr(), storage.nbytes()
    if size == 0:
        return None
    if not start:
        raise TypeError("No data address for this parameter dependency.")
    return str(value.device), start, start + size


def _overlapping_storage_groups(storage_groups):
    """Connect storage wrappers that address any of the same bytes."""
    groups_by_span = {}
    device, end = None, 0
    for span in sorted(storage_groups):
        next_device, start, stop = span
        if next_device != device or start >= end:
            groups = set()
            device, end = next_device, 0
        groups.update(storage_groups[span])
        groups_by_span[span] = groups
        end = max(end, stop)
    return groups_by_span


def _parameter_names(owner):
    return set().union(*(getattr(type(owner), name) for name in _PARAMETER_CATEGORIES))


def _has_plain_sources(owner, names):
    if (
        owner.in_graph_parametrizations
        or owner.additional_parameters
        or owner.runtime_noises
        or owner.random_parameters
        or torch.nn.utils.parametrize.is_parametrized(owner)
        or type(owner).__getattr__ is not torch.nn.Module.__getattr__
        or type(owner).__getattribute__ is not object.__getattribute__
        or any(
            isinstance(value, staticproperty) and name != "t"
            for cls in type(owner).__mro__
            for name, value in vars(cls).items()
        )
    ):
        return False
    for name in names:
        source_name = name + "_param"
        source = owner._parameters.get(source_name)
        if source is None:
            source = owner._modules.get(source_name)
        if type(source) not in (torch.nn.Parameter, PositiveParam, NegativeParam):
            return False
        if inspect.getattr_static(owner, source_name, None) is not None:
            return False
    return True


def _has_local_derived_inputs(owner, field_names):
    if not owner._derived_buffers:
        return True
    # Use the existing callable audit to reject hidden globals, closures, and
    # authored descriptors, then restrict receiver reads to the input frame.
    from dendra.func._population import (
        _direct_receiver_attribute_names,
        _initialization_callable_dependency_signature,
    )
    from dendra.func._types import FunctionalizationError

    builder = owner.derive_buffers
    routine = getattr(builder, "__func__", None)
    if routine is None or getattr(builder, "__self__", None) is not owner:
        return False
    receiver = routine.__code__.co_varnames[0]
    # The shared audit recognizes direct LOAD_FAST receiver accesses. Reject
    # fused receiver loads that newer Python versions may introduce.
    if any(
        instruction.opname.count("LOAD_FAST") > 1
        and isinstance(instruction.argval, tuple)
        and receiver in instruction.argval
        for instruction in dis.get_instructions(routine)
    ):
        return False
    reads, escapes = _direct_receiver_attribute_names(routine, receiver)
    if escapes or not reads <= field_names | {"celsius", "diam"}:
        return False
    if any(name not in owner._buffers for name in reads):
        return False
    try:
        _initialization_callable_dependency_signature(
            builder, label="derive_buffers", owner=owner, strict=True
        )
    except FunctionalizationError:
        return False
    return True


def _has_standard_ion_sources(handler):
    from dendra.models.mechanisms._ions import Ion

    # Custom initial-field sources can run an imperative callback between
    # parameter population and the post transforms.
    return not handler.materials and all(
        type(ion) is Ion
        and all(
            getattr(getattr(ion, name), "__func__", None) is getattr(Ion, name)
            for name in (
                "initialize",
                "advance",
                "_derive_initial_field_values",
                "_derive_advanced_field_updates",
            )
        )
        and all(
            type(getattr(ion, name))
            in (torch.nn.Parameter, PositiveParam, NegativeParam)
            for name in ("e_init", "i_init", "o_init")
        )
        for ion in handler.ions.values()
    )


def _initialization_parameter_dependents(population, mechanism_groups, targets):
    """Return ordered affected mechanisms or None for an incomplete graph.

    Nested State trees are refreshed as part of their owning mechanism. Inspect
    live storage rather than persisting a graph across model edits, conversions,
    or assignment of a new Parameter that shares the old storage.
    """
    from dendra.models.mechanisms._mechanism import Mechanism

    if (
        torch.nn.modules.module._global_forward_hooks
        or torch.nn.modules.module._global_forward_pre_hooks
        or not _has_standard_ion_sources(population.mech)
    ):
        return None
    module_groups = defaultdict(set)
    for group, mechanism in mechanism_groups:
        for module in mechanism.modules():
            module_groups[id(module)].add(group)
    # A population-level parameter wrapper can also be registered below a
    # mechanism. Preserve the full population refresh when that source changes.
    for name in _parameter_names(population):
        source = population._modules.get(name + "_param")
        if source is not None:
            for module in source.modules():
                module_groups[id(module)].add(None)
    storage_groups = defaultdict(set)
    if not _has_plain_sources(population, _parameter_names(population)):
        return None
    try:
        changed = {_storage_span(target) for target in targets}
        if None in changed:
            return None
        for module in population.modules():
            if module._forward_hooks or module._forward_pre_hooks:
                return None
            groups = module_groups.get(id(module), {None})
            for registry in (module._parameters, module._buffers):
                for value in registry.values():
                    if value is not None:
                        span = _storage_span(value)
                        if span is not None:
                            storage_groups[span].update(groups)
        storage_groups = _overlapping_storage_groups(storage_groups)
        affected = set()
        written_groups = set()
        for span in changed:
            groups = storage_groups.get(span, {None})
            if None in groups:
                return None
            written_groups.add(id(groups))
            affected.update(groups)
        for group, mechanism in mechanism_groups:
            if any(
                getattr(getattr(mechanism, name), "__func__", None)
                is not getattr(Mechanism, name)
                for name in ("_init_buffers_s", "_initialize_declared_values")
            ):
                return None
            for owner in (mechanism, *mechanism.DE.values()):
                if any(
                    getattr(getattr(owner, method), "__func__", None)
                    is not getattr(Parameterized, method)
                    for method in _PARAMETER_METHODS
                ):
                    return None
                fields = _parameter_names(owner)
                if not (
                    _has_plain_sources(owner, fields)
                    and _has_local_derived_inputs(owner, fields)
                ):
                    return None
                for name in fields:
                    value = owner._buffers.get(name)
                    if value is None:
                        return None
                    span = _storage_span(value)
                    if span is None:
                        continue
                    groups = storage_groups.get(span)
                    # Keep full install order for aliases of writable effective
                    # buffers, including writes through an aliased raw target.
                    if groups != {group} or id(groups) in written_groups:
                        return None
    except (TypeError, RuntimeError, NotImplementedError):
        return None

    return tuple(
        mechanism for group, mechanism in mechanism_groups if group in affected
    )
