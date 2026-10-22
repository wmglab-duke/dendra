"""Static TABLE contracts over the shared, registered interpolation modules."""

from __future__ import annotations

import hashlib
import inspect
from types import CodeType, MethodType

import torch

from dendra.models import parametric
from dendra.models.parametric import Parameterized
from dendra.utils import PreparedInterp1d

from ._types import FunctionalizationError

_TABLE_TEMPLATE = parametric.table_function_template
_INSTANTIATE_TABLES = inspect.getattr_static(Parameterized, "instantiate_tables")
_INSTANTIATE_CODE = _INSTANTIATE_TABLES.__code__
_INTERPOLATOR_METHODS = {
    name: value
    for name, value in vars(PreparedInterp1d).items()
    if inspect.isfunction(value)
}
_INTERPOLATOR_METHODS.update(
    (name, inspect.getattr_static(PreparedInterp1d, name))
    for name in ("__call__", "_wrapped_call_impl", "_call_impl")
)
_INTERPOLATOR_CODES = {
    name: method.__code__ for name, method in _INTERPOLATOR_METHODS.items()
}


def _owner_paths(population, owners=None):
    from ._population import _parameterized_owner_paths

    selected = None if owners is None else {id(owner) for owner in owners}
    return tuple(
        (path, owner)
        for path, owner in _parameterized_owner_paths(population)
        if selected is None or id(owner) in selected
    )


def table_owners(population, owners=None):
    """Return each declared TABLE in canonical Parameterized owner order."""
    return tuple(
        (path, owner, name, owner._modules.get(f"{name}_table"))
        for path, owner in _owner_paths(population, owners)
        for name in owner.__class__._table
    )


def table_buffer_slots(owner):
    """Yield registered interpolation constants relative to one TABLE owner."""
    for name in owner.__class__._table:
        table = owner._modules.get(f"{name}_table")
        if table is None:
            raise FunctionalizationError(f"TABLE {name!r} is not instantiated")
        for buffer_name, value in table._buffers.items():
            if value is not None:
                yield f"{name}_table.{buffer_name}", value


def table_parameter_names(population, owners=None):
    """Return canonical learnable TABLE ordinate names, without shape padding."""
    identities = {
        id(parameter)
        for _path, _owner, _name, table in table_owners(population, owners)
        if table is not None
        for parameter in table._parameters.values()
        if parameter is not None
    }
    return frozenset(
        name
        for name, parameter in population.named_parameters()
        if id(parameter) in identities
    )


def canonical_table_wrapper(owner, name, method=None):
    """Recognize exactly the generated TABLE dispatch wrapper, without running it."""
    if name not in owner.__class__._table:
        return False
    method = getattr(owner, name, None) if method is None else method
    if getattr(method, "__self__", None) is not owner:
        return False
    function = getattr(method, "__func__", None)
    if function is None:
        return False
    source = _TABLE_TEMPLATE.format(func_name=name)
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    filename = f"<dendra.parametric.{name}_with_table:{digest}>"
    if (
        getattr(function, "__source__", None) != source
        or getattr(function, "__generated_filename__", None) != filename
        or function.__code__.co_filename != filename
        or set(vars(function)) != {"__source__", "__generated_filename__"}
    ):
        return False
    expected = next(
        item
        for item in compile(source, filename, "exec", dont_inherit=True).co_consts
        if isinstance(item, CodeType)
    )
    return function.__code__ == expected


def table_wrapper_original(owner, method):
    """Resolve a canonical generated dispatcher to its audited authored hook.

    The initialization dependency walker can use this to traverse user code
    without treating the registered interpolation Module as an opaque callable.
    Its tensor leaves are independently routed through TABLE constants and
    parameters, and :class:`TablePlan` guards its dispatch and layout.
    """
    function = getattr(method, "__func__", None)
    if function is None:
        return None
    for name in owner.__class__._table:
        wrapper = getattr(owner, name, None)
        if getattr(wrapper, "__func__", None) is function and canonical_table_wrapper(
            owner, name, method
        ):
            return name, getattr(owner, f"{name}_original")
    return None


def _tensor_layout(value):
    return tuple(value.shape), value.dtype, value.device, value.layout


def _literal_binding(value):
    # Do not invoke equality/truthiness on an opaque replacement while checking
    # a source captured by Dynamo. Admission rejects such metadata separately.
    if type(value) in (bool, int, float, str):
        return value
    return "nonliteral", id(type(value)), id(value)


def _interpolator_signature(table):
    return (
        _literal_binding(table.batched),
        _literal_binding(table.D),
        _literal_binding(table.N),
        _literal_binding(table.outside),
        _literal_binding(table.fill_value),
        _literal_binding(table.learnable_y),
        _literal_binding(table.exact_clamp),
        _literal_binding(table._use_uniform),
        _literal_binding(table._ind_hi),
        tuple((name, _tensor_layout(value)) for name, value in table._buffers.items()),
        tuple(
            (name, _tensor_layout(value)) for name, value in table._parameters.items()
        ),
    )


def _tensor_alias_signature(population, tables):
    """Freeze registered TABLE routing while allowing numerical replacements."""
    if not tables:
        return ()
    from ._population import _module_tensor_slots

    aliases = {}
    for path, value in _module_tensor_slots(population):
        aliases.setdefault(id(value), []).append(path)
    return tuple(
        (
            path,
            name,
            kind,
            tensor_name,
            tuple(aliases.get(id(value), ())),
        )
        for path, _owner, name, table in tables
        for kind, slots in (
            ("parameter", table._parameters),
            ("buffer", table._buffers),
        )
        for tensor_name, value in slots.items()
    )


def network_table_alias_signature(network):
    """Include TABLE tensor aliases crossing Population or connection owners."""
    tables = tuple(
        (
            f"populations.{population_name}{'.' + path if path else ''}",
            owner,
            name,
            table,
        )
        for population_name, population in network.populations.items()
        for path, owner, name, table in table_owners(population)
    )
    return _tensor_alias_signature(network, tables)


def table_failures(owner):
    """Reject noncanonical dispatch/workspaces before their numerical purity audit."""
    declarations = owner.__class__._table
    if not declarations:
        return ()
    failures = []
    if (
        "instantiate_tables" in vars(owner)
        or inspect.getattr_static(type(owner), "instantiate_tables")
        is not _INSTANTIATE_TABLES
        or _INSTANTIATE_TABLES.__code__ is not _INSTANTIATE_CODE
        or parametric.table_function_template != _TABLE_TEMPLATE
    ):
        failures.append("TABLE requires canonical table instantiation")
    for name, declaration in declarations.items():
        label = f"TABLE {type(owner).__qualname__}.{name}"
        if not canonical_table_wrapper(owner, name):
            failures.append(
                f"{label} requires its canonical generated dispatch wrapper"
            )
        original = getattr(owner, f"{name}_original", None)
        if not callable(original) or (
            inspect.ismethod(original)
            and original.__self__ is not owner
            and original.__self__ is not type(owner)
        ):
            failures.append(f"{label} requires its original authored function")
        flag = getattr(owner, f"usetable_{name}", None)
        if type(flag) is not bool:
            failures.append(f"{label} requires a boolean usetable flag")
        table = owner._modules.get(f"{name}_table")
        if type(table) is not PreparedInterp1d:
            failures.append(f"{label} requires a canonical PreparedInterp1d module")
            continue
        if any(
            hook in vars(table)
            or inspect.getattr_static(type(table), hook) is not canonical
            or canonical.__code__ is not _INTERPOLATOR_CODES[hook]
            or canonical.__globals__.get("torch") is not torch
            for hook, canonical in _INTERPOLATOR_METHODS.items()
        ):
            failures.append(f"{label} has a replaced interpolation implementation")
        if getattr(table, "_compiled_call_impl", None) is not None:
            failures.append(f"{label} has a custom compiled interpolation adapter")
        if any(
            getattr(table, attribute)
            for attribute in (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        ):
            failures.append(f"{label} interpolation hooks are currently unsupported")
        literal_metadata = {
            "batched": bool,
            "D": int,
            "N": int,
            "outside": str,
            "fill_value": float,
            "learnable_y": bool,
            "exact_clamp": bool,
            "_use_uniform": bool,
            "_ind_hi": int,
        }
        if any(
            type(getattr(table, field, None)) is not expected
            for field, expected in literal_metadata.items()
        ):
            failures.append(f"{label} requires literal interpolation metadata")
            continue
        if table.outside not in ("clamp", "zero", "fill"):
            failures.append(f"{label} has an unsupported outside policy")
        learnable = declaration.get("learnable", False)
        if (
            table.batched
            or table.D != 1
            or table.N != declaration["n"]
            or table.learnable_y != learnable
            or not table._use_uniform
            or table._ind_hi != table.N - 2
        ):
            failures.append(
                f"{label} interpolation metadata differs from its declaration"
            )
        expected_buffers = {
            "_x_search",
            "_x0",
            "_x_min",
            "_x_max",
            "_inv_dx_uniform",
            "_dx_uniform",
            "_inv_dx",
        }
        if not learnable:
            expected_buffers.update(("_y", "_y0", "_slopes", "_y_first", "_y_last"))
        if (
            set(table._buffers) != expected_buffers
            or set(table._parameters) != ({"y"} if learnable else set())
            or table._modules
            or any(
                not torch.is_tensor(value)
                for value in (*table._buffers.values(), *table._parameters.values())
            )
        ):
            failures.append(f"{label} has a noncanonical registered tensor schema")
            continue
        scalar = {
            "_x_min",
            "_x_max",
            "_inv_dx_uniform",
            "_dx_uniform",
            "_y_first",
            "_y_last",
        }
        full = {"_x_search", "_y", "y"}
        if any(
            tuple(value.shape)
            != (() if name in scalar else (table.N if name in full else table.N - 1,))
            or value.dtype != table._x_search.dtype
            or value.device != table._x_search.device
            or value.layout is not torch.strided
            for name, value in (*table._buffers.items(), *table._parameters.items())
        ):
            failures.append(f"{label} has a noncanonical table layout")
    return tuple(failures)


def _contract(population):
    from ._population import _initialization_callable_dependency_signature

    aliases = {}
    records = []
    tables = table_owners(population)
    for path, owner, name, table in tables:
        failures = table_failures(owner)
        if failures:
            raise FunctionalizationError("; ".join(failures))
        records.append(
            (
                path,
                name,
                tuple(
                    (_literal_binding(field), _literal_binding(value))
                    for field, value in owner.__class__._table[name].items()
                ),
                _literal_binding(getattr(owner, f"usetable_{name}")),
                aliases.setdefault(id(table), len(aliases)),
                _interpolator_signature(table),
                (
                    None
                    if getattr(owner, f"usetable_{name}")
                    else _initialization_callable_dependency_signature(
                        getattr(owner, f"{name}_original"),
                        label=f"TABLE {path}.{name}_original",
                        owner=owner,
                        strict=True,
                        include_identity=False,
                    )
                ),
            )
        )
    return tuple(records), _tensor_alias_signature(population, tables)


class TablePlan:
    """Frozen TABLE dispatch/layout contract; ordinates remain explicit leaves."""

    def __init__(self, population):
        self.population = population
        self.contract = _contract(population)
        self._source_bindings = self._bindings(population)

    @staticmethod
    def _bindings(population):
        return tuple(
            (
                path,
                name,
                getattr(owner, name).__func__,
                getattr(
                    getattr(owner, f"{name}_original"),
                    "__func__",
                    getattr(owner, f"{name}_original"),
                ),
            )
            for path, owner, name, _table in table_owners(population)
        )

    def validate_population(self, population, *, source=False):
        if _contract(population) != self.contract:
            raise FunctionalizationError(
                "TABLE declarations, flags, interpolation layout, or authored helpers "
                "changed; lower the Population again"
            )
        if source and self._bindings(population) != self._source_bindings:
            raise FunctionalizationError("TABLE callable bindings changed; lower again")

    def validate_source(self):
        self.validate_population(self.population, source=True)

    def source_binding_signature(self, population=None):
        """Expose source attributes directly to Dynamo's ordinary guards.

        The complete admission audit remains a Python-only boundary check.
        These literal values and callable identities must also be read while
        tracing, so a captured graph cannot reuse stale TABLE dispatch after
        the source configuration changes.
        """
        population = self.population if population is None else population

        def callable_binding(value):
            function = value.__func__ if isinstance(value, MethodType) else value
            return id(function), id(getattr(function, "__code__", None))

        tables = table_owners(population)
        records = tuple(
            (
                path,
                name,
                tuple(
                    (_literal_binding(field), _literal_binding(value))
                    for field, value in owner.__class__._table[name].items()
                ),
                _literal_binding(getattr(owner, f"usetable_{name}")),
                id(table),
                _interpolator_signature(table),
                callable_binding(getattr(owner, name)),
                callable_binding(getattr(owner, f"{name}_original")),
                tuple(
                    (
                        method,
                        callable_binding(getattr(type(table), method)),
                        callable_binding(vars(table).get(method)),
                    )
                    for method in _INTERPOLATOR_METHODS
                ),
                callable_binding(getattr(table, "_compiled_call_impl", None)),
                tuple(
                    (
                        attribute,
                        tuple(
                            (key, id(hook))
                            for key, hook in getattr(table, attribute).items()
                        ),
                    )
                    for attribute in (
                        "_forward_hooks",
                        "_forward_pre_hooks",
                        "_backward_hooks",
                        "_backward_pre_hooks",
                    )
                ),
            )
            for path, owner, name, table in tables
        )
        return records, _tensor_alias_signature(population, tables)
