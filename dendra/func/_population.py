"""Experimental, fail-closed functional Population lowering."""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.func import functional_call

from dendra.models.core import Population, Unmyelinated
from dendra.models.integrators.implicit import _bwd_euler_ub
from dendra.models.mechanisms._mechanism import Mechanism, PointProcess
from dendra.models.mechanisms._support import SupportKind
from dendra.models.parametric import PositiveParam, cacheable

from ._types import (
    FunctionalizationError,
    PopulationTensors,
    RolloutInput,
    StepInput,
)

_POPULATION_PARAMETER_NAMES = {
    "celsius_param",
    "cm_param",
    "rhoa_param",
    "rhoa_scale_param.rho",
    "cm_scale_param.rho",
    "area_scale_param.rho",
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


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.preserve_format)


def _clone_execution_population(population: Unmyelinated) -> Unmyelinated:
    """Copy a model without retaining live autograd scratch or buffer aliases."""
    memo = {}
    for _name, value in population.named_buffers(remove_duplicate=False):
        memo.setdefault(id(value), _clone_tensor(value))
    handler = population.mech
    for values in (handler._buf_i, handler._buf_g):
        for value in values:
            if torch.is_tensor(value):
                memo.setdefault(id(value), _clone_tensor(value))
    for value in getattr(population.integrator, "_last_bands", ()) or ():
        if torch.is_tensor(value):
            memo.setdefault(id(value), _clone_tensor(value))
    cloned = copy.deepcopy(population, memo)
    cloned.integrator.clear_jit_cache()
    # Imperative execution installs conditional Dynamo barriers around shared
    # ion/material synchronization.  This private program is explicitly driven
    # by an outer functional_call/torch.compile boundary, so retain the original
    # class methods instead of those per-instance wrappers.
    for method_name in cloned.mech._SYNC_METHODS:
        cloned.mech.__dict__.pop(method_name, None)
    return cloned


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


def _tensor_binding(values: Mapping[str, torch.Tensor]) -> tuple:
    """Bind a derived workspace to the exact tensor versions that created it."""
    return tuple(
        (name, id(values[name]), values[name]._version) for name in sorted(values)
    )


class _PreparedPopulation:
    """Opaque plan-bound tensor bundle returned by :meth:`prepare`."""

    __slots__ = (
        "token",
        "parameter_binding",
        "constant_binding",
        "constant_sources",
        "values",
    )

    def __init__(
        self,
        token,
        parameter_binding,
        constant_binding,
        constant_sources,
        values,
    ):
        self.token = token
        self.parameter_binding = parameter_binding
        self.constant_binding = constant_binding
        self.constant_sources = constant_sources
        self.values = values


def _positive_parameter_value(module: PositiveParam, rho: torch.Tensor) -> torch.Tensor:
    """Evaluate the standard lower-bounded Population scale parameter."""
    if (
        module.min_val != 0.0
        or module.max_val is not None
        or module.lower_mode != "softplus"
    ):
        raise FunctionalizationError(
            "Functional Unmyelinated geometry currently supports only the "
            "standard lower-bounded PositiveParam used by Population scales."
        )
    return F.softplus(rho, beta=module.beta, threshold=module.threshold)


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
class _PreparedBuffer:
    key: str
    module_path: str
    buffer_name: str
    shape: tuple[int, ...]
    kind: str


def _derived_buffer_names(module) -> tuple[str, ...]:
    """Return the statically declared initialization-derived buffer names."""
    return tuple(sorted(getattr(module, "_derived_buffers", ())))


def _derived_handler_keys(handler) -> set[str]:
    """Return checkpoint keys reconstructed from prepared workspace tensors."""
    keys = set()
    for mechanism_name, mechanism in handler.mechanisms.items():
        keys.update(
            f"{mechanism_name}.{buffer_name}"
            for buffer_name in _derived_buffer_names(mechanism)
        )
        for state_name, state in mechanism.DE.items():
            keys.update(
                f"{mechanism_name}.DE.{state_name}.{buffer_name}"
                for buffer_name in _derived_buffer_names(state)
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
        for buffer_name in sorted(
            set(mechanism._assigned) - set(_derived_buffer_names(mechanism))
        ):
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
            for buffer_name in sorted(
                set(state._state_buffers) - set(_derived_buffer_names(state))
            ):
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
        for buffer_name in sorted(process._assigned):
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
            for buffer_name in sorted(state._state_buffers):
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
    leaves = [
        _StateLeaf(
            public_path=("integrator", "v"),
            module_path="",
            buffer_name="v",
            mapping_slots=(("", "v"),),
            checkpoint_key=None,
            shape=_shape_tuple(population.v),
            dtype=population.v.dtype,
            device=population.v.device,
        )
    ]
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
    for module_path in _parameterized_module_paths(population):
        module = _module_at(population, module_path)
        for name in _declared_parameter_names(module):
            if name not in module._buffers or not hasattr(module, f"{name}_param"):
                continue
            effective.append(
                _PreparedBuffer(
                    key=f"{module_path}.{name}",
                    module_path=module_path,
                    buffer_name=name,
                    shape=_shape_tuple(module._buffers[name]),
                    kind="effective_parameter",
                )
            )
        if not hasattr(module, "DE") and getattr(module, "has_q10", False):
            if "q10_cache" in module._buffers:
                effective.append(
                    _PreparedBuffer(
                        key=f"{module_path}.q10_cache",
                        module_path=module_path,
                        buffer_name="q10_cache",
                        shape=_shape_tuple(module._buffers["q10_cache"]),
                        kind="q10",
                    )
                )
        for name in _derived_buffer_names(module):
            derived.append(
                _PreparedBuffer(
                    key=f"{module_path}.{name}",
                    module_path=module_path,
                    buffer_name=name,
                    shape=_shape_tuple(module._buffers[name]),
                    kind="derived",
                )
            )
    return tuple((*effective, *derived))


def _geometry_buffer_layout(population: Population) -> tuple[tuple[str, str], ...]:
    """Map registered runtime geometry slots to explicit geometry constants."""
    values = [("population.diam", "diam"), ("population.dx", "dx")]
    for module_path in _parameterized_module_paths(population):
        module = _module_at(population, module_path)
        # Mechanisms and their nested State objects receive a registered local
        # diameter view when the handler is built. Dense support means that
        # view is the whole Population geometry, so it must follow the explicit
        # constant rather than the execution clone used to build this plan.
        if "diam" in module._buffers:
            values.append((f"population.{module_path}.diam", "diam"))
    return tuple(values)


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
    for name, value in values.items():
        if not torch.is_tensor(value):
            raise FunctionalizationError(
                f"{type(module).__qualname__}.derive_buffers()[{name!r}] "
                "must be a Tensor"
            )
    return values


class _PopulationPreparation(torch.nn.Module):
    """Pure materialization of effective parameters and static workspaces."""

    def __init__(self, population: Unmyelinated):
        super().__init__()
        self.population = population
        self.layout = _prepared_buffer_layout(population)
        self.module_paths = _parameterized_module_paths(population)
        self.derived_module_paths = tuple(
            module_path
            for module_path in self.module_paths
            if _derived_buffer_names(_module_at(population, module_path))
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
            for index, entry in enumerate(self.layout):
                if entry.kind == "derived":
                    offset_names.append(None)
                    continue
                owner = _module_at(population, entry.module_path)
                if entry.kind == "q10":
                    evaluated = owner.calc_q10()
                else:
                    evaluated = _resolve_parameter_value(
                        owner,
                        f"{entry.buffer_name}_param",
                    )
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
                owner = _module_at(population, module_path)
                values = _evaluate_derived_buffers(owner)
                for name in _derived_buffer_names(owner):
                    value = values[name]
                    live = owner._buffers[name]
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
        self.offset_names = tuple(offset_names)

    def forward(self, celsius, diam):
        handler = self.population.mech
        handler._sync_celsius(celsius)
        # The supported slice is dense, so every local geometry view is the
        # explicit Population diameter tensor. Rebind it before derived builders
        # run so geometry substitutions and their gradients cannot fall back to
        # the extraction clone.
        for module_path in self.module_paths:
            owner = _module_at(self.population, module_path)
            if "diam" in owner._buffers:
                owner._buffers["diam"] = diam

        outputs = []
        q10_entries = []
        for index, entry in enumerate(self.layout):
            if entry.kind == "derived":
                continue
            owner = _module_at(self.population, entry.module_path)
            if entry.kind == "q10":
                q10_entries.append((index, entry, owner))
                continue
            value = _resolve_parameter_value(owner, f"{entry.buffer_name}_param")
            value = torch.broadcast_to(value, entry.shape)
            offset_name = self.offset_names[index]
            if offset_name is not None:
                value = value + getattr(self, offset_name)
            owner._buffers[entry.buffer_name] = value
            outputs.append((entry.key, value))

        for index, entry, owner in q10_entries:
            value = torch.broadcast_to(owner.calc_q10(), entry.shape)
            offset_name = self.offset_names[index]
            if offset_name is not None:
                value = value + getattr(self, offset_name)
            owner._buffers[entry.buffer_name] = value
            outputs.append((entry.key, value))

        for module_path in self.derived_module_paths:
            owner = _module_at(self.population, module_path)
            values = _evaluate_derived_buffers(owner)
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

        by_key = dict(outputs)
        return tuple(by_key[entry.key] for entry in self.layout)


class _UnmyelinatedTransition(torch.nn.Module):
    """Functional facade over the established deterministic step implementation."""

    def __init__(self, population: Unmyelinated, solver, layout):
        super().__init__()
        self.population = population
        self.solver = solver
        self.layout = layout
        self.dt_value = float(population.integrator.dt)
        object.__setattr__(
            self,
            "_state_owners",
            tuple(_module_at(population, entry.module_path) for entry in layout),
        )

    def forward(self, v, t, duration_remainder, dt, celsius, ve, intra, steps: int):
        handler = self.population.mech
        handler._sync_celsius(celsius)
        handler.read_from_ions()
        handler.read_from_materials()

        integrator = self.population.integrator
        for index in range(steps):
            ve_step = None if ve is None else ve[index]
            intra_step = None if intra is None else intra[index]
            v, _i_membrane = integrator._functional_step_reference(
                v,
                dt,
                celsius,
                ve_step,
                intra_step,
                solver=self.solver,
            )
            # PyTorch's scalar-plus-constant vmap rule selects lane zero and
            # therefore rejects an empty lane batch. Materializing the fixed
            # timestep through the batched scalar preserves the same arithmetic
            # while making the zero-lane boundary well-defined.
            t = t + torch.full_like(t, self.dt_value)

        outputs = []
        for entry, owner in zip(self.layout, self._state_owners, strict=True):
            if entry.public_path == ("integrator", "v"):
                value = v
            elif entry.public_path == ("clock", "t"):
                value = t
            elif entry.public_path == ("control", "duration_remainder"):
                value = duration_remainder
            else:
                value = owner._buffers[entry.buffer_name]
            outputs.append(value)
        return tuple(outputs)


def _select_transformable_solver(integrator):
    solver = integrator._solve
    solver_name = getattr(solver, "__name__", None)
    solver_module = getattr(solver, "__module__", None)

    if solver_module == "torch._ops.dendra_solvers":
        try:
            from dendra_solvers import pcr_solve_t, thomas_solve_t
        except ImportError as exc:  # pragma: no cover - guarded by selected op
            raise FunctionalizationError(
                "The selected native CPU solver requires a dendra-solvers build "
                "that exports torch.func-compatible tridiagonal facades."
            ) from exc
        if solver_name == "thomas_solve_t":
            return thomas_solve_t
        if solver_name == "pcr_solve_t":
            return pcr_solve_t

    if solver_name == "pcr_solve_t" and solver_module == (
        "dendra.models.integrators.tridiag.pcr"
    ):
        return solver

    raise FunctionalizationError(
        "The selected bwd_euler_ub solver is not yet transform-compatible. "
        "Use method='thomas' with current dendra-solvers or method='pcr'."
    )


def _uses_standard_unmyelinated_geometry(population) -> bool:
    for name in ("area", "edge_resistance_ohm"):
        if inspect.getattr_static(type(population), name) is not inspect.getattr_static(
            Unmyelinated, name
        ):
            return False
    return True


def _parameterized_owners(population):
    yield population
    for module_path in _parameterized_module_paths(population):
        yield _module_at(population, module_path)


def _functionalization_failures(population, dt: float) -> list[str]:
    failures = []
    if not isinstance(population, Population):
        return ["expected an initialized dendra Population"]
    if not population.initialized:
        failures.append("Population must be initialized before functionalization")
    if not isinstance(population, Unmyelinated):
        failures.append("the current supported topology is Unmyelinated")
    elif not _uses_standard_unmyelinated_geometry(population):
        failures.append(
            "subclasses overriding Unmyelinated area or axial geometry are not "
            "supported; exactly Unmyelinated geometry semantics are required"
        )
    if population.device().type != "cpu":
        failures.append("the current supported device is CPU")
    if population.dtype() not in (torch.float32, torch.float64):
        failures.append("dtype must be float32 or float64")
    if not isinstance(population.integrator, _bwd_euler_ub):
        failures.append("the current supported integrator is bwd_euler_ub")
    elif population.integrator.imem:
        failures.append("i_membrane recording is not supported yet")
    elif population.integrator.use_gc_variant:
        failures.append("the gradient-clipped Thomas variant is not supported yet")

    handler = getattr(population, "mech", None)
    mechanisms = () if handler is None else tuple(handler.mechanisms.values())
    if not mechanisms:
        failures.append("at least one membrane mechanism must be inserted")
    point_processes = tuple(
        mechanism for mechanism in mechanisms if isinstance(mechanism, PointProcess)
    )
    if point_processes:
        failures.append(
            "PointProcesses have area-dependent current-scaling workspaces "
            "that are not lowered yet"
        )

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
        if inspect.getattr_static(
            type(mechanism), "set_dt"
        ) is not inspect.getattr_static(Mechanism, "set_dt"):
            failures.append(
                f"{mechanism.name} overrides set_dt with a timestep-dependent "
                "workspace that is not lowered yet"
            )
        support_map = getattr(mechanism, "support_map", None)
        if support_map is None or support_map.spec.kind is not SupportKind.DENSE:
            failures.append(f"{mechanism.name} must use dense whole-Population support")
        if mechanism._delayed_state_specs:
            failures.append(
                f"{mechanism.name} uses delayed state, which is not supported yet"
            )
        if mechanism._injection_specs:
            failures.append(f"{mechanism.name} uses a registered waveform injection")
        for module in (mechanism, *mechanism.DE.values()):
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
    if population.injections or population.mechanism_injections:
        failures.append(
            "registered injections are not lowered yet; pass explicit intra"
        )

    for owner in _parameterized_owners(population):
        if getattr(owner, "additional_parameters", None):
            failures.append("regional parameter overrides are not supported yet")
            break
        if getattr(owner, "in_graph_parametrizations", None):
            failures.append("custom in-graph parameterizations are not supported yet")
            break
        parametrizations = getattr(owner, "parametrizations", None)
        if parametrizations is not None and len(parametrizations):
            failures.append("torch parameterizations are not supported yet")
            break

    named_parameters = dict(population.named_parameters())
    missing_population_parameters = _POPULATION_PARAMETER_NAMES - set(named_parameters)
    if missing_population_parameters:
        failures.append(
            "required Unmyelinated parameters are missing: "
            f"{sorted(missing_population_parameters)}"
        )
    if any(parameter.ndim != 0 for parameter in named_parameters.values()):
        failures.append("the current slice requires scalar raw parameters")

    if not math.isfinite(dt) or dt <= 0.0:
        failures.append("dt must be a finite positive scalar")
    return failures


def _support_signature(module):
    support_map = getattr(module, "support_map", None)
    return None if support_map is None else support_map.spec.ordered_signature


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

    def module_signature(name, module):
        return (
            name,
            type(module).__module__,
            type(module).__qualname__,
            _support_signature(module),
            _flag_signature(module),
            tuple(
                (
                    state_name,
                    type(state).__module__,
                    type(state).__qualname__,
                    tuple(state._state),
                    tuple(state._state_buffers),
                    tuple(_derived_buffer_names(state)),
                    _flag_signature(state),
                    state.method,
                    _configuration_signature(state.method_kwargs),
                    bool(state.include_q10_in_comp_graph),
                )
                for state_name, state in module.DE.items()
            ),
            tuple(sorted(module._assigned)),
            tuple(_derived_buffer_names(module)),
            tuple(sorted(module._save)),
        )

    return (
        (type(population).__module__, type(population).__qualname__),
        tuple(population.shape),
        population.device().type,
        population.dtype(),
        (
            type(population.integrator).__module__,
            type(population.integrator).__qualname__,
            getattr(population.integrator, "method", None),
            bool(population.integrator.imem),
            bool(getattr(population.integrator, "use_gc_variant", False)),
            tuple(population.integrator.v_vars),
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


class FunctionalPopulation:
    """Immutable execution plan for deterministic scalar Unmyelinated models.

    Construct instances with :func:`make_functional`. Existing Population
    execution is not redirected through this reference path.
    """

    def __init__(self, population: Unmyelinated, *, dt: float):
        self.shape = tuple(population.shape)
        self.dt = float(dt)
        self.dtype = population.dtype()
        self.device = population.device()
        self._population = population
        self._source_ids = _structure_ids(population)
        self._fingerprint = _structure_signature(population)
        self._prepared_token = object()

        execution_population = _clone_execution_population(population)
        dt_tensor = torch.as_tensor(self.dt, device=self.device, dtype=self.dtype)
        execution_population.integrator._initialize(
            execution_population,
            dt_tensor,
            force=execution_population.force_integrator_reinit(),
            compile_scope="population",
        )
        solver = _select_transformable_solver(execution_population.integrator)
        self._state_layout = _state_layout(execution_population)
        self._transition = _UnmyelinatedTransition(
            execution_population,
            solver,
            self._state_layout,
        )
        self._base_mapping = {
            name: _clone_tensor(value)
            for name, value in _module_tensor_slots(self._transition)
        }
        self._geometry_buffers = _geometry_buffer_layout(execution_population)

        preparation_population = _clone_execution_population(execution_population)
        self._preparation = _PopulationPreparation(preparation_population)
        self._prepared_buffers = self._preparation.layout
        self._preparation_base_mapping = {
            name: _clone_tensor(value)
            for name, value in _module_tensor_slots(self._preparation)
        }

        self._parameter_names = tuple(dict(population.named_parameters()))
        self._parameter_shapes = {
            name: _shape_tuple(parameter)
            for name, parameter in population.named_parameters()
        }
        self._state_schema = self._make_state_schema(self._state_layout)

    @staticmethod
    def _make_state_schema(layout):
        schema = {}
        for leaf in layout:
            _put_path(schema, leaf.public_path, leaf)
        return schema

    def _validate_source(self) -> None:
        if _structure_ids(self._population) != self._source_ids or (
            _structure_signature(self._population) != self._fingerprint
        ):
            raise FunctionalizationError(
                "The source Population structure changed after make_functional(); "
                "lower it again after insert(), batch(), rebuild(), .to(), or dtype changes."
            )

    def _validate_compatible_population(self, population) -> None:
        if not isinstance(population, Population):
            raise TypeError("population must be a dendra Population")
        if _structure_signature(population) != self._fingerprint:
            raise FunctionalizationError(
                "Population structure does not match this functional execution plan."
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
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError(
                    f"parameter {name!r} must use {self.device}/{self.dtype}"
                )

    def _validate_constants(self, constants) -> None:
        if not isinstance(constants, Mapping):
            raise TypeError("constants must be a mapping of named tensors")
        if set(constants) != {"diam", "dx"}:
            raise KeyError("constants must contain exactly 'diam' and 'dx'")
        for name in ("diam", "dx"):
            value = constants[name]
            if not torch.is_tensor(value) or _shape_tuple(value) != self.shape:
                raise ValueError(f"constant {name!r} must have shape {self.shape}")
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError(
                    f"constant {name!r} must use {self.device}/{self.dtype}"
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
        """Extract independent constants/state and explicit raw parameters."""
        population = self._population if population is None else population
        self._validate_compatible_population(population)
        state = {}
        for leaf in _state_layout(population):
            value = _buffer_at(population, leaf.module_path, leaf.buffer_name)
            _put_path(state, leaf.public_path, _clone_tensor(value))
        return PopulationTensors(
            parameters=dict(population.named_parameters()),
            constants={
                "diam": _clone_tensor(population.diam),
                "dx": _clone_tensor(population.dx),
            },
            state=state,
        )

    def _prepare_mechanism_buffers(self, parameters, celsius, diam):
        mapping = dict(self._preparation_base_mapping)
        for name, value in parameters.items():
            mapping[f"population.{name}"] = value
        outputs = functional_call(
            self._preparation,
            mapping,
            (celsius, diam),
            tie_weights=False,
        )
        return {
            entry.key: value
            for entry, value in zip(self._prepared_buffers, outputs, strict=True)
        }

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

        values = self._prepare_values(parameters, constants)
        return _PreparedPopulation(
            self._prepared_token,
            _tensor_binding(parameters),
            _tensor_binding(constants),
            dict(constants),
            values,
        )

    def _prepare_values(self, parameters, constants):
        shape = self.shape
        diam = constants["diam"]
        dx = constants["dx"]
        dt = diam.new_tensor(self.dt)
        dt_s = dt * 1.0e-3

        celsius = parameters["celsius_param"]
        cm = parameters["cm_param"].expand(shape)
        rhoa = parameters["rhoa_param"].expand(shape)
        rhoa_scale = _positive_parameter_value(
            self._population.rhoa_scale_param,
            parameters["rhoa_scale_param.rho"],
        )
        cm_scale = _positive_parameter_value(
            self._population.cm_scale_param,
            parameters["cm_scale_param.rho"],
        )
        area_scale = _positive_parameter_value(
            self._population.area_scale_param,
            parameters["area_scale_param.rho"],
        )

        area = diam * 1.0e-4 * torch.pi * dx * 1.0e-4
        area_scaled = area * area_scale
        capacitance = 1.0e-6 * cm * cm_scale * area_scaled
        cm_inv = capacitance.reciprocal()

        radius_cm = 1.0e-4 * diam / 2.0
        dx_cm = 1.0e-4 * dx
        segment_resistance = rhoa * rhoa_scale * dx_cm / (torch.pi * radius_cm.square())
        edge_conductance = 2.0 / (
            segment_resistance[..., :-1] + segment_resistance[..., 1:]
        )
        g_left = edge_conductance / capacitance[..., :-1]
        g_right = edge_conductance / capacitance[..., 1:]
        zeros = torch.zeros_like(capacitance[..., :1])
        diag_base = torch.cat((-g_left, zeros), dim=-1) + torch.cat(
            (zeros, -g_right),
            dim=-1,
        )

        return {
            "geometry": {
                "diam": diam,
                "dx": dx,
            },
            "population": {
                "celsius": celsius,
                "cm": cm,
                "rhoa": rhoa,
                "rhoa_scale": rhoa_scale,
                "cm_scale": cm_scale,
                "area_scale": area_scale,
            },
            "mechanisms": self._prepare_mechanism_buffers(
                parameters,
                celsius,
                diam,
            ),
            "integrator": {
                "dt": dt,
                "diag_base": diag_base,
                "lower": -dt_s * g_right,
                "upper": -dt_s * g_left,
                "g_edge_Cinv": g_left,
                "g_edge_Cinv_right": g_right,
                "cm_inv": cm_inv,
                "scale": area_scaled * cm_inv,
            },
        }

    def _validate_prepared(self, prepared, parameters) -> Mapping:
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "Opaque prepared plans cannot cross torch.compile graph "
                "boundaries; compile prepare_and_step() or "
                "prepare_and_rollout() instead"
            )
        if not isinstance(prepared, _PreparedPopulation):
            raise TypeError("prepared must be the opaque value returned by prepare()")
        if prepared.token is not self._prepared_token:
            raise FunctionalizationError(
                "prepared belongs to a different functional Population plan"
            )
        if prepared.parameter_binding != _tensor_binding(parameters):
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

        values = prepared.values
        if not isinstance(values, Mapping) or set(values) != {
            "geometry",
            "population",
            "mechanisms",
            "integrator",
        }:
            raise FunctionalizationError("prepared tensor schema is invalid")
        if not isinstance(values["geometry"], Mapping) or set(values["geometry"]) != {
            "diam",
            "dx",
        }:
            raise FunctionalizationError("prepared geometry tensor schema is invalid")
        expected_population = {
            "celsius",
            "cm",
            "rhoa",
            "rhoa_scale",
            "cm_scale",
            "area_scale",
        }
        if (
            not isinstance(values["population"], Mapping)
            or set(values["population"]) != expected_population
        ):
            raise FunctionalizationError("prepared population tensor schema is invalid")
        expected_mechanisms = {entry.key for entry in self._prepared_buffers}
        if (
            not isinstance(values["mechanisms"], Mapping)
            or set(values["mechanisms"]) != expected_mechanisms
        ):
            raise FunctionalizationError("prepared mechanism tensor schema is invalid")
        expected_integrator = {
            "dt",
            "diag_base",
            "lower",
            "upper",
            "g_edge_Cinv",
            "g_edge_Cinv_right",
            "cm_inv",
            "scale",
        }
        if (
            not isinstance(values["integrator"], Mapping)
            or set(values["integrator"]) != expected_integrator
        ):
            raise FunctionalizationError("prepared integrator tensor schema is invalid")

        edge_shape = (*self.shape[:-1], self.shape[-1] - 1)
        tensor_shapes = {
            ("geometry", "diam"): self.shape,
            ("geometry", "dx"): self.shape,
            ("population", "celsius"): (),
            ("population", "cm"): self.shape,
            ("population", "rhoa"): self.shape,
            ("population", "rhoa_scale"): (),
            ("population", "cm_scale"): (),
            ("population", "area_scale"): (),
            ("integrator", "dt"): (),
            ("integrator", "diag_base"): self.shape,
            ("integrator", "lower"): edge_shape,
            ("integrator", "upper"): edge_shape,
            ("integrator", "g_edge_Cinv"): edge_shape,
            ("integrator", "g_edge_Cinv_right"): edge_shape,
            ("integrator", "cm_inv"): self.shape,
            ("integrator", "scale"): self.shape,
        }
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
        for entry in self._prepared_buffers:
            value = values["mechanisms"][entry.key]
            if not torch.is_tensor(value) or _shape_tuple(value) != entry.shape:
                raise FunctionalizationError(
                    f"prepared mechanism {entry.key} must have shape {entry.shape}"
                )
            if value.device != self.device or value.dtype != self.dtype:
                raise FunctionalizationError(
                    f"prepared mechanism {entry.key} must use {self.device}/{self.dtype}"
                )
        return values

    def _mapping(self, parameters, prepared, state):
        mapping = dict(self._base_mapping)
        prefix = "population."
        for name, value in parameters.items():
            mapping[f"{prefix}{name}"] = value

        for name, value in prepared["population"].items():
            mapping[f"{prefix}{name}"] = value
        for slot, constant_name in self._geometry_buffers:
            mapping[slot] = prepared["geometry"][constant_name]
        for entry in self._prepared_buffers:
            value = prepared["mechanisms"][entry.key]
            if entry.kind == "derived":
                # Derived workspaces are read-only transition inputs. A private
                # clone keeps authored accidental in-place writes from mutating
                # the caller's reusable prepared tensor or crossing vmap lanes.
                value = value.clone()
            mapping[f"{prefix}{entry.module_path}.{entry.buffer_name}"] = value

        for name in (
            "diag_base",
            "lower",
            "upper",
            "g_edge_Cinv",
            "g_edge_Cinv_right",
            "cm_inv",
            "scale",
        ):
            mapping[f"{prefix}integrator.{name}"] = prepared["integrator"][name]

        for leaf in self._state_layout:
            # Authored breakpoint/state code may legally update a declared
            # mutable buffer in place. Give the private execution skeleton one
            # differentiable working copy, shared by every registered alias, so
            # those updates become output carry without modifying (or
            # leaf-in-place invalidating) the caller's explicit state tensor.
            value = _get_path(state, leaf.public_path).clone()
            for module_path, buffer_name in leaf.mapping_slots:
                path = f"{module_path}." if module_path else ""
                key = f"{prefix}{path}{buffer_name}"
                # The duration carry deliberately is a plain host-side
                # attribute, not a registered module buffer. It is already an
                # explicit transition input.
                if key in self._base_mapping:
                    mapping[key] = value
        return mapping

    def _normalize_rollout_value(self, value, *, name: str, steps: int | None):
        if value is None:
            return None, steps
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a Tensor or None")
        expected_tail = self.shape
        if (
            value.ndim != len(expected_tail) + 1
            or tuple(value.shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"{name} must have time-first shape "
                f"(steps, {', '.join(map(str, expected_tail))})"
            )
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError(f"{name} must use {self.device}/{self.dtype}")
        value_steps = int(value.shape[0])
        if steps is not None and value_steps != steps:
            raise ValueError(
                f"{name} has {value_steps} steps, but the rollout expects {steps}"
            )
        return value, value_steps

    def _rollout_values(
        self,
        parameters,
        prepared,
        state,
        inputs: RolloutInput | None = None,
        *,
        steps: int | None = None,
    ):
        if inputs is None:
            inputs = RolloutInput()
        if not isinstance(inputs, RolloutInput):
            raise TypeError("inputs must be a RolloutInput or None")
        if steps is not None:
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise ValueError("steps must be a non-negative integer")

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

        mapping = self._mapping(parameters, prepared, state)
        outputs = functional_call(
            self._transition,
            mapping,
            (
                state["integrator"]["v"],
                state["clock"]["t"],
                state["control"]["duration_remainder"],
                prepared["integrator"]["dt"],
                prepared["population"]["celsius"],
                ve,
                intra,
                resolved_steps,
            ),
            tie_weights=False,
        )
        next_state = {}
        for leaf, value in zip(self._state_layout, outputs, strict=True):
            _put_path(next_state, leaf.public_path, value)
        return next_state, {"v": next_state["integrator"]["v"]}

    def rollout(
        self,
        parameters,
        prepared,
        state,
        inputs: RolloutInput | None = None,
        *,
        steps: int | None = None,
    ):
        """Run an eager/torch.func rollout from an opaque prepared plan."""
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
        )

    def prepare_and_rollout(
        self,
        parameters,
        constants,
        state,
        inputs: RolloutInput | None = None,
        *,
        steps: int | None = None,
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
        )

    def step(
        self,
        parameters,
        prepared,
        state,
        inputs: StepInput | None = None,
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
        )

    def prepare_and_step(
        self,
        parameters,
        constants,
        state,
        inputs: StepInput | None = None,
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
        population.restore_dict_from_checkpoint(checkpoint)
        return population


def make_functional(
    population: Population,
    *,
    dt: float,
) -> tuple[FunctionalPopulation, PopulationTensors]:
    """Lower an initialized deterministic Population into a pure transition."""
    if isinstance(dt, bool):
        raise TypeError("dt must be a finite positive scalar")
    try:
        dt_value = float(dt)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt must be a finite positive scalar") from exc
    failures = _functionalization_failures(population, dt_value)
    if failures:
        details = "\n".join(f"- {failure}" for failure in dict.fromkeys(failures))
        raise FunctionalizationError(
            "Population cannot be functionalized by the current capability slice:\n"
            f"{details}"
        )
    functional = FunctionalPopulation(population, dt=dt_value)
    return functional, functional.extract(population)


__all__ = ["FunctionalPopulation", "make_functional"]
