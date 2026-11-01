"""Detached, source-aware publication of functional continuation to native models."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch.utils import _pytree as pytree

from dendra.models._checkpoint import _copy_state_containers
from dendra.models.core import Cable
from dendra.models.networks.net import Network

from ._continuation import Continuation
from ._continuation_checkpoints import _cold_boundary, _validate_frames
from ._handoff_staging import stage_native_handoff
from ._native_handoff_sources import (
    _NativeSourceGroup,
    _registered_owner,
    population_source_frames,
)
from ._native_network_sources import network_source_frames, scalar_source_writes
from ._network import FunctionalNetwork
from ._population import FunctionalPopulation, _rebind_population_time_references
from ._types import FunctionalizationError


def _tensor_paths(tree, prefix=()):
    if isinstance(tree, _NativeSourceGroup):
        for value in tree.tensors:
            yield prefix, value
    elif torch.is_tensor(tree):
        yield prefix, tree
    elif isinstance(tree, Mapping):
        for key, value in tree.items():
            yield from _tensor_paths(value, (*prefix, key))
    elif isinstance(tree, tuple):
        names = getattr(tree, "_fields", range(len(tree)))
        for key, value in zip(names, tree, strict=True):
            yield from _tensor_paths(value, (*prefix, key))


def _at(tree, path):
    for key in path:
        tree = (
            getattr(tree, key)
            if isinstance(tree, tuple) and isinstance(key, str)
            else tree[key]
        )
    return tree


def _equal(left, right):
    # Compare logical contents exactly. Numeric equality would hide signed
    # zeros or NaN payload changes that custom pure transforms can observe.
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    left = left.detach().resolve_conj().resolve_neg()
    right = right.detach().resolve_conj().resolve_neg()
    if left.is_complex():
        left, right = torch.view_as_real(left), torch.view_as_real(right)
    # Equal-sized integer views also handle noncontiguous and singleton
    # zero-stride operands without copying or changing their logical layout.
    bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        left.element_size()
    ]
    return torch.equal(left.view(bits), right.view(bits))


def _owned(tree):
    memo = {}

    def clone(value):
        if not torch.is_tensor(value):
            return value
        if id(value) not in memo:
            memo[id(value)] = value.detach().clone()
        return memo[id(value)]

    return pytree.tree_map(clone, tree)


def _source_frames(functional, target):
    reader = (
        network_source_frames
        if isinstance(functional, FunctionalNetwork)
        else population_source_frames
    )
    return reader(functional, target)


def _source_writes(frames, parameters, constants):
    """Own incoming values and qualify each distinct destination before writes."""
    result, coordinates, storages = {}, {}, {}
    for domain, references, values in (
        ("parameters", frames[0], parameters),
        ("constants", frames[1], constants),
    ):
        for path, destination in _tensor_paths(references):
            label = ".".join((domain, *map(str, path)))
            value = _at(values, path)
            if type(destination) not in (torch.Tensor, torch.nn.Parameter) or type(
                value
            ) not in (torch.Tensor, torch.nn.Parameter):
                raise FunctionalizationError(
                    f"Native handoff source {label!r} requires an ordinary Tensor or Parameter"
                )
            if (
                destination.layout is not torch.strided
                or torch.is_inference(destination)
                or torch._debug_has_internal_overlap(destination) != 0
            ):
                raise FunctionalizationError(
                    f"Native handoff source {label!r} has unsupported overlapping storage"
                )
            if destination.shape != value.shape:
                raise FunctionalizationError(
                    f"Native handoff source {label!r} has no matching native shape"
                )
            converted = value.detach().to(
                device=destination.device, dtype=destination.dtype
            )
            if not _equal(converted.to(device=value.device, dtype=value.dtype), value):
                raise FunctionalizationError(
                    f"Native handoff source {label!r} cannot be represented by its native dtype"
                )
            previous = result.get(id(destination))
            if previous is not None:
                if not _equal(previous[1], converted):
                    raise FunctionalizationError(
                        f"Native handoff source aliases disagree at {label!r}"
                    )
                continue
            if destination.numel():
                storage = (destination.device, destination.untyped_storage().data_ptr())
                coordinate = (
                    storage,
                    destination.dtype,
                    destination.storage_offset(),
                    tuple(destination.shape),
                    tuple(destination.stride()),
                    destination.is_conj(),
                    destination.is_neg(),
                )
                if storage in storages and storages[storage] != coordinate:
                    raise FunctionalizationError(
                        f"Native handoff source {label!r} shares unsupported native storage views"
                    )
                storages[storage] = coordinate
                if coordinate in coordinates:
                    if not _equal(coordinates[coordinate], converted):
                        raise FunctionalizationError(
                            f"Native handoff source aliases disagree at {label!r}"
                        )
                else:
                    coordinates[coordinate] = converted
            result[id(destination)] = (destination, converted.clone(), label)
    return tuple(result.values())


def _validate_readonly(functional, constants):
    populations = (
        (
            (plan, constants["populations"][name])
            for name, plan in functional.populations.items()
        )
        if isinstance(functional, FunctionalNetwork)
        else ((functional, constants),)
    )
    for plan, frame in populations:
        if not _equal(frame["dt"], frame["dt"].new_tensor(plan.dt)):
            raise FunctionalizationError(
                "Native handoff cannot change the initialized timestep"
            )
    if isinstance(functional, FunctionalNetwork) and functional.netstim is not None:
        for key in ("heap_times", "heap_row_offsets"):
            if not _equal(
                constants["netstim"][key], getattr(functional.netstim._source, key)
            ):
                raise FunctionalizationError(
                    "Native handoff cannot change a frozen NetStim heap calendar"
                )


def _refresh_population(population, dt, *, compile_scope="population"):
    # This is the existing continuation refresh, performed eagerly so inference
    # also sees the transferred sources. It never runs INITIAL or draws RAND.
    population._deterministically_rematerialize_initialization_parameters(
        preserve_state=True
    )
    population.integrator.clear_jit_cache()
    population.integrator.initialized = False
    population.integrator._initialize(
        population,
        torch.tensor(dt, device=population.device(), dtype=population.dtype()),
        force=True,
        compile_scope=compile_scope,
    )


def _connection_draws(source_plan, role, constants):
    if not source_plan._draw_schemas[role]:
        return {}
    if not bool(constants[f"draw_valid.{role}"]):
        return None
    fresh = source_plan._alternate_draws[role] and bool(
        constants[f"use_fresh_draw.{role}"]
    )
    prefix = "fresh_draw" if fresh else "draw"
    return {
        str(index): constants[f"{prefix}.{role}.{index}"]
        for index in source_plan._draw_schemas[role]
    }


def _refresh_native(functional, target, parameters, constants):
    if not isinstance(functional, FunctionalNetwork):
        _refresh_population(target, functional.dt)
        return
    for name, plan in functional.populations.items():
        _refresh_population(
            target.populations[name], plan.dt, compile_scope="network_population"
        )
    for (
        collection,
        name,
    ), source_plan in functional._connection_parameter_plans.items():
        connection = getattr(target, collection)[name]
        frame = constants[collection][name]
        expected = source_plan.materialize(parameters[collection][name], frame)
        for role, attribute in (("weights", "weight"), ("delays", "delay_ms")):
            materializer = getattr(connection, attribute)
            if source_plan._raw[role]:
                draws = _connection_draws(source_plan, role, frame)
                materializer.set_extra_state(draws)
                actual = materializer._derive_detached_values()
                if not _equal(actual, expected[role]):
                    raise FunctionalizationError(
                        f"Native handoff cannot represent retained {collection}.{name}.{role} overrides"
                    )
                materializer.w = actual.detach().clone()
            elif not _equal(materializer.w, expected[role]):
                raise FunctionalizationError(
                    f"Native handoff has no writable raw source for {collection}.{name}.{role}"
                )
        connection._validate_detached_parameter_values(
            connection.weight.w, connection.delay_ms.w
        )


def _compare_preparation(functional, target, parameters, constants):
    """Reject synthetic overrides which cannot survive native rematerialization."""
    actual = functional.extract(target)
    populations = (
        ((("populations", name), plan) for name, plan in functional.populations.items())
        if isinstance(functional, FunctionalNetwork)
        else (((), functional),)
    )
    # Randomized bank schedules are runtime history, not raw sources. Their
    # parameter operands are replaced by invocation packets during execution;
    # restored native buffers instead hold the continuation's latest schedule.
    runtime_schedules = {
        (*prefix, entry.parameter)
        for prefix, plan in populations
        if plan._poisson_plan is not None
        for entry in plan._poisson_plan.entries
        if entry.kind != "mechanism"
    }
    # Parameters without an owned raw source (external extra, opaque samples,
    # and synthetic scalar sources) must still agree after staged restoration.
    for path, value in _tensor_paths(parameters):
        if path in runtime_schedules:
            continue
        if not _equal(value, _at(actual.parameters, path)):
            raise FunctionalizationError(
                f"Native handoff has no equivalent source for parameters.{'.'.join(map(str, path))}"
            )
    expected_values = functional._prepare_values(parameters, constants)
    actual_values = functional._prepare_values(actual.parameters, actual.constants)
    for path, value in _tensor_paths(expected_values):
        if not _equal(value, _at(actual_values, path)):
            raise FunctionalizationError(
                f"Native handoff cannot represent prepared override {'.'.join(map(str, path))!r}"
            )


@contextmanager
def _preserve_native_bindings(target):
    """Rollback publication failures before any raw source copy takes place."""
    modules = tuple(target.modules())
    namespaces = tuple(
        (module, type(module), _copy_state_containers(vars(module)))
        for module in modules
    )
    generators, seen = {}, set()

    def collect(value):
        if isinstance(value, torch.Generator):
            generators.setdefault(id(value), (value, value.get_state().clone()))
        elif isinstance(value, (Mapping, list, tuple)) and id(value) not in seen:
            seen.add(id(value))
            for item in value.values() if isinstance(value, Mapping) else value:
                collect(item)

    for module in modules:
        collect(vars(module))
    try:
        yield
    except BaseException:
        for module, cls, namespace in namespaces:
            object.__setattr__(module, "__class__", cls)
            module.__dict__.clear()
            module.__dict__.update(namespace)
        for generator, state in generators.values():
            generator.set_state(state)
        raise


def _canonical_geometry_buffer(owner, name):
    return (
        issubclass(type(owner), Cable)
        and vars(owner).get("_compartment_graph") is not None
        and name in Cable._CANONICAL_GEOMETRY_TENSORS
    )


def _publication_buffers(target, staged, protected):
    result, memo = [], {}
    originals = dict(target.named_modules())
    for path, module in staged.named_modules():
        owner = originals.get(path)
        for name, value in module._buffers.items():
            previous = None if owner is None else owner._buffers.get(name)
            if not torch.is_tensor(value) or id(previous) in protected:
                continue
            unchanged = (
                torch.is_tensor(previous)
                and (
                    not previous.requires_grad
                    or _canonical_geometry_buffer(owner, name)
                )
                and previous.shape == value.shape
                and previous.dtype == value.dtype
                and previous.device == value.device
                and _equal(previous, value)
            )
            if id(value) not in memo:
                memo[id(value)] = value.detach().clone()
            result.append((path, owner, name, memo[id(value)], unchanged))
    return result


def _publish_runtime(target, staged, buffers, source_bindings, scalar_writes):
    # Some native inference restores donate runtime storage. Isolate it first
    # so a later error can restore the original tensors and version counters.
    protected = {id(value) for _path, _registry, _name, value in source_bindings}
    original_buffers = tuple(
        (module, name, value)
        for module in target.modules()
        for name, value in module._buffers.items()
        if torch.is_tensor(value) and id(value) not in protected
    )
    isolated = {}
    for owner, name, value in original_buffers:
        if id(value) not in isolated:
            isolated[id(value)] = value.detach().clone()
        owner._buffers[name] = isolated[id(value)]
    target.restore_dict_from_checkpoint(staged.state_dict_for_checkpoint())
    for owner, name, value in original_buffers:
        restored = owner._buffers.get(name)
        if (
            torch.is_tensor(restored)
            and (not value.requires_grad or _canonical_geometry_buffer(owner, name))
            and restored.shape == value.shape
            and restored.dtype == value.dtype
            and restored.device == value.device
            and _equal(restored, value)
        ):
            owner._buffers[name] = value
    for path, original_owner, name, value, unchanged in buffers:
        owner = _registered_owner(target, path)
        if owner is not original_owner or not unchanged:
            owner._buffers[name] = value
    # Checkpoint restoration can rebind schedule value buffers. Raw identities
    # belong to the receiving model and must survive that restoration.
    for path, registry, name, value in source_bindings:
        getattr(_registered_owner(target, path), registry)[name] = value
    for owner, name, value in scalar_writes:
        setattr(owner, name, value)
    for module in target.modules():
        if "_cache" in vars(module):
            module.__dict__["_cache"] = None
    populations = (
        target.populations.values() if isinstance(target, Network) else (target,)
    )
    for population in populations:
        population.integrator.initialized = True
        population._integrator_reinit_pending = False
        population._detached_materializations_pending = True
    if isinstance(target, Network):
        for collection in ("synapses", "continuous_synapses"):
            for name, connection in getattr(target, collection).items():
                counterpart = getattr(staged, collection)[name]
                for attribute in ("weight", "delay_ms"):
                    materializer = getattr(connection, attribute)
                    prepared = getattr(counterpart, attribute)
                    materializer._sampling_draws = _owned(prepared._sampling_draws)
                    materializer._block_lengths = prepared._block_lengths
                    materializer._detached_materializations_pending = True
        target._detached_materializations_pending = True


def _workspace_completion(target):
    """Resolve workspace metadata while publication can still be rolled back.

    Only ordinary version-counter reads and Python tuple assembly remain after
    source copies. In particular no authored getter or workspace builder runs
    after a Parameter version has changed.
    """
    tensors = {
        id(value): value
        for module in target.modules()
        for registry in (module._parameters, module._buffers)
        for value in registry.values()
        if torch.is_tensor(value)
    }

    def bind(signature):
        if signature is None:
            return lambda: None
        if type(signature) is not tuple:
            raise FunctionalizationError(
                "Native handoff cannot rebase this solver workspace contract"
            )
        if (
            len(signature) >= 3
            and isinstance(signature[0], str)
            and type(signature[1]) is int
        ):
            reference = tensors.get(signature[1])
            if (
                reference is None
                or type(reference) not in (torch.Tensor, torch.nn.Parameter)
                or torch.is_inference(reference)
            ):
                raise FunctionalizationError(
                    "Native handoff requires versioned registered solver workspace inputs"
                )
            head, tail = signature[:2], signature[3:]
            return lambda: (*head, reference._version, *tail)
        children = tuple(
            bind(item)
            if isinstance(item, tuple) or item is None
            else lambda item=item: item
            for item in signature
        )
        return lambda: tuple(child() for child in children)

    completions = []
    populations = (
        target.populations.values() if isinstance(target, Network) else (target,)
    )
    for population in populations:
        signature = population._integrator_workspace_contract_signature()
        if signature is not None:
            completions.append((vars(population), bind(signature)))
    return completions


def commit_continuation_(functional, target, continuation):
    """Transfer detached sources, runtime carry and RNG to a native model.

    Accept a lowered Population/Network plan or a runner from ``dn.compile``.
    ``target`` must be a compatible initialized native model. Return that model.
    Existing native Parameter objects, their trainability, gradients and source
    aliases are retained. Incoming source and state Tensor graphs are detached;
    subsequent native gradients belong to the receiving model's raw sources.

    Validate and materialize on an isolated model before publication. Unsupported
    source layouts, changed fixed configuration and effective overrides without
    an equivalent native recipe reject without changing the receiving model.
    No initialization or random sampling is performed. Ordinary state-only
    ``plan.commit_state_`` methods retain their existing graph-preserving API.

    Callback carry is separate: this operation neither installs callbacks nor
    opens storage. Reuse its prepared collection and explicit carry, or use
    export/restore between collections. Inputs and call-local extracellular
    ``extra`` remain invocation arguments. Native ``step`` accepts one
    ``RuntimeDraws`` frame for NOISE/SDE, Network NetStim and Population/Network
    mechanism-owned injections and solver-owned intra/call-local extra Poisson
    banks. Native ``run``,
    ``longrun`` and ``longrun_checkpointed`` accept time-leading draw tapes.
    """
    _cold_boundary("commit_continuation_")
    if type(continuation) is not Continuation:
        raise TypeError("commit_continuation_ requires an exact Continuation")
    from ._convenience import CompiledRunner
    from ._network_compile import CompiledNetworkRunner

    if type(functional) in (CompiledRunner, CompiledNetworkRunner):
        functional = functional.functional
    if type(functional) not in (FunctionalPopulation, FunctionalNetwork):
        raise TypeError(
            "commit_continuation_ requires a Population/Network plan or compiled runner"
        )
    with torch.inference_mode(False), torch.no_grad():
        functional._validate_source()
        _validate_frames(functional, continuation)
        _validate_readonly(functional, continuation.constants)
        if isinstance(functional, FunctionalNetwork):
            functional._validate_compatible(target)
        else:
            functional._validate_compatible_population(target)
        frames = _source_frames(functional, target)
        writes = _source_writes(frames, continuation.parameters, continuation.constants)
        parameters, constants, state = _owned(
            (continuation.parameters, continuation.constants, continuation.state)
        )
        staged, staged_functional = stage_native_handoff(functional, target)
        if isinstance(functional, FunctionalNetwork):
            for population in staged.populations.values():
                _rebind_population_time_references(population)
        else:
            _rebind_population_time_references(staged)
        # The public state-only adapter checks live raw/effective consistency.
        # Restore while the private native clone still has coherent sources,
        # then apply new sources and rebuild against the restored draw history.
        staged_functional.commit_state_(staged, state)
        for destination, value, _label in _source_writes(
            _source_frames(staged_functional, staged), parameters, constants
        ):
            destination.copy_(value)
        scalar_writes = []
        if isinstance(functional, FunctionalNetwork):
            for owner, name, value in scalar_source_writes(
                staged_functional, staged, parameters, constants
            ):
                setattr(owner, name, value)
            scalar_writes = scalar_source_writes(
                functional, target, parameters, constants
            )
            constants = functional._bind_parameter_constants(parameters, constants)
        _refresh_native(staged_functional, staged, parameters, constants)
        _compare_preparation(staged_functional, staged, parameters, constants)
        protected = {id(destination) for destination, _value, _label in writes}
        source_bindings = tuple(
            (path, registry_name, name, value)
            for path, module in target.named_modules()
            for registry_name in ("_parameters", "_buffers")
            for name, value in getattr(module, registry_name).items()
            if id(value) in protected
        )
        buffers = _publication_buffers(target, staged, protected)
        copies = tuple(
            (destination, value)
            for destination, value, _label in writes
            if not _equal(destination, value)
        )
        with _preserve_native_bindings(target):
            _publish_runtime(target, staged, buffers, source_bindings, scalar_writes)
            workspace_completions = _workspace_completion(target)
            # All authored code, allocations, schema checks and runtime restores
            # precede these ordinary dense source copies. Never copy then roll
            # back raw Parameters: doing so would invalidate their old graphs.
            for destination, value in copies:
                destination.copy_(value)
            for namespace, complete in workspace_completions:
                namespace["_validated_integrator_workspace_signature"] = complete()
                namespace["_validated_unversioned_workspace_values"] = None
    return target


__all__ = ["commit_continuation_"]
