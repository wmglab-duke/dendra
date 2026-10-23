"""Pure initialization of a frozen Network from explicit component inputs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import NamedTuple

import torch

from ._connection_sampling import ConnectionSamplingDraw
from ._netstim import NetStimDraw
from ._network_bindings import _canonicalize_mapping_tree
from ._types import FunctionalizationError, InitializationInput


class NetworkInitializationCache(NamedTuple):
    """Detached steady-state Population and normalized connection cache values.

    Population entries use the native state-dict names. Connection entries are
    Tensor-only packets normalized for the plan's fixed topology and timestep.
    Cache leaves are independent editable initialization inputs, not runtime
    checkpoints; restoration does not differentiate through the warm-up run.
    """

    populations: Mapping
    synapses: Mapping
    continuous_synapses: Mapping


class NetworkInitializationInput(NamedTuple):
    """Initialization-only Population values, starting time, and optional draw.

    ``populations`` maps every authored Population name to an
    :class:`InitializationInput`. ``t`` is a scalar Tensor matching the Network
    clock's dtype and device. It is detached during initialization, matching
    native clock synchronization. ``netstim_draw`` is an explicit initialization
    packet; the host runner samples one when omitted. Step draw tapes remain
    separate runtime inputs. A bound Network-wide ``cache`` takes precedence
    over individual Population caches; otherwise each Population's input
    controls its own initialization at its position in the authored order.
    ``reinit_weights`` and ``reinit_delays`` refresh connection materializations
    from their raw sources. A false flag retains the preceding explicit
    materialization, matching native Network initialization.
    ``connection_draw`` supplies parameter-independent connection base draws
    and resulting RNG checkpoints. Pure initialization requires this packet
    when refreshing supported stochastic roles; the host runner samples it
    after Population/cache writes. Its sampling flags must match this input.
    """

    populations: dict[str, InitializationInput]
    t: torch.Tensor
    netstim_draw: NetStimDraw | None = None
    cache: NetworkInitializationCache | None = None
    reinit_weights: bool = True
    reinit_delays: bool = True
    connection_draw: ConnectionSamplingDraw | None = None


def nonpersistent_material_cache_failures(functional):
    """Reject cache writes omitted from another Population's config reader."""
    from ._custom_material_processes import custom_material_constant_slots
    from ._population_cache import _cache_storage_frame
    from ._torch_parametrizations import buffer_slots as torch_buffer_slots

    plans = (
        functional._population_cache_plans
        if functional._cache_enabled
        else {
            name: plan._owned_cache_plan
            for name, plan in functional.populations.items()
            if plan._owned_cache_plan is not None
        }
    )
    if not plans:
        return ()
    bindings = (
        functional._cache_alias_bindings
        if functional._cache_enabled
        else tuple(
            binding
            for entries, _replays in functional._owned_cache_alias_plans.values()
            for binding in entries
        )
    )
    represented = {
        path
        for _winner, _parameters, constant_groups, _states in bindings
        for paths in constant_groups
        for path in paths
    }

    def overlaps(left, right):
        if left[0] != "storage" or right[0] != "storage":
            return False
        if (left[1], left[3]) != (right[1], right[3]):
            return False

        def bounds(frame):
            offset, shape, strides = frame[4:]
            last = offset + sum(
                (size - 1) * stride for size, stride in zip(shape, strides, strict=True)
            )
            return offset * frame[2].itemsize, (last + 1) * frame[2].itemsize

        start, end = bounds(left)
        first, last = bounds(right)
        return start < last and first < end

    failures = []
    for name, population_plan in functional.populations.items():
        population = population_plan._population
        config_ids = {
            id(value)
            for process in population.mech.material_processes.values()
            for _slot, value in custom_material_constant_slots(process)
        }
        torch_config_ids = {
            id(value) for _slot, value in torch_buffer_slots(population)
        }
        config_ids.update(torch_config_ids)
        for entry in population_plan._parametrization_constants:
            if ("populations", name, entry.key) in represented:
                continue
            owner = population.get_submodule(entry.module_path)
            if entry.buffer_name not in owner._non_persistent_buffers_set:
                continue
            value = owner._buffers[entry.buffer_name]
            if id(value) not in config_ids:
                continue
            reader = _cache_storage_frame(value)
            if any(
                writer_name != name
                and any(
                    overlaps(reader, plan.storage_frames[key])
                    for key in plan.write_keys
                )
                for writer_name, plan in plans.items()
            ):
                failures.append(
                    f"Population {name!r} nonpersistent "
                    f"{'torch parametrization' if id(value) in torch_config_ids else 'custom MaterialProcess'} "
                    f"config {entry.key!r} shares storage with another "
                    "Population's cached tensor. This cache restoration is "
                    "currently unsupported; use independent config buffers "
                    "or register the shared config as persistent on every reader."
                )
    return tuple(failures)


def initialization_alias_failures(functional):
    """Reject native retained materializations absent from the explicit packet."""
    from ._initialization_sources import (
        initialization_source_constant_slots,
        initialization_source_roots,
    )
    from ._population import (
        _initialization_only_parameter_names,
        _initialization_transform_hooks,
    )

    order = {name: index for index, name in enumerate(functional.populations)}
    initial_only = {
        name: _initialization_only_parameter_names(plan._population)
        for name, plan in functional.populations.items()
    }
    failures = []
    initial_constants = {}
    for name, plan in functional.populations.items():
        if plan._owned_cache_bound:
            continue
        population = plan._population
        source_buffer_ids = {
            id(value)
            for _path, source in initialization_source_roots(population)
            for _slot, value in initialization_source_constant_slots(source)
        }
        for entry in plan._parametrization_constants:
            value = population.get_submodule(entry.module_path)._buffers[
                entry.buffer_name
            ]
            if id(value) in source_buffer_ids:
                initial_constants[("populations", name, entry.key)] = name
    for parameter_path, constant_paths in functional._parameter_constant_aliases:
        for constant_path in constant_paths:
            name = initial_constants.get(constant_path)
            if name is None or parameter_path[:2] == ("populations", name):
                continue
            failures.append(
                f"Population {name!r} initialization source buffer "
                f"{constant_path[2]!r} aliases parameter "
                f"{'.'.join(parameter_path)!r} registered only outside that "
                "Population. Cross-Population initializer buffer/parameter "
                "aliases are currently unsupported; register the same Parameter "
                "on the initializer module as a Parameter attribute to preserve "
                "its canonical shared binding"
            )
    for name, plan in functional.populations.items():
        if plan._owned_cache_bound:
            # Owned-cache initialization skips both pre transforms and
            # structured post transforms, matching Population.initialize.
            continue
        population = plan._population
        writes = {
            reference.removeprefix("parameters.")
            for phase in ("pre", "post")
            for hook in _initialization_transform_hooks(population, phase)
            for reference in population._initialization_transforms[
                hook.action_name
            ].writes
            if reference.startswith("parameters.")
        }
        for path, constant_paths in functional._parameter_constant_aliases:
            if path[:2] == ("populations", name) and path[2] in writes:
                bindings = ", ".join(
                    ".".join(map(str, target)) for target in constant_paths
                )
                failures.append(
                    f"Population {name!r} initialization writes a parameter shared "
                    f"with explicit Network constants ({bindings}). These writes "
                    "are currently unsupported; parameter replacements do not "
                    "replace constant bindings"
                )
        for group in functional._parameter_aliases:
            if not any(
                path[:2] == ("populations", name) and path[2] in writes
                for path in group
            ):
                continue
            for path in group:
                if (
                    path[0] == "populations"
                    and order[path[1]] < order[name]
                    and path[2] not in initial_only[path[1]]
                ):
                    failures.append(
                        f"Population {name!r} writes a shared runtime parameter of "
                        f"already initialized Population {path[1]!r}; native initialization "
                        "retains its earlier materialized coefficients. This ordering is "
                        "currently unsupported; initialize shared runtime parameters "
                        "before their first Population reader"
                    )
    for owner, (bindings, _replays) in functional._owned_cache_alias_plans.items():
        for _winner, parameter_groups, _constants, _states in bindings:
            for paths in parameter_groups:
                for path in paths:
                    if (
                        path[0] == "populations"
                        and order[path[1]] < order[owner]
                        and not functional.populations[path[1]]._owned_cache_bound
                        and path[2] not in initial_only[path[1]]
                    ):
                        failures.append(
                            f"Population {owner!r} cache writes a shared runtime "
                            f"parameter of already initialized fresh Population "
                            f"{path[1]!r}; native initialization retains its earlier "
                            "materialized coefficients. This ordering is currently "
                            "unsupported; initialize the cached owner before its "
                            "fresh parameter readers"
                        )

    if not functional._owned_cache_alias_plans:
        return tuple(dict.fromkeys(failures))

    from ._population_cache import _cache_storage_frame

    def overlaps(left, right):
        if left == right:
            return True
        if (
            left[0] != "storage"
            or right[0] != "storage"
            or left[1] != right[1]
            or left[3] != right[3]
        ):
            return False

        def bounds(frame):
            offset, shape, strides = frame[4:]
            width = frame[2].itemsize
            end = offset + sum(
                (length - 1) * stride
                for length, stride in zip(shape, strides, strict=True)
            )
            return offset * width, (end + 1) * width - 1

        start, end = bounds(left)
        other_start, other_end = bounds(right)
        return start <= other_end and other_start <= end

    fresh_materializations = {}
    for name, plan in functional.populations.items():
        if plan._owned_cache_bound:
            continue
        population = plan._population
        paths = [
            name
            for name in plan._preparation.population_parameter_names
            if name in population._buffers
        ]
        paths.extend(
            f"{entry.module_path + '.' if entry.module_path else ''}{entry.buffer_name}"
            for entry in plan._prepared_buffers
            if entry.kind in {"effective_parameter", "derived"}
        )
        for record, preparation in zip(
            plan._multi_component_plans,
            getattr(plan._preparation, "component_physical_preparations", ()),
            strict=True,
        ):
            component = population.populations[record.name]
            paths.extend(
                f"populations.{record.name}.{field}"
                for field in preparation.population_parameter_names
                if field in component._buffers
            )
        fresh_materializations[name] = tuple(
            _cache_storage_frame(
                population.get_buffer(path), buffer=True, owner=population
            )
            for path in dict.fromkeys(paths)
        )
    for owner in functional._owned_cache_alias_plans:
        cache_plan = functional.populations[owner]._owned_cache_plan
        writes = tuple(cache_plan.storage_frames[key] for key in cache_plan.write_keys)
        represented = tuple(
            cache_plan.storage_frames[source]
            for source in (
                *cache_plan.parameters.values(),
                *cache_plan.constants.values(),
                *(source for _path, source in cache_plan.states),
            )
        )
        for name, materializations in fresh_materializations.items():
            # Later cache copies can overwrite earlier retained coefficients;
            # later fresh population can overwrite the cached owner's readers.
            # Neither mutation is represented by recomputing fresh preparation
            # beside immutable detached cache materializations.
            affected = writes if order[name] < order[owner] else represented
            if any(
                overlaps(saved, current)
                for saved in affected
                for current in materializations
            ):
                failures.append(
                    f"Population {owner!r} cache shares retained materializations "
                    f"with fresh Population {name!r}. Shared runtime buffers with "
                    "different initialization materializations are currently unsupported"
                )
    return tuple(dict.fromkeys(failures))


def _replace_at(tree, path, value):
    """Replace one leaf in an owned container tree, retaining tensor history."""
    if not path:
        return value
    key, *remaining = path
    if isinstance(tree, Mapping):
        result = dict(tree)
        result[key] = _replace_at(tree[key], remaining, value)
        return result
    values = list(tree)
    values[key] = _replace_at(tree[key], remaining, value)
    return type(tree)(*values) if hasattr(tree, "_fields") else tuple(values)


def _propagate_population_parameters(functional, parameters, name, replacement):
    """Publish a Population transaction through shared Network parameter aliases."""
    from ._network import _at

    result = _replace_at(parameters, ("populations", name), replacement)
    for group in functional._parameter_aliases:
        changed = [
            path
            for path in group
            if path[:2] == ("populations", name)
            and _at(result, path) is not _at(parameters, path)
        ]
        if not changed:
            continue
        value = _at(result, changed[-1])
        if any(_at(result, path) is not value for path in changed[:-1]):
            raise FunctionalizationError(
                "Population initialization produced conflicting shared parameter replacements"
            )
        for path in group:
            result = _replace_at(result, path, value)
    return result


def _cache_alias_plan(functional, writers, readers):
    """Freeze one native cache-copy transaction and all of its readers."""
    from ._population_cache import _cache_storage_frame, cache_storage_replays

    groups = {}
    constant_frames = {}
    constant_sources = {}
    frame_sources, writes = {}, []
    for name, plan in writers.items():
        for key in plan.write_keys:
            frame = plan.storage_frames[key]
            group = groups.setdefault(frame, [None, [], [], []])
            group[0] = (name, key)
            frame_sources.setdefault(frame, plan.sources[key])
            writes.append(((name, key), plan.sources[key], frame))

    def overlaps(frame, other):
        if (
            frame[0] != "storage"
            or other[0] != "storage"
            or frame[1] != other[1]
            or frame[3] != other[3]
        ):
            return False

        def bounds(value):
            offset, shape, strides = value[4:]
            size = value[2].itemsize
            end = offset + sum(
                (length - 1) * stride
                for length, stride in zip(shape, strides, strict=True)
            )
            return offset * size, (end + 1) * size - 1

        start, end = bounds(frame)
        other_start, other_end = bounds(other)
        return start <= other_end and other_start <= end

    written_frames = tuple(groups)
    for name, plan in readers.items():
        for collection, bindings in (
            (1, plan.parameters.items()),
            (2, plan.constants.items()),
            (3, plan.states),
        ):
            for public, source in bindings:
                frame = plan.storage_frames[source]
                if frame not in groups:
                    if not any(overlaps(frame, other) for other in written_frames):
                        continue
                    groups[frame] = [None, [], [], []]
                    frame_sources[frame] = plan.sources[source]
                group = groups[frame]
                path = (
                    (name, *public)
                    if collection == 3
                    else ("populations", name, public)
                )
                group[collection].append(path)
                if collection == 2:
                    constant_frames[path] = frame
                    constant_sources[path] = plan.sources[source]
    # Some saved Population buffers are exposed only as cache materializations,
    # so their packet constants intentionally do not share snapshot identity.
    # Resolve outside execution parameters against the original native Tensor
    # as well; a native state-dict copy still updates every such reader.
    from ._continuous_transform import _named_tensors
    from ._network import _tensor_paths
    from ._schedules import TensorScheduleTensors, _TensorSchedulePlan

    native = {
        "populations": {},
        "synapses": {},
        "continuous_synapses": {},
        "netstim": None,
    }
    native_constants = {"populations": {}, "synapses": {}, "continuous_synapses": {}}
    for name, plan in functional.populations.items():
        native["populations"][name] = plan._parameter_sources()
        native_constants["populations"][name] = {
            entry.key: plan._population.get_submodule(entry.module_path)._buffers[
                entry.buffer_name
            ]
            for entry in plan._parametrization_constants
        }
    for name, plan in functional.synapses.items():
        connection = plan._source
        sources = functional._connection_parameter_plans["synapses", name]
        native["synapses"][name] = {
            **sources.parameter_sources(connection),
            "schedule": TensorScheduleTensors(**_TensorSchedulePlan._values(connection))
            if functional._parameter_schema["synapses"][name]["schedule"] is not None
            else None,
        }
        native_constants["synapses"][name] = sources.cache_constant_sources(connection)
    for name, plan in functional.continuous_synapses.items():
        connection = plan._source
        sources = functional._connection_parameter_plans["continuous_synapses", name]
        native["continuous_synapses"][name] = {
            **sources.parameter_sources(connection),
            "transform_parameters": _named_tensors(connection.transform)
            if isinstance(connection.transform, torch.nn.Module)
            else None,
        }
        native_constants["continuous_synapses"][name] = sources.cache_constant_sources(
            connection
        )
    if functional.netstim is not None:
        source = functional.netstim._source
        sources = functional._netstim_parameter_plan
        native["netstim"] = {
            "parameters": sources.parameter_sources(source),
            "schedule": TensorScheduleTensors(**_TensorSchedulePlan._values(source))
            if functional._parameter_schema["netstim"]["schedule"] is not None
            else None,
        }
        native_constants["netstim"] = {
            "sources": sources.cache_constant_sources(source)
        }
    native_parameters = dict(_tensor_paths(native))
    parameter_frames = {
        path: _cache_storage_frame(source) for path, source in native_parameters.items()
    }
    for path, frame in parameter_frames.items():
        if frame not in groups and any(
            overlaps(frame, other) for other in written_frames
        ):
            groups[frame] = [None, [], [], []]
            frame_sources[frame] = native_parameters[path]
        if frame in groups:
            parameters = groups[frame][1]
            if path not in parameters:
                parameters.append(path)
    for path, source in _tensor_paths(native_constants):
        constant_sources[path] = source
        # Population cache bindings own their destination frames, including
        # expanded buffers rebound to owned storage. Missing Population readers
        # are admitted only by the exact parameter-alias closure below.
        if path[0] == "populations" or path in constant_frames:
            continue
        frame = _cache_storage_frame(source)
        if frame not in groups and any(
            overlaps(frame, other) for other in written_frames
        ):
            groups[frame] = [None, [], [], []]
            frame_sources[frame] = source
        if frame in groups:
            constants = groups[frame][2]
            if path not in constants:
                constants.append(path)
            constant_frames[path] = frame
    result, outputs = [], []
    for frame, (winner, parameters, constants, states) in groups.items():
        if not (parameters or constants or states):
            continue
        changed = True
        while changed:
            count = len(parameters) + len(constants)
            for aliases in functional._parameter_aliases:
                if any(path in aliases for path in parameters):
                    parameters.extend(
                        path
                        for path in aliases
                        if path not in parameters
                        and parameter_frames.get(path) == frame
                    )
            for path, constant_paths in functional._parameter_constant_aliases:
                if parameter_frames.get(path) == frame and (
                    path in parameters
                    or any(target in constants for target in constant_paths)
                ):
                    if path not in parameters:
                        parameters.append(path)
                    constants.extend(
                        target
                        for target in constant_paths
                        if target not in constants
                        and constant_frames.get(target, frame) == frame
                    )
            changed = len(parameters) + len(constants) != count
        parameter_groups = {}
        for path in parameters:
            parameter_groups.setdefault(id(native_parameters[path]), []).append(path)
        # A cache copy updates every reader of the same authored Tensor.
        # Distinct Tensor views can share a storage frame without sharing identity.
        constant_groups = {}
        for path in constants:
            constant_groups.setdefault(id(constant_sources[path]), []).append(path)
        outputs.append((len(result), frame_sources[frame], frame))
        result.append(
            (
                winner,
                tuple(tuple(paths) for paths in parameter_groups.values()),
                tuple(tuple(paths) for paths in constant_groups.values()),
                tuple(states),
            )
        )
    replays = cache_storage_replays(writes, outputs)
    resolved = {identifier for replay in replays for identifier in replay.output_ids}
    if any(
        winner is None and index not in resolved
        for index, (winner, *_rest) in enumerate(result)
    ):
        raise FunctionalizationError(
            "Population cache cannot fully restore an external Tensor view; this alias is currently unsupported"
        )
    return tuple(result), replays


def cache_alias_bindings(functional):
    """Freeze the complete Network-owned cache-copy phase."""
    functional._cache_storage_replays = ()
    if not functional._cache_enabled or functional._cache_error is not None:
        return ()
    try:
        bindings, replays = _cache_alias_plan(
            functional,
            functional._population_cache_plans,
            functional._population_cache_plans,
        )
    except (FunctionalizationError, ValueError, TypeError, RuntimeError) as error:
        functional._cache_error = str(error)
        return ()
    functional._cache_storage_replays = replays
    return bindings


def population_cache_alias_bindings(functional):
    """Freeze individual Population restores without moving their native phase."""
    if functional._cache_enabled:
        return {}, None
    owners = {
        name: plan._owned_cache_plan
        for name, plan in functional.populations.items()
        if plan._owned_cache_plan is not None
    }
    if not owners:
        return {}, None
    from ._population_cache import PopulationCachePlan

    try:
        readers = {
            name: owners[name]
            if name in owners
            else PopulationCachePlan(plan, None, bindings_only=True)
            for name, plan in functional.populations.items()
        }
        plans = {
            name: _cache_alias_plan(functional, {name: owner}, readers)
            for name, owner in owners.items()
        }
    except (FunctionalizationError, ValueError, TypeError, RuntimeError) as error:
        return {}, str(error)
    return plans, None


def _restore_cache_aliases(
    functional,
    parameters,
    effective,
    constants,
    populations,
    cache,
    *,
    bindings=None,
    replays=None,
):
    """Publish final native cache writes before rebuilding runtime workspaces."""
    from dendra.models.networks._population_cache import (
        replay_population_cache_storage,
        restore_population_cache_values,
    )

    from ._network import _at

    resolved = {}
    for replay in functional._cache_storage_replays if replays is None else replays:
        values = replay_population_cache_storage(
            tuple(_at(cache.populations, path) for path in replay.write_paths),
            replay.write_indices,
            replay.output_indices,
            replay.output_shapes,
            size=replay.size,
        )
        resolved.update(zip(replay.output_ids, values, strict=True))
    for index, (winner, parameter_groups, constant_groups, state_paths) in enumerate(
        functional._cache_alias_bindings if bindings is None else bindings
    ):
        saved = resolved[index] if index in resolved else _at(cache.populations, winner)
        for parameter_paths in parameter_groups:
            reference = _at(parameters, parameter_paths[0])
            # NetStim and stimulation fields may expose a broadcasted view of
            # their authored Tensor. Keep that existing explicit packet shape.
            parameter_saved = saved.expand_as(reference)
            value = restore_population_cache_values(
                {"value": reference},
                {"value": parameter_saved},
                parameter_names=("value",),
            )["value"]
            for path in parameter_paths:
                effective = _replace_at(effective, path, value)
        for constant_paths in constant_groups:
            reference = _at(constants, constant_paths[0])
            value = saved.detach().to(reference).clone()
            for path in constant_paths:
                constants = _replace_at(constants, path, value)
        for path in state_paths:
            reference = _at(populations, path)
            populations = _replace_at(
                populations, path, saved.detach().to(reference).clone()
            )
    return effective, constants, populations


def _validate(functional, parameters, constants, state, inputs, draw, *, require_draw):
    from ._network import _validate_tree

    functional._validate_initialization_source()
    functional._validate_parameters(parameters)
    _validate_tree(constants, functional._constant_schema, "constants")
    functional._validate_state(state)
    if not isinstance(inputs, NetworkInitializationInput):
        raise TypeError("inputs must be a NetworkInitializationInput")
    for name in ("reinit_weights", "reinit_delays"):
        if type(getattr(inputs, name)) is not bool:
            raise TypeError(f"initialization.{name} must be a bool")
    functional._validate_initialization_cache(inputs.cache)
    if not isinstance(inputs.populations, Mapping) or set(inputs.populations) != set(
        functional.populations
    ):
        raise ValueError("initialization populations must match the Network")
    _validate_tree(inputs.t, functional._state_schema["clock"]["t"], "initialization.t")
    if not torch._C._are_functorch_transforms_active():
        if torch.compiler.is_compiling():
            torch._assert_async(
                torch.isfinite(inputs.t), "initialization.t must be finite"
            )
        elif not bool(torch.isfinite(inputs.t)):
            raise ValueError("initialization.t must be finite")
    if functional.netstim is None:
        if draw is not None:
            raise ValueError(
                "a Network without NetStim does not accept an initialization draw"
            )
    elif draw is None and require_draw:
        raise ValueError(
            "NetStim initialization requires an explicit draw; use sample_initialization_draw() outside transformed execution"
        )
    elif draw is not None:
        functional.netstim._validate_draw(draw)
    sampling = functional._connection_sampling_plan
    sampling.validate_draw(
        inputs.connection_draw,
        reinit_weights=inputs.reinit_weights,
        reinit_delays=inputs.reinit_delays,
        required=require_draw
        and sampling.requires_draw(
            reinit_weights=inputs.reinit_weights, reinit_delays=inputs.reinit_delays
        ),
    )
    if inputs.connection_draw is not None:
        _validate_tree(
            inputs.connection_draw.rng,
            functional._state_schema["connection_sampling"]["rng"],
            "initialization.connection_draw.rng",
        )


def _initialize_components(
    functional, parameters, constants, state, inputs, *, sample_random=False
):
    """Complete the native phases preceding NetStim initialization."""
    from ._network import NetworkTensors

    populations = dict(state["populations"])
    effective = parameters
    effective_constants = functional._bind_parameter_constants(parameters, constants)
    with nullcontext() if functional.training else torch.no_grad():
        # Native initialize completes each Population in authored order before
        # resetting connections. Parameter-writing transforms affect every
        # alias before the next Population observes its input frame.
        for name, plan in functional.populations.items():
            incoming_parameters = effective
            if inputs.cache is None:
                initialize_population = (
                    plan.initialize_host if sample_random else plan.initialize
                )
                initialized = initialize_population(
                    effective["populations"][name],
                    effective_constants["populations"][name],
                    inputs.populations[name],
                    state=populations[name],
                )
            else:
                initialized = functional._population_cache_plans[name].restore(
                    effective["populations"][name],
                    effective_constants["populations"][name],
                    state["populations"][name],
                    inputs.cache.populations[name],
                )
            effective_constants = _replace_at(
                effective_constants, ("populations", name), initialized.constants
            )
            effective = _propagate_population_parameters(
                functional, effective, name, initialized.parameters
            )
            populations[name] = initialized.state
            if inputs.cache is None and name in functional._owned_cache_alias_plans:
                bindings, replays = functional._owned_cache_alias_plans[name]
                effective, effective_constants, populations = _restore_cache_aliases(
                    functional,
                    incoming_parameters,
                    effective,
                    effective_constants,
                    populations,
                    NetworkInitializationCache(
                        {name: inputs.populations[name].cache}, {}, {}
                    ),
                    bindings=bindings,
                    replays=replays,
                )
            # A cache transaction can replace a selectable raw input while
            # restoring the same authored Tensor through a config-buffer slot.
            # The resulting raw input owns all exact-identity config readers.
            effective_constants = functional._bind_parameter_constants(
                effective, effective_constants
            )
        if inputs.cache is not None:
            effective, effective_constants, populations = _restore_cache_aliases(
                functional,
                parameters,
                effective,
                effective_constants,
                populations,
                inputs.cache,
            )
            effective_constants = functional._bind_parameter_constants(
                effective, effective_constants
            )
        time = inputs.t.detach().clone()
        populations = {
            name: functional._snap_population_clock(value, time)
            for name, value in populations.items()
        }
        clock = {
            "t": time,
            "origin": time.clone(),
            "step": torch.zeros_like(state["clock"]["step"]),
            "duration_remainder": torch.zeros_like(
                state["clock"]["duration_remainder"]
            ),
        }
        draw = inputs.connection_draw
        if draw is None and sample_random:
            draw = functional._connection_sampling_plan.sample(
                effective,
                effective_constants,
                state,
                reinit_weights=inputs.reinit_weights,
                reinit_delays=inputs.reinit_delays,
            )
        functional._connection_sampling_plan.validate_draw(
            draw,
            reinit_weights=inputs.reinit_weights,
            reinit_delays=inputs.reinit_delays,
            required=functional._connection_sampling_plan.requires_draw(
                reinit_weights=inputs.reinit_weights,
                reinit_delays=inputs.reinit_delays,
            ),
        )
        for (collection, name), plan in functional._connection_parameter_plans.items():
            connection_constants = plan.initialize_constants(
                effective[collection][name],
                effective_constants[collection][name],
                reinit_weights=inputs.reinit_weights,
                reinit_delays=inputs.reinit_delays,
                draws=None if draw is None else draw.values[collection][name],
            )
            effective_constants = _replace_at(
                effective_constants, (collection, name), connection_constants
            )
        connections = functional._materialize_connections(
            effective, effective_constants
        )
        synapses = {
            name: plan.initial_state(state["synapses"][name], t=time)
            for name, plan in functional.synapses.items()
        }
        continuous = {
            name: plan.initial_state(state["continuous_synapses"][name], t=time)
            for name, plan in functional.continuous_synapses.items()
        }
        if inputs.cache is not None:
            # Native restores cached history after resetting connections and
            # refreshing their parameters, and before NetStim initialization.
            for name, cache in inputs.cache.synapses.items():
                synapses[name] = functional.synapses[name].restore_cache(
                    {
                        **connections["synapses"][name],
                        "schedule": effective["synapses"][name]["schedule"],
                    },
                    synapses[name],
                    cache,
                    t=time,
                )
            for name, cache in inputs.cache.continuous_synapses.items():
                continuous[name] = functional.continuous_synapses[name].restore_cache(
                    {
                        "weights": connections["continuous_synapses"][name]["weights"],
                        "transform_parameters": effective["continuous_synapses"][name][
                            "transform_parameters"
                        ],
                    },
                    continuous[name],
                    cache,
                    t=time,
                )
        initialized_state = {
            "populations": populations,
            "synapses": synapses,
            "continuous_synapses": continuous,
            "netstim": state["netstim"],
            "clock": clock,
            **(
                {
                    "connection_sampling": {
                        "rng": {
                            name: value.detach().clone()
                            for name, value in draw.rng.items()
                        }
                    }
                    if draw is not None
                    else state["connection_sampling"]
                }
                if functional._connection_sampling_plan.devices
                else {}
            ),
        }
    return NetworkTensors(effective, effective_constants, initialized_state)


def _finish_initialization(functional, packet, draw):
    from ._network import NetworkTensors

    state = dict(packet.state)
    if functional.netstim is not None:
        state["netstim"] = functional.netstim.initial_state(
            functional._materialize_netstim_parameters(
                packet.parameters, packet.constants
            ),
            functional._netstim_constants(packet.constants),
            state["netstim"],
            draw,
        )
    functional._validate_parameters(packet.parameters)
    functional._validate_state(state)
    return NetworkTensors(packet.parameters, packet.constants, state)


def initialize(functional, parameters, constants, state, inputs, *, netstim_draw=None):
    """Initialize one complete boundary without sampling or mutating operands."""
    (parameters, constants, state, inputs, netstim_draw), _ = (
        _canonicalize_mapping_tree((parameters, constants, state, inputs, netstim_draw))
    )
    if not isinstance(inputs, NetworkInitializationInput):
        raise TypeError("inputs must be a NetworkInitializationInput")
    if netstim_draw is not None and inputs.netstim_draw is not None:
        raise ValueError("supply the NetStim initialization draw only once")
    draw = inputs.netstim_draw if netstim_draw is None else netstim_draw
    _validate(functional, parameters, constants, state, inputs, draw, require_draw=True)
    packet = _initialize_components(functional, parameters, constants, state, inputs)
    return _finish_initialization(functional, packet, draw)


def initialize_host(functional, parameters, constants, state, inputs):
    """Cold sampling adapter over the same pure initialization phases."""
    (parameters, constants, state, inputs), _ = _canonicalize_mapping_tree(
        (parameters, constants, state, inputs)
    )
    if not isinstance(inputs, NetworkInitializationInput):
        raise TypeError("inputs must be a NetworkInitializationInput")
    _validate(
        functional,
        parameters,
        constants,
        state,
        inputs,
        inputs.netstim_draw,
        require_draw=False,
    )
    packet = _initialize_components(
        functional, parameters, constants, state, inputs, sample_random=True
    )
    draw = inputs.netstim_draw
    if functional.netstim is not None and draw is None:
        # Native initialization samples last. Population transforms may have
        # changed a parameter shared with NetStim, including conditional noise.
        draw = functional.sample_initialization_draw(
            packet.parameters, packet.state, constants=packet.constants
        )
    return _finish_initialization(functional, packet, draw)


__all__ = ["NetworkInitializationCache", "NetworkInitializationInput"]
