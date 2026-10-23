"""Cold state-dict bindings for shared pure Population cache restoration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch

from dendra.models.networks._population_cache import (
    replay_population_cache_storage,
    restore_population_cache_values,
)

from ._population import _buffer_at, _geometry_constant_value, _get_path, _put_path
from ._types import FunctionalizationError, PopulationTensors


def _cache_storage_frame(value, *, buffer=False, owner=None):
    """Identify an exact native cache-copy destination without tying AD leaves.

    Native loading copies through detached views, so distinct Tensor objects
    can still overwrite the same storage frame. Expanded buffers are rebound
    to owned storage before loading, memoized by their original identity.
    """
    if value.is_neg() or value.is_conj():
        raise FunctionalizationError(
            "Population cache negative or conjugate Tensor views are currently unsupported"
        )
    if buffer and any(
        size > 1 and stride == 0 for size, stride in zip(value.shape, value.stride())
    ):
        return ("owned_buffer", id(owner), id(value))
    if not value.numel():
        return ("identity", id(value))
    return (
        "storage",
        value.device,
        value.dtype,
        value.untyped_storage().data_ptr(),
        value.storage_offset(),
        tuple(value.shape),
        tuple(value.stride()),
    )


class CacheStorageReplay(NamedTuple):
    size: int
    write_paths: tuple
    write_indices: tuple
    output_ids: tuple
    output_indices: tuple
    output_shapes: tuple


def _frame_indices(frame, origin):
    offset, shape, strides = frame[4:]
    indices = torch.tensor(offset - origin, dtype=torch.long, device="cpu")
    for axis, (length, stride) in enumerate(zip(shape, strides, strict=True)):
        view = [1] * len(shape)
        view[axis] = length
        indices = indices + stride * torch.arange(
            length, dtype=torch.long, device="cpu"
        ).reshape(view)
    return indices.expand(shape).reshape(-1)


def cache_storage_replays(writes, outputs):
    """Resolve partial native cache-copy views into immutable replay metadata.

    ``writes`` and ``outputs`` contain ``(identifier, native_tensor, frame)``
    triples. Exact-frame groups retain their direct cache-key fast path.
    """
    groups = {}
    for kind, entries in ((0, writes), (1, outputs)):
        for order, (identifier, source, frame) in enumerate(entries):
            if frame[0] != "storage":
                continue
            key = (frame[1], frame[3])
            group = groups.setdefault(key, [[], []])
            group[kind].append((order, identifier, source, frame))

    recipes = []
    for write_entries, output_entries in groups.values():
        if not write_entries or not output_entries:
            continue
        frames = {entry[3] for entry in (*write_entries, *output_entries)}
        if len(frames) == 1:
            continue
        if len({frame[2] for frame in frames}) != 1:
            raise FunctionalizationError(
                "Population cache views sharing storage with different dtypes "
                "are currently unsupported"
            )
        final_writes = {entry[3]: entry for entry in write_entries}
        frame_outputs = {}
        for entry in output_entries:
            frame_outputs.setdefault(entry[3], []).append(entry)

        def bounds(frame):
            offset, shape, strides = frame[4:]
            return offset, offset + sum(
                (length - 1) * stride
                for length, stride in zip(shape, strides, strict=True)
            )

        components = []
        for frame in sorted(frames, key=bounds):
            start, end = bounds(frame)
            if not components or start > components[-1][1]:
                components.append([start, end, {frame}])
            else:
                components[-1][1] = max(components[-1][1], end)
                components[-1][2].add(frame)
        for origin, _end, component in components:
            if len(component) == 1:
                continue
            selected_outputs = sorted(
                entry for frame in component for entry in frame_outputs.get(frame, ())
            )
            if not selected_outputs:
                continue
            # An earlier write to the exact same frame is fully overwritten by
            # its last occurrence, including intervening partial writes.
            last_writes = {
                frame: final_writes[frame]
                for frame in component
                if frame in final_writes
            }
            selected_writes = sorted(last_writes.values())
            if not selected_writes:
                continue
            indices = {frame: _frame_indices(frame, origin) for frame in component}
            for frame in last_writes:
                index = indices[frame]
                if index.unique().numel() != index.numel():
                    raise FunctionalizationError(
                        "Population cache views with overlapping elements are "
                        "currently unsupported"
                    )
            covered = torch.cat(
                [indices[entry[3]] for entry in selected_writes]
            ).unique()
            for _order, _identifier, _source, frame in selected_outputs:
                positions = torch.searchsorted(covered, indices[frame])
                safe = positions.clamp(max=covered.numel() - 1)
                if not bool(
                    (
                        (positions < covered.numel())
                        & (covered[safe] == indices[frame])
                    ).all()
                ):
                    raise FunctionalizationError(
                        "Population cache writes cover only part of an external "
                        "Tensor view; this alias is currently unsupported"
                    )
            indices = {
                frame: torch.searchsorted(covered, index)
                for frame, index in indices.items()
            }
            device = selected_writes[0][2].device
            recipes.append(
                CacheStorageReplay(
                    size=covered.numel(),
                    write_paths=tuple(entry[1] for entry in selected_writes),
                    write_indices=tuple(
                        indices[entry[3]].to(device) for entry in selected_writes
                    ),
                    output_ids=tuple(entry[1] for entry in selected_outputs),
                    output_indices=tuple(
                        indices[entry[3]].to(device) for entry in selected_outputs
                    ),
                    output_shapes=tuple(
                        tuple(entry[2].shape) for entry in selected_outputs
                    ),
                )
            )
    return tuple(recipes)


class PopulationCachePlan:
    """Resolve saved module slots once, then restore explicit tensor packets."""

    def __init__(self, plan, cache=None, *, bindings_only=False):
        self.plan = plan
        population = plan._population
        reference = population.state_dict(keep_vars=True)
        preserved = {f"integrator.{name}" for name in population.integrator._buffers}
        if not bindings_only and isinstance(cache, Mapping):
            cache = {
                **cache,
                **{name: reference[name] for name in preserved if name in reference},
            }
        if not bindings_only and (
            not isinstance(cache, Mapping) or set(cache) != set(reference)
        ):
            raise FunctionalizationError(
                "Population state cache must match the initialized state-dict schema"
            )
        from ._random_parameters import cache_metadata_tensors

        self._reference = {
            name: value if bindings_only else cache_metadata_tensors(value, plan.device)
            for name, value in reference.items()
        }
        parameter_slots = dict(population.named_parameters(remove_duplicate=False))
        buffer_slots = dict(population.named_buffers(remove_duplicate=False))
        slots = {
            **parameter_slots,
            **buffer_slots,
            "_duration_remainder": population._duration_remainder,
        }
        self.sources = slots
        self.storage_frames = {
            name: _cache_storage_frame(
                value, buffer=name in buffer_slots, owner=population
            )
            for name, value in slots.items()
        }
        # Population's unregistered duration carrier is replaced by its load
        # hook; copying the cache never mutates an outside reader of its old
        # Tensor storage.
        self.storage_frames["_duration_remainder"] = (
            "replacement",
            id(population),
            "_duration_remainder",
        )
        self.write_keys = tuple(
            name for name in reference if name in slots and name not in preserved
        )
        aliases = {}
        for name in self.write_keys:
            aliases.setdefault(self.storage_frames[name], []).append(name)

        def key(value, fallback):
            frame = self.storage_frames.get(fallback)
            if frame is None:
                frame = _cache_storage_frame(value)
            names = aliases.get(frame, ())
            if names:
                return names[-1]
            if fallback in reference:
                return fallback
            raise FunctionalizationError(
                f"Population cache has no source for {fallback!r}"
            )

        self.parameters = {
            name: key(value, name)
            for name, value in population.named_parameters()
            if name in plan._population_parameter_names
        }
        self.states = tuple(
            (
                leaf.public_path,
                key(
                    _buffer_at(population, leaf.module_path, leaf.buffer_name),
                    f"{leaf.module_path + '.' if leaf.module_path else ''}{leaf.buffer_name}",
                ),
            )
            for leaf in plan._state_layout
        )
        self.constants = {}

        def constant_source(value, fallback, owner, buffer_name):
            # Native state-dict restoration leaves nonpersistent configuration
            # alone unless a persistent alias writes the same storage frame.
            if (
                fallback not in reference
                and not aliases.get(self.storage_frames.get(fallback), ())
                and buffer_name in owner._non_persistent_buffers_set
            ):
                # This reader is absent from the native load, so expanded
                # views retain their original storage rather than being owned.
                frame = _cache_storage_frame(value)
                if frame is not None and frame[0] == "storage":

                    def byte_bounds(tensor, layout):
                        offset, shape, strides = layout[4:]
                        last = offset + sum(
                            (size - 1) * stride
                            for size, stride in zip(shape, strides, strict=True)
                        )
                        return offset * tensor.element_size(), (
                            last + 1
                        ) * tensor.element_size()

                    start, end = byte_bounds(value, frame)
                    for written in self.write_keys:
                        target = self.storage_frames[written]
                        if target[0] != "storage" or (target[1], target[3]) != (
                            frame[1],
                            frame[3],
                        ):
                            continue
                        first, last = byte_bounds(slots[written], target)
                        if start < last and first < end:
                            raise FunctionalizationError(
                                "Nonpersistent config views partially overlapping "
                                "cached tensor storage are currently unsupported; "
                                "use independent config buffers or an exact shared binding"
                            )
                return None
            return key(value, fallback)

        special = {
            "diam_original": "parametrizations.diam.original",
            "canonical_area_cm2": "_canonical_area_cm2",
            "canonical_edge_resistance_ohm": "_canonical_edge_resistance_ohm",
        }
        geometry_names = (
            () if plan._multi_component_plans else plan._geometry_constant_names
        )
        for name in geometry_names:
            value = _geometry_constant_value(population, name)
            if name in getattr(population, "parametrizations", {}):
                # Computed properties have no saved effective storage. Their
                # original/config tensor bindings restore the native getter.
                continue
            if (
                name in {"canonical_area_cm2", "canonical_edge_resistance_ohm"}
                and special[name] not in reference
            ):
                # Canonical Tree geometry is an immutable CompartmentGraph
                # source, not a loadable state-dict buffer.
                continue
            self.constants[name] = key(value, special.get(name, name))
        for entry in plan._parametrization_constants:
            source = constant_source(
                _buffer_at(population, entry.module_path, entry.buffer_name),
                f"{entry.module_path + '.' if entry.module_path else ''}{entry.buffer_name}",
                population.get_submodule(entry.module_path),
                entry.buffer_name,
            )
            if source is not None:
                self.constants[entry.key] = source
        for record in plan._multi_component_plans:
            component = population.populations[record.name]
            entries = {entry.key: entry for entry in record.parametrization_constants}
            for relative, public in record.constant_sources:
                if relative in getattr(component, "parametrizations", {}):
                    continue
                if relative in record.geometry_constant_names:
                    value = _geometry_constant_value(component, relative)
                    fallback = special.get(relative, relative)
                    if (
                        relative
                        in {"canonical_area_cm2", "canonical_edge_resistance_ohm"}
                        and f"populations.{record.name}.{fallback}" not in reference
                    ):
                        continue
                else:
                    entry = entries[relative]
                    value = _buffer_at(component, entry.module_path, entry.buffer_name)
                    fallback = f"{entry.module_path + '.' if entry.module_path else ''}{entry.buffer_name}"
                path = f"populations.{record.name}.{fallback}"
                source = (
                    key(value, path)
                    if relative in record.geometry_constant_names
                    else constant_source(
                        value,
                        path,
                        component.get_submodule(entry.module_path),
                        entry.buffer_name,
                    )
                )
                if source is not None:
                    self.constants[public] = source
        for public, source, _shape, _dtype, _device in (
            *plan._cache_value_specs,
            *plan._materialization_value_specs,
        ):
            value = slots[source]
            self.constants[public] = key(value, source)
        represented = {
            id(slots[source])
            for source in (
                *self.parameters.values(),
                *self.constants.values(),
                *(source for _path, source in self.states),
            )
        }
        # Index maps and spatial coordinates participate in the frozen graph,
        # rather than the numerical restoration packet. A same-shaped edit
        # must not silently change a native topology absent from the pure plan.
        self.frozen = tuple(
            name
            for name, value in slots.items()
            if name in reference
            and name not in preserved
            and id(value) not in represented
            and (not value.is_floating_point() or name in {"x", "y", "z"})
        )
        self.default = None if bindings_only else self.normalize(cache)
        self._owned_parameter_bindings = ()
        self._owned_replays = ()

    def bind_parameter_sources(self, sources):
        """Freeze standalone promoted stimulation readers beside cache outputs."""
        winners = {self.storage_frames[name]: name for name in self.write_keys}
        writes = [
            (name, self.sources[name], self.storage_frames[name])
            for name in self.write_keys
        ]
        output_sources = tuple(
            dict.fromkeys(
                (
                    *self.parameters.values(),
                    *self.constants.values(),
                    *(source for _path, source in self.states),
                )
            )
        )
        outputs = [
            (name, self.sources[name], self.storage_frames[name])
            for name in output_sources
        ]
        storage_keys = {
            (frame[1], frame[3]) for frame in winners if frame[0] == "storage"
        }
        groups = {}
        for name, source in sources.items():
            frame = _cache_storage_frame(source)
            if frame not in winners and (
                frame[0] != "storage" or (frame[1], frame[3]) not in storage_keys
            ):
                continue
            group = groups.setdefault((frame, id(source)), [])
            group.append(name)
        bindings = []
        for index, ((frame, _identity), names) in enumerate(groups.items()):
            identifier = ("parameter", index)
            outputs.append((identifier, sources[names[0]], frame))
            bindings.append((identifier, winners.get(frame), tuple(names)))
        self._owned_replays = cache_storage_replays(writes, outputs)
        self._owned_parameter_bindings = tuple(bindings)

    def resolve(self, cache):
        """Replay local storage writes without retaining an initialization graph."""
        resolved = dict(cache)
        for replay in self._owned_replays:
            values = replay_population_cache_storage(
                tuple(cache[path] for path in replay.write_paths),
                replay.write_indices,
                replay.output_indices,
                replay.output_shapes,
                size=replay.size,
            )
            resolved.update(zip(replay.output_ids, values, strict=True))
        return resolved

    def restore_owned(self, parameters, constants, state, cache):
        resolved = self.resolve(cache)
        packet = self.restore(parameters, constants, state, resolved)
        effective = dict(packet.parameters)
        for identifier, winner, names in self._owned_parameter_bindings:
            if identifier in resolved:
                saved = resolved[identifier]
            elif winner is not None:
                saved = resolved[winner]
            else:
                continue
            reference = parameters[names[0]]
            value = restore_population_cache_values(
                {"value": reference},
                {"value": saved.expand_as(reference)},
                parameter_names=("value",),
            )["value"]
            for name in names:
                effective[name] = value
        return packet._replace(parameters=effective)

    def validate_values(self, cache):
        if torch._C._are_functorch_transforms_active():
            return
        for name in self.frozen:
            value, reference = cache[name], self._reference[name]
            equal = value == reference
            if value.is_floating_point():
                equal = equal | (torch.isnan(value) & torch.isnan(reference))
            message = f"Population cache {name!r} changes frozen topology; lower again"
            if torch.compiler.is_compiling():
                torch._assert_async(equal.all(), message)
            elif not bool(equal.all()):
                raise FunctionalizationError(message)

    def normalize(self, cache):
        if not isinstance(cache, Mapping) or set(cache) != set(self._reference):
            raise FunctionalizationError(
                "Population state cache schema changed; lower again"
            )
        from ._random_parameters import cache_metadata_tensors

        if self.plan._random_plan is not None:
            self.plan._random_plan.validate_native_cache(cache)
        converted = {
            name: cache_metadata_tensors(value, self.plan.device)
            for name, value in cache.items()
        }
        if self.plan._random_plan is not None:
            self.plan._random_plan.validate_cache(converted)
        for name, reference in self._reference.items():
            expected, spec = torch.utils._pytree.tree_flatten(reference)
            actual, other = torch.utils._pytree.tree_flatten(converted[name])
            if spec != other or any(
                not torch.is_tensor(value) or tuple(value.shape) != tuple(target.shape)
                for value, target in zip(actual, expected, strict=True)
            ):
                raise FunctionalizationError(
                    f"Population cache {name!r} has incompatible shape"
                )
        normalized = dict(restore_population_cache_values(self._reference, converted))
        self.validate_values(normalized)
        if self.plan._random_plan is not None:
            self.plan._random_plan.validate_cache(normalized)
        return normalized

    def restore(self, parameters, constants, state, cache):
        self.validate_values(cache)
        plan = self.plan
        selected = {name: parameters[name] for name in self.parameters}
        restored = restore_population_cache_values(
            selected,
            {name: cache[source] for name, source in self.parameters.items()},
            parameter_names=self.parameters,
        )
        effective = {**parameters, **restored}
        effective_constants = {
            **constants,
            **{
                name: cache[source].detach().to(constants[name]).clone()
                for name, source in self.constants.items()
            },
        }
        if plan._cache_value_specs:
            effective_constants["cache.active"] = torch.ones_like(
                constants["cache.active"]
            )
        effective_constants = plan._clear_retained_materializations(effective_constants)
        restored_state = {}
        for path, source in self.states:
            reference = _get_path(state, path)
            value = cache[source].detach().to(reference).clone()
            _put_path(restored_state, path, value)
        if plan._random_plan is not None:
            effective_constants, restored_state["random"] = (
                plan._random_plan.restore_cache(cache, effective_constants)
            )
        plan._validate_parameters(effective)
        plan._validate_constants(effective_constants)
        plan._validate_state(restored_state)
        return PopulationTensors(effective, effective_constants, restored_state, None)


__all__ = ["PopulationCachePlan"]
