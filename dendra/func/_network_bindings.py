"""Cold source bindings and identity-preserving Network tensor snapshots.

Connections observe the latest explicit Population frame. No binding resolves
through a native module during execution, including sources changed by an
earlier receive in the same network timestep.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from dendra.models.mechanisms._handler import MechanismHandler

from ._tree_paths import _get_path as _tree_value
from ._types import FunctionalizationError


def _map_containers(tree, mapping):
    """Snapshot public container structure without copying differentiable leaves."""
    if isinstance(tree, Mapping):
        return mapping(
            {key: _map_containers(value, mapping) for key, value in tree.items()}
        )
    if isinstance(tree, tuple):
        values = tuple(_map_containers(value, mapping) for value in tree)
        return type(tree)(*values) if hasattr(tree, "_fields") else values
    return tree


def _canonicalize_mapping_tree(tree):
    """Normalize public Mappings to compiler PyTrees without copying tensors.

    Standard dict/tuple packets retain their containers when no child changes.
    Mapping accessors may execute external Python; complete normalization
    before checking source, schema, or prepared-coefficient freshness.
    """
    if isinstance(tree, Mapping):
        changed = type(tree) is not dict
        values = {}
        for name, value in tree.items():
            normalized, child_changed = _canonicalize_mapping_tree(value)
            values[name] = normalized
            changed |= child_changed
        return (values if changed else tree), changed
    if isinstance(tree, tuple):
        values = []
        changed = False
        for value in tree:
            normalized, child_changed = _canonicalize_mapping_tree(value)
            values.append(normalized)
            changed |= child_changed
        if changed:
            result = type(tree)(*values) if hasattr(tree, "_fields") else tuple(values)
            return result, True
    return tree, False


def _registered_source(population, reference):
    if not isinstance(reference, str) or not reference:
        raise FunctionalizationError("Network pre_var must name a registered Tensor.")
    value = population
    owner, field = None, None
    for name in reference.split("."):
        if isinstance(value, MechanismHandler) and name in value.mechanisms:
            value = value.mechanisms[name]
            continue
        if not isinstance(value, torch.nn.Module):
            raise FunctionalizationError(
                f"Network source {reference!r} is not a registered Tensor path."
            )
        slots = {**value._modules, **value._parameters, **value._buffers}
        if name not in slots:
            raise FunctionalizationError(
                f"Network source {reference!r} uses an unstored attribute or property."
            )
        owner, field = value, name
        value = slots[name]
    if not torch.is_tensor(value) or owner is None:
        raise FunctionalizationError(
            f"Network source {reference!r} must resolve to a registered Tensor."
        )
    module_paths = {}
    for path, module in population.named_modules():
        module_paths.setdefault(id(module), path)
    path = module_paths[id(owner)]
    return f"population.{path + '.' if path else ''}{field}", value


class SourceBinding:
    """A stored Native source mapped to a current functional Population frame.

    ``prepared`` in :meth:`resolve` is the raw result of
    ``FunctionalPopulation._prepare_values``. Dynamic state/carry is read
    directly, while effective parameters, derived buffers and geometry use the
    Population's own transition mapping. Unrepresented workspaces fail cold
    rather than accidentally sampling their frozen lowering-time values.
    """

    def __init__(self, population_plan, population, reference):
        self.reference = reference
        self.slot, value = _registered_source(population, reference)
        self.shape = tuple(value.shape)
        self.dtype, self.device = value.dtype, value.device
        self._plan = population_plan
        self._kind, self._key = None, None
        for entry in population_plan._state_layout:
            slots = (
                f"population.{path + '.' if path else ''}{name}"
                for path, name in entry.mapping_slots
            )
            if self.slot in slots:
                self._kind, self._key = "state", entry.public_path
                return
        for name, slots in population_plan._transition_parameter_slots.items():
            if self.slot in slots:
                self._kind, self._key = "parameter", name
                return
        explicit_slots = {
            f"population.{entry.module_path}.{entry.buffer_name}"
            for entry in (
                *population_plan._prepared_buffers,
                *population_plan._point_area_factors,
            )
        }
        explicit_slots.update(population_plan._mechanism_dt_slots)
        explicit_slots.update(population_plan._mechanism_celsius_slots)
        explicit_slots.update(
            f"population.{name}"
            for name in population_plan._preparation.population_parameter_names
        )
        explicit_slots.update(
            f"population.{path + '.' if path else ''}{binding.buffer_name}"
            for binding in population_plan._geometry_bindings
            for path in binding.module_paths
        )
        if self.slot in explicit_slots and self.slot in population_plan._base_mapping:
            self._kind = "mapping"
            return
        raise FunctionalizationError(
            f"Network source {reference!r} has no explicit functional state, "
            "parameter, or prepared-buffer binding."
        )

    def resolve(self, parameters, prepared, state, *, _prepared_only=False):
        if self._kind == "state":
            return _tree_value(state, self._key)
        if self._kind == "parameter":
            return parameters[self._key]
        if _prepared_only:
            # Network execution admits ordinary state before graph capture.
            # Prepared sources need the canonical read-only bindings, without
            # acquiring an independent writable copy of every state slot.
            mapping, *_ = self._plan._read_only_mapping(
                parameters, prepared, isolate_read_only=False
            )
            if self._plan._poisson_plan is not None and "stimulation" in state:
                mapping.update(
                    self._plan._poisson_plan.state_mapping(state["stimulation"])
                )
        else:
            mapping, *_ = self._plan._mapping(
                parameters, prepared, state, isolate_read_only=False
            )
        value = mapping[self.slot]
        # Prepared scalar coefficients intentionally retain their logical rank;
        # ordinary native parameter buffers expose their broadcast field shape.
        # Execution may add leading singleton axes even to a native scalar.
        # Remove those views before expanding to the registered source shape.
        shape = tuple(value.shape)
        if shape == self.shape:
            return value
        extra = len(shape) - len(self.shape)
        if extra > 0 and all(size == 1 for size in shape[:extra]):
            shape = shape[extra:]
            value = value.reshape(shape)
        return value if shape == self.shape else value.expand(self.shape)


def _bind_materializer_aliases(mapping, canonical, aliases, values):
    """Clone each canonical source once and rebind its relative paths."""
    for relative, key in aliases.items():
        if key not in canonical:
            canonical[key] = values[key].clone()
        mapping[f"source.{relative}"] = canonical[key]


class AliasSnapshots:
    """Detached independent snapshots retaining genuine native Tensor ties.

    Keep one instance for an aggregate extraction. ``native`` identifies the
    source storage contract, while ``value`` supplies the component adapter's
    effective value and gradient convention. References are retained with the
    memo so temporary materializations cannot collide through Python id reuse.
    Views with different identities remain different inputs; no storage alias
    or parameterization relationship is inferred.

    Native-boundary capture opts into the native source's trainability; ordinary
    component extraction retains the adapter's existing gradient convention.
    """

    def __init__(self, *, preserve_native_grad=False):
        self._memo = {}
        self._preserve_native_grad = preserve_native_grad

    def resolve(self, native):
        """Resolve a source already represented by this extraction's owned leaf."""
        entry = self._memo.get(id(native))
        return native if entry is None else entry[2]

    def remember(self, native, copied):
        """Seed an owned invariant before visiting recurrent source aliases."""
        schema = (tuple(copied.shape), copied.dtype, copied.device)
        self._memo[id(native)] = native, schema, copied
        return copied

    def snapshot(self, native, value):
        if not torch.is_tensor(value):
            return value
        source = native if torch.is_tensor(native) else value
        requires_grad = (
            source.requires_grad if self._preserve_native_grad else value.requires_grad
        )
        identity = id(source)
        schema = (tuple(value.shape), value.dtype, value.device)
        if identity in self._memo:
            _, previous_schema, copied = self._memo[identity]
            if schema != previous_schema:
                raise FunctionalizationError(
                    "A shared Network Tensor has incompatible explicit component schemas; "
                    "use separate native tensors or a matching parameterization."
                )
            # A raw source may be a trainable input to one component and a
            # config buffer to another. Keep their common owned leaf while
            # retaining the trainability requested by either reader.
            if requires_grad and not copied.requires_grad:
                with torch.inference_mode(False):
                    copied.requires_grad_(True)
            return copied
        with torch.inference_mode(False):
            copied = value.detach().clone(memory_format=torch.preserve_format)
            copied.requires_grad_(requires_grad)
        self._memo[identity] = source, schema, copied
        return copied

    def tree(self, native, values):
        """Snapshot a tensor tree paired with optional native correspondence."""
        if torch.is_tensor(values):
            return self.snapshot(native, values)
        if isinstance(values, Mapping):
            native = native if isinstance(native, Mapping) else {}
            return {
                name: self.tree(native.get(name), value)
                for name, value in values.items()
            }
        if isinstance(values, tuple) and hasattr(values, "_fields"):
            return type(values)(
                *(
                    self.tree(
                        native.get(name)
                        if isinstance(native, Mapping)
                        else getattr(native, name, None),
                        value,
                    )
                    for name, value in zip(values._fields, values, strict=True)
                )
            )
        if isinstance(values, (tuple, list)):
            native = native if isinstance(native, (tuple, list)) else ()
            return type(values)(
                self.tree(native[index] if index < len(native) else None, value)
                for index, value in enumerate(values)
            )
        return values
