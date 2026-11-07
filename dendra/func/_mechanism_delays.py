"""Frozen delay-history contracts over the shared Mechanism helpers."""

from __future__ import annotations

import inspect
from types import MethodType

import torch

from dendra.models.mechanisms import Mechanism, _delays
from dendra.models.mechanisms._handler import (
    _delay_buffer_shape,
    _validate_delayed_state_specs,
)

from ._types import FunctionalizationError

_HELPER_NAMES = (
    "register_delayed_state",
    "register_delayed_states",
    "reset_delayed_states",
    "delayed_state",
    "delayed_states",
    "delayed_state_values",
    "delayed_states_values",
    "initial_delay_values",
    "_delay_values_spec",
    "_refresh_delay_history_layout",
    "_sync_delay_history_shapes",
    "_delayed_state_shift",
    "_delayed_state_circular",
    "_delayed_states_steps",
    "_delayed_states_gather",
    "_delayed_states_zero_mask",
    "_delayed_states_shift",
    "_delayed_states_circular",
)
_PURE_HELPER_NAMES = (
    "delayed_state_values",
    "delayed_states_values",
    "initial_delay_values",
)
# These are captured once after the native module has finished importing. The
# pure helpers and native adapters share their numerical implementation; no
# functional copy is installed onto the model.
_CANONICAL_HELPERS = {
    name: inspect.getattr_static(Mechanism, name) for name in _HELPER_NAMES
}
_CANONICAL_CODES = {
    name: function.__code__ for name, function in _CANONICAL_HELPERS.items()
}
_NUMERICAL_HELPERS = {
    name: value for name, value in vars(_delays).items() if inspect.isfunction(value)
}
_NUMERICAL_CODES = {name: value.__code__ for name, value in _NUMERICAL_HELPERS.items()}
_IMPORTED_NUMERICAL_HELPERS = frozenset(
    (
        "resolve_delay_mode",
        "shifted_history_values",
        "gather_history_values",
        "circular_history_plan",
        "delay_history_values",
        "initial_delay_history_values",
    )
)
_OVERWRITE_HISTORY = _delays._OverwriteHistory
_OVERWRITE_METHODS = {
    name: inspect.getattr_static(_OVERWRITE_HISTORY, name)
    for name in ("forward", "setup_context", "backward", "jvp")
}
_OVERWRITE_CODES = {
    name: getattr(method, "__func__", method).__code__
    for name, method in _OVERWRITE_METHODS.items()
}
_RAW_OVERWRITE_HISTORY = _delays._raw_overwrite_history
_RAW_CALLBACK_FIELDS = (
    "_init_fn",
    "_abstract_fn",
    "_setup_context_fn",
    "_backward_fn",
    "_vmap_fn",
)
_SINGLE_FIELDS = frozenset(
    ("buffer", "pointer", "steps", "depth", "axis", "value_shape", "mode")
)
_MULTI_FIELDS = _SINGLE_FIELDS | {
    "steps_buffer",
    "stream_axis",
    "value_stream_axis",
    "n_streams",
    "batched",
    "has_zero_delay",
    "all_zero_delay",
}


def mechanism_delay_owners(population, owners=None):
    """Return registered histories in canonical handler owner order."""
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
            type(getattr(owner, "_delayed_state_specs", None)) is not dict
            or len(owner._delayed_state_specs) != 0
        )
    )
    if not registered:
        return ()
    paths = {id(owner): path for path, owner in population.named_modules()}
    return tuple((paths[id(owner)], owner) for owner in registered)


def mechanism_delay_state_slots(owner):
    """Yield history queues and pointers, including undeclared CARRY storage."""
    seen = set()
    for _name, spec in sorted(getattr(owner, "_delayed_state_specs", {}).items()):
        for field in ("buffer", "pointer"):
            name = spec[field]
            if name not in seen:
                seen.add(name)
                yield name, owner._buffers.get(name)


def mechanism_delay_constant_slots(owner):
    """Yield registered fixed multistream delay vectors relative to an owner."""
    seen = set()
    for _name, spec in sorted(getattr(owner, "_delayed_state_specs", {}).items()):
        name = spec.get("steps_buffer")
        if name is not None and name not in seen:
            seen.add(name)
            yield name, owner._buffers.get(name)


def mechanism_delay_helper(owner, method):
    """Recognize a pure returned-output helper for the authored-code audit."""
    if getattr(method, "__self__", None) is not owner:
        return None
    function = getattr(method, "__func__", None)
    for name in _PURE_HELPER_NAMES:
        if (
            function is _CANONICAL_HELPERS[name]
            and function.__code__ is _CANONICAL_CODES[name]
            and name not in vars(owner)
            and inspect.getattr_static(type(owner), name) is function
        ):
            return name
    return None


def _literal(value):
    if type(value) in (str, bool, int, float, type(None)):
        return value
    if type(value) in (tuple, list):
        return tuple(_literal(item) for item in value)
    # Guard fingerprinting must not execute replacement metadata's __eq__,
    # __bool__, or iterator while deciding whether the source is still valid.
    return "nonliteral", id(type(value)), id(value)


def _raw_history_signature():
    """Guard opaque FX-node implementations as well as the Python entry point."""
    raw = _delays._raw_overwrite_history
    if id(raw) != id(_RAW_OVERWRITE_HISTORY):
        return "replaced-operator", id(raw)

    def callback(function):
        return id(function), id(getattr(function, "__code__", None))

    return (
        id(raw),
        id(type(raw)),
        callback(type(raw).__call__),
        tuple(
            (name, _literal(getattr(raw, name)))
            for name in ("_namespace", "_name", "_schema", "_is_inplace", "_is_out")
        ),
        id(raw._opoverload),
        tuple((name, callback(getattr(raw, name))) for name in _RAW_CALLBACK_FIELDS),
        tuple(
            (device, callback(function), id(function.__wrapped__.__code__))
            for device, function in raw._backend_fns.items()
        ),
        tuple(
            (id(key), callback(function))
            for key, function in raw._torch_dispatch_fns.items()
        ),
        tuple((device, id(dtype)) for device, dtype in raw._autocast_dtype.items()),
        tuple(raw._disabled_kernel),
    )


_RAW_HISTORY_SIGNATURE = _raw_history_signature()


def _tensor_layout(value):
    if not torch.is_tensor(value):
        return "not-tensor", id(type(value)), id(value)
    return tuple(value.shape), value.dtype, value.device, value.layout


def _metadata_failures(owner):
    specs = owner._delayed_state_specs
    if type(specs) is not dict or any(type(name) is not str for name in specs):
        return (
            "Mechanism delayed-state declarations require a plain string-keyed mapping",
        )
    for name, spec in specs.items():
        label = f"Mechanism delayed state {name!r}"
        if type(spec) is not dict:
            return (f"{label} requires plain literal metadata",)
        if any(type(field) is not str for field in spec):
            return (f"{label} requires string metadata fields",)
        batched = spec.get("batched", False)
        if type(batched) is not bool:
            return (f"{label} requires a boolean batched flag",)
        if set(spec) != (_MULTI_FIELDS if batched else _SINGLE_FIELDS):
            return (f"{label} has noncanonical metadata fields",)
        names = (
            ("buffer", "pointer", "steps_buffer") if batched else ("buffer", "pointer")
        )
        if any(type(spec[field]) is not str for field in names):
            return (f"{label} requires string buffer names",)
        integers = ("steps", "depth", "axis") + (
            ("stream_axis", "value_stream_axis", "n_streams") if batched else ()
        )
        if any(type(spec[field]) is not int for field in integers):
            return (f"{label} requires literal integer dimensions and delays",)
        if (
            type(spec["value_shape"]) not in (tuple, list)
            or any(type(size) is not int for size in spec["value_shape"])
            or type(spec["mode"]) is not str
        ):
            return (f"{label} requires a literal payload shape and mode",)
        if batched and any(
            type(spec[field]) is not bool
            for field in ("has_zero_delay", "all_zero_delay")
        ):
            return (f"{label} requires literal zero-delay flags",)
    return ()


def mechanism_delay_failures(owner):
    """Fail closed before evaluating malformed registrations or custom adapters."""
    if type(owner._delayed_state_specs) is dict and not owner._delayed_state_specs:
        return ()
    failures = list(_metadata_failures(owner))
    if failures:
        return tuple(failures)
    history_names = tuple(
        dict.fromkeys(
            name
            for spec in owner._delayed_state_specs.values()
            for name in (spec["buffer"], spec["pointer"])
        )
    )
    frame_names = getattr(owner, "_transition_input_names", None)
    if (
        type(getattr(owner, "_delay_history_buffer_names", None)) is not tuple
        or any(type(name) is not str for name in owner._delay_history_buffer_names)
        or owner._delay_history_buffer_names != history_names
        or type(frame_names) is not tuple
        or any(type(name) is not str for name in frame_names)
        or len(set(frame_names)) != len(frame_names)
        or any(name not in frame_names for name in history_names)
    ):
        failures.append(
            "Mechanism delayed-state registration has a stale history value-frame layout"
        )
    if type(owner.training) is not bool:
        failures.append(
            "Mechanism delayed-state execution requires a boolean training policy"
        )
    for name, canonical in _CANONICAL_HELPERS.items():
        if (
            name in vars(owner)
            or inspect.getattr_static(type(owner), name) is not canonical
            or canonical.__code__ is not _CANONICAL_CODES[name]
        ):
            failures.append(
                f"Mechanism delayed-state execution requires canonical {name}"
            )
    mechanism_globals = _CANONICAL_HELPERS["delayed_state_values"].__globals__
    if (
        any(
            getattr(_delays, name, None) is not canonical
            or canonical.__code__ is not _NUMERICAL_CODES[name]
            or (
                name in _IMPORTED_NUMERICAL_HELPERS
                and mechanism_globals.get(name) is not canonical
            )
            for name, canonical in _NUMERICAL_HELPERS.items()
        )
        or _delays.torch is not torch
        or mechanism_globals.get("torch") is not torch
    ):
        failures.append(
            "Mechanism delayed-state execution requires canonical shared history evaluation"
        )
    if (
        _delays._OverwriteHistory is not _OVERWRITE_HISTORY
        or _OVERWRITE_HISTORY.generate_vmap_rule is not True
        or any(
            inspect.getattr_static(_OVERWRITE_HISTORY, name) is not canonical
            or getattr(canonical, "__func__", canonical).__code__
            is not _OVERWRITE_CODES[name]
            for name, canonical in _OVERWRITE_METHODS.items()
        )
    ):
        failures.append(
            "Mechanism delayed-state execution requires canonical history gradient evaluation"
        )
    if _raw_history_signature() != _RAW_HISTORY_SIGNATURE:
        failures.append(
            "Mechanism delayed-state execution requires canonical opaque history evaluation"
        )
    try:
        specs = _validate_delayed_state_specs(
            owner, owner._delayed_state_specs, label="Mechanism"
        )
    except (TypeError, ValueError, KeyError) as error:
        failures.append(str(error))
        return tuple(failures)
    for name, spec in specs.items():
        label = f"Mechanism delayed state {name!r}"
        tensor_names = (spec["buffer"], spec["pointer"]) + (
            (spec["steps_buffer"],) if spec.get("batched", False) else ()
        )
        if len(set(tensor_names)) != len(tensor_names) or any(
            tensor_name not in owner._buffers
            or not torch.is_tensor(owner._buffers[tensor_name])
            or owner._buffers[tensor_name].layout is not torch.strided
            or tensor_name in owner._parameters
            for tensor_name in tensor_names
        ):
            failures.append(f"{label} requires distinct registered Tensor buffer slots")
            continue
        queue, pointer = (
            owner._buffers[spec[field]] for field in ("buffer", "pointer")
        )
        if tuple(queue.shape) != _delay_buffer_shape(spec):
            failures.append(
                f"{label} history shape differs from its registered capacity"
            )
        if (
            pointer.shape != torch.Size([])
            or pointer.dtype is not torch.long
            or pointer.device != queue.device
            or pointer.requires_grad
        ):
            failures.append(
                f"{label} requires a scalar long pointer on the queue device"
            )
        elif not 0 <= int(pointer.detach().cpu().item()) < spec["depth"]:
            failures.append(f"{label} pointer is outside its registered capacity")
        if spec.get("batched", False):
            value_shape = spec["value_shape"]
            value_axis = spec["value_stream_axis"]
            if (
                value_axis >= len(value_shape)
                or value_shape[value_axis] != spec["n_streams"]
                or spec["stream_axis"] != value_axis + (spec["axis"] <= value_axis)
            ):
                failures.append(f"{label} has inconsistent stream axes")
            steps = owner._buffers[spec["steps_buffer"]]
            if (
                tuple(steps.shape) != (spec["n_streams"],)
                or steps.dtype is not torch.long
                or steps.device != queue.device
                or steps.requires_grad
            ):
                failures.append(f"{label} requires a registered long delay vector")
                continue
            delays = tuple(int(value) for value in steps.detach().cpu().tolist())
            if (
                min(delays, default=0) < 0
                or max(delays, default=0) != spec["steps"]
                or any(delay == 0 for delay in delays) != spec["has_zero_delay"]
                or all(delay == 0 for delay in delays) != spec["all_zero_delay"]
            ):
                failures.append(
                    f"{label} delay vector differs from its registered metadata"
                )
    return tuple(failures)


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
        if not _metadata_failures(owner)
        for name, value in (
            *mechanism_delay_state_slots(owner),
            *mechanism_delay_constant_slots(owner),
        )
    )


def network_mechanism_delay_alias_signature(network):
    """Include aliases crossing population and connection owners."""
    owners = tuple(
        (f"populations.{name}.{path}", owner)
        for name, population in network.populations.items()
        for path, owner in mechanism_delay_owners(population)
    )
    return _tensor_alias_signature(network, owners)


def _contract(population):
    owners = mechanism_delay_owners(population)
    records = []
    for path, owner in owners:
        failures = mechanism_delay_failures(owner)
        if failures:
            raise FunctionalizationError("; ".join(failures))
        records.append(
            (
                path,
                _literal(owner.training),
                _literal(owner._delay_history_buffer_names),
                _literal(owner._transition_input_names),
                tuple(
                    (
                        name,
                        tuple(
                            (field, _literal(value))
                            for field, value in sorted(spec.items())
                        ),
                    )
                    for name, spec in sorted(owner._delayed_state_specs.items())
                ),
                tuple(
                    (name, _tensor_layout(value))
                    for name, value in (
                        *mechanism_delay_state_slots(owner),
                        *mechanism_delay_constant_slots(owner),
                    )
                ),
            )
        )
    return tuple(records), _tensor_alias_signature(population, owners)


class MechanismDelayPlan:
    """Frozen fixed-capacity registrations; history tensors remain explicit carry."""

    def __init__(self, population):
        self.population = population
        self.contract = _contract(population)
        self.constant_entries = tuple(
            (path, name, tuple(value.detach().cpu().tolist()))
            for path, owner in mechanism_delay_owners(population)
            for name, value in mechanism_delay_constant_slots(owner)
        )
        self._constant_schemas = tuple(
            (
                path,
                spec["steps_buffer"],
                spec["steps"],
                spec["has_zero_delay"],
                spec["all_zero_delay"],
            )
            for path, owner in mechanism_delay_owners(population)
            for spec in owner._delayed_state_specs.values()
            if spec.get("batched", False)
        )
        self._source_bindings = self.source_binding_signature()

    def validate_population(self, population, *, source=False):
        if _contract(population) != self.contract:
            raise FunctionalizationError(
                "Mechanism delayed-state registration, delay vector, layout, or aliases changed; lower again"
            )
        if (
            source
            and self.source_binding_signature(population) != self._source_bindings
        ):
            raise FunctionalizationError(
                "Mechanism delayed-state helper bindings changed; lower again"
            )
        if source:
            modules = dict(population.named_modules())
            if any(
                tuple(modules[path]._buffers[name].detach().cpu().tolist()) != expected
                for path, name, expected in self.constant_entries
            ):
                raise FunctionalizationError(
                    "Mechanism delayed-state source delay vector changed; lower again"
                )

    def validate_source(self):
        self.validate_population(self.population, source=True)

    def source_binding_signature(self, population=None):
        """Expose literal registrations and callable identities to Dynamo guards."""
        population = self.population if population is None else population

        def binding(function):
            function = (
                function.__func__ if isinstance(function, MethodType) else function
            )
            return id(function), id(getattr(function, "__code__", None))

        owners = mechanism_delay_owners(population)
        return (
            tuple(
                (
                    path,
                    _literal(owner.training),
                    _literal(owner._delay_history_buffer_names),
                    _literal(owner._transition_input_names),
                    tuple(
                        (
                            _literal(name),
                            tuple(
                                (_literal(field), _literal(value))
                                for field, value in spec.items()
                            ),
                        )
                        if type(spec) is dict
                        else (_literal(name), _literal(spec))
                        for name, spec in owner._delayed_state_specs.items()
                    )
                    if not _metadata_failures(owner)
                    else ("invalid-metadata", _literal(owner._delayed_state_specs)),
                    tuple(
                        (name, _tensor_layout(value))
                        for name, value in (
                            *mechanism_delay_state_slots(owner),
                            *mechanism_delay_constant_slots(owner),
                        )
                    )
                    if not _metadata_failures(owner)
                    else (),
                    tuple(
                        (
                            name,
                            binding(getattr(type(owner), name)),
                            binding(getattr(owner, name)),
                        )
                        for name in _HELPER_NAMES
                    ),
                    tuple(
                        (
                            name,
                            binding(getattr(_delays, name, None)),
                            binding(
                                _CANONICAL_HELPERS[
                                    "delayed_state_values"
                                ].__globals__.get(name)
                            ),
                        )
                        for name in _NUMERICAL_HELPERS
                    ),
                    id(_delays.torch),
                    id(
                        _CANONICAL_HELPERS["delayed_state_values"].__globals__.get(
                            "torch"
                        )
                    ),
                    id(_delays._OverwriteHistory),
                    id(_delays._OverwriteHistory.generate_vmap_rule),
                    tuple(
                        (
                            name,
                            id(getattr(_delays._OverwriteHistory, name)),
                            id(getattr(_delays._OverwriteHistory, name).__code__),
                        )
                        for name in _OVERWRITE_METHODS
                    ),
                    _raw_history_signature(),
                )
                for path, owner in owners
            ),
            _tensor_alias_signature(population, owners),
        )

    def validate_source_tensor_values(self):
        """Retain fixed integer-routing checks inside warmed compiled graphs."""
        if not self.constant_entries:
            return
        modules = dict(self.population.named_modules())
        for path, name, expected in self.constant_entries:
            value = modules[path]._buffers[name]
            torch._check_tensor_all(
                value == value.new_tensor(expected),
                "Mechanism delayed-state delay vector changed; lower again",
            )

    def validate_constant_values(self, values):
        """Check explicit delay values against the native checkpoint schema."""
        for path, name, maximum, has_zero, all_zero in self._constant_schemas:
            value = values[f"{path}.{name}" if path else name]
            zero = value == 0
            torch._check_tensor_all(
                (value >= 0).all()
                & (value.amax() == maximum)
                & (zero.any() == has_zero)
                & (zero.all() == all_zero),
                "Mechanism delayed-state delay values must preserve registered capacity and zero-delay flags",
            )


__all__ = []
