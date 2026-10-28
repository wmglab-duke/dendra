"""Checked plain-data checkpoints for complete functional continuations."""

from __future__ import annotations

import types
from collections.abc import Mapping
from numbers import Integral, Real

import torch

from ._callback_checkpoints import _keys, _same
from ._continuation import Continuation
from ._continuation_codec import decode_frames, encode_frames
from ._module_traversal import _registered_modules
from ._network import FunctionalNetwork, _validate_tree
from ._population import FunctionalPopulation, _support_key_modules, _support_signature
from ._tree_paths import _get_path
from ._types import FunctionalizationError

_FORMAT = "dendra.func.continuation"
_VERSION = 1


def _cold_boundary():
    if (
        torch.compiler.is_compiling()
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    ):
        raise FunctionalizationError(
            "continuation checkpoint export/restore must run outside torch.compile, "
            "torch.func and forward AD"
        )


def _type_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _configuration_value(value, active=None):
    if isinstance(value, torch.Tensor):
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            raise FunctionalizationError(
                "continuation checkpoints require ordinary configuration Tensors"
            )
        return value.detach().to(device="cpu").clone()
    if isinstance(value, str):
        return str.__str__(value)
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return {"type": _type_name(value), "value": str(value)}
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is complex:
        return {"complex": [value.real, value.imag]}
    if isinstance(value, types.ModuleType):
        return {"module": value.__name__}
    if isinstance(value, type):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        name = f"{value.__module__}.{value.__qualname__}"
        if not isinstance(value, types.FunctionType):
            return {"function": name}
        active = set() if active is None else active
        if id(value) in active:
            return {"function_reference": name}
        active.add(id(value))
        try:
            from ._continuous_transform import _global_values

            globals_used = _global_values(value)
            return {
                "function": name,
                "defaults": _configuration_value(value.__defaults__, active),
                "kwdefaults": _configuration_value(value.__kwdefaults__, active),
                "closure": [
                    _configuration_value(cell.cell_contents, active)
                    for cell in value.__closure__ or ()
                ],
                "globals": _configuration_value(
                    {name: globals_used[name] for name in sorted(globals_used)}, active
                ),
            }
        finally:
            active.remove(id(value))
    if isinstance(value, (tuple, list)):
        return [_configuration_value(child, active) for child in value]
    if isinstance(value, Mapping):
        return {
            str.__str__(key): _configuration_value(child, active)
            for key, child in value.items()
        }
    raise FunctionalizationError(
        "continuation checkpoints require plain model configuration metadata"
    )


def _population_layout(population):
    layout = {
        "shape": list(population.shape),
        "core_shape": list(population.core_shape()),
        "placements": [
            [path, _configuration_value(_support_signature(module))]
            for path, module in _support_key_modules(population)
        ],
    }
    graph = getattr(population, "compartment_graph", None)
    if graph is not None:
        layout["parents"] = [int(parent) for parent in graph.topology.parent_index]
    fingerprint = getattr(
        population, "_canonical_morphology_fingerprint_reference", None
    )
    if fingerprint is not None:
        layout["morphology"] = _configuration_value(fingerprint)
    if getattr(population, "populations", None) is not None:
        layout["components"] = [
            [str.__str__(name), _population_layout(component)]
            for name, component in population.populations.items()
        ]
    return layout


def _transform_configuration(plan):
    transform = plan._transform
    if not isinstance(transform, torch.nn.Module):
        return _configuration_value(transform)
    modules = list(_registered_modules(transform))
    paths = {id(module): path for path, module in modules}
    return {
        "modules": [[path, _type_name(module)] for path, module in modules],
        "attributes": [
            [paths[id(owner)], name, _configuration_value(value)]
            for owner, name, value, kind in plan._transform_plan._guard_attributes
            if kind == "config" and id(owner) in paths
        ],
    }


def _population_owner(name, plan):
    population = plan._population
    return {
        "name": None if name is None else str.__str__(name),
        "shape": list(plan.shape),
        "dtype": str(plan.dtype),
        "device": str(plan.device),
        "training": bool(population.training),
        "topology": plan._topology_kind,
        "layout": _population_layout(population),
        "modules": [
            [path, _type_name(module)]
            for path, module in _registered_modules(population)
        ],
        "random_streams": _configuration_value(
            {} if plan._random_plan is None else plan._random_plan.streams
        ),
        "borrowed_streams": _configuration_value(plan._stimulation_rng_plan.borrowed),
    }


def _connection_owner(functional, collection, name, plan):
    fields = (
        "_pre_var",
        "_dt",
        "_pre_idx",
        "_post_idx",
        "_delay_steps",
        "_depth",
        "_mode",
        "_runtime",
        "_history_mode",
        "_history_metadata",
        "_track_events",
        "_skip_thresholding",
        "_threshold",
        "_nan_mask",
        "_apply_masking",
        "_diff_spiking",
        "_diff_delays",
        "_taps",
        "_sigma",
        "_tau",
        "_diff_scheduled_times",
        "_sched_width",
        "input",
        "reduce",
    )
    binding = functional._bindings[collection, name]
    return {
        "name": str.__str__(name),
        "type": _type_name(plan._source),
        "modules": [
            [path, _type_name(module)]
            for path, module in _registered_modules(plan._source)
        ],
        "source": None
        if binding is None
        else _configuration_value([binding[0], binding[1].reference]),
        "target": _configuration_value(functional._targets[collection, name]),
        "transform": _transform_configuration(plan)
        if collection == "continuous_synapses"
        else None,
        "schedule": _schedule_configuration(plan._schedule)
        if collection == "synapses"
        else None,
        "configuration": {
            key: _configuration_value(getattr(plan, key))
            for key in fields
            if hasattr(plan, key)
        },
    }


def _schedule_configuration(schedule):
    return {
        field: _configuration_value(getattr(schedule, field))
        for field in ("indices", "time_idx", "amplitude_idx")
    }


def _owner(functional):
    network = isinstance(functional, FunctionalNetwork)
    populations = functional.populations.items() if network else ((None, functional),)
    result = {
        "kind": "Network" if network else "Population",
        "dt": functional.dt,
        "populations": [_population_owner(name, plan) for name, plan in populations],
    }
    if network:
        result["type"] = _type_name(functional._source)
        result["training"] = functional.training
        result["rng_ownership"] = _configuration_value(functional._rng_plan._topology)
        result["stimulation_streams"] = _configuration_value(
            functional._stimulation_plan.local
        )
        result["borrowed_streams"] = _configuration_value(
            functional._stimulation_rng_plan.borrowed
        )
        result["connections"] = {
            collection: [
                _connection_owner(functional, collection, name, plan)
                for name, plan in getattr(functional, collection).items()
            ]
            for collection in ("synapses", "continuous_synapses")
        }
        stimulus = functional.netstim
        result["netstim"] = (
            None
            if stimulus is None
            else {
                "type": _type_name(stimulus._source),
                "modules": [
                    [path, _type_name(module)]
                    for path, module in _registered_modules(stimulus._source)
                ],
                "shape": list(stimulus.shape),
                "dtype": str(stimulus.dtype),
                "device": str(stimulus.device),
                "training": stimulus._training,
                "schedule_dt": _configuration_value(stimulus._dt),
                "mode": _configuration_value(stimulus._schedule_mode),
                "schedule": _schedule_configuration(stimulus._schedule),
            }
        )
    return result


def _validate_frames(functional, continuation):
    functional._validate_parameters(continuation.parameters)
    if isinstance(functional, FunctionalNetwork):
        _validate_tree(continuation.constants, functional._constant_schema, "constants")
        for name, plan in functional.populations.items():
            plan._validate_constants(continuation.constants["populations"][name])
        for path, readers in functional._parameter_constant_aliases:
            value = _get_path(continuation.parameters, path)
            if any(
                _get_path(continuation.constants, reader) is not value
                for reader in readers
            ):
                raise FunctionalizationError(
                    "continuation constants must preserve their shared parameter aliases"
                )
    else:
        functional._validate_constants(continuation.constants)
    functional._validate_state(continuation.state)


def _callback_packet(callbacks, state, *, owner="this runner"):
    if callbacks is None:
        if state is not None:
            raise FunctionalizationError(
                f"continuation callback state does not match {owner}'s callbacks"
            )
        return None
    if state is None:
        raise FunctionalizationError(
            f"continuation callback state is required for {owner}'s callbacks"
        )
    return callbacks.export_state(state)


def _encode_continuation(functional, continuation, callbacks):
    return {
        "format": _FORMAT,
        "version": _VERSION,
        "owner": _owner(functional),
        "frames": encode_frames(
            {
                "state": continuation.state,
                "parameters": continuation.parameters,
                "constants": continuation.constants,
            }
        ),
        "callbacks": callbacks,
    }


def _validate_packet(packet):
    _keys(
        packet,
        ("format", "version", "owner", "frames", "callbacks"),
        label="continuation checkpoint",
    )
    if type(packet["format"]) is not str or packet["format"] != _FORMAT:
        raise ValueError("unsupported continuation checkpoint format")
    if type(packet["version"]) is not int or packet["version"] != _VERSION:
        raise ValueError("unsupported continuation checkpoint version")


def _validate_owner(functional, packet, *, owner):
    functional._validate_source()
    if not _same(_owner(functional), packet["owner"]):
        raise FunctionalizationError(
            f"continuation checkpoint owner configuration does not match {owner}"
        )


def _validate_callback_presence(callbacks, packet, *, owner):
    if (callbacks is None) != (packet["callbacks"] is None):
        raise FunctionalizationError(
            f"continuation checkpoint callbacks do not match {owner}"
        )


def _decode_continuation(functional, packet, callbacks, templates):
    frames = decode_frames(packet["frames"], templates)
    callback_state = (
        None if callbacks is None else callbacks.restore_state(packet["callbacks"])
    )
    continuation = Continuation(callback_state=callback_state, **frames)
    _validate_frames(functional, continuation)
    return continuation


def _validate_functional(functional):
    if type(functional) not in (FunctionalPopulation, FunctionalNetwork):
        raise TypeError(
            "functional must be a FunctionalPopulation or FunctionalNetwork"
        )


def _validate_callbacks(functional, callbacks):
    if callbacks is not None:
        from ._callbacks import FunctionalCallbacks

        if not isinstance(callbacks, FunctionalCallbacks):
            raise TypeError(
                "callbacks must be a FunctionalCallbacks collection or None"
            )
        callbacks._validate_owner(functional)


def export_continuation(functional, continuation, *, callbacks=None):
    """Export a portable continuation for a lowered Population or Network.

    ``functional`` is the lowered plan returned by ``make_functional``.
    Supply its bound ``FunctionalCallbacks`` collection when callback carry is
    present. The versioned plain data/Tensor packet supports ``torch.save`` and
    ``torch.load(..., weights_only=True)`` and the existing runner methods.

    Tensor identity ties and trainability are preserved as owned detached leaves;
    upstream graphs, recording history, input/draw tapes and optimizer state are
    excluded. Distinct views become independent snapshots. Export runs outside
    compilation, function transforms and forward AD.
    """
    _cold_boundary()
    if not isinstance(continuation, Continuation):
        raise TypeError("export_continuation requires a Continuation")
    _validate_functional(functional)
    _validate_callbacks(functional, callbacks)
    with torch.inference_mode(False), torch.no_grad():
        functional._validate_source()
        _validate_frames(functional, continuation)
        callback_packet = _callback_packet(
            callbacks, continuation.callback_state, owner="this functional plan"
        )
        return _encode_continuation(functional, continuation, callback_packet)


def restore_continuation(functional, packet, *, callbacks=None):
    """Restore portable data against a compatible lowered Population or Network.

    The existing receiving plan supplies trusted shape, dtype, device and
    container metadata. Its model configuration and explicitly supplied callbacks
    must match the checkpoint. Restore does not extract native values, initialize,
    prepare, sample, execute or change plan defaults.

    The returned ``Continuation`` owns detached ordinary Tensor leaves with saved
    identity ties and trainability. Bind those leaves before resuming with
    ``dn.func.run``::

        bound = kernel.bind(restored.parameters, constants=restored.constants)
        state, aux = dn.func.run(bound, restored, steps=100, callbacks=callbacks)

    Saved retained initialization values are independent effective sources after
    restore; their former dependencies on raw parameters are not reconstructed.
    """
    _cold_boundary()
    with torch.inference_mode(False), torch.no_grad():
        _validate_packet(packet)
        _validate_functional(functional)
        _validate_callbacks(functional, callbacks)
        _validate_owner(functional, packet, owner="this functional plan")
        _validate_callback_presence(callbacks, packet, owner="this functional plan")
        from ._continuation_schema import _receiving_frames

        return _decode_continuation(
            functional, packet, callbacks, _receiving_frames(functional)
        )


def _export_runner_continuation(runner, continuation):
    _cold_boundary()
    if not isinstance(continuation, Continuation):
        raise TypeError("export_continuation requires a Continuation")
    with torch.inference_mode(False), torch.no_grad():
        functional = runner.functional
        functional._validate_source()
        _validate_frames(functional, continuation)
        callbacks = _callback_packet(runner.callbacks, continuation.callback_state)
        return _encode_continuation(functional, continuation, callbacks)


def _restore_runner_continuation(runner, packet):
    _cold_boundary()
    with torch.inference_mode(False), torch.no_grad():
        _validate_packet(packet)
        functional = runner.functional
        _validate_owner(functional, packet, owner="this runner")
        _validate_callback_presence(runner.callbacks, packet, owner="this runner")
        return _decode_continuation(
            functional,
            packet,
            runner.callbacks,
            {
                "state": runner.initial_state,
                "parameters": runner.parameters,
                "constants": runner.constants,
            },
        )


__all__ = ["export_continuation", "restore_continuation"]
