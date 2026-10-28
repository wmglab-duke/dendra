"""Cold, portable checkpoint adapters for qualified functional callbacks."""

from __future__ import annotations

import math
from dataclasses import fields
from numbers import Integral

import torch
from torch.utils import _pytree as pytree

from ._callbacks import (
    AnomalyDetector,
    FunctionalCallbackState,
    _BoundActiveAL,
    _BoundActiveALCount,
    _BoundAPCount,
    _BoundLFP,
    _BoundPopulationCallback,
    _BoundRaster,
    _BoundRecorder,
    _TensorLeafSchema,
    _TensorTreeSchema,
    _validate_tree,
)
from ._regions import _RecorderRegion
from ._types import FunctionalizationError

_FORMAT = "dendra.func.callback_state"
_VERSION = 1
_KINDS = {
    _BoundRecorder: "Recorder",
    _BoundLFP: "LFP",
    _BoundRaster: "Raster",
    _BoundAPCount: "APCount",
    _BoundActiveAL: "ActiveAL",
    _BoundActiveALCount: "ActiveALCount",
    _BoundPopulationCallback: "PopulationCallback",
    _RecorderRegion: "RecorderRegion",
}
_DTYPES = {
    str(value): value
    for value in vars(torch).values()
    if isinstance(value, torch.dtype)
}


def _cold_boundary():
    if (
        torch.compiler.is_compiling()
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    ):
        raise FunctionalizationError(
            "callback checkpoint export/restore must run outside torch.compile, "
            "torch.func and forward AD"
        )


def _tensor(value, *, copy):
    if type(value) not in (torch.Tensor, torch.nn.Parameter):
        raise FunctionalizationError(
            "callback checkpoints require ordinary Tensors; Tensor subclasses "
            "are currently unsupported"
        )
    return value.detach().clone() if copy else value


def _tensor_tree(value, *, copy, active=None):
    """Validate plain acyclic containers before using the PyTree validators."""
    if isinstance(value, torch.Tensor):
        return _tensor(value, copy=copy)
    if value is None:
        return None
    if type(value) not in (dict, list, tuple):
        raise TypeError("callback checkpoint carry must be a plain Tensor tree")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("callback checkpoint carry must not contain cycles")
    active.add(identity)
    try:
        if type(value) is dict:
            if not all(
                isinstance(key, str) if copy else type(key) is str for key in value
            ):
                raise TypeError("callback checkpoint carry keys must be strings")
            result = {
                str.__str__(key): _tensor_tree(child, copy=copy, active=active)
                for key, child in value.items()
            }
        else:
            children = [
                _tensor_tree(child, copy=copy, active=active) for child in value
            ]
            result = tuple(children) if type(value) is tuple else children
        return result if copy else value
    finally:
        active.remove(identity)


def _configuration_value(value):
    """Project only exact trusted implementations, never authored object state."""
    if isinstance(value, str):
        return str.__str__(value)
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, torch.Tensor):
        value = _tensor(value, copy=False)
        return {
            "tensor": value.detach().to(device="cpu").clone(),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if type(value) in (tuple, list):
        return [_configuration_value(child) for child in value]
    if type(value) is slice:
        components = (value.start, value.stop, value.step)
        if not all(part is None or isinstance(part, Integral) for part in components):
            raise FunctionalizationError(
                "callback checkpoints require plain integer Slice selectors"
            )
        return {"slice": [None if part is None else int(part) for part in components]}
    if type(value) is AnomalyDetector:
        return {"kind": "AnomalyDetector"}
    kind = _KINDS.get(type(value))
    if kind is None:
        raise FunctionalizationError(
            "callback checkpoints currently support built-in callbacks only; "
            "custom callbacks are unsupported"
        )
    return {
        "kind": kind,
        "fields": {
            field.name: _configuration_value(getattr(value, field.name))
            for field in fields(value)
        },
    }


def _configuration(callbacks):
    from ._network import FunctionalNetwork

    functional = callbacks._functional
    if isinstance(functional, FunctionalNetwork):
        populations = functional.populations.items()
        kind = "Network"
    else:
        populations = ((None, functional),)
        kind = "Population"
    return {
        "owner": {
            "kind": kind,
            "populations": [
                {
                    "name": None if name is None else str.__str__(name),
                    "shape": list(population.shape),
                    "dtype": str(population.dtype),
                    "device": str(population.device),
                }
                for name, population in populations
            ],
            "dt": functional.dt,
        },
        "callbacks": [
            {
                "name": str.__str__(plan.name),
                "callback": _configuration_value(plan.callback),
            }
            for plan in callbacks._plans
        ],
    }


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        if (
            left.dtype != right.dtype
            or left.device != right.device
            or left.shape != right.shape
        ):
            return False
        if torch.equal(left, right):
            return True
        if left.is_complex():
            return _same(left.real, right.real) and _same(left.imag, right.imag)
        if left.is_floating_point():
            return bool(((left == right) | (left.isnan() & right.isnan())).all())
        return False
    if type(left) is dict:
        return (
            all(type(key) is str for key in right)
            and list(left) == list(right)
            and all(_same(left[key], right[key]) for key in left)
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


def _encode_schema(schema):
    if schema is None:
        return None
    leaves = iter(schema.leaves)

    def encode(spec):
        if spec.is_leaf():
            leaf = next(leaves)
            return {
                "kind": "tensor",
                "shape": list(leaf.shape),
                "dtype": str(leaf.dtype),
                "device": str(leaf.device),
            }
        if spec.type not in (dict, tuple, list, type(None)):
            raise FunctionalizationError(
                "callback checkpoints require plain built-in tensor PyTrees"
            )
        result = {"kind": spec.type.__name__}
        if spec.type is dict:
            if not all(isinstance(key, str) for key in spec.context):
                raise TypeError("callback checkpoint schema keys must be strings")
            result["keys"] = [str.__str__(key) for key in spec.context]
        result["children"] = [encode(child) for child in spec.children()]
        return result

    result = encode(schema.spec)
    if next(leaves, None) is not None:
        raise ValueError("callback checkpoint schema has an invalid leaf count")
    return result


def _keys(value, expected, *, label):
    if type(value) is not dict:
        raise TypeError(f"{label} must be a plain dict")
    if not all(type(key) is str for key in value) or set(value) != set(expected):
        raise ValueError(f"{label} has missing or unknown keys")


def _decode_schema(value):
    if value is None:
        return None
    leaves = []
    active = set()

    def decode(node):
        if type(node) is not dict:
            raise TypeError("callback checkpoint emission schema must be a plain dict")
        identity = id(node)
        if identity in active:
            raise ValueError("callback checkpoint emission schema contains a cycle")
        active.add(identity)
        try:
            kind = node.get("kind")
            if type(kind) is not str:
                raise ValueError("callback checkpoint emission kind is invalid")
            if kind == "tensor":
                _keys(
                    node, ("kind", "shape", "dtype", "device"), label="emission schema"
                )
                shape = node["shape"]
                if type(shape) is not list or not all(
                    type(size) is int and size >= 0 for size in shape
                ):
                    raise ValueError("callback checkpoint emission shape is invalid")
                dtype = node["dtype"]
                device = node["device"]
                if type(dtype) is not str or dtype not in _DTYPES:
                    raise ValueError("callback checkpoint emission dtype is invalid")
                if type(device) is not str:
                    raise ValueError("callback checkpoint emission device is invalid")
                try:
                    device = torch.device(device)
                except (RuntimeError, ValueError) as error:
                    raise ValueError(
                        "callback checkpoint emission device is invalid"
                    ) from error
                leaves.append(_TensorLeafSchema(tuple(shape), _DTYPES[dtype], device))
                return object()
            expected = (
                ("kind", "keys", "children") if kind == "dict" else ("kind", "children")
            )
            _keys(node, expected, label="emission schema")
            if kind not in ("dict", "list", "tuple", "NoneType"):
                raise ValueError("callback checkpoint emission container is invalid")
            children = node["children"]
            if type(children) is not list:
                raise TypeError("callback checkpoint emission children must be a list")
            decoded = [decode(child) for child in children]
            if kind == "dict":
                keys = node["keys"]
                if (
                    type(keys) is not list
                    or not all(type(key) is str for key in keys)
                    or len(set(keys)) != len(keys)
                    or len(keys) != len(decoded)
                ):
                    raise ValueError("callback checkpoint emission keys are invalid")
                return dict(zip(keys, decoded, strict=True))
            if kind == "NoneType":
                if decoded:
                    raise ValueError("callback checkpoint None schema has children")
                return None
            return tuple(decoded) if kind == "tuple" else decoded
        finally:
            active.remove(identity)

    spec = pytree.tree_structure(decode(value))
    if spec.num_leaves != len(leaves):
        raise ValueError(
            "callback checkpoint emission schema has an invalid leaf count"
        )
    return _TensorTreeSchema(spec, tuple(leaves))


def _validate_emissions(callbacks, schemas):
    if type(schemas) is not tuple or len(schemas) != len(callbacks._plans):
        raise ValueError(
            "callback checkpoint emission schema count does not match the plan"
        )
    for plan, schema in zip(callbacks._plans, schemas, strict=True):
        implementation = plan.callback
        if type(implementation) is _BoundPopulationCallback:
            implementation = implementation.callback
        expected = plan.emission_schema
        if type(implementation) is _BoundRaster and schema is not None:
            expected = _TensorTreeSchema(
                plan.result_schema.spec,
                tuple(
                    _TensorLeafSchema(leaf.shape[1:], leaf.dtype, leaf.device)
                    for leaf in plan.result_schema.leaves
                ),
            )
        if schema != expected:
            raise FunctionalizationError(
                f"callback checkpoint emission schema for {plan.name!r} "
                "does not match the callback configuration"
            )


def _validate_invariants(callbacks, carries):
    for plan in callbacks._plans:
        implementation = plan.callback
        if type(implementation) is _BoundPopulationCallback:
            implementation = implementation.callback
        if type(implementation) is not _BoundLFP:
            continue
        expected = torch.stack(
            [
                value.to(device=implementation.device, dtype=implementation.dtype)
                for value in implementation.fields
            ]
        )
        if implementation.rule is None:
            expected = expected.reshape(
                expected.shape[0],
                *((1,) * implementation.padding),
                *expected.shape[1:],
            )
        if not _same(expected, carries[plan.name]):
            raise FunctionalizationError(
                f"callback checkpoint LFP lead-field carry for {plan.name!r} "
                "does not match the callback configuration"
            )


def export_state(callbacks, state):
    _cold_boundary()
    with torch.inference_mode(False), torch.no_grad():
        callbacks._validate_state(state)
        configuration = _configuration(callbacks)
        _validate_emissions(callbacks, state.emission_schemas)
        _validate_invariants(callbacks, state.carries)
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "configuration": configuration,
            "carries": _tensor_tree(state.carries, copy=True),
            "emission_schemas": [
                _encode_schema(schema) for schema in state.emission_schemas
            ],
        }


def restore_state(callbacks, packet):
    _cold_boundary()
    with torch.inference_mode(False), torch.no_grad():
        _keys(
            packet,
            ("format", "version", "configuration", "carries", "emission_schemas"),
            label="callback checkpoint",
        )
        if type(packet["format"]) is not str or packet["format"] != _FORMAT:
            raise ValueError("unsupported callback checkpoint format")
        if type(packet["version"]) is not int or packet["version"] != _VERSION:
            raise ValueError("unsupported callback checkpoint version")
        if not _same(_configuration(callbacks), packet["configuration"]):
            raise FunctionalizationError(
                "callback checkpoint configuration does not match this callback plan"
            )
        if type(packet["emission_schemas"]) is not list:
            raise TypeError("callback checkpoint emission_schemas must be a list")
        schemas = tuple(_decode_schema(value) for value in packet["emission_schemas"])
        _validate_emissions(callbacks, schemas)
        carries = packet["carries"]
        _tensor_tree(carries, copy=False)
        _keys(
            carries,
            [plan.name for plan in callbacks._plans],
            label="callback checkpoint carries",
        )
        for plan in callbacks._plans:
            _validate_tree(
                carries[plan.name],
                plan.carry_schema,
                label=f"callback checkpoint carry {plan.name!r}",
            )
        _validate_invariants(callbacks, carries)
        state = FunctionalCallbackState(
            carries=_tensor_tree(carries, copy=True),
            _token=callbacks._state_token,
            _emission_schemas=schemas,
        )
        return callbacks._validate_state(state)


__all__ = []
