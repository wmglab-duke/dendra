"""Plain-data encoding for portable continuation tensor frames.

Tensor references preserve object-identity ties across complete frames. Distinct
Tensor objects, including views of one storage, remain distinct owned snapshots;
no physical-storage alias or upstream computation graph is reconstructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ._schedules import TensorScheduleTensors
from ._types import FunctionalizationError

_NAMED_TUPLES = (TensorScheduleTensors,)
_SCALARS = (bool, int, float, str)


@dataclass(frozen=True)
class _TensorDescriptor:
    """Receiving Tensor metadata supplied by a trusted lowered plan."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def _keys(value, expected, *, label):
    if type(value) is not dict:
        raise TypeError(f"{label} must be a plain dict")
    if not all(type(key) is str for key in value) or set(value) != set(expected):
        raise ValueError(f"{label} has missing or unknown keys")


def _ordinary_tensor(value, *, packet=False):
    allowed = (torch.Tensor,) if packet else (torch.Tensor, torch.nn.Parameter)
    if type(value) not in allowed:
        raise FunctionalizationError(
            "continuation checkpoints require ordinary Tensors; Tensor subclasses "
            "are currently unsupported"
        )
    if value.layout is not torch.strided:
        raise FunctionalizationError(
            "continuation checkpoints require dense strided Tensors"
        )
    return value


def _owned_tensor(value, requires_grad):
    result = value.detach().clone(memory_format=torch.preserve_format)
    result.requires_grad_(requires_grad)
    return result


def _string_keys(value, *, label):
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    keys = [str.__str__(key) for key in value]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} keys must be unique plain strings")
    return keys


def encode_frames(frames):
    """Own detached ordinary Tensor leaves and encode acyclic frame containers.

    All leaf trainability flags are retained on new leaf Tensors, including
    flags shared across parameters, constants and state. The outer checkpoint
    adapter owns format/version and model compatibility metadata.
    """
    tensors = []
    flags = []
    identities = {}
    active = set()

    def encode(value):
        if isinstance(value, torch.Tensor):
            _ordinary_tensor(value)
            identity = id(value)
            if identity not in identities:
                index = len(tensors)
                identities[identity] = (value, index)
                flags.append(value.requires_grad)
                tensors.append(_owned_tensor(value, value.requires_grad))
            return {"kind": "tensor", "index": identities[identity][1]}
        if value is None:
            return {"kind": "none"}
        if isinstance(value, str):
            return {"kind": "scalar", "value": str.__str__(value)}
        if type(value) in _SCALARS:
            return {"kind": "scalar", "value": value}
        if type(value) not in (dict, list, tuple, *_NAMED_TUPLES):
            raise FunctionalizationError(
                "continuation checkpoints require plain containers or trusted "
                "TensorScheduleTensors packets; custom containers are unsupported"
            )
        identity = id(value)
        if identity in active:
            raise ValueError("continuation checkpoint frames must not contain cycles")
        active.add(identity)
        try:
            if type(value) is dict:
                return {
                    "kind": "dict",
                    "keys": _string_keys(value, label="continuation frame"),
                    "children": [encode(child) for child in value.values()],
                }
            result = {
                "kind": "list" if type(value) is list else "tuple",
                "children": [encode(child) for child in value],
            }
            if type(value) in _NAMED_TUPLES:
                result["kind"] = "namedtuple"
                result["fields"] = list(type(value)._fields)
            return result
        finally:
            active.remove(identity)

    with torch.inference_mode(False), torch.no_grad():
        tree = encode(frames)
    return {"tree": tree, "tensors": tensors, "requires_grad": flags}


def decode_frames(packet, templates):
    """Validate plain data against trusted receiving frames and own fresh leaves.

    Container classes and Tensor descriptors come only from receiving templates.
    Tensor values and trainability can differ from defaults; shape, dtype and
    device cannot. Shared references yield one owned Tensor per table entry.
    """
    _keys(packet, ("tree", "tensors", "requires_grad"), label="continuation frames")
    tensors, flags = packet["tensors"], packet["requires_grad"]
    if type(tensors) is not list or type(flags) is not list:
        raise TypeError("continuation tensors and requires_grad must be plain lists")
    if len(tensors) != len(flags):
        raise ValueError("continuation Tensor and trainability counts do not match")
    if len({id(value) for value in tensors}) != len(tensors):
        raise ValueError("continuation Tensor table must not contain duplicate entries")
    for value, flag in zip(tensors, flags, strict=True):
        _ordinary_tensor(value, packet=True)
        if type(flag) is not bool:
            raise TypeError("continuation requires_grad flags must be booleans")
        if value.requires_grad is not flag:
            raise ValueError("continuation Tensor trainability metadata does not match")
    copies = {}
    active = set()

    def decode(node, template, path):
        if type(node) is not dict:
            raise TypeError("continuation frame nodes must be plain dicts")
        identity = id(node)
        if identity in active:
            raise ValueError("continuation checkpoint frames must not contain cycles")
        active.add(identity)
        try:
            kind = node.get("kind")
            if type(kind) is not str:
                raise ValueError("continuation frame kind must be a plain string")
            if (
                isinstance(template, torch.Tensor)
                or type(template) is _TensorDescriptor
            ):
                if isinstance(template, torch.Tensor):
                    _ordinary_tensor(template)
                if kind != "tensor":
                    raise ValueError(f"{path} must contain a Tensor reference")
                _keys(node, ("kind", "index"), label="continuation Tensor reference")
                index = node["index"]
                if type(index) is not int or not 0 <= index < len(tensors):
                    raise ValueError("continuation Tensor reference index is invalid")
                value = tensors[index]
                if (
                    tuple(value.shape) != tuple(template.shape)
                    or value.dtype != template.dtype
                    or value.device != template.device
                ):
                    raise ValueError(
                        f"{path} must match the receiving shape, dtype and device"
                    )
                if index not in copies:
                    copies[index] = _owned_tensor(value, flags[index])
                return copies[index]
            if template is None:
                _keys(node, ("kind",), label="continuation None node")
                if kind != "none":
                    raise ValueError(f"{path} must be None")
                return None
            if isinstance(template, str) or type(template) in _SCALARS:
                _keys(node, ("kind", "value"), label="continuation scalar")
                expected_type = str if isinstance(template, str) else type(template)
                if kind != "scalar" or type(node["value"]) is not expected_type:
                    raise ValueError(f"{path} has an incompatible scalar schema")
                return node["value"]
            if isinstance(template, Mapping):
                if kind != "dict":
                    raise ValueError(f"{path} must contain a mapping")
                _keys(node, ("kind", "keys", "children"), label="continuation mapping")
                keys, children = node["keys"], node["children"]
                if type(keys) is not list or not all(type(key) is str for key in keys):
                    raise TypeError("continuation mapping keys must be plain strings")
                if len(set(keys)) != len(keys):
                    raise ValueError("continuation mapping keys must be unique")
                if type(children) is not list or len(children) != len(keys):
                    raise ValueError(
                        "continuation mapping children do not match its keys"
                    )
                expected_keys = _string_keys(template, label="receiving frame")
                if set(keys) != set(expected_keys):
                    raise ValueError(f"{path} must match the receiving frame keys")
                expected = dict(zip(expected_keys, template.values(), strict=True))
                return {
                    name: decode(child, expected[name], f"{path}.{name}")
                    for name, child in zip(keys, children, strict=True)
                }
            if type(template) not in (list, tuple, *_NAMED_TUPLES):
                raise FunctionalizationError(
                    "continuation checkpoint receiving frames contain an unsupported "
                    "container type"
                )
            named = type(template) in _NAMED_TUPLES
            expected_kind = (
                "namedtuple"
                if named
                else ("list" if type(template) is list else "tuple")
            )
            if kind != expected_kind:
                raise ValueError(f"{path} has an incompatible sequence schema")
            _keys(
                node,
                ("kind", "children", "fields") if named else ("kind", "children"),
                label="continuation sequence",
            )
            children = node["children"]
            if type(children) is not list or len(children) != len(template):
                raise ValueError(f"{path} has an incompatible sequence length")
            if named:
                fields = node["fields"]
                if (
                    type(fields) is not list
                    or not all(type(field) is str for field in fields)
                    or fields != list(type(template)._fields)
                ):
                    raise ValueError(f"{path} has incompatible namedtuple fields")
            values = [
                decode(child, expected, f"{path}[{index}]")
                for index, (child, expected) in enumerate(
                    zip(children, template, strict=True)
                )
            ]
            return type(template)(*values) if named else type(template)(values)
        finally:
            active.remove(identity)

    with torch.inference_mode(False), torch.no_grad():
        frames = decode(packet["tree"], templates, "continuation")
    if len(copies) != len(tensors):
        raise ValueError("continuation Tensor table contains unused entries")
    return frames


__all__ = []
