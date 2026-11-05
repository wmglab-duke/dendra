"""Shared tensor contracts for validation, preparation, and loop capture.

Schema validation, identity/version freshness, and capture layout describe
different boundaries. Keep those contracts separate even when they inspect
the same tensor.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils import _pytree as pytree


class _CaptureTensorType:
    """Retain a subclass identity without invoking its authored metaclass."""

    __slots__ = ("kind",)

    def __init__(self, kind):
        self.kind = kind

    def __hash__(self):
        return id(self.kind)

    def __eq__(self, other):
        return type(other) is _CaptureTensorType and self.kind is other.kind


def _capture_tensor_type(value):
    kind = type(value)
    if kind is torch.Tensor or kind is torch.nn.Parameter:
        return kind
    return _CaptureTensorType(kind)


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.preserve_format)


def _clone_contiguous_tree(tree):
    """Own each Tensor occurrence independently, retaining its autograd history."""
    return pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.clone(memory_format=torch.contiguous_format),
        tree,
    )


def _shape_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def validate_tensor(value, *, shape, dtype, device, label):
    if not torch.is_tensor(value):
        raise TypeError(f"{label} must be a Tensor")
    if tuple(value.shape) != shape or value.dtype != dtype or value.device != device:
        raise ValueError(f"{label} must match {device}/{dtype}/{shape}")


def _unpack_forward_ad(value: torch.Tensor):
    """Expose a dual tangent even when the caller has inference mode enabled."""
    if torch.is_inference_mode_enabled():
        # Inference mode normally hides forward-AD tangents from unpack_dual().
        # Freshness metadata must describe the tensor itself rather than the
        # ambient execution mode, or a plan could silently lose its tangent
        # binding when prepared/consumed under inference mode.
        with torch.inference_mode(False):
            return torch.autograd.forward_ad.unpack_dual(value)
    return torch.autograd.forward_ad.unpack_dual(value)


def _has_inference_tensor_component(value: torch.Tensor) -> bool:
    tangent = _unpack_forward_ad(value).tangent
    return torch.is_inference(value) or (
        tangent is not None and torch.is_inference(tangent)
    )


def _tensor_leaf_binding(value: torch.Tensor) -> tuple:
    """Bind one primal (or ordinary Tensor) and its optional dual tangent."""
    unpacked = _unpack_forward_ad(value)
    tangent = unpacked.tangent
    if torch.is_inference(value) or (
        tangent is not None and torch.is_inference(tangent)
    ):
        raise TypeError(
            "tensor bindings require versioned primal and forward-AD tangent values"
        )
    tangent_binding = (
        None
        if tangent is None
        else (id(tangent), tangent._version, tangent.requires_grad)
    )
    # ``value`` is the stable owner for a dual's primal storage. The primal
    # view returned by unpack_dual() is recreated on demand, so its Python
    # identity is deliberately not part of the freshness contract.
    return (id(value), value._version, value.requires_grad, tangent_binding)


def _tensor_binding(values: Mapping[str, torch.Tensor]) -> tuple:
    """Bind a derived workspace to the exact tensor versions that created it."""
    return tuple((name, *_tensor_leaf_binding(values[name])) for name in sorted(values))


def _tensor_tree_binding(values) -> tuple:
    """Bind every tensor leaf in a prepared workspace tree."""
    leaves, tree_spec = torch.utils._pytree.tree_flatten(values)
    if any(not torch.is_tensor(value) for value in leaves):
        raise TypeError("prepared workspaces must contain only Tensor leaves")
    return (
        tree_spec,
        tuple(_tensor_leaf_binding(value) for value in leaves),
    )


def capture_schema(tree):
    """Describe authored tensor classes and views used to capture a loop."""
    values, spec = pytree.tree_flatten(tree)
    return spec, tuple(
        (
            tuple(value.shape),
            value.dtype,
            value.device,
            tuple(value.stride()),
            value.storage_offset(),
            value.requires_grad,
            value.is_leaf,
            value.is_conj(),
            value.is_neg(),
            torch.is_inference(value),
            _capture_tensor_type(value),
        )
        if torch.is_tensor(value)
        else (type(value), value)
        for value in values
    )
