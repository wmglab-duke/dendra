"""Canonical state views for an invocation-private inference pack.

Projection is a capability of the standard clone adapters, not a replacement
for arbitrary clone policies. Observable containers, tensors and Torch modes
retain the ordinary per-occurrence clone path.
"""

from __future__ import annotations

from types import (
    BuiltinFunctionType,
    FunctionType,
    GetSetDescriptorType,
    MethodDescriptorType,
    MethodType,
    WrapperDescriptorType,
)

import torch
from torch.utils._device import DeviceContext

from dendra.models.integrators.core import _population_runtime_modes_plain


def _clone_state_leaf(value):
    return value.clone(memory_format=torch.contiguous_format)


def _identity_state_leaf(value):
    return value


def _state_projection(clone, *, guard=None):
    """Associate a standard projection with its exact ordinary clone policy."""

    def decorate(project):
        project._state_clone = (
            clone,
            clone.__code__,
            clone.__defaults__,
            project.__code__,
            project.__defaults__,
            guard,
        )
        return project

    return decorate


def _projection_matches(clone, project):
    if type(clone) is MethodType and type(project) is MethodType:
        if clone.__self__ is not project.__self__:
            return False
        clone, project = clone.__func__, project.__func__
    if type(clone) is not FunctionType or type(project) is not FunctionType:
        return False
    binding = vars(project).get("_state_clone")
    return (
        type(binding) is tuple
        and len(binding) == 6
        and clone is binding[0]
        and clone.__code__ is binding[1]
        and clone.__defaults__ is binding[2]
        and project.__code__ is binding[3]
        and project.__defaults__ is binding[4]
        and (binding[5] is None or binding[5]())
    )


_TENSOR = torch.Tensor
_MRO_GET = type.__dict__["__mro__"].__get__
_CLASS_DICT = type.__dict__["__dict__"].__get__
_TENSOR_MRO = _MRO_GET(_TENSOR)
_NAMES = (
    "clone",
    "reshape",
    "numel",
    "shape",
    "dtype",
    "device",
    "layout",
    "is_nested",
    "__dict__",
    "__getattribute__",
    "__torch_function__",
    "__torch_dispatch__",
    "is_contiguous",
    "storage_offset",
    "is_conj",
    "is_neg",
)


def _tensor_descriptor(name):
    return next(
        (_CLASS_DICT(cls)[name] for cls in _TENSOR_MRO if name in _CLASS_DICT(cls)),
        None,
    )


_DESCRIPTORS = tuple(_tensor_descriptor(name) for name in _NAMES)
_TENSOR_DICT = _DESCRIPTORS[8]
_DEVICE_FUNCTION = DeviceContext.__torch_function__
_DEVICE_CODE = _DEVICE_FUNCTION.__code__
_DEVICE_DEFAULTS = _DEVICE_FUNCTION.__defaults__
_DEVICE_CONSTRUCTORS = _DEVICE_FUNCTION.__globals__.get("_device_constructors")
_IS_FUNCTIONAL = getattr(torch, "_is_functional_tensor", None)
_IS_WRAPPED = getattr(
    getattr(torch._C, "_functorch", None), "is_functorch_wrapped_tensor", None
)


def _plain_projection_context(*, check_saved_tensors=True):
    try:
        return (
            torch.Tensor is _TENSOR
            and _MRO_GET(_TENSOR) is _TENSOR_MRO
            and all(
                _tensor_descriptor(name) is descriptor
                for name, descriptor in zip(_NAMES, _DESCRIPTORS, strict=True)
            )
            and all(
                type(descriptor) is MethodDescriptorType
                for descriptor in _DESCRIPTORS[:3]
            )
            and all(
                type(descriptor) is GetSetDescriptorType
                for descriptor in _DESCRIPTORS[3:9]
            )
            and type(_DESCRIPTORS[9]) is WrapperDescriptorType
            and all(
                type(descriptor) is MethodDescriptorType
                for descriptor in _DESCRIPTORS[12:]
            )
            and DeviceContext.__torch_function__ is _DEVICE_FUNCTION
            and _DEVICE_FUNCTION.__code__ is _DEVICE_CODE
            and _DEVICE_FUNCTION.__defaults__ is _DEVICE_DEFAULTS
            and _DEVICE_FUNCTION.__globals__.get("_device_constructors")
            is _DEVICE_CONSTRUCTORS
            and torch._is_functional_tensor is _IS_FUNCTIONAL
            and torch._C._functorch.is_functorch_wrapped_tensor is _IS_WRAPPED
            and type(_IS_FUNCTIONAL) is BuiltinFunctionType
            and type(_IS_WRAPPED) is BuiltinFunctionType
            and not torch._C._are_functorch_transforms_active()
            and torch.autograd.forward_ad._current_level < 0
            and not torch.compiler.is_compiling()
            and not torch.jit.is_tracing()
            and _population_runtime_modes_plain(check_saved_tensors=check_saved_tensors)
        )
    except AttributeError:
        return False


def _plain_state_tree(tree, active=None):
    """Inspect only builtin containers and unmodified ordinary tensor leaves."""
    if tree is None:
        return True
    if type(tree) is _TENSOR:
        return (
            not _TENSOR_DICT.__get__(tree)
            and not _IS_FUNCTIONAL(tree)
            and not _IS_WRAPPED(tree)
            and tree.layout is torch.strided
            and not tree.is_nested
        )
    kind = type(tree)
    if kind not in (dict, tuple, list):
        return False
    if active is None:
        active = set()
    identity = id(tree)
    if identity in active:
        return False
    active.add(identity)
    try:
        if kind is dict:
            return all(type(key) in (str, int) for key in tree) and all(
                _plain_state_tree(value, active) for value in tree.values()
            )
        return all(_plain_state_tree(value, active) for value in tree)
    finally:
        active.remove(identity)


def _can_project_state(clone, project, state, carries, partition):
    if (
        not _projection_matches(clone, project)
        or not _plain_projection_context()
        or not _plain_state_tree(state)
    ):
        return False
    if not partition.invariant_names:
        return _plain_state_tree(carries)
    # Do not inspect/read through authored callback mappings, and leave LFP's
    # invariant tensors on their existing explicit-operand path.
    if type(carries) is not dict or any(type(name) is not str for name in carries):
        return False
    return all(
        name in carries
        and (name in partition.invariant_names or _plain_state_tree(carries[name]))
        for name in partition.names
    )


__all__ = []
