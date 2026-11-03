"""Independent lifted loop operands with authored tensor-view layouts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodDescriptorType

import torch
from torch.utils import _pytree as pytree

from ._state_projection import (
    _CLASS_DICT,
    _MRO_GET,
    _NAMES,
    _TENSOR_DICT,
    _plain_projection_context,
    _plain_state_tree,
    _tensor_descriptor,
)

_PARAMETER = torch.nn.Parameter
_PARAMETER_MRO = _MRO_GET(_PARAMETER)
_PARAMETER_TORCH_FUNCTION = torch._C._disabled_torch_function_impl


def _plain_parameter_class():
    """Keep authored Parameter methods on the original operand path."""
    if (
        torch.nn.Parameter is not _PARAMETER
        or _MRO_GET(_PARAMETER) is not _PARAMETER_MRO
        or getattr(torch._C, "_disabled_torch_function_impl", None)
        is not _PARAMETER_TORCH_FUNCTION
    ):
        return False
    namespaces = tuple(_CLASS_DICT(cls) for cls in _PARAMETER_MRO)
    if any("__getattr__" in namespace for namespace in namespaces):
        return False
    for name in _NAMES:
        actual = next(
            (namespace[name] for namespace in namespaces if name in namespace), None
        )
        expected = (
            _PARAMETER_TORCH_FUNCTION
            if name == "__torch_function__"
            else _tensor_descriptor(name)
        )
        if actual is not expected:
            return False
    return True


_DRAW_STRIDE_DESCRIPTOR = _tensor_descriptor("stride")


def _own_runtime_draw_packets(tree):
    """Own only canonical plain runtime draw tapes before scan specialization.

    Base draws reach authored distribution transforms through existing owned
    per-frame clones. Their leading tape offsets do not describe authored
    state. Ordinary deterministic inputs and observable tensor layouts retain
    their original objects, metadata, gradients and dispatch behavior.
    """
    from ._draws import RuntimeDraws
    from ._inputs import Inputs, _RuntimeInputs
    from ._netstim import NetStimDraw
    from ._types import RolloutInput, StepInput

    if not (
        _plain_projection_context()
        and type(_DRAW_STRIDE_DESCRIPTOR) is MethodDescriptorType
        and _tensor_descriptor("stride") is _DRAW_STRIDE_DESCRIPTOR
    ):
        return tree

    packets = (RuntimeDraws, NetStimDraw)
    envelopes = (Inputs, _RuntimeInputs, StepInput, RolloutInput)

    def visit(value):
        kind = type(value)
        if kind in packets:
            fields = tuple(value)
            if not _plain_state_tree(fields):
                return value
            leaves = pytree.tree_leaves(fields)
            if not any(leaf is not None for leaf in leaves):
                return value
            for leaf in leaves:
                if leaf is None:
                    continue
                # Preserve every visible frame stride, including singleton
                # dimensions. A contiguous clone of other layouts could change
                # a custom registered distribution's preserve_format clone.
                stride = 1
                for length, actual in zip(
                    reversed(leaf.shape), reversed(leaf.stride()), strict=True
                ):
                    if actual != stride:
                        return value
                    stride *= max(length, 1)
            # Retain gradients on live explicit operands. Inference tensors
            # become ordinary owned tensors, matching disposable capture data.
            with torch.inference_mode(False):
                return pytree.tree_map_only(
                    torch.Tensor,
                    lambda leaf: leaf.clone(memory_format=torch.contiguous_format),
                    value,
                )
        if kind is dict:
            if any(type(name) is not str for name in value):
                return value
            updated = {name: visit(child) for name, child in value.items()}
            return (
                updated
                if any(updated[name] is not child for name, child in value.items())
                else value
            )
        if kind in (tuple, list) or kind in envelopes:
            updated = tuple(visit(child) for child in value)
            if all(left is right for left, right in zip(updated, value, strict=True)):
                return value
            if kind in envelopes:
                return kind(*updated)
            return list(updated) if kind is list else updated
        return value

    return visit(tree)


def _dense_zero_offset(value):
    return (
        value.is_contiguous()
        and value.storage_offset() == 0
        and not (value.is_conj() or value.is_neg())
    )


def _copy_view(value):
    # Copy the physical span before restoring logical views. In particular,
    # clone(preserve_format) still compacts slices and expanded tensors.
    # Captured programs can use their original strides in explicit view ops.
    physical = value
    conjugate, negative = value.is_conj(), value.is_neg()
    if _dense_zero_offset(value):
        return value.clone(memory_format=torch.preserve_format)
    if conjugate:
        physical = physical.conj()
    if negative:
        physical = torch._neg_view(physical)
    if value.numel():
        size = value.storage_offset() + 1
        for length, stride in zip(value.shape, value.stride(), strict=True):
            size += (length - 1) * stride
        storage = physical.as_strided((size,), (1,), 0).clone()
    else:
        storage = physical.new_empty((0,))
    result = storage.as_strided(value.shape, value.stride(), value.storage_offset())
    if negative:
        result = torch._neg_view(result)
    if conjugate:
        result = result.conj()
    return result


class _IndependentView(torch.autograd.Function):
    """Storage isolation is the identity in logical tensor coordinates."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value):
        return _copy_view(value)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return gradient

    @staticmethod
    def jvp(ctx, tangent):
        return _copy_view(tangent)


@torch.compiler.allow_in_graph
def _isolate_lifted_tensor(value):
    # AOTAutograd keeps the logical identity derivative of overlapping views.
    # Dynamo cannot inline an autograd.Function with a custom JVP; the audited
    # operation also reads only static tensor-view metadata.
    return _IndependentView.apply(value)


def _input_usage(graph, arguments):
    """Retain only static placeholder-use metadata from an authored step."""
    values, spec = pytree.tree_flatten(arguments)
    if isinstance(graph, torch.fx.GraphModule):
        placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
        if len(placeholders) != len(values):
            raise ValueError("loop input usage changed its captured argument structure")
        used = [bool(node.users) for node in placeholders]
    else:
        # Non-FX capture adapters conservatively retain tensor arguments.
        used = [torch.is_tensor(value) for value in values]
    return pytree.tree_unflatten(used, spec)


@dataclass(frozen=True)
class _LiftedViewIsolation:
    spec: object
    indices: tuple[int, ...]
    captured_inputs: bool = False

    @classmethod
    def from_usage(cls, usage, *, captured_inputs=False):
        values, spec = pytree.tree_flatten(usage)
        return cls(
            spec,
            tuple(index for index, used in enumerate(values) if used),
            captured_inputs,
        )

    def __call__(self, tree):
        # Authored views enter the graph as owned Tensor inputs. Creating them
        # inside it can leave identity ReinterpretViews lifted into a HOP; the
        # C++ wrapper can reuse a freed temporary across condition/body calls.
        # Ignore unused inputs and keep dense copies in the compiled graph.
        values, spec = pytree.tree_flatten(tree)
        if spec != self.spec:
            # FX binds dictionary arguments by key. Reuse that captured order
            # when an equivalent invocation mapping has a different insertion
            # order, so the usage indices still identify the intended leaves.
            try:
                values = self.spec.flatten_up_to(tree)
                canonical = pytree.tree_unflatten(values, self.spec)
                if pytree.tree_structure(canonical) != self.spec:
                    raise ValueError("loop input leaf became a subtree")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "loop view isolation changed its argument structure"
                ) from error
            spec = self.spec
        copied = False
        if self.captured_inputs:
            # The authored graph has already resolved metadata-dependent
            # branches against the raw schema. Unused Tensor placeholders need
            # no runtime operand or compiler guard (e.g. Parameter vs Tensor
            # after portable restoration). Keep their captured positions.
            used = set(self.indices)
            for index, value in enumerate(values):
                if index not in used and torch.is_tensor(value):
                    values[index] = None
                    copied = True
        parameter_views, plain_modes = {}, None
        for index in self.indices:
            value = values[index]
            if torch.is_tensor(value) and not _dense_zero_offset(value):
                values[index] = _isolate_lifted_tensor(value)
                copied = True
            elif self.captured_inputs and type(value) is _PARAMETER:
                if plain_modes is None:
                    # Alias is an identity view and saves no tensors. It is
                    # safe under checkpoint save hooks, but authored execution
                    # modes retain the original operand/dispatch behavior.
                    plain_modes = (
                        _plain_projection_context(check_saved_tensors=False)
                        and _plain_parameter_class()
                    )
                if plain_modes:
                    namespace = _TENSOR_DICT.__get__(value)
                    if type(namespace) is dict and not namespace:
                        identity = id(value)
                        if identity not in parameter_views:
                            parameter_views[identity] = torch.ops.aten.alias.default(
                                value
                            )
                        values[index] = parameter_views[identity]
                        copied = True
        return pytree.tree_unflatten(values, spec) if copied else tree


@torch.compiler.allow_in_graph
def _isolate_dense_lifted_tensor(value):
    # AOT traces static view metadata; Dynamo cannot inline storage_offset().
    return (
        value.clone(memory_format=torch.preserve_format)
        if _dense_zero_offset(value)
        else value
    )


def _isolate_dense_lifted_inputs(tree):
    # Dense copies do not need view handles. Keep these inside compilation so
    # unused copies disappear and numerical kernels can fuse with their users.
    # Non-dense operands already own storage at the invocation boundary.
    return pytree.tree_map_only(
        torch.Tensor,
        _isolate_dense_lifted_tensor,
        tree,
    )


__all__ = []
