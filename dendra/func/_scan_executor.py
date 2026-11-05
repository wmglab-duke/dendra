"""Conservative executor identity after raw scan-step capture.

Different tape offsets still author their own graphs. Pool only identical,
read-only programs whose sample reads cannot depend on a physical storage
prefix. Live operands and their existing isolation are never changed here.
"""

from __future__ import annotations

import operator
from collections import namedtuple
from types import CodeType, FunctionType

import torch
from torch.fx import Graph, Node
from torch.fx.graph import _PyTreeCodeGen
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.utils import _pytree as pytree

from ._state_projection import (
    _IS_FUNCTIONAL,
    _IS_WRAPPED,
    _TENSOR_DICT,
    _plain_projection_context,
    _plain_state_tree,
    _tensor_descriptor,
)

_DESCRIPTOR_NAMES = ("stride", "requires_grad", "is_leaf", "_version")
_DESCRIPTORS = tuple(_tensor_descriptor(name) for name in _DESCRIPTOR_NAMES)
_IS_INFERENCE = torch.is_inference

_ALIASES = frozenset(
    id(value)
    for value in (
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.real.default,
        torch.ops.aten.imag.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.select.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.broadcast_to.default,
    )
)
_ALLOCATIONS = frozenset(
    id(value)
    for value in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.neg.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.index_select.default,
        torch.ops.aten._to_copy.default,
    )
)


def _literal(value, nodes=None):
    kind = type(value)
    if kind is Node:
        if nodes is None:
            raise TypeError
        return ("node", nodes[value])
    if value is None or kind in (str, int, bool):
        return kind, value
    if kind is float:
        return kind, value.hex()
    if kind in (torch.dtype, torch.device, torch.layout, torch.memory_format):
        return kind, str(value)
    if kind is type:
        return kind, id(value)
    if kind is slice:
        return kind, tuple(
            _literal(item, nodes) for item in (value.start, value.stop, value.step)
        )
    if kind in (tuple, list, immutable_list):
        return kind, tuple(_literal(item, nodes) for item in value)
    if kind in (dict, immutable_dict):
        return kind, tuple(
            (_literal(key, nodes), _literal(item, nodes)) for key, item in value.items()
        )
    raise TypeError


def _spec_key(spec):
    if type(spec) is pytree.LeafSpec:
        return ("leaf",)
    if type(spec) is not pytree.TreeSpec:
        raise TypeError
    if not any(spec.type is kind for kind in (dict, tuple, list)):
        from ._draws import RuntimeDraws
        from ._inputs import Inputs, _RuntimeInputs
        from ._netstim import NetStimDraw
        from ._types import RolloutInput, StepInput

        if spec.type is not namedtuple or not any(
            spec.context is kind
            for kind in (
                RuntimeDraws,
                Inputs,
                _RuntimeInputs,
                NetStimDraw,
                RolloutInput,
                StepInput,
            )
        ):
            raise TypeError
    return (
        id(spec.type),
        _literal(spec.context),
        tuple(_spec_key(child) for child in spec.children()),
    )


def _forward_key(graph):
    fields = vars(graph)
    if "forward" in fields or any(
        fields.get(name)
        for name in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
    ):
        raise TypeError
    from torch.nn.modules import module

    if any(
        getattr(module, name)
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    ):
        raise TypeError
    forward = vars(type(graph)).get("forward")
    source = fields.get("_code")
    if (
        type(forward) is not FunctionType
        or forward.__closure__ is not None
        or forward.__defaults__ is not None
        or forward.__kwdefaults__ is not None
        or type(source) is not str
    ):
        raise TypeError
    regenerated = fields["_graph"].python_code("self")
    if regenerated.src != source or any(
        forward.__globals__.get(name) is not value
        for name, value in regenerated.globals.items()
    ):
        raise TypeError
    generated = next(
        value
        for value in compile(
            source, "<scan-executor>", "exec", dont_inherit=True
        ).co_consts
        if type(value) is CodeType and value.co_name == "forward"
    )
    if (forward.__code__.co_flags & ~annotations.compiler_flag) != generated.co_flags:
        raise TypeError
    for name in (
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_freevars",
        "co_cellvars",
        "co_exceptiontable",
    ):
        if getattr(forward.__code__, name, None) != getattr(generated, name, None):
            raise TypeError
    return source, tuple(
        (name, id(forward.__globals__[name]))
        for name in forward.__code__.co_names
        if name in forward.__globals__
    )


def _logical_samples(placeholders):
    pending, seen = list(placeholders), set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        for user in node.users:
            if user.op != "call_function":
                return False
            target = id(user.target)
            if target in _ALLOCATIONS:
                continue
            if target not in _ALIASES:
                return False
            pending.append(user)
    return True


def _sample_contract(signature):
    layout, (spec, metadata), sample_layout, assembly, grad_enabled = signature
    children = spec.children()
    if len(children) != 5:
        raise TypeError
    count = children[-1].num_leaves
    rows = list(metadata)
    for index in range(len(rows) - count, len(rows)):
        row = rows[index]
        if type(row) is not tuple or len(row) != 10:
            raise TypeError
        shape, _dtype, _device, strides, offset, _grad, _leaf, conj, neg, inference = (
            row
        )
        if conj or neg or inference or type(offset) is not int or offset < 0:
            raise TypeError
        stride = 1
        for length, actual in zip(reversed(shape), reversed(strides), strict=True):
            if type(length) is not int or length < 0 or actual != stride:
                raise TypeError
            stride *= max(length, 1)
        rows[index] = (*row[:4], 0, *row[5:])
    tape_spec, positions = sample_layout
    return (
        layout,
        _spec_key(spec),
        _literal(tuple(rows)),
        _spec_key(tape_spec),
        _literal(positions),
        _literal(assembly),
        grad_enabled,
    )


def _plain_constant(value):
    if _plain_state_tree(value):
        return True
    namespace = _TENSOR_DICT.__get__(value)
    if type(namespace) is not dict or len(namespace) != 1:
        return False
    name, marker = next(iter(namespace.items()))
    # Dynamo marks lifted module buffers after their first compilation. This
    # annotation changes no tensor values/layout and grants no authored method
    # overrides. Keep this exception local to exact lifted constant owners.
    return (
        type(name) is str
        and name == "_dynamo_static_input_type"
        and type(marker) is str
        and marker in ("guarded", "unguarded")
        and not _IS_FUNCTIONAL(value)
        and not _IS_WRAPPED(value)
        and value.layout is torch.strided
        and not value.is_nested
    )


def _executor_key(graph, signature, options):
    """Return a cold, metadata-only pool key, or retain ordinary compilation."""
    try:
        if (
            signature is None
            or not _plain_projection_context(check_saved_tensors=False)
            or torch.is_inference is not _IS_INFERENCE
            or any(
                _tensor_descriptor(name) is not descriptor
                for name, descriptor in zip(
                    _DESCRIPTOR_NAMES, _DESCRIPTORS, strict=True
                )
            )
            or type(options) is not dict
        ):
            return None
        backend = options.get("backend")
        if backend is None:
            from torch._dynamo.backends.registry import get_default_backend

            backend = get_default_backend()
        if type(backend) is not str or backend not in (
            "inductor",
            "aot_eager",
            "eager",
        ):
            return None
        fields = vars(graph)
        fx_graph = fields.get("_graph")
        if type(fx_graph) is not Graph or type(fx_graph._codegen) is not _PyTreeCodeGen:
            return None
        input_spec, output_spec = fields["_in_spec"], fields["_out_spec"]
        input_children = input_spec.children()
        if len(input_children) != 6:
            return None
        nodes = tuple(fx_graph.nodes)
        if any(type(node) is not Node for node in nodes):
            return None
        indices = {node: index for index, node in enumerate(nodes)}
        placeholders = [node for node in nodes if node.op == "placeholder"]
        count = input_children[-1].num_leaves
        if len(placeholders) != input_spec.num_leaves or not _logical_samples(
            placeholders[len(placeholders) - count :]
        ):
            return None
        program = []
        for node in nodes:
            if node.op == "call_function":
                target = node.target
                if type(target) is torch._ops.OpOverload:
                    if target._schema.is_mutable:
                        return None
                elif target is not operator.getitem:
                    return None
                target_key = id(target)
            elif node.op == "get_attr":
                # make_fx registers its lifted constants directly as buffers.
                # The pooled graph keeps those exact owners alive; never equate
                # distinct tensors by their numerical values.
                name = node.target
                if type(name) is not str or "." in name:
                    return None
                value = fields.get("_buffers", {}).get(name)
                if (
                    type(value) is not torch.Tensor
                    or not _plain_constant(value)
                    or _IS_INFERENCE(value)
                    or value.requires_grad
                ):
                    return None
                target_key = (
                    name,
                    id(value),
                    value._version,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                    value.dtype,
                    value.device,
                    value.is_leaf,
                    value.is_conj(),
                    value.is_neg(),
                )
            elif node.op in ("placeholder", "output"):
                target_key = _literal(node.target)
            else:
                return None
            program.append(
                (
                    node.op,
                    target_key,
                    _literal(node.args, indices),
                    _literal(node.kwargs, indices),
                )
            )
        return (
            _forward_key(graph),
            tuple(program),
            _spec_key(input_spec),
            _spec_key(output_spec),
            _sample_contract(signature),
            _literal(options),
            backend,
        )
    except Exception:
        # Admission cannot replace the caller's capture/compiler diagnostics.
        return None


__all__ = []
