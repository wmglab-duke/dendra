"""Retain authored current VJPs across primal-only conditional/scan capture.

Only phases with authored AD boundaries use these operators. AOTAutograd
supplies both tensor programs; local Inductor decompositions recover their
fusion after scan has constructed its derivative. The weak registry owns no
model tensors or programs beyond their functional plan's lifetime.
"""

from __future__ import annotations

import itertools
import uuid
import weakref

import torch
from torch.utils import _pytree as pytree

from dendra._bootstrap import torch_compiler_warning_context

_PROGRAMS = weakref.WeakValueDictionary()
_IDENTIFIERS = itertools.count()
# AOT's persistent cache hashes opaque calls before local decompositions
# expose their graphs. Counters alone could reuse another program's VJP after
# restart. Expanded kernels retain Inductor's ordinary content-based keys.
_SESSION = uuid.uuid4().hex


def _forward(program_id, operands):
    program = _PROGRAMS[program_id]
    return [
        value.clone(memory_format=torch.contiguous_format)
        for value in program.forward(*operands)[: program.output_count]
    ]


def _backward(program_id, operands, tangents):
    program = _PROGRAMS[program_id]
    # Recompute only this pure current frame's saved activations. Mechanism
    # advancement, stimulation, and random draws lie outside these programs.
    activations = program.forward(*operands)[program.output_count :]
    gradients = program.backward(*activations, *tangents)
    return [
        gradients[index].clone(memory_format=torch.contiguous_format)
        for index in program.gradient_indices
    ]


@torch.library.custom_op("dendra_dufort::current_frame", mutates_args=())
def _current_frame(program_id: str, operands: list[torch.Tensor]) -> list[torch.Tensor]:
    return _forward(program_id, operands)


@_current_frame.register_fake
def _current_frame_fake(program_id, operands):
    return [
        operands[0].new_empty(shape, dtype=dtype)
        for shape, dtype in _PROGRAMS[program_id].output_metadata
    ]


@torch.library.custom_op("dendra_dufort::current_vjp", mutates_args=())
def _current_vjp(
    program_id: str, operands: list[torch.Tensor], tangents: list[torch.Tensor]
) -> list[torch.Tensor]:
    return _backward(program_id, operands, tangents)


@_current_vjp.register_fake
def _current_vjp_fake(program_id, operands, tangents):
    return [
        torch.empty_like(operands[index], memory_format=torch.contiguous_format)
        for index in _PROGRAMS[program_id].gradient_indices
    ]


def _setup_context(ctx, inputs, output):
    program_id, operands = inputs
    ctx.program_id = program_id
    ctx.program = _PROGRAMS[program_id]
    ctx.save_for_backward(*operands)
    ctx.mark_non_differentiable(
        *(
            value
            for value, enabled in zip(output, ctx.program.output_grad)
            if not enabled
        )
    )


def _current_backward(ctx, tangents):
    operands = list(ctx.saved_tensors)
    program = ctx.program
    live_tangents = [
        tangent if tangent is not None else operands[0].new_zeros(shape, dtype=dtype)
        for tangent, enabled, (shape, dtype) in zip(
            tangents, program.output_grad, program.output_metadata, strict=True
        )
        if enabled
    ]
    live_tangents = [
        value.clone(memory_format=torch.contiguous_format) for value in live_tangents
    ]
    computed = _current_vjp(ctx.program_id, operands, live_tangents)
    gradients = [None] * len(operands)
    for index, gradient in zip(program.gradient_indices, computed, strict=True):
        gradients[index] = gradient
    return None, gradients


_current_frame.register_autograd(_current_backward, setup_context=_setup_context)


class _AutogradProgram:
    def __deepcopy__(self, memo):
        # Programs contain immutable graphs/metadata; all live state is supplied
        # as operands. Sharing also retains the weak registry's sole identity.
        memo[id(self)] = self
        return self

    def __call__(self, *operands):
        return pytree.tree_unflatten(
            _current_frame(self.identifier, list(operands)), self.output_spec
        )


def capture_autograd_program(evaluate, arguments):
    """Capture actual native forward/backward contracts with explicit inputs."""
    from torch._functorch.aot_autograd import aot_function, make_boxed_func

    program = _AutogradProgram()

    def forward_compiler(graph, _arguments):
        program.forward = graph
        return make_boxed_func(graph.forward)

    def backward_compiler(graph, _arguments):
        program.backward = graph
        return make_boxed_func(graph.forward)

    compiled = aot_function(
        evaluate, fw_compiler=forward_compiler, bw_compiler=backward_compiler
    )
    with torch_compiler_warning_context():
        outputs, program.output_spec = pytree.tree_flatten(compiled(*arguments))
        program.output_count = len(outputs)
        program.output_metadata = tuple(
            (tuple(value.shape), value.dtype) for value in outputs
        )
        program.output_grad = tuple(value.requires_grad for value in outputs)
        differentiable = tuple(value for value in outputs if value.requires_grad)
        if differentiable:
            argument_indices = tuple(
                index for index, value in enumerate(arguments) if value.requires_grad
            )
            gradients = torch.autograd.grad(
                differentiable,
                tuple(arguments[index] for index in argument_indices),
                tuple(torch.ones_like(value) for value in differentiable),
                allow_unused=True,
            )
            program.gradient_indices = tuple(
                index
                for index, gradient in zip(argument_indices, gradients, strict=True)
                if gradient is not None
            )
        else:
            program.gradient_indices = ()
    program.identifier = f"{_SESSION}:{next(_IDENTIFIERS)}"
    _PROGRAMS[program.identifier] = program
    return program


def _inductor_decompositions():
    """Expand tensor programs only in the owned post-AD compiler boundary."""
    return {
        torch.ops.dendra_dufort.current_frame.default: _forward,
        torch.ops.dendra_dufort.current_vjp.default: _backward,
    }
