"""Separately compiled detached Newton programs for Population training scans."""

import uuid
import weakref
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    _temp_remove_pre_dispatch_torch_function_mode,
    disable_proxy_modes_tracing,
    make_fx,
)

from dendra._bootstrap import compile_owned

_PROGRAMS = weakref.WeakValueDictionary()
_CAPTURE = ContextVar("dendra_derivimplicit_forward_capture", default=None)


@contextmanager
def _outside_step_trace():
    # The inner while capture installs its own metadata modes. Leaving the
    # outer modes active can make Dynamo specialize against another step trace.
    with (
        disable_proxy_modes_tracing(),
        _temp_remove_pre_dispatch_torch_function_mode(),
        _temp_remove_metadata_torch_function_mode(),
    ):
        yield


@contextmanager
def _capture_forward_programs(backend):
    if not isinstance(backend, str) or backend not in (
        "eager",
        "aot_eager",
        "inductor",
    ):
        backend = "inductor"
    programs = []
    token = _CAPTURE.set((programs, backend))
    try:
        yield programs
    finally:
        _CAPTURE.reset(token)


class _ForwardProgram:
    def __init__(self, graph, backend, device):
        # The opaque call is part of the outer AOT cache key. A fresh identifier
        # prevents a cached graph from looking up another program in this process.
        self.identifier = uuid.uuid4().hex
        self.graph = graph
        options = {}
        if backend == "inductor" and device.type == "cuda":
            # The independent primal's host loop reads a CUDA predicate. Use
            # ordinary launches even if ambient Inductor settings enable graphs.
            options["options"] = {"triton.cudagraphs": False}
        self.compiled = compile_owned(
            graph.forward, backend=backend, fullgraph=True, dynamic=False, **options
        )


@torch.library.custom_op(
    "dendra_derivimplicit::detached_newton",
    mutates_args=(),
    tags=torch.Tag.cudagraph_unsafe,
)
def _detached_newton(
    program_id: str, operands: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    program = _PROGRAMS[program_id]
    with torch.no_grad():
        if _CAPTURE.get() is not None:
            # Real make_fx traces need the primal value, but cannot enter a
            # Dynamo-optimized callable while tracing the enclosing step.
            with _outside_step_trace():
                return program.graph(*operands)
        return program.compiled(*operands)


@_detached_newton.register_fake
def _(program_id, operands):
    x = operands[0]
    return x.new_empty(x.shape), x.new_empty(x.shape[:-1], dtype=torch.int64)


def _try_separate_newton(x, dt, make_evaluators, coefficients, options):
    capture = _CAPTURE.get()
    if capture is None or not all(torch.is_tensor(v) for v in (dt, *coefficients)):
        return None

    from ._derivimplicit import _get_while_loop, _solve_derivimplicit

    programs, backend = capture
    loop = _get_while_loop()
    operands = [value.detach() for value in (x, dt, *coefficients)]

    def primal(initial, step, *values):
        # Structured control flow requires independent lifted operands, even
        # when authored coefficients are read-only views of the same tensor.
        rhs, jacobian = make_evaluators(*(value.clone() for value in values))
        root, status = _solve_derivimplicit(
            initial,
            step,
            rhs,
            None,
            None,
            jacobian,
            **options,
            create_graph=False,
            detach_newton=True,
            loop=loop,
        )
        return root.contiguous().clone(), status.contiguous().clone()

    with _outside_step_trace(), torch.no_grad():
        graph = make_fx(primal)(*operands)
    program = _ForwardProgram(graph, backend, x.device)
    programs.append(program)
    _PROGRAMS[program.identifier] = program
    return _detached_newton(program.identifier, operands)
