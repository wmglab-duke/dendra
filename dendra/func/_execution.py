"""Descriptive execution contracts and tensor-free dispatch reports."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExecutionCapabilities:
    """Execution choices offered by one compiled functional kernel.

    These are Dendra's dispatch contracts, subject to the selected model and
    PyTorch backend supporting their tensor operations. They do not guarantee
    native code generation, speed, memory use, or successful compilation.
    ``compiled_higher_order_reverse_mode`` describes differentiating an already
    compiled kernel: transform fallback does not intercept ``create_graph=True``.
    """

    execution: Literal["default", "scan"]
    inference: bool
    compiled_reverse_mode: Literal["first_order"]
    compiled_higher_order_reverse_mode: Literal["backend_limited", "rejected"]
    functional_transforms: Literal["eager_fallback"]
    forward_mode: Literal["eager_fallback"]
    structured_inference: Literal["conditional_while_loop", "scan"]
    structured_training: bool


@dataclass(frozen=True)
class ExecutionReport:
    """The most recent successfully completed kernel dispatch.

    Tail and callback specializations share this report with their original
    kernel. Checkpoint replay can update it during backward. A failed call
    leaves the previous report unchanged, and an empty run makes no dispatch.
    This is not a summary of a whole host run or a performance measurement.

    A ``compiled_*`` strategy means Dendra called its ``torch.compile`` boundary;
    it does not promise native code generation by ``backend``. The while-loop
    strategy additionally means the captured inference loop was selected and
    the timestep count includes at least one loop iteration. Reports contain
    only scalar metadata and never retain live tensors or autograd graphs.
    """

    requested_execution: Literal["default", "scan"]
    strategy: Literal[
        "compiled_unrolled",
        "compiled_while_loop",
        "compiled_scan",
        "eager_transform_fallback",
        "eager_forward_ad_fallback",
        "eager_disabled",
    ]
    steps: int
    callbacks: bool
    grad_enabled: bool
    backend: str
    reason: str | None = None


class _ExecutionReportState:
    """A shared report slot with no references back to kernels or operands."""

    __slots__ = ("report",)

    def __init__(self):
        self.report: ExecutionReport | None = None


def _capabilities(execution: str) -> ExecutionCapabilities:
    return ExecutionCapabilities(
        execution=execution,
        inference=True,
        compiled_reverse_mode="first_order",
        compiled_higher_order_reverse_mode=(
            "rejected" if execution == "scan" else "backend_limited"
        ),
        functional_transforms="eager_fallback",
        forward_mode="eager_fallback",
        structured_inference=(
            "scan" if execution == "scan" else "conditional_while_loop"
        ),
        structured_training=execution == "scan",
    )


def _backend_name(compile_options) -> str:
    backend = compile_options.get("backend", "inductor")
    if isinstance(backend, str):
        return backend
    name = getattr(backend, "__qualname__", None)
    return name if isinstance(name, str) else type(backend).__name__


__all__ = ["ExecutionCapabilities", "ExecutionReport"]
