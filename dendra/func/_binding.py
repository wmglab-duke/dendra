"""Explicit, graph-local bindings for functional Population execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from ._types import FunctionalizationError, RolloutInput, StepInput

if TYPE_CHECKING:
    from ._compiled import CompiledPopulationChunk
    from ._population import FunctionalPopulation


@dataclass(frozen=True, eq=False, init=False)
class BoundPopulation:
    """A callable transition with explicit model ownership and prepared inputs.

    Construct with ``fmodel.bind(parameters, prepared)`` for one eager step,
    or ``kernel.bind(parameters, prepared)`` for a fixed compiled chunk.
    Parameter keys are snapshotted; tensor leaves remain live and differentiable.
    Replacing a caller mapping entry requires a new binding. In-place changes
    to preparation inputs are rejected by the usual freshness checks.

    Reuse a binding throughout one forward/backward graph, then prepare and
    bind again for the next training iteration. The reusable compiled kernel
    never owns this binding. Call ``run(bound, state, ...)`` to schedule a run.
    """

    _functional: FunctionalPopulation
    _parameters: dict[str, torch.Tensor]
    _prepared: object
    _compiled_chunk: CompiledPopulationChunk | None
    _eager_step: object

    def __init__(self, functional, parameters, prepared, *, compiled_chunk=None):
        from ._compiled import CompiledPopulationChunk
        from ._population import FunctionalPopulation

        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "BoundPopulation is an eager binding; use prepare_and_step() "
                "or prepare_and_rollout() inside torch.compile"
            )
        if type(functional) is not FunctionalPopulation:
            raise TypeError("functional must be an exact FunctionalPopulation")
        eager_step = functional.step
        if compiled_chunk is None and (
            getattr(eager_step, "__self__", None) is not functional
            or getattr(eager_step, "__func__", None) is not FunctionalPopulation.step
        ):
            raise FunctionalizationError(
                "eager binding requires the owned FunctionalPopulation.step; "
                "use a generic one-step callable with the legacy runner for adapters"
            )
        if compiled_chunk is not None:
            if type(compiled_chunk) is not CompiledPopulationChunk:
                raise TypeError(
                    "compiled_chunk must be an exact CompiledPopulationChunk"
                )
            if compiled_chunk._functional is not functional:
                raise FunctionalizationError(
                    "compiled chunk belongs to a different FunctionalPopulation plan"
                )
        functional._validate_source()
        functional._validate_parameters(parameters)
        snapshot = dict(parameters)
        functional._validate_prepared(prepared, snapshot)
        object.__setattr__(self, "_functional", functional)
        object.__setattr__(self, "_parameters", snapshot)
        object.__setattr__(self, "_prepared", prepared)
        object.__setattr__(self, "_compiled_chunk", compiled_chunk)
        object.__setattr__(
            self, "_eager_step", eager_step if compiled_chunk is None else None
        )

    @property
    def functional(self) -> FunctionalPopulation:
        return self._functional

    @property
    def parameters(self) -> Mapping[str, torch.Tensor]:
        """Read-only parameter mapping; its tensor leaves are not copied."""
        return MappingProxyType(self._parameters)

    @property
    def prepared(self):
        return self._prepared

    @property
    def compiled_chunk(self) -> CompiledPopulationChunk | None:
        return self._compiled_chunk

    @property
    def steps(self) -> int:
        """Exact timestep count consumed by a direct invocation."""
        return 1 if self._compiled_chunk is None else self._compiled_chunk.steps

    def __call__(self, state, inputs: RolloutInput | StepInput | None = None):
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "BoundPopulation is an eager binding; use prepare_and_step() "
                "or prepare_and_rollout() inside torch.compile"
            )
        # MappingProxyType is an opaque leaf to PyTorch's PyTree machinery.
        # Only the owned ordinary dict may cross a tensor execution boundary.
        if self._compiled_chunk is None:
            return self._eager_step(self._parameters, self._prepared, state, inputs)
        return self._compiled_chunk(self._parameters, self._prepared, state, inputs)


__all__ = ["BoundPopulation"]
