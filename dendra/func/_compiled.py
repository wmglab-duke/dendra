"""Safe Python wrappers around tensor-only compiled functional kernels."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch

from ._lowered import _LoweredPopulationChunk, _PopulationChunkOperands
from ._types import FunctionalizationError, RolloutInput, StepInput

if TYPE_CHECKING:
    from ._population import FunctionalPopulation


class CompiledPopulationChunk:
    """A fixed-length compiled rollout with preparation hoisted by the caller.

    Construct this object with :meth:`FunctionalPopulation.compile_rollout_chunk`.
    The outer Python call retains source, schema, and prepared-plan freshness
    checks. Only explicit tensor trees are passed to the internal
    :func:`torch.compile` boundary, so one prepared plan can be reused across
    multiple differentiable chunk calls without rebuilding its workspaces.

    Source structure is checked when this immutable lowered-plan wrapper is
    constructed. Later edits to the imperative source Population do not alter
    the compiled kernel; create a new functional plan after structural edits.
    """

    def __init__(
        self,
        functional: FunctionalPopulation,
        *,
        steps: int,
        compile_options: dict,
    ):
        self._lowered = _LoweredPopulationChunk(functional, steps=steps)
        self._functional = functional
        self._steps = self._lowered.steps
        # A preparation can own an autograd graph derived from parameters and
        # geometry. Keep only a non-owning reference here so the wrapper does
        # not extend that graph's lifetime after the caller finishes a
        # forward/backward iteration.
        self._cached_prepared_ref = None
        options = dict(compile_options)
        options.setdefault("fullgraph", True)
        options.setdefault("dynamic", False)
        self._compile_options = options

        self._compiled = torch.compile(self._lowered, **options)

    @property
    def steps(self) -> int:
        """The exact number of timesteps consumed by each invocation."""
        return self._steps

    def _execute_operands(self, operands, *, prewarm_structured: bool):
        """Execute validated tensor operands through the owned kernel."""
        # Dynamo/AOTAutograd cannot currently be entered from an active
        # functorch transform, and compiled autograd functions do not expose a
        # direct forward-AD rule consistently across backends. The same
        # tensor-only transition is already transform-safe in eager mode, so
        # preserve vmap/jacrev/jacfwd and dual-tensor semantics without
        # attempting nested compilation.
        if (
            torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level >= 0
        ):
            return self._lowered(*operands)
        if prewarm_structured and not torch.is_grad_enabled():
            functional = self._functional
            functional._ensure_structured_step_graph(
                operands.ve is not None or functional.extra.enabled,
                operands.intra is not None or functional.intra.enabled,
            )
        return self._compiled(*operands)

    def _step_values(self, parameters, prepared, state, inputs: StepInput):
        """Execute one runner-validated frame without public Python checks."""
        if self._steps != 1:  # Defensive: runner recognition also enforces this.
            raise FunctionalizationError(
                "host schedulers require a one-step CompiledPopulationChunk"
            )
        operands = _PopulationChunkOperands(
            parameters,
            prepared,
            state,
            None if inputs.ve is None else inputs.ve.unsqueeze(0),
            None if inputs.intra is None else inputs.intra.unsqueeze(0),
        )
        # A one-step kernel has no large rollout graph to bound, so the
        # inference-only structured prewarm is unnecessary in this hot path.
        return self._execute_operands(operands, prewarm_structured=False)

    def __call__(
        self,
        parameters,
        prepared,
        state,
        inputs: RolloutInput | StepInput | None = None,
    ):
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "CompiledPopulationChunk is already an outer Python wrapper "
                "around a compiled tensor kernel; call it directly instead of "
                "passing it through torch.compile"
            )
        if isinstance(inputs, StepInput):
            if self._steps != 1:
                raise TypeError(
                    "StepInput is accepted only by a one-step compiled chunk"
                )
            for name, value in zip(("ve", "intra"), inputs):
                if value is not None and not torch.is_tensor(value):
                    raise TypeError(f"{name} must be a Tensor or None")
            inputs = RolloutInput(
                ve=None if inputs.ve is None else inputs.ve.unsqueeze(0),
                intra=None if inputs.intra is None else inputs.intra.unsqueeze(0),
            )
        functional = self._functional
        functional._validate_state(state)
        # Prepared freshness intentionally depends only on Population dynamics
        # parameters and geometry. Stimulation leaves may therefore be replaced
        # without preparing again, but every replacement must still satisfy the
        # complete public parameter schema, including on the weak-cache path.
        functional._validate_parameters(parameters)
        cached_prepared = (
            None if self._cached_prepared_ref is None else self._cached_prepared_ref()
        )
        if prepared is cached_prepared:
            prepared_values = functional._validate_prepared_freshness(
                prepared,
                parameters,
            )
        else:
            prepared_values = functional._validate_prepared(prepared, parameters)
            self._cached_prepared_ref = weakref.ref(prepared)
        ve, intra, _resolved_steps = functional._resolve_rollout_inputs(
            inputs,
            self._steps,
        )
        operands = _PopulationChunkOperands(
            parameters,
            prepared_values,
            state,
            ve,
            intra,
        )
        return self._execute_operands(operands, prewarm_structured=True)


__all__ = ["CompiledPopulationChunk"]
