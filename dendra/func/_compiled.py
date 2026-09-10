"""Safe Python wrappers around tensor-only compiled functional kernels."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch

from ._lowered import (
    _LoweredCallbackPopulationChunk,
    _LoweredPopulationChunk,
    _PopulationChunkOperands,
)
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
        _stepwise_stimulation: bool = False,
    ):
        self._lowered = _LoweredPopulationChunk(
            functional,
            steps=steps,
            stepwise_stimulation=_stepwise_stimulation,
        )
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
        # Host schedules may end at a shorter tail. Cache only kernels here,
        # never bound parameters or preparation, so optimization graphs retain
        # their ordinary lifetime.
        self._runner_chunks: dict[int, CompiledPopulationChunk] = {}
        # A transient callback plan may own a large immutable target trace.
        # Lowerers retain its static entries, never the collection used here as
        # the weak key, so dropping that collection also releases its kernels.
        self._callback_chunks = weakref.WeakKeyDictionary()

        self._compiled = torch.compile(self._lowered, **options)

    @property
    def steps(self) -> int:
        """The exact number of timesteps consumed by each invocation."""
        return self._steps

    def _runner_chunk(self, steps: int) -> CompiledPopulationChunk:
        """Reuse a host-compatible specialization with these compile options."""
        stepwise_stimulation = steps > 1 and (
            self._functional.intra.enabled or self._functional.extra.enabled
        )
        if steps == self._steps and (
            not stepwise_stimulation or self._lowered._stepwise_stimulation
        ):
            return self
        if not 1 <= steps <= self._steps:
            raise ValueError("runner chunk steps must be between 1 and chunk.steps")
        if steps not in self._runner_chunks:
            self._runner_chunks[steps] = CompiledPopulationChunk(
                self._functional,
                steps=steps,
                compile_options=self._compile_options,
                _stepwise_stimulation=stepwise_stimulation,
            )
        return self._runner_chunks[steps]

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

    def _runner_callback_chunk(self, callbacks, steps: int):
        """Cache a joint transition/reducer kernel for one plan and width."""
        if not 1 <= steps <= self._steps:
            raise ValueError("runner chunk steps must be between 1 and chunk.steps")
        kernels = self._callback_chunks.setdefault(callbacks, {})
        if steps not in kernels:
            kernels[steps] = _CompiledCallbackPopulationChunk(
                self._functional,
                steps=steps,
                plans=callbacks._plans,
                compile_options=self._compile_options,
            )
        return kernels[steps]

    def _step_values(self, parameters, prepared, state, inputs: StepInput):
        """Execute one runner-validated frame without public Python checks."""
        if self._steps != 1:
            raise FunctionalizationError(
                "one-step execution requires a CompiledPopulationChunk with steps=1"
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

    def _rollout_values(self, parameters, prepared, state, inputs: RolloutInput):
        """Execute one runner-validated fixed chunk without public tree checks."""
        operands = _PopulationChunkOperands(
            parameters,
            prepared,
            state,
            inputs.ve,
            inputs.intra,
        )
        return self._execute_operands(
            operands,
            prewarm_structured=self._steps > 1,
        )

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


class _CompiledCallbackPopulationChunk:
    """Runner-owned tensor boundary for a joint model/callback recurrence."""

    def __init__(self, functional, *, steps, plans, compile_options):
        self._lowered = _LoweredCallbackPopulationChunk(
            functional, steps=steps, plans=plans
        )
        self._compiled = torch.compile(self._lowered, **compile_options)

    def _rollout_values(self, parameters, prepared, state, inputs, carries):
        operands = (parameters, prepared, state, inputs.ve, inputs.intra, carries)
        if (
            torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level >= 0
        ):
            return self._lowered(*operands)
        return self._compiled(*operands)


__all__ = ["CompiledPopulationChunk"]
