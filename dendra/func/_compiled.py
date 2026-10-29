"""Safe Python wrappers around tensor-only compiled functional kernels."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch

from dendra._bootstrap import isolate_compilation_callable
from dendra.helpers import MEMBRANE_ASSEMBLY

from ._binding import _BINDING_UNSET
from ._draws import _with_runtime_draws
from ._execution import (
    ExecutionCapabilities,
    ExecutionReport,
    _backend_name,
    _capabilities,
    _ExecutionReportState,
)
from ._inference import _canonicalize_inference_tree
from ._inputs import Inputs, _population_inputs, _RuntimeInputs
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

    ``inputs=Inputs(...)`` supplies exactly ``steps`` time-leading samples;
    ``draws=RuntimeDraws(...)`` supplies explicit stochastic tapes. Optional
    ``callbacks`` from ``functional.make_callbacks(...)`` use the same
    stimulation clock and return results in ``auxiliary['callbacks']``.
    Continue with the returned simulation state and
    ``callback_state=auxiliary['callbacks'].state``; resumed recording contains
    only samples produced by the new call.

    Bind raw sources with ``kernel.bind(tensors_or_continuation)`` and schedule
    arbitrary horizons with ``dn.func.run(bound, state_or_continuation, ...)``.
    Direct chunk calls retain the explicit parameters/prepared/state contract
    and return callback carry separately.
    """

    def __init__(
        self,
        functional: FunctionalPopulation,
        *,
        steps: int,
        compile_options: dict,
        execution: str = "default",
        _stepwise_stimulation: bool = False,
        _report_state: _ExecutionReportState | None = None,
    ):
        if execution not in ("default", "unrolled", "scan"):
            raise ValueError("execution must be 'default', 'unrolled', or 'scan'")
        self._execution = execution
        self._capabilities = _capabilities(execution)
        self._report_state = (
            _ExecutionReportState() if _report_state is None else _report_state
        )
        self._lowered = _LoweredPopulationChunk(
            functional,
            steps=steps,
            stepwise_stimulation=_stepwise_stimulation,
            force_unrolled=execution == "unrolled",
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
        self._noise_while = None
        if (
            execution == "default"
            and (
                functional._noise_plan is not None
                or getattr(functional, "_poisson_plan", None) is not None
            )
            and not options.get("disable", False)
        ):
            from ._recorder_while import _RecorderWhileExecutor

            self._noise_while = _RecorderWhileExecutor(
                functional,
                steps=steps,
                plans=(),
                compile_options=options,
                allow_empty_plans=True,
                stepwise_stimulation=_stepwise_stimulation,
            )

        if execution == "scan":
            from ._scan import _CompiledScanPopulationChunk

            self._compiled = _CompiledScanPopulationChunk(
                functional,
                steps=steps,
                compile_options=options,
                stepwise_stimulation=_stepwise_stimulation,
            )
        elif execution == "default":
            from ._while_inductor import _compile_while

            self._compiled = _compile_while(self._lowered.__call__, options)
        else:
            self._compiled = torch.compile(
                isolate_compilation_callable(self._lowered.__call__), **options
            )

    @property
    def steps(self) -> int:
        """The exact number of timesteps consumed by each invocation."""
        return self._steps

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """The immutable execution and differentiation contract for this kernel."""
        return self._capabilities

    def execution_report(self) -> ExecutionReport | None:
        """Return the last successful dispatch, including tails and replay.

        Returns ``None`` before any call completes. The report covers one kernel
        invocation, not an entire run; checkpoint backward can replace it.
        """
        return self._report_state.report

    def bind(self, parameters, prepared=_BINDING_UNSET, *, constants=_BINDING_UNSET):
        """Bind a matching tensor bundle, Continuation, or explicit raw/prepared sources.

        Packet forms select only parameters/constants and cannot be combined
        with ``prepared`` or ``constants``. State and callback carry remain
        explicit operands of ``dn.func.run``.
        """
        from ._binding import BoundPopulation

        return BoundPopulation(
            self._functional,
            parameters,
            prepared,
            constants=constants,
            compiled_chunk=self,
        )

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
                execution=self._execution,
                _stepwise_stimulation=stepwise_stimulation,
                _report_state=self._report_state,
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
        transform_active = torch._C._are_functorch_transforms_active()
        forward_ad_active = torch.autograd.forward_ad._current_level >= 0
        grad_enabled = torch.is_grad_enabled()
        if transform_active or forward_ad_active:
            result = self._lowered(*operands)
            strategy = (
                "eager_transform_fallback"
                if transform_active
                else "eager_forward_ad_fallback"
            )
            self._record_execution(strategy, grad_enabled=grad_enabled)
            return result
        if not self._compile_options.get("disable", False):
            state = _canonicalize_inference_tree(operands.state)
            if state is not operands.state:
                operands = operands._replace(state=state)
        if (
            self._noise_while is not None
            and (
                operands.noise is not None
                or operands.stimulation is not None
                or operands.draws is not None
            )
            and not grad_enabled
        ):
            kernel, reason = self._noise_while.kernel_for(
                operands.parameters,
                operands.prepared,
                operands.state,
                operands.ve,
                operands.intra,
                (),
                operands.noise,
                operands.stimulation,
                operands.draws,
            )
            if kernel is not None:
                final, auxiliary, _carries, _emitted = kernel(
                    operands.parameters,
                    operands.prepared,
                    operands.state,
                    operands.ve,
                    operands.intra,
                    (),
                    operands.noise,
                    operands.stimulation,
                    operands.draws,
                )
                self._record_execution("compiled_while_loop", grad_enabled=False)
                return final, auxiliary
        if (
            self._execution == "default"
            and operands.noise is None
            and operands.stimulation is None
            and operands.draws is None
            and prewarm_structured
            and not torch.is_grad_enabled()
        ):
            functional = self._functional
            functional._ensure_structured_step_graph(
                operands.ve is not None or functional.extra.enabled,
                operands.intra is not None or functional.intra.enabled,
            )
        strategy, reason = self._compiled_strategy(operands, grad_enabled)
        result = self._compiled(*operands)
        self._record_execution(strategy, grad_enabled=grad_enabled, reason=reason)
        return result

    def _compiled_strategy(self, operands, grad_enabled):
        """Mirror the lowered inference dispatch without inspecting live graphs."""
        if self._compile_options.get("disable", False):
            return "eager_disabled", "Compilation was explicitly disabled."
        if self._execution == "scan":
            return "compiled_scan", None
        if self._execution == "unrolled":
            return "compiled_unrolled", "Unrolled execution was explicitly selected."
        if grad_enabled:
            return (
                "compiled_unrolled",
                "Gradient recording uses the unrolled recurrence.",
            )
        if self._lowered._stepwise_stimulation:
            return (
                "compiled_unrolled",
                "Bound waveforms are sampled at each accepted timestep.",
            )
        # Keep the threshold identical to the actual loop body. A prewarmed
        # one-/two-/three-step call executes only the unrolled tail.
        from ._population import _STRUCTURED_ROLLOUT_CHUNK

        if self._steps < _STRUCTURED_ROLLOUT_CHUNK:
            return (
                "compiled_unrolled",
                "The chunk is shorter than one inference loop block.",
            )
        if (
            operands.noise is not None
            or operands.stimulation is not None
            or operands.draws is not None
        ):
            return (
                "compiled_unrolled",
                "No captured stochastic inference step graph is available.",
            )
        signature = (
            operands.ve is not None or self._functional.extra.enabled,
            operands.intra is not None or self._functional.intra.enabled,
            MEMBRANE_ASSEMBLY.value,
        )
        if signature in self._functional._structured_step_graphs:
            return "compiled_while_loop", None
        return "compiled_unrolled", "No captured inference step graph is available."

    def _record_execution(self, strategy, *, grad_enabled, reason=None):
        self._report_state.report = ExecutionReport(
            requested_execution=self._execution,
            strategy=strategy,
            steps=self._steps,
            callbacks=False,
            grad_enabled=grad_enabled,
            backend=_backend_name(self._compile_options),
            reason=reason,
        )

    def _runner_callback_chunk(
        self, callbacks, steps: int, *, stepwise_stimulation: bool = True
    ):
        """Cache one joint recurrence with its caller's stimulation clock."""
        if not 1 <= steps <= self._steps:
            raise ValueError("runner chunk steps must be between 1 and chunk.steps")
        kernels = self._callback_chunks.setdefault(callbacks, {})
        key = (steps, stepwise_stimulation)
        if key not in kernels:
            kernels[key] = _CompiledCallbackPopulationChunk(
                self._functional,
                steps=steps,
                plans=callbacks._plans,
                compile_options=self._compile_options,
                execution=self._execution,
                report_state=self._report_state,
                stepwise_stimulation=stepwise_stimulation,
            )
        return kernels[key]

    def _step_values(self, parameters, prepared, state, inputs: _RuntimeInputs):
        """Execute one runner-validated frame without public Python checks."""
        if self._steps != 1:
            raise FunctionalizationError(
                "one-step execution requires a CompiledPopulationChunk with steps=1"
            )
        inputs = _population_inputs(inputs, step=True)
        operands = _PopulationChunkOperands(
            parameters,
            prepared,
            state,
            None if inputs.ve is None else inputs.ve.unsqueeze(0),
            None if inputs.intra is None else inputs.intra.unsqueeze(0),
            None,
            None
            if inputs.noise is None
            else torch.utils._pytree.tree_map_only(
                torch.Tensor, lambda value: value.unsqueeze(0), inputs.noise
            ),
            None
            if inputs.stimulation is None
            else torch.utils._pytree.tree_map_only(
                torch.Tensor, lambda value: value.unsqueeze(0), inputs.stimulation
            ),
            None
            if inputs.draws is None
            else torch.utils._pytree.tree_map_only(
                torch.Tensor, lambda value: value.unsqueeze(0), inputs.draws
            ),
        )
        # A one-step kernel has no large rollout graph to bound, so the
        # inference-only structured prewarm is unnecessary in this hot path.
        return self._execute_operands(operands, prewarm_structured=False)

    def _rollout_values(self, parameters, prepared, state, inputs: _RuntimeInputs):
        """Execute one runner-validated fixed chunk without public tree checks."""
        inputs = _population_inputs(inputs)
        operands = _PopulationChunkOperands(
            parameters,
            prepared,
            state,
            inputs.ve,
            inputs.intra,
            None,
            inputs.noise,
            inputs.stimulation,
            inputs.draws,
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
        inputs: Inputs | RolloutInput | StepInput | None = None,
        *,
        callbacks=None,
        callback_state=None,
        draws=None,
    ):
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "CompiledPopulationChunk is already an outer Python wrapper "
                "around a compiled tensor kernel; call it directly instead of "
                "passing it through torch.compile"
            )
        legacy_step = isinstance(inputs, StepInput)
        if draws is not None or legacy_step:
            inputs = _with_runtime_draws(inputs, draws, step=legacy_step)
        if legacy_step:
            if self._steps != 1:
                raise TypeError(
                    "StepInput is accepted only by a one-step compiled chunk"
                )
            for name, value in zip(("ve", "intra"), inputs):
                if value is not None and not torch.is_tensor(value):
                    raise TypeError(f"{name} must be a Tensor or None")
            inputs = _RuntimeInputs(
                ve=None if inputs.ve is None else inputs.ve.unsqueeze(0),
                intra=None if inputs.intra is None else inputs.intra.unsqueeze(0),
                noise=None
                if inputs.noise is None
                else torch.utils._pytree.tree_map_only(
                    torch.Tensor, lambda value: value.unsqueeze(0), inputs.noise
                ),
                stimulation=None
                if inputs.stimulation is None
                else torch.utils._pytree.tree_map_only(
                    torch.Tensor, lambda value: value.unsqueeze(0), inputs.stimulation
                ),
                draws=None
                if inputs.draws is None
                else torch.utils._pytree.tree_map_only(
                    torch.Tensor, lambda value: value.unsqueeze(0), inputs.draws
                ),
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
        resolved, _resolved_steps = functional._resolve_rollout_input_packets(
            inputs,
            self._steps,
        )
        if callbacks is not None:
            from ._callbacks import FunctionalCallbacks, _setup_callback_state
            from ._runners import _attach_callback_results

            if not isinstance(callbacks, FunctionalCallbacks):
                raise TypeError(
                    "callbacks must be a FunctionalCallbacks plan created by "
                    "functional.make_callbacks(...)"
                )
            callbacks._validate_owner(functional)
            callbacks._validate_runtime_dt_tensor(prepared_values["integrator"]["dt"])
            previous = self._report_state.report
            try:
                callback_state, parts = _setup_callback_state(
                    callbacks, callback_state, state
                )
                kernel = self._runner_callback_chunk(
                    callbacks,
                    self._steps,
                    stepwise_stimulation=self._lowered._stepwise_stimulation,
                )
                final, auxiliary, carries, emitted = kernel._rollout_values(
                    parameters, prepared_values, state, resolved, callback_state.carries
                )
                callback_state = callbacks._wrap_compiled_update(
                    callback_state, carries, emitted
                )
                if callbacks._has_emissions(emitted):
                    parts.append(emitted)
                return final, _attach_callback_results(
                    auxiliary, callbacks, callback_state, parts
                )
            except BaseException:
                self._report_state.report = previous
                raise
        if callback_state is not None:
            raise ValueError("callback_state requires callbacks")
        operands = _PopulationChunkOperands(
            parameters,
            prepared_values,
            state,
            resolved.ve,
            resolved.intra,
            None,
            resolved.noise,
            resolved.stimulation,
            resolved.draws,
        )
        return self._execute_operands(operands, prewarm_structured=True)


class _CompiledCallbackPopulationChunk:
    """Runner-owned tensor boundary for a joint model/callback recurrence."""

    def __init__(
        self,
        functional,
        *,
        steps,
        plans,
        compile_options,
        execution="default",
        report_state,
        stepwise_stimulation=True,
    ):
        self._execution = execution
        self._steps = steps
        self._compile_options = compile_options
        self._report_state = report_state
        self._lowered = _LoweredCallbackPopulationChunk(
            functional,
            steps=steps,
            plans=plans,
            stepwise_stimulation=stepwise_stimulation,
        )
        self._recorder_while = None
        if execution == "default" and not compile_options.get("disable", False):
            from ._recorder_while import _RecorderWhileExecutor

            self._recorder_while = _RecorderWhileExecutor(
                functional,
                steps=steps,
                plans=plans,
                compile_options=compile_options,
                stepwise_stimulation=stepwise_stimulation,
            )
        if execution == "scan":
            from ._scan import _CompiledScanPopulationChunk

            self._compiled = _CompiledScanPopulationChunk(
                functional,
                steps=steps,
                plans=plans,
                compile_options=compile_options,
                stepwise_stimulation=stepwise_stimulation,
            )
        else:
            self._compiled = torch.compile(
                isolate_compilation_callable(self._lowered.__call__), **compile_options
            )

    def _rollout_values(self, parameters, prepared, state, inputs, carries):
        inputs = _population_inputs(inputs)
        operands = (
            parameters,
            prepared,
            state,
            inputs.ve,
            inputs.intra,
            carries,
            inputs.noise,
            inputs.stimulation,
            inputs.draws,
        )
        transform_active = torch._C._are_functorch_transforms_active()
        forward_ad_active = torch.autograd.forward_ad._current_level >= 0
        grad_enabled = torch.is_grad_enabled()
        reason = None
        if transform_active or forward_ad_active:
            result = self._lowered(*operands)
            strategy = (
                "eager_transform_fallback"
                if transform_active
                else "eager_forward_ad_fallback"
            )
        else:
            if not self._compile_options.get("disable", False):
                state, carries = _canonicalize_inference_tree((state, carries))
                operands = (
                    parameters,
                    prepared,
                    state,
                    inputs.ve,
                    inputs.intra,
                    carries,
                    inputs.noise,
                    inputs.stimulation,
                    inputs.draws,
                )
            while_kernel = None
            if self._recorder_while is not None and not grad_enabled:
                while_kernel, reason = self._recorder_while.kernel_for(*operands)
            if while_kernel is not None:
                result = while_kernel(*operands)
                strategy = "compiled_while_loop"
            else:
                result = self._compiled(*operands)
                if self._compile_options.get("disable", False):
                    strategy = "eager_disabled"
                    reason = "Compilation was explicitly disabled."
                elif self._execution == "scan":
                    strategy = "compiled_scan"
                else:
                    strategy = "compiled_unrolled"
                    if self._execution == "unrolled":
                        reason = "Unrolled execution was explicitly selected."
                    elif grad_enabled:
                        reason = (
                            "Gradient recording uses the unrolled callback recurrence."
                        )
                    elif reason is None:
                        reason = "Callback updates use the unrolled recurrence."
        self._report_state.report = ExecutionReport(
            requested_execution=self._execution,
            strategy=strategy,
            steps=self._steps,
            callbacks=True,
            grad_enabled=grad_enabled,
            backend=_backend_name(self._compile_options),
            reason=reason,
        )
        return result


__all__ = ["CompiledPopulationChunk"]
