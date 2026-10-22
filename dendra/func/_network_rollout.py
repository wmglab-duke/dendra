"""Network adapters for shared tensor-tree scans and host checkpoint spans."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from typing import NamedTuple

import torch
from torch.utils import _pytree as pytree

from ._callbacks import _update_callback_values
from ._execution import (
    ExecutionCapabilities,
    ExecutionReport,
    _backend_name,
    _capabilities,
    _ExecutionReportState,
)
from ._netstim import NetStimDraw, NetStimOutputs
from ._network import _schema, _tensor_paths, _validate_tree
from ._network_bindings import _canonicalize_mapping_tree
from ._population import _tensor_tree_binding
from ._runners import (
    _attach_callback_results,
    _checkpoint_tensor_tree,
    _execution_spans,
    _prepare_callback_collection,
    _validate_chunklength,
)
from ._types import FunctionalizationError, RolloutInput, StepInput


class NetworkRolloutInput(NamedTuple):
    """Time-leading per-Population inputs and optional explicit NetStim draws.

    ``populations`` maps names to :class:`RolloutInput`. ``netstim`` is a
    :class:`NetStimDraw` whose two Tensor fields have one leading time axis.
    Keep draw values and their matching post-draw RNG bytes together. Pure
    rollout/chunk calls require a supplied tape when NetStim is present;
    the host ``run`` scheduler can sample bounded tapes from explicit carry.
    """

    populations: Mapping[str, RolloutInput] | None = None
    netstim: NetStimDraw | None = None


def _steps(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("steps must be a non-negative integer")
    return value


def _slice(tape, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], tape)


def _sample(tape, index):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[index], tape)


def _binding(tree):
    # Public packets accept arbitrary Mappings, including unregistered PyTree
    # containers. Traverse the same structural contract as Network validation.
    values = tuple(value for _, value in _tensor_paths(tree))
    # Checkpoint operands clone inference tensors into independent, versioned
    # snapshots. Their original owners have no version counter to fingerprint.
    inference = tuple(
        id(value) if torch.is_inference(value) else None for value in values
    )
    return (
        _schema(tree),
        _tensor_tree_binding(
            tuple(value for value in values if not torch.is_inference(value))
        ),
        inference,
    )


def _resolve_inputs(functional, inputs, steps, *, allow_missing_draws=False):
    if inputs is None:
        inputs = NetworkRolloutInput()
    if not isinstance(inputs, NetworkRolloutInput):
        raise TypeError("inputs must be a NetworkRolloutInput or None")
    populations = {} if inputs.populations is None else inputs.populations
    if not isinstance(populations, Mapping) or not set(populations) <= set(
        functional.populations
    ):
        raise ValueError(
            "populations must map known Network names to RolloutInput tapes"
        )
    lengths = []
    for name, selected in populations.items():
        if not isinstance(selected, RolloutInput):
            raise TypeError(f"Population {name!r} inputs must be a RolloutInput")
        for value in selected:
            if value is not None:
                if not torch.is_tensor(value) or value.ndim < 1:
                    raise ValueError("Population input tapes need a leading time axis")
                lengths.append(value.shape[0])
    if inputs.netstim is not None:
        if functional.netstim is None:
            raise ValueError("A Network without NetStim requires netstim=None")
        if not isinstance(inputs.netstim, NetStimDraw):
            raise TypeError("netstim must be a NetStimDraw tape")
        for value in inputs.netstim:
            if not torch.is_tensor(value) or value.ndim < 1:
                raise ValueError("NetStim draw tapes need a leading time axis")
            lengths.append(value.shape[0])
    if steps is None:
        if not lengths:
            raise ValueError("steps is required when no input tape supplies a horizon")
        steps = lengths[0]
    steps = _steps(steps)
    if any(length != steps for length in lengths):
        raise ValueError("All Network input tapes must have exactly steps samples")
    normalized = {}
    for name, selected in populations.items():
        ve, intra, _ = functional.populations[name]._resolve_rollout_inputs(
            selected, steps
        )
        selected = {
            key: value
            for key, value in (("ve", ve), ("intra", intra))
            if value is not None
        }
        if selected:
            normalized[name] = selected
    draws = inputs.netstim
    if functional.netstim is not None:
        if draws is None:
            if steps and not allow_missing_draws:
                raise ValueError(
                    "NetStim requires an explicit draw/RNG tape; use sample_draw_tape"
                )
        else:
            plan = functional.netstim
            shape, dtype, device = plan._state_schema["rng_state"]
            plan._validate_tensor(
                draws.exp_rand,
                shape=(steps, *plan.shape),
                dtype=plan.dtype,
                device=plan.device,
                label="netstim.exp_rand",
            )
            plan._validate_tensor(
                draws.rng_state,
                shape=(steps, *shape),
                dtype=dtype,
                device=device,
                label="netstim.rng_state",
            )
    tape = {"populations": normalized}
    if draws is not None:
        tape["netstim"] = draws
    return tape, steps


def sample_draw_tape(functional, state, steps):
    """Sample sequentially from an independent generator, never a batched rand."""
    if torch.compiler.is_compiling():
        raise FunctionalizationError("Sample draw tapes outside torch.compile")
    functional._validate_source()
    functional._validate_state(state)
    steps = _steps(steps)
    plan = functional.netstim
    if plan is None:
        return None
    if not steps:
        shape, dtype, device = plan._state_schema["rng_state"]
        return NetStimDraw(
            torch.empty((0, *plan.shape), device=plan.device, dtype=plan.dtype),
            torch.empty((0, *shape), dtype=dtype, device=device),
        )
    current = state["netstim"]
    draws = []
    for _ in range(steps):
        packet = plan.sample_step_draw(current)
        draws.append(packet)
        current = {**current, "rng_state": packet.rng_state}
    return NetStimDraw(
        torch.stack([draw.exp_rand for draw in draws]),
        torch.stack([draw.rng_state for draw in draws]),
    )


def _auxiliary(functional, state, *, scan=False):
    stimulus = None
    if scan and functional.netstim is not None:
        plan = functional.netstim
        stimulus = NetStimOutputs(
            torch.zeros(plan.shape, dtype=torch.bool, device=plan.device),
            torch.zeros(plan.shape, dtype=plan.dtype, device=plan.device),
        )
    return {
        "populations": {
            name: {"v": state["populations"][name]["integrator"]["v"]}
            for name in functional.populations
        },
        "netstim": stimulus,
    }


def _step(functional, parameters, prepared, state, sample):
    inputs = {
        name: StepInput(selected.get("ve"), selected.get("intra"))
        for name, selected in sample.get("populations", {}).items()
    }
    return functional._step_values(
        parameters, prepared, state, inputs=inputs, netstim_draw=sample.get("netstim")
    )


def _eager_values(
    functional, parameters, prepared, state, tape, steps, callbacks, callback_state
):
    auxiliary = _auxiliary(functional, state)
    emissions = []
    for index in range(steps):
        state, auxiliary = _step(
            functional, parameters, prepared, state, _sample(tape, index)
        )
        if callbacks is not None:
            callback_state, emitted = callbacks._update(
                callback_state, state, auxiliary
            )
            emissions.append(emitted)
    stacked = None if callbacks is None else callbacks._stack(emissions)
    return state, auxiliary, callback_state, stacked


def _validate(functional, parameters, state, callbacks, callback_state):
    functional._validate_source()
    functional._validate_parameters(parameters)
    functional._validate_state(state)
    with nullcontext() if functional.training else torch.no_grad():
        return _prepare_callback_collection(
            functional, callbacks, callback_state, state, functional.dt
        )


def rollout(
    functional,
    parameters,
    prepared,
    state,
    inputs=None,
    *,
    steps=None,
    callbacks=None,
    callback_state=None,
    constants=None,
):
    (parameters, prepared, state, inputs, constants), _ = _canonicalize_mapping_tree(
        (parameters, prepared, state, inputs, constants)
    )
    callbacks, callback_state, parts = _validate(
        functional, parameters, state, callbacks, callback_state
    )
    tape, steps = _resolve_inputs(functional, inputs, steps)
    if constants is None:
        values = functional._validate_prepared(prepared, parameters)
    else:
        _validate_tree(constants, functional._constant_schema, "constants")
        values = functional._prepare_values(parameters, constants)
    with nullcontext() if functional.training else torch.no_grad():
        state, auxiliary, callback_state, emitted = _eager_values(
            functional,
            parameters,
            values,
            state,
            tape,
            steps,
            callbacks,
            callback_state,
        )
    functional.check_overflow(state)
    if callbacks is not None and callbacks._has_emissions(emitted):
        parts.append(emitted)
    return state, _attach_callback_results(auxiliary, callbacks, callback_state, parts)


class CompiledNetworkChunk:
    """A fixed-horizon Network adapter over shared tensor-tree loop engines.

    ``execution='scan'`` preserves the existing first-order reverse-mode scan
    contract. ``execution='unrolled'`` uses the same authored body without a
    HOP. ``execution='while'`` captures an inference-only bounded loop using
    Population's packed carry and invocation-local recording machinery.
    Public calls validate source/schema/preparation before entering the
    tensor-only kernel. Kernel caches retain no live input/preparation graphs.
    """

    def __init__(
        self,
        functional,
        *,
        steps,
        execution="scan",
        _report_state=None,
        **compile_options,
    ):
        if _steps(steps) < 1:
            raise ValueError("compiled chunks require a positive step count")
        if execution not in {"scan", "unrolled", "while"}:
            raise ValueError(
                "Network chunk execution must be 'scan', 'unrolled', or 'while'"
            )
        functional._validate_source()
        if execution == "scan":
            from ._scan_compat import install_scan_compatibility

            # Capability probes can enable gradients. Select compatibility at
            # the public cold boundary, before any checkpoint forward/replay.
            install_scan_compatibility()
        self._functional, self._steps, self._execution = functional, steps, execution
        self._capabilities = _capabilities(execution)
        self._report_state = (
            _ExecutionReportState() if _report_state is None else _report_state
        )
        self._options = dict(compile_options)
        self._options.setdefault("fullgraph", True)
        self._options.setdefault("dynamic", False)
        self._plain_kernel = None
        self._callback_kernels = weakref.WeakKeyDictionary()
        self._tails = {}

    @property
    def steps(self):
        return self._steps

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """The immutable execution and differentiation contract for this chunk."""
        return self._capabilities

    def execution_report(self) -> ExecutionReport | None:
        """Describe the latest successful dispatch, including tails and replay.

        Reports contain scalar metadata only. An empty host run makes no
        dispatch, and checkpoint replay may replace the report during backward.
        """
        return self._report_state.report

    def _record_execution(self, strategy, *, callbacks, grad_enabled, reason=None):
        self._report_state.report = ExecutionReport(
            requested_execution=self._execution,
            strategy=strategy,
            steps=self.steps,
            callbacks=callbacks is not None,
            grad_enabled=grad_enabled,
            backend=_backend_name(self._options),
            reason=reason,
        )

    def _runner_chunk(self, steps):
        if steps == self.steps:
            return self
        if not 1 <= steps <= self.steps:
            raise ValueError("tail steps must be between 1 and chunk.steps")
        if steps not in self._tails:
            self._tails[steps] = CompiledNetworkChunk(
                self._functional,
                steps=steps,
                execution=self._execution,
                _report_state=self._report_state,
                **self._options,
            )
        return self._tails[steps]

    def _validate_while_callbacks(self, callbacks):
        if self._execution != "while" or callbacks is None:
            return
        from ._callbacks import _BoundRecorder

        if any(type(plan.callback) is not _BoundRecorder for plan in callbacks._plans):
            raise FunctionalizationError(
                "Network execution='while' supports qualified Recorders; "
                "use scan for custom callbacks or reducers."
            )

    def _make_kernel(self, callbacks):
        functional = self._functional
        plans = None if callbacks is None else callbacks._plans

        def body(parameters, prepared, loop, carries, sample):
            state, auxiliary = _step(
                functional, parameters, prepared, loop["network"], sample
            )
            emitted = {}
            if plans is not None:
                carries, emitted = _update_callback_values(
                    plans, carries, state, auxiliary
                )
            return ({"network": state, "auxiliary": auxiliary}, carries), emitted or {}

        if self._execution == "while":
            from ._tensor_tree_while import _CompiledTensorTreeWhile

            def clone_loop(loop):
                return {
                    "network": functional._clone_state_tree(loop["network"]),
                    "auxiliary": pytree.tree_map_only(
                        torch.Tensor,
                        lambda value: value.clone(
                            memory_format=torch.contiguous_format
                        ),
                        loop["auxiliary"],
                    ),
                }

            return _CompiledTensorTreeWhile(
                body,
                steps=self.steps,
                compile_options=self._options,
                device=functional.device,
                plans=plans,
                clone_state=clone_loop,
            )

        if self._execution == "unrolled":
            from dendra._bootstrap import isolate_compilation_callable

            steps = self.steps

            def execute(parameters, prepared, loop, carries, tape):
                emissions = []
                for index in range(steps):
                    (loop, carries), emitted = body(
                        parameters, prepared, loop, carries, _sample(tape, index)
                    )
                    emissions.append(emitted)
                stacked = pytree.tree_map(
                    lambda *values: torch.stack(values), *emissions
                )
                return loop, carries, stacked

            return torch.compile(isolate_compilation_callable(execute), **self._options)
        from ._scan_tree import _CompiledTensorTreeScan

        static = [
            ("network", "clock", "origin"),
            ("network", "clock", "duration_remainder"),
        ]
        detached = [("network", "clock", "t")]
        for name in functional.populations:
            static.append(
                ("network", "populations", name, "control", "duration_remainder")
            )
            detached.append(("network", "populations", name, "clock", "t"))
        return _CompiledTensorTreeScan(
            body,
            steps=self.steps,
            compile_options=self._options,
            device=functional.device,
            plans=plans,
            static_state_paths=static,
            detached_state_paths=detached,
        )

    def _values(self, parameters, prepared, state, tape, callbacks, callback_state):
        self._validate_while_callbacks(callbacks)
        transform = torch._C._are_functorch_transforms_active()
        forward_ad = torch.autograd.forward_ad._current_level >= 0
        disabled = self._options.get("disable", False)
        grad_enabled = torch.is_grad_enabled()
        if transform or forward_ad or disabled:
            result = _eager_values(
                self._functional,
                parameters,
                prepared,
                state,
                tape,
                self.steps,
                callbacks,
                callback_state,
            )
            self._functional.check_overflow(result[0])
            strategy = (
                "eager_transform_fallback"
                if transform
                else "eager_forward_ad_fallback"
                if forward_ad
                else "eager_disabled"
            )
            self._record_execution(
                strategy,
                callbacks=callbacks,
                grad_enabled=grad_enabled,
                reason=(
                    "Compilation was explicitly disabled."
                    if strategy == "eager_disabled"
                    else None
                ),
            )
            return result
        if self._execution == "while" and grad_enabled:
            raise FunctionalizationError(
                "Network execution='while' is inference-only; use torch.no_grad() "
                "or scan for gradients."
            )
        if callbacks is None:
            if self._plain_kernel is None:
                self._plain_kernel = self._make_kernel(None)
            kernel = self._plain_kernel
        else:
            if callbacks not in self._callback_kernels:
                self._callback_kernels[callbacks] = self._make_kernel(callbacks)
            kernel = self._callback_kernels[callbacks]
        loop = {
            "network": state,
            "auxiliary": _auxiliary(self._functional, state, scan=True),
        }
        carries = {} if callback_state is None else callback_state.carries
        try:
            loop, carries, emitted = kernel(parameters, prepared, loop, carries, tape)
        except Exception as error:
            if self._execution == "while":
                raise FunctionalizationError(
                    f"Network execution='while' is unavailable: {error}. "
                    "Use scan for this configuration."
                ) from error
            raise
        self._functional.check_overflow(loop["network"])
        if callbacks is not None:
            emitted = {plan.name: emitted.get(plan.name) for plan in callbacks._plans}
            callback_state = callbacks._wrap_compiled_update(
                callback_state, carries, emitted
            )
        from ._population import _STRUCTURED_ROLLOUT_CHUNK

        while_loop = (
            self._execution == "while" and self.steps >= _STRUCTURED_ROLLOUT_CHUNK
        )
        self._record_execution(
            "compiled_while_loop"
            if while_loop
            else "compiled_scan"
            if self._execution == "scan"
            else "compiled_unrolled",
            callbacks=callbacks,
            grad_enabled=grad_enabled,
            reason=(
                "Unrolled execution was explicitly selected."
                if self._execution == "unrolled"
                else "The chunk is shorter than one inference loop block."
                if self._execution == "while" and not while_loop
                else None
            ),
        )
        return loop["network"], loop["auxiliary"], callback_state, emitted

    def __call__(
        self,
        parameters,
        prepared,
        state,
        inputs=None,
        *,
        callbacks=None,
        callback_state=None,
    ):
        previous = self._report_state.report
        try:
            # Mapping accessors may execute external Python. Finish all public
            # packet traversal before validating source/preparation freshness.
            (parameters, prepared, state, inputs), _ = _canonicalize_mapping_tree(
                (parameters, prepared, state, inputs)
            )
            functional = self._functional
            callbacks, callback_state, parts = _validate(
                functional, parameters, state, callbacks, callback_state
            )
            tape, _ = _resolve_inputs(functional, inputs, self.steps)
            values = functional._validate_prepared(prepared, parameters)
            with nullcontext() if functional.training else torch.no_grad():
                state, auxiliary, callback_state, emitted = self._values(
                    parameters, values, state, tape, callbacks, callback_state
                )
            if callbacks is not None and callbacks._has_emissions(emitted):
                parts.append(emitted)
            return state, _attach_callback_results(
                auxiliary, callbacks, callback_state, parts
            )
        except BaseException:
            self._report_state.report = previous
            raise


@contextmanager
def _restore_report_on_error(chunk):
    previous = None if chunk is None else chunk.execution_report()
    try:
        yield
    except BaseException:
        if chunk is not None:
            chunk._report_state.report = previous
        raise


def run(
    functional,
    parameters,
    constants,
    state,
    inputs=None,
    *,
    steps=None,
    chunklength=64,
    checkpoint_every=None,
    callbacks=None,
    callback_state=None,
    compiled_chunk=None,
):
    """Host scheduling with shared exact spans and non-reentrant checkpoints.

    Explicit tapes are sliced without detaching. If NetStim draws are omitted,
    sample each bounded host span sequentially from explicit RNG carry before
    entering checkpoint/compiled execution. Backward reuses those same packets.
    """
    public_parameters, public_constants, public_inputs = parameters, constants, inputs
    (parameters, constants, state, inputs), _ = _canonicalize_mapping_tree(
        (parameters, constants, state, inputs)
    )
    callbacks, callback_state, parts = _validate(
        functional, parameters, state, callbacks, callback_state
    )
    _validate_tree(constants, functional._constant_schema, "constants")
    tape, steps = _resolve_inputs(functional, inputs, steps, allow_missing_draws=True)
    chunklength = _validate_chunklength(chunklength)
    if checkpoint_every is not None:
        checkpoint_every = _validate_chunklength(checkpoint_every)
    if compiled_chunk is not None and (
        type(compiled_chunk) is not CompiledNetworkChunk
        or compiled_chunk._functional is not functional
    ):
        raise FunctionalizationError(
            "compiled_chunk belongs to a different Network plan"
        )
    horizon = checkpoint_every or chunklength
    checkpoint_active = (
        checkpoint_every is not None and functional.training and torch.is_grad_enabled()
    )
    if compiled_chunk is not None:
        kernel_width = min(chunklength, compiled_chunk.steps)
    else:
        kernel_width = chunklength
    # Capture the live public bindings as well as explicit checkpoint operands.
    # Container rebinding and tensor writes after forward invalidate replay.
    binding = (
        _binding((public_parameters, public_constants)) if checkpoint_active else None
    )
    tape_binding = _binding((public_inputs, tape)) if checkpoint_active else None
    auxiliary = _auxiliary(functional, state)

    def span(parameters_in, constants_in, state_in, tape_in, callback_in, span_steps):
        if binding is not None and (
            _binding((public_parameters, public_constants)) != binding
            or _binding((public_inputs, tape)) != tape_binding
        ):
            raise FunctionalizationError(
                "Network parameters, constants, or input tapes changed after checkpoint forward"
            )
        functional._validate_source()
        functional._validate_parameters(parameters_in)
        functional._validate_state(state_in)
        values = functional._prepare_values(parameters_in, constants_in)
        emitted_parts = []
        auxiliary_in = _auxiliary(functional, state_in)
        for start, stop in _execution_spans(span_steps, kernel_width):
            selected = _slice(tape_in, start, stop)
            if compiled_chunk is None:
                state_in, auxiliary_in, callback_in, emitted = _eager_values(
                    functional,
                    parameters_in,
                    values,
                    state_in,
                    selected,
                    stop - start,
                    callbacks,
                    callback_in,
                )
            else:
                kernel = compiled_chunk._runner_chunk(stop - start)
                state_in, auxiliary_in, callback_in, emitted = kernel._values(
                    parameters_in, values, state_in, selected, callbacks, callback_in
                )
            functional.check_overflow(state_in)
            if callbacks is not None and callbacks._has_emissions(emitted):
                emitted_parts.append(emitted)
        emitted = None if callbacks is None else callbacks._concatenate(emitted_parts)
        return state_in, auxiliary_in, callback_in, emitted

    with (
        _restore_report_on_error(compiled_chunk),
        nullcontext() if functional.training else torch.no_grad(),
    ):
        for start, stop in _execution_spans(steps, horizon):
            selected = _slice(tape, start, stop)
            if functional.netstim is not None and selected.get("netstim") is None:
                selected = {
                    **selected,
                    "netstim": sample_draw_tape(functional, state, stop - start),
                }
            span_steps = stop - start

            def execute(
                parameters_in,
                constants_in,
                state_in,
                tape_in,
                callback_in,
                span_steps=span_steps,
            ):
                return span(
                    parameters_in,
                    constants_in,
                    state_in,
                    tape_in,
                    callback_in,
                    span_steps,
                )

            operands = (parameters, constants, state, selected, callback_state)
            if not checkpoint_active:
                state, auxiliary, callback_state, emitted = execute(*operands)
            else:
                state, auxiliary, callback_state, emitted = _checkpoint_tensor_tree(
                    execute, operands
                )
            if callbacks is not None and callbacks._has_emissions(emitted):
                parts.append(emitted)
        return state, _attach_callback_results(
            auxiliary, callbacks, callback_state, parts
        )


__all__ = ["NetworkRolloutInput", "CompiledNetworkChunk"]
