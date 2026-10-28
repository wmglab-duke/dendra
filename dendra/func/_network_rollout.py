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
from ._inputs import Inputs, _population_inputs, _RuntimeInputs
from ._netstim import NetStimDraw, NetStimOutputs
from ._network import _schema, _tensor_paths, _validate_tree
from ._network_bindings import _canonicalize_mapping_tree
from ._network_draws import _legacy_population_noise
from ._runners import (
    _assemble_extra_chunk,
    _attach_callback_results,
    _checkpoint_tensor_tree,
    _execution_spans,
    _prepare_callback_collection,
    _prepare_extra,
    _runtime_extra_checkpoint_binding,
    _validate_chunklength,
)
from ._stimuli import StimulusTensors
from ._tensor_contracts import _tensor_tree_binding
from ._types import FunctionalizationError, RolloutInput


class NetworkRolloutInput(NamedTuple):
    """Time-leading per-Population inputs and explicit stochastic draw tapes.

    ``populations`` maps names to :class:`RolloutInput`, including each
    Population's ``noise`` and ``stimulation`` draw/RNG tapes. ``netstim`` is a
    :class:`NetStimDraw` whose two Tensor fields have one leading time axis.
    Keep draw values and their matching post-draw RNG bytes together. Pure
    rollout/chunk calls require tapes for recurrent NOISE, stimulation and NetStim;
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


def _resolve_inputs(
    functional, inputs, steps, *, allow_missing_draws=False, draws=None
):
    from ._network_draws import _network_inputs, _with_network_runtime_draws

    inputs, draws = _with_network_runtime_draws(inputs, draws)
    populations, netstim = _network_inputs(inputs)
    if not isinstance(populations, Mapping) or not set(populations) <= set(
        functional.populations
    ):
        raise ValueError(
            "populations must map known Network names to RolloutInput tapes"
        )
    lengths = []
    for name, selected in populations.items():
        if not isinstance(selected, (Inputs, _RuntimeInputs, RolloutInput)):
            raise TypeError(f"Population {name!r} inputs must be a RolloutInput")
        for _path, value in _tensor_paths(selected):
            if value.ndim < 1:
                raise ValueError("Population input tapes need a leading time axis")
            lengths.append(value.shape[0])
    if netstim is not None:
        if functional.netstim is None:
            raise ValueError("A Network without NetStim requires netstim=None")
        if not isinstance(netstim, NetStimDraw):
            raise TypeError("netstim must be a NetStimDraw tape")
        for value in netstim:
            if not torch.is_tensor(value) or value.ndim < 1:
                raise ValueError("NetStim draw tapes need a leading time axis")
            lengths.append(value.shape[0])
    if draws is not None:
        for _path, value in _tensor_paths(draws):
            if value.ndim < 1:
                raise ValueError("Runtime draw tapes need a leading time axis")
            lengths.append(value.shape[0])
    if steps is None:
        if not lengths:
            raise ValueError("steps is required when no input tape supplies a horizon")
        steps = lengths[0]
    steps = _steps(steps)
    if any(length != steps for length in lengths):
        raise ValueError("All Network input tapes must have exactly steps samples")
    normalized = {}
    for name, plan in functional.populations.items():
        selected = _population_inputs(populations.get(name))
        ve, intra, _ = plan._resolve_rollout_inputs(selected, steps)
        if draws is None and selected.draws is not None:
            plan._runtime_draw_plan.validate(selected.draws, steps=steps)
        elif draws is None and (selected.noise is not None or not allow_missing_draws):
            plan._validate_noise_draw(selected.noise, steps=steps)
        if draws is None and (
            selected.stimulation is not None or not allow_missing_draws
        ):
            plan._validate_stimulation_draw(selected.stimulation, steps=steps)
        frame = selected._replace(ve=ve, intra=intra)
        if any(value is not None for value in frame):
            normalized[name] = frame
    if draws is None:
        functional._stimulation_rng_plan.validate_packets(
            {
                name: _legacy_population_noise(selected)
                for name, selected in normalized.items()
            },
            {name: selected.stimulation for name, selected in normalized.items()},
        )
    if draws is None and functional.netstim is not None:
        if netstim is None:
            if steps and not allow_missing_draws:
                raise ValueError(
                    "NetStim requires an explicit draw/RNG tape; use sample_draw_tape"
                )
        else:
            plan = functional.netstim
            shape, dtype, device = plan._state_schema["rng_state"]
            plan._validate_tensor(
                netstim.exp_rand,
                shape=(steps, *plan.shape),
                dtype=plan.dtype,
                device=plan.device,
                label="netstim.exp_rand",
            )
            plan._validate_tensor(
                netstim.rng_state,
                shape=(steps, *shape),
                dtype=dtype,
                device=device,
                label="netstim.rng_state",
            )
    tape = {"populations": normalized}
    if netstim is not None:
        tape["netstim"] = netstim
    if draws is not None:
        functional._runtime_draw_plan.validate(draws, steps=steps)
        tape["draws"] = draws
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
    inputs = sample.get("populations", {})
    return functional._step_values(
        parameters,
        prepared,
        state,
        inputs=inputs,
        netstim_draw=sample.get("netstim"),
        **({"draws": sample["draws"]} if "draws" in sample else {}),
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
    draws=None,
):
    from ._network_draws import _with_network_runtime_draws

    inputs, draws = _with_network_runtime_draws(inputs, draws)
    (parameters, prepared, state, inputs, constants, draws), _ = (
        _canonicalize_mapping_tree(
            (parameters, prepared, state, inputs, constants, draws)
        )
    )
    callbacks, callback_state, parts = _validate(
        functional, parameters, state, callbacks, callback_state
    )
    tape, steps = _resolve_inputs(functional, inputs, steps, draws=draws)
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

    def bind(self, parameters, constants):
        """Bind raw inputs with ``bind(parameters, constants=constants)``.

        Positional constants remain supported. Each direct call prepares fresh
        values; host scheduling retains its per-span preparation boundaries.
        """
        from ._network_binding import BoundNetwork

        return BoundNetwork(
            self._functional, parameters, constants, compiled_chunk=self
        )

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
        from ._callbacks import _is_builtin_callback

        if any(not _is_builtin_callback(plan.callback) for plan in callbacks._plans):
            raise FunctionalizationError(
                "Network execution='while' supports qualified Recorders and "
                "PopulationCallback scopes around supported built-ins; "
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
        for name, population in functional.populations.items():
            static.append(
                ("network", "populations", name, "control", "duration_remainder")
            )
            detached.append(("network", "populations", name, "clock", "t"))
            detached.extend(
                ("network", "populations", name, *leaf.public_path)
                for leaf in population._state_layout
                if leaf.public_path[0] == "runtime_noise"
            )
            if population._poisson_plan is not None:
                detached.extend(
                    (
                        "network",
                        "populations",
                        name,
                        "stimulation",
                        "schedules",
                        entry.key,
                    )
                    for entry in population._poisson_plan.entries
                )
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
        draws=None,
    ):
        previous = self._report_state.report
        try:
            # Mapping accessors may execute external Python. Finish all public
            # packet traversal before validating source/preparation freshness.
            from ._network_draws import _with_network_runtime_draws

            inputs, draws = _with_network_runtime_draws(inputs, draws)
            (parameters, prepared, state, inputs, draws), _ = (
                _canonicalize_mapping_tree((parameters, prepared, state, inputs, draws))
            )
            functional = self._functional
            callbacks, callback_state, parts = _validate(
                functional, parameters, state, callbacks, callback_state
            )
            tape, _ = _resolve_inputs(functional, inputs, self.steps, draws=draws)
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
    _runtime_extra=None,
    draws=None,
):
    """Host scheduling with shared exact spans and non-reentrant checkpoints.

    Explicit tapes are sliced without detaching. Missing NOISE or NetStim
    tapes are sampled for each bounded host span sequentially from explicit
    RNG carry before entering checkpoint/compiled execution. Backward reuses
    those same packets. Call-local extracellular recipes are assembled inside
    each checkpoint closure, one bounded kernel at a time, from explicit
    stimulus leaves and the Network's canonical pre-step clock.
    """
    from ._draws import _runtime_draws_binding
    from ._network_draws import _owned_runtime_sampling, _with_network_runtime_draws

    public_parameters, public_constants, public_inputs = parameters, constants, inputs
    public_draws = draws
    inputs, draws = _with_network_runtime_draws(inputs, draws)
    (parameters, constants, state, inputs, draws), _ = _canonicalize_mapping_tree(
        (parameters, constants, state, inputs, draws)
    )
    callbacks, callback_state, parts = _validate(
        functional, parameters, state, callbacks, callback_state
    )
    _validate_tree(constants, functional._constant_schema, "constants")
    tape, steps = _resolve_inputs(
        functional, inputs, steps, allow_missing_draws=True, draws=draws
    )
    automatic_draws = (
        draws is None
        and (
            functional.netstim is not None
            or any(
                plan._runtime_draw_plan.has_noise or plan._poisson_plan is not None
                for plan in functional.populations.values()
            )
        )
        and tape.get("netstim") is None
        and not any(
            values.noise is not None
            or values.stimulation is not None
            or values.draws is not None
            for values in tape["populations"].values()
        )
    )
    extra_plans, extra_tensors, extra_values = {}, {}, {}
    if _runtime_extra is not None:
        for name, extra in _runtime_extra.items():
            selected = tape["populations"].get(name, _RuntimeInputs())
            if selected.ve is not None:
                raise ValueError(
                    f"explicit ve for Population {name!r} cannot be combined "
                    "with a declared runtime extra slot"
                )
            plan, tensors = _prepare_extra(
                functional.populations[name], extra, RolloutInput()
            )
            extra_plans[name], extra_tensors[name] = plan, tensors
            extra_values[name] = (
                None
                if tensors is None
                else {"parameters": tensors.parameters, "constants": tensors.constants}
            )
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
    draw_binding = (
        _binding((public_draws, draws))
        if checkpoint_active and draws is not None
        else None
    )

    def runtime_extra_binding():
        return tuple(
            (name, _runtime_extra_checkpoint_binding(plan, extra_tensors[name]))
            for name, plan in extra_plans.items()
            if plan is not None
        )

    extra_binding = (
        runtime_extra_binding() if checkpoint_active and extra_plans else None
    )
    auxiliary = _auxiliary(functional, state)

    def span(
        parameters_in,
        constants_in,
        state_in,
        tape_in,
        callback_in,
        extra_in,
        span_steps,
        span_start,
    ):
        if binding is not None and (
            _binding((public_parameters, public_constants)) != binding
            or _binding((public_inputs, tape)) != tape_binding
        ):
            raise FunctionalizationError(
                "Network parameters, constants, or input tapes changed after checkpoint forward"
            )
        if draw_binding is not None and _binding((public_draws, draws)) != draw_binding:
            raise FunctionalizationError(
                "Network runtime draws changed after checkpoint forward"
            )
        if extra_binding is not None and runtime_extra_binding() != extra_binding:
            raise FunctionalizationError(
                "Network runtime extra tensors changed after checkpoint forward"
            )
        functional._validate_source()
        functional._validate_parameters(parameters_in)
        functional._validate_state(state_in)
        values = functional._prepare_values(parameters_in, constants_in)
        emitted_parts = []
        auxiliary_in = _auxiliary(functional, state_in)
        for start, stop in _execution_spans(span_steps, kernel_width):
            selected = _slice(tape_in, start, stop)
            if extra_plans:
                populations = dict(selected["populations"])
                clock = state_in["clock"]
                indices = clock["step"] + torch.arange(
                    stop - start,
                    device=clock["step"].device,
                    dtype=clock["step"].dtype,
                )
                times = (
                    clock["origin"] + indices.to(clock["origin"].dtype) * functional.dt
                ).detach()
                for name, extra_plan in extra_plans.items():
                    plan = functional.populations[name]
                    packet = extra_in[name]
                    tensors = (
                        None
                        if packet is None
                        else StimulusTensors(
                            packet["parameters"],
                            packet["constants"],
                            extra_tensors[name].plan_token,
                        )
                    )
                    ve = _assemble_extra_chunk(
                        plan,
                        extra_plan,
                        tensors,
                        state_in["populations"][name],
                        span_start + start,
                        stop - start,
                        steps,
                        functional.dt,
                        runtime_extra_slot=True,
                        times=times.to(device=plan.device, dtype=plan.dtype),
                    )
                    populations[name] = populations.get(
                        name, _RuntimeInputs()
                    )._replace(ve=ve)
                selected = {**selected, "populations": populations}
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
            if (
                "draws" not in selected
                and automatic_draws
                and _owned_runtime_sampling(functional)
            ):
                packet = functional.sample_runtime_draws(state, steps=stop - start)
                from ._draws import _normalize_runtime_draws

                packet = _normalize_runtime_draws(packet)
                functional._runtime_draw_plan.validate(packet, steps=stop - start)
                selected = {**selected, "draws": packet}
            elif "draws" not in selected:
                populations = dict(selected["populations"])
                provided_noise = {
                    name: _legacy_population_noise(values)
                    for name, values in populations.items()
                    if _legacy_population_noise(values) is not None
                }
                provided_stimulation = {
                    name: values.stimulation
                    for name, values in populations.items()
                    if values.stimulation is not None
                }
                if functional._stimulation_rng_plan.enabled:
                    noise_tapes, stimulation_tapes = (
                        functional._stimulation_rng_plan.sample_tapes(
                            state,
                            stop - start,
                            noise=provided_noise,
                            stimulation=provided_stimulation,
                        )
                    )
                else:
                    noise_tapes = functional._rng_plan.sample_noise_tapes(
                        state,
                        stop - start,
                        provided=provided_noise,
                    )
                    stimulation_tapes = functional.sample_population_stimulation_tapes(
                        state,
                        stop - start,
                        provided=provided_stimulation,
                    )
                for name, plan in functional.populations.items():
                    values = populations.get(name, _RuntimeInputs())
                    if values.stimulation is None:
                        stimulation = stimulation_tapes[name]
                        if stimulation is not None:
                            values = values._replace(stimulation=stimulation)
                            populations[name] = values
                    if values.noise is None and values.draws is None:
                        noise = noise_tapes[name]
                        if noise is not None:
                            from ._draws import RuntimeDraws

                            populations[name] = (
                                values._replace(draws=noise)
                                if isinstance(noise, RuntimeDraws)
                                else values._replace(noise=noise)
                            )
                selected = {**selected, "populations": populations}
                if functional.netstim is not None and selected.get("netstim") is None:
                    selected = {
                        **selected,
                        "netstim": sample_draw_tape(functional, state, stop - start),
                    }
            span_steps = stop - start
            packet_source = selected.get("draws")
            packet_binding = (
                _runtime_draws_binding(packet_source)
                if checkpoint_active and packet_source is not None
                else None
            )

            def execute(
                parameters_in,
                constants_in,
                state_in,
                tape_in,
                callback_in,
                extra_in,
                span_steps=span_steps,
                span_start=start,
                packet_source=packet_source,
                packet_binding=packet_binding,
            ):
                if packet_binding is not None and (
                    _runtime_draws_binding(packet_source) != packet_binding
                ):
                    raise FunctionalizationError(
                        "Network runtime draws changed after checkpoint forward"
                    )
                return span(
                    parameters_in,
                    constants_in,
                    state_in,
                    tape_in,
                    callback_in,
                    extra_in,
                    span_steps,
                    span_start,
                )

            operands = (
                parameters,
                constants,
                state,
                selected,
                callback_state,
                extra_values,
            )
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

_OWNED_NETSTIM_DRAW_TAPE = sample_draw_tape
