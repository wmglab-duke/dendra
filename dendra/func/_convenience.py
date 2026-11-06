"""Opt-in compiled runners over the existing functional execution adapters."""

from __future__ import annotations

from contextlib import nullcontext
from types import MappingProxyType

import torch
from torch.utils import _pytree as pytree

from ._binding import BoundPopulation
from ._compiled import CompiledPopulationChunk
from ._continuation import _complete_continuation, _resolve_continuation
from ._draws import _with_runtime_draws
from ._initialization import _initialize_fresh, _runner_sources
from ._inputs import _population_inputs
from ._parameter_queries import matching_names
from ._parameter_queries import parameter_name as _parameter_name
from ._population import (
    FunctionalPopulation,
    _initialization_transform_hooks,
    make_functional,
)
from ._runner_policy import (
    _configured_callbacks,
    _inference_compile_options,
    _runner_call_horizon,
    _runner_chunk_steps,
    _runner_default_horizon,
    _validate_host_call,
    _validate_host_construction,
)
from ._runners import (
    _bound_runner_dt,
    _canonical_runtime_sampling_available,
    _normalize_sampled_inputs,
    _prepare_chunk_inputs,
    _run_source_validated,
    _slice_inputs,
    _stage_duration,
    _validate_duration_remainder,
    run,
)
from ._runtime_extra import (
    _MISSING_RUNTIME_EXTRA,
    VarInput,
)
from ._snapshot import _snapshot_frames
from ._stimulation_clock import _accepted_bank_clock, _sampling_time_tape
from ._types import FunctionalizationError

_OWNED_PREPARE = FunctionalPopulation.prepare
_OWNED_BIND = CompiledPopulationChunk.bind
_OWNED_RUN = run
_OWNED_NOISE_TAPE = FunctionalPopulation.sample_noise_draw_tape
_OWNED_STIMULATION_TAPE = FunctionalPopulation.sample_stimulation_draw_tape
_OWNED_MISSING_DRAWS = FunctionalPopulation._sample_missing_runtime_draws
_OWNED_RUNTIME_DRAW_SAMPLER = FunctionalPopulation.sample_runtime_draws


def _initialization_writes_parameters(functional):
    """Continuation must explicitly carry any initialization-adjusted parameters."""
    population = functional._population
    return any(
        reference.startswith("parameters.")
        for phase in ("pre", "post")
        for hook in _initialization_transform_hooks(population, phase)
        for reference in population._initialization_transforms[hook.action_name].writes
    )


class CompiledRunner:
    """Compiled Population execution with explicit inputs and continuations.

    Construct with :func:`compile`, usually exposed as ``dn.compile``. Calling the
    runner returns ``(state, auxiliary)`` and does not advance its native source.
    Omitting state restarts the configured snapshot/fresh initialization policy;
    passing ``auxiliary["continuation"]`` continues effective parameters,
    constants, simulation state and callback carry together.

    ``steps=`` or ``duration=`` overrides a call's default horizon without changing
    ``chunk_steps``. ``inputs=Inputs(...)`` supplies deterministic tapes and
    ``draws=RuntimeDraws(...)`` replays stochastic tapes. Named callbacks return
    results under ``auxiliary["callbacks"]``. Per-call source overrides retain
    their Tensor identities and gradients but do not replace runner defaults.

    Set ``inference=False`` for training. Scan supports first-order gradients;
    while is inference-only. ``checkpoint_every`` trades recomputation for saved
    activations while retaining full-run gradients and requested recordings.
    Independent training forwards own their preparation graphs; safe snapshot
    inference may reuse preparation.

    ``extra=VarInput()`` requires call-local ``extra=`` on every invocation;
    ``extra=None`` selects zero drive. Waveform types and contact counts may vary.
    These slots accept deterministic waveforms. Bind randomized Poisson waveforms
    when compiling. See :func:`compile` for construction options and
    :meth:`__call__` for invocation arguments.
    """

    def __init__(
        self,
        population,
        *,
        dt,
        steps=None,
        duration=None,
        extra=None,
        chunk_steps=None,
        execution="scan",
        callbacks=None,
        inference=True,
        initialization="snapshot",
        checkpoint_every=None,
        trainable=(),
        backend="inductor",
        options=None,
        **compile_options,
    ):
        _validate_host_construction(compile_options)
        steps, duration, nominal_steps = _runner_default_horizon(steps, duration, dt)
        if execution not in ("scan", "while"):
            raise ValueError("execution must be 'scan' or 'while'")
        if not isinstance(inference, bool):
            raise TypeError("inference must be a bool")
        if initialization not in ("snapshot", "fresh"):
            raise ValueError("initialization must be 'snapshot' or 'fresh'")
        if checkpoint_every is not None:
            if (
                isinstance(checkpoint_every, bool)
                or not isinstance(checkpoint_every, int)
                or checkpoint_every < 1
            ):
                raise ValueError("checkpoint_every must be a positive integer")
            if execution != "scan" or inference:
                raise ValueError(
                    "checkpoint_every requires execution='scan' and inference=False"
                )
        configured_callbacks = _configured_callbacks(callbacks)
        if execution == "while":
            if not inference:
                raise ValueError(
                    "execution='while' is inference-only; use scan for gradients"
                )
            if duration is None and steps < 4:
                raise ValueError("execution='while' requires at least four steps")
            if compile_options.get("disable", False):
                raise ValueError("execution='while' requires compilation to be enabled")
        self._inference = inference
        self._initialization = initialization
        self._execution = execution
        self._steps = steps
        self._duration = duration
        self._checkpoint_every = checkpoint_every
        self._runtime_extra = isinstance(extra, VarInput)
        self._chunk_steps = _runner_chunk_steps(
            chunk_steps,
            steps=max(nominal_steps, 4) if execution == "while" else nominal_steps,
            runtime_extra=self._runtime_extra,
        )
        if execution == "while" and self._chunk_steps < 4:
            raise ValueError(
                "execution='while' requires "
                + ("runtime extra " if self._runtime_extra else "")
                + "chunk_steps of at least four"
            )
        # Versioned snapshots must remain ordinary tensors even when the
        # convenience constructor is called inside inference_mode().
        with torch.inference_mode(False), torch.no_grad():
            self._functional, tensors = make_functional(
                population, dt=dt, extra=None if self._runtime_extra else extra
            )
            self._parameters, self._constants, self._initial_state = _snapshot_frames(
                self._functional, tensors, trainable=trainable
            )
            self._initialization_inputs = pytree.tree_map_only(
                torch.Tensor,
                lambda value: value.detach().clone(),
                tensors.initialization,
            )
            self._initialization_writes_parameters = initialization == "fresh" and (
                _initialization_writes_parameters(self._functional)
                or self._functional._owned_cache_bound
            )
            self._initialization_writes_constants = initialization == "fresh" and (
                self._functional._owned_cache_bound
                or (
                    self._functional._random_plan is not None
                    and bool(self._functional._random_plan.entries)
                )
            )
            self._callbacks = (
                None
                if not configured_callbacks
                else self._functional.make_callbacks(configured_callbacks)
            )
        keywords = _inference_compile_options(
            self._functional, inference, backend, options, compile_options
        )
        self._kernel = self._functional.compile_rollout_chunk(
            self._chunk_steps,
            execution="default" if execution == "while" else "scan",
            **keywords,
        )
        self._prepared = None

    @property
    def initial_state(self):
        """Captured restart snapshot; tensor leaves may be edited by the caller.

        Fresh-policy calls ignore this snapshot unless it is explicitly passed
        as state, which always bypasses initialization.
        """
        return self._initial_state

    @property
    def initialization_inputs(self):
        """Independent fresh-initialization defaults; tensor leaves are editable.

        Use ``requires_grad_()`` or pass an explicit ``InitializationInput`` per
        fresh call to differentiate these initialization-only inputs. They are
        not included in :attr:`parameters` or selected by ``trainable``.
        """
        return self._initialization_inputs

    @property
    def parameters(self):
        """Read-only mapping of independent, live parameter tensor leaves."""
        return MappingProxyType(self._parameters)

    @property
    def named_parameters(self):
        """Read-only canonical names for the runner-owned default parameter leaves."""
        return MappingProxyType(self._parameters)

    @property
    def trainable_parameters(self):
        """Read-only named defaults currently requiring gradients, including aliases.

        Tensor objects remain live; freezing or unfreezing a leaf is reflected
        on the next access. Per-call parameter overrides are not included.
        Deduplicate values by identity before supplying them to an optimizer.
        Inference execution suppresses gradients even for selected leaves.
        """
        return MappingProxyType(
            {
                name: value
                for name, value in self._parameters.items()
                if value.requires_grad
            }
        )

    def find_parameters(self, query):
        """Return all named defaults matching a substring, glob, or regex query."""
        return {
            name: self._parameters[name]
            for name in matching_names(self._parameters, query)
        }

    def parameter_name(self, query):
        """Resolve one exact, partial, glob, or regex match among named defaults."""
        return _parameter_name(self._parameters, query)

    def get_parameter(self, query):
        """Return the live default Tensor selected by one unambiguous query."""
        return self._parameters[self.parameter_name(query)]

    @property
    def constants(self):
        """Read-only mapping of independent, version-checked constants."""
        return MappingProxyType(self._constants)

    @property
    def functional(self):
        """The lowered execution plan owned by this runner."""
        return self._functional

    @property
    def callbacks(self):
        """Bound callback collection, or ``None`` when callbacks are omitted.

        Use its ``export_state`` and ``restore_state`` methods to save and
        resume explicit callback carry with an equivalent reconstructed runner.
        """
        return self._callbacks

    def export_continuation(self, continuation):
        """Export owned detached data for portable continuation checkpoints.

        Effective raw sources retain their Tensor ties and trainability as new
        leaves. Save the versioned packet with ``torch.save``. Export runs
        outside compilation and function transforms; prior graphs and recording
        history are excluded. Packets from ``dn.func.initialize`` begin callbacks
        on their first execution.
        """
        from ._continuation_checkpoints import export_continuation

        return export_continuation(self, continuation)

    def restore_continuation(self, packet):
        """Restore a plain checkpoint against this compatible initialized runner.

        The receiving model/schema, dtype/device and active callback carry
        must match. Initialized packets begin callbacks on their first execution.
        Returns independent ordinary Tensor leaves in a
        ``Continuation`` without initialization, preparation, sampling or
        execution. Runner defaults are unchanged.
        """
        from ._continuation_checkpoints import restore_continuation

        return restore_continuation(self, packet)

    @property
    def kernel(self):
        """The fixed-width compiled kernel used by this runner."""
        return self._kernel

    def execution_report(self):
        """Return the underlying kernel's latest successful execution report."""
        return self._kernel.execution_report()

    def warm(self, state=None, *, verbose=False, **call_kwargs):
        """Warm forward execution with a discarded representative run.

        Parameters
        ----------
        state : tensor tree or Continuation, optional
            Select the same restart or continuation as :meth:`__call__`.
        verbose : bool, optional
            Print start, elapsed-time updates and completion to standard error.
            Defaults to False. Progress describes the warm-up call, not a
            compiler percentage.
        **call_kwargs
            Invocation arguments accepted by :meth:`__call__`, including
            inputs, draws, runtime extra, source overrides and the horizon.

        Returns
        -------
        CompiledRunner
            This runner, with the encountered execution variants warmed.

        Notes
        -----
        Executes the selected simulation and discards its results. Model/source
        state, RNG carry and parameter gradients retain their values. Recorder
        file publication is suppressed while its numerical callback and chunk
        schedule are preserved. Compilation and safe preparation caches remain.
        Accelerator work completes before return.
        Training warms the forward path only; a subsequent backward may compile.
        Different shapes, gradient flags or horizons can require other variants.
        """
        from ._warm import _warm_runner

        return _warm_runner(self, state, verbose=verbose, **call_kwargs)

    def _prepare(self, parameters, *, cache, constants=None):
        return self._prepare_with(
            parameters,
            cache=cache,
            constants=constants,
            prepare=self._functional.prepare,
        )

    def _prepare_source_validated(self, parameters, *, cache, constants=None):
        return self._prepare_with(
            parameters,
            cache=cache,
            constants=constants,
            prepare=self._functional._prepare_source_validated,
        )

    def _prepare_with(self, parameters, *, cache, prepare, constants=None):
        from ._preparation import _select_population_preparation

        prepared = _select_population_preparation(
            self._functional,
            parameters,
            self._constants if constants is None else constants,
            cached=self._prepared,
            cache=cache,
            prepare=prepare,
        )
        if cache:
            self._prepared = prepared
        return prepared

    def _require_while(
        self, parameters, prepared, state, inputs, callback_state, horizon
    ):
        """Fail before a rejected structured loop could compile an unrolled horizon."""
        if self._callbacks is None and callback_state is not None:
            raise ValueError("callback_state requires callbacks")
        functional = self._functional
        explicit_stimulation = inputs is not None and (
            inputs.draws is not None or inputs.stimulation is not None
        )
        functional._validate_state(state)
        values = functional._validate_prepared(prepared, parameters)
        steps = min(self._kernel.steps, horizon)
        from ._recording import _recording_span

        steps = _recording_span(self._callbacks, steps)
        if steps < 4:
            # These calls execute only the bounded tail; there is no structured
            # loop body to capture or reject.
            if self._callbacks is None:
                self._kernel._enable_model_while()
            return
        if self._runtime_extra or steps < horizon:
            if not self._runtime_extra:
                # Qualify the full call before a shorter preflight could hide
                # an invalid time axis. Missing drives remain unmaterialized.
                functional._resolve_rollout_inputs(inputs, horizon)
            # Only inspect one bounded loop signature. The actual schedule
            # samples draw packets and assembles stimulation span by span.
            inputs = _slice_inputs(inputs, 0, steps)
            if self._runtime_extra:
                inputs = inputs._replace(
                    ve=torch.zeros(
                        (steps, *functional.shape),
                        device=functional.device,
                        dtype=functional.dtype,
                    )
                )
            if _canonical_runtime_sampling_available(functional, inputs):
                inputs = _prepare_chunk_inputs(
                    functional, state, inputs, 0, steps, canonical=True
                )
            elif inputs.draws is None:
                inputs = functional._sample_missing_runtime_draws(state, inputs, steps)
                supplied_draws = getattr(inputs, "draws", None)
                if supplied_draws is None and inputs.noise is None:
                    inputs = inputs._replace(
                        noise=functional.sample_noise_draw_tape(state, steps)
                    )
                if (
                    supplied_draws is None
                    and inputs.stimulation is None
                    and functional._poisson_plan is not None
                ):
                    inputs = inputs._replace(
                        stimulation=functional.sample_stimulation_draw_tape(
                            state, steps
                        )
                    )
                inputs = _normalize_sampled_inputs(
                    functional, inputs, steps, qualify_draws=True
                )
            # Public sampler replacements can execute authored code. Qualify
            # the source after their calls and before capturing a loop body.
            functional._validate_source()
            functional._validate_parameters(parameters)
            values = functional._validate_prepared(prepared, parameters)
        ve, intra, _ = functional._resolve_rollout_inputs(inputs, steps)
        callbacks = self._callbacks
        if callbacks is None:
            carries = ()
            chunk = self._kernel._runner_chunk(steps)
            executor = chunk._model_while_executor()
        else:
            carry = (
                callbacks._initialize(state)[0]
                if callback_state is None
                else callbacks._validate_state(callback_state)
            )
            carries = carry.carries
            chunk = self._kernel._runner_callback_chunk(callbacks, steps)
            executor = chunk._callback_while
        if executor is None:
            raise FunctionalizationError(
                "execution='while' has no functional callback loop executor"
            )
        sampling_times = None if inputs is None else inputs.sampling_time
        if (
            sampling_times is None
            and (functional.intra.enabled or functional.extra.enabled)
            and not _accepted_bank_clock(functional, explicit_stimulation)
        ):
            sampling_times = _sampling_time_tape(
                functional,
                state["clock"]["t"],
                0,
                steps,
                _bound_runner_dt(functional, values),
                clock_dt=values["integrator"]["dt"],
            )
        kernel, reason = executor.kernel_for(
            parameters,
            values,
            state,
            ve,
            intra,
            carries,
            None if inputs is None else inputs.noise,
            None if inputs is None else inputs.stimulation,
            None if inputs is None else inputs.draws,
            sampling_times,
        )
        if kernel is None:
            raise FunctionalizationError(f"execution='while' is unavailable: {reason}")
        if callbacks is None:
            self._kernel._enable_model_while()

    def __call__(
        self,
        state=None,
        *,
        steps=None,
        duration=None,
        parameters=None,
        constants=None,
        inputs=None,
        callback_state=None,
        initialization=None,
        extra=_MISSING_RUNTIME_EXTRA,
        draws=None,
    ):
        """Run from a restart snapshot, explicit state or Continuation.

        Parameters
        ----------
        state : tensor tree or Continuation, optional
            Omission/None restarts the configured initialization policy. A Continuation
            also supplies effective parameters/constants and callback carry.
        steps, duration : int or float, optional
            Override the default horizon with an exact count or elapsed time in ms.
            They are mutually exclusive; per-call values may be zero.
        parameters, constants : mapping, optional
            Call-local source overrides. They cannot accompany a Continuation.
        inputs : Inputs or mapping, optional
            Deterministic time-leading tapes. Networks use Population-name mappings.
        draws : RuntimeDraws, optional
            Explicit stochastic tape; omission retains automatic sampling.
        callback_state : FunctionalCallbackState, optional
            Carry for an explicit-state call. A Continuation includes this already.
        initialization : InitializationInput or NetworkInitializationInput, optional
            Fresh-policy initialization override; incompatible with explicit state.
        extra : stimulation specification or mapping, optional
            Required when runtime ``VarInput`` slots were declared. None is zero drive.

        Returns
        -------
        state : tensor tree
            Final explicit simulation state.
        auxiliary : dict
            Contains ``continuation`` and, when configured, callback results. Fresh
            initialization also publishes effective ``parameters`` and ``constants``.
        """
        return _execute_population_runner(
            self,
            state,
            steps=steps,
            duration=duration,
            parameters=parameters,
            constants=constants,
            inputs=inputs,
            callback_state=callback_state,
            initialization=initialization,
            extra=extra,
            draws=draws,
        )


def _execute_population_runner(
    self,
    state=None,
    *,
    steps=None,
    duration=None,
    parameters=None,
    constants=None,
    inputs=None,
    callback_state=None,
    initialization=None,
    extra=_MISSING_RUNTIME_EXTRA,
    draws=None,
):
    _validate_host_call()
    selected_steps, selected_duration = _runner_call_horizon(
        self._steps, self._duration, steps, duration
    )
    if draws is not None:
        inputs = _with_runtime_draws(inputs, draws)
    if self._runtime_extra:
        if extra is _MISSING_RUNTIME_EXTRA:
            raise TypeError("extra must be supplied for extra=VarInput()")
    elif extra is not _MISSING_RUNTIME_EXTRA:
        raise TypeError("runtime extra requires compile(..., extra=VarInput())")
    state, parameters, constants, callback_state = _resolve_continuation(
        state, parameters, constants, callback_state, initialization
    )
    fresh_policy = self._initialization == "fresh"
    if initialization is not None:
        if not fresh_policy:
            raise ValueError("initialization overrides require initialization='fresh'")
        if state is not None:
            raise ValueError("initialization cannot be combined with explicit state")
    fresh = fresh_policy and state is None
    if fresh and callback_state is not None:
        raise ValueError(
            "fresh initialization cannot continue callback_state without state"
        )
    if (
        fresh_policy
        and state is not None
        and parameters is None
        and self._initialization_writes_parameters
    ):
        raise ValueError(
            "initialization writes parameters; continue with "
            "parameters=aux['parameters'] from the preceding call"
        )
    if (
        fresh_policy
        and state is not None
        and constants is None
        and self._initialization_writes_constants
    ):
        raise ValueError(
            "cached initialization changes constants; continue with constants=aux['constants'] from the preceding call"
        )
    selected_state = self._initial_state if state is None else state
    # An external Mapping can execute arbitrary Python while materializing
    # its entries. Snapshot it before checking source structure, rather than
    # letting iteration invalidate that check between owned boundaries.
    selected_parameters, selected_constants, selected_initialization, _ = (
        _runner_sources(self, parameters, constants, initialization, fresh=fresh)
    )
    if fresh:
        # Physical preparation requires ordinary, versioned tensors even
        # for inference. Respect the caller's grad context and never retain
        # this initialization graph or its parameter replacements.
        initialized = _initialize_fresh(
            self._functional,
            selected_parameters,
            selected_constants,
            selected_state,
            selected_initialization,
            gradient=torch.is_grad_enabled() and not self._inference,
        )
        selected_state = initialized.state
        selected_parameters = initialized.parameters
        selected_constants = initialized.constants
    owned_boundaries = (
        type(self._functional) is FunctionalPopulation
        and type(self._kernel) is CompiledPopulationChunk
        and getattr(self._functional.prepare, "__self__", None) is self._functional
        and getattr(self._functional.prepare, "__func__", None) is _OWNED_PREPARE
        and getattr(self._kernel.bind, "__self__", None) is self._kernel
        and getattr(self._kernel.bind, "__func__", None) is _OWNED_BIND
        and run is _OWNED_RUN
        and (
            self._execution != "while"
            or all(
                getattr(sampler, "__self__", None) is self._functional
                and getattr(sampler, "__func__", None) is original
                for sampler, original in (
                    (
                        self._functional.sample_noise_draw_tape,
                        _OWNED_NOISE_TAPE,
                    ),
                    (
                        self._functional.sample_stimulation_draw_tape,
                        _OWNED_STIMULATION_TAPE,
                    ),
                    (
                        self._functional._sample_missing_runtime_draws,
                        _OWNED_MISSING_DRAWS,
                    ),
                    (
                        self._functional.sample_runtime_draws,
                        _OWNED_RUNTIME_DRAW_SAMPLER,
                    ),
                )
            )
        )
    )
    context = torch.inference_mode() if self._inference else nullcontext()
    with context:
        if owned_boundaries:
            self._functional._validate_source()
        prepare = self._prepare_source_validated if owned_boundaries else self._prepare
        prepared = prepare(
            selected_parameters,
            cache=self._inference
            and parameters is None
            and constants is None
            and not fresh_policy,
            constants=selected_constants,
        )
        bound = (
            BoundPopulation._from_source_validated(
                self._functional,
                selected_parameters,
                prepared,
                compiled_chunk=self._kernel,
            )
            if owned_boundaries
            else self._kernel.bind(selected_parameters, prepared)
        )
        if selected_duration is not None:
            _validate_duration_remainder(selected_state)
            selected_state, selected_steps = _stage_duration(
                selected_state,
                selected_duration,
                _bound_runner_dt(self._functional, prepared.values),
                name="duration",
            )
        inputs = _population_inputs(inputs)
        bounded_while = (
            self._runtime_extra
            or self._chunk_steps < selected_steps
            or bool(getattr(self._callbacks, "_storage", None))
        )
        if (
            self._execution == "while"
            and not bounded_while
            and (self._functional.intra.enabled or self._functional.extra.enabled)
            and not _accepted_bank_clock(
                self._functional,
                inputs.draws is not None or inputs.stimulation is not None,
            )
        ):
            # Freeze the caller's bank policy before automatic sampling fills
            # draw packets; generated packets do not select native replay mode.
            inputs = inputs._replace(
                sampling_time=_sampling_time_tape(
                    self._functional,
                    selected_state["clock"]["t"],
                    0,
                    selected_steps,
                    _bound_runner_dt(self._functional, prepared.values),
                    clock_dt=prepared.values["integrator"]["dt"],
                )
            )
        if self._runtime_extra and inputs.ve is not None:
            raise ValueError("explicit ve cannot be combined with runtime extra")
        if inputs.draws is not None:
            self._functional._runtime_draw_plan.validate(
                inputs.draws, steps=selected_steps
            )
        if (
            self._execution == "while"
            and not bounded_while
            and _canonical_runtime_sampling_available(self._functional, inputs)
        ):
            inputs = _prepare_chunk_inputs(
                self._functional,
                selected_state,
                inputs,
                0,
                selected_steps,
                canonical=True,
            )
        if self._execution == "while" and not bounded_while and inputs.draws is None:
            inputs = self._functional._sample_missing_runtime_draws(
                selected_state, inputs, selected_steps
            )
        if (
            self._execution == "while"
            and not bounded_while
            and getattr(inputs, "draws", None) is None
            and inputs.noise is None
        ):
            sample_noise = (
                self._functional._sample_noise_draw_tape_source_validated
                if owned_boundaries
                else self._functional.sample_noise_draw_tape
            )
            inputs = inputs._replace(noise=sample_noise(selected_state, selected_steps))
        if (
            self._execution == "while"
            and not bounded_while
            and getattr(inputs, "draws", None) is None
            and inputs.stimulation is None
            and getattr(self._functional, "_poisson_plan", None) is not None
        ):
            sample_stimulation = (
                self._functional._sample_stimulation_draw_tape_source_validated
                if owned_boundaries
                else self._functional.sample_stimulation_draw_tape
            )
            inputs = inputs._replace(
                stimulation=sample_stimulation(selected_state, selected_steps)
            )
        if self._execution == "while":
            if not bounded_while:
                inputs = _normalize_sampled_inputs(
                    self._functional,
                    inputs,
                    selected_steps,
                    qualify_draws=not owned_boundaries,
                )
            if not owned_boundaries:
                # A replaced public sampler can run user code after its
                # own check. Recheck before preflight captures a step.
                self._functional._validate_source()
            if selected_steps:
                self._require_while(
                    bound._parameters,
                    prepared,
                    selected_state,
                    inputs,
                    callback_state,
                    selected_steps,
                )
        schedule = _run_source_validated if owned_boundaries else run
        schedule_options = (
            {}
            if self._checkpoint_every is None
            else {"checkpoint_every": self._checkpoint_every}
        )
        if self._runtime_extra:
            schedule_options.update(extra=extra, _runtime_extra_slot=True)
        if self._checkpoint_every is None:
            if self._runtime_extra or self._chunk_steps < selected_steps:
                schedule_options["host_span_steps"] = self._chunk_steps
        elif self._runtime_extra:
            schedule_options["checkpoint_every"] = min(
                self._checkpoint_every, self._chunk_steps
            )
        previous_report = self._kernel.execution_report()
        final, auxiliary = schedule(
            bound,
            selected_state,
            inputs=inputs,
            steps=selected_steps,
            callbacks=self._callbacks,
            callback_state=callback_state,
            **schedule_options,
        )
        if auxiliary is None:
            auxiliary = {}
        if fresh_policy:
            auxiliary = dict(auxiliary)
            auxiliary["parameters"] = dict(selected_parameters)
            auxiliary["constants"] = dict(selected_constants)
        try:
            return _complete_continuation(
                final, auxiliary, selected_parameters, selected_constants
            )
        except Exception:
            self._kernel._report_state.report = previous_report
            raise


def compile(
    population,
    *,
    dt,
    steps=None,
    duration=None,
    extra=None,
    chunk_steps=None,
    execution="scan",
    callbacks=None,
    inference=True,
    initialization="snapshot",
    checkpoint_every=None,
    trainable=(),
    pending_capacity=None,
    backend="inductor",
    options=None,
    **compile_options,
):
    """Create a compiled runner for an initialized Population or Network.

    Parameters
    ----------
    population : Population or Network
        Initialized source model. A Network's timestep and training mode must
        match its initialized configuration.
    dt : float
        Positive solver timestep in ms. For a Network it must match initialization.
    steps : int, optional
        Positive default timestep count. Supply either ``steps`` or ``duration``.
    duration : float, optional
        Non-negative default elapsed time in ms. Fractional time is retained in
        the returned continuation until a complete timestep can be accepted.
    extra : stimulation specification, mapping, or VarInput, optional
        Extracellular ``(field, waveform)`` pair or contact sequence. Networks use
        a Population-name mapping. ``VarInput()`` declares call-local deterministic
        stimulation; each call then supplies ``extra=``. Explicit ``Inputs.ve``
        cannot drive a Population that also has bound extracellular stimulation.
    chunk_steps : int, optional
        Compiled kernel width. Calls may use other horizons and shorter tails.
        The default is the nominal default horizon, capped at 64 for runtime-extra
        slots. This width does not limit recording history.
    execution : {"scan", "while"}, default "scan"
        Scan supports first-order training. While is inference-only; Population
        while runners require a compiled width of at least four steps.
    callbacks : mapping, optional
        Named built-in or pure custom callback configurations. For example,
        ``{"trace": dn.Recorder(("v", "t"))}`` records a Population trajectory.
    inference : bool, default True
        Suppress execution gradients. Set False for training and select sources
        with ``trainable``.
    initialization : {"snapshot", "fresh"}, default "snapshot"
        No-state calls restart the captured snapshot or run fresh initialization.
        Explicit state or a Continuation always continues without initialization.
    checkpoint_every : int, optional
        Training-scan activation-checkpoint span. Backward recomputes these spans;
        gradients still cover the whole run. This is independent of kernel width.
    trainable : selector or iterable of selectors, optional
        Exact/unique partial names, full-name globs, or ``re:``/compiled regular
        expressions selecting independent runner-owned parameter leaves.
    pending_capacity : int or mapping, optional
        Network pending-event calendar capacities. A full calendar raises before
        publishing a result. Lower again with more capacity from the pre-call native
        boundary; calendar tensor shapes belong to that plan.
    backend : str or callable, default "inductor"
        Backend passed to ``torch.compile``.
    options : dict, optional
        Backend options. Explicit choices override Dendra's inference defaults.
    **compile_options
        Additional ``torch.compile`` options, such as ``fullgraph`` or ``disable``.

    Returns
    -------
    CompiledRunner or CompiledNetworkRunner
        Callable returning ``(state, auxiliary)``. Continue with
        ``dn.func.run(program, auxiliary["continuation"])``. Callback results are
        available at ``auxiliary["callbacks"]``.

    Notes
    -----
    Compilation is lazy. Execution leaves the native model unchanged and preserves
    shared parameter aliases. Use :func:`inspect_sources` to inspect effective
    sources, :func:`resolve_steps` to size deterministic inputs, and
    :func:`initialize` for an explicit fresh trial. Call-local ``VarInput`` slots
    accept deterministic waveforms; randomized Poisson waveforms must be bound
    when compiling. Host runners run outside ``torch.compile`` and ``torch.func``;
    use :func:`make_functional` for explicit tensor transforms.
    """
    from dendra.models.networks.net import Network

    runner_type = CompiledRunner
    runner_keywords = {}
    if isinstance(population, Network):
        from ._network_compile import CompiledNetworkRunner

        runner_type = CompiledNetworkRunner
        runner_keywords["pending_capacity"] = pending_capacity
    elif pending_capacity is not None:
        raise ValueError("pending_capacity applies only to Network compilation")
    return runner_type(
        population,
        dt=dt,
        steps=steps,
        duration=duration,
        extra=extra,
        chunk_steps=chunk_steps,
        execution=execution,
        callbacks=callbacks,
        inference=inference,
        initialization=initialization,
        checkpoint_every=checkpoint_every,
        trainable=trainable,
        backend=backend,
        options=options,
        **runner_keywords,
        **compile_options,
    )


__all__ = ["CompiledRunner", "compile"]
