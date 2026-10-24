"""Opt-in compiled runners over the existing functional execution adapters."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from types import MappingProxyType

import torch
from torch.utils import _pytree as pytree

from ._binding import BoundPopulation
from ._compiled import CompiledPopulationChunk
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
    _validate_host_call,
    _validate_host_construction,
)
from ._runners import _run_source_validated, run
from ._types import FunctionalizationError, InitializationInput, RolloutInput

_OWNED_PREPARE = FunctionalPopulation.prepare
_OWNED_BIND = CompiledPopulationChunk.bind
_OWNED_RUN = run
_OWNED_NOISE_TAPE = FunctionalPopulation.sample_noise_draw_tape
_OWNED_STIMULATION_TAPE = FunctionalPopulation.sample_stimulation_draw_tape


def _initialization_input_mapping(inputs):
    """Snapshot external mappings without severing differentiable leaves."""
    if not isinstance(inputs, InitializationInput):
        raise TypeError("initialization must be an InitializationInput")
    mappings = []
    for name, values in (
        ("states", inputs.states),
        ("transforms", inputs.transforms),
        ("cache", inputs.cache),
    ):
        if values is not None and not isinstance(values, Mapping):
            raise TypeError(f"initialization {name} must be a mapping")
        mappings.append(None if values is None else dict(values))
    return InitializationInput(
        inputs.v_init, *mappings, inputs.random_draw, inputs.stimulation
    )


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
    """Reusable, fixed-horizon scan or inference-only while-loop execution.

    Construct with :func:`dendra.func.compile`. A call returns ``(state, aux)``
    and never commits state or parameters to the source Population. With no
    state argument, the default ``initialization='snapshot'`` restarts from
    :attr:`initial_state`. Opt in to ``initialization='fresh'`` to recompute
    differentiable initial state from current parameters and
    :attr:`initialization_inputs` instead. A bound Population-owned steady-state
    cache is restored through ``initialization_inputs.cache``; its saved
    materializations are detached. Explicit state always continues
    without reinitializing. Parameters
    are independent copies; select differentiable leaves with ``trainable``
    and ``inference=False``, or pass an explicit parameter mapping per call.
    Selectors accept exact or unique partial names, glob patterns, and ``re:``
    or compiled regular expressions. :attr:`named_parameters` lists the
    runner-owned defaults; :attr:`trainable_parameters` shows their current
    ``requires_grad`` leaves. Per-call overrides are not retained for inspection.
    ``inference=True`` suppresses execution gradients regardless of selection.

    ``callbacks`` is a named mapping of functional callback configurations.
    Recording is explicit: use ``callbacks={'trace': dn.func.Recorder(('v',))}``
    to retain voltage at ``aux['callbacks']['trace']['v']``. Omitting callbacks
    or passing an empty mapping records no trajectory. Callback names have no
    special meaning. Scan accepts pure custom callbacks; while supports the
    built-in callbacks and requires a nonempty mapping.

    Compilation is lazy. Snapshot inference reuses preparation with version
    checks; fresh initialization, training, and explicit parameter overrides
    receive fresh preparation. No
    output, live binding, or training preparation is retained between calls.
    For training scan, ``checkpoint_every=H`` recomputes compiled spans of at
    most H steps during backward to reduce saved activation storage. This
    preserves full-horizon gradients and callback history; requested recordings
    remain retained. Callbacks must be pure, deterministic, and replay-safe.
    This is a host-side convenience object, not a torch.compile/torch.func
    transform target; those uses retain the lower-level functional API.

    ``extra`` binds one ``(field, Waveform)`` pair or a sequence of contacts.
    Field and waveform leaves become independent ``stimulation.extra.*``
    parameters, selectable with ``trainable`` like model parameters. Bound
    waveforms follow the accepted simulation clock across continuation and
    checkpoint replay. Explicit ``RolloutInput.ve`` cannot be combined with
    bound ``extra``; use it instead for already assembled time-first voltages.

    Fresh calls accept an ``initialization=InitializationInput(...)`` override;
    it cannot be combined with explicit state. Fresh-policy outputs include
    ``aux['parameters']`` and ``aux['constants']``, the effective mappings after
    initialization. Continue parameter-writing initialization with the returned
    parameters; cached initialization also requires the returned constants.
    Neither mapping nor its autograd graph is retained by the
    runner. Initialization-only inputs remain separate from runtime parameters.
    """

    def __init__(
        self,
        population,
        *,
        dt,
        steps,
        extra=None,
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
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
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
            if not configured_callbacks:
                raise ValueError(
                    "execution='while' requires a nonempty callbacks mapping"
                )
            if steps < 4:
                raise ValueError("execution='while' requires at least four steps")
            if compile_options.get("disable", False):
                raise ValueError("execution='while' requires compilation to be enabled")
        self._inference = inference
        self._initialization = initialization
        self._execution = execution
        self._steps = steps
        self._checkpoint_every = checkpoint_every
        # Versioned snapshots must remain ordinary tensors even when the
        # convenience constructor is called inside inference_mode().
        with torch.inference_mode(False), torch.no_grad():
            self._functional, tensors = make_functional(population, dt=dt, extra=extra)
            self._parameters = tensors.independent_parameters(trainable=trainable)
            self._constants = dict(tensors.constants)
            self._initial_state = pytree.tree_map_only(
                torch.Tensor, lambda value: value.detach().clone(), tensors.state
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
            steps,
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
        return self._functional

    @property
    def kernel(self):
        return self._kernel

    def execution_report(self):
        """Return the underlying kernel's latest successful execution report."""
        return self._kernel.execution_report()

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
        if cache and self._prepared is not None:
            try:
                self._functional._validate_prepared_freshness(
                    self._prepared, parameters
                )
            except FunctionalizationError as exc:
                # Refresh normal model/constant edits, but do not hide a
                # corrupted workspace, invalid schema, or foreign plan.
                if str(exc) not in (
                    "parameters changed after prepare(); call prepare() again",
                    "constants changed after prepare(); call prepare() again",
                ):
                    raise
            else:
                return self._prepared
        prepared = prepare(
            parameters, self._constants if constants is None else constants
        )
        if cache:
            self._prepared = prepared
        return prepared

    def _require_while(self, parameters, prepared, state, inputs, callback_state):
        """Fail before a rejected structured loop could compile an unrolled horizon."""
        functional = self._functional
        functional._validate_state(state)
        values = functional._validate_prepared(prepared, parameters)
        ve, intra, _ = functional._resolve_rollout_inputs(inputs, self._steps)
        callbacks = self._callbacks
        carry = (
            callbacks._initialize(state)[0]
            if callback_state is None
            else callbacks._validate_state(callback_state)
        )
        chunk = self._kernel._runner_callback_chunk(callbacks, self._steps)
        executor = chunk._recorder_while
        if executor is None:
            raise FunctionalizationError(
                "execution='while' has no functional callback loop executor"
            )
        kernel, reason = executor.kernel_for(
            parameters,
            values,
            state,
            ve,
            intra,
            carry.carries,
            None if inputs is None else inputs.noise,
            None if inputs is None else inputs.stimulation,
        )
        if kernel is None:
            raise FunctionalizationError(f"execution='while' is unavailable: {reason}")

    def __call__(
        self,
        state=None,
        *,
        parameters=None,
        constants=None,
        inputs=None,
        callback_state=None,
        initialization=None,
    ):
        _validate_host_call()
        fresh_policy = self._initialization == "fresh"
        if initialization is not None:
            if not fresh_policy:
                raise ValueError(
                    "initialization overrides require initialization='fresh'"
                )
            if state is not None:
                raise ValueError(
                    "initialization cannot be combined with explicit state"
                )
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
        selected_parameters = (
            self._parameters if parameters is None else dict(parameters)
        )
        selected_initialization = (
            _initialization_input_mapping(
                self._initialization_inputs
                if initialization is None
                else initialization
            )
            if fresh
            else None
        )
        selected_constants = self._constants if constants is None else dict(constants)
        if fresh:
            # Physical preparation requires ordinary, versioned tensors even
            # for inference. Respect the caller's grad context and never retain
            # this initialization graph or its parameter replacements.
            initialization_grad = torch.is_grad_enabled() and not self._inference
            with (
                torch.inference_mode(False),
                torch.set_grad_enabled(initialization_grad),
            ):
                initialized = self._functional.initialize_host(
                    selected_parameters,
                    selected_constants,
                    selected_initialization,
                    state=selected_state,
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
                    )
                )
            )
        )
        context = torch.inference_mode() if self._inference else nullcontext()
        with context:
            if owned_boundaries:
                self._functional._validate_source()
            prepare = (
                self._prepare_source_validated if owned_boundaries else self._prepare
            )
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
            if inputs is None:
                inputs = RolloutInput()
            if not isinstance(inputs, RolloutInput):
                raise TypeError("inputs must be a RolloutInput or None")
            if self._execution == "while" and inputs.noise is None:
                sample_noise = (
                    self._functional._sample_noise_draw_tape_source_validated
                    if owned_boundaries
                    else self._functional.sample_noise_draw_tape
                )
                inputs = inputs._replace(
                    noise=sample_noise(selected_state, self._steps)
                )
            if (
                self._execution == "while"
                and inputs.stimulation is None
                and getattr(self._functional, "_poisson_plan", None) is not None
            ):
                sample_stimulation = (
                    self._functional._sample_stimulation_draw_tape_source_validated
                    if owned_boundaries
                    else self._functional.sample_stimulation_draw_tape
                )
                inputs = inputs._replace(
                    stimulation=sample_stimulation(selected_state, self._steps)
                )
            if self._execution == "while":
                if not owned_boundaries:
                    # A replaced public sampler can run user code after its
                    # own check. Recheck before preflight captures a step.
                    self._functional._validate_source()
                self._require_while(
                    bound._parameters, prepared, selected_state, inputs, callback_state
                )
            schedule = _run_source_validated if owned_boundaries else run
            schedule_options = (
                {}
                if self._checkpoint_every is None
                else {"checkpoint_every": self._checkpoint_every}
            )
            final, auxiliary = schedule(
                bound,
                selected_state,
                inputs=inputs,
                steps=self._steps,
                callbacks=self._callbacks,
                callback_state=callback_state,
                **schedule_options,
            )
            if fresh_policy:
                auxiliary = dict(auxiliary)
                auxiliary["parameters"] = dict(selected_parameters)
                auxiliary["constants"] = dict(selected_constants)
            return final, auxiliary


def compile(
    population,
    *,
    dt,
    steps,
    extra=None,
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
    """Create a reusable Population or Network runner (scan by default).

    Initialized canonical Networks return :class:`CompiledNetworkRunner` with
    their existing timestep/mode. Their
    inputs use :class:`NetworkRolloutInput`, recordings use qualified names
    such as ``cell.v``, and parameters retain shared tensor identities. See
    :class:`CompiledNetworkRunner` for its restart and continuation contract.
    Network inference supports ``execution='while'`` with qualified Recorders,
    ``PopulationCallback`` scopes around supported built-ins, or without
    callbacks; use scan for gradients and custom callbacks.
    ``pending_capacity`` configures Network pending-event calendars with one
    positive entry count or a mapping from connection names to counts. Overflow
    raises before a result is returned; increase capacity and retry from a valid
    checkpoint. Population runners do not accept this option.

    Recording and reductions are configured only through named ``callbacks``.
    For example, ``callbacks={'trace': dn.func.Recorder(('v',))}`` records
    voltage at ``aux['callbacks']['trace']['v']`` after ``state, aux = runner()``.
    Omit callbacks to return the final state without a trajectory. Select
    compartments with ``dn.func.Recorder(..., node_indices=[...])``, not a
    runner argument. CPU inference with ordinary Inductor defaults to
    ``options={'cpp_wrapper': True}``; explicit compiler choices win. Use
    ``inference=False, trainable=(...)`` for scan gradients. Supply
    exact/unique partial names, glob patterns, or ``re:``/compiled regex selectors
    to ``trainable``; inspect ``runner.trainable_parameters`` for the live named
    defaults selected for gradients. Inference execution still suppresses gradients.
    Add ``checkpoint_every=H`` to trade backward recomputation of H-step compiled
    scans for lower activation memory, without truncating gradients. This is
    valid only for training scan; requested recording storage remains.
    By default, calls without state restart the captured initialized snapshot.
    Population and Network runners accept ``initialization='fresh'`` to initialize
    from current parameters on each such call; explicit state still continues
    without reinitialization. Bound steady-state caches restore their explicit
    saved state and detached materializations on fresh calls. Edit
    ``runner.initialization_inputs`` or pass an ``InitializationInput`` (Population)
    or ``NetworkInitializationInput`` (Network) as the
    call's ``initialization=`` override. Fresh-policy results expose effective
    mappings at ``aux['parameters']`` and ``aux['constants']``. Parameter-writing
    initialization requires the returned parameters for continuation; cached
    initialization also requires the returned constants.
    For Populations, ``extra=(field, waveform)`` or a sequence of such pairs
    binds extracellular stimulation. For Networks, ``extra`` maps Population
    names to those same bindings. Select independent fields and waveform leaves
    through ``trainable``; inspect ``runner.parameters`` for canonical names.
    See :class:`CompiledRunner` and :class:`CompiledNetworkRunner` for all
    constructor options.
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
        extra=extra,
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
