"""Network convenience execution over shared pure components."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from types import MappingProxyType

import torch
from torch.utils import _pytree as pytree

from dendra.models.core import _validate_time_scalar

from ._continuation import _complete_continuation, _resolve_continuation
from ._draws import _normalize_runtime_draws
from ._initialization import _initialize_fresh, _runner_sources
from ._lowering import _lower_network
from ._network_bindings import _map_containers
from ._network_initialization import NetworkInitializationInput
from ._network_parameters import (
    find_parameters,
    get_parameter,
    named_parameters,
    parameter_name,
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
from ._runtime_extra import (
    _MISSING_RUNTIME_EXTRA,
    VarInput,
)
from ._snapshot import _snapshot_frames
from ._types import FunctionalizationError


class CompiledNetworkRunner:
    """Compiled Network execution with shared parameters and explicit continuation.

    Create with :func:`compile`. Calls return ``(state, auxiliary)`` without
    advancing the native Network. A no-state call restarts the configured
    snapshot/fresh policy, including its RNG and delivery state. Pass the returned
    ``auxiliary["continuation"]`` to continue sources, state and callback phase.

    Parameters are independent nested Tensor mappings with shared aliases
    preserved. ``named_parameters`` provides dotted names; ``trainable_parameters``
    lists live defaults requiring gradients. Per-call overrides remain local.
    Training requires a Network initialized in training mode and
    ``inference=False``. Scan supports gradients; while is inference-only.

    Inputs map Population names to :class:`Inputs`; :class:`RuntimeDraws` supplies
    stochastic tapes. Recorders use qualified paths such as ``left.v`` or
    :class:`PopulationCallback` for a local callback. ``steps``/``duration``
    overrides reuse the configured kernel width and cached tails. Activation
    checkpointing preserves whole-run gradients, RNG and callback continuation.

    ``extra`` maps Population names to extracellular field/waveform bindings.
    ``VarInput()`` declares deterministic runtime slots; omitted names in a supplied
    runtime mapping receive zero drive. Bind randomized Poisson waveforms during
    compilation. ``pending_capacity`` controls event-calendar capacity; overflow
    raises without publishing an incomplete result.

    Use :func:`capture_continuation` to snapshot a native boundary and
    :func:`commit_continuation_` to transfer detached effective sources and state
    back to a compatible native model. See :func:`compile` and :meth:`__call__` for
    construction and invocation arguments.
    """

    def __init__(
        self,
        network,
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
        _validate_host_construction(compile_options)
        steps, duration, nominal_steps = _runner_default_horizon(steps, duration, dt)
        if execution not in {"scan", "while"}:
            raise ValueError("execution must be 'scan' or 'while'")
        if not isinstance(inference, bool):
            raise TypeError("inference must be a bool")
        if execution == "while":
            if not inference:
                raise ValueError(
                    "execution='while' is inference-only; use scan for gradients"
                )
            if compile_options.get("disable", False):
                raise ValueError("execution='while' requires compilation to be enabled")
        if initialization not in ("snapshot", "fresh"):
            raise ValueError("initialization must be 'snapshot' or 'fresh'")
        if checkpoint_every is not None:
            if (
                isinstance(checkpoint_every, bool)
                or not isinstance(checkpoint_every, int)
                or checkpoint_every < 1
            ):
                raise ValueError("checkpoint_every must be a positive integer")
            if inference or execution != "scan":
                raise ValueError(
                    "checkpoint_every requires execution='scan' and inference=False"
                )
        configured = _configured_callbacks(callbacks)
        dt_value = _validate_time_scalar(dt, name="dt", positive=True)
        if isinstance(extra, VarInput):
            runtime_extra_names = tuple(network.populations)
            bound_extra = None
        elif isinstance(extra, Mapping):
            extra = dict(extra)
            unknown = extra.keys() - network.populations.keys()
            if unknown:
                raise ValueError(
                    f"extra names unknown Populations: {sorted(unknown, key=str)!r}"
                )
            runtime_extra_names = tuple(
                name for name, value in extra.items() if isinstance(value, VarInput)
            )
            bound_extra = {
                name: value
                for name, value in extra.items()
                if not isinstance(value, VarInput)
            }
        else:
            runtime_extra_names, bound_extra = (), extra
        self._runtime_extra_names = runtime_extra_names
        self._chunk_steps = _runner_chunk_steps(
            chunk_steps, steps=nominal_steps, runtime_extra=bool(runtime_extra_names)
        )
        with torch.inference_mode(False), torch.no_grad():
            self._functional, tensors = _lower_network(
                network,
                dt=dt_value,
                extra=bound_extra,
                pending_capacity=pending_capacity,
            )
            if not inference and not self._functional.training:
                raise FunctionalizationError(
                    "inference=False requires a Network initialized in training mode; call train() and initialize() before compiling"
                )
            self._parameters, self._constants, self._initial_state = _snapshot_frames(
                self._functional, tensors, trainable=trainable
            )
            self._initialization_inputs = pytree.tree_map_only(
                torch.Tensor,
                lambda value: value.detach().clone(),
                self._functional.initialization_inputs,
            )
            from ._convenience import _initialization_writes_parameters

            self._initialization_writes_parameters = initialization == "fresh" and any(
                plan._owned_cache_bound or _initialization_writes_parameters(plan)
                for plan in self._functional.populations.values()
            )
            if initialization == "fresh" and self._functional._cache_enabled:
                self._initialization_writes_parameters = any(
                    plan.parameters
                    for plan in self._functional._population_cache_plans.values()
                )
            self._initialization_writes_constants = initialization == "fresh" and (
                self._initialization_writes_parameters
                or bool(self._functional._connection_sampling_plan.devices)
                or any(
                    plan.constants
                    for plan in self._functional._population_cache_plans.values()
                )
                or any(
                    plan._owned_cache_bound
                    or bool(plan._has_snapshot_materialization_offsets)
                    or (
                        plan._random_plan is not None
                        and bool(plan._random_plan.entries)
                    )
                    for plan in self._functional.populations.values()
                )
            )
            self._callbacks = (
                None if not configured else self._functional.make_callbacks(configured)
            )
        self._inference = inference
        self._execution = execution
        self._initialization = initialization
        self._steps = steps
        self._duration = duration
        self._checkpoint_every = (
            min(checkpoint_every, self._chunk_steps)
            if checkpoint_every is not None and runtime_extra_names
            else checkpoint_every
        )
        keywords = _inference_compile_options(
            self._functional, inference, backend, options, compile_options
        )
        self._kernel = self._functional.compile_rollout_chunk(
            self._chunk_steps, execution=execution, **keywords
        )

    @property
    def initial_state(self):
        """Complete restart snapshot; explicit state always bypasses initialization.

        Fresh-policy no-state calls use :attr:`initialization_inputs` instead.
        """
        return self._initial_state

    @property
    def initialization_inputs(self):
        """Editable independent defaults for Population, clock and cache initialization.

        Initialization-only leaves are separate from :attr:`parameters`; use
        ``requires_grad_()`` or explicit overrides to differentiate fresh
        initial values. Captured steady-state cache values are detached.
        """
        return self._initialization_inputs

    @property
    def parameters(self):
        """Read-only nested parameter packet with independent, live Tensor leaves."""
        return _map_containers(self._parameters, MappingProxyType)

    @property
    def named_parameters(self):
        """Read-only dotted names for the same live leaves as ``parameters``."""
        return MappingProxyType(named_parameters(self._parameters))

    @property
    def trainable_parameters(self):
        """Read-only named defaults currently requiring gradients, including aliases.

        Reflects live freezing/unfreezing on access, not per-call overrides.
        Deduplicate values by identity before supplying them to an optimizer.
        Inference execution suppresses gradients even for selected leaves.
        """
        return MappingProxyType(
            {
                name: value
                for name, value in named_parameters(self._parameters).items()
                if value.requires_grad
            }
        )

    @property
    def constants(self):
        """Read-only nested constants; edits to Tensor values apply on the next call."""
        return _map_containers(self._constants, MappingProxyType)

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
        """Export detached owned continuation data, retaining source ties/flags.

        The versioned plain data/Tensor packet supports ``torch.save`` and
        ``torch.load(..., weights_only=True)``. It excludes upstream graphs,
        recording history, input/draw tapes and optimizer state. Packets from
        ``dn.func.initialize`` begin callbacks on their first execution.
        """
        from ._continuation_checkpoints import export_continuation

        return export_continuation(self, continuation)

    def restore_continuation(self, packet):
        """Restore a checkpoint into this compatible initialized Network runner.

        Owner/schema, dtype/device, shared source ties and active callback carry are
        validated before returning owned ordinary Tensor leaves in a
        ``Continuation``. Initialized packets begin callbacks on their first
        execution. No initialization, preparation, RNG sampling or execution runs,
        and runner defaults retain their values.
        """
        from ._continuation_checkpoints import restore_continuation

        return restore_continuation(self, packet)

    @property
    def kernel(self):
        """The fixed-width compiled kernel used by this runner."""
        return self._kernel

    def parameter_name(self, query):
        """Resolve one exact, partial, glob, or regex match among named defaults."""
        return parameter_name(self._parameters, query)

    def get_parameter(self, query):
        """Return the live default Tensor selected by one unambiguous query."""
        return get_parameter(self._parameters, query)

    def find_parameters(self, query):
        """Return all named defaults matching a substring, glob, or regex query."""
        return find_parameters(self._parameters, query)

    def execution_report(self):
        """Return the latest successful kernel dispatch, including tails/replay."""
        return self._kernel.execution_report()

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
        return _execute_network_runner(
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


def _execute_network_runner(
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
    steps, duration = _runner_call_horizon(self._steps, self._duration, steps, duration)
    if draws is not None:
        draws = _normalize_runtime_draws(draws)
    if self._runtime_extra_names:
        if extra is _MISSING_RUNTIME_EXTRA:
            raise TypeError(
                "extra is required for runtime extra slots declared with VarInput()"
            )
        if extra is not None and not isinstance(extra, Mapping):
            raise TypeError("extra must map declared Population names to bindings")
        selected_extra = {} if extra is None else dict(extra)
        unknown = selected_extra.keys() - set(self._runtime_extra_names)
        if unknown:
            raise ValueError(
                "extra names Populations without runtime slots: "
                f"{sorted(unknown, key=str)!r}"
            )
        selected_extra = {
            name: _map_containers(selected_extra.get(name), dict)
            for name in self._runtime_extra_names
        }
    else:
        if extra is not _MISSING_RUNTIME_EXTRA:
            raise TypeError(
                "call-time extra requires slots declared with extra=VarInput()"
            )
        selected_extra = None
    state, parameters, constants, callback_state = _resolve_continuation(
        state, parameters, constants, callback_state, initialization
    )
    fresh_policy = self._initialization == "fresh"
    if initialization is not None:
        if not fresh_policy:
            raise ValueError("initialization overrides require initialization='fresh'")
        if state is not None:
            raise ValueError("initialization cannot be combined with explicit state")
    if callback_state is not None and state is None:
        raise ValueError("callback_state continuation requires explicit Network state")
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
            "initialization writes constants; continue with "
            "constants=aux['constants'] from the preceding call"
        )
    fresh = fresh_policy and state is None
    selected_state = self._initial_state if state is None else state
    # Materialize external container code before the plan validates native
    # source structure. Tensor identities/history, including ties, survive.
    (
        selected_parameters,
        selected_constants,
        selected_initialization,
        selected_inputs,
    ) = _runner_sources(
        self, parameters, constants, initialization, fresh=fresh, inputs=inputs
    )
    previous = self._kernel.execution_report()
    try:
        if fresh:
            if not isinstance(selected_initialization, NetworkInitializationInput):
                raise TypeError("initialization must be a NetworkInitializationInput")
            initialized = _initialize_fresh(
                self._functional,
                selected_parameters,
                selected_constants,
                selected_state,
                selected_initialization,
                gradient=torch.is_grad_enabled() and not self._inference,
            )
            selected_parameters = initialized.parameters
            selected_constants = initialized.constants
            selected_state = initialized.state
        selected_constants = self._functional._bind_parameter_constants(
            selected_parameters, selected_constants
        )
        with torch.inference_mode() if self._inference else nullcontext():
            if duration is not None:
                from ._runners import _stage_network_duration

                selected_state, steps = _stage_network_duration(
                    self._functional, selected_state, duration
                )
            schedule_args = (
                selected_parameters,
                selected_constants,
                selected_state,
                selected_inputs,
            )
            if selected_extra is None:
                schedule = self._functional.run
                schedule_options = {}
            else:
                from ._network_rollout import run

                schedule = run
                schedule_args = (self._functional, *schedule_args)
                schedule_options = {"_runtime_extra": selected_extra}
            if draws is not None:
                schedule_options["draws"] = draws
            final, auxiliary = schedule(
                *schedule_args,
                steps=steps,
                callbacks=self._callbacks,
                callback_state=callback_state,
                checkpoint_every=self._checkpoint_every,
                compiled_chunk=self._kernel,
                chunklength=self._chunk_steps,
                **schedule_options,
            )
            if fresh_policy:
                auxiliary = dict(auxiliary)
                auxiliary["parameters"] = _map_containers(selected_parameters, dict)
                auxiliary["constants"] = _map_containers(selected_constants, dict)
            return _complete_continuation(
                final, auxiliary, selected_parameters, selected_constants
            )
    except Exception:
        self._kernel._report_state.report = previous
        raise


__all__ = ["CompiledNetworkRunner"]
