"""Network convenience execution over shared pure components."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from types import MappingProxyType

import torch
from torch.utils import _pytree as pytree

from dendra.models.core import _validate_time_scalar

from ._network import make_functional_network
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
    _validate_host_call,
    _validate_host_construction,
)
from ._types import FunctionalizationError


def _map_containers(tree, mapping):
    """Snapshot public container structure without copying differentiable leaves."""
    if isinstance(tree, Mapping):
        return mapping(
            {key: _map_containers(value, mapping) for key, value in tree.items()}
        )
    if isinstance(tree, tuple):
        values = tuple(_map_containers(value, mapping) for value in tree)
        return type(tree)(*values) if hasattr(tree, "_fields") else values
    return tree


class CompiledNetworkRunner:
    """Fixed-horizon ``dn.compile(Network)`` execution with explicit restart policy.

    No-state calls restart :attr:`initial_state`, including delivery queues and
    NetStim clock/RNG carry. ``initialization='fresh'`` instead constructs state
    from current parameters and :attr:`initialization_inputs`. An explicit
    state continues without initialization;
    continue recordings with ``callback_state=aux['callbacks'].state``. Calls
    return ``(state, aux)`` and leave the native Network and restart snapshot
    unchanged. Use :meth:`FunctionalNetwork.commit_state_` to publish carry.

    :attr:`parameters` is a nested read-only view with editable, independent
    Tensor leaves. :attr:`named_parameters` exposes dotted query names. Select
    exact or unique partial names, glob patterns, or ``re:``/compiled regular
    expressions with ``trainable`` and ``inference=False``.
    Every alias of a selected shared Tensor remains the same trainable leaf.
    :attr:`trainable_parameters` reports every named default currently requiring
    gradients, including aliases. Per-call overrides are not retained for
    inspection. Inference execution suppresses gradients regardless of selection.
    Per-call parameter overrides preserve their Tensor objects and autograd
    history, and must retain the Network's parameter ties and component schema.

    Runtime inputs use :class:`NetworkRolloutInput`. Supplied draw/RNG tapes
    are replayed explicitly; omitted NetStim tapes are sampled by the shared
    host scheduler from the supplied state's RNG carry. Qualified Recorders and
    pure callback reducers use the existing Network callback binding.

    Preparation and output graphs are invocation-local; only compiled kernels
    are cached. ``checkpoint_every`` preserves gradients across exact spans.
    The initialized Network's mode and timestep are frozen. Gradient execution
    requires a Network initialized in training mode. Fresh calls accept a
    :class:`NetworkInitializationInput` override and return effective parameters
    and constants in ``aux['parameters']`` and ``aux['constants']``. Continue
    those packets explicitly when initialization replaces them. A captured
    Network-wide steady-state cache supplies detached Population
    materializations and rebuilds connection carry using the shared cache
    restoration helpers. Without that global cache, each Population restores
    its own bound cache or runs its fresh initializer in authored order.
    Initialization runs once before
    rollout/checkpoint spans; its graph is never cached. Inference while-loop
    execution uses ``execution='while', inference=True`` with the same pure
    Network timestep and the Population packed loop engine. Qualified Recorders,
    built-ins wrapped in ``PopulationCallback``, and execution without callbacks
    are supported; use scan for custom callbacks and gradients. Short chunks
    contain only the bounded unrolled tail. ``extra`` maps Population names to
    ``(field, Waveform)`` pairs or contact sequences, using the Population
    stimulation machinery. Its fields and waveform leaves are independent
    parameters selected through ``trainable``, with shared aliases preserved.
    Binding stimulation and supplying explicit ``ve`` inputs to the same
    Population cannot be combined.

    ``pending_capacity`` sets the entry count for packed scheduled traffic and
    sparse inference calendars. It accepts one positive integer or a mapping
    keyed by event-connection name. A full calendar raises without changing
    the input state, restart snapshot, native model, or execution report. Compile
    with a larger capacity and retry from a valid checkpoint.
    """

    def __init__(
        self,
        network,
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
        _validate_host_construction(compile_options)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
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
        with torch.inference_mode(False), torch.no_grad():
            self._functional, tensors = make_functional_network(
                network, extra=extra, pending_capacity=pending_capacity
            )
            if dt_value != self._functional.dt:
                raise ValueError(
                    "dt must match the initialized Network timestep; initialize and compile again to change it"
                )
            if not inference and not self._functional.training:
                raise FunctionalizationError(
                    "inference=False requires a Network initialized in training mode; call train() and initialize() before compiling"
                )
            self._parameters = tensors.independent_parameters(trainable)
            self._constants = _map_containers(tensors.constants, dict)
            self._initial_state = pytree.tree_map_only(
                torch.Tensor, lambda value: value.detach().clone(), tensors.state
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
                any(
                    plan.constants
                    for plan in self._functional._population_cache_plans.values()
                )
                or any(
                    plan._owned_cache_bound
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
        self._checkpoint_every = checkpoint_every
        keywords = _inference_compile_options(
            self._functional, inference, backend, options, compile_options
        )
        self._kernel = self._functional.compile_rollout_chunk(
            steps, execution=execution, **keywords
        )
        self._kernel._validate_while_callbacks(self._callbacks)

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
        return self._functional

    @property
    def kernel(self):
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
        if callback_state is not None and state is None:
            raise ValueError(
                "callback_state continuation requires explicit Network state"
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
                "initialization writes constants; continue with "
                "constants=aux['constants'] from the preceding call"
            )
        fresh = fresh_policy and state is None
        selected_state = self._initial_state if state is None else state
        # Materialize external container code before the plan validates native
        # source structure. Tensor identities/history, including ties, survive.
        selected_parameters = (
            self._parameters
            if parameters is None
            else _map_containers(parameters, dict)
        )
        selected_inputs = _map_containers(inputs, dict)
        selected_initialization = (
            _map_containers(
                self._initialization_inputs
                if initialization is None
                else initialization,
                dict,
            )
            if fresh
            else None
        )
        selected_constants = (
            self._constants if constants is None else _map_containers(constants, dict)
        )
        previous = self._kernel.execution_report()
        try:
            if fresh:
                if not isinstance(selected_initialization, NetworkInitializationInput):
                    raise TypeError(
                        "initialization must be a NetworkInitializationInput"
                    )
                # Fresh coefficients and effective parameters require ordinary,
                # versioned tensors even under an outer inference context.
                initialization_grad = torch.is_grad_enabled() and not self._inference
                with (
                    torch.inference_mode(False),
                    torch.set_grad_enabled(initialization_grad),
                ):
                    initialized = self._functional._initialize_host(
                        selected_parameters,
                        selected_constants,
                        selected_state,
                        selected_initialization,
                    )
                selected_parameters = initialized.parameters
                selected_constants = initialized.constants
                selected_state = initialized.state
            with torch.inference_mode() if self._inference else nullcontext():
                final, auxiliary = self._functional.run(
                    selected_parameters,
                    selected_constants,
                    selected_state,
                    selected_inputs,
                    steps=self._steps,
                    callbacks=self._callbacks,
                    callback_state=callback_state,
                    checkpoint_every=self._checkpoint_every,
                    compiled_chunk=self._kernel,
                    chunklength=self._steps,
                )
                if fresh_policy:
                    auxiliary = dict(auxiliary)
                    auxiliary["parameters"] = _map_containers(selected_parameters, dict)
                    auxiliary["constants"] = _map_containers(selected_constants, dict)
                return final, auxiliary
        except Exception:
            self._kernel._report_state.report = previous
            raise


__all__ = ["CompiledNetworkRunner"]
