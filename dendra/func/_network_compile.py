"""Snapshot-based Network convenience execution over shared pure components."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from types import MappingProxyType

import torch
from torch.utils import _pytree as pytree

from dendra.models.core import _validate_time_scalar

from ._network import make_functional_network
from ._network_parameters import get_parameter, named_parameters, parameter_name
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
    """Fixed-horizon ``dn.compile(Network)`` execution from an initialized snapshot.

    No-state calls restart :attr:`initial_state`, including delivery queues and
    NetStim clock/RNG carry. An explicit state continues without initialization;
    continue recordings with ``callback_state=aux['callbacks'].state``. Calls
    return ``(state, aux)`` and leave the native Network and restart snapshot
    unchanged. Use :meth:`FunctionalNetwork.commit_state_` to publish carry.

    :attr:`parameters` is a nested read-only view with editable, independent
    Tensor leaves. :attr:`named_parameters` exposes dotted query names. Select
    exact or unique partial names with ``trainable`` and ``inference=False``.
    Every alias of a selected shared Tensor remains the same trainable leaf.
    Per-call parameter overrides preserve their Tensor objects and autograd
    history, and must retain the Network's parameter ties and component schema.

    Runtime inputs use :class:`NetworkRolloutInput`. Supplied draw/RNG tapes
    are replayed explicitly; omitted NetStim tapes are sampled by the shared
    host scheduler from the supplied state's RNG carry. Qualified Recorders and
    pure callback reducers use the existing Network callback binding.

    Preparation and output graphs are invocation-local; only compiled kernels
    are cached. ``checkpoint_every`` preserves gradients across exact spans.
    The initialized Network's mode and timestep are frozen. Gradient execution
    requires a Network initialized in training mode. Fresh initialization,
    inference while loops, and constructor-bound ``extra`` are not supported
    by this snapshot adapter; explicit ``ve`` tapes remain available.
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
        backend="inductor",
        options=None,
        **compile_options,
    ):
        _validate_host_construction(compile_options)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if execution != "scan":
            raise ValueError(
                "Network compiled runners require execution='scan'; while execution is not yet supported"
            )
        if not isinstance(inference, bool):
            raise TypeError("inference must be a bool")
        if initialization != "snapshot":
            raise ValueError(
                "Network compiled runners require initialization='snapshot'; fresh initialization is not yet supported"
            )
        if extra is not None:
            raise ValueError(
                "Network compiled runners do not bind extra; supply per-population NetworkRolloutInput ve tapes"
            )
        if checkpoint_every is not None:
            if (
                isinstance(checkpoint_every, bool)
                or not isinstance(checkpoint_every, int)
                or checkpoint_every < 1
            ):
                raise ValueError("checkpoint_every must be a positive integer")
            if inference:
                raise ValueError(
                    "checkpoint_every requires execution='scan' and inference=False"
                )
        configured = _configured_callbacks(callbacks)
        dt_value = _validate_time_scalar(dt, name="dt", positive=True)
        with torch.inference_mode(False), torch.no_grad():
            self._functional, tensors = make_functional_network(network)
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
            self._callbacks = (
                None if not configured else self._functional.make_callbacks(configured)
            )
        self._inference = inference
        self._steps = steps
        self._checkpoint_every = checkpoint_every
        keywords = _inference_compile_options(
            self._functional, inference, backend, options, compile_options
        )
        self._kernel = self._functional.compile_rollout_chunk(
            steps, execution="scan", **keywords
        )

    @property
    def initial_state(self):
        """Complete restart snapshot; Tensor leaves are editable by the caller."""
        return self._initial_state

    @property
    def parameters(self):
        """Read-only nested parameter packet with independent, live Tensor leaves."""
        return _map_containers(self._parameters, MappingProxyType)

    @property
    def named_parameters(self):
        """Read-only dotted names for the same live leaves as ``parameters``."""
        return MappingProxyType(named_parameters(self._parameters))

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
        """Resolve an exact name or a unique partial parameter query."""
        return parameter_name(self._parameters, query)

    def get_parameter(self, query):
        """Return the live Tensor selected by an exact or unique partial query."""
        return get_parameter(self._parameters, query)

    def execution_report(self):
        """Return the latest successful kernel dispatch, including tails/replay."""
        return self._kernel.execution_report()

    def __call__(
        self,
        state=None,
        *,
        parameters=None,
        inputs=None,
        callback_state=None,
        initialization=None,
    ):
        _validate_host_call()
        if initialization is not None:
            raise ValueError(
                "Network compiled runners use initialization='snapshot' and do not accept initialization overrides"
            )
        if callback_state is not None and state is None:
            raise ValueError(
                "callback_state continuation requires explicit Network state"
            )
        selected_state = self._initial_state if state is None else state
        # Materialize external container code before the plan validates native
        # source structure. Tensor identities/history, including ties, survive.
        selected_parameters = (
            self._parameters
            if parameters is None
            else _map_containers(parameters, dict)
        )
        previous = self._kernel.execution_report()
        try:
            with torch.inference_mode() if self._inference else nullcontext():
                return self._functional.run(
                    selected_parameters,
                    self._constants,
                    selected_state,
                    inputs,
                    steps=self._steps,
                    callbacks=self._callbacks,
                    callback_state=callback_state,
                    checkpoint_every=self._checkpoint_every,
                    compiled_chunk=self._kernel,
                    chunklength=self._steps,
                )
        except Exception:
            self._kernel._report_state.report = previous
            raise


__all__ = ["CompiledNetworkRunner"]
