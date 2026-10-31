"""Plan-aware truncation of live recurrent history without rebinding sources."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

from ._callback_carry import _CallbackCarryPartition
from ._callbacks import FunctionalCallbackState, _callback_implementation
from ._continuation import Continuation, _make_continuation
from ._continuation_checkpoints import (
    _cold_boundary,
    _continuation_lifecycle,
    _is_runner_target,
    _validate_callback_state_presence,
    _validate_frames,
)
from ._types import FunctionalizationError


def _lfp_recipe(callback):
    source = callback._stacked_source
    return (
        ("stacked", id(source))
        if source is not None
        else ("fields", *(id(field) for field in callback.fields)),
        callback.device,
        callback.dtype,
        callback.padding if callback.rule is None else None,
    )


def _refresh_invariants(callbacks, state, memo):
    partition = _CallbackCarryPartition.from_plans(callbacks._plans)
    recipes = {}
    for plan in callbacks._plans:
        if plan.name not in partition.invariant_names:
            continue
        implementation = _callback_implementation(plan.callback)
        key = id(state[plan.name])
        recipe = _lfp_recipe(implementation)
        if key in recipes:
            if recipe != recipes[key]:
                raise FunctionalizationError(
                    "aliased LFP carry cannot be refreshed from different field sources"
                )
            continue
        recipes[key] = recipe
        memo[key] = implementation._materialize_carry()


def detach_continuation(functional, continuation, *, callbacks=None):
    """Truncate recurrent history for a lowered plan or ``dn.compile`` runner.

    ``functional`` accepts the same Population/Network plans and compiled
    runners as :func:`export_continuation`. Lowered plans require their bound
    callback collection when the packet carries callback state. Runners use
    their constructor-configured callbacks and require ``callbacks`` omitted.

    Returns a new :class:`Continuation` with detached simulation state and
    recurrent callback carry. Parameter and constant Tensor objects, aliases,
    trainability and upstream graphs remain untouched. Callback plan identity,
    emission metadata, cumulative values, cadence, clocks, RNG and lifecycle
    are retained. Initialization, sampling, execution and storage are not run.
    Unstarted packets remain unstarted and do not initialize callbacks.

    Built-in LFP carry holds observation inputs rather than recurrent history.
    Its field stack is rematerialized from the bound current field sources,
    retaining fresh suffix gradients when called with gradients enabled.
    All custom callback carry is recurrent and is detached. Refresh follows
    the caller's gradient mode, including ``no_grad`` and ``inference_mode``.

    Detachment shares storage like :meth:`torch.Tensor.detach`; it is not a
    snapshot. Tensor identity aliases across state and callback carry are
    retained, and distinct views retain their storage and strides. A Tensor
    shared between raw sources and recurrent data is detached only in its
    recurrent role. Aliases to an LFP invariant use its refreshed field stack;
    conflicting field sources for one aliased stack are rejected.

    This operation cuts recurrent history, not every graph in the packet.
    Externally computed nonleaf sources and callback field expressions remain
    caller-owned and must be rebuilt when their upstream graph is consumed.
    Previous outputs are also caller-owned. Use portable export/restore for
    detached owned checkpoint storage. Construction runs outside compilation,
    function transforms and forward AD.
    """
    _cold_boundary("detach_continuation")
    if not isinstance(continuation, Continuation):
        raise TypeError("detach_continuation requires a Continuation")
    runner = functional if _is_runner_target(functional, callbacks) else None
    owner = "this functional plan" if runner is None else "this runner"
    if runner is not None:
        functional = runner.functional
        callbacks = runner.callbacks
    grad_enabled = torch.is_grad_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
        with torch.no_grad():
            functional._validate_source()
            _validate_frames(functional, continuation)
            lifecycle = _continuation_lifecycle(continuation)
            has_carry = _validate_callback_state_presence(
                callbacks, continuation.callback_state, lifecycle=lifecycle, owner=owner
            )
            if has_carry:
                callbacks._validate_state(continuation.callback_state)

        memo = {}
        if has_carry:
            # Seed every invariant before visiting recurrent values: a custom
            # carry or model state can alias an invariant's old Tensor.
            _refresh_invariants(callbacks, continuation.callback_state, memo)

        def detach(value):
            if isinstance(value, torch.Tensor):
                key = id(value)
                if key not in memo:
                    memo[key] = value.detach()
                return memo[key]
            return value

        state = pytree.tree_map(
            detach,
            continuation.state,
            is_leaf=lambda value: isinstance(value, torch.Tensor),
        )
        carry = None
        if has_carry:
            previous = continuation.callback_state
            carry = FunctionalCallbackState(
                carries=pytree.tree_map(
                    detach,
                    previous.carries,
                    is_leaf=lambda value: isinstance(value, torch.Tensor),
                ),
                _token=previous.plan_token,
                _emission_schemas=previous.emission_schemas,
            )
            callbacks._validate_state(carry)
        return _make_continuation(
            state,
            continuation.parameters,
            continuation.constants,
            carry,
            lifecycle=lifecycle,
        )


__all__ = ["detach_continuation"]
