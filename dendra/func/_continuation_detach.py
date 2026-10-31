"""Plan-aware truncation of live recurrent history without rebinding sources."""

from __future__ import annotations

import torch

from ._callback_detach import _detach_recurrent_values
from ._continuation import Continuation, _make_continuation
from ._continuation_checkpoints import (
    _cold_boundary,
    _continuation_lifecycle,
    _is_runner_target,
    _validate_callback_state_presence,
    _validate_frames,
)


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

        state, carry = _detach_recurrent_values(
            continuation.state,
            callbacks,
            continuation.callback_state,
        )
        return _make_continuation(
            state,
            continuation.parameters,
            continuation.constants,
            carry,
            lifecycle=lifecycle,
        )


__all__ = ["detach_continuation"]
