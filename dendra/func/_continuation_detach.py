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
    """Cut recurrent autograd history while retaining live source ownership.

    Parameters
    ----------
    functional : FunctionalPopulation, FunctionalNetwork or compiled runner
        Plan/runner defining simulation and callback roles.
    continuation : Continuation
        Packet whose simulation state and recurrent callback carry are detached.
    callbacks : FunctionalCallbacks, optional
        Explicit plan's prepared collection when callback carry is present.
        Runners select their configured callbacks and require this omitted.

    Returns
    -------
    Continuation
        Detached recurrent state with unchanged parameters/constants, aliases,
        trainability, RNG, clocks, accumulated reductions and recording phase.

    Notes
    -----
    Detachment shares storage like ``Tensor.detach``; it is not an owned snapshot.
    Use :func:`export_continuation` for portable owned storage. Source graphs and
    previous outputs remain caller-owned. Rebuild externally computed nonleaf
    source/field expressions when their upstream graph has been consumed.

    Built-in LFP observation fields are refreshed under the caller's gradient
    mode, preserving fresh suffix gradients. Other custom callback carry is
    recurrent. This operation does not initialize callbacks, sample, execute or
    write storage. Call outside compilation, function transforms and forward AD.
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
