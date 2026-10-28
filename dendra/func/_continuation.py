"""Graph-preserving host continuation for compiled Population/Network runners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.utils import _pytree as pytree

from ._callbacks import FunctionalCallbackState


def _snapshot_sources(tree):
    """Own public containers while leaving Tensor leaves completely untouched."""
    if isinstance(tree, Mapping):
        return {name: _snapshot_sources(value) for name, value in tree.items()}
    return pytree.tree_map(
        lambda value: _snapshot_sources(value) if isinstance(value, Mapping) else value,
        tree,
        is_leaf=lambda value: isinstance(value, (Mapping, torch.Tensor)),
    )


@dataclass(frozen=True, eq=False, init=False)
class Continuation:
    """Explicit in-memory state and effective sources for a compiled runner.

    Both ``dn.compile`` runners publish this packet at
    ``auxiliary['continuation']``. Pass it as ``program(packet)`` to continue
    simulation state, raw parameters/constants and callback carry together.
    The packet captures the sources used by that call, including per-call
    overrides and fresh/cache initialization replacements.

    Fields cannot be reassigned. Ordinary containers are independent snapshots
    and remain editable, like explicit simulation state. Tensor objects, views,
    aliases and autograd history are retained without cloning or detachment.
    The packet contains no runner, prepared coefficients or recording history.
    It is an in-memory continuation, not a portable detached checkpoint.

    Parameters/constants and callback carry follow the existing receiving
    runner's validation and ownership rules. Packet sources are explicit
    overrides and are not retained in the runner's inference preparation cache.
    ``inputs``, ``draws`` and runtime ``extra`` remain call-local arguments.
    """

    state: object
    parameters: Mapping
    constants: Mapping
    callback_state: FunctionalCallbackState | None

    def __init__(self, state, parameters, constants, callback_state=None):
        if state is None:
            raise ValueError("continuation requires explicit non-None state")
        if not isinstance(parameters, Mapping):
            raise TypeError("continuation parameters must be a mapping")
        if not isinstance(constants, Mapping):
            raise TypeError("continuation constants must be a mapping")
        if callback_state is not None and not isinstance(
            callback_state, FunctionalCallbackState
        ):
            raise TypeError(
                "continuation callback_state must be a FunctionalCallbackState or None"
            )
        object.__setattr__(self, "state", _snapshot_sources(state))
        object.__setattr__(self, "parameters", _snapshot_sources(parameters))
        object.__setattr__(self, "constants", _snapshot_sources(constants))
        # Retain plan identity/metadata and custom carry node types while
        # snapshotting the ordinary containers in each callback's Tensor tree.
        carry = (
            None
            if callback_state is None
            else FunctionalCallbackState(
                carries={
                    name: pytree.tree_map(
                        lambda value: value,
                        carry,
                        is_leaf=lambda value: isinstance(value, torch.Tensor),
                    )
                    for name, carry in callback_state.carries.items()
                },
                _token=callback_state.plan_token,
                _emission_schemas=callback_state.emission_schemas,
            )
        )
        object.__setattr__(self, "callback_state", carry)


def _resolve_continuation(state, parameters, constants, callback_state, initialization):
    if not isinstance(state, Continuation):
        return state, parameters, constants, callback_state
    conflicts = [
        name
        for name, value in (
            ("parameters", parameters),
            ("constants", constants),
            ("callback_state", callback_state),
            ("initialization", initialization),
        )
        if value is not None
    ]
    if conflicts:
        raise ValueError("continuation cannot be combined with " + ", ".join(conflicts))
    return state.state, state.parameters, state.constants, state.callback_state


def _complete_continuation(state, auxiliary, parameters, constants):
    result = dict(auxiliary)
    callbacks = result.get("callbacks")
    result["continuation"] = Continuation(
        state,
        parameters,
        constants,
        None if callbacks is None else callbacks.state,
    )
    return state, result


__all__ = ["Continuation"]
