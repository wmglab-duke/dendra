"""Graph-preserving host continuation for Population/Network execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.utils import _pytree as pytree

from ._callbacks import FunctionalCallbacks, FunctionalCallbackState
from ._types import FunctionalizationError


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
    """Explicit in-memory state and effective sources for functional execution.

    Both ``dn.compile`` runners publish this packet at
    ``auxiliary['continuation']``. Pass it as ``program(packet)`` to continue
    simulation state, raw parameters/constants and callback carry together.
    The packet captures the sources used by that call, including per-call
    overrides and fresh/cache initialization replacements.

    Raw-bound ``run(bound, state, ...)`` calls also publish this packet. Resume
    with ``run(bound, packet, ...)`` when its parameter/constant Tensor objects
    match the binding's sources. Different sources require an explicit new raw
    binding. Prepared Population bindings retain explicit state/carry calls.

    Fields cannot be reassigned. Ordinary containers are independent snapshots
    and remain editable, like explicit simulation state. Tensor objects, views,
    aliases and autograd history are retained without cloning or detachment.
    The packet contains no runner, opaque prepared frame or recording history.
    It is an in-memory continuation, not a portable detached checkpoint.

    Parameters/constants and callback carry follow the existing receiving
    execution's validation and ownership rules. For ``dn.compile`` runners,
    packet sources are explicit overrides and are not retained in the runner's
    inference preparation cache. Raw bindings validate matching sources.
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


def _bound_continuation_sources(bound):
    """Project effective raw readers without changing authored binding operands."""
    from ._network_binding import BoundNetwork

    parameters, constants = bound._parameters, bound._constants
    if type(bound) is BoundNetwork:
        constants = bound.functional._bind_parameter_constants(parameters, constants)
    return parameters, constants


def _same_source_tree(actual, expected):
    """Match structure and Tensor objects, never Tensor values or storage."""
    if isinstance(expected, torch.Tensor):
        return actual is expected
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not all(
            isinstance(key, str) for key in actual
        ):
            return False
        return set(actual) == set(expected) and all(
            _same_source_tree(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, tuple):
        return (
            type(actual) is type(expected)
            and len(actual) == len(expected)
            and all(
                _same_source_tree(value, reference)
                for value, reference in zip(actual, expected, strict=True)
            )
        )
    return expected is None and actual is None


def _resolve_bound_continuation(bound, state, callback_state, callbacks):
    """Qualify packet ownership before preparation; ordinary state calls pass through."""
    if not isinstance(state, Continuation):
        return state, callback_state
    state, parameters, constants, callback_state = _resolve_continuation(
        state, None, None, callback_state, None
    )
    if bound._constants is None:
        raise FunctionalizationError(
            "continuation requires a raw binding created with constants=; "
            "prepared Population bindings use explicit state and callback_state"
        )
    expected_parameters, expected_constants = _bound_continuation_sources(bound)
    for name, actual, expected in (
        ("parameters", parameters, expected_parameters),
        ("constants", constants, expected_constants),
    ):
        if not _same_source_tree(actual, expected):
            raise FunctionalizationError(
                f"continuation {name} do not match the bound sources; "
                "create a new raw binding for different sources"
            )
    if callbacks is None:
        if callback_state is not None:
            raise ValueError("callback_state requires callbacks")
    else:
        if not isinstance(callbacks, FunctionalCallbacks):
            raise TypeError(
                "callbacks must be a FunctionalCallbacks plan created by "
                "functional.make_callbacks(...)"
            )
        callbacks._validate_owner(bound.functional)
        if callback_state is not None:
            callbacks._validate_state(callback_state)
    return state, callback_state


def _complete_bound_continuation(bound, result, previous_report):
    """Publish raw sources and restore reports only if this added completion fails."""
    try:
        parameters, constants = _bound_continuation_sources(bound)
        state, auxiliary = result
        return _complete_continuation(
            state, {} if auxiliary is None else auxiliary, parameters, constants
        )
    except BaseException:
        if bound.compiled_chunk is not None:
            bound.compiled_chunk._report_state.report = previous_report
        raise


__all__ = ["Continuation"]
