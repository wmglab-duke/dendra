"""Shared recurrent tree detachment and plan-aware callback invariant refresh."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

from ._callback_carry import _CallbackCarryPartition
from ._callback_checkpoints import _cold_boundary
from ._callbacks import FunctionalCallbackState, _callback_implementation
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


def _refresh_invariants(callbacks, state, memo, transform=None):
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
        value = implementation._materialize_carry()
        memo[key] = value if transform is None else transform(state[plan.name], value)


def _map_recurrent_values(
    state, callbacks, callback_state, transform, *, invariant_transform=None
):
    """Map validated recurrent frames after refreshing their shared invariants."""
    memo = {}
    if callback_state is not None:
        # Seed every invariant before visiting recurrent values: a custom
        # carry or model state can alias an invariant's old Tensor.
        _refresh_invariants(callbacks, callback_state, memo, invariant_transform)

    def mapped(value):
        if isinstance(value, torch.Tensor):
            key = id(value)
            if key not in memo:
                memo[key] = transform(value)
            return memo[key]
        return value

    state = pytree.tree_map(
        mapped, state, is_leaf=lambda value: isinstance(value, torch.Tensor)
    )
    carry = None
    if callback_state is not None:
        carry = FunctionalCallbackState(
            carries=pytree.tree_map(
                mapped,
                callback_state.carries,
                is_leaf=lambda value: isinstance(value, torch.Tensor),
            ),
            _token=callback_state.plan_token,
            _emission_schemas=callback_state.emission_schemas,
        )
        callbacks._validate_state(carry)
    return state, carry


def _detach_recurrent_values(state, callbacks, callback_state):
    """Detach validated recurrent frames with one memo, preserving traversal."""
    return _map_recurrent_values(
        state, callbacks, callback_state, lambda value: value.detach()
    )


def _detach_callback_state(callbacks, state):
    _cold_boundary("FunctionalCallbacks.detach_state")
    grad_enabled = torch.is_grad_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
        with torch.no_grad():
            callbacks._validate_state(state)
        return _detach_recurrent_values(None, callbacks, state)[1]


__all__ = []
