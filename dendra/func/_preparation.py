"""Invocation-owned Population preparation and detached inference reuse."""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

from ._types import FunctionalizationError


def _cacheable_preparation_inputs(parameters, constants):
    """Reuse ordinary, versioned inputs without differentiation or autocast."""
    if (
        torch.is_grad_enabled()
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
        or torch._C._is_any_autocast_enabled()
    ):
        return False
    for value in pytree.tree_leaves((parameters, constants)):
        if not torch.is_tensor(value):
            continue
        if (
            type(value) not in (torch.Tensor, torch.nn.Parameter)
            or torch.is_inference(value)
            or value.grad_fn is not None
        ):
            return False
    return True


def _cacheable_preparation_values(prepared):
    """Do not retain graphs from hooks that locally enable differentiation."""
    return all(
        type(value) in (torch.Tensor, torch.nn.Parameter)
        and not torch.is_inference(value)
        and value.grad_fn is None
        for value in pytree.tree_leaves(prepared.values)
        if torch.is_tensor(value)
    )


def _select_population_preparation(
    functional,
    parameters,
    constants,
    *,
    cached=None,
    cache=False,
    prepare=None,
):
    """Share freshness selection without owning or retaining a training graph.

    Callers retain a result only when their inference policy permits it. Normal
    input edits refresh cached values; invalid workspaces and foreign plans
    keep their existing errors instead of being hidden by a new preparation.
    """
    if cache and cached is not None:
        try:
            functional._validate_prepared_freshness(cached, parameters)
        except FunctionalizationError as error:
            if str(error) not in (
                "parameters changed after prepare(); call prepare() again",
                "constants changed after prepare(); call prepare() again",
            ):
                raise
        else:
            return cached
    if prepare is None:
        prepare = functional.prepare
    return prepare(parameters, constants)
