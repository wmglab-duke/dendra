"""Utilities for making optimizer gradients safe to apply."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch.optim import Optimizer


def _validate_clip_limit(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")

    limit = float(value)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return limit


def sanitize_grad(
    optimizer: Optimizer,
    clip_norm: float | None = None,
    clip_value: float | None = None,
) -> None:
    """Sanitize and optionally clip an optimizer's gradients in place.

    ``NaN``, positive infinity, and negative infinity are replaced with zero.
    If both clipping limits are supplied, element-wise value clipping is
    applied first, followed by L2-norm clipping.  The norm is computed over all
    gradients in all optimizer parameter groups as if they formed one vector.
    Parameters whose gradient is ``None`` are ignored.

    Parameters
    ----------
    optimizer
        Optimizer containing the parameters whose gradients should be
        sanitized. Call this function after backpropagation and before
        :meth:`~torch.optim.Optimizer.step`.
    clip_norm
        Optional finite, non-negative maximum for the global L2 norm.
    clip_value
        Optional finite, non-negative maximum absolute value for each gradient
        element.

    Raises
    ------
    TypeError
        If a clipping limit is not a real number or ``None``, or if an
        optimizer parameter has a non-strided (for example, sparse) gradient.
    ValueError
        If a clipping limit is negative or non-finite.

    Notes
    -----
    Only dense, strided gradients are supported. Unsupported layouts are
    detected before any gradient is mutated. The operation runs under
    ``torch.no_grad()`` and does not replace existing gradient tensors. It is
    an optimizer hygiene operation, not a differentiable transformation of
    the backward graph.

    Examples
    --------
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> sanitize_grad(optimizer, clip_norm=1.0, clip_value=10.0)
    >>> optimizer.step()
    """
    max_norm = _validate_clip_limit(clip_norm, "clip_norm")
    max_value = _validate_clip_limit(clip_value, "clip_value")
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]

    if not parameters:
        return

    unsupported_layouts = {
        parameter.grad.layout
        for parameter in parameters
        if parameter.grad.layout != torch.strided
    }
    if unsupported_layouts:
        layouts = ", ".join(sorted(str(layout) for layout in unsupported_layouts))
        message = (
            f"sanitize_grad supports only dense, strided gradients; found {layouts}"
        )
        raise TypeError(message)

    with torch.no_grad():
        for parameter in parameters:
            parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        if max_value is not None:
            torch.nn.utils.clip_grad_value_(parameters, max_value)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
