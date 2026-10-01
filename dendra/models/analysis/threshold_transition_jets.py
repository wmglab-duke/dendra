"""Compose hard transition roots after independent simulator reverse passes.

A stateful simulator can reuse backward buffers on its next forward. Extract
each root's parameter VJP immediately, release that graph, then use this
first-order jet to combine all roots into one expected-hard scalar. Work
scales with observed transitions and reverse passes, not the number of model
parameter coordinates. Root VJPs still need independent hard-protocol checks.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .threshold_transition_metrics import gaussian_expected_hard_from_transition_roots


def gaussian_expected_hard_from_root_parameter_jets(
    center: float | torch.Tensor,
    root_values: torch.Tensor,
    jumps: torch.Tensor,
    left_value: float | torch.Tensor,
    noise_std: float | torch.Tensor,
    *,
    parameters: Sequence[torch.Tensor],
    root_parameter_gradients: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    r"""Build a first-order differentiable Gaussian hard-descriptor mean.

    ``root_values`` are sorted hard-search root estimates. For each root,
    ``root_parameter_gradients[j][k]`` is the already extracted VJP of that
    root with respect to ``parameters[k]``, in the *same tensor coordinates*.
    Each VJP must come from its own fresh simulator replay before the next
    simulator forward. The function attaches these detached slopes to the
    hard roots and composes signed jumps analytically. It therefore supports
    one optimizer backward through all connected parameters even when the
    original simulator graphs cannot coexist. This construction preserves
    first-order derivatives only; it does not provide simulator Hessians.

    A hard-forward root and a differentiable slope are separate claims.
    Validate every supplied root VJP against complete hard-search parameter
    perturbations on a stable transition branch before using it to train.
    """
    if not isinstance(root_values, torch.Tensor) or root_values.ndim != 1:
        raise ValueError("root_values must be a one-dimensional tensor.")
    if not isinstance(jumps, torch.Tensor) or jumps.shape != root_values.shape:
        raise ValueError("jumps must match root_values.")
    if not root_values.is_floating_point():
        raise ValueError("root_values must be floating point.")
    parameters = tuple(parameters)
    root_parameter_gradients = tuple(tuple(row) for row in root_parameter_gradients)
    if not parameters:
        raise ValueError("At least one trainable parameter is required.")
    if len(root_parameter_gradients) != root_values.numel():
        raise ValueError("Provide one parameter-gradient row per root.")
    for parameter in parameters:
        if (
            not isinstance(parameter, torch.Tensor)
            or not parameter.is_floating_point()
            or not parameter.requires_grad
        ):
            raise ValueError(
                "Parameters must be differentiable floating-point tensors."
            )
        if (
            parameter.device != root_values.device
            or parameter.dtype != root_values.dtype
        ):
            raise ValueError("Parameters must share root_values' device and dtype.")

    jets = []
    for index, gradient_row in enumerate(root_parameter_gradients):
        if len(gradient_row) != len(parameters):
            raise ValueError("Each root needs a VJP for every parameter tensor.")
        jet = root_values[index].detach()
        for parameter, gradient in zip(parameters, gradient_row):
            if (
                not isinstance(gradient, torch.Tensor)
                or gradient.shape != parameter.shape
                or gradient.device != parameter.device
                or gradient.dtype != parameter.dtype
                or not bool(torch.isfinite(gradient.detach()).all())
            ):
                raise ValueError(
                    "Each root VJP must be finite and match its parameter."
                )
            jet = jet + ((parameter - parameter.detach()) * gradient.detach()).sum()
        jets.append(jet)
    roots = torch.stack(jets) if jets else root_values.detach()
    result = gaussian_expected_hard_from_transition_roots(
        center,
        roots,
        jumps,
        left_value,
        noise_std,
    )
    if not jets:
        # Keep a useful zero-gradient connection on a constant hard branch.
        result = result + sum(
            (parameter - parameter.detach()).sum() * 0 for parameter in parameters
        )
    return result
