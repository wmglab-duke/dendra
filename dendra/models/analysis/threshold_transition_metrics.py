"""Gaussian expectations of complete hard descriptors from transition roots.

This module composes an already discovered table of hard-protocol transitions.
It does not discover roots, run a simulator, or certify their parameter slopes.
In particular, roots supplied by a trace-tangent threshold proxy must still be
checked against independent complete hard-protocol reruns.
"""

from __future__ import annotations

import math

import torch


def gaussian_expected_activation(
    center: float | torch.Tensor,
    threshold: torch.Tensor,
    noise_std: float | torch.Tensor,
) -> torch.Tensor:
    r"""Return activation probability for a Gaussian-distributed amplitude.

    The complete hard protocol is assumed to change from inactive to active at
    the scalar ``threshold``.  If the delivered amplitude is

    ``A = center + noise_std * Z``, with ``Z ~ N(0, 1)``, this function returns

    ``P(A >= threshold) = Phi((center - threshold) / noise_std)``.

    ``threshold`` can be the proxy from :func:`trace_tangent_threshold_proxy`.
    Its forward value is then the complete hard threshold, while autograd uses
    the validated local trace-tangent direction.  The Gaussian width is a
    user-selected smoothing scale in the same amplitude units; it describes
    the training objective and is not inferred from the voltage trace.

    Parameters
    ----------
    center
        Mean stimulus amplitude. A tensor may have any shape.
    threshold
        A scalar floating-point tensor. Gradients propagate through it.
    noise_std
        Positive Gaussian standard deviation in stimulus-amplitude units.
    """
    if not isinstance(threshold, torch.Tensor):
        raise TypeError("threshold must be a tensor.")
    if not threshold.is_floating_point():
        raise ValueError("threshold must be floating point.")
    if threshold.numel() != 1:
        raise ValueError("threshold must be scalar.")
    if not bool(torch.isfinite(threshold.detach()).all()):
        raise ValueError("threshold must be finite.")

    root = threshold.reshape(1)
    return gaussian_expected_hard_from_transition_roots(
        center,
        root,
        torch.ones_like(root),
        0.0,
        noise_std,
    )


def _coerce_input(
    value: float | torch.Tensor,
    *,
    name: str,
    like: torch.Tensor,
    scalar: bool,
) -> torch.Tensor:
    """Make Python numbers match roots while rejecting silent tensor casts."""
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point.")
        if value.device != like.device or value.dtype != like.dtype:
            raise ValueError(f"{name} must share the roots' device and dtype.")
        out = value
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{name} must be a floating-point tensor or number."
            ) from error
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite.")
        out = torch.as_tensor(numeric, device=like.device, dtype=like.dtype)
    if scalar and out.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    if not bool(torch.isfinite(out.detach()).all()):
        raise ValueError(f"{name} must be finite.")
    return out


def gaussian_expected_hard_from_transition_roots(
    center: float | torch.Tensor,
    roots: torch.Tensor,
    jumps: torch.Tensor,
    left_value: float | torch.Tensor,
    noise_std: float | torch.Tensor,
) -> torch.Tensor:
    r"""Compose sorted hard transitions under Gaussian stimulus variation.

    Suppose the *complete* hard protocol at amplitude ``A`` is the step
    function

    ``H(A) = H_left + sum_j Delta_j 1{A >= r_j}``,

    with strictly increasing roots ``r_j`` and signed jumps ``Delta_j``.
    For ``A = center + noise_std * Z`` and ``Z ~ N(0, 1)``, return

    ``G(center) = E[H(A)] = H_left + sum_j Delta_j Phi((center-r_j)/noise_std)``.

    ``roots`` and ``jumps`` are one-dimensional tensors of equal length;
    an empty table gives the constant ``H_left``. ``center`` may have any
    shape, so one root table can be evaluated on an amplitude grid. The other
    two inputs are scalars. Torch autograd propagates through ``center``,
    ``roots``, ``jumps``, ``left_value``, and a tensor ``noise_std``. For hard
    count/success descriptors, ``jumps`` and ``left_value`` are fixed event
    values; keeping them constant gives the parameter derivative

    ``dG/dq = -sum_j Delta_j phi((center-r_j)/noise_std) (dr_j/dq)/noise_std``.

    A root may be a hard-forward trace-tangent proxy, allowing one backward
    pass through all connected model parameters. This is a valid local
    training direction only if the root identities and jump sizes remain
    stable, all transitions with relevant Gaussian mass are included, and
    each root tangent follows the corresponding complete hard boundary.
    Signed jumps represent nonmonotone firing or following responses; the
    function does not assume monotone recruitment. Duplicate roots must be
    combined into one signed jump before calling. A full Gaussian requires
    the hard protocol to be defined on its amplitude support; truncation or
    omitted tails require an explicit error bound from the caller.

    Tensor inputs must share dtype and device with ``roots``; Python scalar
    inputs are converted using the roots' dtype and device. Unsorted,
    duplicate, nonfinite, or malformed transition tables are rejected.
    """
    if not isinstance(roots, torch.Tensor) or not isinstance(jumps, torch.Tensor):
        raise TypeError("roots and jumps must be tensors.")
    if roots.ndim != 1 or jumps.ndim != 1 or roots.shape != jumps.shape:
        raise ValueError("roots and jumps must be equal-length vectors.")
    if not roots.is_floating_point() or not jumps.is_floating_point():
        raise ValueError("roots and jumps must be floating point.")
    if roots.device != jumps.device or roots.dtype != jumps.dtype:
        raise ValueError("roots and jumps must share device and dtype.")
    if not bool(torch.isfinite(roots.detach()).all()):
        raise ValueError("roots must be finite.")
    if not bool(torch.isfinite(jumps.detach()).all()):
        raise ValueError("jumps must be finite.")
    if roots.numel() > 1 and not bool((roots.detach()[1:] > roots.detach()[:-1]).all()):
        raise ValueError("roots must be strictly increasing; combine duplicate roots.")

    mu = _coerce_input(center, name="center", like=roots, scalar=False)
    left = _coerce_input(left_value, name="left_value", like=roots, scalar=True)
    sigma = _coerce_input(noise_std, name="noise_std", like=roots, scalar=True)
    if not bool((sigma.detach() > 0).item()):
        raise ValueError("noise_std must be positive.")

    standardized = (mu.unsqueeze(-1) - roots) / sigma
    normal_cdf = 0.5 * torch.erfc(-standardized / math.sqrt(2.0))
    return left + (normal_cdf * jumps).sum(dim=-1)
