"""Expected hard spike descriptors under a shared noisy detection threshold.

These functions smooth a *specified measurement protocol*: draw one logistic
threshold offset per fiber, then apply the ordinary hard crossing rule to all
selected samples and sites in that fiber.  They are not the derivatives of a
deterministic binary descriptor.  Their derivatives describe sensitivity of
the expected hard descriptor under the stated threshold noise model.
"""

from __future__ import annotations

import math

import torch


def expected_hard_active(
    V: torch.Tensor,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    V_spk: float = 0.0,
    threshold_noise_scale_mV: float = 3.0,
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,
    time_window: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    r"""Probability that ``hard_active`` detects an upward crossing.

    For each fiber, independently draw one threshold offset
    :math:`\epsilon\sim\mathrm{Logistic}(0,s)` and replace the detection threshold
    by :math:`V_{spk}+\epsilon` at *every* selected compartment and sample.
    This function integrates the resulting hard active indicator exactly.
    It uses the same sampled upward-crossing rule as ``hard_active`` and the
    same half-open sample-index window.  For each rising sample pair, eligible
    thresholds form the interval :math:`(V_t,V_{t+1}]`.  The active probability
    is the logistic probability mass of the union of these intervals.

    When ``use_dv_gate`` is true, the hard dV/dt condition selects intervals.
    This selection is intentionally discrete; gradients do not anticipate an
    interval entering or leaving that gate.  The value is still the exact
    expectation of the corresponding hard measurement.

    The result is differentiable almost everywhere with respect to ``V`` and
    tends to the deterministic hard value as ``s`` decreases away from event
    boundaries.  Very small ``s`` can saturate gradients far from a boundary.

    Returns ``active`` (shape ``(F,)``), the expected binary hard descriptor.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, F, C).")
    T, fibs, compartments = V.shape
    if not V.is_floating_point():
        raise ValueError("V must have a floating-point dtype.")
    scale = float(threshold_noise_scale_mV)
    if not 0.0 < scale < float("inf"):
        raise ValueError("threshold_noise_scale_mV must be finite and positive.")
    dt = torch.as_tensor(dt_ms, device=V.device, dtype=V.dtype)
    if dt.ndim != 0 or not bool(torch.isfinite(dt)) or not bool(dt > 0):
        raise ValueError("dt_ms must be a finite positive scalar.")

    if time_window is None:
        start, end = 0, T
    else:
        start = max(0, int(time_window[0]))
        end = min(T, int(time_window[1]))
    if end - start < 2:
        return {"active": V.new_zeros(fibs)}

    if node_mask is None:
        selected = torch.ones((fibs, compartments), device=V.device, dtype=torch.bool)
    else:
        mask = node_mask.to(device=V.device)
        if mask.shape == (compartments,):
            selected = (mask != 0)[None, :].expand(fibs, compartments)
        elif mask.shape == (fibs, compartments):
            selected = mask != 0
        else:
            raise ValueError("node_mask must have shape (C,) or (F, C).")

    window = V[start:end]
    lower = window[:-1]
    upper = window[1:]
    valid = (upper > lower) & selected[None, :, :]
    if use_dv_gate:
        valid = valid & ((upper - lower) / dt >= dv_spk)

    # Sort all eligible threshold intervals by lower endpoint.  Masked pairs
    # are placed at the end; using constants makes their gradient exactly zero.
    low = torch.where(valid, lower, torch.full_like(lower, float("inf")))
    high = torch.where(valid, upper, torch.full_like(upper, float("-inf")))
    low = low.permute(1, 0, 2).reshape(fibs, -1)
    high = high.permute(1, 0, 2).reshape(fibs, -1)
    sorted_low, order = torch.sort(low, dim=1, stable=True)
    sorted_high = torch.gather(high, 1, order)

    previous_high = torch.cat(
        (
            torch.full_like(sorted_high[:, :1], float("-inf")),
            torch.cummax(sorted_high, dim=1).values[:, :-1],
        ),
        dim=1,
    )
    new_start = torch.maximum(sorted_low, previous_high)
    cdf_high = torch.sigmoid((sorted_high - V_spk) / scale)
    cdf_start = torch.sigmoid((new_start - V_spk) / scale)
    increment = torch.clamp_min(cdf_high - cdf_start, 0.0)
    probability = increment.sum(dim=1).clamp(0.0, 1.0)
    return {"active": probability}


def gaussian_smoothed_hard_measure(
    center: torch.Tensor,
    hard_values: torch.Tensor,
    perturbations: torch.Tensor,
    noise_std: float,
    *,
    weights: torch.Tensor | None = None,
    baseline: float = 0.0,
) -> dict[str, torch.Tensor]:
    r"""Attach a score-function gradient to an expected hard descriptor.

    Let ``hard_values[i]`` be the detached output of an entire hard measurement
    protocol rerun at scalar parameter ``center + perturbations[i]``.  If the
    perturbations sample :math:`N(0,\sigma^2)`, this estimates

    .. math::

       G_\sigma(q)=\mathbb{E}[H(q+\epsilon)],\qquad
       G_\sigma'(q)=\mathbb{E}[H(q+\epsilon)\epsilon]/\sigma^2.

    ``surrogate`` has the sampled expected hard value on its forward pass and
    the score-function estimate on its backward pass.  It works even if the
    simulator and hard descriptor have no differentiable path.  The derivative
    is for the *noise-averaged* metric, not the deterministic hard metric, and
    can be noisy unless samples cover the transition.  Symmetric perturbations
    and independent reruns of the complete measurement protocol are advised.

    ``weights`` can provide normalized Gaussian quadrature weights; otherwise
    each run has equal weight. ``baseline`` is a fixed, parameter-independent
    control variate; a separately measured hard value is a useful choice.
    This utility accepts a scalar ``center`` and one scalar hard value per run.
    """
    if center.ndim != 0 or not center.is_floating_point():
        raise ValueError("center must be a floating-point scalar tensor.")
    if hard_values.ndim != 1 or perturbations.ndim != 1:
        raise ValueError("hard_values and perturbations must be vectors.")
    if hard_values.numel() == 0 or hard_values.shape != perturbations.shape:
        raise ValueError(
            "hard_values and perturbations must have equal nonzero length."
        )
    sigma = float(noise_std)
    if not 0.0 < sigma < float("inf"):
        raise ValueError("noise_std must be finite and positive.")
    values = hard_values.detach().to(device=center.device, dtype=center.dtype)
    offsets = perturbations.detach().to(device=center.device, dtype=center.dtype)
    if not bool(torch.isfinite(values).all()) or not bool(
        torch.isfinite(offsets).all()
    ):
        raise ValueError("hard_values and perturbations must be finite.")
    if weights is None:
        normalized_weights = torch.full_like(values, 1.0 / values.numel())
    else:
        normalized_weights = weights.detach().to(
            device=center.device, dtype=center.dtype
        )
        if (
            normalized_weights.shape != values.shape
            or not bool(torch.isfinite(normalized_weights).all())
            or bool((normalized_weights < 0).any())
            or not bool(normalized_weights.sum() > 0)
        ):
            raise ValueError(
                "weights must be finite, nonnegative, and match hard_values."
            )
        normalized_weights = normalized_weights / normalized_weights.sum()
    if not math.isfinite(float(baseline)):
        raise ValueError("baseline must be finite.")
    mean = (normalized_weights * values).sum()
    score = (normalized_weights * (values - baseline) * offsets).sum() / sigma**2
    surrogate = mean + (center - center.detach()) * score
    return {"surrogate": surrogate, "value": mean, "gradient_estimate": score}
