"""Opt-in training directions for discontinuous spike descriptors.

The complete hard protocol remains the source of the descriptor and its task
loss.  A differentiable voltage replay supplies a *training direction* for
the currently mismatched event slots.  This is not a derivative of the hard
descriptor, which is locally constant almost everywhere and discontinuous at
spike transitions.

This module owns neither a stimulus protocol nor event-to-pulse assignment.
The caller must use the same checked sites, sample pairs, voltage threshold,
upstroke gate, and response windows as its hard protocol.  It must assign each
hard event to at most one slot, handle refractory exclusions, and create extra
slots if more than one spike may occur in a response window.  Those discrete
choices are recomputed after every parameter update and are not differentiated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EventCrossingMargin:
    """One signed crossing margin per event slot.

    ``margin`` has the leading shape of ``voltage`` before its time/site axes.
    Its sign matches ``hard_crossed`` except when a sampled voltage is exactly
    equal to the threshold at the *previous* sample: the hard rule uses a
    strict ``v_previous < threshold`` test, whereas a continuous margin has a
    zero at that boundary.  ``pair_index`` and ``site_index`` identify the
    currently selected branch within each caller-supplied slot.
    """

    margin: torch.Tensor
    hard_crossed: torch.Tensor
    pair_index: torch.Tensor
    site_index: torch.Tensor


@dataclass(frozen=True)
class EventTrainingLoss:
    """Exact hard task loss in forward, chosen trace objective in backward."""

    loss: torch.Tensor
    trace_objective: torch.Tensor
    mismatched_events: torch.Tensor
    local_sensitivity_fraction: torch.Tensor


def event_crossing_margin(
    voltage: torch.Tensor,
    *,
    threshold_mV: float = 0.0,
    valid_pairs: torch.Tensor | None = None,
    dv_threshold_mV_per_ms: float | None = None,
    dt_ms: float | None = None,
) -> EventCrossingMargin:
    r"""Score one upward crossing in each caller-defined event slot.

    ``voltage`` has shape ``(*slots, time, site)`` with at least two samples.
    ``valid_pairs`` must broadcast to ``(*slots, time-1, site)``; use it to
    enforce the hard protocol's site mask, pulse window, and ownership rule.
    At each valid sample pair and site, the voltage margin is

    ``m_tc = min(threshold-v_t, v_(t+1)-threshold)``.

    When an upstroke gate is requested, it also includes
    ``v_(t+1)-v_t-dt*dv_threshold`` in the minimum.  The event margin is the
    maximum over valid pairs and sites.  The corresponding hard crossing uses
    ``v_t < threshold <= v_(t+1)`` and the same upstroke gate.  Max/min branches
    may switch as parameters change; autograd follows the branch selected in
    the current replay.  This statistic detects one-or-more crossings, not a
    refractory-aware count or pulse attribution.
    """
    if not isinstance(voltage, torch.Tensor) or voltage.ndim < 2:
        raise ValueError("voltage must be a tensor with time and site axes.")
    if not voltage.is_floating_point() or not bool(
        torch.isfinite(voltage.detach()).all()
    ):
        raise ValueError("voltage must be finite and floating point.")
    if voltage.shape[-2] < 2 or voltage.shape[-1] < 1:
        raise ValueError("Each event slot needs at least two samples and one site.")
    if not math.isfinite(threshold_mV):
        raise ValueError("threshold_mV must be finite.")
    if dv_threshold_mV_per_ms is not None:
        if not math.isfinite(dv_threshold_mV_per_ms):
            raise ValueError("dv_threshold_mV_per_ms must be finite.")
        if dt_ms is None or not math.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("An upstroke gate requires finite positive dt_ms.")

    v0, v1 = voltage[..., :-1, :], voltage[..., 1:, :]
    if valid_pairs is None:
        valid = torch.ones_like(v0, dtype=torch.bool)
    else:
        if not isinstance(valid_pairs, torch.Tensor) or valid_pairs.dtype != torch.bool:
            raise TypeError("valid_pairs must be a boolean tensor.")
        try:
            valid = torch.broadcast_to(valid_pairs.to(device=voltage.device), v0.shape)
        except RuntimeError as error:
            raise ValueError("valid_pairs must broadcast to sample pairs.") from error
    if not bool(valid.any(dim=(-2, -1)).all()):
        raise ValueError("Every event slot must have a valid sample pair and site.")

    margin_by_pair = torch.minimum(threshold_mV - v0, v1 - threshold_mV)
    hard_by_pair = (v0 < threshold_mV) & (v1 >= threshold_mV)
    if dv_threshold_mV_per_ms is not None:
        upstroke_margin = v1 - v0 - dt_ms * dv_threshold_mV_per_ms
        margin_by_pair = torch.minimum(margin_by_pair, upstroke_margin)
        hard_by_pair = hard_by_pair & (upstroke_margin >= 0)
    margin_by_pair = torch.where(valid, margin_by_pair, float("-inf"))
    hard_by_pair = hard_by_pair & valid

    site_count = voltage.shape[-1]
    flat = margin_by_pair.flatten(start_dim=-2)
    margin, flat_index = flat.max(dim=-1)
    return EventCrossingMargin(
        margin=margin,
        hard_crossed=hard_by_pair.any(dim=(-2, -1)),
        pair_index=torch.div(flat_index, site_count, rounding_mode="floor"),
        site_index=flat_index.remainder(site_count),
    )


def hard_forward_event_training_loss(
    hard_task_loss: float | torch.Tensor,
    event_margins: torch.Tensor,
    hard_events: torch.Tensor,
    target_events: torch.Tensor,
    *,
    event_weights: torch.Tensor | None = None,
    temperature_mV: float = 1.0,
    margin_target_mV: float = 0.0,
    backward_scale: float = 1.0,
    require_inactive_carrier: bool = True,
    min_local_sensitivity_fraction: float = 0.01,
) -> EventTrainingLoss:
    r"""Use trace margins to train toward targets while reporting the hard loss.

    ``hard_task_loss`` is computed by the *complete* hard measurement protocol.
    ``hard_events`` and ``target_events`` are boolean event-slot labels after
    the caller's hard, one-to-one event assignment.  ``event_margins`` comes
    from a differentiable trace at the same stimulus or a validated nearby
    probe.  By default each mismatched slot must have a *negative* carrier
    margin.  A missing event can use its current inactive trace; an unwanted
    event should use a nearby stimulus probe where that event is absent.
    This avoids differentiating the sampled crossing pair of a full spike,
    whose gradient may merely move the crossing to an adjacent sample.
    ``require_inactive_carrier=False`` permits exploratory alternative
    statistics, but requires independent hard-step validation.

    An inactive carrier far below threshold can silently saturate the
    suppression penalty. ``min_local_sensitivity_fraction`` rejects a
    mismatched slot when the magnitude of its local softplus derivative,
    relative to the maximum ``1/temperature_mV``, falls below the configured
    fraction. The default 0.01 therefore rejects effectively zero directions;
    setting zero explicitly disables this local sensitivity gate. Passing it
    still does not certify a useful parameter VJP or hard-protocol update.

    For mismatched slots ``j``, let ``y_j=+1`` if an event is wanted and ``-1``
    if it must be removed.  The dimensionless trace objective is the weighted
    mean of

    ``softplus((margin_target_mV - y_j*m_j)/temperature_mV)``.

    The returned scalar is

    ``L = stopgrad(L_hard) + backward_scale * (R - stopgrad(R))``.

    Thus its forward value is exactly the complete hard task loss, and one
    simulator VJP supplies ``backward_scale * grad(R)`` for *all* connected
    model parameters when the margins share one replay. If several probes are
    needed, accumulate their weighted VJPs sequentially; a stateful simulator
    may invalidate an earlier graph when a later probe runs. ``-grad(R)`` is a
    proposed descent direction, not the
    derivative of ``L_hard`` and not a guarantee of hard improvement.  When all
    event labels match, the trace gradient is zero.  Recompute hard labels,
    windows, and probes after each update; validate directions with complete
    hard-protocol reruns at finite parameter steps.
    """
    if not isinstance(event_margins, torch.Tensor):
        raise TypeError("event_margins must be a tensor.")
    if (
        not event_margins.is_floating_point()
        or not event_margins.requires_grad
        or not bool(torch.isfinite(event_margins.detach()).all())
    ):
        raise ValueError("event_margins must be finite and connected to autograd.")
    if not isinstance(hard_events, torch.Tensor) or not isinstance(
        target_events, torch.Tensor
    ):
        raise TypeError("hard_events and target_events must be tensors.")
    if hard_events.dtype != torch.bool or target_events.dtype != torch.bool:
        raise TypeError("hard_events and target_events must be boolean.")
    if (
        hard_events.shape != event_margins.shape
        or target_events.shape != event_margins.shape
    ):
        raise ValueError("Event margins and labels must have identical shapes.")
    if not math.isfinite(temperature_mV) or temperature_mV <= 0:
        raise ValueError("temperature_mV must be finite and positive.")
    if not math.isfinite(margin_target_mV) or margin_target_mV < 0:
        raise ValueError("margin_target_mV must be finite and nonnegative.")
    if not math.isfinite(backward_scale) or backward_scale < 0:
        raise ValueError("backward_scale must be finite and nonnegative.")
    if (
        not math.isfinite(min_local_sensitivity_fraction)
        or not 0 <= min_local_sensitivity_fraction <= 1
    ):
        raise ValueError("min_local_sensitivity_fraction must lie in [0, 1].")

    # Keep a tensor hard loss in its measurement precision. In particular, a
    # float64 hard protocol value must not be rounded to a float32 replay's
    # margin dtype merely to attach the replay's training direction.
    hard_dtype = (
        hard_task_loss.dtype
        if isinstance(hard_task_loss, torch.Tensor)
        and hard_task_loss.is_floating_point()
        else event_margins.dtype
    )
    hard = torch.as_tensor(
        hard_task_loss, dtype=hard_dtype, device=event_margins.device
    )
    if hard.numel() != 1 or not bool(torch.isfinite(hard.detach()).all()):
        raise ValueError("hard_task_loss must be a finite scalar.")
    if event_weights is None:
        weights = torch.ones_like(event_margins)
    else:
        if not isinstance(event_weights, torch.Tensor):
            raise TypeError("event_weights must be a tensor.")
        try:
            weights = torch.broadcast_to(
                event_weights.detach().to(event_margins),
                event_margins.shape,
            )
        except RuntimeError as error:
            raise ValueError(
                "event_weights must broadcast to event margins."
            ) from error
        if not bool((torch.isfinite(weights) & (weights >= 0)).all()):
            raise ValueError("event_weights must be finite and nonnegative.")

    hard_events = hard_events.to(device=event_margins.device)
    target_events = target_events.to(device=event_margins.device)
    mismatched = hard_events != target_events
    selected_weights = weights * mismatched.to(event_margins.dtype)
    if require_inactive_carrier and bool(
        ((selected_weights > 0) & (event_margins.detach() >= 0)).any()
    ):
        raise ValueError(
            "Mismatched slots need negative inactive-carrier margins; "
            "select an absent-event probe or explicitly disable this guard."
        )
    desired_sign = torch.where(
        target_events,
        torch.ones_like(event_margins),
        -torch.ones_like(event_margins),
    )
    selected = selected_weights > 0
    logits = (margin_target_mV - desired_sign * event_margins) / temperature_mV
    if not bool(torch.isfinite(logits.detach()[selected]).all()):
        raise ValueError("Selected event-margin penalty logits must be finite.")
    local_sensitivity = torch.sigmoid(logits.detach())
    saturated = selected & (local_sensitivity < min_local_sensitivity_fraction)
    if bool(saturated.any()):
        smallest = float(local_sensitivity[saturated].min())
        raise ValueError(
            "inactive_carrier_saturated: local sensitivity fraction "
            f"{smallest:.3g} is below {min_local_sensitivity_fraction:.3g}; "
            "choose a closer carrier or another validated trace statistic."
        )
    # Do not evaluate a numerically extreme penalty for an unselected slot:
    # zero weight times infinity would otherwise contaminate the reduction.
    penalties = F.softplus(torch.where(selected, logits, torch.zeros_like(logits)))
    objective = (selected_weights * penalties).sum() / selected_weights.sum().clamp_min(
        torch.finfo(event_margins.dtype).tiny
    )
    trace_direction = backward_scale * (objective - objective.detach())
    loss = hard.detach().reshape(()) + trace_direction.to(dtype=hard.dtype)
    return EventTrainingLoss(loss, objective, mismatched, local_sensitivity)
