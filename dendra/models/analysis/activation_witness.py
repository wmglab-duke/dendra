"""Select an upstream event-margin witness from two complete voltage traces.

This is an opt-in candidate selector for a local activation threshold gradient.
The caller supplies matched lower-event-absent and upper-event-present trials,
the monitored upper event sample, and its protocol's admissible pair/site mask.
Temporal precedence is evidence for a candidate witness, not proof that the
selected site caused the monitored spike. Revalidate the selected branch and
direction with complete hard-threshold reruns at finite parameter steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch

from .event_margin_training import event_crossing_margin


@dataclass(frozen=True)
class ActivationWitness:
    """Selected lower margin and detached diagnostics for its upstream site.

    ``upper_crossing_pair_index`` is the pair selected by the maximum-margin
    reduction. ``upper_first_crossing_pair_index`` and
    ``upper_crossing_count`` describe hard upper crossings at that same site
    among admissible pre-event pairs; the first pair need not maximize margin.
    """

    lower_margin: torch.Tensor | None
    site_index: int | None
    lower_peak_sample_index: int | None
    upper_crossing_pair_index: int | None
    candidate_count: int
    rejection_reasons: tuple[str, ...]
    upper_first_crossing_pair_index: int | None = None
    upper_crossing_count: int = 0

    @property
    def valid(self) -> bool:
        return self.lower_margin is not None and not self.rejection_reasons


def select_pre_event_activation_witness(
    lower_voltage: torch.Tensor,
    upper_voltage: torch.Tensor,
    *,
    monitored_upper_event_sample_index: int,
    valid_pairs: torch.Tensor | None = None,
    threshold_mV: float = 0.0,
    dv_threshold_mV_per_ms: float | None = None,
    dt_ms: float | None = None,
) -> ActivationWitness:
    r"""Choose the closest lower-trace crossing margin at an upstream site.

    Both traces have shape ``(time, site)`` and represent the same initialized
    model and stimulus protocol at the inactive/active ends of one hard
    amplitude bracket. ``valid_pairs`` is the protocol's boolean mask of
    allowed sample pairs and sites, including any site/readout restriction.
    Only pairs whose *upper sample* precedes
    ``monitored_upper_event_sample_index`` are considered.

    A site is eligible when the upper trace has a hard upward crossing in
    those pairs, the lower trace has none, its lower voltage stays below the
    detection threshold, and its lower peak is strictly inside the allowed
    sample window. Among eligible sites the selected lower crossing margin is
    the greatest (closest to zero). Event decisions and selection indices are
    detached; the returned lower margin retains its autograd connection.

    A caller using a stateful simulator should obtain the detached upper
    trace first, then make the gradient-bearing lower replay last. This helper
    does not search stimulus amplitudes, identify the monitored event, or
    establish that the witness controls it. A selected margin is only a
    candidate for a later local tangent or training direction.
    """
    if (
        not isinstance(lower_voltage, torch.Tensor)
        or not isinstance(upper_voltage, torch.Tensor)
        or lower_voltage.ndim != 2
        or upper_voltage.shape != lower_voltage.shape
    ):
        raise ValueError("Matched lower/upper voltage must have shape (time, site).")
    if lower_voltage.shape[0] < 3 or lower_voltage.shape[1] < 1:
        raise ValueError("Need at least three time samples and one site.")
    if (
        not lower_voltage.is_floating_point()
        or not lower_voltage.requires_grad
        or not upper_voltage.is_floating_point()
        or lower_voltage.device != upper_voltage.device
        or lower_voltage.dtype != upper_voltage.dtype
        or not bool(torch.isfinite(lower_voltage.detach()).all())
        or not bool(torch.isfinite(upper_voltage.detach()).all())
    ):
        raise ValueError(
            "Matched voltage must be finite, floating point, and lower differentiable."
        )
    if (
        isinstance(monitored_upper_event_sample_index, bool)
        or not isinstance(monitored_upper_event_sample_index, Integral)
        or not 1 < monitored_upper_event_sample_index < lower_voltage.shape[0]
    ):
        raise ValueError("monitored_upper_event_sample_index must be inside the trace.")

    time_count, site_count = lower_voltage.shape
    pair_shape = (time_count - 1, site_count)
    if valid_pairs is None:
        allowed = torch.ones(pair_shape, device=lower_voltage.device, dtype=torch.bool)
    else:
        if not isinstance(valid_pairs, torch.Tensor) or valid_pairs.dtype != torch.bool:
            raise TypeError("valid_pairs must be a boolean tensor.")
        try:
            allowed = torch.broadcast_to(
                valid_pairs.to(device=lower_voltage.device),
                pair_shape,
            )
        except RuntimeError as error:
            raise ValueError(
                "valid_pairs must broadcast to sample pairs and sites."
            ) from error
    before = (
        torch.arange(time_count - 1, device=lower_voltage.device) + 1
        < monitored_upper_event_sample_index
    )
    allowed = allowed & before[:, None]
    possible_sites = torch.where(allowed.any(dim=0))[0]
    if possible_sites.numel() == 0:
        return ActivationWitness(None, None, None, None, 0, ("no_pre_event_pairs",))

    # Treat each checked site as one event slot. Upper is detached so its
    # selection cannot silently acquire a derivative from the active branch.
    lower_selected = lower_voltage[:, possible_sites].transpose(0, 1).unsqueeze(-1)
    upper_selected = (
        upper_voltage.detach()[:, possible_sites].transpose(0, 1).unsqueeze(-1)
    )
    pair_selected = allowed[:, possible_sites].transpose(0, 1).unsqueeze(-1)
    lower = event_crossing_margin(
        lower_selected,
        threshold_mV=threshold_mV,
        valid_pairs=pair_selected,
        dv_threshold_mV_per_ms=dv_threshold_mV_per_ms,
        dt_ms=dt_ms,
    )
    upper = event_crossing_margin(
        upper_selected,
        threshold_mV=threshold_mV,
        valid_pairs=pair_selected,
        dv_threshold_mV_per_ms=dv_threshold_mV_per_ms,
        dt_ms=dt_ms,
    )

    # The lower peak must be interior to the actual admissible frame window;
    # a peak on either boundary may represent an unobserved continuation.
    selected_mask = allowed[:, possible_sites]
    frame_mask = torch.zeros(
        (time_count, possible_sites.numel()),
        device=lower_voltage.device,
        dtype=torch.bool,
    )
    frame_mask[:-1] |= selected_mask
    frame_mask[1:] |= selected_mask
    frame_first = frame_mask.to(torch.int64).argmax(dim=0)
    frame_last = (
        time_count - 1 - frame_mask.flip(dims=(0,)).to(torch.int64).argmax(dim=0)
    )
    lower_for_peak = torch.where(
        frame_mask,
        lower_voltage.detach()[:, possible_sites],
        float("-inf"),
    )
    peak_value, peak_sample = lower_for_peak.max(dim=0)
    interior = (peak_sample > frame_first) & (peak_sample < frame_last)
    candidates = (
        upper.hard_crossed
        & ~lower.hard_crossed
        & (lower.margin.detach() < 0)
        & (peak_value < threshold_mV)
        & interior
    )
    candidate_count = int(candidates.sum())
    if candidate_count == 0:
        return ActivationWitness(None, None, None, None, 0, ("no_eligible_witness",))

    rank = torch.where(
        candidates,
        lower.margin.detach(),
        float("-inf"),
    )
    chosen = int(rank.argmax())
    chosen_site = int(possible_sites[chosen])
    upper_trace = upper_voltage.detach()[:, chosen_site]
    hard_upper_crossings = (
        allowed[:, chosen_site]
        & (upper_trace[:-1] < threshold_mV)
        & (upper_trace[1:] >= threshold_mV)
    )
    if dv_threshold_mV_per_ms is not None:
        hard_upper_crossings &= (
            upper_trace[1:] - upper_trace[:-1] >= dt_ms * dv_threshold_mV_per_ms
        )
    crossing_pairs = torch.where(hard_upper_crossings)[0]
    return ActivationWitness(
        lower_margin=lower.margin[chosen],
        site_index=chosen_site,
        lower_peak_sample_index=int(peak_sample[chosen]),
        upper_crossing_pair_index=int(upper.pair_index[chosen]),
        candidate_count=candidate_count,
        rejection_reasons=(),
        upper_first_crossing_pair_index=int(crossing_pairs[0]),
        upper_crossing_count=int(crossing_pairs.numel()),
    )
