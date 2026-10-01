"""Spike timing descriptors on a fixed crossing branch.

The hard and differentiable functions use the same upward-crossing decision.
The latter differentiates interpolated crossing times through the supplied
voltage trace, while treating the selected sample pairs as discrete data.
Neither function assumes a stimulus protocol or a particular model.
"""

from __future__ import annotations

import math
from typing import Literal

import torch


def _spike_timing(
    V: torch.Tensor,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None,
    *,
    V_spk: float | torch.Tensor,
    use_dv_gate: bool,
    dv_spk: float,
    time_window: tuple[int, int] | None,
    time_window_ms: tuple[float, float] | None,
    refractory_ms: float,
    aggregate: Literal["mean", "max", "sum"],
) -> dict[str, object]:
    if V.ndim != 3 or not V.is_floating_point():
        raise ValueError("V must be a floating-point tensor of shape (T, F, C).")
    T, fibers, compartments = V.shape
    dt = torch.as_tensor(dt_ms, device=V.device, dtype=V.dtype)
    if (
        dt.numel() != 1
        or not math.isfinite(float(dt.detach()))
        or float(dt.detach()) <= 0
    ):
        raise ValueError("dt_ms must be a finite positive scalar.")
    dt_value = float(dt.detach())
    threshold = torch.as_tensor(V_spk, device=V.device, dtype=V.dtype)
    if threshold.numel() != 1 or not math.isfinite(float(threshold.detach())):
        raise ValueError("V_spk must be a finite scalar.")
    if not math.isfinite(dv_spk):
        raise ValueError("dv_spk must be finite.")
    if not math.isfinite(refractory_ms) or refractory_ms < 0:
        raise ValueError("refractory_ms must be finite and nonnegative.")
    if aggregate not in {"mean", "max", "sum"}:
        raise ValueError("aggregate must be one of {'mean', 'max', 'sum'}.")

    if time_window is not None and time_window_ms is not None:
        raise ValueError("Pass only one of time_window or time_window_ms.")
    if time_window_ms is not None:
        window_start_ms = float(time_window_ms[0])
        window_end_ms = float(time_window_ms[1])
        if (
            not math.isfinite(window_start_ms)
            or not math.isfinite(window_end_ms)
            or window_end_ms <= window_start_ms
        ):
            raise ValueError(
                "time_window_ms must contain finite values with end > start."
            )
        # Include a conservative neighboring pair at each bound, then filter
        # by the interpolated roots below. This avoids losing aligned events
        # when decimal bounds and dt divide just above an integer in binary.
        start = max(0, math.floor(window_start_ms / dt_value) - 1)
        end = min(T, math.floor(window_end_ms / dt_value) + 2)
        recorded_end_ms = (T - 1) * dt_value
        analyzed_window_ms = min(window_end_ms, recorded_end_ms) - max(
            window_start_ms, 0.0
        )
    elif time_window is not None:
        start = max(0, int(time_window[0]))
        end = min(T, int(time_window[1]))
        analyzed_window_ms = (end - start - 1) * dt_value
    else:
        start, end = 0, T
        analyzed_window_ms = (end - start - 1) * dt_value
    if end - start < 2:
        raise ValueError("Selected time window must contain at least 2 samples.")

    if node_mask is None:
        selected = torch.ones((fibers, compartments), device=V.device, dtype=torch.bool)
    else:
        mask = node_mask.to(device=V.device)
        if mask.shape == (compartments,):
            selected = (mask != 0)[None, :].expand(fibers, compartments)
        elif mask.shape == (fibers, compartments):
            selected = mask != 0
        else:
            raise ValueError("node_mask must have shape (C,) or (F, C).")

    # Pair selection is intentionally outside autograd. The roots below still
    # use V itself, so a scalar timing objective needs just one simulation VJP.
    detached = V.detach()
    crossings = (detached[start : end - 1] < threshold.detach()) & (
        detached[start + 1 : end] >= threshold.detach()
    )
    if use_dv_gate:
        slopes = (detached[start + 1 : end] - detached[start : end - 1]) / dt_value
        crossings = crossings & (slopes >= dv_spk)
    refractory_steps = (
        max(1, int(round(refractory_ms / dt_value))) if refractory_ms > 0 else 0
    )

    count_comp = torch.zeros((fibers, compartments), device=V.device, dtype=torch.long)
    indices_by_site: list[list[torch.Tensor]] = []
    signature_rows: list[tuple[tuple[int, ...] | None, ...]] = []
    count_rows: list[tuple[int | None, ...]] = []
    for f in range(fibers):
        site_indices: list[torch.Tensor] = []
        signature_row: list[tuple[int, ...] | None] = []
        count_row: list[int | None] = []
        for c in range(compartments):
            indices = torch.where(crossings[:, f, c])[0] + start
            if time_window_ms is not None and indices.numel():
                before = detached[indices, f, c]
                after = detached[indices + 1, f, c]
                roots = (
                    indices.to(dtype=V.dtype)
                    + (threshold.detach() - before) / (after - before)
                ) * dt.detach()
                in_window = (roots >= window_start_ms) & (roots <= window_end_ms)
                indices = indices[in_window]
            if refractory_steps:
                retained: list[int] = []
                last = -(10**12)
                for index in indices.tolist():
                    if index - last >= refractory_steps:
                        retained.append(index)
                        last = index
                indices = torch.tensor(retained, device=V.device, dtype=torch.long)
            count_comp[f, c] = indices.numel()
            site_indices.append(indices)
            is_selected = bool(selected[f, c])
            signature_row.append(tuple(indices.tolist()) if is_selected else None)
            count_row.append(indices.numel() if is_selected else None)
        indices_by_site.append(site_indices)
        signature_rows.append(tuple(signature_row))
        count_rows.append(tuple(count_row))

    # NaN padding preserves the event order and makes undefined sites explicit.
    max_events = max(1, int(count_comp.max()) if count_comp.numel() else 0)
    event_times = V.new_full((max_events, fibers, compartments), float("nan"))
    for f, site_indices in enumerate(indices_by_site):
        for c, indices in enumerate(site_indices):
            n = indices.numel()
            if n:
                before = V[indices, f, c]
                after = V[indices + 1, f, c]
                roots = (
                    indices.to(dtype=V.dtype) + (threshold - before) / (after - before)
                ) * dt
                event_times[:n, f, c] = roots

    # Keep every interval as well as its span average. The span average
    # telescopes and cannot detect redistribution of interior event times.
    interspike_intervals = V.new_full(
        (max(1, max_events - 1), fibers, compartments), float("nan")
    )
    for f in range(fibers):
        for c in range(compartments):
            n = int(count_comp[f, c])
            if n >= 2:
                interspike_intervals[: n - 1, f, c] = (
                    event_times[1:n, f, c] - event_times[: n - 1, f, c]
                )

    valid_comp = count_comp >= 2
    selected_valid_comp = valid_comp & selected
    valid = selected_valid_comp.any(dim=-1)
    first = torch.nan_to_num(event_times[0], nan=0.0)
    last_index = (count_comp - 1).clamp_min(0)[None]
    last = torch.nan_to_num(event_times.gather(0, last_index).squeeze(0), nan=0.0)
    # The safe span avoids a zero or NaN division in undefined sites.
    span = torch.where(valid_comp, last - first, torch.ones_like(last))
    safe_intervals = (count_comp - 1).clamp_min(1).to(V.dtype)
    mean_isi_comp = torch.where(
        valid_comp, span / safe_intervals, torch.full_like(span, float("nan"))
    )
    frequency_comp = torch.where(
        valid_comp,
        1000.0 * safe_intervals / span,
        torch.full_like(span, float("nan")),
    )
    defined_frequency = torch.where(
        selected_valid_comp, frequency_comp, torch.zeros_like(frequency_comp)
    )
    if aggregate == "mean":
        frequency = defined_frequency.sum(dim=-1) / selected_valid_comp.sum(
            dim=-1
        ).clamp_min(1)
    elif aggregate == "sum":
        frequency = defined_frequency.sum(dim=-1)
    else:
        frequency = (
            torch.where(
                selected_valid_comp,
                frequency_comp,
                torch.full_like(frequency_comp, float("-inf")),
            )
            .max(dim=-1)
            .values
        )
    frequency = torch.where(valid, frequency, torch.full_like(frequency, float("nan")))

    return {
        "event_times_ms": event_times,
        "interspike_intervals_ms": interspike_intervals,
        "count_comp": count_comp,
        "mean_isi_ms_comp": mean_isi_comp,
        "span_frequency_hz_comp": frequency_comp,
        "span_frequency_hz": frequency,
        "valid_comp": valid_comp,
        "selected_valid_comp": selected_valid_comp,
        "valid": valid,
        "weights": selected.to(dtype=V.dtype),
        "window_ms": dt.new_tensor(analyzed_window_ms),
        "branch_signature": tuple(signature_rows),
        "count_signature": tuple(count_rows),
    }


def branch_conditioned_spike_timing(
    V: torch.Tensor,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    V_spk: float | torch.Tensor = 0.0,
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,
    time_window: tuple[int, int] | None = None,
    time_window_ms: tuple[float, float] | None = None,
    refractory_ms: float = 0.0,
    aggregate: Literal["mean", "max", "sum"] = "mean",
) -> dict[str, object]:
    """Return differentiable spike times and a span-based frequency.

    The event times are linearly interpolated roots of sample pairs satisfying
    ``V[i] < V_spk <= V[i+1]``. Pair indices are selected from detached voltage;
    event times use graph-connected voltage. For N >= 2 events at a site,
    ``mean_isi_ms_comp = (t_last - t_first)/(N-1)`` and
    ``span_frequency_hz_comp = 1000/mean_isi_ms_comp``. These values are NaN
    and ``valid_comp`` is false for N < 2. The per-fiber frequency aggregates
    only selected, defined sites; it is NaN when no site is defined.
    ``interspike_intervals_ms`` retains the ordered adjacent intervals, with
    NaN padding. A caller can form a protocol-defined loss over these
    intervals when regularity or local spacing matters: the mean span
    frequency sees only the first and last event times when count is fixed.
    ``time_window_ms`` includes events whose interpolated crossing times lie
    within the inclusive interval; its bounds need not align with samples.

    ``branch_signature`` contains retained global crossing-pair indices at
    selected sites (None for unselected sites). Compare it between evaluations
    when checking a local gradient. The derivative describes only the selected
    crossing branch; event births, losses, gate changes, and refractory changes
    are discrete and have no derivative here. There are no pulse bins.
    """
    return _spike_timing(
        V,
        dt_ms,
        node_mask,
        V_spk=V_spk,
        use_dv_gate=use_dv_gate,
        dv_spk=dv_spk,
        time_window=time_window,
        time_window_ms=time_window_ms,
        refractory_ms=refractory_ms,
        aggregate=aggregate,
    )


@torch.no_grad()
def hard_spike_timing(
    V: torch.Tensor,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    V_spk: float | torch.Tensor = 0.0,
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,
    time_window: tuple[int, int] | None = None,
    time_window_ms: tuple[float, float] | None = None,
    refractory_ms: float = 0.0,
    aggregate: Literal["mean", "max", "sum"] = "mean",
) -> dict[str, object]:
    """Hard reference for :func:`branch_conditioned_spike_timing`."""
    return _spike_timing(
        V,
        dt_ms,
        node_mask,
        V_spk=V_spk,
        use_dv_gate=use_dv_gate,
        dv_spk=dv_spk,
        time_window=time_window,
        time_window_ms=time_window_ms,
        refractory_ms=refractory_ms,
        aggregate=aggregate,
    )
