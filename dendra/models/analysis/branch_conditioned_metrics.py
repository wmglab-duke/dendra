"""Hard-forward, branch-conditioned derivatives for spike timing descriptors.

These estimators make exactly the same discrete event choices as the hard
AP-width and ADS descriptors. Within a selected branch, Torch tracks the
linearly interpolated crossing times and downstream arithmetic. Signature
changes need interpretation: a crossing's interpolation pair can advance by
one sample while following the *same* physical event continuously. Changing
the selected spike, validity or window-boundary state, or the historical
sampled baseline window can instead introduce a genuine derivative
discontinuity.

An optimizer may cross to another branch between steps. Recompute this metric
after every parameter update so the next derivative uses the newly selected
events. A signature mismatch makes a secant unsuitable as an unqualified
same-branch derivative check; it does not by itself prove the descriptor or
its derivative is discontinuous.

The hard value is attached to the selected-branch derivative by
``hard + selected - selected.detach()``. This keeps the published reference
value on the forward pass, while exposing its ordinary piecewise derivative.
This is deliberately not a smooth proxy for absent spikes or event changes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from .descriptors import (
    _as_F_vector,
    _coerce_lengths_um,
    _coerce_pulse_times_ms,
    _gather_time_windows_FNKC,
    _piecewise_linear_window_mean,
    _resolve_ads_stabilizers,
    hard_action_potential_width,
    hard_activity_dependent_slowing,
)


def compare_branch_signatures(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compare signatures from two reruns, returning masks for each choice.

    The ``all_stable`` mask has the common signature shape. A false entry means
    at least one recorded integer choice changed; it does *not* establish that
    the hard descriptor jumped. Adjacent arrival/rise/fall pair changes can
    represent the same linearly interpolated crossing passing a sample time.
    Compare event times and neighboring traces before treating such a change
    as a different spike. A changed sampled baseline index can alter the mean
    discontinuously; continuous-time baseline mode records clipping/emptiness
    instead of those sample indices.
    """
    if left.keys() != right.keys():
        raise ValueError("Branch signatures must have identical keys.")
    same: dict[str, torch.Tensor] = {}
    for key in left:
        if left[key].shape != right[key].shape:
            raise ValueError(f"Branch signature shape differs for {key!r}.")
        same[key] = left[key] == right[key]
    same["all_stable"] = torch.stack(list(same.values()), dim=0).all(dim=0)
    return same


def _attach(hard: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Use the hard forward value and selected branch's Torch derivative."""
    return hard + (selected - selected.detach())


def _crossing_time(
    trace: torch.Tensor,
    level: torch.Tensor,
    index: int,
    dt: torch.Tensor,
    *,
    falling: bool,
) -> torch.Tensor:
    if falling:
        fraction = (
            (trace[index] - level) / (trace[index] - trace[index + 1] + 1e-12)
        ).clamp(0, 1)
    else:
        fraction = (
            (level - trace[index]) / (trace[index + 1] - trace[index] + 1e-12)
        ).clamp(0, 1)
    return (index + fraction) * dt


def _width_branch(
    trace: torch.Tensor,
    level: torch.Tensor,
    peak: int,
    start: int,
    end: int,
    dt: torch.Tensor,
) -> tuple[torch.Tensor | None, tuple[int, int, int, int]]:
    """Select hard rise/fall crossing pairs; codes 0=crossing, 1=censor."""
    T = trace.numel()
    start, end = max(0, start), min(T, end)
    if end - start < 2:
        return None, (-1, -1, -1, -1)
    peak = max(start, min(peak, end - 1))
    v = trace.detach()
    lev = level.detach()
    rise = -1
    for i in range(peak - 1, start - 1, -1):
        if bool((v[i] < lev).item()) and bool((v[i + 1] >= lev).item()):
            rise = i
            break
    if rise >= 0:
        t_up = _crossing_time(trace, level, rise, dt, falling=False)
        rise_kind = 0
    elif bool((v[start] >= lev).item()) and bool((v[peak] >= lev).item()):
        t_up = start * dt
        rise_kind = 1
        rise = start
    else:
        return None, (-1, -1, -1, -1)

    fall = -1
    for i in range(peak, end - 1):
        if bool((v[i] >= lev).item()) and bool((v[i + 1] < lev).item()):
            fall = i
            break
    if fall >= 0:
        t_down = _crossing_time(trace, level, fall, dt, falling=True)
        fall_kind = 0
    elif bool((v[end - 1] >= lev).item()) and bool((v[peak] >= lev).item()):
        t_down = (end - 1) * dt
        fall_kind = 1
        fall = end - 1
    else:
        return None, (rise, rise_kind, -1, -1)
    return t_down - t_up, (rise, rise_kind, fall, fall_kind)


def branch_conditioned_action_potential_width(
    V: torch.Tensor,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    V_spk: float = 0.0,
    dv_th: float | None = None,
    mode: Literal["half_height", "half_peak_to_peak"] = "half_height",
    baseline_pre_ms: float = 1.0,
    baseline_guard_ms: float = 0.15,
    baseline_mean_mode: Literal["sampled", "continuous_time"] = "sampled",
    peak_pre_ms: float = 0.2,
    peak_post_ms: float = 1.5,
    trough_pre_ms: float = 0.3,
    trough_post_ms: float = 3.0,
    width_pre_ms: float = 1.0,
    width_post_ms: float = 3.0,
    half_level: float = 0.5,
    baseline_margin_mV: float = 1.0,
    full_width_pre_ms: float | None = None,
    full_width_post_ms: float = 8.0,
    amp_min_mV: float = 5.0,
) -> dict[str, Any]:
    """AP widths with exact hard forward values and stable-branch derivatives.

    The branch signature records the arrival crossing, baseline and peak
    windows, selected peak/trough, and rising/falling crossing pairs. ``-1``
    marks an absent choice. A fall or rise ``kind`` of one marks a boundary
    substitution because the required crossing was not observed inside the
    measurement window. The caller selects the recording duration and search
    windows to contain the required events for the recording sites and
    stimulation protocol. Extending the recording alone does not extend
    ``width_post_ms`` or ``full_width_post_ms``; their return searches must
    also contain the falling crossings. In
    ``baseline_mean_mode="continuous_time"``, ``baseline_start/end`` are ``-2``
    because the baseline has no discrete sample-window branch; separate
    ``baseline_start_clipped``, ``baseline_end_clipped``, and ``baseline_empty``
    flags retain the genuine boundary/validity states. ``arrival_pair`` and
    rise/fall pairs identify interpolation *segments*, not persistent spike
    identities: a one-sample pair change may be benign when crossing time is
    continuous. Peak-window bounds can change without changing the selected
    peak, but can also admit a competing peak. Validity, boundary substitution,
    and selected-peak changes require closer inspection of the actual voltage
    event before declaring a stable derivative.
    """
    kwargs = dict(
        V_spk=V_spk,
        dv_th=dv_th,
        mode=mode,
        baseline_pre_ms=baseline_pre_ms,
        baseline_guard_ms=baseline_guard_ms,
        baseline_mean_mode=baseline_mean_mode,
        peak_pre_ms=peak_pre_ms,
        peak_post_ms=peak_post_ms,
        trough_pre_ms=trough_pre_ms,
        trough_post_ms=trough_post_ms,
        width_pre_ms=width_pre_ms,
        width_post_ms=width_post_ms,
        half_level=half_level,
        baseline_margin_mV=baseline_margin_mV,
        full_width_pre_ms=full_width_pre_ms,
        full_width_post_ms=full_width_post_ms,
        amp_min_mV=amp_min_mV,
    )
    hard = hard_action_potential_width(V, dt_ms, node_mask, **kwargs)
    T, F, C = V.shape
    dt = torch.as_tensor(dt_ms, device=V.device, dtype=V.dtype)
    dt_float = float(dt.detach().item())
    if full_width_pre_ms is None:
        full_width_pre_ms = width_pre_ms

    names = (
        "arrival_pair",
        "baseline_start",
        "baseline_end",
        "baseline_mean_mode",
        "baseline_start_clipped",
        "baseline_end_clipped",
        "baseline_empty",
        "peak_window_start",
        "peak_window_end",
        "peak_sample",
        "trough_sample",
        "half_rise_pair",
        "half_rise_kind",
        "half_fall_pair",
        "half_fall_kind",
        "full_rise_pair",
        "full_rise_kind",
        "full_fall_pair",
        "full_fall_kind",
        "half_valid",
        "full_valid",
        "half_used",
        "full_used",
    )
    sig = {
        name: torch.full((F, C), -1, device=V.device, dtype=torch.long)
        for name in names
    }
    sig["baseline_mean_mode"].fill_(1 if baseline_mean_mode == "continuous_time" else 0)
    half_rows: list[torch.Tensor] = []
    full_rows: list[torch.Tensor] = []
    arrival_rows: list[torch.Tensor] = []
    base_rows: list[torch.Tensor] = []
    low_rows: list[torch.Tensor] = []
    peak_rows: list[torch.Tensor] = []
    level_rows: list[torch.Tensor] = []
    amp_rows: list[torch.Tensor] = []
    zero = V.new_zeros(())

    with torch.no_grad():
        arrival_candidates = (V[:-1] < V_spk) & (V[1:] >= V_spk)
        if dv_th is not None:
            arrival_candidates &= (V[1:] - V[:-1]) / dt >= dv_th
        arrival_idx = arrival_candidates.to(torch.int64).argmax(dim=0)

    for f in range(F):
        h_row: list[torch.Tensor] = []
        fw_row: list[torch.Tensor] = []
        ar_row: list[torch.Tensor] = []
        b_row: list[torch.Tensor] = []
        l_row: list[torch.Tensor] = []
        p_row: list[torch.Tensor] = []
        vh_row: list[torch.Tensor] = []
        a_row: list[torch.Tensor] = []
        for c in range(C):
            arrival_val = zero
            width_val = full_val = base_val = low_val = peak_val = level_val = (
                amp_val
            ) = zero
            if bool(hard["has_crossing"][f, c].item()):
                sig["arrival_pair"][f, c] = arrival_idx[f, c]
                trace = V[:, f, c]
                arrival_val = _crossing_time(
                    trace,
                    trace.new_tensor(V_spk),
                    int(arrival_idx[f, c].item()),
                    dt,
                    falling=False,
                )
                tc = float(hard["t_cross_ms"][f, c].item())
                b0 = max(0, math.floor((tc - baseline_pre_ms) / dt_float))
                b1 = max(0, math.ceil((tc - baseline_guard_ms) / dt_float))
                sig["baseline_start_clipped"][f, c] = int(tc - baseline_pre_ms < 0)
                sig["baseline_end_clipped"][f, c] = int(
                    tc - baseline_guard_ms > (T - 1) * dt_float
                )
                if baseline_mean_mode == "sampled":
                    sig["baseline_start"][f, c] = b0
                    sig["baseline_end"][f, c] = b1
                    base_candidate = trace[b0:b1].mean() if b1 > b0 else None
                else:
                    sig["baseline_start"][f, c] = -2
                    sig["baseline_end"][f, c] = -2
                    base_candidate = _piecewise_linear_window_mean(
                        trace,
                        dt,
                        arrival_val - baseline_pre_ms,
                        arrival_val - baseline_guard_ms,
                    )
                sig["baseline_empty"][f, c] = int(base_candidate is None)
                if base_candidate is not None:
                    base_val = base_candidate
                    p0 = max(0, math.floor((tc - peak_pre_ms) / dt_float))
                    p1 = min(T, math.ceil((tc + peak_post_ms) / dt_float) + 1)
                    sig["peak_window_start"][f, c] = p0
                    sig["peak_window_end"][f, c] = p1
                    if p1 > p0:
                        pk = p0 + int(trace[p0:p1].detach().argmax().item())
                        sig["peak_sample"][f, c] = pk
                        peak_val = trace[pk]
                        if mode == "half_height":
                            low_val = base_val
                        elif mode == "half_peak_to_peak":
                            tp = float(hard["t_peak_ms"][f, c].item())
                            tr0 = max(0, math.floor((tp - trough_pre_ms) / dt_float))
                            tr1 = min(
                                T, math.ceil((tp + trough_post_ms) / dt_float) + 1
                            )
                            if tr1 > tr0:
                                tr = tr0 + int(trace[tr0:tr1].detach().argmin().item())
                                sig["trough_sample"][f, c] = tr
                                low_val = trace[tr]
                            else:
                                low_val = base_val
                        else:
                            raise ValueError(
                                "mode must be 'half_height' or 'half_peak_to_peak'."
                            )
                        amp_val = peak_val - low_val
                        if float(amp_val.detach().item()) >= amp_min_mV:
                            level_val = low_val + half_level * amp_val
                            tp = float(hard["t_peak_ms"][f, c].item())
                            w0 = max(0, math.floor((tp - width_pre_ms) / dt_float))
                            w1 = min(T, math.ceil((tp + width_post_ms) / dt_float) + 1)
                            width_candidate, branch = _width_branch(
                                trace, level_val, pk, w0, w1, dt
                            )
                            for name, value in zip(
                                (
                                    "half_rise_pair",
                                    "half_rise_kind",
                                    "half_fall_pair",
                                    "half_fall_kind",
                                ),
                                branch,
                            ):
                                sig[name][f, c] = value
                            if width_candidate is not None:
                                width_val = width_candidate
                            fw0 = max(
                                0, math.floor((tp - full_width_pre_ms) / dt_float)
                            )
                            fw1 = min(
                                T, math.ceil((tp + full_width_post_ms) / dt_float) + 1
                            )
                            full_level = base_val + baseline_margin_mV
                            full_candidate, branch = _width_branch(
                                trace, full_level, pk, fw0, fw1, dt
                            )
                            for name, value in zip(
                                (
                                    "full_rise_pair",
                                    "full_rise_kind",
                                    "full_fall_pair",
                                    "full_fall_kind",
                                ),
                                branch,
                            ):
                                sig[name][f, c] = value
                            if full_candidate is not None:
                                full_val = full_candidate
            h_row.append(width_val)
            fw_row.append(full_val)
            ar_row.append(arrival_val)
            b_row.append(base_val)
            l_row.append(low_val)
            p_row.append(peak_val)
            vh_row.append(level_val)
            a_row.append(amp_val)
        half_rows.append(torch.stack(h_row))
        full_rows.append(torch.stack(fw_row))
        arrival_rows.append(torch.stack(ar_row))
        base_rows.append(torch.stack(b_row))
        low_rows.append(torch.stack(l_row))
        peak_rows.append(torch.stack(p_row))
        level_rows.append(torch.stack(vh_row))
        amp_rows.append(torch.stack(a_row))

    half_sel = torch.stack(half_rows)
    full_sel = torch.stack(full_rows)
    arrival_sel = torch.stack(arrival_rows)
    b_sel = torch.stack(base_rows)
    l_sel = torch.stack(low_rows)
    p_sel = torch.stack(peak_rows)
    vh_sel = torch.stack(level_rows)
    a_sel = torch.stack(amp_rows)
    half_valid = torch.isfinite(hard["width_ms_comp"])
    full_valid = torch.isfinite(hard["full_width_ms_comp"])
    mask = hard["weights"]
    sig["half_valid"] = half_valid.long()
    sig["full_valid"] = full_valid.long()
    sig["half_used"] = (half_valid & mask).long()
    sig["full_used"] = (full_valid & mask).long()

    out: dict[str, Any] = dict(hard)
    for key, selected in (
        ("t_cross_ms", arrival_sel),
        ("width_ms_comp", half_sel),
        ("full_width_ms_comp", full_sel),
        ("V_base_mV", b_sel),
        ("V_low_mV", l_sel),
        ("V_peak_mV", p_sel),
        ("V_half_mV", vh_sel),
        ("amp_mV", a_sel),
    ):
        out[key] = _attach(hard[key], selected)
    out["width_ms"] = _attach(
        hard["width_ms"],
        torch.stack(
            [
                half_sel[f, sig["half_used"][f].bool()].mean()
                if bool(sig["half_used"][f].any().item())
                else zero
                for f in range(F)
            ]
        ),
    )
    out["full_width_ms"] = _attach(
        hard["full_width_ms"],
        torch.stack(
            [
                full_sel[f, sig["full_used"][f].bool()].mean()
                if bool(sig["full_used"][f].any().item())
                else zero
                for f in range(F)
            ]
        ),
    )
    out["branch_signature"] = sig
    tolerance = max(1e-10, 32 * torch.finfo(V.dtype).eps)
    half_selected = (sig["half_rise_pair"] >= 0) & (sig["half_fall_pair"] >= 0)
    full_selected = (sig["full_rise_pair"] >= 0) & (sig["full_fall_pair"] >= 0)
    half_matches = (half_valid == half_selected) & (
        ~half_valid
        | torch.isclose(
            half_sel.detach(), hard["width_ms_comp"], rtol=tolerance, atol=tolerance
        )
    )
    full_matches = (full_valid == full_selected) & (
        ~full_valid
        | torch.isclose(
            full_sel.detach(),
            hard["full_width_ms_comp"],
            rtol=tolerance,
            atol=tolerance,
        )
    )
    out["branch_matches_hard"] = half_matches & full_matches
    return out


def branch_conditioned_activity_dependent_slowing(
    V: torch.Tensor,
    pulse_times_ms: torch.Tensor | Sequence[float],
    dt_ms: float | torch.Tensor,
    *,
    lengths_um: Any = None,
    node_mask: torch.Tensor | None = None,
    response_window_ms: tuple[float, float] = (0.1, 10.0),
    V_th: float = 0.0,
    dv_th: float | None = 10.0,
    baseline_pulse_indices: torch.Tensor | Sequence[int] | None = None,
    baseline_n_pulses: int = 1,
    reference_latency_ms: float | torch.Tensor | None = None,
    reference_velocity_m_per_s: float | torch.Tensor | None = None,
    tail_n_pulses: int = 10,
    interpolate: bool = True,
    eps: float = 1e-12,
    eps_time_ms2: float | None = None,
    eps_latency_ms: float | None = None,
    eps_velocity_m_per_s: float | None = None,
    window_margin_ms: float | None = None,
) -> dict[str, Any]:
    """Return hard ADS values with selected-crossing derivatives.

    Spike presence, first-crossing interval, selected sites, earliest site, and
    baseline/tail inclusion are discrete choices. The derivative is meaningful
    only while these signatures remain fixed. ``interpolate=False`` correctly
    gives zero voltage derivative of sampled crossing times.

    ``ads_percent`` is the percentage latency increase relative to baseline, so
    positive values mean slower arrival. When ``lengths_um`` is supplied,
    ``velocity_change_percent`` is the signed percentage change in propagation
    velocity and is therefore negative for slowing;
    ``velocity_slowing_percent`` is its sign-reversed, positive-for-slowing
    counterpart.

    ``eps_time_ms2`` regularizes the arrival-time variance in the velocity
    regression, ``eps_latency_ms`` regularizes the latency normalization, and
    ``eps_velocity_m_per_s`` regularizes the velocity normalization. Omitted
    values fall back independently to the legacy ``eps`` argument, preserving
    existing calls and default forward values.
    """
    (
        eps,
        eps_time_ms2,
        eps_latency_ms,
        eps_velocity_m_per_s,
        _,
    ) = _resolve_ads_stabilizers(
        eps,
        eps_time_ms2,
        eps_latency_ms,
        eps_velocity_m_per_s,
    )
    kwargs = dict(
        lengths_um=lengths_um,
        node_mask=node_mask,
        response_window_ms=response_window_ms,
        V_th=V_th,
        dv_th=dv_th,
        baseline_pulse_indices=baseline_pulse_indices,
        baseline_n_pulses=baseline_n_pulses,
        reference_latency_ms=reference_latency_ms,
        reference_velocity_m_per_s=reference_velocity_m_per_s,
        tail_n_pulses=tail_n_pulses,
        interpolate=interpolate,
        eps=eps,
        eps_time_ms2=eps_time_ms2,
        eps_latency_ms=eps_latency_ms,
        eps_velocity_m_per_s=eps_velocity_m_per_s,
        window_margin_ms=window_margin_ms,
    )
    hard = hard_activity_dependent_slowing(V, pulse_times_ms, dt_ms, **kwargs)
    _, F, C = V.shape
    dt = torch.as_tensor(dt_ms, device=V.device, dtype=V.dtype)
    pulse_times = _coerce_pulse_times_ms(
        pulse_times_ms, Fibs=F, device=V.device, dtype=V.dtype
    )
    N = pulse_times.shape[1]
    win_start = hard["pulse_times_ms"] + response_window_ms[0]
    win_end = hard["pulse_times_ms"] + response_window_ms[1]
    margin = (
        float(dt.detach().item())
        if window_margin_ms is None
        else float(window_margin_ms)
    )
    V_win, t_win, valid = _gather_time_windows_FNKC(
        V, dt, win_start, win_end, margin_ms=margin, sample_offset=0.0
    )
    v0, v1 = V_win[:, :, :-1, :], V_win[:, :, 1:, :]
    t0 = t_win[:, :, :-1]
    valid_pair = valid[:, :, :-1] & valid[:, :, 1:]
    with torch.no_grad():
        crossings = (v0 < V_th) & (v1 >= V_th) & valid_pair[:, :, :, None]
        if dv_th is not None:
            crossings &= (v1 - v0) / dt >= dv_th
        if interpolate:
            candidate_frac = ((V_th - v0) / (v1 - v0 + 1e-12)).clamp(0, 1)
            candidate_time = t0[:, :, :, None] + candidate_frac * dt
        else:
            candidate_time = t0[:, :, :, None] + dt
        crossings &= (candidate_time >= win_start[:, :, None, None]) & (
            candidate_time <= win_end[:, :, None, None]
        )
        has_comp = crossings.any(dim=2)
        first_idx = crossings.to(torch.int64).argmax(dim=2)

    gather_idx = first_idx[:, :, None, :]
    v0_sel = v0.gather(2, gather_idx).squeeze(2)
    v1_sel = v1.gather(2, gather_idx).squeeze(2)
    t0_sel = t0[:, :, :, None].expand_as(v0).gather(2, gather_idx).squeeze(2)
    safe_v0 = torch.where(has_comp, v0_sel, torch.full_like(v0_sel, V_th - 1.0))
    safe_v1 = torch.where(has_comp, v1_sel, torch.full_like(v1_sel, V_th + 1.0))
    if interpolate:
        frac = ((V_th - safe_v0) / (safe_v1 - safe_v0 + 1e-12)).clamp(0, 1)
        cross_safe = t0_sel + frac * dt
    else:
        cross_safe = t0_sel + dt
    valid_selected = has_comp & hard["weights"][:, None, :]
    earliest = torch.where(
        valid_selected, cross_safe, torch.full_like(cross_safe, float("inf"))
    )
    earliest_site = earliest.argmin(dim=-1)
    has_any = valid_selected.any(dim=-1)
    first_arrival_safe = earliest.min(dim=-1).values
    first_arrival_safe = torch.where(
        has_any, first_arrival_safe, torch.zeros_like(first_arrival_safe)
    )
    latency_safe = first_arrival_safe - pulse_times

    baseline_idx = hard["baseline_pulse_indices"]
    tail_idx = hard["tail_pulse_indices"]
    ref_latency = _as_F_vector(
        reference_latency_ms,
        Fibs=F,
        device=V.device,
        dtype=V.dtype,
        name="reference_latency_ms",
    )
    if ref_latency is None:
        valid_base = has_any[:, baseline_idx]
        vals = torch.where(
            valid_base,
            latency_safe[:, baseline_idx],
            torch.zeros_like(latency_safe[:, baseline_idx]),
        )
        count = valid_base.sum(dim=-1).clamp(min=1)
        baseline_safe = vals.sum(dim=-1) / count
    else:
        baseline_safe = ref_latency
    ads_safe = (
        100
        * (latency_safe - baseline_safe[:, None])
        / (baseline_safe[:, None] + eps_latency_ms)
    )
    tail_valid = has_any[:, tail_idx]
    tail_sum = torch.where(
        tail_valid, ads_safe[:, tail_idx], torch.zeros_like(ads_safe[:, tail_idx])
    ).sum(-1)
    tail_safe = tail_sum / tail_valid.sum(-1).clamp(min=1)

    out: dict[str, Any] = dict(hard)
    cross_out = cross_safe.permute(1, 0, 2).contiguous()
    out["t_cross_ms_comp"] = _attach(hard["t_cross_ms_comp"], cross_out)
    for key, selected in (
        ("arrival_time_ms", first_arrival_safe),
        ("latency_ms", latency_safe),
        ("baseline_latency_ms", baseline_safe),
        ("latency_shift_ms", latency_safe - baseline_safe[:, None]),
        ("ads_percent", ads_safe),
        ("final_ads_percent", ads_safe[:, -1]),
        ("tail_ads_percent", tail_safe),
    ):
        out[key] = _attach(hard[key], selected)
    out["latency_slowing_percent"] = out["ads_percent"]
    if N >= 2:
        output_isi_safe = first_arrival_safe[:, 1:] - first_arrival_safe[:, :-1]
        input_isi = pulse_times[:, 1:] - pulse_times[:, :-1]
        out["output_isi_ms"] = _attach(hard["output_isi_ms"], output_isi_safe)
        out["input_isi_ms"] = _attach(hard["input_isi_ms"], input_isi)
        out["isi_error_ms"] = _attach(hard["isi_error_ms"], output_isi_safe - input_isi)

    if lengths_um is not None:
        L = _coerce_lengths_um(lengths_um, Fibs=F, C=C, device=V.device, dtype=V.dtype)
        x = (torch.cumsum(L, dim=-1) - 0.5 * L)[:, None, :]
        w = valid_selected.to(dtype=V.dtype)
        W = w.sum(-1)
        cross_zero = torch.where(
            valid_selected, cross_safe, torch.zeros_like(cross_safe)
        )
        t_bar = (w * cross_zero).sum(-1, keepdim=True) / W[:, :, None].clamp(min=1)
        x_bar = (w * x).sum(-1, keepdim=True) / W[:, :, None].clamp(min=1)
        dtc = torch.where(
            valid_selected, cross_safe - t_bar, torch.zeros_like(cross_safe)
        )
        dxc = x - x_bar
        var_t = (w * dtc * dtc).sum(-1) / W.clamp(min=1)
        cov = (w * dtc * dxc).sum(-1) / W.clamp(min=1)
        vel_safe = 1e-3 * cov / (var_t + eps_time_ms2)
        vel_valid = (W >= 2) & (var_t > eps_time_ms2)
        vel_safe = torch.where(vel_valid, vel_safe, torch.zeros_like(vel_safe))
        out["v_m_per_s"] = _attach(hard["v_m_per_s"], vel_safe)
        out["speed_m_per_s"] = _attach(hard["speed_m_per_s"], vel_safe.abs())
        ref_velocity = _as_F_vector(
            reference_velocity_m_per_s,
            Fibs=F,
            device=V.device,
            dtype=V.dtype,
            name="reference_velocity_m_per_s",
        )
        if ref_velocity is None:
            base_valid = vel_valid[:, baseline_idx]
            base_vals = torch.where(
                base_valid,
                vel_safe[:, baseline_idx],
                torch.zeros_like(vel_safe[:, baseline_idx]),
            )
            baseline_vel_safe = base_vals.sum(-1) / base_valid.sum(-1).clamp(min=1)
        else:
            baseline_vel_safe = ref_velocity
        change_safe = (
            100
            * (vel_safe - baseline_vel_safe[:, None])
            / (baseline_vel_safe[:, None] + eps_velocity_m_per_s)
        )
        tail_vel_valid = vel_valid[:, tail_idx]
        tail_change = torch.where(
            tail_vel_valid,
            change_safe[:, tail_idx],
            torch.zeros_like(change_safe[:, tail_idx]),
        )
        tail_change_safe = tail_change.sum(-1) / tail_vel_valid.sum(-1).clamp(min=1)
        out["baseline_velocity_m_per_s"] = _attach(
            hard["baseline_velocity_m_per_s"], baseline_vel_safe
        )
        out["velocity_change_percent"] = _attach(
            hard["velocity_change_percent"], change_safe
        )
        out["velocity_slowing_percent"] = -out["velocity_change_percent"]
        out["final_velocity_slowing_percent"] = _attach(
            hard["final_velocity_slowing_percent"], -change_safe[:, -1]
        )
        out["tail_velocity_slowing_percent"] = _attach(
            hard["tail_velocity_slowing_percent"], -tail_change_safe
        )

    # Return a common (F,N,C) signature shape so the generic comparator can
    # detect event switches without broadcasting or opaque string encodings.
    time_pair = (t0_sel.detach() / dt.detach()).round().long()
    time_pair = torch.where(has_comp, time_pair, torch.full_like(time_pair, -1))
    earliest_site = torch.where(
        has_any, earliest_site, torch.full_like(earliest_site, -1)
    )
    sig = {
        "crossing_pair": time_pair,
        "crossing_present": has_comp.long(),
        "site_used": valid_selected.long(),
        "earliest_site": earliest_site[:, :, None].expand(F, N, C),
        "baseline_included": torch.zeros((F, N, C), device=V.device, dtype=torch.long),
        "tail_included": torch.zeros((F, N, C), device=V.device, dtype=torch.long),
    }
    if lengths_um is not None:
        sign = torch.sign(hard["v_m_per_s"].nan_to_num()).long()
        sig["velocity_sign"] = sign[:, :, None].expand(F, N, C)
    sig["baseline_included"][:, baseline_idx, :] = has_any[:, baseline_idx, None].long()
    sig["tail_included"][:, tail_idx, :] = has_any[:, tail_idx, None].long()
    out["branch_signature"] = sig
    hard_has = hard["has_crossing_comp"].permute(1, 0, 2)
    hard_cross = hard["t_cross_ms_comp"].permute(1, 0, 2)
    out["branch_matches_hard"] = (has_comp == hard_has) & (
        ~has_comp
        | torch.isclose(cross_safe.detach(), hard_cross, rtol=1e-10, atol=1e-10)
    )
    return out
