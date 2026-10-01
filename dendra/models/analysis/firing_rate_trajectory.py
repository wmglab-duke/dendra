"""Event-train firing-rate trajectories on existing crossing branches.

The hard reference and the local differentiable descriptor use identical
threshold-crossing decisions. The latter differentiates interpolated event
times through voltage on the selected crossing branch. Neither implementation
chooses a neuronal model, readout compartment, stimulus, or kernel time scale.
"""

from __future__ import annotations

import math

import torch

from .spike_timing import branch_conditioned_spike_timing, hard_spike_timing


def _trajectory(
    V: torch.Tensor,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor,
    *,
    tau_ms: float | torch.Tensor,
    V_spk: float | torch.Tensor,
    use_dv_gate: bool,
    dv_spk: float,
    refractory_ms: float,
    hard: bool,
) -> dict[str, object]:
    if V.ndim != 3 or not V.is_floating_point():
        raise ValueError("V must be a floating-point tensor of shape (T, F, C).")
    T, fibers, compartments = V.shape
    if T < 2:
        raise ValueError("V must contain at least 2 time samples.")
    if node_mask is None:
        raise ValueError("node_mask must select at least one readout site per neuron.")
    mask = torch.as_tensor(node_mask, device=V.device)
    if mask.shape == (compartments,):
        selected = (mask != 0)[None, :].expand(fibers, compartments)
    elif mask.shape == (fibers, compartments):
        selected = mask != 0
    else:
        raise ValueError("node_mask must have shape (C,) or (F, C).")
    if not bool(selected.any(dim=-1).all()):
        raise ValueError("node_mask must select at least one readout site per neuron.")

    dt = torch.as_tensor(dt_ms, device=V.device, dtype=V.dtype)
    tau = torch.as_tensor(tau_ms, device=V.device, dtype=V.dtype)
    if (
        dt.numel() != 1
        or not math.isfinite(float(dt.detach()))
        or float(dt.detach()) <= 0
    ):
        raise ValueError("dt_ms must be a finite positive scalar.")
    if (
        tau.numel() != 1
        or not math.isfinite(float(tau.detach()))
        or float(tau.detach()) <= 0
    ):
        raise ValueError("tau_ms must be a finite positive scalar.")

    timing_fn = hard_spike_timing if hard else branch_conditioned_spike_timing
    timing = timing_fn(
        V,
        dt,
        selected,
        V_spk=V_spk,
        use_dv_gate=use_dv_gate,
        dv_spk=dv_spk,
        refractory_ms=refractory_ms,
    )
    time_ms = torch.arange(T, device=V.device, dtype=V.dtype) * dt
    event_times = timing["event_times_ms"]
    counts = timing["count_comp"]
    neuron_curves: list[torch.Tensor] = []
    for f in range(fibers):
        site_curves: list[torch.Tensor] = []
        for c in torch.where(selected[f])[0].tolist():
            n = int(counts[f, c])
            if n == 0:
                site_curves.append(torch.zeros_like(time_ms))
                continue
            lag = time_ms[:, None] - event_times[None, :n, f, c]
            # K_tau(s) = 1{s >= 0} s exp(-s/tau) / tau^2, in 1/ms.
            # Clamp before exp to avoid overflow for pre-event samples.
            positive_lag = lag.clamp_min(0)
            kernel = positive_lag * torch.exp(-positive_lag / tau) / tau.square()
            site_curves.append(1000.0 * kernel.sum(dim=-1))
        neuron_curves.append(torch.stack(site_curves, dim=0).mean(dim=0))
    rate_hz = torch.stack(neuron_curves, dim=-1)
    count = (counts.to(V.dtype) * selected).sum(dim=-1) / selected.sum(dim=-1)
    return {
        "time_ms": time_ms,
        "rate_hz": rate_hz,
        "count": count,
        "count_comp": counts,
        "event_times_ms": event_times,
        "weights": selected.to(V.dtype),
        "branch_signature": timing["branch_signature"],
        "tau_ms": tau,
    }


def branch_conditioned_firing_rate_trajectory(
    V: torch.Tensor,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor,
    *,
    tau_ms: float | torch.Tensor,
    V_spk: float | torch.Tensor = 0.0,
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,
    refractory_ms: float = 0.0,
) -> dict[str, object]:
    r"""Return a local differentiable rate trajectory, in Hz, for every neuron.

    For each selected site, let ``t_i`` be its upward threshold-crossing
    times, linearly interpolated between recorded samples. At sample time
    ``t``, the site rate is ``1000 * sum_i K_tau(t - t_i)``, where
    ``K_tau(s) = 1{s >= 0} * s * exp(-s/tau) / tau**2`` has units ``1/ms``.
    The per-neuron ``rate_hz`` has shape ``(T, F)`` and is the mean of the
    selected site rates. ``node_mask`` must select at least one site per neuron;
    it can have shape ``(C,)`` or ``(F, C)``. ``tau_ms`` is a caller-chosen
    temporal smoothing scale. This causal kernel integrates to one event on an
    unbounded time axis; its integral on a finite recording can differ from
    the fixed-window crossing count because of onset lag and end truncation.

    The crossing-pair indices, optional upstroke gate, and refractory filter
    are discrete decisions made from detached voltage. Interpolated crossing
    times remain connected to the voltage graph, so spike-timing changes have
    gradients. Births and losses of events do not: in particular, a silent
    neuron has zero rate and no voltage gradient from this descriptor alone.
    ``branch_signature`` exposes the selected pair indices for local gradient
    checks; it is not a rule for rejecting candidate parameter updates.
    The fixed-window count and this kernelized event train are distinct hard
    descriptors: a spike's event count is discrete, while its contribution to
    this sampled trajectory depends on event time and ``tau_ms``. This is a
    local existing-event descriptor, not a birth/loss training objective. A
    separate voltage-intensity trajectory based on the smooth occupancy
    up-crossings in :func:`dendra.models.analysis.firing_rate` could supply
    birth/loss gradients, but needs its own hard-value and gradient validation.
    """
    return _trajectory(
        V,
        dt_ms,
        node_mask,
        tau_ms=tau_ms,
        V_spk=V_spk,
        use_dv_gate=use_dv_gate,
        dv_spk=dv_spk,
        refractory_ms=refractory_ms,
        hard=False,
    )


@torch.no_grad()
def hard_firing_rate_trajectory(
    V: torch.Tensor,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor,
    *,
    tau_ms: float | torch.Tensor,
    V_spk: float | torch.Tensor = 0.0,
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,
    refractory_ms: float = 0.0,
) -> dict[str, object]:
    """Hard event-train reference for the matching branch-conditioned trajectory."""
    return _trajectory(
        V,
        dt_ms,
        node_mask,
        tau_ms=tau_ms,
        V_spk=V_spk,
        use_dv_gate=use_dv_gate,
        dv_spk=dv_spk,
        refractory_ms=refractory_ms,
        hard=True,
    )
