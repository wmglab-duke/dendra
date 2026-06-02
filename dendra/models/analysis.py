from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ArrayLike1D = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[int]]
LengthLike = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[int], float, int]


def _soft_arrival_and_spike_gate(
    V: torch.Tensor,  # (T, F, C)
    dt_ms: float | torch.Tensor,
    *,
    # Soft arrival time settings (upstroke-based)
    beta: float = 50.0,
    dv0: float = 0.0,
    dv_scale: float = 1.0,
    lambda_early: float = 0.0,
    # Spike-present confidence gate (voltage-based)
    V_spk: float = 0.0,
    kappa_V: float = 20.0,
    gate_V_scale: float = 5.0,
    # Optional dV/dt gate
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,
    kappa_dv: float = 10.0,
    gate_dv_scale: float = 5.0,
):
    """
    Shared helper for soft spike localization.

    Returns
    -------
    t_hat_ms : torch.Tensor
        Soft upstroke arrival times, shape (F, C).
    p_spike : torch.Tensor
        Spike-present confidence, shape (F, C).
    """
    assert V.ndim == 3, "V must be (T, F, C)"
    T, Fibs, C = V.shape
    device, dtype = V.device, V.dtype

    if not torch.is_tensor(dt_ms):
        dt_ms = torch.tensor(dt_ms, device=device, dtype=dtype)
    else:
        dt_ms = dt_ms.to(device=device, dtype=dtype)

    # Time grid
    t_ms = torch.arange(T, device=device, dtype=dtype) * dt_ms  # (T,)

    # Upstroke-based soft arrival time
    dV = (V[1:] - V[:-1]) / dt_ms  # (T-1, F, C), mV/ms
    t_mid = 0.5 * (t_ms[1:] + t_ms[:-1])  # (T-1,)

    U = F.softplus((dV - dv0) / dv_scale)  # (T-1, F, C)
    logits = beta * U - lambda_early * t_mid[:, None, None]
    w_time = torch.softmax(logits, dim=0)
    t_hat_ms = (w_time * t_mid[:, None, None]).sum(dim=0)  # (F, C)

    # Spike-present confidence
    a = torch.logsumexp(kappa_V * (V - V_spk), dim=0) / kappa_V  # (F, C)
    p_spike = torch.sigmoid(a / gate_V_scale)

    if use_dv_gate:
        u = torch.logsumexp(kappa_dv * (dV - dv_spk), dim=0) / kappa_dv  # (F, C)
        p_dv = torch.sigmoid(u / gate_dv_scale)
        p_spike = p_spike * p_dv

    return t_hat_ms, p_spike


def conduction_velocity(
    V: torch.Tensor,  # (T, F, C) membrane potentials
    lengths_um: LengthLike,  # scalar, (C,), or (F, C) compartment lengths in um
    dt_ms: float | torch.Tensor,  # timestep in ms
    node_mask: torch.Tensor
    | None = None,  # optional (F, C) bool/float; 1 for nodes, 0 otherwise
    *,
    # Soft arrival time settings (upstroke-based)
    beta: float = 50.0,  # sharpness for soft-argmax over time
    dv0: float = 0.0,  # upstroke baseline threshold (mV/ms)
    dv_scale: float = 1.0,  # softness for upstroke score (mV/ms)
    lambda_early: float = 0.0,  # early-time bias (1/ms); use >0 if multiple spikes possible
    # Spike-present confidence gate (voltage-based)
    V_spk: float = 0.0,  # "spike-ish" voltage threshold (mV), e.g. 0 or -20
    kappa_V: float = 20.0,  # sharpness for smooth max over time
    gate_V_scale: float = 5.0,  # softness for sigmoid gate in mV
    # Optional dV/dt gate (often not necessary, but can help)
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,  # upstroke threshold for dv gate (mV/ms)
    kappa_dv: float = 10.0,  # sharpness for smooth max of dv
    gate_dv_scale: float = 5.0,  # softness for dv gate
    # Robustness / regularization knobs (return reg terms you can add to your loss)
    min_ess: float = 3.0,  # minimum effective # of contributing compartments per fiber
    min_var_ms2: float = 1e-3,  # minimum weighted Var(t_hat) to avoid blowups
    reg_ess_weight: float = 0.0,  # set >0 if you want to enforce min_ess
    reg_var_weight: float = 0.0,  # set >0 if you want to enforce min_var_ms2
    # Numerics
    eps: float = 1e-6,
    eps_speed: float = 1e-8,
):
    r"""
    Differentiably estimate per-fiber conduction velocity from spatiotemporal membrane potentials.

    This function computes a smooth, differentiable surrogate for conduction velocity for each
    fiber in a multicompartment model, given a tensor of membrane potentials `V[t, f, c]`
    and per-compartment physical lengths.

    The estimator has three conceptual steps:

    1. **Compartment positions** :math:`x_{f,c}` in :math:`\mu m` are formed from the cumulative
       sum of compartment lengths (midpoint convention).
    2. **Soft arrival times** :math:`\hat{t}_{f,c}` (ms) are estimated per compartment using a
       soft-argmax over time of an upstroke score based on :math:`dV/dt`.
    3. **(Weighted) linear regression** estimates a propagation speed by fitting
       :math:`x_{f,c} \approx b_f + v_f \hat{t}_{f,c}` across compartments within each fiber.
       Optional weights (gating) reduce the influence of compartments where an AP does not
       "arrive" or where the compartment should be excluded (e.g., internodes).

    All operations are differentiable w.r.t. `V` (and thus w.r.t. upstream model parameters).

    Parameters
    ----------
    V : torch.Tensor
        Membrane potentials with shape ``(T, F, C)`` where:

        - ``T`` : number of time samples
        - ``F`` : number of fibers
        - ``C`` : number of compartments per fiber

        Units in Axon are typically mV (but any consistent voltage units are acceptable; see Notes).

    lengths_um : scalar, array-like, or torch.Tensor
        Compartment physical lengths in micrometers. Accepts:

        - scalar: every compartment in every fiber has this length;
        - shape ``(C,)``: a compartment-length vector shared by all fibers;
        - shape ``(F, C)``: fiber-specific compartment lengths.

        These are used to compute compartment midpoint positions along each fiber.
        For `dn.Myelinated`, this should be internodal length
        (`model.deltax(model.diameters)[:, None].expand_as(model.dx)`).

    dt_ms : float or torch.Tensor
        Time step between samples in milliseconds. If provided as a tensor, it will be
        cast to ``V.dtype`` and moved to ``V.device``.

    node_mask : torch.Tensor or None, optional
        Optional compartment inclusion mask with shape ``(F, C)``. If provided:

        - boolean mask: ``True`` entries are included, ``False`` excluded
        - float mask: values act as multiplicative weights (e.g., soft node-ness)

        This is useful for complex myelinated models (e.g., MRG) where APs are
        generated only at nodes of Ranvier. If ``None``, all compartments are eligible.

    Other Parameters
    ----------------
    beta : float, default=50.0
        Soft-argmax sharpness over time for arrival time estimation. Larger values approach a
        hard argmax (peakier gradients); smaller values yield smoother arrival times.

    dv0 : float, default=0.0
        Baseline upstroke threshold in mV/ms applied to :math:`dV/dt` before the softplus.
        Increasing this can suppress small passive slopes.

    dv_scale : float, default=1.0
        Softness (scale) in mV/ms for the upstroke score
        :math:`U=\mathrm{softplus}((dV/dt - dv0)/dv\_scale)`.

    lambda_early : float, default=0.0
        Optional early-time bias (units 1/ms). The time logits are:
        ``beta * U - lambda_early * t_mid``.
        Use ``lambda_early > 0`` when multiple spikes may occur and you want to bias toward
        earlier events.

    V_spk : float, default=0.0
        Voltage reference (mV) for spike-present gating. The gating uses a smooth max of
        ``V - V_spk`` over time. Common choices: 0 mV, -20 mV, etc.

    kappa_V : float, default=20.0
        Sharpness of the smooth max over time for voltage gating. Larger values approach
        a hard maximum (more threshold-like).

    gate_V_scale : float, default=5.0
        Softness (mV) of the sigmoid converting the smooth max voltage score into a confidence.

    use_dv_gate : bool, default=False
        If ``True``, additionally gates weights using a smooth max of :math:`dV/dt` over time.
        This can help prevent passive compartments with small voltage peaks from contributing.

    dv_spk : float, default=10.0
        Reference threshold (mV/ms) for the optional :math:`dV/dt` gating.

    kappa_dv : float, default=10.0
        Sharpness for the smooth max over time applied to ``(dV/dt - dv_spk)`` in the dv gate.

    gate_dv_scale : float, default=5.0
        Softness for the sigmoid converting dv smooth max into a confidence.

    min_ess : float, default=3.0
        Minimum desired *effective sample size* (ESS) of contributing compartments per fiber.
        If the ESS is too small (e.g., too few nodes/compartments have non-negligible weight),
        the regression slope becomes noisy.

    min_var_ms2 : float, default=1e-3
        Minimum desired weighted variance of estimated arrival times (ms^2). If the time spread
        is too small, the velocity estimate can explode because the regression denominator
        approaches zero.

    reg_ess_weight : float, default=0.0
        Weight for an optional differentiable penalty encouraging ESS >= ``min_ess``.
        The penalty returned in ``out["reg"]`` includes:
        ``reg_ess_weight * softplus(min_ess - ess).mean()``.

    reg_var_weight : float, default=0.0
        Weight for an optional differentiable penalty encouraging Var(t_hat) >= ``min_var_ms2``.
        The penalty returned in ``out["reg"]`` includes:
        ``reg_var_weight * softplus(min_var_ms2 - var_t).mean()``.

    eps : float, default=1e-6
        Small constant for numerical stability in denominators.

    eps_speed : float, default=1e-8
        Small constant used for smooth absolute value:
        ``speed = sqrt(v^2 + eps_speed)``.

    Returns
    -------
    out : dict[str, torch.Tensor]
        Dictionary of per-fiber estimates and intermediate quantities:

        - ``"v_m_per_s"`` : torch.Tensor, shape ``(F,)``
            Signed conduction velocity estimate in meters/second.

        - ``"speed_m_per_s"`` : torch.Tensor, shape ``(F,)``
            Smooth speed (m/s), computed as ``sqrt(v^2 + eps_speed)`` (differentiable at 0).

        - ``"v_um_per_ms"`` : torch.Tensor, shape ``(F,)``
            Same as ``v_m_per_s`` but in micrometers/millisecond.
            Note: :math:`1~\mu m/ms = 10^{-3}~m/s`.

        - ``"t_hat_ms"`` : torch.Tensor, shape ``(F, C)``
            Soft arrival time per compartment (ms), computed via soft-argmax of an upstroke score.

        - ``"p_spike"`` : torch.Tensor, shape ``(F, C)``
            Spike-present confidence per compartment prior to applying ``node_mask``.
            Values in (0, 1).

        - ``"weights"`` : torch.Tensor, shape ``(F, C)``
            Final regression weights per compartment, equal to ``p_spike * node_mask`` (or
            ``p_spike`` if ``node_mask is None``).

        - ``"ess"`` : torch.Tensor, shape ``(F,)``
            Soft effective sample size per fiber:
            :math:`ESS = (\sum w)^2 / \sum w^2`.

        - ``"var_t_ms2"`` : torch.Tensor, shape ``(F,)``
            Weighted variance of ``t_hat_ms`` used in the regression denominator (ms^2).

        - ``"reg"`` : torch.Tensor, scalar
            Optional regularization penalty (0 if ``reg_ess_weight==reg_var_weight==0``).
            Add this to your task loss if you want to enforce robustness constraints.

    Notes
    -----
    **Arrival time definition.**
    Arrival times are computed from the time derivative:

    - ``dV = (V[t+1] - V[t]) / dt_ms`` (mV/ms)
    - upstroke score ``U = softplus((dV - dv0)/dv_scale)``
    - soft-argmax weights over time ``softmax(beta*U - lambda_early*t_mid)``
    - ``t_hat = sum_t w_t * t_mid``

    This tends to be more stable than threshold-crossing times because it uses a smooth peak
    of the upstroke rather than an event time.

    **Gating and node masks.**
    In myelinated models, internodes may not generate spikes. Passing ``node_mask`` (nodes=1,
    internodes=0) ensures the regression is based on node-to-node timing. If ``node_mask`` is
    omitted, the voltage gate often downweights internodes automatically, but explicit masking
    is usually cleaner.

    **When to use the optional dv gate.**
    If you observe that some compartments reach modest voltages (raising ``p_spike``) but do not
    exhibit a strong upstroke, enabling ``use_dv_gate`` can suppress their influence.

    **Units.**
    The code assumes lengths are in :math:`\mu m` and time is in ms, producing :math:`\mu m/ms`.
    This is consistent with Dendra's internal practice of representing lengths in :math:`\mu m`,
    therefore `model.dx` etc. can often be used directly. For `ax.Myelinated`, internodal length
    should be used instead of compartment length. The conversion to m/s is a fixed factor of :math:`10^{-3}`.

    **Directionality.**
    The returned ``v_m_per_s`` is signed: positive means increasing x with increasing arrival time.
    Use ``speed_m_per_s`` when you only care about magnitude.

    **Assumptions.**
    This estimator is best behaved when:

    - a single traveling AP (or a dominant one) is present within the time window,
    - recordings exclude the initiation region,
    - compartments used for fitting span a meaningful propagation distance.

    Examples
    --------
    Basic usage (single AP, known nodes):

    >>> out = conduction_velocity(
    ...     V, lengths_um, dt_ms,
    ...     node_mask=node_mask,
    ...     V_spk=-10.0,
    ...     beta=60.0,
    ... )
    >>> v = out["v_m_per_s"]          # (F,)
    >>> loss = task_loss(v) + out["reg"]
    >>> loss.backward()

    Add robustness penalties when training:

    >>> out = conduction_velocity(
    ...     V, lengths_um, dt_ms,
    ...     node_mask=node_mask,
    ...     reg_ess_weight=1e-2, min_ess=4.0,
    ...     reg_var_weight=1e-2, min_var_ms2=5e-3,
    ... )
    >>> loss = task_loss + out["reg"]

    See Also
    --------
    torch.logsumexp : Used to compute smooth maxima (differentiable approximation).
    torch.softmax : Used to implement soft-argmax over time.

    """
    assert V.ndim == 3, "V must be (T, F, C)"
    _, Fibs, C = V.shape

    device, dtype = V.device, V.dtype

    # ---- 1) Positions x (um): midpoint of each compartment along the fiber
    lengths_um = _coerce_lengths_um(
        lengths_um, Fibs=Fibs, C=C, device=device, dtype=dtype
    )
    x_um = torch.cumsum(lengths_um, dim=-1) - 0.5 * lengths_um  # (F, C)

    # ---- 2-4) Shared temporal localization + spike-present confidence
    t_hat_ms, p_spike = _soft_arrival_and_spike_gate(
        V,
        dt_ms,
        beta=beta,
        dv0=dv0,
        dv_scale=dv_scale,
        lambda_early=lambda_early,
        V_spk=V_spk,
        kappa_V=kappa_V,
        gate_V_scale=gate_V_scale,
        use_dv_gate=use_dv_gate,
        dv_spk=dv_spk,
        kappa_dv=kappa_dv,
        gate_dv_scale=gate_dv_scale,
    )

    # ---- 5) Optional node_mask restriction
    if node_mask is None:
        nm = torch.ones((Fibs, C), device=device, dtype=dtype)
    else:
        nm = node_mask.to(device=device, dtype=dtype)
        assert nm.shape == (Fibs, C), "node_mask must be (F, C)"

    weights = p_spike * nm  # (F, C)

    # ---- 6) Weighted least-squares fit: x ≈ b + v * t_hat
    W = weights.sum(dim=-1, keepdim=True) + eps  # (F, 1)

    t_bar = (weights * t_hat_ms).sum(dim=-1, keepdim=True) / W  # (F, 1)
    x_bar = (weights * x_um).sum(dim=-1, keepdim=True) / W  # (F, 1)

    dtc = t_hat_ms - t_bar  # (F, C)
    dxc = x_um - x_bar  # (F, C)

    cov = (weights * dtc * dxc).sum(dim=-1) / W.squeeze(-1)  # (F,)
    var_t = (weights * dtc * dtc).sum(dim=-1) / W.squeeze(-1)  # (F,) ms^2

    v_um_per_ms = cov / (var_t + eps)  # (F,) um/ms
    v_m_per_s = v_um_per_ms * 1e-3  # (F,) m/s
    speed_m_per_s = torch.sqrt(v_m_per_s * v_m_per_s + eps_speed)  # smooth |v|

    # ---- 7) ESS and optional penalties
    w2 = (weights * weights).sum(dim=-1) + eps  # (F,)
    ess = (weights.sum(dim=-1) ** 2) / w2  # (F,)

    reg = V.new_zeros(())
    if reg_ess_weight > 0.0:
        reg = (
            reg
            + reg_ess_weight
            * F.softplus(torch.tensor(min_ess, device=device, dtype=dtype) - ess).mean()
        )
    if reg_var_weight > 0.0:
        reg = (
            reg
            + reg_var_weight
            * F.softplus(
                torch.tensor(min_var_ms2, device=device, dtype=dtype) - var_t
            ).mean()
        )

    return {
        "v_m_per_s": v_m_per_s,
        "speed_m_per_s": speed_m_per_s,
        "v_um_per_ms": v_um_per_ms,
        "t_hat_ms": t_hat_ms,
        "p_spike": p_spike,
        "weights": weights,
        "ess": ess,
        "var_t_ms2": var_t,
        "reg": reg,
    }


def action_potential_width(
    V: torch.Tensor,  # (T, F, C)
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    # Optionally reuse from conduction_velocity(...)
    t_hat_ms: torch.Tensor | None = None,  # (F, C)
    p_spike: torch.Tensor | None = None,  # (F, C)
    # Width definition
    mode: str = "half_height",  # {"half_height", "half_peak_to_peak"}
    # Same localization / gating knobs as conduction_velocity
    beta: float = 50.0,
    dv0: float = 0.0,
    dv_scale: float = 1.0,
    lambda_early: float = 0.0,
    V_spk: float = 0.0,
    kappa_V: float = 20.0,
    gate_V_scale: float = 5.0,
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,
    kappa_dv: float = 10.0,
    gate_dv_scale: float = 5.0,
    # Local windows around the spike (ms)
    baseline_pre_ms: float = 1.0,
    baseline_guard_ms: float = 0.15,
    peak_pre_ms: float = 0.2,
    peak_post_ms: float = 1.5,
    trough_pre_ms: float = 0.3,  # used only in mode="half_peak_to_peak"
    trough_post_ms: float = 3.0,
    width_pre_ms: float = 1.0,
    width_post_ms: float = 3.0,
    gate_t_scale_ms: float = 0.05,
    # Full-width / return-to-baseline metric
    full_width_pre_ms: float | None = None,
    full_width_post_ms: float = 8.0,
    baseline_margin_mV: float = 1.0,
    V_full_width_scale: float | None = None,
    return_tail_ms: float = 0.5,
    # Smooth extrema / thresholding
    kappa_peak: float = 10.0,
    kappa_trough: float = 10.0,
    half_level: float = 0.5,
    V_width_scale: float = 1.0,
    # Weighting / regularization
    amp_min_mV: float = 20.0,
    amp_scale_mV: float = 5.0,
    min_ess: float = 3.0,
    min_amp_mV: float = 5.0,
    reg_ess_weight: float = 0.0,
    reg_amp_weight: float = 0.0,
    # Optional full-width / return regularizers
    reg_full_width_weight: float = 0.0,
    max_full_width_ms: float = 8.0,
    reg_return_weight: float = 0.0,
    eps: float = 1e-6,
    # Set True only if you want dense T x F x C diagnostic traces returned.
    # Leaving this False avoids retaining large extra tensors in the graph.
    return_time_traces: bool = False,
):
    """
    Differentiable AP width surrogate.

    Returns per-fiber AP width estimates by:
      1) localizing the main spike per compartment,
      2) constructing a differentiable half-amplitude level,
      3) integrating a soft indicator of V > V_half over time.

    Also returns a differentiable baseline-relative full-width metric:

        full_width_ms ~= time spent above V_base + baseline_margin_mV

    within a spike-centered window.

    The extra return-to-baseline diagnostic penalizes / measures whether the
    trace is still above baseline near the end of the allowed post-spike window.

    Assumes these are available in the outer scope:

        import torch
        import torch.nn.functional as F

    and that _soft_arrival_and_spike_gate(...) is defined.
    """

    assert V.ndim == 3, "V must be (T, F, C)"
    T, Fibs, C = V.shape
    device, dtype = V.device, V.dtype

    assert baseline_pre_ms > baseline_guard_ms, (
        "baseline_pre_ms must be > baseline_guard_ms"
    )
    assert full_width_post_ms > 0.0, "full_width_post_ms must be positive"
    assert return_tail_ms > 0.0, "return_tail_ms must be positive"

    if full_width_pre_ms is None:
        full_width_pre_ms = width_pre_ms

    if V_full_width_scale is None:
        V_full_width_scale = V_width_scale

    if not torch.is_tensor(dt_ms):
        dt_ms = torch.tensor(dt_ms, device=device, dtype=dtype)
    else:
        dt_ms = dt_ms.to(device=device, dtype=dtype)

    if t_hat_ms is None or p_spike is None:
        _t_hat_ms, _p_spike = _soft_arrival_and_spike_gate(
            V,
            dt_ms,
            beta=beta,
            dv0=dv0,
            dv_scale=dv_scale,
            lambda_early=lambda_early,
            V_spk=V_spk,
            kappa_V=kappa_V,
            gate_V_scale=gate_V_scale,
            use_dv_gate=use_dv_gate,
            dv_spk=dv_spk,
            kappa_dv=kappa_dv,
            gate_dv_scale=gate_dv_scale,
        )
        if t_hat_ms is None:
            t_hat_ms = _t_hat_ms
        if p_spike is None:
            p_spike = _p_spike

    t_hat_ms = t_hat_ms.to(device=device, dtype=dtype)
    p_spike = p_spike.to(device=device, dtype=dtype)

    assert t_hat_ms.shape == (Fibs, C)
    assert p_spike.shape == (Fibs, C)

    if node_mask is None:
        nm = torch.ones((Fibs, C), device=device, dtype=dtype)
    else:
        nm = node_mask.to(device=device, dtype=dtype)
        assert nm.shape == (Fibs, C), "node_mask must be (F, C)"

    t_ms = torch.arange(T, device=device, dtype=dtype) * dt_ms
    t = t_ms[:, None, None]  # (T, 1, 1)

    def interval_gate(
        t: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        tau_ms: float | torch.Tensor,
    ) -> torch.Tensor:
        tau = torch.as_tensor(tau_ms, device=device, dtype=dtype)
        return torch.sigmoid((t - a[None, :, :]) / tau) * torch.sigmoid(
            (b[None, :, :] - t) / tau
        )

    # ---------------------------------------------------------------------
    # 1. Pre-spike baseline
    # ---------------------------------------------------------------------
    g_base = interval_gate(
        t,
        t_hat_ms - baseline_pre_ms,
        t_hat_ms - baseline_guard_ms,
        gate_t_scale_ms,
    )
    V_base = (g_base * V).sum(dim=0) / (g_base.sum(dim=0) + eps)  # (F, C)

    # ---------------------------------------------------------------------
    # 2. Local soft peak
    # ---------------------------------------------------------------------
    g_peak = interval_gate(
        t,
        t_hat_ms - peak_pre_ms,
        t_hat_ms + peak_post_ms,
        gate_t_scale_ms,
    )

    alpha_peak = torch.softmax(
        kappa_peak * V + torch.log(g_peak + eps),
        dim=0,
    )

    V_peak = (alpha_peak * V).sum(dim=0)  # (F, C)
    t_peak_ms = (alpha_peak * t).sum(dim=0)  # (F, C)

    # ---------------------------------------------------------------------
    # 3. Low reference level for half-width
    # ---------------------------------------------------------------------
    if mode == "half_height":
        V_low = V_base

    elif mode == "half_peak_to_peak":
        g_trough = interval_gate(
            t,
            t_peak_ms - trough_pre_ms,
            t_peak_ms + trough_post_ms,
            gate_t_scale_ms,
        )

        alpha_trough = torch.softmax(
            -kappa_trough * V + torch.log(g_trough + eps),
            dim=0,
        )

        V_low = (alpha_trough * V).sum(dim=0)  # (F, C)

    else:
        raise ValueError("mode must be 'half_height' or 'half_peak_to_peak'")

    amp_mV = V_peak - V_low
    V_half = V_low + half_level * amp_mV

    # ---------------------------------------------------------------------
    # 4. Half-width = soft time spent above half-level
    # ---------------------------------------------------------------------
    V_width_scale_t = torch.as_tensor(
        V_width_scale,
        device=device,
        dtype=dtype,
    )

    g_width = interval_gate(
        t,
        t_peak_ms - width_pre_ms,
        t_peak_ms + width_post_ms,
        gate_t_scale_ms,
    )

    q_half = torch.sigmoid((V - V_half[None, :, :]) / V_width_scale_t)

    width_ms_comp = (g_width * q_half).sum(dim=0) * dt_ms  # (F, C)

    # ---------------------------------------------------------------------
    # 5. Full-width = soft time spent above baseline + margin
    # ---------------------------------------------------------------------
    baseline_margin_t = torch.as_tensor(
        baseline_margin_mV,
        device=device,
        dtype=dtype,
    )

    V_full_width_scale_t = torch.as_tensor(
        V_full_width_scale,
        device=device,
        dtype=dtype,
    )

    # Threshold for "above baseline". Use a small positive margin to avoid
    # counting noise / numerical jitter around rest.
    V_full = V_base + baseline_margin_t  # (F, C)

    g_full_width = interval_gate(
        t,
        t_peak_ms - full_width_pre_ms,
        t_peak_ms + full_width_post_ms,
        gate_t_scale_ms,
    )

    q_full = torch.sigmoid((V - V_full[None, :, :]) / V_full_width_scale_t)

    full_width_ms_comp = (g_full_width * q_full).sum(dim=0) * dt_ms  # (F, C)

    # ---------------------------------------------------------------------
    # 6. Return-to-baseline tail diagnostics
    # ---------------------------------------------------------------------
    # This asks: near the end of the allowed post-spike window, is the voltage
    # still above the baseline threshold?
    #
    # return_above_baseline_prob_comp is bounded in [0, 1].
    # return_excess_mV_comp is usually better as a loss term because it keeps
    # useful gradients when the trace is far above baseline.
    g_return_tail = interval_gate(
        t,
        t_peak_ms + full_width_post_ms - return_tail_ms,
        t_peak_ms + full_width_post_ms,
        gate_t_scale_ms,
    )

    return_above_baseline_prob_comp = (g_return_tail * q_full).sum(dim=0) / (
        g_return_tail.sum(dim=0) + eps
    )  # (F, C)

    return_excess_mV_comp = (
        g_return_tail
        * F.softplus((V - V_full[None, :, :]) / V_full_width_scale_t)
        * V_full_width_scale_t
    ).sum(dim=0) / (g_return_tail.sum(dim=0) + eps)  # (F, C)

    # ---------------------------------------------------------------------
    # 7. Aggregate across compartments / nodes
    # ---------------------------------------------------------------------
    p_amp = torch.sigmoid((amp_mV - amp_min_mV) / amp_scale_mV)
    weights = nm * p_spike * p_amp

    W = weights.sum(dim=-1) + eps

    width_ms = (weights * width_ms_comp).sum(dim=-1) / W  # (F,)
    full_width_ms = (weights * full_width_ms_comp).sum(dim=-1) / W  # (F,)

    return_above_baseline_prob = (weights * return_above_baseline_prob_comp).sum(
        dim=-1
    ) / W  # (F,)

    return_excess_mV = (weights * return_excess_mV_comp).sum(dim=-1) / W  # (F,)

    # Effective number of contributing compartments per fiber
    ess = (weights.sum(dim=-1) ** 2) / ((weights * weights).sum(dim=-1) + eps)

    # ---------------------------------------------------------------------
    # 8. Optional regularization
    # ---------------------------------------------------------------------
    reg = V.new_zeros(())

    if reg_ess_weight > 0.0:
        reg = (
            reg
            + reg_ess_weight
            * F.softplus(torch.tensor(min_ess, device=device, dtype=dtype) - ess).mean()
        )

    if reg_amp_weight > 0.0:
        reg = (
            reg
            + reg_amp_weight
            * F.softplus(
                torch.tensor(min_amp_mV, device=device, dtype=dtype) - amp_mV
            ).mean()
        )

    if reg_full_width_weight > 0.0:
        reg = (
            reg
            + reg_full_width_weight
            * F.softplus(
                full_width_ms
                - torch.tensor(max_full_width_ms, device=device, dtype=dtype)
            ).mean()
        )

    if reg_return_weight > 0.0:
        reg = reg + reg_return_weight * return_excess_mV.mean()

    # ---------------------------------------------------------------------
    # 9. Return metrics
    # ---------------------------------------------------------------------
    out = {
        # Existing half-width outputs
        "width_ms": width_ms,  # (F,)
        "width_ms_comp": width_ms_comp,  # (F, C)
        # New baseline-relative full-width outputs
        "full_width_ms": full_width_ms,  # (F,)
        "full_width_ms_comp": full_width_ms_comp,  # (F, C)
        "V_full_mV": V_full,  # (F, C)
        # New return-to-baseline diagnostics
        "return_above_baseline_prob": return_above_baseline_prob,  # (F,)
        "return_above_baseline_prob_comp": return_above_baseline_prob_comp,  # (F, C)
        "return_excess_mV": return_excess_mV,  # (F,)
        "return_excess_mV_comp": return_excess_mV_comp,  # (F, C)
        # Existing localization / amplitude diagnostics
        "t_hat_ms": t_hat_ms,  # (F, C)
        "t_peak_ms": t_peak_ms,  # (F, C)
        "V_base_mV": V_base,  # (F, C)
        "V_low_mV": V_low,  # (F, C)
        "V_peak_mV": V_peak,  # (F, C)
        "V_half_mV": V_half,  # (F, C)
        "amp_mV": amp_mV,  # (F, C)
        "p_spike": p_spike,  # (F, C)
        "p_amp": p_amp,  # (F, C)
        "weights": weights,  # (F, C)
        "ess": ess,  # (F,)
        "reg": reg,  # scalar
    }

    if return_time_traces:
        out.update(
            {
                "q_half": q_half,  # (T, F, C)
                "q_full": q_full,  # (T, F, C)
                "g_width": g_width,  # (T, F, C)
                "g_full_width": g_full_width,  # (T, F, C)
                "g_return_tail": g_return_tail,  # (T, F, C)
            }
        )

    return out


def chronaxie_from_trials(
    V: torch.Tensor,  # (T, P, C)
    amplitudes: torch.Tensor,  # (P,)
    pws_ms: torch.Tensor,  # (P,)
    *,
    # --- Activation surrogate from voltages ---
    V_spk: float = 0.0,
    kappa_V: float = 20.0,
    gate_V_scale: float = 5.0,
    node_mask: Optional[torch.Tensor] = None,  # (C,) bool/float weights
    time_window: Optional[Tuple[int, int]] = None,
    # --- Strength axis convention ---
    strength: Optional[
        torch.Tensor
    ] = None,  # (P,) use e.g. -amplitudes for cathodic-negative
    use_abs_strength: bool = False,
    # --- Pulse-width grouping ---
    pw_round_decimals: Optional[int] = None,
    enforce_min_trials_per_pw: bool = True,
    min_trials_per_pw: int = 2,
    # --- Threshold extraction ---
    # onset            -> I_th = I_on (soft lowest-active)
    # onset_midpoint   -> I_th = 0.5*(I_pre + I_on) (nearest inactive below onset)
    # ptarget          -> soft-argmin around p_target (needs samples near boundary)
    # bracket_midpoint -> historical; can fail under block
    threshold_method: Literal[
        "onset", "onset_midpoint", "ptarget", "bracket_midpoint"
    ] = "onset_midpoint",
    # ptarget method
    p_target: float = 0.5,
    alpha_thresh: float = 200.0,
    # Soft extreme controls (used for onset / block boundaries)
    alpha_extreme: float = 50.0,
    membership_power: float = 10.0,
    # Robust "nearest inactive below onset" (gap-based; avoids being fooled by block)
    gap_alpha: float = 200.0,
    below_gate_frac: float = 0.02,
    # Also estimate block boundary diagnostics
    compute_block: bool = True,
    # --- Optional robustness regularizers ---
    # NOTE: monotone regularizer conflicts with block; use unimodal instead.
    p_low: float = 0.05,
    p_high: float = 0.95,
    reg_bracket_weight: float = 0.0,
    reg_monotone_weight: float = 0.0,
    reg_unimodal_weight: float = 0.0,
    unimodal_beta_peak: float = 50.0,
    unimodal_tau_idx: float = 1.0,
    reg_weiss_fit_weight: float = 0.0,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    r"""
    Differentiably estimate chronaxie and rheobase from trial-wise voltage recordings,
    with explicit support for **non-monotone activation vs amplitude** (e.g., high-amplitude
    conduction block).

    This function is designed for the common simulation/recording format where each trial
    corresponds to a single `(pulse width, amplitude)` pair:

    - ``V[t, p, c]``: membrane potential time series for trial ``p`` across compartments
    - ``amplitudes[p]``: stimulus amplitude for trial ``p``
    - ``pws_ms[p]``: pulse width (ms) for trial ``p``

    The key difference vs. many threshold extractors is that it can separate:

    - the **onset threshold** ("first activation": lowest amplitude that yields an AP),
      which is what chronaxie typically refers to, and
    - an optional **block boundary** at high amplitude (if activation becomes non-monotone).

    The output includes both the estimated chronaxie/rheobase and intermediate per-pulse-width
    threshold/boundary estimates.

    The computation has three stages:

    1) **Smooth activation surrogate per trial**
       A scalar "activation confidence" ``p_spike_trial[p] ∈ (0, 1)`` is computed from each
       voltage recording using a differentiable approximation to a max-over-time-and-space
       spike detector:

       - compute ``Z = V - V_spk``
       - approximate ``max_{t,c} Z[t,p,c]`` with a log-sum-exp (softmax) using ``kappa_V``
       - map to (0,1) with a sigmoid with scale ``gate_V_scale``

       This avoids non-differentiable threshold crossing and keeps gradients defined everywhere.

    2) **Group trials by pulse width**
       Trials are grouped by unique values in ``pws_ms`` (optionally rounded).

    3) **Extract a differentiable threshold strength per pulse width**
       For each pulse width group, a differentiable onset threshold is computed. The default
       (``threshold_method="onset_midpoint"``) is robust to high-amplitude block:

       - ``I_on(d)``: a soft estimate of the **lowest activating strength** (soft-min over
         strength weighted by activation membership)
       - ``I_pre(d)``: a soft estimate of the **highest inactive strength below onset**
         (selected by smallest positive gap to ``I_on``, weighted by inactivity membership)

       Then the onset threshold is defined as:
           ``I_th(d) = 0.5 * (I_pre(d) + I_on(d))``

       This definition matches the common empirical rule "midpoint between highest inactive
       and lowest active", while remaining differentiable and avoiding contamination from
       blocked high-amplitude non-activation.

       Optionally, the function also estimates high-amplitude block boundaries:
       - ``I_last(d)``: soft estimate of the **highest activating strength**
       - ``I_post(d)``: soft estimate of the **lowest inactive strength above I_last**
       - ``I_block_mid(d) = 0.5*(I_last + I_post)`` (diagnostic)


    Given per-pulse-width thresholds ``I_th(d)``, the function uses the Weiss charge form:

        ``Q_th(d) = d * I_th(d) ≈ I_r * d + I_r * c``

    where:
    - ``I_r`` is rheobase (slope),
    - ``c`` is chronaxie (intercept/slope).

    A **weighted least-squares** fit is used. Pulse widths where the threshold extraction
    is ambiguous (e.g., no strong evidence of a subthreshold point and an activating point)
    are automatically downweighted via ``pw_weight`` (see Returns).

    Parameters
    ----------
    V : torch.Tensor
        Voltage recordings with shape ``(T, P, C)``:

        - ``T``: number of time samples
        - ``P``: number of trials
        - ``C``: number of compartments

        Units are typically mV. Gradients propagate through ``V`` (and therefore through any
        upstream differentiable simulator).

        Important: the activation surrogate detects *any* spike-like event within the scored
        compartments. If you care about **propagating** activation (not just local initiation),
        ensure that `node_mask` selects compartments in the region where propagation is assessed
        (often distal compartments).

    amplitudes : torch.Tensor
        Stimulus amplitudes, shape ``(P,)``. These are typically design constants and do not
        need gradients. Units are arbitrary (uA, mA, etc.) but must be consistent across trials.

    pws_ms : torch.Tensor
        Pulse widths in milliseconds, shape ``(P,)``. Must contain at least 2 distinct pulse
        widths to estimate chronaxie.

    V_spk : float, default=0.0
        Voltage reference for spike scoring (same units as ``V``). The spike score is based on
        ``V - V_spk``. Example choices: 0 mV, -20 mV.

    kappa_V : float, default=20.0
        Sharpness of the log-sum-exp used to approximate a max. Larger values make spike scoring
        closer to a hard max but can produce peakier gradients.

    gate_V_scale : float, default=5.0
        Softness of the sigmoid mapping the spike score to confidence in (0,1). Smaller values
        act more threshold-like.

    node_mask : torch.Tensor, optional
        Compartment weights/mask, shape ``(C,)`` (bool or float).

        - bool: True compartments included, False excluded
        - float: nonnegative weights (0 excludes)

        Used inside the smooth max over compartments. This is the primary mechanism to focus
        activation scoring on nodes of Ranvier or distal compartments.

    time_window : tuple[int, int], optional
        Time index window ``(t_start, t_end)`` applied as ``V[t_start:t_end]`` before scoring.
        Useful to exclude baseline and/or late artifacts.

    strength : torch.Tensor, optional
        A per-trial monotone “strength” variable, shape ``(P,)``, used as the 1D axis along which
        thresholds are extracted. If None, defaults to ``amplitudes``.

        Use this when “stronger stimulus” is not numerically larger amplitude. Examples:
        - cathodic negative currents: use ``strength = -amplitudes`` so stronger means larger
        - magnitude-only experiments: use ``use_abs_strength=True``

        All threshold outputs (I_th, rheobase, boundaries) are in the units of `strength`.

    use_abs_strength : bool, default=False
        If True, thresholds are extracted on ``abs(strength)`` (or ``abs(amplitudes)``).

    pw_round_decimals : int, optional
        If provided, pulse widths are rounded to this number of decimals before grouping.
        Useful if `pws_ms` contains float representation noise.

        Note: grouping is discrete (not differentiable w.r.t. pws), which is typically fine
        since pws are design constants.

    enforce_min_trials_per_pw : bool, default=True
        If True, raises ValueError if any pulse width group has fewer than `min_trials_per_pw` trials.

    min_trials_per_pw : int, default=2
        Minimum trial count per pulse width group when enforcement is enabled.

    threshold_method : {"onset", "onset_midpoint", "ptarget", "bracket_midpoint"}, default="onset_midpoint"
        How to compute the per-pulse-width threshold:

        - ``"onset"``:
          ``I_th(d) = I_on(d)``, i.e. soft estimate of the **lowest active** strength.
          Closest to the empirical "lowest activating amplitude".

        - ``"onset_midpoint"`` (recommended for chronaxie):
          ``I_th(d) = 0.5*(I_pre(d) + I_on(d))`` where I_pre is the **highest inactive below onset**
          (soft, block-robust). Closest to empirical "midpoint between highest inactive and lowest active".

        - ``"ptarget"``:
          Selects the strength whose activation confidence is closest to ``p_target`` using a soft-argmin.
          This requires trials near the boundary; it is often poorly conditioned for all-or-none activation.

        - ``"bracket_midpoint"``:
          Legacy "global bracket" midpoint between soft max inactive and soft min active. This can fail under
          block because high-amplitude blocked trials appear inactive and dominate the max-inactive selection.

    p_target : float, default=0.5
        Used only for ``threshold_method="ptarget"``.

    alpha_thresh : float, default=200.0
        Used only for ``threshold_method="ptarget"``; soft-argmin sharpness.

    alpha_extreme : float, default=50.0
        Sharpness for soft extreme selection (soft-min / soft-max) used in onset/block boundaries.
        Internally strengths are normalized within each PW group, so this is dimensionless.

    membership_power : float, default=10.0
        Sharpens activation/inactivation membership by exponentiating probabilities in log-space.
        Larger values make "active" behave more like a hard set (p≈1) and "inactive" more like p≈0.
        This improves extreme selection when p_spike is nearly binary.

    gap_alpha : float, default=200.0
        Sharpness for selecting the **nearest** inactive point below onset (and optionally above last active).
        Larger values behave more like selecting the single closest gap.

    below_gate_frac : float, default=0.02
        Scale (fraction of within-group strength range) controlling a smooth gate that suppresses points on
        the wrong side of the boundary when selecting "inactive below onset" (or "inactive above last active").

    compute_block : bool, default=True
        If True, compute high-amplitude block boundary diagnostics (I_last, I_post, I_block_mid).
        These diagnostics do not affect the onset threshold unless you explicitly use them.

    p_low, p_high : float, default=0.05, 0.95
        Targets for optional bracketing regularization. See `reg_bracket_weight`.

    reg_bracket_weight : float, default=0.0
        Optional regularizer that encourages each PW group to contain evidence of both:
        - a low-strength non-activating trial (p at minimum strength <= p_low),
        - at least one strongly activating trial (max p >= p_high).

        This is *block compatible* because it uses `max p` rather than p at max strength.

    reg_monotone_weight : float, default=0.0
        Optional regularizer that encourages p_spike to be nondecreasing with strength within each PW group.
        This **conflicts with block** (0→1→0 behavior), so leave it at 0 if block is expected.

    reg_unimodal_weight : float, default=0.0
        Optional regularizer that encourages **unimodality** of p_spike vs strength within each PW group:
        allows 0→1→0 (single peak) but discourages oscillations (e.g., 0→1→0→1).
        This is the recommended shape prior if block is expected.

    unimodal_beta_peak : float, default=50.0
        Sharpness of the soft peak-location estimate used by the unimodality regularizer.

    unimodal_tau_idx : float, default=1.0
        Softness of the “before/after peak” partition used by the unimodality regularizer
        (in index units along the sorted strength axis).

    reg_weiss_fit_weight : float, default=0.0
        Optional regularizer penalizing Weiss-fit residual MSE across pulse widths, weighted by pw_weight.
        Useful if you want extracted thresholds to adhere closely to Weiss/Lapicque behavior.

    eps : float, default=1e-8
        Numerical stability constant used in logs/divisions.

    Returns
    -------
    out : dict[str, Any]
        Dictionary of differentiable outputs (gradients w.r.t. V):

        chronaxie_ms : torch.Tensor, shape ``()``
            Estimated chronaxie in milliseconds.

        rheobase : torch.Tensor, shape ``()``
            Estimated rheobase in the units of `strength` (or amplitudes if strength is None).

        pw_unique_ms : torch.Tensor, shape ``(D,)``
            Unique pulse widths used for estimation, sorted ascending.

        I_th : torch.Tensor, shape ``(D,)``
            Differentiable onset threshold per pulse width (definition depends on `threshold_method`).

        Q_th : torch.Tensor, shape ``(D,)``
            Charge thresholds: ``Q_th[d] = pw_unique_ms[d] * I_th[d]``.

        p_spike_trial : torch.Tensor, shape ``(P,)``
            Smooth activation confidence per trial in (0,1).

        pw_group_id : torch.Tensor, shape ``(P,)``
            Integer PW group index per trial (maps trials to entries of pw_unique_ms).
            This is discrete and mainly for inspection/debugging.

        pw_weight : torch.Tensor, shape ``(D,)``
            Reliability weights per pulse width used in the weighted Weiss fit. These downweight
            pulse widths where evidence of bracketing (inactive at low strength and some activation)
            is weak.

        boundaries : dict[str, torch.Tensor]
            Boundary diagnostics (each shape ``(D,)``):

            - ``I_on``:
                soft estimate of lowest activating strength (“lowest active”).

            - ``I_pre_inactive``:
                soft estimate of highest inactive strength below onset (block-robust).

            - ``I_last_active``:
                soft estimate of highest activating strength (“last active”).

            - ``I_post_inactive``:
                soft estimate of lowest inactive strength above last active (for block diagnostics).

            - ``I_block_mid``:
                midpoint between I_last_active and I_post_inactive (diagnostic block threshold).

        fit : dict
            Weiss fit diagnostics:
              - ``slope``     (rheobase)
              - ``intercept`` (rheobase * chronaxie)
              - ``weiss_mse`` (weighted MSE)

        reg : torch.Tensor, shape ``()``
            Optional regularization term (0 if all reg_*_weight are 0). Add to your training loss
            if you are optimizing model parameters through this chronaxie estimator.

    Notes
    -----

    - You must have at least 2 distinct pulse widths to estimate chronaxie.
    - For each pulse width, threshold extraction is meaningful only if you have trials that
      bracket the onset transition (some inactive below onset and some active at/above onset).
    - Under block, you can still estimate onset threshold if low-end bracketing exists.

    - The outputs are differentiable w.r.t. V (and thus w.r.t. upstream simulator parameters).
    - Grouping by pulse width is discrete; do not expect gradients w.r.t. pws_ms grouping.
    - The `amplitudes`/`strength` are typically constants; threshold extraction does not require
      gradients w.r.t. these design variables.

    Spike scoring is a soft max over time and the selected compartments. If your goal is
    *propagating* activation, choose a `node_mask` that reflects the observation site(s)
    where propagation is assessed (often distal nodes/compartments). Otherwise, the surrogate
    may report activation even if propagation fails downstream.

    - If block can occur, prefer:
      - ``threshold_method="onset"`` or ``"onset_midpoint"`` (default)
      - avoid ``"bracket_midpoint"``
      - avoid monotonicity regularization; use unimodality regularization if desired.

    Examples
    --------
    Basic usage (robust onset midpoint thresholds):

    >>> out = chronaxie_from_trials(
    ...     V, amps, pws_ms,
    ...     node_mask=distal_mask,
    ...     threshold_method="onset_midpoint",
    ... )
    >>> c_ms = out["chronaxie_ms"]
    >>> Ir = out["rheobase"]

    Cathodic-negative amplitudes (stronger means more negative):
    pass strength=-amps so strength increases with stimulus intensity:

    >>> out = chronaxie_from_trials(
    ...     V, amps, pws_ms,
    ...     strength=-amps,
    ...     threshold_method="onset_midpoint",
    ... )

    Non-monotone activation due to high-amplitude block:
    extract onset threshold but also inspect block boundary diagnostics:

    >>> out = chronaxie_from_trials(
    ...     V, amps, pws_ms,
    ...     threshold_method="onset_midpoint",
    ...     compute_block=True,
    ... )
    >>> onset = out["boundaries"]["I_on"]
    >>> block = out["boundaries"]["I_block_mid"]

    Add shape prior that allows 0→1→0 but discourages oscillations:

    >>> out = chronaxie_from_trials(
    ...     V, amps, pws_ms,
    ...     threshold_method="onset_midpoint",
    ...     reg_unimodal_weight=1e-2,
    ... )
    >>> loss = task_loss(out["chronaxie_ms"]) + out["reg"]

    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, P, C).")
    T, P, C = V.shape
    device, dtype = V.device, V.dtype

    amplitudes = amplitudes.to(device=device, dtype=dtype)
    pws_ms = pws_ms.to(device=device, dtype=dtype)
    if amplitudes.shape != (P,) or pws_ms.shape != (P,):
        raise ValueError("amplitudes and pws_ms must have shape (P,).")

    # Strength variable used for thresholding / x-axis; must be monotone with "strongness"
    if strength is None:
        S = amplitudes
    else:
        S = strength.to(device=device, dtype=dtype)
        if S.shape != (P,):
            raise ValueError("strength must have shape (P,).")
    if use_abs_strength:
        S = S.abs()

    # Time window
    Vw = V[slice(*time_window)] if time_window is not None else V

    # Compartment mask/weights
    if node_mask is None:
        comp_w = torch.ones((C,), device=device, dtype=dtype)
    else:
        m = node_mask.to(device=device)
        if m.shape != (C,):
            raise ValueError("node_mask must have shape (C,).")
        comp_w = m.to(dtype=dtype) if m.dtype != torch.bool else m.to(dtype=dtype)
    comp_w = torch.clamp(comp_w, min=0.0)
    logw = torch.log(torch.clamp(comp_w, min=eps))

    # Smooth activation confidence per trial
    Z = Vw - V_spk
    logits = kappa_V * Z + logw[None, None, :]
    score = torch.logsumexp(logits, dim=(0, 2)) / kappa_V
    p_spike_trial = torch.sigmoid(score / gate_V_scale)  # (P,)

    # Group pulse widths
    if pw_round_decimals is not None:
        factor = float(10**pw_round_decimals)
        pws_group = torch.round(pws_ms * factor) / factor
    else:
        pws_group = pws_ms

    pw_unique_ms, pw_group_id, pw_counts = torch.unique(
        pws_group, sorted=True, return_inverse=True, return_counts=True
    )
    D = int(pw_unique_ms.numel())
    if D < 2:
        raise ValueError("Need at least 2 distinct pulse widths to estimate chronaxie.")

    if enforce_min_trials_per_pw and int(pw_counts.min().item()) < min_trials_per_pw:
        raise ValueError(
            f"Some PW groups have < {min_trials_per_pw} trials. counts={pw_counts.tolist()}"
        )

    def _soft_pre_inactive(
        Sg: torch.Tensor, pg: torch.Tensor, I_on: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft estimate of 'highest inactive below I_on' robust to block.

        Uses gap g = I_on - Sg and selects the smallest positive gap among inactive points
        (instead of selecting max inactive strength globally).
        """
        Smin, Smax = Sg.min(), Sg.max()
        Srange = (Smax - Smin).clamp_min(eps)

        g = I_on - Sg  # positive if below onset
        inact = torch.clamp(1.0 - pg, min=eps, max=1.0 - eps)
        log_inact = membership_power * torch.log(inact)

        gate_below = torch.sigmoid(
            g / (below_gate_frac * Srange + eps)
        )  # suppress S > I_on
        err = (g / (Srange + eps)) ** 2
        w = torch.softmax(
            -gap_alpha * err + log_inact + torch.log(torch.clamp(gate_below, min=eps)),
            dim=0,
        )
        g_hat = (w * g).sum()
        return I_on - g_hat

    def _soft_post_inactive(
        Sg: torch.Tensor, pg: torch.Tensor, I_last: torch.Tensor
    ) -> torch.Tensor:
        """Soft estimate of 'lowest inactive above I_last' (for block boundary diagnostics)."""
        Smin, Smax = Sg.min(), Sg.max()
        Srange = (Smax - Smin).clamp_min(eps)

        g = Sg - I_last  # positive if above last active
        inact = torch.clamp(1.0 - pg, min=eps, max=1.0 - eps)
        log_inact = membership_power * torch.log(inact)

        gate_above = torch.sigmoid(g / (below_gate_frac * Srange + eps))
        err = (g / (Srange + eps)) ** 2
        w = torch.softmax(
            -gap_alpha * err + log_inact + torch.log(torch.clamp(gate_above, min=eps)),
            dim=0,
        )
        g_hat = (w * g).sum()
        return I_last + g_hat

    I_th_list, I_on_list, I_pre_list = [], [], []
    I_last_list, I_post_list, I_block_mid_list = [], [], []
    pw_weight_list = []

    reg = V.new_zeros(())
    bracket_terms, mono_terms, unimodal_terms = [], [], []

    for g in range(D):
        idx = torch.where(pw_group_id == g)[0]
        Sg = S[idx]
        pg = p_spike_trial[idx]

        # Sort by strength for diagnostics/regularizers
        if idx.numel() > 1:
            perm = torch.argsort(Sg)
            pg_sort = pg[perm]
        else:
            pg_sort = pg

        # Sharpen membership so "inactive" really excludes p≈1 points (and vice versa)
        act = torch.clamp(pg, min=eps, max=1.0 - eps)
        log_act = membership_power * torch.log(act)

        # Normalize strength within group for stable extreme selection
        Smin, Smax = Sg.min(), Sg.max()
        Srange = (Smax - Smin).clamp_min(eps)
        Sunit = (Sg - Smin) / Srange

        # Lower boundary (onset): soft-min of strength among active membership
        w_on = torch.softmax(alpha_extreme * (-Sunit) + log_act, dim=0)
        I_on = (w_on * Sg).sum()

        # Optional upper boundary (block diagnostics): soft-max of strength among active membership
        w_last = torch.softmax(alpha_extreme * (Sunit) + log_act, dim=0)
        I_last = (w_last * Sg).sum()

        # Nearest inactive below onset (robust to blocked high-amplitude inactives)
        I_pre = _soft_pre_inactive(Sg, pg, I_on)

        if compute_block:
            I_post = _soft_post_inactive(Sg, pg, I_last)
            I_block_mid = 0.5 * (I_last + I_post)
        else:
            I_post = torch.tensor(float("nan"), device=device, dtype=dtype)
            I_block_mid = torch.tensor(float("nan"), device=device, dtype=dtype)

        # Choose I_th(d)
        if threshold_method == "onset":
            I_th = I_on
        elif threshold_method == "onset_midpoint":
            I_th = 0.5 * (I_pre + I_on)
        elif threshold_method == "ptarget":
            err = (pg - p_target) ** 2
            wA = torch.softmax(-alpha_thresh * err, dim=0)
            I_th = (wA * Sg).sum()
        elif threshold_method == "bracket_midpoint":
            # Historical behavior: can be wrong under block
            inact = torch.clamp(1.0 - pg, min=eps, max=1.0 - eps)
            log_inact = membership_power * torch.log(inact)
            w_low = torch.softmax(
                alpha_extreme * (Sunit) + log_inact, dim=0
            )  # global max inactive (bad under block)
            I_low = (w_low * Sg).sum()
            I_th = 0.5 * (I_low + I_on)
        else:
            raise ValueError(f"Unknown threshold_method={threshold_method!r}")

        I_th_list.append(I_th)
        I_on_list.append(I_on)
        I_pre_list.append(I_pre)
        I_last_list.append(I_last)
        I_post_list.append(I_post)
        I_block_mid_list.append(I_block_mid)

        # Reliability weight: want low-end inactive AND at least one active anywhere (block-safe)
        p_at_min = pg_sort[0]
        p_any_active = pg_sort.max()
        tau = 0.05
        w_pw = torch.sigmoid((p_low - p_at_min) / tau) * torch.sigmoid(
            (p_any_active - p_high) / tau
        )
        pw_weight_list.append(w_pw)

        # Optional bracketing reg (block-safe definition)
        if reg_bracket_weight > 0.0:
            bracket_terms.append(
                F.softplus(p_at_min - p_low) + F.softplus(p_high - p_any_active)
            )

        # Optional monotonic reg (NOT appropriate if you expect block)
        if reg_monotone_weight > 0.0 and pg_sort.numel() >= 2:
            dp = pg_sort[1:] - pg_sort[:-1]
            mono_terms.append(F.softplus(-dp).mean())

        # Optional unimodal reg (allows 0->1->0; discourages oscillations)
        if reg_unimodal_weight > 0.0 and pg_sort.numel() >= 3:
            idxs = torch.arange(pg_sort.numel(), device=device, dtype=dtype)
            w_peak = torch.softmax(unimodal_beta_peak * pg_sort, dim=0)
            k_hat = (w_peak * idxs).sum()
            dp = pg_sort[1:] - pg_sort[:-1]
            i = torch.arange(dp.numel(), device=device, dtype=dtype)
            before = torch.sigmoid((k_hat - i) / unimodal_tau_idx)
            after = 1.0 - before
            unimodal_terms.append(
                (before * F.softplus(-dp) + after * F.softplus(dp)).mean()
            )

    I_th = torch.stack(I_th_list)  # (D,)
    Q_th = pw_unique_ms * I_th  # (D,)

    I_on = torch.stack(I_on_list)
    I_pre = torch.stack(I_pre_list)
    I_last = torch.stack(I_last_list)
    I_post = torch.stack(I_post_list)
    I_block_mid = torch.stack(I_block_mid_list)

    w_pw = torch.stack(pw_weight_list)  # (D,)

    # Weighted Weiss fit: Q_th(d) ≈ I_r*d + I_r*c
    W = w_pw.sum() + eps
    d = pw_unique_ms
    d_bar = (w_pw * d).sum() / W
    Q_bar = (w_pw * Q_th).sum() / W
    cov = (w_pw * (d - d_bar) * (Q_th - Q_bar)).sum() / W
    var = (w_pw * (d - d_bar) ** 2).sum() / W
    slope = cov / (var + eps)  # rheobase
    intercept = Q_bar - slope * d_bar  # rheobase * chronaxie
    chronaxie_ms = intercept / (slope + eps)

    Q_pred = slope * d + intercept
    weiss_mse = (w_pw * (Q_th - Q_pred) ** 2).sum() / W

    if reg_weiss_fit_weight > 0.0:
        reg = reg + reg_weiss_fit_weight * weiss_mse
    if reg_bracket_weight > 0.0 and bracket_terms:
        reg = reg + reg_bracket_weight * torch.stack(bracket_terms).mean()
    if reg_monotone_weight > 0.0 and mono_terms:
        reg = reg + reg_monotone_weight * torch.stack(mono_terms).mean()
    if reg_unimodal_weight > 0.0 and unimodal_terms:
        reg = reg + reg_unimodal_weight * torch.stack(unimodal_terms).mean()

    return {
        "chronaxie_ms": chronaxie_ms,
        "rheobase": slope,
        "pw_unique_ms": pw_unique_ms,
        "I_th": I_th,
        "Q_th": Q_th,
        "p_spike_trial": p_spike_trial,
        "pw_group_id": pw_group_id,
        "pw_weight": w_pw,
        "boundaries": {
            "I_on": I_on,  # lowest-active (soft)
            "I_pre_inactive": I_pre,  # highest-inactive below onset (soft, block-safe)
            "I_last_active": I_last,  # highest-active (soft)
            "I_post_inactive": I_post,  # lowest-inactive above last active (soft)
            "I_block_mid": I_block_mid,  # midpoint block boundary (diagnostic)
        },
        "fit": {"slope": slope, "intercept": intercept, "weiss_mse": weiss_mse},
        "reg": reg,
    }


def plot_activation_heatmap_from_chronaxie_output(
    chron_out: Dict[str, Any],
    amplitudes: ArrayLike1D,
    *,
    # What to visualize in the heatmap
    mode: Literal["binary", "prob"] = "binary",
    p_thresh: float = 0.5,
    agg: Literal["mean", "max", "min"] = "mean",
    # Optional discretization of amplitude levels
    amp_round_decimals: Optional[int] = None,
    # Plot configuration
    ax=None,
    show_colorbar: bool = True,
    cmap=None,
    title: Optional[str] = None,
    xlabel: str = "Pulse width (ms)",
    ylabel: str = "Amplitude",
    # Tick density controls (None -> show all)
    max_xticks: Optional[int] = 20,
    max_yticks: Optional[int] = 20,
    # ---- Overlay: inferred threshold curve (strength-duration curve) ----
    overlay_threshold: bool = True,
    threshold_kind: Literal["I_th", "fit", "both"] = "I_th",
    overlay_points: bool = True,
    overlay_line: bool = True,
    overlay_bracket_band: bool = False,
    # Style kwargs for overlays
    threshold_line_kwargs: Optional[Dict[str, Any]] = None,
    threshold_point_kwargs: Optional[Dict[str, Any]] = None,
    bracket_line_kwargs: Optional[Dict[str, Any]] = None,
    bracket_fill_kwargs: Optional[Dict[str, Any]] = None,
    # ---- Overlay: empirical threshold points inferred from the heatmap grid ----
    overlay_empirical_threshold: bool = False,
    empirical_kind: Literal[
        "lowest_active", "highest_inactive", "midpoint", "all"
    ] = "all",
    empirical_use_thresholded_prob_in_prob_mode: bool = True,
    empirical_low_kwargs: Optional[Dict[str, Any]] = None,
    empirical_high_kwargs: Optional[Dict[str, Any]] = None,
    empirical_mid_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Plot an activation heatmap (activated vs non-activated) vs amplitude and pulse width,
    with optional overlays for (i) estimated strength-duration threshold curves and (ii)
    empirical threshold points derived directly from the binned heatmap.

    This helper is intended to work with the dictionary output from
    ``chronaxie_from_trials(...)`` (or a compatible estimator that returns
    per-trial activation confidences and pulse-width grouping information).

    The plot is constructed by binning trials onto a 2D grid:

    - x-axis: pulse width (ms)
    - y-axis: stimulus amplitude (or a monotone "strength" variable you pass as `amplitudes`)
    - cell value: activation (binary) or activation confidence (probability-like)

    The plot include various overlays:

    1) **Estimated threshold curve overlay** (`overlay_threshold=True`)

       - `threshold_kind="I_th"` overlays `chron_out["I_th"]` vs `chron_out["pw_unique_ms"]`
       - `threshold_kind="fit"` overlays the Lapicque curve computed from
         `chron_out["rheobase"]` and `chron_out["chronaxie_ms"]`:
           I_fit(d) = I_r * (1 + c/d)
       - `threshold_kind="both"` overlays both, if available

       If `overlay_bracket_band=True` and `chron_out["bracket"]` exists with keys
       `"I_low"` and `"I_high"`, the function can optionally overlay those bracket lines
       and a filled band between them.

    2) **Empirical threshold points overlay** (`overlay_empirical_threshold=True`)

       Empirical thresholds are derived *directly* from the binned heatmap grid, per pulse width:

       - **lowest_active(d)**: the smallest amplitude level in that PW column with activation == 1
       - **highest_inactive(d)**: the largest amplitude level with activation == 0
       - **midpoint(d)**: 0.5*(lowest_active + highest_inactive), when both exist

       These are useful to visually compare the estimator output (I_th / fit) against the
       “data-derived” boundary.

       - This assumes that “activation likelihood increases monotonically with amplitude/strength”
         within each pulse-width column. If your design violates this, empirical thresholds may be
         ambiguous (you can use the monotonicity regularizer during fitting, or inspect manually).
       - If a column has only active trials or only inactive trials (or missing data), the
         corresponding empirical threshold for that column will be undefined (NaN) and not plotted.

    Parameters
    ----------
    chron_out : dict
        Output dictionary from ``chronaxie_from_trials`` (or equivalent).
        Required keys:

        - ``"p_spike_trial"`` : torch.Tensor, shape (P,)
            Smooth activation confidence per trial in (0, 1).

        - ``"pw_group_id"`` : torch.Tensor, shape (P,)
            Integer group id mapping each trial to a pulse-width index.

        - ``"pw_unique_ms"`` : torch.Tensor, shape (D,)
            Unique pulse widths (ms), sorted ascending.

        Optional keys used for estimated-threshold overlays:

        - ``"I_th"`` : torch.Tensor, shape (D,)
        - ``"rheobase"`` : torch.Tensor scalar (or float)
        - ``"chronaxie_ms"`` : torch.Tensor scalar (or float)
        - ``"bracket"`` : dict with keys ``"I_low"``, ``"I_high"`` (each shape (D,))

    amplitudes : array-like, shape (P,)
        Per-trial amplitude values aligned with the trials used to compute `chron_out`.

        **Important sign/strength convention**
        If your chronaxie estimator used a different monotone “strength” variable than raw
        amplitudes (e.g., `strength = -amplitudes` for cathodic negative pulses), you should
        pass that same strength array here. The heatmap y-axis and all overlays assume the
        same coordinate system.

    mode : {"binary", "prob"}, default="binary"
        Heatmap cell values:

        - ``"binary"``: after aggregating p_spike within each (PW, amplitude) cell,
          mark activated if >= `p_thresh`.
        - ``"prob"``: show aggregated p_spike in [0, 1].

    p_thresh : float, default=0.5
        Threshold used for binary heatmap mode, and also for empirical threshold extraction
        when `mode="prob"` and `empirical_use_thresholded_prob_in_prob_mode=True`.

    agg : {"mean", "max", "min"}, default="mean"
        Aggregation rule when multiple trials map to the same (PW, amplitude) cell.

    amp_round_decimals : int, optional
        If provided, amplitudes are grouped after rounding to this number of decimals.

    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure/axes is created.

    show_colorbar : bool, default=True
        If True, add a colorbar.

    cmap : matplotlib colormap, optional
        Colormap passed to `pcolormesh`. If None, matplotlib default is used.

    title : str, optional
        Plot title. If None, a default is chosen from `mode`.

    xlabel, ylabel : str
        Axis labels.

    max_xticks, max_yticks : int, optional
        Maximum number of tick labels; values are subsampled for readability when exceeded.

    overlay_threshold : bool, default=True
        Overlay estimator-derived threshold curve(s).

    threshold_kind : {"I_th", "fit", "both"}, default="I_th"
        Which estimator-derived threshold curve(s) to overlay.

    overlay_points : bool, default=True
        Plot threshold points at each pulse width.

    overlay_line : bool, default=True
        Connect threshold points with a line.

    overlay_bracket_band : bool, default=False
        Overlay the `bracket` band's I_low/I_high lines and fill between them if available.

    threshold_line_kwargs, threshold_point_kwargs, bracket_line_kwargs, bracket_fill_kwargs : dict, optional
        Matplotlib kwargs for overlay artists. Colors are not set explicitly by default.

    overlay_empirical_threshold : bool, default=False
        If True, overlay empirical threshold points derived from the binned heatmap.

    empirical_kind : {"lowest_active", "highest_inactive", "midpoint", "all"}, default="all"
        Which empirical threshold points to overlay.

    empirical_use_thresholded_prob_in_prob_mode : bool, default=True
        If `mode="prob"`, compute empirical thresholds by thresholding the aggregated probabilities
        at `p_thresh` (i.e., treat >=p_thresh as active). If False, empirical overlay is skipped
        in prob mode unless `mode="binary"`.

    empirical_low_kwargs, empirical_high_kwargs, empirical_mid_kwargs : dict, optional
        Matplotlib kwargs for plotting empirical threshold points:

        - low (lowest_active): defaults to marker '^'
        - high (highest_inactive): defaults to marker 'v'
        - mid (midpoint): defaults to marker 'x'

        Colors are not set explicitly by default.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    ax : matplotlib.axes.Axes
        Axes handle.

    mesh : matplotlib.collections.QuadMesh
        Result of `ax.pcolormesh`.

    grid : torch.Tensor
        The plotted grid, shape (A_unique, D). Values are NaN where no trials exist for a cell.

    amp_unique : torch.Tensor
        Sorted unique amplitude levels used for y-axis binning, shape (A_unique,).

    pw_unique_ms : torch.Tensor
        Sorted unique pulse widths used for x-axis binning, shape (D,).

    Notes
    -----
    - This is a visualization utility; tensors are detached and moved to CPU.
    - If your trials do not form a full Cartesian grid, missing cells remain NaN and are masked.
    - For non-uniform amplitude/PW spacing, `pcolormesh` with computed bin edges preserves geometry.

    """
    # Local import to keep plotting dependency optional for core code
    import matplotlib.pyplot as plt

    # ----------------------------
    # Extract required fields
    # ----------------------------
    required = ("p_spike_trial", "pw_group_id", "pw_unique_ms")
    for k in required:
        if k not in chron_out:
            raise KeyError(f"chron_out must contain key '{k}'.")

    p_spike = chron_out["p_spike_trial"]
    pw_group_id = chron_out["pw_group_id"]
    pw_unique_ms = chron_out["pw_unique_ms"]

    if not isinstance(p_spike, torch.Tensor):
        raise TypeError("chron_out['p_spike_trial'] must be a torch.Tensor.")
    if not isinstance(pw_group_id, torch.Tensor):
        raise TypeError("chron_out['pw_group_id'] must be a torch.Tensor.")
    if not isinstance(pw_unique_ms, torch.Tensor):
        raise TypeError("chron_out['pw_unique_ms'] must be a torch.Tensor.")

    # Move to CPU for plotting
    p_spike = p_spike.detach().cpu()
    pw_group_id = pw_group_id.detach().cpu().to(torch.int64)
    pw_unique_ms = pw_unique_ms.detach().cpu()

    # amplitudes -> CPU tensor
    amps = torch.as_tensor(amplitudes).detach().cpu()
    if amps.ndim != 1:
        amps = amps.reshape(-1)

    P = int(p_spike.numel())
    if amps.numel() != P or pw_group_id.numel() != P:
        raise ValueError(
            f"Mismatched P: p_spike has {P}, amplitudes has {amps.numel()}, pw_group_id has {pw_group_id.numel()}."
        )

    D = int(pw_unique_ms.numel())
    if D < 1:
        raise ValueError("pw_unique_ms is empty.")
    if pw_group_id.min() < 0 or pw_group_id.max() >= D:
        raise ValueError("pw_group_id contains indices outside [0, D-1].")

    # ----------------------------
    # Group amplitudes (optional rounding)
    # ----------------------------
    if amp_round_decimals is not None:
        factor = float(10**amp_round_decimals)
        amps_group = torch.round(amps * factor) / factor
    else:
        amps_group = amps

    amp_unique, amp_inv = torch.unique(amps_group, sorted=True, return_inverse=True)
    Auniq = int(amp_unique.numel())

    # ----------------------------
    # Build grid (Auniq x D) with NaNs
    # ----------------------------
    grid = torch.full((Auniq, D), float("nan"), dtype=torch.float32)

    row = amp_inv.to(torch.int64)
    col = pw_group_id
    bin_idx = row * D + col  # (P,)

    if agg == "mean":
        sums = torch.zeros((Auniq * D,), dtype=torch.float32)
        cnts = torch.zeros((Auniq * D,), dtype=torch.float32)
        sums.scatter_add_(0, bin_idx, p_spike.to(torch.float32))
        cnts.scatter_add_(0, bin_idx, torch.ones_like(p_spike, dtype=torch.float32))
        mean = sums / torch.clamp(cnts, min=1.0)
        mean[cnts == 0] = float("nan")
        grid = mean.reshape(Auniq, D)

    elif agg in ("max", "min"):
        if agg == "max":
            init = float("-inf")
            reduce_fn = max
        else:
            init = float("inf")
            reduce_fn = min

        vals = torch.full((Auniq * D,), init, dtype=torch.float32)
        seen = torch.zeros((Auniq * D,), dtype=torch.bool)

        for i in range(P):
            b = int(bin_idx[i])
            v = float(p_spike[i])
            if not seen[b]:
                vals[b] = torch.tensor(v, dtype=torch.float32)
                seen[b] = True
            else:
                vals[b] = torch.tensor(
                    reduce_fn(float(vals[b].item()), v), dtype=torch.float32
                )

        vals[~seen] = float("nan")
        grid = vals.reshape(Auniq, D)

    else:
        raise ValueError("agg must be one of {'mean', 'max', 'min'}.")

    # ----------------------------
    # Convert to plot values
    # ----------------------------
    if mode == "binary":
        plot_grid = (grid >= p_thresh).to(torch.float32)
        plot_grid[torch.isnan(grid)] = float("nan")
        default_title = f"Activation (>= {p_thresh:g})"
    elif mode == "prob":
        plot_grid = grid
        default_title = "Activation probability"
    else:
        raise ValueError("mode must be one of {'binary', 'prob'}.")

    # ----------------------------
    # Compute bin edges for pcolormesh
    # ----------------------------
    def _bin_edges(vals_1d: np.ndarray) -> np.ndarray:
        """Compute edges from sorted centers."""
        n = vals_1d.size
        if n == 1:
            v0 = float(vals_1d[0])
            span = 0.5 * (abs(v0) + 1.0)  # ensures nonzero width even if v0==0
            return np.array([v0 - span, v0 + span], dtype=float)
        mids = 0.5 * (vals_1d[1:] + vals_1d[:-1])
        first = vals_1d[0] - (mids[0] - vals_1d[0])
        last = vals_1d[-1] + (vals_1d[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]]).astype(float)

    x_centers = pw_unique_ms.numpy().astype(float)
    y_centers = amp_unique.numpy().astype(float)

    x_edges = _bin_edges(x_centers)  # (D+1,)
    y_edges = _bin_edges(y_centers)  # (Auniq+1,)

    # ----------------------------
    # Plot (pcolormesh + masked NaNs)
    # ----------------------------
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    arr = plot_grid.numpy().astype(float)
    arr_masked = np.ma.masked_invalid(arr)

    mesh = ax.pcolormesh(
        x_edges, y_edges, arr_masked, shading="auto", vmin=0.0, vmax=1.0, cmap=cmap
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(default_title if title is None else title)

    # Ticks (subsample for readability)
    if max_xticks is None or D <= max_xticks:
        xtick_vals = x_centers
    else:
        idx = np.unique(np.linspace(0, D - 1, max_xticks).round().astype(int))
        xtick_vals = x_centers[idx]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels([f"{v:g}" for v in xtick_vals], rotation=45, ha="right")

    if max_yticks is None or Auniq <= max_yticks:
        ytick_vals = y_centers
    else:
        idx = np.unique(np.linspace(0, Auniq - 1, max_yticks).round().astype(int))
        ytick_vals = y_centers[idx]
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels([f"{v:g}" for v in ytick_vals])

    if show_colorbar:
        fig.colorbar(mesh, ax=ax)

    # ----------------------------
    # Overlay threshold curve(s) from the estimator
    # ----------------------------
    if overlay_threshold:
        if threshold_line_kwargs is None:
            threshold_line_kwargs = {"linewidth": 2}
        if threshold_point_kwargs is None:
            threshold_point_kwargs = {
                "marker": "o",
                "linestyle": "None",
                "markersize": 4,
            }
        if bracket_line_kwargs is None:
            bracket_line_kwargs = {"linestyle": "--", "linewidth": 1.5}
        if bracket_fill_kwargs is None:
            bracket_fill_kwargs = {"alpha": 0.15}

        pw_x = pw_unique_ms.detach().cpu().numpy().astype(float)

        def _overlay_curve(y_vals: np.ndarray, label: Optional[str] = None):
            y_vals = np.asarray(y_vals, dtype=float)
            ok = np.isfinite(pw_x) & np.isfinite(y_vals)
            if overlay_line and ok.sum() >= 2:
                ax.plot(pw_x[ok], y_vals[ok], label=label, **threshold_line_kwargs)
            if overlay_points and ok.sum() >= 1:
                ax.plot(
                    pw_x[ok],
                    y_vals[ok],
                    label=None if overlay_line else label,
                    **threshold_point_kwargs,
                )

        if threshold_kind in ("I_th", "both"):
            if "I_th" not in chron_out:
                raise KeyError(
                    "threshold_kind includes 'I_th' but chron_out has no key 'I_th'."
                )
            I_th = chron_out["I_th"]
            if isinstance(I_th, torch.Tensor):
                I_th = I_th.detach().cpu().numpy()
            _overlay_curve(I_th, label="I_th")

        if threshold_kind in ("fit", "both"):
            if "rheobase" not in chron_out or "chronaxie_ms" not in chron_out:
                raise KeyError(
                    "threshold_kind includes 'fit' but chron_out does not contain both "
                    "'rheobase' and 'chronaxie_ms'."
                )
            Ir = chron_out["rheobase"]
            c = chron_out["chronaxie_ms"]
            Ir = (
                float(Ir.detach().cpu().item())
                if isinstance(Ir, torch.Tensor)
                else float(Ir)
            )
            c = (
                float(c.detach().cpu().item())
                if isinstance(c, torch.Tensor)
                else float(c)
            )

            d = pw_x
            I_fit = Ir * (1.0 + (c / np.clip(d, 1e-12, None)))
            _overlay_curve(I_fit, label="fit")

        if overlay_bracket_band and "bracket" in chron_out:
            br = chron_out["bracket"]
            if not isinstance(br, dict) or ("I_low" not in br) or ("I_high" not in br):
                raise KeyError(
                    "chron_out['bracket'] must be a dict with keys 'I_low' and 'I_high'."
                )

            I_low = br["I_low"]
            I_high = br["I_high"]
            if isinstance(I_low, torch.Tensor):
                I_low = I_low.detach().cpu().numpy()
            if isinstance(I_high, torch.Tensor):
                I_high = I_high.detach().cpu().numpy()

            I_low = np.asarray(I_low, dtype=float)
            I_high = np.asarray(I_high, dtype=float)
            ok = np.isfinite(pw_x) & np.isfinite(I_low) & np.isfinite(I_high)

            if ok.sum() >= 2:
                ax.plot(pw_x[ok], I_low[ok], **bracket_line_kwargs)
                ax.plot(pw_x[ok], I_high[ok], **bracket_line_kwargs)
                ax.fill_between(pw_x[ok], I_low[ok], I_high[ok], **bracket_fill_kwargs)

    # ----------------------------
    # Overlay empirical thresholds derived from the binned grid
    # ----------------------------
    if overlay_empirical_threshold:
        # Determine a binary grid for empirical extraction
        if mode == "binary":
            binary_grid = plot_grid  # already 0/1 with NaNs
        else:
            if not empirical_use_thresholded_prob_in_prob_mode:
                binary_grid = None
            else:
                binary_grid = (grid >= p_thresh).to(torch.float32)
                binary_grid[torch.isnan(grid)] = float("nan")

        if binary_grid is not None:
            # Defaults for markers (no explicit colors)
            if empirical_low_kwargs is None:
                empirical_low_kwargs = {
                    "marker": "^",
                    "linestyle": "None",
                    "markersize": 5,
                }
            if empirical_high_kwargs is None:
                empirical_high_kwargs = {
                    "marker": "v",
                    "linestyle": "None",
                    "markersize": 5,
                }
            if empirical_mid_kwargs is None:
                empirical_mid_kwargs = {
                    "marker": "x",
                    "linestyle": "None",
                    "markersize": 5,
                }

            pw_x = pw_unique_ms.detach().cpu().numpy().astype(float)

            # Per-PW empirical thresholds
            emp_low = np.full((D,), np.nan, dtype=float)  # lowest activating amplitude
            emp_high = np.full(
                (D,), np.nan, dtype=float
            )  # highest non-activating amplitude
            emp_mid = np.full((D,), np.nan, dtype=float)  # midpoint between them

            # amp_unique is sorted ascending
            for j in range(D):
                colj = binary_grid[:, j]  # (Auniq,)
                finite = torch.isfinite(colj)
                if not finite.any():
                    continue

                act = (colj == 1.0) & finite
                ina = (colj == 0.0) & finite

                if act.any():
                    emp_low[j] = float(amp_unique[act].min().item())
                if ina.any():
                    emp_high[j] = float(amp_unique[ina].max().item())
                if np.isfinite(emp_low[j]) and np.isfinite(emp_high[j]):
                    emp_mid[j] = 0.5 * (emp_low[j] + emp_high[j])

            def _plot_emp(y: np.ndarray, label: str, kwargs: Dict[str, Any]):
                ok = np.isfinite(pw_x) & np.isfinite(y)
                if ok.sum() >= 1:
                    ax.plot(pw_x[ok], y[ok], label=label, **kwargs)

            if empirical_kind in ("lowest_active", "all"):
                _plot_emp(emp_low, "empirical lowest active", empirical_low_kwargs)
            if empirical_kind in ("highest_inactive", "all"):
                _plot_emp(emp_high, "empirical highest inactive", empirical_high_kwargs)
            if empirical_kind in ("midpoint", "all"):
                _plot_emp(emp_mid, "empirical midpoint", empirical_mid_kwargs)

    # Legend if any labeled artists exist
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        ax.legend()

    fig.tight_layout()
    return fig, ax, mesh, plot_grid, amp_unique, pw_unique_ms


def firing_rate(
    V: torch.Tensor,  # (T, F, C)
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    # Optionally reuse a precomputed per-compartment spike-present confidence
    p_spike: torch.Tensor | None = None,  # (F, C)
    # Spike-state thresholding
    V_spk: float = 0.0,
    gate_V_scale: float = 5.0,
    # Optional dV/dt gate to suppress slow depolarizations
    use_dv_gate: bool = True,
    dv_spk: float = 10.0,
    kappa_V: float = 20.0,
    kappa_dv: float = 10.0,
    gate_dv_scale: float = 5.0,
    # Optional time restriction, applied before counting
    time_window: tuple[int, int] | None = None,
    # Robustness / regularization
    min_ess: float = 1.0,
    reg_ess_weight: float = 0.0,
    eps: float = 1e-6,
    confidence_weighted: bool = False,
):
    r"""
    Differentiable firing-rate surrogate in Hz.

    For each compartment:
        s[t] = sigmoid((V[t] - V_spk) / gate_V_scale)

    Then count soft upward crossings:
        up[t] = relu(s[t+1] - s[t])

    For a full 0->1->0 excursion, sum_t up[t] is approximately 1, so each AP
    contributes about one count. Optionally, `up` is multiplied by a smooth dV/dt
    gate to reject slow threshold crossings.

    Returns
    -------
    out : dict[str, torch.Tensor]
        "rate_hz"      : (F,)   weighted per-fiber firing-rate surrogate
        "rate_hz_comp" : (F, C) per-compartment firing-rate surrogate
        "count"        : (F,)   weighted soft spike count over the window
        "count_comp"   : (F, C) per-compartment soft spike count
        "p_spike"      : (F, C) spike-present confidence used for weighting
        "weights"      : (F, C) final aggregation weights
        "ess"          : (F,)   effective sample size of weights
        "window_ms"    : scalar analyzed duration in ms
        "reg"          : scalar optional regularizer
    """

    assert V.ndim == 3, "V must be (T, F, C)"
    T, Fibs, C = V.shape
    device, dtype = V.device, V.dtype

    if T < 2:
        raise ValueError("Need at least 2 time samples to estimate firing rate.")

    if not torch.is_tensor(dt_ms):
        dt_ms = torch.tensor(dt_ms, device=device, dtype=dtype)
    else:
        dt_ms = dt_ms.to(device=device, dtype=dtype)

    Vw = V[slice(*time_window)] if time_window is not None else V
    Tw = Vw.shape[0]
    if Tw < 2:
        raise ValueError("Selected time window must contain at least 2 samples.")

    if node_mask is None:
        nm = torch.ones((Fibs, C), device=device, dtype=dtype)
    else:
        nm = node_mask.to(device=device, dtype=dtype)
        assert nm.shape == (Fibs, C), "node_mask must be (F, C)"

    dV = (Vw[1:] - Vw[:-1]) / dt_ms  # (Tw-1, F, C), mV/ms

    # Smooth compartment-level spike-present confidence, same style as the other surrogates
    if p_spike is None:
        a = torch.logsumexp(kappa_V * (Vw - V_spk), dim=0) / kappa_V  # (F, C)
        p_spike = torch.sigmoid(a / gate_V_scale)

        if use_dv_gate:
            u = torch.logsumexp(kappa_dv * (dV - dv_spk), dim=0) / kappa_dv  # (F, C)
            p_dv = torch.sigmoid(u / gate_dv_scale)
            p_spike = p_spike * p_dv
    else:
        p_spike = p_spike.to(device=device, dtype=dtype)
        assert p_spike.shape == (Fibs, C), "p_spike must be (F, C)"

    # Smooth spike-state occupancy in (0, 1)
    s = torch.sigmoid((Vw - V_spk) / gate_V_scale)  # (Tw, F, C)

    # Positive changes in occupancy. A full 0->1 excursion contributes ~1.
    ds = s[1:] - s[:-1]  # (Tw-1, F, C)
    up = torch.relu(ds)

    # Optional upstroke gate to reject slow threshold crossings / depolarizations.
    if use_dv_gate:
        g_up = torch.sigmoid((dV - dv_spk) / gate_dv_scale)
        up = up * g_up

    count_comp = up.sum(dim=0)  # (F, C), soft spike count per compartment

    window_ms = (Tw - 1) * dt_ms
    rate_hz_comp = 1000.0 * count_comp / window_ms  # (F, C)

    # Aggregate across compartments in the same style as the other functions
    if confidence_weighted:
        # Weight by p_spike to get a confidence-weighted rate estimate
        weights = nm * p_spike
    else:
        # Just use the node mask for weighting, without p_spike confidence
        weights = nm

    W = weights.sum(dim=-1) + eps
    count = (weights * count_comp).sum(dim=-1) / W  # (F,)
    rate_hz = (weights * rate_hz_comp).sum(dim=-1) / W  # (F,)

    ess = (weights.sum(dim=-1) ** 2) / ((weights * weights).sum(dim=-1) + eps)

    reg = V.new_zeros(())
    if reg_ess_weight > 0.0:
        reg = (
            reg
            + reg_ess_weight
            * F.softplus(torch.tensor(min_ess, device=device, dtype=dtype) - ess).mean()
        )

    return {
        "rate_hz": rate_hz,  # (F,)
        "rate_hz_comp": rate_hz_comp,  # (F, C)
        "count": count,  # (F,)
        "count_comp": count_comp,  # (F, C)
        "p_spike": p_spike,  # (F, C)
        "weights": weights,  # (F, C)
        "ess": ess,  # (F,)
        "window_ms": window_ms,  # scalar
        "reg": reg,  # scalar
    }


def active(
    V: torch.Tensor,  # (T, F, C)
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    # Spike-present confidence from voltages
    V_spk: float = 0.0,
    kappa_V: float = 20.0,
    gate_V_scale: float = 5.0,
    # Optional dV/dt gate to suppress slow depolarizations
    use_dv_gate: bool = True,
    dv_spk: float = 10.0,
    kappa_dv: float = 10.0,
    gate_dv_scale: float = 5.0,
    # Optional time restriction, applied before scoring
    time_window: tuple[int, int] | None = None,
    # Compartment aggregation
    aggregate: Literal["softmax", "soft_or", "mean"] = "softmax",
    # Robustness / regularization
    min_ess: float = 1.0,
    reg_ess_weight: float = 0.0,
    eps: float = 1e-6,
):
    r"""
    Differentiable spike-present / "active" surrogate.

    `active[f]` is a smooth confidence that at least one AP occurred in the
    user-selected readout region of fiber `f` within the analysis window.

    Aggregation options
    -------------------
    softmax:
        Region-level smooth max over time and selected compartments.
        Best default for "did any selected trace spike?"
    soft_or:
        Probabilistic soft-OR over per-compartment activity confidences.
    mean:
        Literal mean local activity confidence over selected compartments.

    Returns
    -------
    out : dict[str, torch.Tensor]
        "active"      : (F,)   aggregated readout-region activity confidence
        "active_comp" : (F, C) per-compartment activity confidence
        "p_spike"     : (F, C) alias for active_comp
        "score_comp"  : (F, C) pre-sigmoid voltage score per compartment
        "weights"     : (F, C) selected compartment weights / mask
        "ess"         : (F,)   effective sample size of the readout weights
        "window_ms"   : scalar analyzed duration in ms
        "reg"         : scalar optional regularizer
    """
    assert V.ndim == 3, "V must be (T, F, C)"
    _, Fibs, C = V.shape
    device, dtype = V.device, V.dtype

    if not torch.is_tensor(dt_ms):
        dt_ms = torch.tensor(dt_ms, device=device, dtype=dtype)
    else:
        dt_ms = dt_ms.to(device=device, dtype=dtype)

    Vw = V[slice(*time_window)] if time_window is not None else V
    Tw = Vw.shape[0]
    if Tw < 1:
        raise ValueError("Selected time window must contain at least 1 sample.")
    if use_dv_gate and Tw < 2:
        raise ValueError("Need at least 2 time samples when use_dv_gate=True.")

    if node_mask is None:
        nm = torch.ones((Fibs, C), device=device, dtype=dtype)
    else:
        nm = node_mask.to(device=device, dtype=dtype)
        assert nm.shape == (Fibs, C), "node_mask must be (F, C)"
    nm = torch.clamp(nm, min=0.0)

    # Local per-compartment spike-present confidence
    score_comp = torch.logsumexp(kappa_V * (Vw - V_spk), dim=0) / kappa_V  # (F, C)
    active_comp = torch.sigmoid(score_comp / gate_V_scale)

    dV = None
    p_dv_comp = None
    if use_dv_gate:
        dV = (Vw[1:] - Vw[:-1]) / dt_ms
        score_dv_comp = torch.logsumexp(kappa_dv * (dV - dv_spk), dim=0) / kappa_dv
        p_dv_comp = torch.sigmoid(score_dv_comp / gate_dv_scale)
        active_comp = active_comp * p_dv_comp

    # Aggregate over the user-selected readout set
    if aggregate == "mean":
        weights = nm
        W = weights.sum(dim=-1) + eps
        active_region = (weights * active_comp).sum(dim=-1) / W

    elif aggregate == "soft_or":
        weights = nm
        log_not = weights * torch.log(torch.clamp(1.0 - active_comp, min=eps))
        active_region = 1.0 - torch.exp(log_not.sum(dim=-1))

    elif aggregate == "softmax":
        weights = nm
        neg_inf = torch.full_like(weights, -torch.inf)
        logw = torch.where(
            weights > 0,
            torch.log(torch.clamp(weights, min=eps)),
            neg_inf,
        )

        score_region = (
            torch.logsumexp(
                kappa_V * (Vw - V_spk) + logw[None, :, :],
                dim=(0, 2),
            )
            / kappa_V
        )
        active_region = torch.sigmoid(score_region / gate_V_scale)

        if use_dv_gate:
            score_dv_region = (
                torch.logsumexp(
                    kappa_dv * (dV - dv_spk) + logw[None, :, :],
                    dim=(0, 2),
                )
                / kappa_dv
            )
            active_region = active_region * torch.sigmoid(
                score_dv_region / gate_dv_scale
            )

        active_region = torch.where(
            weights.sum(dim=-1) > 0,
            active_region,
            torch.zeros_like(active_region),
        )

    else:
        raise ValueError("aggregate must be one of {'softmax', 'soft_or', 'mean'}.")

    ess = (nm.sum(dim=-1) ** 2) / ((nm * nm).sum(dim=-1) + eps)

    reg = V.new_zeros(())
    if reg_ess_weight > 0.0:
        reg = (
            reg
            + reg_ess_weight
            * F.softplus(torch.tensor(min_ess, device=device, dtype=dtype) - ess).mean()
        )

    window_ms = (
        (Tw - 1) * dt_ms if Tw >= 2 else torch.zeros((), device=device, dtype=dtype)
    )

    return {
        "active": active_region,  # (F,)
        "active_comp": active_comp,  # (F, C)
        "p_spike": active_comp,  # alias for suite consistency
        "p_dv_comp": p_dv_comp,  # (F, C) or None
        "score_comp": score_comp,  # (F, C)
        "weights": weights,  # (F, C)
        "ess": ess,  # (F,)
        "window_ms": window_ms,  # scalar
        "reg": reg,  # scalar
    }


def paired_pulse_recovery_from_trials(
    V: torch.Tensor,  # (T, P, C)
    test_amplitudes: torch.Tensor,  # (P,)
    isis_ms: torch.Tensor,  # (P,)
    dt_ms: float | torch.Tensor,
    *,
    # Timing
    condition_pulse_time_ms: float | torch.Tensor = 0.0,
    test_pulse_times_ms: torch.Tensor
    | None = None,  # optional absolute test times, (P,)
    response_window_ms: tuple[float, float] = (0.1, 10.0),
    gate_t_scale_ms: float = 0.05,
    # Optional normalization / references
    baseline_threshold: float | torch.Tensor | None = None,  # scalar or (D,)
    reference_latency_ms: float | torch.Tensor | None = None,  # scalar or (D,)
    reference_velocity_m_per_s: float | torch.Tensor | None = None,  # scalar or (D,)
    # Optional velocity recovery calculation
    lengths_um: Optional[LengthLike] = None,  # scalar, (C,), or (P, C)
    node_mask: torch.Tensor | None = None,  # (C,) or (P, C)
    # Test-response / spike-present scoring
    V_spk: float = 0.0,
    kappa_V: float = 20.0,
    gate_V_scale: float = 5.0,
    use_dv_gate: bool = True,
    dv_spk: float = 10.0,
    kappa_dv: float = 10.0,
    gate_dv_scale: float = 5.0,
    # Soft arrival-time settings for latency / velocity
    compute_latency: bool = True,
    beta: float = 50.0,
    dv0: float = 0.0,
    dv_scale: float = 1.0,
    lambda_early: float = 0.0,
    min_ess: float = 1.0,
    min_var_ms2: float = 1e-3,
    reg_ess_weight: float = 0.0,
    reg_var_weight: float = 0.0,
    # Strength-axis convention
    strength: torch.Tensor
    | None = None,  # (P,); e.g. -test_amplitudes for cathodic-negative pulses
    use_abs_strength: bool = False,
    # ISI grouping
    isi_round_decimals: int | None = None,
    enforce_min_trials_per_isi: bool = True,
    min_trials_per_isi: int = 2,
    # Threshold extraction per ISI
    threshold_method: Literal[
        "onset", "onset_midpoint", "ptarget", "bracket_midpoint"
    ] = "onset_midpoint",
    p_target: float = 0.5,
    alpha_thresh: float = 200.0,
    alpha_extreme: float = 50.0,
    membership_power: float = 10.0,
    gap_alpha: float = 200.0,
    below_gate_frac: float = 0.02,
    compute_block: bool = True,
    # Group summaries and optional shape priors
    success_aggregate: Literal["mean", "max", "soft_or"] = "mean",
    success_beta: float = 50.0,
    p_low: float = 0.05,
    p_high: float = 0.95,
    reg_bracket_weight: float = 0.0,
    reg_monotone_weight: float = 0.0,
    reg_unimodal_weight: float = 0.0,
    unimodal_beta_peak: float = 50.0,
    unimodal_tau_idx: float = 1.0,
    eps: float = 1e-8,
    return_time_traces: bool = False,
) -> Dict[str, Any]:
    r"""
    Differentiably estimate paired-pulse recovery and threshold recovery-cycle metrics.

    This function implements a smooth surrogate for the electrophysiological paired-pulse
    recovery / threshold recovery-cycle experiment. Each trial should contain a conditioning
    pulse followed by a test pulse. The metric scores the response to the **test pulse only**
    in a pulse-locked differentiable readout window.

    The function supports two common protocols:

    **1. Threshold recovery cycle.**
        For each interstimulus interval (ISI), simulate a sweep over second-pulse/test-pulse
        amplitudes. The first/conditioning pulse is usually fixed and suprathreshold. The input
        tensor therefore contains one trial per ``(ISI, test_amplitude)`` pair. The function
        extracts a differentiable second-pulse threshold for each ISI. If ``baseline_threshold``
        is supplied, it returns

        ``threshold_ratio[isi] = I_th_test[isi] / baseline_threshold``.

        Values greater than 1 indicate elevated threshold / relative refractoriness. Values less
        than 1 indicate supernormal excitability.

    **2. Fixed-amplitude paired-pulse success, latency, or velocity recovery.**
        For each ISI, simulate one or more paired-pulse trials at fixed test-pulse amplitude.
        Threshold extraction is not well bracketed in this mode, but the function still returns
        smooth test-response probabilities, latencies, and optional velocity recovery. Set
        ``enforce_min_trials_per_isi=False`` if there is only one amplitude per ISI.

    Required input data
    -------------------
    V : torch.Tensor
        Voltage recordings with shape ``(T, P, C)``:

        - ``T``: number of time samples
        - ``P``: number of paired-pulse trials
        - ``C``: number of compartments / recording sites

        ``V`` should include both the conditioning response and the test response. The test
        response is scored only within a window relative to the test-pulse time. Units are
        usually mV. Gradients propagate through ``V`` and thus through the upstream simulator.

    test_amplitudes : torch.Tensor
        Test-pulse amplitudes with shape ``(P,)``. These are the second-pulse amplitudes, not
        the conditioning-pulse amplitudes. Units are arbitrary but must match
        ``baseline_threshold`` and any target threshold data.

    isis_ms : torch.Tensor
        Interstimulus intervals with shape ``(P,)``. Trials with the same ISI are grouped to
        define one recovery-cycle point.

    dt_ms : float or torch.Tensor
        Simulation time step in ms.

    Simulations required to produce the input data
    ----------------------------------------------
    For threshold recovery, run a batched grid such as:

    ``for isi in ISIs:``
        ``for amp in test_amplitudes_for_this_isi:``
            simulate a conditioning pulse at ``condition_pulse_time_ms`` and a test pulse at
            ``condition_pulse_time_ms + isi``; store the voltage trace as one trial.

    The conditioning pulse should usually be suprathreshold and fixed across trials. The
    test-pulse amplitude should span subthreshold and suprathreshold responses for each ISI.
    The readout compartments selected by ``node_mask`` should reflect the experimental endpoint:
    use distal compartments for propagation recovery, or local compartments for local
    excitability recovery.

    To normalize the threshold recovery cycle, separately simulate a single-pulse threshold
    experiment with the same test-pulse waveform and readout site but without the conditioning
    pulse. Pass that threshold as ``baseline_threshold``. It can be obtained with a compatible
    single-pulse threshold extractor, e.g. a fixed-pulse-width use of ``chronaxie_from_trials``.

    Timing convention
    -----------------
    If ``test_pulse_times_ms`` is ``None``, trial ``p`` uses

    ``t_test[p] = condition_pulse_time_ms + isis_ms[p]``.

    If your simulations use absolute test-pulse times that differ from this convention, pass
    them directly via ``test_pulse_times_ms``. The test response is scored in

    ``[t_test + response_window_ms[0], t_test + response_window_ms[1]]``.

    For C-fibers or long axons, this window should be long enough for the test spike to reach
    the selected readout site.

    Returns
    -------
    out : dict[str, Any]
        Main threshold-recovery outputs:

        ``"isi_unique_ms"`` : torch.Tensor, shape ``(D,)``
            Sorted unique ISIs defining the recovery-cycle x-axis.

        ``"I_th_test"`` : torch.Tensor, shape ``(D,)``
            Differentiable second-pulse threshold per ISI. The default method,
            ``threshold_method="onset_midpoint"``, estimates the midpoint between the nearest
            inactive strength below onset and the lowest active strength.

        ``"threshold_ratio"`` : torch.Tensor or None, shape ``(D,)``
            ``I_th_test / baseline_threshold`` if ``baseline_threshold`` is supplied.

        ``"threshold_percent_change"`` : torch.Tensor or None, shape ``(D,)``
            ``100 * (threshold_ratio - 1)`` if ``baseline_threshold`` is supplied. Positive
            values indicate refractoriness; negative values indicate supernormality.

        ``"p_response_trial"`` : torch.Tensor, shape ``(P,)``
            Smooth test-response confidence for each paired-pulse trial.

        ``"p_response_by_isi"`` : torch.Tensor, shape ``(D,)``
            Group-level response confidence per ISI. This is most interpretable for
            fixed-amplitude paired-pulse success curves. For amplitude sweeps, it is a diagnostic
            summary across all tested amplitudes.

        ``"isi_weight"`` : torch.Tensor, shape ``(D,)``
            Reliability weight per ISI. This is high when the group contains evidence of both
            low-strength non-activation and at least one activating trial.

        ``"boundaries"`` : dict[str, torch.Tensor]
            Per-ISI boundary diagnostics, each shape ``(D,)``:
            ``I_on``, ``I_pre_inactive``, ``I_last_active``, ``I_post_inactive``, and
            ``I_block_mid``. The last three are useful when high-amplitude block occurs.

        Optional latency / velocity outputs:

        ``"latency_ms_trial"`` and ``"latency_ms_by_isi"``
            Test-spike latency relative to test-pulse onset, computed from a soft upstroke
            arrival time in the test-response window.

        ``"latency_shift_ms"`` and ``"latency_shift_percent"``
            Returned when ``reference_latency_ms`` is supplied.

        ``"v_m_per_s_trial"`` and ``"v_m_per_s_by_isi"``
            Optional velocity recovery estimates when ``lengths_um`` is supplied. Velocity is
            computed by weighted regression of compartment position against soft test-spike
            arrival time. ``lengths_um`` may be scalar, shape ``(C,)``, or shape ``(P, C)``.

        ``"velocity_percent_change"``
            Returned when both ``lengths_um`` and ``reference_velocity_m_per_s`` are supplied.
            Negative values correspond to conduction slowing.

        Other diagnostics include ``p_response_comp``, ``p_dv_comp``, ``test_pulse_times_ms``,
        ``isi_group_id``, latency effective sample size, arrival-time variance, and ``reg``.

    Notes
    -----
    - Outputs are differentiable with respect to ``V``. Grouping by ISI is discrete because ISIs
      are experimental design constants.
    - If cathodic stimuli are negative-valued, pass ``strength=-test_amplitudes`` so that larger
      strength means stronger stimulation. Alternatively set ``use_abs_strength=True``.
    - If high-amplitude block can occur, prefer ``threshold_method="onset"`` or
      ``"onset_midpoint"``. Avoid ``"bracket_midpoint"`` and monotonicity regularization under
      block because blocked high-amplitude trials are inactive but should not define the onset.
    - For propagation recovery rather than local excitability, select distal readout
      compartments via ``node_mask``.

    Examples
    --------
    Threshold recovery curve with cathodic-negative second pulses:

    >>> out = paired_pulse_recovery_from_trials(
    ...     V_pair, amps_test, isis, dt_ms,
    ...     condition_pulse_time_ms=5.0,
    ...     strength=-amps_test,
    ...     baseline_threshold=I0,
    ...     node_mask=distal_mask,
    ...     response_window_ms=(0.2, 8.0),
    ... )
    >>> loss = ((out["threshold_ratio"] - target_ratio) ** 2).mean() + out["reg"]

    Fixed-amplitude paired-pulse success curve:

    >>> out = paired_pulse_recovery_from_trials(
    ...     V_pair_fixed, amps_fixed, isis, dt_ms,
    ...     condition_pulse_time_ms=5.0,
    ...     enforce_min_trials_per_isi=False,
    ...     node_mask=distal_mask,
    ... )
    >>> p_success = out["p_response_by_isi"]
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, P, C).")

    T, n_trials, n_comp = V.shape
    device, dtype = V.device, V.dtype

    if T < 2:
        raise ValueError("Need at least 2 time samples.")

    if not torch.is_tensor(dt_ms):
        dt_ms = torch.tensor(dt_ms, device=device, dtype=dtype)
    else:
        dt_ms = dt_ms.to(device=device, dtype=dtype)

    test_amplitudes = test_amplitudes.to(device=device, dtype=dtype)
    isis_ms = isis_ms.to(device=device, dtype=dtype)
    if test_amplitudes.shape != (n_trials,) or isis_ms.shape != (n_trials,):
        raise ValueError("test_amplitudes and isis_ms must both have shape (P,).")

    if response_window_ms[1] <= response_window_ms[0]:
        raise ValueError("response_window_ms must satisfy end > start.")

    if test_pulse_times_ms is None:
        t_cond = torch.as_tensor(condition_pulse_time_ms, device=device, dtype=dtype)
        test_pulse_times = t_cond + isis_ms
    else:
        test_pulse_times = test_pulse_times_ms.to(device=device, dtype=dtype)
        if test_pulse_times.shape != (n_trials,):
            raise ValueError("test_pulse_times_ms must have shape (P,).")

    win_start = test_pulse_times + torch.as_tensor(
        response_window_ms[0], device=device, dtype=dtype
    )
    win_end = test_pulse_times + torch.as_tensor(
        response_window_ms[1], device=device, dtype=dtype
    )

    # Strength variable used for thresholding. It must increase with stimulus strength.
    if strength is None:
        S = test_amplitudes
    else:
        S = strength.to(device=device, dtype=dtype)
        if S.shape != (n_trials,):
            raise ValueError("strength must have shape (P,).")
    if use_abs_strength:
        S = S.abs()

    # Compartment mask/weights. Allow a shared (C,) mask or trial-specific (P, C) weights.
    if node_mask is None:
        comp_w = torch.ones((n_trials, n_comp), device=device, dtype=dtype)
    else:
        m = node_mask.to(device=device)
        if m.shape == (n_comp,):
            comp_w = m.to(dtype=dtype)[None, :].expand(n_trials, n_comp)
        elif m.shape == (n_trials, n_comp):
            comp_w = m.to(dtype=dtype)
        else:
            raise ValueError("node_mask must have shape (C,) or (P, C).")
    comp_w = torch.clamp(comp_w, min=0.0)
    neg_inf = torch.full_like(comp_w, -torch.inf)
    logw = torch.where(comp_w > 0, torch.log(torch.clamp(comp_w, min=eps)), neg_inf)

    t_ms = torch.arange(T, device=device, dtype=dtype) * dt_ms
    t = t_ms[:, None, None]
    tau_t = torch.as_tensor(gate_t_scale_ms, device=device, dtype=dtype)

    g_time = torch.sigmoid((t - win_start[None, :, None]) / tau_t) * torch.sigmoid(
        (win_end[None, :, None] - t) / tau_t
    )
    log_g_time = torch.log(torch.clamp(g_time, min=eps))

    # ------------------------------------------------------------------
    # 1) Test-response confidence in the window after the second pulse
    # ------------------------------------------------------------------
    Z = V - V_spk
    score_comp = torch.logsumexp(kappa_V * Z + log_g_time, dim=0) / kappa_V
    p_response_comp = torch.sigmoid(score_comp / gate_V_scale)

    score_region = (
        torch.logsumexp(kappa_V * Z + log_g_time + logw[None, :, :], dim=(0, 2))
        / kappa_V
    )
    p_response_trial = torch.sigmoid(score_region / gate_V_scale)

    dV = (V[1:] - V[:-1]) / dt_ms
    t_mid = 0.5 * (t_ms[1:] + t_ms[:-1])
    tmid = t_mid[:, None, None]
    g_mid = torch.sigmoid((tmid - win_start[None, :, None]) / tau_t) * torch.sigmoid(
        (win_end[None, :, None] - tmid) / tau_t
    )
    log_g_mid = torch.log(torch.clamp(g_mid, min=eps))

    p_dv_comp = None
    if use_dv_gate:
        score_dv_comp = (
            torch.logsumexp(kappa_dv * (dV - dv_spk) + log_g_mid, dim=0) / kappa_dv
        )
        p_dv_comp = torch.sigmoid(score_dv_comp / gate_dv_scale)
        p_response_comp = p_response_comp * p_dv_comp

        score_dv_region = (
            torch.logsumexp(
                kappa_dv * (dV - dv_spk) + log_g_mid + logw[None, :, :],
                dim=(0, 2),
            )
            / kappa_dv
        )
        p_response_trial = p_response_trial * torch.sigmoid(
            score_dv_region / gate_dv_scale
        )

    p_response_trial = torch.where(
        comp_w.sum(dim=-1) > 0,
        p_response_trial,
        torch.zeros_like(p_response_trial),
    )

    # ------------------------------------------------------------------
    # 2) Optional latency and velocity estimates for the test spike
    # ------------------------------------------------------------------
    t_hat_test_ms = None
    latency_ms_comp = None
    latency_ms_trial = None
    latency_ess = None
    var_t_ms2 = None
    v_um_per_ms_trial = None
    v_m_per_s_trial = None
    speed_m_per_s_trial = None
    reg = V.new_zeros(())

    if compute_latency:
        U = F.softplus((dV - dv0) / dv_scale)
        rel_t_mid = tmid - win_start[None, :, None]
        arrival_logits = beta * U + log_g_mid - lambda_early * rel_t_mid
        w_time = torch.softmax(arrival_logits, dim=0)
        t_hat_test_ms = (w_time * tmid).sum(dim=0)
        latency_ms_comp = t_hat_test_ms - test_pulse_times[:, None]

        latency_weights = comp_w * p_response_comp
        Wc = latency_weights.sum(dim=-1) + eps
        latency_ms_trial = (latency_weights * latency_ms_comp).sum(dim=-1) / Wc
        t_bar = (latency_weights * t_hat_test_ms).sum(dim=-1, keepdim=True) / Wc[
            :, None
        ]
        dtc = t_hat_test_ms - t_bar
        var_t_ms2 = (latency_weights * dtc * dtc).sum(dim=-1) / Wc
        latency_ess = (latency_weights.sum(dim=-1) ** 2) / (
            (latency_weights * latency_weights).sum(dim=-1) + eps
        )

        if reg_ess_weight > 0.0:
            reg = (
                reg
                + reg_ess_weight
                * F.softplus(
                    torch.tensor(min_ess, device=device, dtype=dtype) - latency_ess
                ).mean()
            )
        if reg_var_weight > 0.0:
            reg = (
                reg
                + reg_var_weight
                * F.softplus(
                    torch.tensor(min_var_ms2, device=device, dtype=dtype) - var_t_ms2
                ).mean()
            )

        if lengths_um is not None:
            L = _coerce_lengths_um(
                lengths_um, Fibs=n_trials, C=n_comp, device=device, dtype=dtype
            )

            x_um = torch.cumsum(L, dim=-1) - 0.5 * L
            x_bar = (latency_weights * x_um).sum(dim=-1, keepdim=True) / Wc[:, None]
            dxc = x_um - x_bar
            cov = (latency_weights * dtc * dxc).sum(dim=-1) / Wc
            v_um_per_ms_trial = cov / (var_t_ms2 + eps)
            v_m_per_s_trial = 1e-3 * v_um_per_ms_trial
            speed_m_per_s_trial = torch.sqrt(v_m_per_s_trial * v_m_per_s_trial + eps)

    # ------------------------------------------------------------------
    # 3) Group by ISI and extract a differentiable threshold per group
    # ------------------------------------------------------------------
    if isi_round_decimals is not None:
        factor = float(10**isi_round_decimals)
        isis_group = torch.round(isis_ms * factor) / factor
    else:
        isis_group = isis_ms

    isi_unique_ms, isi_group_id, isi_counts = torch.unique(
        isis_group, sorted=True, return_inverse=True, return_counts=True
    )
    n_isi = int(isi_unique_ms.numel())

    if enforce_min_trials_per_isi and int(isi_counts.min().item()) < min_trials_per_isi:
        raise ValueError(
            f"Some ISI groups have < {min_trials_per_isi} trials. "
            f"counts={isi_counts.tolist()}. Set enforce_min_trials_per_isi=False "
            "for fixed-amplitude paired-pulse success curves."
        )

    def _soft_pre_inactive(
        Sg: torch.Tensor, pg: torch.Tensor, I_on: torch.Tensor
    ) -> torch.Tensor:
        Smin, Smax = Sg.min(), Sg.max()
        Srange = (Smax - Smin).clamp_min(eps)
        g = I_on - Sg
        inact = torch.clamp(1.0 - pg, min=eps, max=1.0 - eps)
        log_inact = membership_power * torch.log(inact)
        gate_below = torch.sigmoid(g / (below_gate_frac * Srange + eps))
        err = (g / (Srange + eps)) ** 2
        w = torch.softmax(
            -gap_alpha * err + log_inact + torch.log(torch.clamp(gate_below, min=eps)),
            dim=0,
        )
        return I_on - (w * g).sum()

    def _soft_post_inactive(
        Sg: torch.Tensor, pg: torch.Tensor, I_last: torch.Tensor
    ) -> torch.Tensor:
        Smin, Smax = Sg.min(), Sg.max()
        Srange = (Smax - Smin).clamp_min(eps)
        g = Sg - I_last
        inact = torch.clamp(1.0 - pg, min=eps, max=1.0 - eps)
        log_inact = membership_power * torch.log(inact)
        gate_above = torch.sigmoid(g / (below_gate_frac * Srange + eps))
        err = (g / (Srange + eps)) ** 2
        w = torch.softmax(
            -gap_alpha * err + log_inact + torch.log(torch.clamp(gate_above, min=eps)),
            dim=0,
        )
        return I_last + (w * g).sum()

    def _aggregate_prob(pg: torch.Tensor) -> torch.Tensor:
        if success_aggregate == "mean":
            return pg.mean()
        if success_aggregate == "max":
            w = torch.softmax(success_beta * pg, dim=0)
            return (w * pg).sum()
        if success_aggregate == "soft_or":
            return 1.0 - torch.exp(torch.log(torch.clamp(1.0 - pg, min=eps)).sum())
        raise ValueError("success_aggregate must be one of {'mean', 'max', 'soft_or'}.")

    I_th_list, I_on_list, I_pre_list = [], [], []
    I_last_list, I_post_list, I_block_mid_list = [], [], []
    isi_weight_list, p_by_isi_list = [], []
    latency_by_isi_list, v_by_isi_list, speed_by_isi_list = [], [], []
    bracket_terms, mono_terms, unimodal_terms = [], [], []

    for gidx in range(n_isi):
        idx = torch.where(isi_group_id == gidx)[0]
        Sg = S[idx]
        pg = p_response_trial[idx]
        p_by_isi_list.append(_aggregate_prob(pg))

        perm = torch.argsort(Sg)
        pg_sort = pg[perm]

        act = torch.clamp(pg, min=eps, max=1.0 - eps)
        log_act = membership_power * torch.log(act)
        Smin, Smax = Sg.min(), Sg.max()
        Srange = (Smax - Smin).clamp_min(eps)
        Sunit = (Sg - Smin) / Srange

        w_on = torch.softmax(alpha_extreme * (-Sunit) + log_act, dim=0)
        I_on = (w_on * Sg).sum()
        w_last = torch.softmax(alpha_extreme * Sunit + log_act, dim=0)
        I_last = (w_last * Sg).sum()
        I_pre = _soft_pre_inactive(Sg, pg, I_on)

        if compute_block:
            I_post = _soft_post_inactive(Sg, pg, I_last)
            I_block_mid = 0.5 * (I_last + I_post)
        else:
            I_post = torch.tensor(float("nan"), device=device, dtype=dtype)
            I_block_mid = torch.tensor(float("nan"), device=device, dtype=dtype)

        if threshold_method == "onset":
            I_th = I_on
        elif threshold_method == "onset_midpoint":
            I_th = 0.5 * (I_pre + I_on)
        elif threshold_method == "ptarget":
            err = (pg - p_target) ** 2
            wA = torch.softmax(-alpha_thresh * err, dim=0)
            I_th = (wA * Sg).sum()
        elif threshold_method == "bracket_midpoint":
            inact = torch.clamp(1.0 - pg, min=eps, max=1.0 - eps)
            log_inact = membership_power * torch.log(inact)
            w_low = torch.softmax(alpha_extreme * Sunit + log_inact, dim=0)
            I_low = (w_low * Sg).sum()
            I_th = 0.5 * (I_low + I_on)
        else:
            raise ValueError(f"Unknown threshold_method={threshold_method!r}")

        I_th_list.append(I_th)
        I_on_list.append(I_on)
        I_pre_list.append(I_pre)
        I_last_list.append(I_last)
        I_post_list.append(I_post)
        I_block_mid_list.append(I_block_mid)

        # A high-quality threshold group has low-end non-activation and at least one active trial.
        p_at_min = pg_sort[0]
        p_any_active = pg_sort.max()
        tau = torch.as_tensor(0.05, device=device, dtype=dtype)
        w_isi = torch.sigmoid((p_low - p_at_min) / tau) * torch.sigmoid(
            (p_any_active - p_high) / tau
        )
        isi_weight_list.append(w_isi)

        if compute_latency and latency_ms_trial is not None:
            wg = torch.clamp(pg, min=eps)
            latency_by_isi_list.append(
                (wg * latency_ms_trial[idx]).sum() / (wg.sum() + eps)
            )
            if v_m_per_s_trial is not None:
                v_by_isi_list.append(
                    (wg * v_m_per_s_trial[idx]).sum() / (wg.sum() + eps)
                )
                speed_by_isi_list.append(
                    (wg * speed_m_per_s_trial[idx]).sum() / (wg.sum() + eps)
                )

        if reg_bracket_weight > 0.0:
            bracket_terms.append(
                F.softplus(p_at_min - p_low) + F.softplus(p_high - p_any_active)
            )
        if reg_monotone_weight > 0.0 and pg_sort.numel() >= 2:
            dp = pg_sort[1:] - pg_sort[:-1]
            mono_terms.append(F.softplus(-dp).mean())
        if reg_unimodal_weight > 0.0 and pg_sort.numel() >= 3:
            idxs = torch.arange(pg_sort.numel(), device=device, dtype=dtype)
            w_peak = torch.softmax(unimodal_beta_peak * pg_sort, dim=0)
            k_hat = (w_peak * idxs).sum()
            dp = pg_sort[1:] - pg_sort[:-1]
            i = torch.arange(dp.numel(), device=device, dtype=dtype)
            before = torch.sigmoid((k_hat - i) / unimodal_tau_idx)
            after = 1.0 - before
            unimodal_terms.append(
                (before * F.softplus(-dp) + after * F.softplus(dp)).mean()
            )

    I_th_test = torch.stack(I_th_list)
    I_on = torch.stack(I_on_list)
    I_pre = torch.stack(I_pre_list)
    I_last = torch.stack(I_last_list)
    I_post = torch.stack(I_post_list)
    I_block_mid = torch.stack(I_block_mid_list)
    isi_weight = torch.stack(isi_weight_list)
    p_response_by_isi = torch.stack(p_by_isi_list)

    latency_ms_by_isi = (
        torch.stack(latency_by_isi_list) if latency_by_isi_list else None
    )
    v_m_per_s_by_isi = torch.stack(v_by_isi_list) if v_by_isi_list else None
    speed_m_per_s_by_isi = torch.stack(speed_by_isi_list) if speed_by_isi_list else None

    def _ref_to_D(x: float | torch.Tensor | None, name: str):
        if x is None:
            return None
        xt = torch.as_tensor(x, device=device, dtype=dtype)
        if xt.ndim == 0:
            return xt.expand(n_isi)
        if xt.shape == (n_isi,):
            return xt
        raise ValueError(f"{name} must be scalar or shape (D,), where D={n_isi}.")

    baseline_ref = _ref_to_D(baseline_threshold, "baseline_threshold")
    if baseline_ref is None:
        threshold_ratio = None
        threshold_percent_change = None
    else:
        threshold_ratio = I_th_test / (baseline_ref + eps)
        threshold_percent_change = 100.0 * (threshold_ratio - 1.0)

    latency_ref = _ref_to_D(reference_latency_ms, "reference_latency_ms")
    if latency_ref is not None and latency_ms_by_isi is not None:
        latency_shift_ms = latency_ms_by_isi - latency_ref
        latency_shift_percent = 100.0 * latency_shift_ms / (latency_ref + eps)
    else:
        latency_shift_ms = None
        latency_shift_percent = None

    velocity_ref = _ref_to_D(reference_velocity_m_per_s, "reference_velocity_m_per_s")
    if velocity_ref is not None and v_m_per_s_by_isi is not None:
        velocity_percent_change = (
            100.0 * (v_m_per_s_by_isi - velocity_ref) / (velocity_ref + eps)
        )
    else:
        velocity_percent_change = None

    if reg_bracket_weight > 0.0 and bracket_terms:
        reg = reg + reg_bracket_weight * torch.stack(bracket_terms).mean()
    if reg_monotone_weight > 0.0 and mono_terms:
        reg = reg + reg_monotone_weight * torch.stack(mono_terms).mean()
    if reg_unimodal_weight > 0.0 and unimodal_terms:
        reg = reg + reg_unimodal_weight * torch.stack(unimodal_terms).mean()

    out: Dict[str, Any] = {
        "isi_unique_ms": isi_unique_ms,
        "I_th_test": I_th_test,
        "threshold_ratio": threshold_ratio,
        "threshold_percent_change": threshold_percent_change,
        "p_response_trial": p_response_trial,
        "p_response_by_isi": p_response_by_isi,
        "isi_weight": isi_weight,
        "isi_group_id": isi_group_id,
        "isi_counts": isi_counts,
        "test_pulse_times_ms": test_pulse_times,
        "test_strength": S,
        "boundaries": {
            "I_on": I_on,
            "I_pre_inactive": I_pre,
            "I_last_active": I_last,
            "I_post_inactive": I_post,
            "I_block_mid": I_block_mid,
        },
        "p_response_comp": p_response_comp,
        "p_dv_comp": p_dv_comp,
        "t_hat_test_ms": t_hat_test_ms,
        "latency_ms_comp": latency_ms_comp,
        "latency_ms_trial": latency_ms_trial,
        "latency_ms_by_isi": latency_ms_by_isi,
        "latency_shift_ms": latency_shift_ms,
        "latency_shift_percent": latency_shift_percent,
        "latency_ess": latency_ess,
        "var_t_ms2": var_t_ms2,
        "v_um_per_ms_trial": v_um_per_ms_trial,
        "v_m_per_s_trial": v_m_per_s_trial,
        "speed_m_per_s_trial": speed_m_per_s_trial,
        "v_m_per_s_by_isi": v_m_per_s_by_isi,
        "speed_m_per_s_by_isi": speed_m_per_s_by_isi,
        "velocity_percent_change": velocity_percent_change,
        "reg": reg,
    }

    if return_time_traces:
        out.update(
            {
                "g_response": g_time,
                "g_response_mid": g_mid,
            }
        )

    return out


# =============================================================================
# Activity-dependent slowing and hard reference descriptors
# =============================================================================


def _coerce_pulse_times_ms(
    pulse_times_ms: torch.Tensor | Sequence[float],
    *,
    Fibs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Internal helper: return pulse times as shape (F, N)."""
    pt = torch.as_tensor(pulse_times_ms, device=device, dtype=dtype)
    if pt.ndim == 1:
        pt = pt[None, :].expand(Fibs, pt.numel())
    elif pt.ndim == 2:
        if pt.shape[0] == 1:
            pt = pt.expand(Fibs, pt.shape[1])
        elif pt.shape[0] != Fibs:
            raise ValueError(
                f"pulse_times_ms must have shape (N,), (1, N), or (F, N); got {tuple(pt.shape)}."
            )
    else:
        raise ValueError("pulse_times_ms must be one- or two-dimensional.")
    return pt


def _coerce_compartment_weights(
    node_mask: torch.Tensor | None,
    *,
    Fibs: int,
    C: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Internal helper: return nonnegative compartment weights as shape (F, C)."""
    if node_mask is None:
        return torch.ones((Fibs, C), device=device, dtype=dtype)
    m = node_mask.to(device=device)
    if m.shape == (C,):
        w = m.to(dtype=dtype)[None, :].expand(Fibs, C)
    elif m.shape == (Fibs, C):
        w = m.to(dtype=dtype)
    else:
        raise ValueError(
            f"node_mask must have shape (C,) or (F, C); got {tuple(m.shape)}."
        )
    return torch.clamp(w, min=0.0)


def _coerce_lengths_um(
    lengths_um: LengthLike,
    *,
    Fibs: int,
    C: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Internal helper: return compartment lengths as shape (F, C).

    Accepted inputs:
      - scalar or 0-d tensor: all compartments in all fibers/trials share the same length;
      - shape (C,): a shared compartment-length vector;
      - shape (F, C): fiber/trial-specific compartment lengths.
    """
    L = torch.as_tensor(lengths_um, device=device, dtype=dtype)

    if L.ndim == 0:
        return L.expand(Fibs, C)

    # Treat one-element vectors/lists as scalar input as well. This is convenient for
    # calls such as lengths_um=torch.tensor([100.0]) or lengths_um=[100.0].
    if L.ndim == 1 and L.numel() == 1:
        return L.reshape(1, 1).expand(Fibs, C)

    if L.shape == (C,):
        return L[None, :].expand(Fibs, C)
    if L.shape == (Fibs, C):
        return L

    raise ValueError(
        f"lengths_um must be scalar, shape (C,), or shape (F, C); got {tuple(L.shape)}."
    )


def _weighted_mean_over_pulses(
    x: torch.Tensor,  # (F, N)
    w: torch.Tensor,  # (F, N)
    idx: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Internal helper for differentiable baseline/tail summaries."""
    x_sel = x[:, idx]
    w_sel = w[:, idx]
    return (w_sel * x_sel).sum(dim=-1) / (w_sel.sum(dim=-1) + eps)


def _as_F_vector(
    x: float | torch.Tensor | None,
    *,
    Fibs: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor | None:
    """Internal helper: scalar or (F,) tensor -> (F,)."""
    if x is None:
        return None
    xt = torch.as_tensor(x, device=device, dtype=dtype)
    if xt.ndim == 0:
        return xt.expand(Fibs)
    if xt.shape == (Fibs,):
        return xt
    raise ValueError(f"{name} must be scalar or shape (F,), where F={Fibs}.")


def _gather_time_windows_FNKC(
    X: torch.Tensor,  # (T, F, C)
    dt_ms: torch.Tensor,
    win_start_ms: torch.Tensor,  # (F, N)
    win_end_ms: torch.Tensor,  # (F, N)
    *,
    margin_ms: float | torch.Tensor = 0.0,
    sample_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gather local pulse-locked time windows without materializing a full T x N gate.

    Returns
    -------
    X_win : torch.Tensor
        Windowed data, shape (F, N, K, C).
    t_win_ms : torch.Tensor
        Sample times for the gathered indices, shape (F, N, K). For derivative traces,
        use ``sample_offset=0.5`` so index i corresponds to midpoint (i+0.5)*dt.
    valid : torch.Tensor
        Boolean validity mask for unclipped indices, shape (F, N, K).
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (T, F, C).")
    T_x, Fibs, C = X.shape
    if win_start_ms.shape != win_end_ms.shape or win_start_ms.ndim != 2:
        raise ValueError("win_start_ms and win_end_ms must both have shape (F, N).")
    if win_start_ms.shape[0] != Fibs:
        raise ValueError("Window fiber dimension must match X.shape[1].")

    device, dtype = X.device, X.dtype
    dt = torch.as_tensor(dt_ms, device=device, dtype=dtype)
    if dt.ndim != 0:
        raise ValueError("dt_ms must be scalar.")
    dt_float = float(dt.detach().cpu().item())
    if dt_float <= 0.0:
        raise ValueError("dt_ms must be positive.")

    margin = torch.as_tensor(margin_ms, device=device, dtype=dtype)
    margin_float = float(margin.detach().cpu().item())
    if margin_float < 0.0:
        raise ValueError("margin_ms must be nonnegative.")

    # K is a Python integer because it fixes the static gathered window length.
    max_width = float((win_end_ms - win_start_ms).detach().max().cpu().item())
    if max_width <= 0.0:
        raise ValueError("All windows must satisfy end > start.")
    K = int(np.ceil((max_width + 2.0 * margin_float) / dt_float)) + 4
    K = max(K, 2)

    base = torch.floor((win_start_ms - margin) / dt).to(torch.long)  # (F, N)
    offsets = torch.arange(K, device=device, dtype=torch.long)  # (K,)
    idx = base[:, :, None] + offsets[None, None, :]  # (F, N, K)
    valid = (idx >= 0) & (idx < T_x)
    idx_clip = idx.clamp(0, T_x - 1)

    Fw, N = win_start_ms.shape
    X_ftc = X.permute(1, 0, 2).contiguous()  # (F, T, C)
    idx_flat = idx_clip.reshape(Fw, N * K)
    X_flat = torch.gather(
        X_ftc,
        dim=1,
        index=idx_flat[:, :, None].expand(Fw, N * K, C),
    )
    X_win = X_flat.reshape(Fw, N, K, C)
    t_win_ms = (
        idx.to(dtype=dtype) + torch.as_tensor(sample_offset, device=device, dtype=dtype)
    ) * dt
    return X_win, t_win_ms, valid


def _finite_mean_over_indices(
    x: torch.Tensor,  # (F, N)
    idx: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Mean over selected pulse indices, ignoring NaNs; returns NaN if no finite values."""
    x_sel = x[:, idx]
    finite = torch.isfinite(x_sel)
    vals = torch.where(finite, x_sel, torch.zeros_like(x_sel))
    cnt = finite.to(dtype=x.dtype).sum(dim=-1)
    out = vals.sum(dim=-1) / torch.clamp(cnt, min=1.0)
    return torch.where(cnt > 0, out, torch.full_like(out, float("nan")))


def activity_dependent_slowing(
    V: torch.Tensor,  # (T, F, C)
    pulse_times_ms: torch.Tensor | Sequence[float],  # (N,), (1, N), or (F, N)
    dt_ms: float | torch.Tensor,
    *,
    lengths_um: Optional[LengthLike] = None,  # scalar, (C,), or (F, C)
    node_mask: torch.Tensor | None = None,  # (C,) or (F, C)
    response_window_ms: tuple[float, float] = (0.1, 10.0),
    gate_t_scale_ms: float = 0.05,
    # Local-window execution controls
    window_margin_ms: float | None = None,
    chunk_pulses: int | None = None,
    return_compartment_metrics: bool = True,
    # Spike-present scoring
    V_spk: float = 0.0,
    kappa_V: float = 20.0,
    gate_V_scale: float = 5.0,
    use_dv_gate: bool = True,
    dv_spk: float = 10.0,
    kappa_dv: float = 10.0,
    gate_dv_scale: float = 5.0,
    # Soft arrival-time settings
    beta: float = 50.0,
    dv0: float = 0.0,
    dv_scale: float = 1.0,
    lambda_early: float = 0.0,
    # Baseline / normalization
    baseline_pulse_indices: torch.Tensor | Sequence[int] | None = None,
    baseline_n_pulses: int = 1,
    reference_latency_ms: float | torch.Tensor | None = None,  # scalar or (F,)
    reference_velocity_m_per_s: float | torch.Tensor | None = None,  # scalar or (F,)
    tail_n_pulses: int = 10,
    # Aggregation and robustness
    success_threshold: float = 0.5,
    min_ess: float = 1.0,
    min_var_ms2: float = 1e-3,
    min_success_prob: float = 0.5,
    reg_ess_weight: float = 0.0,
    reg_var_weight: float = 0.0,
    reg_success_weight: float = 0.0,
    eps: float = 1e-8,
    return_time_traces: bool = False,
) -> Dict[str, Any]:
    r"""
    Differentiably estimate activity-dependent slowing (ADS), train following, latency drift,
    and optional velocity drift from pulse-train simulations.

    This optimized implementation avoids the original per-pulse full-time gating cost. Instead
    of constructing a ``T x F x C`` soft gate for every pulse in the train, it gathers a short
    pulse-locked local window around each stimulus and performs all soft maxima / soft arrivals
    on the resulting ``F x N x K x C`` tensor, where ``K`` is the number of samples in
    ``response_window_ms`` plus a small sigmoid-gate margin. For long ADS traces this changes the
    dominant cost from approximately ``O(N*T*F*C)`` to ``O(N*K*F*C)`` with ``K << T``.

    Required input data
    -------------------
    V : torch.Tensor
        Simulated voltage traces with shape ``(T, F, C)``:

        - ``T``: number of time samples
        - ``F``: number of fibers / model instances
        - ``C``: number of compartments or recording sites

        The trace must contain the whole pulse train. Units are typically mV. Gradients
        propagate through ``V`` and therefore through the upstream differentiable simulator.

    pulse_times_ms : array-like or torch.Tensor
        Stimulus pulse onset times in ms. Accepts shape ``(N,)`` for a pulse train shared by
        all fibers, ``(1, N)`` for broadcastable pulse times, or ``(F, N)`` for fiber-specific
        pulse times. ``N`` is the number of pulses in the train.

    dt_ms : float or torch.Tensor
        Simulation time step in milliseconds.

    Simulations required to produce the input data
    ----------------------------------------------
    To compute ADS, run a train-stimulation simulation such as:

    ``for n in range(N):``
        deliver the same suprathreshold pulse at ``pulse_times_ms[n]``

    and record the membrane potential across the compartments used for propagation readout.
    The response window should be long enough for each pulse-evoked spike to reach the selected
    readout region but short enough to avoid overlap with the next pulse. Select distal
    compartments in ``node_mask`` when ADS should reflect propagation to a distal endpoint.

    Performance controls
    --------------------
    window_margin_ms : float, optional
        Extra time included before and after the response window when gathering local samples.
        If ``None``, this defaults to ``max(6*gate_t_scale_ms, 2*dt_ms)``. Increasing this makes
        the local-window result closer to a full-trace sigmoid gate; decreasing it saves memory.

    chunk_pulses : int, optional
        If provided, process at most this many pulses at once. This keeps the fast local-window
        method memory bounded while retaining vectorization within each chunk. Use e.g.
        ``chunk_pulses=16`` for very large ``F*C*N*K``.

    return_compartment_metrics : bool, default=True
        If True, return compartment-level arrays such as ``p_success_comp`` and
        ``t_hat_ms_comp``. Set to False inside optimization loops to reduce memory retained in
        the autograd graph; fiber-level outputs are still returned.

    Optional velocity inputs
    ------------------------
    lengths_um : scalar, torch.Tensor, or sequence, optional
        Compartment lengths in micrometers. Accepts a scalar, shape ``(C,)``, or shape ``(F, C)``.
        When provided, a per-pulse conduction velocity is estimated by weighted regression of
        compartment position against pulse-locked soft arrival time. If omitted, the function
        still returns latency-based ADS.

    node_mask : torch.Tensor, optional
        Compartment mask/weights with shape ``(C,)`` or ``(F, C)``. Use this to select the
        readout region. For latency-only ADS, this is often a distal compartment or distal
        group. For velocity ADS, it should include multiple compartments spanning a meaningful
        propagation distance.

    Returns
    -------
    out : dict[str, Any]
        Main outputs:

        ``"pulse_times_ms"`` : torch.Tensor, shape ``(F, N)``
            Pulse onset times used for the analysis.

        ``"p_success"`` : torch.Tensor, shape ``(F, N)``
            Smooth train-following / spike-success confidence for each pulse.

        ``"follow_fraction"`` : torch.Tensor, shape ``(F,)``
            Mean pulse-following probability across the train.

        ``"latency_ms"`` : torch.Tensor, shape ``(F, N)``
            Pulse-locked latency relative to each stimulus onset.

        ``"ads_percent"`` / ``"latency_slowing_percent"`` : torch.Tensor, shape ``(F, N)``
            Percent latency increase relative to baseline. Positive values indicate slowing.

        ``"final_ads_percent"`` and ``"tail_ads_percent"`` : torch.Tensor, shape ``(F,)``
            ADS at the final pulse and success-weighted mean ADS over the final pulses.

        Optional velocity outputs, returned as ``None`` when ``lengths_um`` is omitted:
        ``"v_m_per_s"``, ``"velocity_change_percent"``, and ``"velocity_slowing_percent"``.

    Notes
    -----
    - All main outputs are differentiable with respect to ``V``. Pulse times and gathered sample
      indices are treated as experiment-design constants.
    - If the train contains multiple spikes per response window, the soft arrival estimator
      returns the dominant upstroke in the local window. Use shorter windows or positive
      ``lambda_early`` to bias toward the first pulse-evoked spike.
    - ``return_time_traces=True`` no longer stores dense pulse-by-time gates; the function returns
      a note because storing those gates would defeat the optimization.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, F, C).")
    T, Fibs, C = V.shape
    if T < 2:
        raise ValueError("Need at least 2 time samples.")
    if response_window_ms[1] <= response_window_ms[0]:
        raise ValueError("response_window_ms must satisfy end > start.")
    if baseline_n_pulses < 1:
        raise ValueError("baseline_n_pulses must be >= 1.")
    if tail_n_pulses < 1:
        raise ValueError("tail_n_pulses must be >= 1.")
    if chunk_pulses is not None and chunk_pulses < 1:
        raise ValueError("chunk_pulses must be None or a positive integer.")

    device, dtype = V.device, V.dtype
    if not torch.is_tensor(dt_ms):
        dt_ms = torch.tensor(dt_ms, device=device, dtype=dtype)
    else:
        dt_ms = dt_ms.to(device=device, dtype=dtype)
    dt_float = float(dt_ms.detach().cpu().item())

    pulse_times = _coerce_pulse_times_ms(
        pulse_times_ms, Fibs=Fibs, device=device, dtype=dtype
    )
    N = pulse_times.shape[1]
    if baseline_n_pulses > N:
        raise ValueError("baseline_n_pulses cannot exceed number of pulses.")

    if baseline_pulse_indices is None:
        baseline_idx = torch.arange(baseline_n_pulses, device=device, dtype=torch.long)
    else:
        baseline_idx = torch.as_tensor(
            baseline_pulse_indices, device=device, dtype=torch.long
        ).flatten()
        if baseline_idx.numel() < 1:
            raise ValueError("baseline_pulse_indices must contain at least one index.")
        if int(baseline_idx.min().item()) < 0 or int(baseline_idx.max().item()) >= N:
            raise ValueError(
                "baseline_pulse_indices contains an out-of-range pulse index."
            )

    tail_k = min(int(tail_n_pulses), N)
    tail_idx = torch.arange(N - tail_k, N, device=device, dtype=torch.long)

    comp_w = _coerce_compartment_weights(
        node_mask, Fibs=Fibs, C=C, device=device, dtype=dtype
    )
    neg_inf = torch.full_like(comp_w, -torch.inf)
    logw_comp = torch.where(
        comp_w > 0, torch.log(torch.clamp(comp_w, min=eps)), neg_inf
    )

    if lengths_um is not None:
        L = _coerce_lengths_um(lengths_um, Fibs=Fibs, C=C, device=device, dtype=dtype)
        x_um = torch.cumsum(L, dim=-1) - 0.5 * L
    else:
        x_um = None

    dV = (V[1:] - V[:-1]) / dt_ms  # (T-1, F, C)
    tau_t = torch.as_tensor(gate_t_scale_ms, device=device, dtype=dtype)
    tiny = torch.finfo(dtype).tiny
    if window_margin_ms is None:
        margin_ms = max(6.0 * float(gate_t_scale_ms), 2.0 * dt_float)
    else:
        margin_ms = float(window_margin_ms)

    p_success_chunks = []
    latency_chunks = []
    ess_chunks = []
    var_t_chunks = []
    v_um_chunks = []
    v_ms_chunks = []
    speed_chunks = []

    p_comp_chunks = []
    p_dv_comp_chunks = []
    t_hat_chunks = []
    latency_comp_chunks = []

    chunk_size = N if chunk_pulses is None else int(chunk_pulses)

    for n0 in range(0, N, chunk_size):
        n1 = min(N, n0 + chunk_size)
        pulse_chunk = pulse_times[:, n0:n1]  # (F, M)
        win_start = pulse_chunk + torch.as_tensor(
            response_window_ms[0], device=device, dtype=dtype
        )
        win_end = pulse_chunk + torch.as_tensor(
            response_window_ms[1], device=device, dtype=dtype
        )

        V_win, t_win, valid_v = _gather_time_windows_FNKC(
            V,
            dt_ms,
            win_start,
            win_end,
            margin_ms=margin_ms,
            sample_offset=0.0,
        )  # (F, M, K, C), (F, M, K)
        g_time = torch.sigmoid((t_win - win_start[:, :, None]) / tau_t) * torch.sigmoid(
            (win_end[:, :, None] - t_win) / tau_t
        )
        g_time = torch.where(valid_v, g_time, torch.zeros_like(g_time))
        log_g_time = torch.log(torch.clamp(g_time, min=tiny))

        Z = V_win - V_spk
        score_comp = (
            torch.logsumexp(
                kappa_V * Z + kappa_V * log_g_time[:, :, :, None],
                dim=2,
            )
            / kappa_V
        )  # (F, M, C)
        p_comp = torch.sigmoid(score_comp / gate_V_scale)

        score_region = (
            torch.logsumexp(
                kappa_V * Z
                + kappa_V * log_g_time[:, :, :, None]
                + logw_comp[:, None, None, :],
                dim=(2, 3),
            )
            / kappa_V
        )  # (F, M)
        p_region = torch.sigmoid(score_region / gate_V_scale)

        dV_win, t_mid_win, valid_dv = _gather_time_windows_FNKC(
            dV,
            dt_ms,
            win_start,
            win_end,
            margin_ms=margin_ms,
            sample_offset=0.5,
        )
        g_mid = torch.sigmoid(
            (t_mid_win - win_start[:, :, None]) / tau_t
        ) * torch.sigmoid((win_end[:, :, None] - t_mid_win) / tau_t)
        g_mid = torch.where(valid_dv, g_mid, torch.zeros_like(g_mid))
        log_g_mid = torch.log(torch.clamp(g_mid, min=tiny))

        p_dv_comp = None
        if use_dv_gate:
            score_dv_comp = (
                torch.logsumexp(
                    kappa_dv * (dV_win - dv_spk) + kappa_dv * log_g_mid[:, :, :, None],
                    dim=2,
                )
                / kappa_dv
            )
            p_dv_comp = torch.sigmoid(score_dv_comp / gate_dv_scale)
            p_comp = p_comp * p_dv_comp

            score_dv_region = (
                torch.logsumexp(
                    kappa_dv * (dV_win - dv_spk)
                    + kappa_dv * log_g_mid[:, :, :, None]
                    + logw_comp[:, None, None, :],
                    dim=(2, 3),
                )
                / kappa_dv
            )
            p_region = p_region * torch.sigmoid(score_dv_region / gate_dv_scale)

        p_region = torch.where(
            comp_w.sum(dim=-1)[:, None] > 0,
            p_region,
            torch.zeros_like(p_region),
        )

        U = F.softplus((dV_win - dv0) / dv_scale)
        rel_t_mid = t_mid_win - win_start[:, :, None]
        arrival_logits = (
            beta * U
            + beta * log_g_mid[:, :, :, None]
            - lambda_early * rel_t_mid[:, :, :, None]
        )
        w_time = torch.softmax(arrival_logits, dim=2)
        t_hat_comp = (w_time * t_mid_win[:, :, :, None]).sum(dim=2)  # (F, M, C)
        latency_comp = t_hat_comp - pulse_chunk[:, :, None]

        latency_weights = comp_w[:, None, :] * p_comp
        Wc = latency_weights.sum(dim=-1) + eps
        latency = (latency_weights * latency_comp).sum(dim=-1) / Wc

        t_bar = (latency_weights * t_hat_comp).sum(dim=-1, keepdim=True) / Wc[
            :, :, None
        ]
        dtc = t_hat_comp - t_bar
        var_t = (latency_weights * dtc * dtc).sum(dim=-1) / Wc
        ess = (latency_weights.sum(dim=-1) ** 2) / (
            (latency_weights * latency_weights).sum(dim=-1) + eps
        )

        if x_um is not None:
            x_bar = (latency_weights * x_um[:, None, :]).sum(dim=-1, keepdim=True) / Wc[
                :, :, None
            ]
            dxc = x_um[:, None, :] - x_bar
            cov = (latency_weights * dtc * dxc).sum(dim=-1) / Wc
            v_um_per_ms = cov / (var_t + eps)
            v_m_per_s = 1e-3 * v_um_per_ms
            speed_m_per_s = torch.sqrt(v_m_per_s * v_m_per_s + eps)
            v_um_chunks.append(v_um_per_ms)
            v_ms_chunks.append(v_m_per_s)
            speed_chunks.append(speed_m_per_s)

        p_success_chunks.append(p_region)
        latency_chunks.append(latency)
        ess_chunks.append(ess)
        var_t_chunks.append(var_t)

        if return_compartment_metrics:
            p_comp_chunks.append(p_comp)
            if use_dv_gate:
                p_dv_comp_chunks.append(p_dv_comp)
            t_hat_chunks.append(t_hat_comp)
            latency_comp_chunks.append(latency_comp)

    p_success = torch.cat(p_success_chunks, dim=1)  # (F, N)
    latency_ms = torch.cat(latency_chunks, dim=1)  # (F, N)
    latency_ess = torch.cat(ess_chunks, dim=1)  # (F, N)
    var_t_ms2 = torch.cat(var_t_chunks, dim=1)  # (F, N)

    if return_compartment_metrics:
        p_success_comp = (
            torch.cat(p_comp_chunks, dim=1).permute(1, 0, 2).contiguous()
        )  # (N, F, C)
        p_dv_comp_out = (
            torch.cat(p_dv_comp_chunks, dim=1).permute(1, 0, 2).contiguous()
            if use_dv_gate
            else None
        )
        t_hat_ms_comp = (
            torch.cat(t_hat_chunks, dim=1).permute(1, 0, 2).contiguous()
        )  # (N, F, C)
        latency_ms_comp = (
            torch.cat(latency_comp_chunks, dim=1).permute(1, 0, 2).contiguous()
        )
    else:
        p_success_comp = None
        p_dv_comp_out = None
        t_hat_ms_comp = None
        latency_ms_comp = None

    baseline_latency_ms = _as_F_vector(
        reference_latency_ms,
        Fibs=Fibs,
        device=device,
        dtype=dtype,
        name="reference_latency_ms",
    )
    if baseline_latency_ms is None:
        baseline_latency_ms = _weighted_mean_over_pulses(
            latency_ms, torch.clamp(p_success, min=eps), baseline_idx, eps
        )

    latency_shift_ms = latency_ms - baseline_latency_ms[:, None]
    ads_percent = 100.0 * latency_shift_ms / (baseline_latency_ms[:, None] + eps)
    arrival_time_ms = pulse_times + latency_ms

    if N >= 2:
        input_isi_ms = pulse_times[:, 1:] - pulse_times[:, :-1]
        output_isi_ms = arrival_time_ms[:, 1:] - arrival_time_ms[:, :-1]
        isi_error_ms = output_isi_ms - input_isi_ms
        instantaneous_frequency_hz = 1000.0 / (input_isi_ms + eps)
    else:
        input_isi_ms = V.new_empty((Fibs, 0))
        output_isi_ms = V.new_empty((Fibs, 0))
        isi_error_ms = V.new_empty((Fibs, 0))
        instantaneous_frequency_hz = V.new_empty((Fibs, 0))

    follow_fraction = p_success.mean(dim=-1)
    p_fail = 1.0 - p_success
    final_ads_percent = ads_percent[:, -1]
    tail_ads_percent = _weighted_mean_over_pulses(
        ads_percent, torch.clamp(p_success, min=eps), tail_idx, eps
    )
    tail_success = p_success[:, tail_idx].mean(dim=-1)
    hard_follow_count_surrogate = (
        (p_success > success_threshold).to(dtype=dtype).sum(dim=-1)
    )

    v_um_per_ms = None
    v_m_per_s = None
    speed_m_per_s = None
    baseline_velocity_m_per_s = None
    velocity_change_percent = None
    velocity_slowing_percent = None
    final_velocity_slowing_percent = None
    tail_velocity_slowing_percent = None

    if v_ms_chunks:
        v_um_per_ms = torch.cat(v_um_chunks, dim=1)
        v_m_per_s = torch.cat(v_ms_chunks, dim=1)
        speed_m_per_s = torch.cat(speed_chunks, dim=1)

        baseline_velocity_m_per_s = _as_F_vector(
            reference_velocity_m_per_s,
            Fibs=Fibs,
            device=device,
            dtype=dtype,
            name="reference_velocity_m_per_s",
        )
        if baseline_velocity_m_per_s is None:
            baseline_velocity_m_per_s = _weighted_mean_over_pulses(
                v_m_per_s, torch.clamp(p_success, min=eps), baseline_idx, eps
            )
        velocity_change_percent = (
            100.0
            * (v_m_per_s - baseline_velocity_m_per_s[:, None])
            / (baseline_velocity_m_per_s[:, None] + eps)
        )
        velocity_slowing_percent = -velocity_change_percent
        final_velocity_slowing_percent = velocity_slowing_percent[:, -1]
        tail_velocity_slowing_percent = _weighted_mean_over_pulses(
            velocity_slowing_percent, torch.clamp(p_success, min=eps), tail_idx, eps
        )

    reg = V.new_zeros(())
    if reg_ess_weight > 0.0:
        reg = (
            reg
            + reg_ess_weight
            * F.softplus(
                torch.tensor(min_ess, device=device, dtype=dtype) - latency_ess
            ).mean()
        )
    if reg_var_weight > 0.0:
        reg = (
            reg
            + reg_var_weight
            * F.softplus(
                torch.tensor(min_var_ms2, device=device, dtype=dtype) - var_t_ms2
            ).mean()
        )
    if reg_success_weight > 0.0:
        reg = (
            reg
            + reg_success_weight
            * F.softplus(
                torch.tensor(min_success_prob, device=device, dtype=dtype) - p_success
            ).mean()
        )

    out: Dict[str, Any] = {
        "pulse_times_ms": pulse_times,
        "p_success": p_success,
        "p_fail": p_fail,
        "follow_fraction": follow_fraction,
        "hard_follow_count_surrogate": hard_follow_count_surrogate,
        "p_success_comp": p_success_comp,
        "p_dv_comp": p_dv_comp_out,
        "latency_ms": latency_ms,
        "baseline_latency_ms": baseline_latency_ms,
        "latency_shift_ms": latency_shift_ms,
        "ads_percent": ads_percent,
        "latency_slowing_percent": ads_percent,
        "final_ads_percent": final_ads_percent,
        "tail_ads_percent": tail_ads_percent,
        "tail_success": tail_success,
        "arrival_time_ms": arrival_time_ms,
        "input_isi_ms": input_isi_ms,
        "instantaneous_frequency_hz": instantaneous_frequency_hz,
        "output_isi_ms": output_isi_ms,
        "isi_error_ms": isi_error_ms,
        "t_hat_ms_comp": t_hat_ms_comp,
        "latency_ms_comp": latency_ms_comp,
        "latency_ess": latency_ess,
        "var_t_ms2": var_t_ms2,
        "weights": comp_w,
        "baseline_pulse_indices": baseline_idx,
        "tail_pulse_indices": tail_idx,
        "v_um_per_ms": v_um_per_ms,
        "v_m_per_s": v_m_per_s,
        "speed_m_per_s": speed_m_per_s,
        "baseline_velocity_m_per_s": baseline_velocity_m_per_s,
        "velocity_change_percent": velocity_change_percent,
        "velocity_slowing_percent": velocity_slowing_percent,
        "final_velocity_slowing_percent": final_velocity_slowing_percent,
        "tail_velocity_slowing_percent": tail_velocity_slowing_percent,
        "reg": reg,
    }

    if return_time_traces:
        out["return_time_traces_note"] = (
            "Dense pulse x time gates are intentionally not stored to avoid O(T*N*F*C) memory use. "
            "This optimized ADS implementation gathers local pulse-locked windows instead."
        )

    return out


@torch.no_grad()
def hard_spike_arrival_times(
    V: torch.Tensor,  # (T, F, C)
    dt_ms: float | torch.Tensor,
    *,
    V_th: float = 0.0,
    dv_th: float | None = None,
    time_window: tuple[int, int] | None = None,
    time_window_ms: tuple[float, float] | None = None,
    interpolate: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Non-differentiable first upward-threshold-crossing arrival times.

    This hard reference descriptor is intended for Experiment 1 comparisons against smooth
    surrogates. It uses boolean threshold crossings and therefore does not provide useful
    gradients. Crossings are detected independently for every fiber and compartment.

    Returns ``t_cross_ms`` and ``has_crossing`` with shape ``(F, C)``.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, F, C).")
    T, Fibs, C = V.shape
    device, dtype = V.device, V.dtype
    dt = torch.as_tensor(dt_ms, device=device, dtype=dtype)

    if time_window is not None and time_window_ms is not None:
        raise ValueError("Pass only one of time_window or time_window_ms.")
    if time_window_ms is not None:
        start = max(0, int(np.floor(float(time_window_ms[0]) / float(dt.item()))))
        end = min(T, int(np.ceil(float(time_window_ms[1]) / float(dt.item()))) + 1)
    elif time_window is not None:
        start, end = int(time_window[0]), int(time_window[1])
        start = max(0, start)
        end = min(T, end)
    else:
        start, end = 0, T

    nan = torch.full((Fibs, C), float("nan"), device=device, dtype=dtype)
    if end - start < 2:
        return {
            "t_cross_ms": nan,
            "has_crossing": torch.zeros((Fibs, C), device=device, dtype=torch.bool),
        }

    Vw = V[start:end]
    crossings = (Vw[:-1] < V_th) & (Vw[1:] >= V_th)
    if dv_th is not None:
        dV = (Vw[1:] - Vw[:-1]) / dt
        crossings = crossings & (dV >= dv_th)

    has = crossings.any(dim=0)
    idx_rel = crossings.to(torch.int64).argmax(dim=0)  # first True, or 0 when absent
    idx0 = idx_rel.clamp(0, end - start - 2)

    fidx = torch.arange(Fibs, device=device)[:, None].expand(Fibs, C)
    cidx = torch.arange(C, device=device)[None, :].expand(Fibs, C)
    v0 = Vw[idx0, fidx, cidx]
    v1 = Vw[idx0 + 1, fidx, cidx]

    if interpolate:
        frac = (
            (torch.as_tensor(V_th, device=device, dtype=dtype) - v0) / (v1 - v0 + 1e-12)
        ).clamp(0.0, 1.0)
        t_cross = (start + idx0.to(dtype) + frac) * dt
    else:
        t_cross = (start + idx0.to(dtype) + 1.0) * dt
    t_cross = torch.where(has, t_cross, nan)
    return {"t_cross_ms": t_cross, "has_crossing": has}


@torch.no_grad()
def hard_active(
    V: torch.Tensor,  # (T, F, C)
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    V_spk: float = 0.0,
    use_dv_gate: bool = True,
    dv_spk: float = 10.0,
    time_window: tuple[int, int] | None = None,
    time_window_ms: tuple[float, float] | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Hard non-differentiable active/inactive descriptor.

    A compartment is active if it crosses ``V_spk`` upward in the specified window, optionally
    requiring ``dV/dt >= dv_spk`` on the crossing interval. A fiber is active if any selected
    compartment is active.
    """
    arr = hard_spike_arrival_times(
        V,
        dt_ms,
        V_th=V_spk,
        dv_th=dv_spk if use_dv_gate else None,
        time_window=time_window,
        time_window_ms=time_window_ms,
    )
    active_comp = arr["has_crossing"]
    Fibs, C = active_comp.shape
    device = V.device
    if node_mask is None:
        mask = torch.ones((Fibs, C), device=device, dtype=torch.bool)
    else:
        m = node_mask.to(device=device)
        if m.shape == (C,):
            mask = (m != 0)[None, :].expand(Fibs, C)
        elif m.shape == (Fibs, C):
            mask = m != 0
        else:
            raise ValueError("node_mask must have shape (C,) or (F, C).")
    active_region = (active_comp & mask).any(dim=-1)
    return {
        "active": active_region,
        "active_float": active_region.to(dtype=V.dtype),
        "active_comp": active_comp,
        "t_cross_ms": arr["t_cross_ms"],
        "weights": mask,
    }


@torch.no_grad()
def hard_firing_rate(
    V: torch.Tensor,  # (T, F, C)
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    V_spk: float = 0.0,
    use_dv_gate: bool = False,
    dv_spk: float = 10.0,
    time_window: tuple[int, int] | None = None,
    time_window_ms: tuple[float, float] | None = None,
    refractory_ms: float = 0.0,
    aggregate: Literal["mean", "max", "sum"] = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Hard spike-count and firing-rate descriptor based on upward threshold crossings.

    This is non-differentiable and intended as an experimental reference for validating the
    smooth ``firing_rate`` surrogate. By default, per-fiber counts are averaged across selected
    compartments, matching the surrogate's mean-over-compartments convention.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, F, C).")
    T, Fibs, C = V.shape
    device, dtype = V.device, V.dtype
    dt = torch.as_tensor(dt_ms, device=device, dtype=dtype)

    if time_window is not None and time_window_ms is not None:
        raise ValueError("Pass only one of time_window or time_window_ms.")
    if time_window_ms is not None:
        start = max(0, int(np.floor(float(time_window_ms[0]) / float(dt.item()))))
        end = min(T, int(np.ceil(float(time_window_ms[1]) / float(dt.item()))) + 1)
    elif time_window is not None:
        start, end = int(time_window[0]), int(time_window[1])
        start = max(0, start)
        end = min(T, end)
    else:
        start, end = 0, T
    if end - start < 2:
        raise ValueError("Selected time window must contain at least 2 samples.")

    Vw = V[start:end]
    crossings = (Vw[:-1] < V_spk) & (Vw[1:] >= V_spk)
    if use_dv_gate:
        dV = (Vw[1:] - Vw[:-1]) / dt
        crossings = crossings & (dV >= dv_spk)

    if refractory_ms > 0.0:
        refractory_steps = max(1, int(round(refractory_ms / float(dt.item()))))
        count_comp = torch.zeros((Fibs, C), device=device, dtype=dtype)
        for f in range(Fibs):
            for c in range(C):
                last = -(10**12)
                n_count = 0
                for idx in torch.where(crossings[:, f, c])[0].detach().cpu().tolist():
                    if idx - last >= refractory_steps:
                        n_count += 1
                        last = idx
                count_comp[f, c] = float(n_count)
    else:
        count_comp = crossings.to(dtype=dtype).sum(dim=0)

    if node_mask is None:
        mask = torch.ones((Fibs, C), device=device, dtype=dtype)
    else:
        m = node_mask.to(device=device)
        if m.shape == (C,):
            mask = (m != 0).to(dtype=dtype)[None, :].expand(Fibs, C)
        elif m.shape == (Fibs, C):
            mask = (m != 0).to(dtype=dtype)
        else:
            raise ValueError("node_mask must have shape (C,) or (F, C).")

    if aggregate == "mean":
        count = (mask * count_comp).sum(dim=-1) / (mask.sum(dim=-1) + 1e-12)
    elif aggregate == "max":
        masked = torch.where(
            mask > 0, count_comp, torch.full_like(count_comp, float("-inf"))
        )
        count = masked.max(dim=-1).values
        count = torch.where(torch.isfinite(count), count, torch.zeros_like(count))
    elif aggregate == "sum":
        count = (mask * count_comp).sum(dim=-1)
    else:
        raise ValueError("aggregate must be one of {'mean', 'max', 'sum'}.")

    window_ms = (end - start - 1) * dt
    rate_hz_comp = 1000.0 * count_comp / window_ms
    rate_hz = 1000.0 * count / window_ms
    return {
        "count": count,
        "count_comp": count_comp,
        "rate_hz": rate_hz,
        "rate_hz_comp": rate_hz_comp,
        "window_ms": window_ms,
        "weights": mask,
    }


@torch.no_grad()
def hard_conduction_velocity(
    V: torch.Tensor,  # (T, F, C)
    lengths_um: LengthLike,
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    V_th: float = 0.0,
    dv_th: float | None = None,
    time_window: tuple[int, int] | None = None,
    time_window_ms: tuple[float, float] | None = None,
    interpolate: bool = True,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Hard non-differentiable conduction velocity from first threshold-crossing times.

    Arrival times are extracted by ``hard_spike_arrival_times`` and then used in ordinary
    least-squares regression of compartment position against arrival time, ignoring inactive
    or masked compartments. Returns NaN when fewer than two selected compartments spike.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, F, C).")
    _, Fibs, C = V.shape
    device, dtype = V.device, V.dtype
    L = _coerce_lengths_um(lengths_um, Fibs=Fibs, C=C, device=device, dtype=dtype)
    x_um = torch.cumsum(L, dim=-1) - 0.5 * L
    arr = hard_spike_arrival_times(
        V,
        dt_ms,
        V_th=V_th,
        dv_th=dv_th,
        time_window=time_window,
        time_window_ms=time_window_ms,
        interpolate=interpolate,
    )
    t_cross = arr["t_cross_ms"]
    has = arr["has_crossing"]
    if node_mask is None:
        mask = torch.ones((Fibs, C), device=device, dtype=torch.bool)
    else:
        m = node_mask.to(device=device)
        if m.shape == (C,):
            mask = (m != 0)[None, :].expand(Fibs, C)
        elif m.shape == (Fibs, C):
            mask = m != 0
        else:
            raise ValueError("node_mask must have shape (C,) or (F, C).")
    valid = has & mask
    v_um = torch.full((Fibs,), float("nan"), device=device, dtype=dtype)
    var_t = torch.full((Fibs,), float("nan"), device=device, dtype=dtype)
    ess = valid.to(dtype=dtype).sum(dim=-1)
    for f in range(Fibs):
        idx = torch.where(valid[f])[0]
        if idx.numel() < 2:
            continue
        tf = t_cross[f, idx]
        xf = x_um[f, idx]
        tb = tf.mean()
        xb = xf.mean()
        denom = ((tf - tb) ** 2).mean()
        var_t[f] = denom
        if float(denom.item()) <= eps:
            continue
        cov = ((tf - tb) * (xf - xb)).mean()
        v_um[f] = cov / denom
    v_ms = 1e-3 * v_um
    return {
        "v_um_per_ms": v_um,
        "v_m_per_s": v_ms,
        "speed_m_per_s": v_ms.abs(),
        "t_cross_ms": t_cross,
        "has_crossing": has,
        "weights": valid,
        "ess": ess,
        "var_t_ms2": var_t,
    }


def _hard_width_crossings_around_peak(
    trace: torch.Tensor,
    dt: torch.Tensor,
    threshold: float | torch.Tensor,
    peak_idx: int,
    start_idx: int,
    end_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal no-grad helper for hard APD-style widths around a known peak."""
    device, dtype = trace.device, trace.dtype
    thr = torch.as_tensor(threshold, device=device, dtype=dtype)
    start_idx = max(0, int(start_idx))
    end_idx = min(int(end_idx), int(trace.numel()))
    if end_idx - start_idx < 2:
        nan = torch.tensor(float("nan"), device=device, dtype=dtype)
        return nan, nan, nan
    peak_idx = int(max(start_idx, min(int(peak_idx), end_idx - 1)))
    above = trace >= thr

    up_time = None
    for i in range(peak_idx - 1, start_idx - 1, -1):
        if bool((trace[i] < thr).item()) and bool((trace[i + 1] >= thr).item()):
            frac = ((thr - trace[i]) / (trace[i + 1] - trace[i] + 1e-12)).clamp(
                0.0, 1.0
            )
            up_time = (
                torch.as_tensor(float(i), device=device, dtype=dtype) + frac
            ) * dt
            break
    if up_time is None:
        if bool(above[start_idx].item()) and bool(above[peak_idx].item()):
            up_time = torch.as_tensor(float(start_idx), device=device, dtype=dtype) * dt
        else:
            nan = torch.tensor(float("nan"), device=device, dtype=dtype)
            return nan, nan, nan

    down_time = None
    for i in range(peak_idx, end_idx - 1):
        if bool((trace[i] >= thr).item()) and bool((trace[i + 1] < thr).item()):
            frac = ((trace[i] - thr) / (trace[i] - trace[i + 1] + 1e-12)).clamp(
                0.0, 1.0
            )
            down_time = (
                torch.as_tensor(float(i), device=device, dtype=dtype) + frac
            ) * dt
            break
    if down_time is None:
        if bool(above[end_idx - 1].item()) and bool(above[peak_idx].item()):
            down_time = (
                torch.as_tensor(float(end_idx - 1), device=device, dtype=dtype) * dt
            )
        else:
            nan = torch.tensor(float("nan"), device=device, dtype=dtype)
            return nan, nan, nan

    return down_time - up_time, up_time, down_time


@torch.no_grad()
def hard_action_potential_width(
    V: torch.Tensor,  # (T, F, C)
    dt_ms: float | torch.Tensor,
    node_mask: torch.Tensor | None = None,
    *,
    V_spk: float = 0.0,
    dv_th: float | None = None,
    mode: Literal["half_height", "half_peak_to_peak"] = "half_height",
    baseline_pre_ms: float = 1.0,
    baseline_guard_ms: float = 0.15,
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
) -> Dict[str, torch.Tensor]:
    """
    Hard AP half-width and baseline-relative full-width descriptors.

    This is the non-differentiable reference analogue of ``action_potential_width``. It detects
    the first hard threshold crossing, estimates baseline/peak/trough from hard windows, and
    computes APD-style widths from threshold crossing times around the peak.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, F, C).")
    T, Fibs, C = V.shape
    device, dtype = V.device, V.dtype
    dt = torch.as_tensor(dt_ms, device=device, dtype=dtype)
    if full_width_pre_ms is None:
        full_width_pre_ms = width_pre_ms

    arr = hard_spike_arrival_times(V, dt, V_th=V_spk, dv_th=dv_th)
    t_cross = arr["t_cross_ms"]
    has = arr["has_crossing"]

    width_comp = torch.full((Fibs, C), float("nan"), device=device, dtype=dtype)
    full_width_comp = torch.full_like(width_comp, float("nan"))
    t_peak = torch.full_like(width_comp, float("nan"))
    V_base = torch.full_like(width_comp, float("nan"))
    V_low = torch.full_like(width_comp, float("nan"))
    V_peak = torch.full_like(width_comp, float("nan"))
    V_half = torch.full_like(width_comp, float("nan"))
    amp = torch.full_like(width_comp, float("nan"))

    for f in range(Fibs):
        for c in range(C):
            if not bool(has[f, c].item()):
                continue
            tc = float(t_cross[f, c].item())
            base0 = max(0, int(np.floor((tc - baseline_pre_ms) / float(dt.item()))))
            base1 = max(0, int(np.ceil((tc - baseline_guard_ms) / float(dt.item()))))
            if base1 <= base0:
                continue
            Vb = V[base0:base1, f, c].mean()
            V_base[f, c] = Vb

            peak0 = max(0, int(np.floor((tc - peak_pre_ms) / float(dt.item()))))
            peak1 = min(T, int(np.ceil((tc + peak_post_ms) / float(dt.item()))) + 1)
            if peak1 <= peak0:
                continue
            local = V[peak0:peak1, f, c]
            pk_rel = int(local.argmax().item())
            pk_idx = peak0 + pk_rel
            Vp = V[pk_idx, f, c]
            V_peak[f, c] = Vp
            t_peak[f, c] = (
                torch.as_tensor(float(pk_idx), device=device, dtype=dtype) * dt
            )

            if mode == "half_height":
                Vl = Vb
            elif mode == "half_peak_to_peak":
                tr0 = max(
                    0,
                    int(
                        np.floor(
                            (float(t_peak[f, c].item()) - trough_pre_ms)
                            / float(dt.item())
                        )
                    ),
                )
                tr1 = min(
                    T,
                    int(
                        np.ceil(
                            (float(t_peak[f, c].item()) + trough_post_ms)
                            / float(dt.item())
                        )
                    )
                    + 1,
                )
                Vl = V[tr0:tr1, f, c].min() if tr1 > tr0 else Vb
            else:
                raise ValueError("mode must be 'half_height' or 'half_peak_to_peak'.")
            V_low[f, c] = Vl
            amp_fc = Vp - Vl
            amp[f, c] = amp_fc
            if float(amp_fc.item()) < amp_min_mV:
                continue
            Vh = Vl + half_level * amp_fc
            V_half[f, c] = Vh

            tp = float(t_peak[f, c].item())
            w0 = max(0, int(np.floor((tp - width_pre_ms) / float(dt.item()))))
            w1 = min(T, int(np.ceil((tp + width_post_ms) / float(dt.item()))) + 1)
            width_comp[f, c], _, _ = _hard_width_crossings_around_peak(
                V[:, f, c], dt, Vh, pk_idx, w0, w1
            )

            fw0 = max(0, int(np.floor((tp - full_width_pre_ms) / float(dt.item()))))
            fw1 = min(T, int(np.ceil((tp + full_width_post_ms) / float(dt.item()))) + 1)
            Vfull = Vb + torch.as_tensor(baseline_margin_mV, device=device, dtype=dtype)
            full_width_comp[f, c], _, _ = _hard_width_crossings_around_peak(
                V[:, f, c], dt, Vfull, pk_idx, fw0, fw1
            )

    if node_mask is None:
        mask = torch.ones((Fibs, C), device=device, dtype=torch.bool)
    else:
        m = node_mask.to(device=device)
        if m.shape == (C,):
            mask = (m != 0)[None, :].expand(Fibs, C)
        elif m.shape == (Fibs, C):
            mask = m != 0
        else:
            raise ValueError("node_mask must have shape (C,) or (F, C).")

    valid_width = torch.isfinite(width_comp) & mask
    valid_full = torch.isfinite(full_width_comp) & mask
    width = torch.full((Fibs,), float("nan"), device=device, dtype=dtype)
    full_width = torch.full_like(width, float("nan"))
    for f in range(Fibs):
        if valid_width[f].any():
            width[f] = width_comp[f, valid_width[f]].mean()
        if valid_full[f].any():
            full_width[f] = full_width_comp[f, valid_full[f]].mean()

    return {
        "width_ms": width,
        "width_ms_comp": width_comp,
        "full_width_ms": full_width,
        "full_width_ms_comp": full_width_comp,
        "t_cross_ms": t_cross,
        "t_peak_ms": t_peak,
        "V_base_mV": V_base,
        "V_low_mV": V_low,
        "V_peak_mV": V_peak,
        "V_half_mV": V_half,
        "amp_mV": amp,
        "has_crossing": has,
        "weights": mask,
    }


@torch.no_grad()
def hard_chronaxie_from_trials(
    V: torch.Tensor,  # (T, P, C)
    amplitudes: torch.Tensor,
    pws_ms: torch.Tensor,
    dt_ms: float | torch.Tensor | None = None,
    *,
    V_spk: float = 0.0,
    use_dv_gate: bool = True,
    dv_spk: float = 10.0,
    node_mask: torch.Tensor | None = None,  # (C,)
    time_window: tuple[int, int] | None = None,
    time_window_ms: tuple[float, float] | None = None,
    strength: torch.Tensor | None = None,
    use_abs_strength: bool = False,
    pw_round_decimals: int | None = None,
    threshold_method: Literal["onset", "onset_midpoint"] = "onset_midpoint",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Hard non-differentiable rheobase/chronaxie descriptor from strength-duration trials.

    Activation is determined by a hard spike detector in selected readout compartments. Per-pulse
    width thresholds are empirical onset boundaries, and rheobase/chronaxie are fit using the
    Weiss charge-duration relation. Use this as the hard reference for ``chronaxie_from_trials``.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, P, C).")
    _, P, C = V.shape
    device, dtype = V.device, V.dtype
    amplitudes = amplitudes.to(device=device, dtype=dtype)
    pws_ms = pws_ms.to(device=device, dtype=dtype)
    if amplitudes.shape != (P,) or pws_ms.shape != (P,):
        raise ValueError("amplitudes and pws_ms must have shape (P,).")
    if time_window_ms is not None and dt_ms is None:
        raise ValueError("dt_ms is required when time_window_ms is used.")
    dt = torch.as_tensor(1.0 if dt_ms is None else dt_ms, device=device, dtype=dtype)

    S = amplitudes if strength is None else strength.to(device=device, dtype=dtype)
    if S.shape != (P,):
        raise ValueError("strength must have shape (P,).")
    if use_abs_strength:
        S = S.abs()

    arr = hard_spike_arrival_times(
        V,
        dt,
        V_th=V_spk,
        dv_th=dv_spk if use_dv_gate else None,
        time_window=time_window,
        time_window_ms=time_window_ms,
    )
    active_comp = arr["has_crossing"]  # (P, C), treating P as the fiber dimension
    if node_mask is None:
        mask = torch.ones((P, C), device=device, dtype=torch.bool)
    else:
        m = node_mask.to(device=device)
        if m.shape != (C,):
            raise ValueError("node_mask must have shape (C,) for trial data.")
        mask = (m != 0)[None, :].expand(P, C)
    p_active = (active_comp & mask).any(dim=-1).to(dtype=dtype)  # (P,)

    if pw_round_decimals is not None:
        factor = float(10**pw_round_decimals)
        pws_group = torch.round(pws_ms * factor) / factor
    else:
        pws_group = pws_ms
    pw_unique_ms, group_id = torch.unique(pws_group, sorted=True, return_inverse=True)
    D = int(pw_unique_ms.numel())
    if D < 2:
        raise ValueError("Need at least 2 distinct pulse widths to estimate chronaxie.")

    I_th = torch.full((D,), float("nan"), device=device, dtype=dtype)
    I_on = torch.full_like(I_th, float("nan"))
    I_pre = torch.full_like(I_th, float("nan"))
    I_last = torch.full_like(I_th, float("nan"))
    I_post = torch.full_like(I_th, float("nan"))
    I_block = torch.full_like(I_th, float("nan"))
    w = torch.zeros((D,), device=device, dtype=dtype)

    for g in range(D):
        idx = torch.where(group_id == g)[0]
        Sg = S[idx]
        pg = p_active[idx].bool()
        perm = torch.argsort(Sg)
        Ssort = Sg[perm]
        psort = pg[perm]
        if not bool(psort.any().item()):
            continue
        first_active = int(torch.where(psort)[0][0].item())
        I_on[g] = Ssort[first_active]
        if first_active > 0:
            I_pre[g] = Ssort[first_active - 1]
            w[g] = 1.0
            I_th[g] = (
                I_on[g] if threshold_method == "onset" else 0.5 * (I_pre[g] + I_on[g])
            )
        else:
            I_th[g] = I_on[g]
            w[g] = 0.5
        last_active = int(torch.where(psort)[0][-1].item())
        I_last[g] = Ssort[last_active]
        if last_active < len(Ssort) - 1:
            I_post[g] = Ssort[last_active + 1]
            I_block[g] = 0.5 * (I_last[g] + I_post[g])

    finite = torch.isfinite(I_th) & (w > 0)
    if finite.sum() < 2:
        rheobase = torch.tensor(float("nan"), device=device, dtype=dtype)
        chron = torch.tensor(float("nan"), device=device, dtype=dtype)
        weiss_mse = torch.tensor(float("nan"), device=device, dtype=dtype)
    else:
        d = pw_unique_ms[finite]
        Q = d * I_th[finite]
        wf = w[finite]
        W = wf.sum() + eps
        db = (wf * d).sum() / W
        Qb = (wf * Q).sum() / W
        var = (wf * (d - db) ** 2).sum() / W
        cov = (wf * (d - db) * (Q - Qb)).sum() / W
        slope = cov / (var + eps)
        intercept = Qb - slope * db
        rheobase = slope
        chron = intercept / (slope + eps)
        weiss_mse = (wf * (Q - (slope * d + intercept)) ** 2).sum() / W

    return {
        "chronaxie_ms": chron,
        "rheobase": rheobase,
        "pw_unique_ms": pw_unique_ms,
        "I_th": I_th,
        "Q_th": pw_unique_ms * I_th,
        "p_spike_trial": p_active,
        "pw_group_id": group_id,
        "pw_weight": w,
        "boundaries": {
            "I_on": I_on,
            "I_pre_inactive": I_pre,
            "I_last_active": I_last,
            "I_post_inactive": I_post,
            "I_block_mid": I_block,
        },
        "fit": {"weiss_mse": weiss_mse},
    }


@torch.no_grad()
def hard_paired_pulse_recovery_from_trials(
    V: torch.Tensor,  # (T, P, C)
    test_amplitudes: torch.Tensor,
    isis_ms: torch.Tensor,
    dt_ms: float | torch.Tensor,
    *,
    condition_pulse_time_ms: float | torch.Tensor = 0.0,
    test_pulse_times_ms: torch.Tensor | None = None,
    response_window_ms: tuple[float, float] = (0.1, 10.0),
    baseline_threshold: float | torch.Tensor | None = None,
    reference_latency_ms: float | torch.Tensor | None = None,
    reference_velocity_m_per_s: float | torch.Tensor | None = None,
    lengths_um: Optional[LengthLike] = None,  # scalar, (C,), or (P, C)
    node_mask: torch.Tensor | None = None,  # (C,) or (P, C)
    V_th: float = 0.0,
    dv_th: float | None = 10.0,
    strength: torch.Tensor | None = None,
    use_abs_strength: bool = False,
    isi_round_decimals: int | None = None,
    threshold_method: Literal["onset", "onset_midpoint"] = "onset_midpoint",
    interpolate: bool = True,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Hard reference descriptor for paired-pulse threshold recovery cycles.

    Each trial contains a conditioning pulse and a test pulse. The hard descriptor scores only
    the test-pulse response in a pulse-locked window, extracts empirical second-pulse thresholds
    per ISI when test amplitude is swept, and optionally computes hard latency / velocity recovery.
    This is the non-differentiable reference analogue of ``paired_pulse_recovery_from_trials``.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, P, C).")
    _, P, C = V.shape
    device, dtype = V.device, V.dtype
    dt = torch.as_tensor(dt_ms, device=device, dtype=dtype)
    test_amplitudes = test_amplitudes.to(device=device, dtype=dtype)
    isis_ms = isis_ms.to(device=device, dtype=dtype)
    if test_amplitudes.shape != (P,) or isis_ms.shape != (P,):
        raise ValueError("test_amplitudes and isis_ms must have shape (P,).")

    if test_pulse_times_ms is None:
        test_times = (
            torch.as_tensor(condition_pulse_time_ms, device=device, dtype=dtype)
            + isis_ms
        )
    else:
        test_times = test_pulse_times_ms.to(device=device, dtype=dtype)
        if test_times.shape != (P,):
            raise ValueError("test_pulse_times_ms must have shape (P,).")

    S = test_amplitudes if strength is None else strength.to(device=device, dtype=dtype)
    if S.shape != (P,):
        raise ValueError("strength must have shape (P,).")
    if use_abs_strength:
        S = S.abs()

    if node_mask is None:
        mask = torch.ones((P, C), device=device, dtype=torch.bool)
    else:
        m = node_mask.to(device=device)
        if m.shape == (C,):
            mask = (m != 0)[None, :].expand(P, C)
        elif m.shape == (P, C):
            mask = m != 0
        else:
            raise ValueError("node_mask must have shape (C,) or (P, C).")

    if lengths_um is not None:
        L = _coerce_lengths_um(lengths_um, Fibs=P, C=C, device=device, dtype=dtype)
        x_um = torch.cumsum(L, dim=-1) - 0.5 * L
    else:
        x_um = None

    p_response_trial = torch.zeros((P,), device=device, dtype=dtype)
    t_cross_comp = torch.full((P, C), float("nan"), device=device, dtype=dtype)
    latency_ms_comp = torch.full((P, C), float("nan"), device=device, dtype=dtype)
    latency_ms_trial = torch.full((P,), float("nan"), device=device, dtype=dtype)
    v_m_per_s_trial = (
        torch.full((P,), float("nan"), device=device, dtype=dtype)
        if x_um is not None
        else None
    )
    speed_m_per_s_trial = (
        torch.full((P,), float("nan"), device=device, dtype=dtype)
        if x_um is not None
        else None
    )

    for p in range(P):
        win0 = float(test_times[p].item() + response_window_ms[0])
        win1 = float(test_times[p].item() + response_window_ms[1])
        arr = hard_spike_arrival_times(
            V[:, p : p + 1, :],
            dt,
            V_th=V_th,
            dv_th=dv_th,
            time_window_ms=(win0, win1),
            interpolate=interpolate,
        )
        tc = arr["t_cross_ms"][0]
        hs = arr["has_crossing"][0]
        t_cross_comp[p] = tc
        latency_ms_comp[p] = tc - test_times[p]
        valid = hs & mask[p]
        if valid.any():
            p_response_trial[p] = 1.0
            earliest = torch.where(valid, tc, torch.full_like(tc, float("inf"))).min()
            latency_ms_trial[p] = earliest - test_times[p]
        if x_um is not None and valid.sum() >= 2:
            idx = torch.where(valid)[0]
            tf = tc[idx]
            xf = x_um[p, idx]
            tb = tf.mean()
            xb = xf.mean()
            denom = ((tf - tb) ** 2).mean()
            if float(denom.item()) > eps:
                vel_um = ((tf - tb) * (xf - xb)).mean() / denom
                v_m_per_s_trial[p] = 1e-3 * vel_um
                speed_m_per_s_trial[p] = v_m_per_s_trial[p].abs()

    if isi_round_decimals is not None:
        factor = float(10**isi_round_decimals)
        isis_group = torch.round(isis_ms * factor) / factor
    else:
        isis_group = isis_ms
    isi_unique_ms, group_id, isi_counts = torch.unique(
        isis_group, sorted=True, return_inverse=True, return_counts=True
    )
    D = int(isi_unique_ms.numel())

    I_th_test = torch.full((D,), float("nan"), device=device, dtype=dtype)
    I_on = torch.full_like(I_th_test, float("nan"))
    I_pre = torch.full_like(I_th_test, float("nan"))
    I_last = torch.full_like(I_th_test, float("nan"))
    I_post = torch.full_like(I_th_test, float("nan"))
    I_block = torch.full_like(I_th_test, float("nan"))
    isi_weight = torch.zeros((D,), device=device, dtype=dtype)
    p_by_isi = torch.zeros((D,), device=device, dtype=dtype)
    latency_by_isi = torch.full((D,), float("nan"), device=device, dtype=dtype)
    v_by_isi = (
        torch.full((D,), float("nan"), device=device, dtype=dtype)
        if v_m_per_s_trial is not None
        else None
    )

    for g in range(D):
        idx = torch.where(group_id == g)[0]
        Sg = S[idx]
        pg = p_response_trial[idx].bool()
        p_by_isi[g] = p_response_trial[idx].mean()
        finite_lat = torch.isfinite(latency_ms_trial[idx])
        if finite_lat.any():
            latency_by_isi[g] = latency_ms_trial[idx][finite_lat].mean()
        if v_m_per_s_trial is not None:
            finite_v = torch.isfinite(v_m_per_s_trial[idx])
            if finite_v.any():
                v_by_isi[g] = v_m_per_s_trial[idx][finite_v].mean()

        perm = torch.argsort(Sg)
        Ssort = Sg[perm]
        psort = pg[perm]
        if not bool(psort.any().item()):
            continue
        first_active = int(torch.where(psort)[0][0].item())
        I_on[g] = Ssort[first_active]
        if first_active > 0:
            I_pre[g] = Ssort[first_active - 1]
            isi_weight[g] = 1.0
            I_th_test[g] = (
                I_on[g] if threshold_method == "onset" else 0.5 * (I_pre[g] + I_on[g])
            )
        else:
            I_th_test[g] = I_on[g]
            isi_weight[g] = 0.5
        last_active = int(torch.where(psort)[0][-1].item())
        I_last[g] = Ssort[last_active]
        if last_active < len(Ssort) - 1:
            I_post[g] = Ssort[last_active + 1]
            I_block[g] = 0.5 * (I_last[g] + I_post[g])

    baseline_ref = (
        None
        if baseline_threshold is None
        else torch.as_tensor(baseline_threshold, device=device, dtype=dtype)
    )
    if baseline_ref is not None:
        if baseline_ref.ndim == 0:
            baseline_ref = baseline_ref.expand(D)
        elif baseline_ref.shape != (D,):
            raise ValueError("baseline_threshold must be scalar or shape (D,).")
        threshold_ratio = I_th_test / (baseline_ref + eps)
        threshold_percent_change = 100.0 * (threshold_ratio - 1.0)
    else:
        threshold_ratio = None
        threshold_percent_change = None

    latency_ref = (
        None
        if reference_latency_ms is None
        else torch.as_tensor(reference_latency_ms, device=device, dtype=dtype)
    )
    if latency_ref is not None:
        latency_ref = latency_ref.expand(D) if latency_ref.ndim == 0 else latency_ref
        latency_shift_ms = latency_by_isi - latency_ref
        latency_shift_percent = 100.0 * latency_shift_ms / (latency_ref + eps)
    else:
        latency_shift_ms = None
        latency_shift_percent = None

    velocity_percent_change = None
    if v_by_isi is not None and reference_velocity_m_per_s is not None:
        vref = torch.as_tensor(reference_velocity_m_per_s, device=device, dtype=dtype)
        vref = vref.expand(D) if vref.ndim == 0 else vref
        velocity_percent_change = 100.0 * (v_by_isi - vref) / (vref + eps)

    return {
        "isi_unique_ms": isi_unique_ms,
        "I_th_test": I_th_test,
        "threshold_ratio": threshold_ratio,
        "threshold_percent_change": threshold_percent_change,
        "p_response_trial": p_response_trial,
        "p_response_by_isi": p_by_isi,
        "isi_weight": isi_weight,
        "isi_group_id": group_id,
        "isi_counts": isi_counts,
        "test_pulse_times_ms": test_times,
        "test_strength": S,
        "boundaries": {
            "I_on": I_on,
            "I_pre_inactive": I_pre,
            "I_last_active": I_last,
            "I_post_inactive": I_post,
            "I_block_mid": I_block,
        },
        "t_cross_comp": t_cross_comp,
        "latency_ms_comp": latency_ms_comp,
        "latency_ms_trial": latency_ms_trial,
        "latency_ms_by_isi": latency_by_isi,
        "latency_shift_ms": latency_shift_ms,
        "latency_shift_percent": latency_shift_percent,
        "v_m_per_s_trial": v_m_per_s_trial,
        "speed_m_per_s_trial": speed_m_per_s_trial,
        "v_m_per_s_by_isi": v_by_isi,
        "velocity_percent_change": velocity_percent_change,
    }


@torch.no_grad()
def hard_activity_dependent_slowing(
    V: torch.Tensor,  # (T, F, C)
    pulse_times_ms: torch.Tensor | Sequence[float],
    dt_ms: float | torch.Tensor,
    *,
    lengths_um: Optional[LengthLike] = None,
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
    window_margin_ms: float | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Hard non-differentiable activity-dependent slowing descriptor.

    This optimized hard reference is the conventional threshold-crossing analogue of
    ``activity_dependent_slowing``. It gathers pulse-locked local windows for all pulses and
    fibers at once, detects the first upward threshold crossing in each selected compartment,
    computes latency relative to pulse onset, and optionally estimates conduction velocity from
    hard per-compartment crossing times.

    It is intended for Experiment 1 hard-vs-surrogate comparisons, not gradient-based fitting.
    """
    if V.ndim != 3:
        raise ValueError("V must have shape (T, F, C).")
    T, Fibs, C = V.shape
    if T < 2:
        raise ValueError("Need at least 2 time samples.")
    if response_window_ms[1] <= response_window_ms[0]:
        raise ValueError("response_window_ms must satisfy end > start.")
    device, dtype = V.device, V.dtype
    dt = torch.as_tensor(dt_ms, device=device, dtype=dtype)
    dt_float = float(dt.detach().cpu().item())

    pulse_times = _coerce_pulse_times_ms(
        pulse_times_ms, Fibs=Fibs, device=device, dtype=dtype
    )
    N = pulse_times.shape[1]

    if baseline_pulse_indices is None:
        baseline_idx = torch.arange(baseline_n_pulses, device=device, dtype=torch.long)
    else:
        baseline_idx = torch.as_tensor(
            baseline_pulse_indices, device=device, dtype=torch.long
        ).flatten()
    tail_k = min(int(tail_n_pulses), N)
    tail_idx = torch.arange(N - tail_k, N, device=device, dtype=torch.long)

    if node_mask is None:
        mask = torch.ones((Fibs, C), device=device, dtype=torch.bool)
    else:
        m = node_mask.to(device=device)
        if m.shape == (C,):
            mask = (m != 0)[None, :].expand(Fibs, C)
        elif m.shape == (Fibs, C):
            mask = m != 0
        else:
            raise ValueError("node_mask must have shape (C,) or (F, C).")

    if lengths_um is not None:
        L = _coerce_lengths_um(lengths_um, Fibs=Fibs, C=C, device=device, dtype=dtype)
        x_um = torch.cumsum(L, dim=-1) - 0.5 * L
    else:
        x_um = None

    win_start = pulse_times + torch.as_tensor(
        response_window_ms[0], device=device, dtype=dtype
    )
    win_end = pulse_times + torch.as_tensor(
        response_window_ms[1], device=device, dtype=dtype
    )
    margin = dt_float if window_margin_ms is None else float(window_margin_ms)

    V_win, t_win, valid = _gather_time_windows_FNKC(
        V,
        dt,
        win_start,
        win_end,
        margin_ms=margin,
        sample_offset=0.0,
    )  # (F, N, K, C), (F, N, K)

    v0 = V_win[:, :, :-1, :]
    v1 = V_win[:, :, 1:, :]
    t0 = t_win[:, :, :-1]
    valid_pair = valid[:, :, :-1] & valid[:, :, 1:]

    crossings = (v0 < V_th) & (v1 >= V_th) & valid_pair[:, :, :, None]
    if dv_th is not None:
        dV_pair = (v1 - v0) / dt
        crossings = crossings & (dV_pair >= dv_th)

    has_comp = crossings.any(dim=2)  # (F, N, C)
    first_idx = crossings.to(torch.int64).argmax(dim=2)  # 0 if absent
    gather_idx = first_idx[:, :, None, :]  # (F, N, 1, C)

    v0_sel = v0.gather(2, gather_idx).squeeze(2)
    v1_sel = v1.gather(2, gather_idx).squeeze(2)
    t0_exp = t0[:, :, :, None].expand_as(v0)
    t0_sel = t0_exp.gather(2, gather_idx).squeeze(2)

    if interpolate:
        frac = (
            (torch.as_tensor(V_th, device=device, dtype=dtype) - v0_sel)
            / (v1_sel - v0_sel + 1e-12)
        ).clamp(0.0, 1.0)
        t_cross = t0_sel + frac * dt
    else:
        t_cross = t0_sel + dt

    nan_fnc = torch.full_like(t_cross, float("nan"))
    t_cross = torch.where(has_comp, t_cross, nan_fnc)  # (F, N, C)
    valid_selected = has_comp & mask[:, None, :]

    p_success = valid_selected.any(dim=-1).to(dtype=dtype)  # (F, N)
    inf = torch.full_like(t_cross, float("inf"))
    earliest = torch.where(valid_selected, t_cross, inf).min(dim=-1).values
    has_any = torch.isfinite(earliest)
    arrival_time_ms = torch.where(
        has_any, earliest, torch.full_like(earliest, float("nan"))
    )
    latency_ms = arrival_time_ms - pulse_times

    t_cross_comp = t_cross.permute(1, 0, 2).contiguous()  # (N, F, C)
    has_comp_out = has_comp.permute(1, 0, 2).contiguous()

    v_m_per_s = None
    speed_m_per_s = None
    if x_um is not None:
        w = valid_selected.to(dtype=dtype)
        W = w.sum(dim=-1)  # (F, N)
        tc0 = torch.where(valid_selected, t_cross, torch.zeros_like(t_cross))
        t_bar = (w * tc0).sum(dim=-1, keepdim=True) / torch.clamp(
            W[:, :, None], min=1.0
        )
        x = x_um[:, None, :]
        x_bar = (w * x).sum(dim=-1, keepdim=True) / torch.clamp(W[:, :, None], min=1.0)
        dtc = torch.where(valid_selected, t_cross - t_bar, torch.zeros_like(t_cross))
        dxc = x - x_bar
        var_t = (w * dtc * dtc).sum(dim=-1) / torch.clamp(W, min=1.0)
        cov = (w * dtc * dxc).sum(dim=-1) / torch.clamp(W, min=1.0)
        vel_um = cov / (var_t + eps)
        valid_vel = (W >= 2.0) & (var_t > eps)
        v_m_per_s = torch.where(
            valid_vel, 1e-3 * vel_um, torch.full_like(vel_um, float("nan"))
        )
        speed_m_per_s = v_m_per_s.abs()

    ref_latency = _as_F_vector(
        reference_latency_ms,
        Fibs=Fibs,
        device=device,
        dtype=dtype,
        name="reference_latency_ms",
    )
    if ref_latency is None:
        baseline_latency = _finite_mean_over_indices(latency_ms, baseline_idx, eps=eps)
    else:
        baseline_latency = ref_latency

    latency_shift_ms = latency_ms - baseline_latency[:, None]
    ads_percent = 100.0 * latency_shift_ms / (baseline_latency[:, None] + eps)

    if N >= 2:
        input_isi_ms = pulse_times[:, 1:] - pulse_times[:, :-1]
        output_isi_ms = arrival_time_ms[:, 1:] - arrival_time_ms[:, :-1]
        isi_error_ms = output_isi_ms - input_isi_ms
        instantaneous_frequency_hz = 1000.0 / (input_isi_ms + eps)
    else:
        input_isi_ms = torch.empty((Fibs, 0), device=device, dtype=dtype)
        output_isi_ms = torch.empty((Fibs, 0), device=device, dtype=dtype)
        isi_error_ms = torch.empty((Fibs, 0), device=device, dtype=dtype)
        instantaneous_frequency_hz = torch.empty((Fibs, 0), device=device, dtype=dtype)

    final_ads_percent = ads_percent[:, -1]
    tail_ads_percent = _finite_mean_over_indices(ads_percent, tail_idx, eps=eps)

    baseline_velocity = None
    velocity_change_percent = None
    velocity_slowing_percent = None
    final_velocity_slowing_percent = None
    tail_velocity_slowing_percent = None
    if v_m_per_s is not None:
        ref_velocity = _as_F_vector(
            reference_velocity_m_per_s,
            Fibs=Fibs,
            device=device,
            dtype=dtype,
            name="reference_velocity_m_per_s",
        )
        if ref_velocity is None:
            baseline_velocity = _finite_mean_over_indices(
                v_m_per_s, baseline_idx, eps=eps
            )
        else:
            baseline_velocity = ref_velocity
        velocity_change_percent = (
            100.0
            * (v_m_per_s - baseline_velocity[:, None])
            / (baseline_velocity[:, None] + eps)
        )
        velocity_slowing_percent = -velocity_change_percent
        final_velocity_slowing_percent = velocity_slowing_percent[:, -1]
        tail_velocity_slowing_percent = _finite_mean_over_indices(
            velocity_slowing_percent, tail_idx, eps=eps
        )

    return {
        "pulse_times_ms": pulse_times,
        "p_success": p_success,
        "p_fail": 1.0 - p_success,
        "follow_fraction": p_success.mean(dim=-1),
        "success_count": p_success.sum(dim=-1),
        "latency_ms": latency_ms,
        "baseline_latency_ms": baseline_latency,
        "latency_shift_ms": latency_shift_ms,
        "ads_percent": ads_percent,
        "latency_slowing_percent": ads_percent,
        "final_ads_percent": final_ads_percent,
        "tail_ads_percent": tail_ads_percent,
        "arrival_time_ms": arrival_time_ms,
        "input_isi_ms": input_isi_ms,
        "instantaneous_frequency_hz": instantaneous_frequency_hz,
        "output_isi_ms": output_isi_ms,
        "isi_error_ms": isi_error_ms,
        "t_cross_ms_comp": t_cross_comp,
        "has_crossing_comp": has_comp_out,
        "weights": mask,
        "baseline_pulse_indices": baseline_idx,
        "tail_pulse_indices": tail_idx,
        "v_m_per_s": v_m_per_s,
        "speed_m_per_s": speed_m_per_s,
        "baseline_velocity_m_per_s": baseline_velocity,
        "velocity_change_percent": velocity_change_percent,
        "velocity_slowing_percent": velocity_slowing_percent,
        "final_velocity_slowing_percent": final_velocity_slowing_percent,
        "tail_velocity_slowing_percent": tail_velocity_slowing_percent,
    }
