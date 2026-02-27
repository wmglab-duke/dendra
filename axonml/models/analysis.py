from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ArrayLike1D = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[int]]


def conduction_velocity(
    V: torch.Tensor,  # (T, F, C) membrane potentials
    lengths_um: torch.Tensor,  # (F, C) compartment lengths in um
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

    lengths_um : torch.Tensor
        Compartment physical lengths (in micrometers) with shape ``(F, C)``.
        These are used to compute compartment midpoint positions along each fiber.
        For `ax.Myelinated`, this should be internodal length
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
    This is consistent with AxonML's internal practice of representing lengths in :math:`\mu m`,
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

    >>> out = differentiable_conduction_velocity(
    ...     V, lengths_um, dt_ms,
    ...     node_mask=node_mask,
    ...     V_spk=-10.0,
    ...     beta=60.0,
    ... )
    >>> v = out["v_m_per_s"]          # (F,)
    >>> loss = task_loss(v) + out["reg"]
    >>> loss.backward()

    Add robustness penalties when training:

    >>> out = differentiable_conduction_velocity(
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
    T, Fibs, C = V.shape
    assert lengths_um.shape == (Fibs, C), "lengths_um must be (F, C)"

    device = V.device
    dtype = V.dtype

    if not torch.is_tensor(dt_ms):
        dt_ms = torch.tensor(dt_ms, device=device, dtype=dtype)
    else:
        dt_ms = dt_ms.to(device=device, dtype=dtype)

    # ---- 1) Positions x (um): midpoint of each compartment along the fiber
    x_um = torch.cumsum(
        lengths_um.to(device=device, dtype=dtype), dim=-1
    ) - 0.5 * lengths_um.to(device=device, dtype=dtype)  # (F, C)

    # ---- 2) Time grid (ms)
    t_ms = torch.arange(T, device=device, dtype=dtype) * dt_ms  # (T,)

    # ---- 3) Upstroke-based soft arrival time per compartment
    dV = (V[1:] - V[:-1]) / dt_ms  # (T-1, F, C) in mV/ms
    t_mid = 0.5 * (t_ms[1:] + t_ms[:-1])  # (T-1,)

    U = F.softplus((dV - dv0) / dv_scale)  # (T-1, F, C), >=0
    logits = beta * U - lambda_early * t_mid[:, None, None]  # (T-1, F, C)
    w_time = torch.softmax(logits, dim=0)  # (T-1, F, C)

    t_hat_ms = (w_time * t_mid[:, None, None]).sum(dim=0)  # (F, C)

    # ---- 4) Spike-present confidence p_spike (voltage-based smooth max)
    a = torch.logsumexp(kappa_V * (V - V_spk), dim=0) / kappa_V  # (F, C) in mV
    pV = torch.sigmoid(a / gate_V_scale)  # (F, C) in (0,1)
    p_spike = pV

    if use_dv_gate:
        u = torch.logsumexp(kappa_dv * (dV - dv_spk), dim=0) / kappa_dv  # (F, C) mV/ms
        pDV = torch.sigmoid(u / gate_dv_scale)
        p_spike = p_spike * pDV

    # ---- 5) Optional node_mask restriction
    if node_mask is None:
        nm = torch.ones((Fibs, C), device=device, dtype=dtype)
    else:
        nm = node_mask.to(device=device)
        if nm.dtype == torch.bool:
            nm = nm.to(dtype=dtype)
        else:
            nm = nm.to(dtype=dtype)
        assert nm.shape == (Fibs, C), "node_mask must be (F, C)"

    weights = p_spike * nm  # (F, C)

    # ---- 6) Weighted least-squares fit: x ≈ b + v * t_hat
    W = weights.sum(dim=-1, keepdim=True) + eps  # (F, 1)

    t_bar = (weights * t_hat_ms).sum(dim=-1, keepdim=True) / W  # (F, 1)
    x_bar = (weights * x_um).sum(dim=-1, keepdim=True) / W  # (F, 1)

    dtc = t_hat_ms - t_bar  # (F, C)
    dxc = x_um - x_bar  # (F, C)

    cov = (weights * dtc * dxc).sum(dim=-1) / W.squeeze(-1)  # (F,)
    var_t = (weights * dtc * dtc).sum(dim=-1) / W.squeeze(-1)  # (F,) in ms^2

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
    """
    Chronaxie estimator robust to high-amplitude block by extracting the *lower* activation
    threshold (first activation) per pulse width, rather than a global bracket midpoint.
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
    with optional overlays for (i) estimated strength–duration threshold curves and (ii)
    empirical threshold points derived directly from the binned heatmap.

    This helper is intended to work with the dictionary output from
    ``differentiable_chronaxie_from_trials(...)`` (or a compatible estimator that returns
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

       Notes:
       - This assumes that “activation likelihood increases monotonically with amplitude/strength”
         within each pulse-width column. If your design violates this, empirical thresholds may be
         ambiguous (you can use the monotonicity regularizer during fitting, or inspect manually).
       - If a column has only active trials or only inactive trials (or missing data), the
         corresponding empirical threshold for that column will be undefined (NaN) and not plotted.

    Parameters
    ----------
    chron_out : dict
        Output dictionary from ``differentiable_chronaxie_from_trials`` (or equivalent).
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
