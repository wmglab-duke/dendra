import torch
import torch.nn.functional as F


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
