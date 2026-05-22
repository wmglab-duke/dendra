from typing import Tuple

import torch
import torch.nn.functional as F


def update_active(has_spiked, vm_new, threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    ge = vm_new >= threshold
    spiked = torch.logical_and(ge, ~has_spiked)
    return ge, spiked


def update_active_diff(
    has_spiked: torch.Tensor,  # previous "ge" (bool)
    vm_new: torch.Tensor,  # float
    threshold: torch.Tensor,  # float
    tau: float = 0.1,  # temperature for the surrogate
):
    """
    Returns:
      ge_hard:     bool  (vm_new >= threshold)
      spiked_hard: bool  (rising edge: ge & ~has_spiked)
      ge_gate:     float in [0,1] with STE (forward==ge_hard, backward==sigmoid)
      spk_gate:    float in [0,1] with STE for *rising edge*
    """
    x = (vm_new - threshold) / tau
    s = torch.sigmoid(x)  # smooth "is-above-threshold"

    ge_hard = vm_new >= threshold  # bool
    spiked_hard = ge_hard & (~has_spiked)  # bool rising edge

    # Straight-through gates:
    # - ge_gate forward equals ge_hard; backward follows s
    ge_gate = ge_hard.to(s.dtype) + (s - s.detach())

    # - rising-edge gate: soft approx is s * (1 - has_spiked)
    s_rise = s * (1.0 - has_spiked.to(s.dtype))
    spk_gate = spiked_hard.to(s.dtype) + (s_rise - s_rise.detach())

    return ge_hard, ge_gate, spk_gate


def sigmoid_ste(x: torch.Tensor, tau: torch.Tensor):
    # Ensure positivity without clamp-induced flat regions (optional)
    tau = F.softplus(tau) + 1e-6

    s = torch.sigmoid(x / tau)  # surrogate
    h = (x >= 0).to(x.dtype)  # hard forward
    return (h - s).detach() + s  # forward==h, grad==∂s/∂x and ∂s/∂tau


def crossing_spike(v_old, v_new, threshold, tau=0.1, ste_scale=1.0):
    """
    Detects if the voltage crosses the threshold from below to above, using a surrogate gradient.

    Returns:
      spiked_hard: bool  (rising edge: v_new >= threshold and v_old < threshold)
      spk_gate:    float in [0,1] with STE for *rising edge*
    """
    hard = ((v_old < threshold) & (v_new > threshold)).to(v_new.dtype)
    g0 = torch.sigmoid((v_old - threshold) / tau)
    g1 = torch.sigmoid((v_new - threshold) / tau)
    soft = torch.relu(g1 - g0)
    return hard + ste_scale * (soft - soft.detach())


def level_spike(v_new, threshold, tau=0.1, ste_scale=1.0):
    """
    Detects if the voltage is above the threshold, using a surrogate gradient.

    Returns:
      spiked_hard: bool  (v_new >= threshold)
      spk_gate:    float in [0,1] with STE for v_new >= threshold
    """
    hard = (v_new >= threshold).to(v_new.dtype)
    soft = torch.sigmoid((v_new - threshold) / tau)
    return hard + ste_scale * (soft - soft.detach())
