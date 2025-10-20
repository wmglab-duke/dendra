from typing import Tuple

import torch


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


def sigmoid_ste(x, tau):
    # Straight-Through Estimator: forward hard step, backward sigmoid
    # We implement as: y = (x>0).float() with custom grad via sigmoid
    class _STE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, tau):
            ctx.save_for_backward(x, tau)
            return (x >= 0).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            x, tau = ctx.saved_tensors
            s = torch.sigmoid(x / tau.clamp_min(1e-6))
            return grad_output * s * (1 - s) / tau.clamp_min(1e-6), None

    return _STE.apply(x, tau)
