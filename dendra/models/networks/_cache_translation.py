"""Shared cold nearest-step indices for legacy cache translation."""

import math

import torch


def _round_half_up_time_indices(
    count: int,
    numerator_dt: float,
    denominator_dt: float,
    *,
    device,
) -> torch.Tensor:
    """Return robust nearest-step indices without device float64 requirements.

    Cache translation is a cold path, so calculate floating-point positions on
    CPU. A scale-aware tolerance keeps documented decimal half-up ties stable.
    """
    count = int(count)
    if count <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    positions = torch.arange(count, dtype=torch.float64) * (
        float(numerator_dt) / float(denominator_dt)
    )
    tolerance = (
        torch.maximum(positions.abs(), torch.ones_like(positions))
        * torch.finfo(torch.float64).eps
        * 8.0
    )
    indices = torch.floor(positions + 0.5 + tolerance).to(torch.long)
    return indices.to(device=device)


def _round_half_up_time_index(
    index: int, numerator_dt: float, denominator_dt: float
) -> int:
    """Scalar counterpart of :func:`_round_half_up_time_indices`."""
    value = float(index) * float(numerator_dt) / float(denominator_dt)
    tolerance = max(abs(value), 1.0) * torch.finfo(torch.float64).eps * 8.0
    return int(math.floor(value + 0.5 + tolerance))
