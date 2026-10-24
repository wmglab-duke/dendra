"""Stable tensor projection onto real symmetric matrix parameters."""

import torch


def _symmetric_part(value):
    """Average transposed entries without overflowing finite equal values.

    Summing first preserves subnormal entries; halving first is required when
    either operand exceeds half the largest finite value of the solve dtype.
    """
    transposed = value.transpose(-1, -2)
    limits = torch.finfo(value.dtype)
    half_max = limits.max / 2.0
    bounded = (value.abs() <= half_max) & (transposed.abs() <= half_max)
    half_value = value * 0.5
    half_transposed = transposed * 0.5
    extreme = torch.where(
        value.abs() < 2.0 * limits.tiny,
        value + half_transposed,
        torch.where(
            transposed.abs() < 2.0 * limits.tiny,
            half_value + transposed,
            half_value + half_transposed,
        ),
    )
    return torch.where(
        bounded,
        (value + transposed) * 0.5,
        extreme,
    )
