"""Invocation-local waveform sampling without changing accepted solver clocks."""

from __future__ import annotations

import torch

from dendra.models.core import _time_tensor_to


def _accepted_bank_clock(functional, explicit_draws):
    """Native explicit Poisson replay evaluates the affected banks per step."""
    plan = functional._poisson_plan
    return bool(
        explicit_draws
        and plan is not None
        and any(entry.kind in ("intra", "extra") for entry in plan.entries)
    )


def _sampling_time_tape(functional, origin, start, steps, dt, *, clock_dt=None):
    """Match native invocation-origin plus binary64 integer-offset sampling.

    Keep integer offsets global to this host invocation, including across
    internal chunks and checkpoint spans. Native Network input preparation
    first converts its clock to the Population's dtype, then widens for this
    calculation. MPS performs the unsupported binary64 work on CPU.
    """
    device, dtype = functional.device, functional.dtype
    calculation_device = torch.device("cpu") if device.type == "mps" else device
    origin = _time_tensor_to(origin, device=calculation_device, dtype=dtype)
    initial = origin.to(dtype=torch.float64)
    offsets = torch.arange(
        start, start + steps, device=calculation_device, dtype=torch.float64
    )
    if clock_dt is None or not (
        clock_dt.requires_grad
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    ):
        effective_dt = dt
    else:
        value = clock_dt.to(device=calculation_device).to(dtype=torch.float64)
        # The host resolves the native Python timestep before scheduling.
        # Preserve its binary64 forward value and the caller's timing tangent.
        effective_dt = value + (value.new_tensor(dt) - value).detach()
    times = initial + offsets * effective_dt
    return _time_tensor_to(times, device=device, dtype=dtype)


__all__ = []
