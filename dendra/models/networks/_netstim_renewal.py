"""Shared tensor transitions for NetStim initialization and renewal.

Random sampling, parameter materialization, schedule heaps, and buffer ownership
belong to adapters. These functions only read tensors and return new state, so
native and functional execution use the same hard and surrogate event semantics.
"""

from typing import NamedTuple

import torch

from dendra.models.core import _time_tensor_to


class NetStimState(NamedTuple):
    next_stoch_time: torch.Tensor
    spike_counts: torch.Tensor
    t_last: torch.Tensor


class NetStimTransition(NamedTuple):
    state: NetStimState
    spikes: torch.Tensor
    spike_gate: torch.Tensor
    heap_fires: torch.Tensor


def sample_netstim_exponential(generator, shape, *, device, dtype):
    """Draw the shared native full-shape unit exponential from an owned stream."""
    uniform = torch.rand(
        shape, generator=generator, device=device, dtype=dtype
    ).clamp_min(torch.finfo(dtype).tiny)
    return -torch.log(uniform)


def broadcast_netstim_time(t, shape, *, device, dtype):
    """Interpret scalar, per-batch, or per-generator simulation times.

    A time tensor matching all leading batch dimensions denotes one time per
    batch item, including when its shape could also broadcast over generators.
    Hidden vmap dimensions remain hidden from this shape policy.
    """
    if (
        isinstance(t, torch.Tensor)
        and t.dtype == torch.float64
        and t.device.type == "cpu"
        and device is not None
        and torch.device(device).type == "mps"
    ):
        # Inductor can transfer a combined cast and copy before narrowing.
        t = _time_tensor_to(t, device=device, dtype=dtype)
    else:
        t = torch.as_tensor(t, device=device, dtype=dtype)
    if tuple(t.shape) == shape:
        return t
    if len(shape) > 1 and tuple(t.shape) == shape[:-1]:
        return t.unsqueeze(-1)
    try:
        return torch.broadcast_to(t, shape)
    except RuntimeError as direct_exc:
        if t.ndim > 0:
            try:
                return torch.broadcast_to(t.unsqueeze(-1), shape)
            except RuntimeError:
                pass
        raise ValueError(
            f"t with shape {tuple(t.shape)} cannot be broadcast to "
            f"NetStim state shape {shape}. Use a scalar, shape "
            f"{shape[:-1]} for per-batch times, or shape {shape}."
        ) from direct_exc


def netstim_initial_state(interval, start, noise, exp_rand):
    """Reset renewal clocks from an explicit unit-exponential draw.

    The native initialization kick has always been outside autograd. Detach
    every input explicitly so reverse and forward differentiation agree. A
    zero draw expresses deterministic initialization without consuming RNG.
    """
    noise = noise.detach()
    baseline = start.detach()
    randomized = baseline + (noise * interval.detach()) * exp_rand.detach()
    # Native initialization bypasses interval evaluation entirely when every
    # generator is deterministic. Preserve that baseline even for an infinite
    # interval, where forming a zero-times-infinity offset would otherwise NaN.
    clock = torch.where(torch.any(noise > 0), randomized, baseline)
    return NetStimState(
        clock,
        torch.zeros_like(clock, dtype=torch.long),
        torch.full_like(clock, -float("inf")),
    )


def netstim_renewal(
    state,
    interval,
    noise,
    max_spikes,
    t,
    exp_rand,
    *,
    tau,
    bptt=True,
    next_sched_time=None,
    scheduled_gate=None,
    scheduled_hard=None,
):
    """Emit at most one hard spike and advance one renewal per call.

    ``exp_rand`` is an explicit unit-rate exponential draw with the full state
    shape, even for deterministic or exhausted generators. Native scheduling
    supplies the next heap head and tensor-schedule gate/mask. The legacy gate
    is computed from the minimum of stochastic and heap clocks *before* the
    renewal update, retaining their existing surrogate-gradient coupling.

    Snapshots let a native caller safely copy the returned state into its owned
    buffers without changing tensors retained by the autograd graph. ``bptt``
    controls whether the incoming stochastic clock retains its history; the
    newly computed clock still differentiates through this call's interval.
    """
    previous = state.next_stoch_time
    stoch = (previous if bptt else previous.detach()).clone()
    sched = (
        torch.full_like(stoch, float("inf"))
        if next_sched_time is None
        else next_sched_time.detach().clone()
    )
    combined = torch.minimum(stoch, sched)
    x = t - combined
    soft = torch.sigmoid(x / tau)
    hard = (x >= 0).to(soft.dtype)
    legacy_gate = hard + (soft - soft.detach())
    stoch_due = t >= stoch
    heap_due = torch.logical_and(t >= sched, torch.isfinite(sched))
    can_spike = state.spike_counts < max_spikes
    legacy_hard = torch.logical_and(torch.logical_or(stoch_due, heap_due), can_spike)
    can_gate = can_spike.to(stoch.dtype)
    legacy_gate = legacy_gate * can_gate
    if scheduled_gate is None:
        spike_gate = legacy_gate
    else:
        spike_gate = legacy_gate + scheduled_gate * can_gate
    if scheduled_hard is None:
        spikes = legacy_hard
    else:
        spikes = torch.logical_or(
            legacy_hard, torch.logical_and(scheduled_hard, can_spike)
        )

    next_interval = interval * (1 - noise) + interval * noise * exp_rand
    stoch_fires = torch.logical_and(stoch_due, can_spike)
    heap_fires = torch.logical_and(heap_due, can_spike)
    delta = next_interval * legacy_gate * stoch_fires.to(stoch.dtype)
    next_state = NetStimState(
        stoch + delta,
        state.spike_counts + spikes.to(torch.long),
        torch.broadcast_to(t.detach(), stoch.shape).clone(),
    )
    return NetStimTransition(next_state, spikes, spike_gate, heap_fires)
