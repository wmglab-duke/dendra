"""Pure fixed-topology tensor schedules shared by network execution adapters.

Scheduling, index expansion, past-event filtering, and reference binding happen
outside these functions. Evaluation is stateless: repeated times reevaluate an
event and skipped hard grid steps do not consume or catch up events. NetStim's
legacy heap calendar has separate consumption semantics.
"""

import math

import torch


def causal_step_index(times: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Map times to the first reconstructed fixed-step grid point at or after them.

    Reconstructing the lower candidate preserves exact decimal grid boundaries
    and adjacent representable times, unlike applying ceil to a rounded ratio.
    """
    dt = torch.as_tensor(dt, device=times.device, dtype=times.dtype)
    lower = torch.floor(times / dt)
    needs_next = lower * dt < times
    return (lower + needs_next.to(lower.dtype)).to(torch.long)


def resolve_schedule_values(
    times,
    weights,
    time_idx,
    weight_idx,
    *,
    time_source=None,
    weight_source=None,
    time_dtype=None,
    weight_dtype=None,
    device=None,
):
    """Add explicitly supplied reference values to event value-mode slots.

    A negative reference index masks its gathered value with zero. Adapters
    decide whether a bound source is active and specify their established dtype
    policy. Cast selected event values after gathering: casting the full source
    first changes the precision of repeated-index gradient accumulation.
    Multiplying a gathered nonfinite value by zero intentionally retains
    ordinary tensor arithmetic semantics.
    """
    if time_source is not None:
        selected = time_source.index_select(0, time_idx.clamp_min(0))
        if time_dtype is not None or device is not None:
            selected = selected.to(device=device, dtype=time_dtype)
        times = times + selected * (time_idx >= 0).to(selected.dtype)
    if weight_source is not None:
        selected = weight_source.index_select(0, weight_idx.clamp_min(0))
        if weight_dtype is not None or device is not None:
            selected = selected.to(device=device, dtype=weight_dtype)
        weights = weights + selected * (weight_idx >= 0).to(weights.dtype)
    return times, weights


def _triangular_kernel(distance, width):
    return (1.0 - distance.abs() / width).clamp(min=0.0, max=1.0)


def _aggregate(indices, values, size, *, dtype=None):
    # Functional scatter remains valid for mapped values and empty vmap lanes.
    return values.new_zeros(size, dtype=dtype).scatter_add(0, indices, values)


def netstim_tensor_schedule(
    flat_idx,
    times,
    weights,
    t,
    dt,
    *,
    shape,
    differentiable_times,
    width=1.0,
):
    """Return state-shaped amplitudes and causal hard spikes for NetStim.

    Triangular amplitudes use continuous per-state absolute time. Hard spikes
    always use causal grid equality, independently of amplitude and support.
    """
    if flat_idx.numel() == 0:
        return times.new_zeros(shape), torch.zeros(
            shape, device=times.device, dtype=torch.bool
        )
    t_evt = torch.broadcast_to(t, shape).reshape(-1).index_select(0, flat_idx)
    event_step = causal_step_index(times, dt)
    current_step = causal_step_index(t_evt, dt)
    hard_evt = event_step == current_step
    if differentiable_times:
        distance = (times - t_evt) / dt
        kernel = _triangular_kernel(distance, max(float(width), 1e-6))
    else:
        kernel = hard_evt.to(times.dtype)
    size = math.prod(shape)
    gate = _aggregate(flat_idx, weights * kernel, size, dtype=times.dtype).reshape(
        shape
    )
    counts = _aggregate(flat_idx, hard_evt.to(torch.int32), size).reshape(shape)
    return gate, counts > 0


def netcon_tensor_schedule(
    con_idx,
    times,
    weights,
    global_step,
    dt,
    *,
    connections,
    differentiable_times,
    width=1.0,
):
    """Return per-connection scheduled amplitudes and event counts for NetCon.

    Training counts triangular support; exact execution counts causal grid
    equality. The native width epsilon and integer global-step clock are kept
    distinct from NetStim's policy.
    """
    if con_idx.numel() == 0:
        return times.new_zeros(connections), torch.zeros(
            connections, device=times.device, dtype=torch.int32
        )
    if differentiable_times:
        distance = times / dt - global_step.to(times.dtype).reshape(1)
        kernel = _triangular_kernel(distance, width + 1e-6)
        active = kernel > 0
    else:
        active = causal_step_index(times, dt) == global_step.reshape(1)
        kernel = active.to(times.dtype)
    gate = _aggregate(con_idx, weights * kernel, connections, dtype=times.dtype)
    counts = _aggregate(con_idx, active.to(torch.int32), connections)
    return gate, counts
