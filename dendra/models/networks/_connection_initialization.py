"""Shared tensor resets for event and continuous connection adapters.

Initialization clears delivery traffic and detector carry without changing
parameters, routing, schedules, or calendar topology. Public tensor operations
are pure; native adapters may donate owned buffers through ``_inplace``.
"""

import torch


def _zero(value, *, inplace):
    return value.zero_() if inplace else torch.zeros_like(value)


def event_initial_step(t, dt, reference):
    """Match NetCon's Python ``round(float(t) / float(dt))`` clock policy."""
    # Native event initialization evaluates the ratio in Python Float64 even
    # when the connection uses Float32 or resides on an accelerator. Keep the
    # calculation on CPU, which also supports MPS simulation devices.
    time = torch.as_tensor(t, device="cpu", dtype=torch.float64).reshape(1)
    step = torch.round(time / dt.to(device="cpu", dtype=torch.float64).reshape(1))
    if not torch._C._are_functorch_transforms_active():
        outside_range = ((step < -(2**63)) | (step >= 2**63)).any()
        if torch.compiler.is_compiling():
            torch._assert_async(
                torch.isfinite(step).all() & ~outside_range,
                "Event initialization step is outside the Int64 range",
            )
        else:
            if bool(torch.isnan(step).any()):
                raise ValueError("cannot convert float NaN to integer")
            if not bool(torch.isfinite(step).all()) or bool(outside_range):
                # Python round returns an unbounded integer. Native fill_
                # rejects values outside Int64 rather than saturating them.
                raise OverflowError(
                    "Event initialization step is outside the Int64 range"
                )
    return step.to(device=reference.device, dtype=reference.dtype).reshape(
        reference.shape
    )


def continuous_initial_step(t, dt, reference, *, dtype):
    """Match ContinuousCon's target-dtype nearest-step clock policy."""
    time = torch.as_tensor(t, device=reference.device, dtype=dtype).reshape(1)
    step = torch.round(time / dt.to(device=reference.device, dtype=dtype).reshape(1))
    return step.to(dtype=reference.dtype).reshape(reference.shape)


def event_runtime_reset(state, *, clear_deliveries=True, step=None, _inplace=False):
    """Reset the ring pointer and optionally traffic/detector carry.

    Omitting ``step`` preserves the absolute counter, as ``NetCon.zero`` does.
    Clearing traffic also resets tracked counts and rising-edge latches. A
    reset never consumes or removes scheduled events.
    """
    result = dict(state)
    result["current_time_step"] = _zero(state["current_time_step"], inplace=_inplace)
    if step is not None:
        result["global_step"] = state["global_step"].copy_(step) if _inplace else step
    if clear_deliveries:
        for name in (
            "delivery_buffer",
            "has_spiked",
            "is_spiking",
            "events",
            "event_queue",
            "spike_history_packed",
            "source_gate_history",
            "bitpack_source_has_spiked",
        ):
            if name in state:
                result[name] = _zero(state[name], inplace=_inplace)
    return result


def continuous_runtime_reset(
    state, *, clear_deliveries=True, step=None, _inplace=False
):
    """Reset continuous counters and optionally payload/presence rings."""
    result = dict(state)
    result["current_time_step"] = _zero(state["current_time_step"], inplace=_inplace)
    result["global_step"] = (
        _zero(state["global_step"], inplace=_inplace)
        if step is None
        else state["global_step"].copy_(step)
        if _inplace
        else step
    )
    if clear_deliveries:
        for name in ("delivery_buffer", "delivery_mask"):
            result[name] = _zero(state[name], inplace=_inplace)
    return result


def event_initial_state(state, *, t, dt):
    """Construct detached event initialization carry from a schema template."""
    step = event_initial_step(t, dt, state["global_step"])
    values = event_runtime_reset(state, step=step)
    return {name: value.detach() for name, value in values.items()}


def continuous_initial_state(state, *, t, dt, dtype):
    """Construct detached continuous initialization carry from a template."""
    step = continuous_initial_step(t, dt, state["global_step"], dtype=dtype)
    values = continuous_runtime_reset(state, step=step)
    return {name: value.detach() for name, value in values.items()}
