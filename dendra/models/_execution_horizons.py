"""Shared host execution requests, separate from physical duration budgeting."""

from __future__ import annotations


def execution_call_horizon(default_steps, default_duration, steps, duration):
    """Select one call's horizon without changing its configured defaults."""
    from dendra.models.core import _validate_time_scalar

    if steps is not None and duration is not None:
        raise ValueError("steps and duration are mutually exclusive")
    if steps is None and duration is None:
        return default_steps, default_duration
    if duration is not None:
        return None, _validate_time_scalar(duration, name="duration", positive=False)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    return steps, None


def native_horizon_request(tstop, *, steps=None, duration=None):
    """Adapt native ``tstop`` without mixing legacy and canonical requests."""
    if steps is not None or duration is not None:
        if tstop is not None:
            raise ValueError("tstop cannot be combined with steps or duration")
        return execution_call_horizon(None, None, steps, duration)
    from dendra.models.core import _validate_time_scalar

    return None, _validate_time_scalar(tstop, name="tstop", positive=False)


def execution_step_budget(request_steps, request_duration, dt, duration_budget):
    """Resolve duration carry only when the request uses physical duration."""
    if request_steps is not None:
        return request_steps, None
    return duration_budget(request_duration, dt)


__all__ = []
