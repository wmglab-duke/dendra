"""Shared host-runner validation and compiler policy."""

from collections.abc import Mapping

import torch

from ._callbacks import _is_callback_configuration
from ._types import FunctionalizationError

_DEFAULT_RUNTIME_EXTRA_CHUNK_STEPS = 64


def _runner_default_horizon(steps, duration, dt):
    """Validate a default horizon and size its reusable numerical kernel."""
    from dendra.models.core import _duration_step_budget, _validate_time_scalar

    if steps is not None and duration is not None:
        raise ValueError("steps and duration are mutually exclusive")
    if duration is None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        return steps, None, steps
    duration = _validate_time_scalar(duration, name="duration", positive=False)
    timestep = _validate_time_scalar(dt, name="dt", positive=True)
    nominal_steps, _ = _duration_step_budget(duration, timestep)
    return None, duration, max(nominal_steps, 1)


def _runner_call_horizon(default_steps, default_duration, steps, duration):
    """Select one call's horizon without changing its runner's defaults."""
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


def _runner_chunk_steps(chunk_steps, *, steps, runtime_extra):
    """Resolve compiled width without changing the runner's complete horizon."""
    if chunk_steps is not None:
        if (
            isinstance(chunk_steps, bool)
            or not isinstance(chunk_steps, int)
            or chunk_steps < 1
        ):
            raise ValueError("chunk_steps must be a positive integer")
        return min(steps, chunk_steps)
    return min(steps, _DEFAULT_RUNTIME_EXTRA_CHUNK_STEPS) if runtime_extra else steps


def _validate_host_construction(compile_options):
    if torch.compiler.is_compiling():
        raise FunctionalizationError("construct compiled runners outside torch.compile")
    removed = [name for name in ("record", "node_indices") if name in compile_options]
    if removed:
        raise TypeError(
            f"Removed compile argument(s): {', '.join(removed)}; configure "
            "recording through callbacks={'trace': "
            "dn.func.Recorder(..., node_indices=...)}"
        )


def _configured_callbacks(callbacks):
    if callbacks is not None and not isinstance(callbacks, Mapping):
        raise TypeError(
            "callbacks must be a named mapping of FunctionalCallback configurations"
        )
    configured = {} if callbacks is None else dict(callbacks)
    for name, callback in configured.items():
        if not isinstance(name, str) or not name:
            raise ValueError("functional callback names must be non-empty strings")
        if not _is_callback_configuration(callback):
            raise TypeError(
                "callback values must be FunctionalCallback configurations or supported built-in Dendra callback configurations"
            )
    return configured


def _validate_host_call():
    if torch.compiler.is_compiling():
        raise FunctionalizationError("call compiled runners outside torch.compile")
    if (
        torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
    ):
        raise FunctionalizationError(
            "compiled runners are host-side callables; use the lower-level "
            "functional API for torch.func or forward-mode AD transforms"
        )


def _inference_compile_options(functional, inference, backend, options, keywords):
    """Prefer a native CPU loop without overriding explicit compiler choices."""
    result = dict(keywords)
    result["backend"] = backend
    selected = None if options is None else dict(options)
    if (
        inference
        and functional.device.type == "cpu"
        and backend == "inductor"
        and not result.get("disable", False)
        and result.get("mode") is None
    ):
        from torch._inductor import config

        candidate = selected or {}
        normalized = {
            name.replace("-", "_"): value for name, value in candidate.items()
        }
        if not (
            normalized.get("use_aoti", False)
            or normalized.get("disable_cpp_codegen", config.disable_cpp_codegen)
            or normalized.get("fx_wrapper", config.fx_wrapper)
        ):
            selected = dict(candidate)
            if "cpp_wrapper" not in normalized:
                selected["cpp_wrapper"] = True
    if selected is not None:
        result["options"] = selected
    return result
