"""Strict deterministic input frames shared by native and functional execution."""

from __future__ import annotations

import torch


def native_packet(inputs):
    """Accept the existing functional packet without replacing its identity."""
    from dendra.func import Inputs

    if inputs is None:
        return Inputs()
    if not isinstance(inputs, Inputs):
        raise TypeError("inputs must be an Inputs packet or None")
    return inputs


def native_input_sources(inputs, *, ve=None, extra=None):
    """Resolve packet ownership without changing legacy coercion policies."""
    packet = native_packet(inputs)
    if packet.ve is not None:
        if ve is not None:
            raise ValueError("inputs.ve cannot be combined with ve")
        if extra is not None:
            raise ValueError("inputs.ve cannot be combined with extra")
    return packet


def normalize_input_tape(value, *, name, steps, shape, device, dtype):
    """Normalize a strict Tensor tape and retain its common leading time axis."""
    if value is None:
        return None, steps
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Tensor or None")
    if value.ndim == 0:
        raise ValueError(f"{name} needs a leading time axis; got a scalar tensor")
    if value.device != device or value.dtype != dtype:
        raise ValueError(f"{name} must use {device}/{dtype}")
    value_steps = int(value.shape[0])
    if steps is not None and value_steps != steps:
        raise ValueError(
            f"{name} has {value_steps} steps, but the rollout expects {steps}"
        )

    # Ordinary trailing broadcasting is authoritative, even when a spatial
    # suffix happens to match a batch axis. Only failed trailing broadcasting
    # permits the explicit batch-only interpretation.
    payload_shape = tuple(value.shape[1:])
    target_shape = (value_steps, *shape)
    trailing_error = None
    if len(payload_shape) <= len(shape):
        aligned_shape = (
            value_steps,
            *((1,) * (len(shape) - len(payload_shape))),
            *payload_shape,
        )
        try:
            normalized = torch.broadcast_to(
                value.reshape(aligned_shape),
                target_shape,
            )
            return normalized.contiguous(), value_steps
        except RuntimeError as exc:
            trailing_error = exc

    batch_shape = shape[:-2]
    if batch_shape and 0 < len(payload_shape) <= len(batch_shape):
        batch_aligned_shape = (
            value_steps,
            *((1,) * (len(batch_shape) - len(payload_shape))),
            *payload_shape,
        )
        try:
            batch_value = torch.broadcast_to(
                value.reshape(batch_aligned_shape),
                (value_steps, *batch_shape),
            )
            normalized = torch.broadcast_to(
                batch_value.reshape(value_steps, *batch_shape, 1, 1),
                target_shape,
            )
            return normalized.contiguous(), value_steps
        except RuntimeError:
            pass

    raise ValueError(
        f"{name} per-step shape {payload_shape} is not broadcastable to "
        f"Population shape {shape}. Ordinary trailing PyTorch "
        "broadcasting is tried first; a batch-only fallback accepts "
        f"payloads broadcastable to explicit batch shape {batch_shape}. "
        "Use singleton axes to disambiguate batch and spatial intent."
    ) from trailing_error


def normalize_input_frame(value, *, name, shape, device, dtype):
    """Normalize one strict frame, including a spatially shared scalar."""
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Tensor or None")
    tape, _steps = normalize_input_tape(
        value.unsqueeze(0),
        name=name,
        steps=1,
        shape=shape,
        device=device,
        dtype=dtype,
    )
    return tape.squeeze(0)


__all__ = []
