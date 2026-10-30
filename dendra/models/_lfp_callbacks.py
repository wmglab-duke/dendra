"""Pure lead-field materialization and projection shared by LFP callbacks."""

from __future__ import annotations

import torch


def _validate_lfp_fields(v_unit):
    if not isinstance(v_unit, list):
        raise TypeError(
            "v_unit must be a non-empty list of lead-field tensors; "
            "pass [lead_field] for one recording contact."
        )
    if not v_unit:
        raise ValueError("v_unit must contain at least one lead-field tensor.")
    if not all(isinstance(field, torch.Tensor) for field in v_unit):
        raise TypeError("Every v_unit entry must be a torch.Tensor.")
    shape = v_unit[0].shape
    if any(field.shape != shape for field in v_unit[1:]):
        raise ValueError("All v_unit lead-field tensors must have the same shape.")
    return tuple(v_unit)


def _materialize_lfp_fields(fields, *, device, dtype, padding=None):
    stacked = torch.stack([field.to(device=device, dtype=dtype) for field in fields])
    if padding is not None:
        stacked = stacked.reshape(
            stacked.shape[0], *((1,) * padding), *stacked.shape[1:]
        )
    return stacked


def _project_lfp(current, fields, *, rule=None, padding=0):
    if rule is not None:
        return torch.einsum(rule, current, fields)
    if padding:
        fields = fields.reshape(fields.shape[0], *((1,) * padding), *fields.shape[1:])
    return (current.unsqueeze(0) * fields).sum(dim=(-1, -2)).movedim(0, -1)


__all__ = []
