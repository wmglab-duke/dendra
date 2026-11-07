"""Cold tensor-schedule extraction and frozen topology for functional adapters."""

from __future__ import annotations

from typing import NamedTuple

import torch

from ._types import FunctionalizationError


class TensorScheduleTensors(NamedTuple):
    """Explicit event values and optional bound source vectors.

    ``times`` and ``amplitudes`` are independent, per-event floating snapshots.
    Bound source snapshots retain their native dtype and shape. Fixed event
    routing and source indices belong to the lowered plan, rather than carry.
    """

    times: torch.Tensor
    amplitudes: torch.Tensor
    time_source: torch.Tensor | None = None
    amplitude_source: torch.Tensor | None = None


def _snapshot(value):
    with torch.inference_mode(False):
        result = value.detach().clone(memory_format=torch.contiguous_format)
        result.requires_grad_(value.requires_grad)
        return result


class _TensorSchedulePlan:
    """Frozen indices and schema checks shared by NetStim and NetCon lowering.

    This helper contains no numerical schedule evaluation. The network layer
    resolves explicit source values and applies each native schedule policy.
    """

    def __init__(self, source, *, index_name, device, dtype):
        self.index_name = index_name
        self.device = device
        self.dtype = dtype
        self.indices = _snapshot(getattr(source, index_name))
        self.time_idx = _snapshot(source.sched_time_idx)
        self.amplitude_idx = _snapshot(source.sched_weight_idx)
        self.events = int(self.indices.numel())
        self._schema = {
            name: (
                None
                if value is None
                else (tuple(value.shape), value.dtype, value.device)
            )
            for name, value in self._values(source).items()
        }
        self.active = self.events > 0 or any(
            self._schema[name] is not None
            for name in ("time_source", "amplitude_source")
        )
        self.has_time_refs = bool((self.time_idx >= 0).any())
        self.has_amplitude_refs = bool((self.amplitude_idx >= 0).any())
        self.validate_compatible(source)

    @staticmethod
    def _values(source):
        return {
            "times": source.sched_time_ms,
            "amplitudes": source.sched_weight,
            "time_source": source._sched_t_source,
            "amplitude_source": source._sched_w_source,
        }

    @staticmethod
    def _validate_tensor(value, schema, label):
        if schema is None:
            if value is not None:
                raise ValueError(f"{label} must be None for this schedule")
            return
        if not torch.is_tensor(value):
            raise TypeError(f"{label} must be a Tensor")
        shape, dtype, device = schema
        if (
            tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != device
        ):
            raise ValueError(f"{label} must match {device}/{dtype}/{shape}")

    def validate(self, packet):
        if not self.active:
            if packet is not None:
                raise ValueError("This plan has no tensor schedule; pass schedule=None")
            return
        if not isinstance(packet, TensorScheduleTensors):
            raise TypeError(
                "This plan requires an explicit TensorScheduleTensors packet"
            )
        for name, schema in self._schema.items():
            self._validate_tensor(getattr(packet, name), schema, f"schedule.{name}")

    def validate_compatible(self, source):
        try:
            for name, expected in (
                (self.index_name, self.indices),
                ("sched_time_idx", self.time_idx),
                ("sched_weight_idx", self.amplitude_idx),
            ):
                value = getattr(source, name)
                self._validate_tensor(
                    value,
                    ((self.events,), torch.long, self.device),
                    name,
                )
                if not torch.equal(value, expected):
                    raise ValueError(
                        "Tensor schedule routing or source indices changed"
                    )
            for name, value in self._values(source).items():
                self._validate_tensor(value, self._schema[name], name)
            for name in ("times", "amplitudes"):
                self._validate_tensor(
                    self._values(source)[name],
                    ((self.events,), self.dtype, self.device),
                    name,
                )
            if bool((self.indices < 0).any()):
                raise ValueError("Tensor schedule event indices must be nonnegative")
            for indices, field in (
                (self.time_idx, "time_source"),
                (self.amplitude_idx, "amplitude_source"),
            ):
                if bool((indices < -1).any()):
                    raise ValueError("Tensor schedule source indices must be >= -1")
                schema = self._schema[field]
                if schema is not None and len(schema[0]) != 1:
                    raise ValueError("Tensor schedule bound sources must be vectors")
                if bool((indices >= 0).any()):
                    if schema is None or bool((indices >= schema[0][0]).any()):
                        raise ValueError(
                            "Tensor schedule reference is outside its source"
                        )
        except (TypeError, ValueError) as error:
            raise FunctionalizationError(
                "Tensor schedule topology or value schema does not match this plan."
            ) from error

    def extract(self, source):
        self.validate_compatible(source)
        if not self.active:
            return None
        return TensorScheduleTensors(
            **{
                name: None if value is None else _snapshot(value)
                for name, value in self._values(source).items()
            }
        )


__all__ = ["TensorScheduleTensors"]
