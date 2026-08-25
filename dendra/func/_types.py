"""Public tensor containers for Dendra's functional execution API."""

from __future__ import annotations

from typing import NamedTuple

import torch

TensorTree = dict[str, object]


class PopulationTensors(NamedTuple):
    """Explicit tensor inputs extracted from one initialized Population."""

    parameters: dict[str, torch.Tensor]
    constants: dict[str, torch.Tensor]
    state: TensorTree


class StepInput(NamedTuple):
    """Already assembled inputs for one functional Population step."""

    ve: torch.Tensor | None = None
    intra: torch.Tensor | None = None


class RolloutInput(NamedTuple):
    """Time-first, already assembled inputs for a functional rollout."""

    ve: torch.Tensor | None = None
    intra: torch.Tensor | None = None


class FunctionalizationError(RuntimeError):
    """Raised when a Population cannot yet be lowered without semantic loss."""


__all__ = [
    "FunctionalizationError",
    "PopulationTensors",
    "RolloutInput",
    "StepInput",
]
