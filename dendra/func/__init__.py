"""Functional transforms for explicit-state Dendra Population execution."""

from ._population import FunctionalPopulation, make_functional
from ._types import (
    FunctionalizationError,
    PopulationTensors,
    RolloutInput,
    StepInput,
)

__all__ = [
    "FunctionalPopulation",
    "FunctionalizationError",
    "PopulationTensors",
    "RolloutInput",
    "StepInput",
    "make_functional",
]
