"""Functional transforms for explicit-state Dendra Population execution."""

from ._binding import BoundPopulation
from ._callbacks import (
    AnomalyDetector,
    APCount,
    FunctionalCallback,
    FunctionalCallbackResults,
    FunctionalCallbacks,
    FunctionalCallbackState,
    Raster,
    Recorder,
)
from ._compiled import CompiledPopulationChunk
from ._execution import ExecutionCapabilities, ExecutionReport
from ._population import FunctionalPopulation, make_functional
from ._runners import longrun, longrun_checkpointed, run
from ._stimuli import FunctionalExtra, FunctionalIntra, StimulusTensors
from ._types import (
    FunctionalizationError,
    InitializationInput,
    PopulationTensors,
    RolloutInput,
    StepInput,
)

__all__ = [
    "APCount",
    "AnomalyDetector",
    "BoundPopulation",
    "CompiledPopulationChunk",
    "ExecutionCapabilities",
    "ExecutionReport",
    "FunctionalCallback",
    "FunctionalCallbackResults",
    "FunctionalCallbackState",
    "FunctionalCallbacks",
    "FunctionalPopulation",
    "FunctionalExtra",
    "FunctionalIntra",
    "FunctionalizationError",
    "InitializationInput",
    "PopulationTensors",
    "Raster",
    "Recorder",
    "RolloutInput",
    "StepInput",
    "StimulusTensors",
    "longrun",
    "longrun_checkpointed",
    "make_functional",
    "run",
]
