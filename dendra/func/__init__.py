"""Functional transforms for explicit-state Dendra execution."""

from ._binding import BoundPopulation
from ._callbacks import (
    LFP,
    Active,
    ActiveAL,
    ActiveALCount,
    AnomalyDetector,
    APCount,
    FunctionalCallback,
    FunctionalCallbackResults,
    FunctionalCallbacks,
    FunctionalCallbackState,
    PopulationCallback,
    Raster,
    Recorder,
)
from ._compiled import CompiledPopulationChunk
from ._connection_sampling import ConnectionSamplingDraw
from ._continuation import Continuation
from ._continuous import (
    ContinuousConTensors,
    FunctionalContinuousCon,
    make_functional_continuous_con,
)
from ._convenience import CompiledRunner, compile
from ._draws import RuntimeDraws
from ._execution import ExecutionCapabilities, ExecutionReport
from ._inputs import Inputs
from ._lowering import make_functional
from ._netcon import FunctionalNetCon, NetConTensors, make_functional_netcon
from ._netstim import (
    FunctionalNetStim,
    NetStimDraw,
    NetStimOutputs,
    NetStimTensors,
    make_functional_netstim,
)
from ._network import FunctionalNetwork, NetworkTensors, make_functional_network
from ._network_binding import BoundNetwork
from ._network_compile import CompiledNetworkRunner
from ._network_initialization import (
    NetworkInitializationCache,
    NetworkInitializationInput,
)
from ._network_rollout import CompiledNetworkChunk, NetworkRolloutInput
from ._population import FunctionalPopulation
from ._runners import longrun, longrun_checkpointed, run
from ._runtime_extra import VarInput
from ._schedules import TensorScheduleTensors
from ._source_inspection import SourceEntry, SourceReport, inspect_sources
from ._stimuli import FunctionalExtra, FunctionalIntra, StimulusTensors
from ._types import (
    FunctionalizationError,
    InitializationInput,
    PopulationTensors,
    RandomParameterDraw,
    RolloutInput,
    RuntimeNoiseDraw,
    StepInput,
    StimulationDraw,
)

__all__ = [
    "Active",
    "ActiveAL",
    "ActiveALCount",
    "APCount",
    "AnomalyDetector",
    "BoundPopulation",
    "BoundNetwork",
    "CompiledNetworkChunk",
    "CompiledNetworkRunner",
    "CompiledPopulationChunk",
    "CompiledRunner",
    "ContinuousConTensors",
    "Continuation",
    "ConnectionSamplingDraw",
    "ExecutionCapabilities",
    "ExecutionReport",
    "FunctionalCallback",
    "FunctionalCallbackResults",
    "FunctionalCallbackState",
    "FunctionalCallbacks",
    "FunctionalContinuousCon",
    "FunctionalPopulation",
    "FunctionalExtra",
    "FunctionalIntra",
    "FunctionalNetCon",
    "FunctionalNetStim",
    "FunctionalNetwork",
    "FunctionalizationError",
    "InitializationInput",
    "Inputs",
    "LFP",
    "NetConTensors",
    "NetStimDraw",
    "NetStimOutputs",
    "NetStimTensors",
    "NetworkInitializationInput",
    "NetworkInitializationCache",
    "NetworkRolloutInput",
    "NetworkTensors",
    "PopulationTensors",
    "RandomParameterDraw",
    "PopulationCallback",
    "Raster",
    "Recorder",
    "RolloutInput",
    "RuntimeNoiseDraw",
    "RuntimeDraws",
    "SourceEntry",
    "SourceReport",
    "StimulationDraw",
    "StepInput",
    "StimulusTensors",
    "TensorScheduleTensors",
    "VarInput",
    "compile",
    "inspect_sources",
    "longrun",
    "longrun_checkpointed",
    "make_functional",
    "make_functional_continuous_con",
    "make_functional_netcon",
    "make_functional_netstim",
    "make_functional_network",
    "run",
]
