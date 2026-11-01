"""Functional transforms for explicit-state Dendra execution."""

from ._binding import BoundPopulation
from ._callback_preparation import make_callbacks
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
from ._continuation_capture import capture_continuation
from ._continuation_checkpoints import export_continuation, restore_continuation
from ._continuation_detach import detach_continuation
from ._continuous import (
    ContinuousConTensors,
    FunctionalContinuousCon,
    make_functional_continuous_con,
)
from ._convenience import CompiledRunner, compile
from ._draws import RuntimeDraws
from ._execution import ExecutionCapabilities, ExecutionReport
from ._initialization import initialize, sample_initialization_draws
from ._initialization_draws import InitializationDraws
from ._inputs import Inputs
from ._lowering import make_functional
from ._native_handoff import commit_continuation_
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
from ._runtime_draw_sampling import sample_runtime_draws
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
    "InitializationDraws",
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
    "capture_continuation",
    "compile",
    "commit_continuation_",
    "detach_continuation",
    "export_continuation",
    "inspect_sources",
    "initialize",
    "longrun",
    "longrun_checkpointed",
    "make_callbacks",
    "make_functional",
    "make_functional_continuous_con",
    "make_functional_netcon",
    "make_functional_netstim",
    "make_functional_network",
    "restore_continuation",
    "run",
    "sample_initialization_draws",
    "sample_runtime_draws",
]
