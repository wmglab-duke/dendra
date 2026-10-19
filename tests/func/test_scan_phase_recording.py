"""Per-phase histories preserve accepted-frame order and output ownership."""

import pytest
import torch
from test_scan_packed import (
    _assert_owned,
    _case,
    _clone,
    _close,
    _eager,
    _gradients,
    _leaves,
)

from dendra.func import _scan

pytestmark = pytest.mark.cpu


def _backend(graph, _examples):
    return graph.forward


@pytest.mark.parametrize("mode", ["no_grad", "inference"])
@pytest.mark.parametrize("steps", [4, 5, 8, 9])
def test_phase_histories_interleave_full_steps_and_tail_without_aliases(mode, steps):
    case = _case(callbacks=True, steps=steps)
    inputs = (
        case.parameters,
        case.prepared,
        case.state,
        case.ve,
        case.intra,
        case.carries,
    )
    originals = _clone(inputs)
    versions = tuple(value._version for value in _leaves(inputs))
    chunk = _scan._CompiledScanPopulationChunk(
        case.functional,
        steps=steps,
        compile_options={"backend": _backend, "fullgraph": True},
        stepwise_stimulation=True,
        plans=case.plans,
    )
    context = torch.inference_mode() if mode == "inference" else torch.no_grad()
    with context:
        actual = chunk(
            case.parameters,
            case.prepared,
            case.state,
            case.ve,
            case.intra,
            case.carries,
        )
        expected = _eager(case)
        _close(actual, expected)
        _assert_owned(actual, inputs)
        assert actual[3]["mixed"]["v"].shape == (steps, 2, 6)
        assert actual[3]["mixed"]["t"].shape == (steps,)
        saved = _clone(actual)
        repeated = chunk(
            {"gain": case.parameters["gain"] + 0.2},
            case.prepared,
            case.state,
            -case.ve,
            case.intra,
            case.carries,
        )
        _assert_owned(repeated, (inputs, actual))
        _close(actual, saved, exact=True)
    _close(inputs, originals, exact=True)
    assert versions == tuple(value._version for value in _leaves(inputs))


@pytest.mark.parametrize("mode", ["no_grad", "inference"])
def test_phase_recording_and_packed_training_reuse_independent_specializations(mode):
    case = _case(callbacks=True, steps=5)
    chunk = _scan._CompiledScanPopulationChunk(
        case.functional,
        steps=case.steps,
        compile_options={"backend": _backend, "fullgraph": True},
        stepwise_stimulation=True,
        plans=case.plans,
    )
    context = torch.inference_mode() if mode == "inference" else torch.no_grad()
    with context:
        forward = chunk(
            case.parameters,
            case.prepared,
            case.state,
            case.ve,
            case.intra,
            case.carries,
        )
        _close(forward, _eager(case))
    with torch.enable_grad():
        actual = chunk(
            case.parameters,
            case.prepared,
            case.state,
            case.ve,
            case.intra,
            case.carries,
        )
        expected = _eager(case)
        _close(actual, expected)
        _close(_gradients(actual, case), _gradients(expected, case))
    context = torch.inference_mode() if mode == "inference" else torch.no_grad()
    with context:
        repeated = chunk(
            case.parameters,
            case.prepared,
            case.state,
            case.ve,
            case.intra,
            case.carries,
        )
    assert len(chunk._kernels) == 2
    _close(repeated, forward, exact=True)
