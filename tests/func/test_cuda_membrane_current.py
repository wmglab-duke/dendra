"""CUDA acceptance for opt-in functional membrane-current observations.

Run on a CUDA machine with::

    python -m pytest tests/func/test_cuda_membrane_current.py -W error -m cuda

All admitted model families compare full current/voltage/gate histories and
nine gradients against the existing imperative integrators. A small orthogonal
compiler slice covers scalar and packed extracellular scan/while execution.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_membrane_current import (
    DT,
    KINDS,
    STEPS,
    _close,
    _functional_result,
    _imperative_result,
    _inputs,
    _model,
    _observable,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_membrane_current_traces_and_gradients(kind, dtype):
    source = _model(kind, device="cuda", dtype=dtype)
    snapshot = _module_snapshot(source)
    actual, gradients = _functional_result(source)
    expected, reference_gradients = _imperative_result(
        _model(kind, device="cuda", dtype=dtype)
    )
    assert len(gradients) == 9
    _close(actual, expected, dtype=dtype)
    _close(gradients, reference_gradients, dtype=dtype)
    assert torch.count_nonzero(actual["i_membrane"][0]) == 0
    _assert_module_snapshot(source, snapshot)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ("point", "multi_block"))
def test_cuda_membrane_current_compiled_scan(kind):
    source = _model(kind, device="cuda", dtype=torch.float32)
    expected, reference_gradients = _functional_result(source)
    with torch_compiler_warning_context():
        actual, gradients = _functional_result(source, compiled="scan")
    _close(actual, expected, dtype=torch.float32)
    _close(gradients, reference_gradients, dtype=torch.float32)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ("point", "multi_block"))
def test_cuda_membrane_current_compiled_while(kind):
    source = _model(kind, device="cuda", dtype=torch.float32)
    expected, _ = _imperative_result(_model(kind, device="cuda", dtype=torch.float32))
    ve, intra = _inputs(source)
    runner = dn.func.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution="while",
        inference=True,
        options={"cpp_wrapper": False, "compile_threads": 1},
        callbacks={"trace": dn.func.Recorder(tuple(_observable(source)))},
    )
    with torch_compiler_warning_context():
        final, auxiliary = runner(inputs=dn.func.RolloutInput(ve=ve, intra=intra))
    assert runner.execution_report().strategy == "compiled_while_loop"
    _close(dict(auxiliary["callbacks"]["trace"]), expected, dtype=torch.float32)
    _close(
        {"i_membrane": final["integrator"]["i_membrane"]},
        {"i_membrane": expected["i_membrane"][-1]},
        dtype=torch.float32,
    )
