"""Bound extracellular fields through dn.compile on a real CUDA device.

Run with ``pytest tests/func/test_cuda_convenience_extra.py -W error -m cuda``.
The shared CPU helpers retain full v/vc/current/HH traces and compare bound
stimulation with independently assembled recurrent-time voltage inputs. Only
two tiny cases compile real GPU Inductor kernels; no unrolled matrix is needed.
"""

from __future__ import annotations

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_convenience_extra import (
    _close,
    _exercise,
    _extra,
    _imperative_reference,
    _model,
    _runner,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _assert_coverage(recorded, gradients, *, block, dtype):
    names = {"v", "i_membrane", "hh.m", "hh.h", "hh.n", "t"}
    if block:
        names.add("vc")
    assert set(recorded) == names
    for name in names - {"t"}:
        assert recorded[name].dtype == dtype
        assert recorded[name].device.type == "cuda"
        assert recorded[name].shape[0] == 5
    assert len(gradients) >= 8
    for name in (
        "stimulus",
        "initial_voltage",
        "gnabar",
        "gkbar",
        "gl",
        "ena",
        "ek",
        "el",
    ):
        assert name in gradients
    extra = {
        name: value
        for name, value in gradients.items()
        if name.startswith("stimulation.extra.")
    }
    for suffix in (".field", ".amp", ".freq", ".phase"):
        assert sum(name.endswith(suffix) for name in extra) == 2
    for name, value in gradients.items():
        assert value.device.type == "cuda", name
        assert torch.isfinite(value).all(), name
        assert torch.count_nonzero(value), name


def _cross_precision_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        value = value.detach().to(device="cpu", dtype=torch.float64)
        reference = expected[name].detach().to(device="cpu", dtype=torch.float64)
        if name == "i_membrane":
            # A small accepted current is a residual of larger ionic and
            # capacitive terms. Binary32 cancellation needs its own mixed-
            # precision gate, distinct from the tighter same-precision oracle
            # inside _exercise. Compare in nA, not public mA.
            torch.testing.assert_close(
                value * 1.0e6, reference * 1.0e6, rtol=2e-3, atol=2e-6
            )
        elif name == "vc":
            # Small shell potentials share an elimination with roughly 60 mV
            # intracellular values. A 1e-5 mV (10 nV) absolute gate is a few
            # binary32 ULPs at that larger circuit scale, rather than an
            # ill-conditioned relative test against a near-zero shell value.
            torch.testing.assert_close(value, reference, rtol=2e-4, atol=1e-5)
        else:
            _close({name: value}, {name: reference}, dtype=torch.float32)


@pytest.mark.parametrize(
    ("kind", "checkpoint_every"), [("path", None), ("multi_block", 2)]
)
def test_cuda_compile_bound_extra_fields_waveforms_and_all_eight_gradients(
    kind, checkpoint_every
):
    recorded, gradients = _exercise(
        kind,
        device="cuda",
        dtype=torch.float64,
        shell_resistance_scale=100.0,
        backend="eager",
        execution="scan",
        checkpoint_every=checkpoint_every,
    )
    _assert_coverage(
        recorded, gradients, block=kind == "multi_block", dtype=torch.float64
    )


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_cuda_compile_extra_float32_matches_float64_cpu_traces_and_gradients(kind):
    pytest.importorskip("dendra_solvers")
    results = []
    for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32)):
        results.append(
            _exercise(
                kind,
                device=device,
                dtype=dtype,
                shell_resistance_scale=100.0,
                backend="eager",
                execution="scan",
                checkpoint_every=2,
            )
        )
    expected, actual = results
    _assert_coverage(
        actual[0], actual[1], block=kind == "multi_block", dtype=torch.float32
    )
    _cross_precision_close(actual[0], expected[0])
    assert actual[1].keys() == expected[1].keys()
    for name, value in actual[1].items():
        reference = expected[1][name].detach().to(device="cpu", dtype=torch.float64)
        value = value.detach().to(device="cpu", dtype=torch.float64)
        # Normalize each target independently so a small field/frequency
        # derivative is not hidden behind a loose absolute tolerance.
        scale = reference.abs().amax()
        assert torch.isfinite(scale) and scale > 0, name
        torch.testing.assert_close(
            value / scale,
            reference / scale,
            rtol=5e-3,
            atol=5e-4,
            msg=lambda message: f"{name}: {message}",
        )


def test_cuda_compile_extra_while_matches_imperative_full_circuit_recording():
    model = _model(
        "multi_block", device="cuda", dtype=torch.float64, shell_resistance_scale=100.0
    )
    extra = _extra(model)
    runner = _runner(model, extra, backend="eager", execution="while")
    with torch_compiler_warning_context():
        final, auxiliary = runner()
    actual = dict(auxiliary["callbacks"]["trace"])
    reference = _model(
        "multi_block", device="cuda", dtype=torch.float64, shell_resistance_scale=100.0
    )
    with torch.no_grad():
        expected = _imperative_reference(reference, _extra(reference))
    _close(actual, expected, dtype=torch.float64)
    assert actual["vc"].shape[-1] == 3
    for name in ("v", "vc", "i_membrane"):
        torch.testing.assert_close(final["integrator"][name], actual[name][-1])
    # Host control metadata is intentionally not coerced onto CUDA/model dtype.
    for leaf in runner.functional._state_layout:
        value = final
        for key in leaf.public_path:
            value = value[key]
        assert value.device == leaf.device
        assert value.dtype == leaf.dtype
    assert runner.execution_report().strategy == "compiled_while_loop"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    [("path", "scan", 2), ("multi_block", "while", None)],
)
def test_cuda_inductor_compile_extra_scan_gradients_and_while_full_traces(
    kind, execution, checkpoint_every
):
    with torch_compiler_warning_context():
        recorded, gradients = _exercise(
            kind,
            device="cuda",
            dtype=torch.float32,
            shell_resistance_scale=100.0,
            backend="inductor",
            execution=execution,
            checkpoint_every=checkpoint_every,
        )
    if execution == "scan":
        _assert_coverage(recorded, gradients, block=False, dtype=torch.float32)
    else:
        assert not gradients
        assert recorded["vc"].shape[-1] == 3
        assert recorded["v"].shape[0] == 5
