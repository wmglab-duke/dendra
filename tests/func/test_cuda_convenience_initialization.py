"""Fresh dn.compile initialization on a real CUDA device.

Run with ``pytest tests/func/test_cuda_convenience_initialization.py -W error -m cuda``.
The shared CPU helpers compare complete recorded state and gradients with an
explicit initialize-then-run reference. Only two tiny cases compile actual GPU
Inductor kernels; the other cases isolate convenience wiring and precision.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_convenience_initialization import (
    _arguments,
    _close,
    _exercise,
    _explicit_reference,
    _gradients,
    _model,
    _recorded,
    _runner,
)
from tests.func.test_cuda_convenience_extra import (
    _cross_precision_close as _extra_cross_precision_close,
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


def _assert_coverage(recorded, gradients, *, block, dtype, extra=False):
    assert {"v", "i_membrane", "t"} <= recorded.keys()
    assert all(any(name.endswith(f"hh.{gate}") for name in recorded) for gate in "mhn")
    # The generic regional initializer and its carry must not disappear when
    # only the HH/voltage fields would be sufficient for a superficial match.
    assert any("_InitializationProbe" in name for name in recorded)
    if block:
        assert "vc" in recorded
        assert recorded["vc"].shape == (*recorded["v"].shape, 3)
        torch.testing.assert_close(
            recorded["vc"][0, ..., 0], recorded["v"][0], rtol=0, atol=0
        )
        assert torch.count_nonzero(recorded["vc"][0, ..., 1:]) == 0
    for name, value in recorded.items():
        assert value.shape[0] == 5, name
        assert torch.isfinite(value).all(), name
        if name != "t":
            assert value.dtype == dtype, name
            assert value.device.type == "cuda", name
    assert recorded["t"][0].item() == 0
    assert torch.count_nonzero(recorded["i_membrane"][0]) == 0
    if not gradients:
        return
    assert {
        "stimulus",
        "initial_voltage",
        "gnabar",
        "gkbar",
        "gl",
        "ena",
        "ek",
        "el",
        "rate",
        "celsius",
        "geometry",
        "ic",
    } <= gradients.keys()
    if extra:
        for suffix in (".field", ".amp", ".freq", ".phase"):
            assert (
                sum(
                    name.startswith("stimulation.extra.") and name.endswith(suffix)
                    for name in gradients
                )
                == 2
            )
    for name, value in gradients.items():
        assert value.device.type == "cuda", name
        assert torch.isfinite(value).all(), name
        assert torch.count_nonzero(value), name


def _cross_precision_close(actual, expected):
    assert actual.keys() == expected.keys()
    _extra_cross_precision_close(
        {name: value for name, value in actual.items() if name != "vc"},
        {name: value for name, value in expected.items() if name != "vc"},
    )
    if "vc" not in actual:
        return
    value = actual["vc"].detach().to(device="cpu", dtype=torch.float64)
    reference = expected["vc"].detach().to(device="cpu", dtype=torch.float64)
    assert value.shape == reference.shape and value.shape[-1] == 3
    assert torch.isfinite(value).all() and torch.isfinite(reference).all()
    torch.testing.assert_close(value[..., 0], reference[..., 0], rtol=2e-4, atol=1e-5)
    # This fresh fixture spans roughly 72 mV intracellular values. Solving
    # small shell potentials with the same circuit needs a 50 nV shell-only
    # budget: about 6.6 binary32 ULPs at 72 mV, not at the tiny shell scale.
    # Neither the intracellular gate, same-precision oracle, nor AD gates change.
    shell_atol = 5e-5  # mV
    shells = reference[..., 1:]
    assert torch.all(shells.abs().reshape(-1, 2).amax(0) > 10 * shell_atol)
    torch.testing.assert_close(value[..., 1:], shells, rtol=2e-4, atol=shell_atol)


@pytest.mark.parametrize(
    ("kind", "checkpoint_every"), (("myelinated", None), ("multi_block", 2))
)
def test_cuda_compile_fresh_scan_full_state_and_initialization_gradients(
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


@pytest.mark.parametrize("kind", ("myelinated", "multi_block"))
def test_cuda_compile_fresh_float32_matches_float64_cpu_traces_and_gradients(kind):
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
        scale = reference.abs().amax()
        assert torch.isfinite(scale) and scale > 0, name
        torch.testing.assert_close(
            value / scale,
            reference / scale,
            rtol=5e-3,
            atol=5e-4,
            msg=lambda message: f"{name}: {message}",
        )


def test_cuda_compile_fresh_while_restarts_full_circuit_and_control_state():
    model = _model(
        "multi_block",
        device="cuda",
        dtype=torch.float64,
        shell_resistance_scale=100.0,
    )
    runner = _runner(model, backend="eager", execution="while")
    parameters, initialization, inputs, _ = _arguments(runner, model)
    result = runner(parameters=parameters, initialization=initialization, inputs=inputs)
    with torch.no_grad():
        expected = _explicit_reference(runner, parameters, initialization, inputs)
    recorded = _recorded(result)
    _close(recorded, _recorded(expected), dtype=torch.float64)
    _close(result[0], expected[0], dtype=torch.float64)
    _close(result[1]["parameters"], expected[1]["parameters"], dtype=torch.float64)
    _assert_coverage(recorded, {}, block=True, dtype=torch.float64)
    assert not recorded["v"].requires_grad
    for leaf in runner.functional._state_layout:
        value = result[0]
        for key in leaf.public_path:
            value = value[key]
        # In particular, duration_remainder intentionally remains host float64.
        assert value.device == leaf.device
        assert value.dtype == leaf.dtype
    assert runner.execution_report().strategy == "compiled_while_loop"


def _nonleaf_arguments(runner, model):
    parameters, initialization, inputs, targets = _arguments(runner, model)
    voltage = initialization.v_init.detach().clone().requires_grad_()
    initialization = initialization._replace(v_init=0.98 * voltage - 0.5)
    targets["initial_voltage"] = voltage
    key = "stimulation.extra.contacts.0.field"
    field = parameters[key].detach().clone().requires_grad_()
    parameters[key] = 1.03 * field
    targets[key] = field
    assert not initialization.v_init.is_leaf
    assert not parameters[key].is_leaf
    return parameters, initialization, inputs, targets


def test_cuda_compile_fresh_nonleaf_initialization_and_extra_rebuild_each_call():
    model = _model("myelinated", device="cuda", dtype=torch.float64)
    runner = _runner(model, checkpoint_every=2, bound_extra=True)
    for _ in range(2):
        parameters, initialization, inputs, targets = _nonleaf_arguments(runner, model)
        actual = _recorded(
            runner(parameters=parameters, initialization=initialization, inputs=inputs)
        )
        gradients = _gradients(actual, targets)
        _assert_coverage(
            actual, gradients, block=False, dtype=torch.float64, extra=True
        )
        torch.testing.assert_close(
            actual["v"][0], initialization.v_init, rtol=0, atol=0
        )

        # Build independent non-leaf graphs for the reference after consuming
        # the first graph; neither the runner nor its stored defaults may retain
        # the previous initialization or electrode-field graph.
        parameters, initialization, inputs, targets = _nonleaf_arguments(runner, model)
        initialized = runner.functional.initialize(
            parameters, runner.constants, initialization
        )
        prepared = runner.functional.prepare(
            initialized.parameters, initialized.constants
        )
        expected = _recorded(
            dn.func.run(
                runner.functional.bind(initialized.parameters, prepared),
                initialized.state,
                steps=4,
                inputs=inputs,
                callbacks=runner._callbacks,
            )
        )
        _close(actual, expected, dtype=torch.float64)
        _close(gradients, _gradients(expected, targets), dtype=torch.float64)
        with torch.no_grad():
            runner.parameters["integrator.mech.mechanisms.hh.gnabar_param"].mul_(1.01)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    (("myelinated", "scan", 2), ("multi_block", "while", None)),
)
def test_cuda_inductor_compile_fresh_scan_gradients_and_while_full_traces(
    kind, execution, checkpoint_every
):
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
        _assert_coverage(recorded, gradients, block=True, dtype=torch.float32)
