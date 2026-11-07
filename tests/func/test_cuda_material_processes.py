"""Clamp, exchange, and diffusion acceptance on a real CUDA device.

Run ``pytest tests/func/test_cuda_material_processes.py -W error -m cuda``.
CPU collection/skips do not establish CUDA correctness. Device-generic
oracles include full state, independent material recurrences, fresh and
checkpointed initialization/continuation, and parameter gradients.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_clearance_process import (
    NAI,
    POOL,
    STEPS,
    _close,
    _explicit_reference,
    _runner,
)
from tests.func.test_cuda_convenience_initialization import _cross_precision_close
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_material_processes import (
    RESERVOIR,
    _arguments,
    _canonical_exchange_geometry,
    _exchange_continuation,
    _exercise,
    _gradients,
    _material_oracle,
    _model,
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


def _assert_coverage(recorded, gradients, *, block, dtype, steps=STEPS, family="clamp"):
    assert {
        "v",
        "i_membrane",
        "t",
        "hh.m",
        "hh.h",
        "hh.n",
        POOL,
        NAI,
        "na_ion.ena",
        "na_ion.ina",
    } <= recorded.keys()
    for name, value in recorded.items():
        assert value.shape[0] == steps + 1, name
        assert torch.isfinite(value).all(), name
        if name != "t":
            assert value.device.type == "cuda" and value.dtype == dtype, name
    assert recorded["t"][0].item() == 0
    assert torch.count_nonzero(recorded["i_membrane"][0]) == 0
    if block:
        assert recorded["vc"].shape == (*recorded["v"].shape, 3)
        torch.testing.assert_close(
            recorded["vc"][0, ..., 0], recorded["v"][0], rtol=0, atol=0
        )
        assert torch.count_nonzero(recorded["vc"][0, ..., 1:]) == 0
    if gradients:
        expected = {
            "stimulus",
            "initial_voltage",
            "gnabar",
            "gkbar",
            "gl",
            "ena",
            "ek",
            "el",
            "geometry",
            "dt",
            "initial_pool",
            "initial_nai",
            "_BeforeClamp.rate",
            "_BeforeClamp.target",
        }
        if family == "clamp":
            expected |= {
                "_BathClamp.value",
                "_BathClamp.lower",
                "_BathClamp.upper",
                "_BathClamp.sodium",
            }
        else:
            assert RESERVOIR in recorded
            expected.add("initial_reservoir")
            if family == "exchange_geometry":
                expected |= {"dx", "_GeometryExchange.conductance"}
            elif family == "exchange_inactive":
                expected |= {
                    "_InactiveExchange.rate",
                    "_InactiveExchange.volume_a",
                    "_InactiveExchange.volume_b",
                }
            else:
                owner = {
                    "exchange": "_ExactExchange",
                    "exchange_implicit": "_ImplicitExchange",
                    "exchange_explicit": "_ExplicitExchange",
                }[family]
                expected |= {
                    f"{owner}.{name}"
                    for name in ("rate", "conductance", "volume_a", "volume_b")
                }
        assert expected == gradients.keys()
        for name, value in gradients.items():
            assert value.device.type == "cuda", name
            assert torch.isfinite(value).all() and torch.count_nonzero(value), name


@pytest.mark.parametrize(
    ("kind", "checkpoint_every"), (("path", None), ("multi_block", 2))
)
def test_cuda_clamp_full_traces_fresh_initialization_and_all_gradients(
    kind, checkpoint_every
):
    recorded, gradients = _exercise(
        kind,
        device="cuda",
        dtype=torch.float64,
        shell_resistance_scale=100.0,
        checkpoint_every=checkpoint_every,
    )
    _assert_coverage(
        recorded, gradients, block=kind == "multi_block", dtype=torch.float64
    )


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_cuda_clamp_float32_matches_cpu_float64_traces_and_all_gradients(kind):
    pytest.importorskip("dendra_solvers")
    expected, actual = [
        _exercise(
            kind,
            device=device,
            dtype=dtype,
            shell_resistance_scale=100.0,
            checkpoint_every=2,
        )
        for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32))
    ]
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


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_cuda_clamp_checkpointed_continuation_full_traces_and_all_gradients(kind):
    model = _model(
        kind, device="cuda", dtype=torch.float64, shell_resistance_scale=100.0
    )
    runner = _runner(model, checkpoint_every=2)
    source = _module_snapshot(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    state, auxiliary = runner(
        parameters=parameters, initialization=initialization, inputs=inputs
    )
    final, continued = runner(
        state,
        parameters=auxiliary["parameters"],
        inputs=inputs,
        callback_state=auxiliary["callbacks"].state,
    )
    recorded = {
        name: torch.cat((value, continued["callbacks"]["trace"][name]))
        for name, value in auxiliary["callbacks"]["trace"].items()
    }
    gradients = _gradients(recorded, targets)
    (
        reference_parameters,
        reference_initialization,
        reference_inputs,
        reference_targets,
    ) = _arguments(runner, model)
    doubled = dn.func.RolloutInput(
        *(torch.cat((value, value)) for value in reference_inputs)
    )
    expected = _explicit_reference(
        runner, reference_parameters, reference_initialization, doubled, steps=2 * STEPS
    )
    _assert_tree_close(final, expected[0], rtol=3e-9, atol=3e-10)
    reference_recorded = dict(expected[1]["callbacks"]["trace"])
    _close(recorded, reference_recorded)
    _close(gradients, _gradients(reference_recorded, reference_targets))
    _material_oracle(recorded, parameters, runner.constants["dt"])
    _assert_coverage(
        recorded,
        gradients,
        block=kind == "multi_block",
        dtype=torch.float64,
        steps=2 * STEPS,
    )
    _assert_module_snapshot(model, source)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    (("path", "scan", 2), ("multi_block", "while", None)),
)
def test_cuda_inductor_clamp_checkpointed_scan_and_while_full_traces(
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
    _assert_coverage(
        recorded, gradients, block=kind == "multi_block", dtype=torch.float32
    )
    if execution == "while":
        assert not gradients and not recorded["v"].requires_grad


@pytest.mark.parametrize(
    ("kind", "checkpoint_every"), (("path", None), ("multi_block", 2))
)
def test_cuda_exchange_full_traces_fresh_initialization_and_all_gradients(
    kind, checkpoint_every
):
    recorded, gradients = _exercise(
        kind,
        family="exchange",
        device="cuda",
        dtype=torch.float64,
        shell_resistance_scale=100.0,
        checkpoint_every=checkpoint_every,
    )
    _assert_coverage(
        recorded,
        gradients,
        family="exchange",
        block=kind == "multi_block",
        dtype=torch.float64,
    )


@pytest.mark.parametrize("family", ("exchange_implicit", "exchange_explicit"))
def test_cuda_exchange_methods_and_conservation(family):
    recorded, gradients = _exercise(
        "point", family=family, device="cuda", dtype=torch.float64
    )
    _assert_coverage(
        recorded, gradients, family=family, block=False, dtype=torch.float64
    )


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_cuda_exchange_float32_matches_cpu_float64_traces_and_all_gradients(kind):
    pytest.importorskip("dendra_solvers")
    expected, actual = [
        _exercise(
            kind,
            family="exchange",
            device=device,
            dtype=dtype,
            shell_resistance_scale=100.0,
            checkpoint_every=2,
        )
        for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32))
    ]
    _assert_coverage(
        actual[0],
        actual[1],
        family="exchange",
        block=kind == "multi_block",
        dtype=torch.float32,
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


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_cuda_exchange_checkpointed_continuation_full_traces_and_all_gradients(kind):
    recorded, gradients = _exchange_continuation(
        kind, device="cuda", dtype=torch.float64, shell_resistance_scale=100.0
    )
    _assert_coverage(
        recorded,
        gradients,
        family="exchange",
        block=kind == "multi_block",
        dtype=torch.float64,
        steps=2 * STEPS,
    )


def test_cuda_exchange_zero_volume_identity_and_gradients_are_finite():
    recorded, gradients = _exercise(
        "point", family="exchange_inactive", device="cuda", dtype=torch.float32
    )
    _assert_coverage(
        recorded,
        gradients,
        family="exchange_inactive",
        block=False,
        dtype=torch.float32,
    )
    for name in ("volume_a", "volume_b"):
        derivative = gradients[f"_InactiveExchange.{name}"]
        assert torch.count_nonzero(derivative[..., :2]) == 0
        assert torch.count_nonzero(derivative[..., 2:]) > 0


def test_cuda_exchange_differentiable_analytic_geometry():
    recorded, gradients = _exercise(
        "path", family="exchange_geometry", device="cuda", dtype=torch.float64
    )
    _assert_coverage(
        recorded,
        gradients,
        family="exchange_geometry",
        block=False,
        dtype=torch.float64,
    )


@pytest.mark.parametrize("kind", ("tree", "multi_block"))
def test_cuda_exchange_canonical_and_packed_area_constants_are_live(kind):
    recorded, gradients = _canonical_exchange_geometry(
        kind, device="cuda", dtype=torch.float64, shell_resistance_scale=100.0
    )
    assert {
        "v",
        "i_membrane",
        "t",
        POOL,
        NAI,
        RESERVOIR,
        "hh.m",
        "hh.h",
        "hh.n",
    } <= recorded.keys()
    assert {
        "stimulus",
        "initial_voltage",
        "gnabar",
        "gkbar",
        "gl",
        "ena",
        "ek",
        "el",
        "canonical_area",
    } <= gradients.keys()
    if kind == "tree":
        assert "exact_volume" in gradients
    else:
        assert "vc" in recorded
    for name, value in gradients.items():
        assert (
            value.device.type == "cuda"
            and torch.isfinite(value).all()
            and torch.count_nonzero(value)
        ), name


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    (("path", "scan", 2), ("multi_block", "while", None)),
)
def test_cuda_inductor_exchange_checkpointed_scan_and_while_full_traces(
    kind, execution, checkpoint_every
):
    recorded, gradients = _exercise(
        kind,
        family="exchange",
        device="cuda",
        dtype=torch.float32,
        shell_resistance_scale=100.0,
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
    _assert_coverage(
        recorded,
        gradients,
        family="exchange",
        block=kind == "multi_block",
        dtype=torch.float32,
    )
    if execution == "while":
        assert not gradients and not recorded["v"].requires_grad
