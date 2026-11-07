"""One pure callback contract serves eager, scan, and while execution."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_lowering_api import DT, _close, _network, _population, _target

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


@dataclass(frozen=True)
class _Statistics(dn.func.FunctionalCallback):
    scale: torch.Tensor

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        voltage = state["integrator"]["v"]
        return {
            "energy": voltage.new_zeros(()),
            "frames": voltage.new_zeros((), dtype=torch.long),
            "seen": voltage.new_zeros((), dtype=torch.bool),
            "scale": self.scale,
        }, None

    def update(self, carry, state, auxiliary):
        voltage = state["integrator"]["v"]
        observed = voltage.mean() * carry["scale"]
        return {
            "energy": carry["energy"] + voltage.square().mean() * carry["scale"],
            "frames": carry["frames"] + 1,
            "seen": carry["seen"] | (voltage > -100).any(),
            "scale": carry["scale"],
        }, {"value": observed, "alias": observed}

    def finalize(self, carry, emissions):
        if emissions is None:
            emissions = {
                "value": carry["energy"].new_empty((0,)),
                "alias": carry["energy"].new_empty((0,)),
            }
        return {"carry": carry, "samples": emissions}


class _VoltageSum(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        return carry + state["integrator"]["v"].sum(), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


def _callbacks(functional, network, scale):
    def scoped(callback):
        return dn.func.PopulationCallback("left", callback) if network else callback

    statistics = _Statistics(scale)
    return functional.make_callbacks(
        {
            "statistics": scoped(statistics),
            "sum": scoped(_VoltageSum()),
            "trace": dn.func.Recorder(("left.v", "t") if network else ("v", "t")),
        }
    ), statistics


def _callback_loss(auxiliary):
    values = auxiliary["callbacks"]["statistics"]
    return values["carry"]["energy"] + values["samples"]["value"].square().mean()


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_custom_callbacks_share_gradients_carry_emissions_and_live_continuation(
    network, execution
):
    model = _network() if network else _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    scale = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    callbacks, authored = _callbacks(functional, network, scale)
    kernel = functional.compile_rollout_chunk(
        4, execution=execution, backend="aot_eager"
    )
    bound = kernel.bind(tensors)
    frozen = scale.detach().clone()
    with torch.no_grad() if execution == "while" else nullcontext():
        expected, reference = dn.func.run(
            functional.bind(tensors), tensors.state, steps=9, callbacks=callbacks
        )
        actual, auxiliary = dn.func.run(
            bound, tensors.state, steps=9, callbacks=callbacks
        )
    _close(actual, expected)
    _close(auxiliary["callbacks"], reference["callbacks"])
    assert authored.scale is scale
    torch.testing.assert_close(scale.detach(), frozen)
    statistics = auxiliary["callbacks"]["statistics"]
    assert statistics["carry"]["frames"].item() == 9
    assert statistics["carry"]["seen"].item()
    assert statistics["samples"]["value"].shape == (9,)
    assert auxiliary["callbacks"]["trace"]["t"].shape[0] == 10
    report = kernel.execution_report()
    assert report.requested_execution == execution and report.steps == 1
    assert report.strategy == (
        "compiled_scan" if execution == "scan" else "compiled_unrolled"
    )
    if execution == "scan":
        targets = (_target(tensors.parameters, network), scale)
        expected_grad = torch.autograd.grad(_callback_loss(reference), targets)
        actual_grad = torch.autograd.grad(_callback_loss(auxiliary), targets)
        _close(actual_grad, expected_grad)
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0
            for value in actual_grad
        )
        _prefix_state, differentiable_prefix = dn.func.run(
            bound, tensors.state, steps=5, callbacks=callbacks
        )
        _resumed_state, differentiable_suffix = dn.func.run(
            bound, differentiable_prefix["continuation"], steps=4, callbacks=callbacks
        )
        combined = torch.cat(
            (
                differentiable_prefix["callbacks"]["statistics"]["samples"]["value"],
                differentiable_suffix["callbacks"]["statistics"]["samples"]["value"],
            )
        )
        split_loss = (
            differentiable_suffix["callbacks"]["statistics"]["carry"]["energy"]
            + combined.square().mean()
        )
        split_grad = torch.autograd.grad(split_loss, targets)
        _close(split_grad, expected_grad)
    else:
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        with pytest.raises(dn.func.FunctionalizationError, match="inference-only"):
            kernel(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
        assert kernel.execution_report() is report
    other = functional.compile_rollout_chunk(
        4, execution="while" if execution == "scan" else "scan", backend="aot_eager"
    )
    with torch.no_grad():
        first, prefix = dn.func.run(
            other.bind(tensors), tensors.state, steps=5, callbacks=callbacks
        )
        resumed, suffix = dn.func.run(
            bound, prefix["continuation"], steps=4, callbacks=callbacks
        )
        previous = kernel.execution_report()
        unchanged, empty = dn.func.run(
            bound, suffix["continuation"], steps=0, callbacks=callbacks
        )
    _close(resumed, expected)
    _close(unchanged, resumed)
    assert kernel.execution_report() is previous
    _close(suffix["callbacks"]["statistics"]["carry"], statistics["carry"])
    _close(suffix["callbacks"]["sum"], auxiliary["callbacks"]["sum"])
    for name in ("value", "alias"):
        observed = torch.cat(
            (
                prefix["callbacks"]["statistics"]["samples"][name],
                suffix["callbacks"]["statistics"]["samples"][name],
            )
        )
        _close(observed, statistics["samples"][name])
        assert empty["callbacks"]["statistics"]["samples"][name].shape == (0,)
    for name in auxiliary["callbacks"]["trace"]:
        joined = torch.cat(
            (prefix["callbacks"]["trace"][name], suffix["callbacks"]["trace"][name])
        )
        _close(joined, auxiliary["callbacks"]["trace"][name])
        assert empty["callbacks"]["trace"][name].shape[0] == 0
    _close(first, prefix["continuation"].state)


class _ChangesCarry(_VoltageSum):
    def update(self, carry, state, auxiliary):
        return carry.unsqueeze(0), None


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_custom_callback_schema_checks_remain_required_for_each_engine(execution):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    callbacks = functional.make_callbacks({"broken": _ChangesCarry()})
    kernel = functional.compile_rollout_chunk(4, execution=execution, backend="eager")
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    with (
        torch.no_grad(),
        pytest.raises(dn.func.FunctionalizationError, match="carry|shape|must use"),
    ):
        kernel(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
    assert kernel.execution_report() is None


class _TensorBranch(_VoltageSum):
    def update(self, carry, state, auxiliary):
        if state["integrator"]["v"].mean().item() > 0:
            return carry + 1, None
        return carry - 1, None


def test_capture_incompatible_custom_callback_preserves_last_report_and_no_fallback():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    graphs = []

    def backend(graph, examples):
        graphs.append(graph)
        return graph.forward

    kernel = functional.compile_rollout_chunk(8, execution="while", backend=backend)
    incompatible = functional.make_callbacks({"incompatible": _TensorBranch()})
    with (
        torch.no_grad(),
        pytest.raises(dn.func.FunctionalizationError, match="unavailable"),
    ):
        kernel(tensors.parameters, prepared, tensors.state, callbacks=incompatible)
    assert not graphs and kernel.execution_report() is None
    callbacks = functional.make_callbacks({"sum": _VoltageSum()})
    with torch.no_grad():
        kernel(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
    previous, count = kernel.execution_report(), len(graphs)
    assert previous.strategy == "compiled_while_loop" and count
    with (
        torch.no_grad(),
        pytest.raises(dn.func.FunctionalizationError, match="unavailable"),
    ):
        kernel(tensors.parameters, prepared, tensors.state, callbacks=incompatible)
    assert kernel.execution_report() is previous and len(graphs) == count


@pytest.mark.mps
def test_mps_custom_callback_while_uses_shared_tensor_carry_and_emissions():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        functional, tensors = dn.func.make_functional(_population(), dt=DT)
        scale = torch.tensor(0.25, device="mps", dtype=torch.float32)
        callbacks, _ = _callbacks(functional, False, scale)
        kernel = functional.compile_rollout_chunk(
            4, execution="while", backend="aot_eager"
        )
        with torch.no_grad():
            expected, reference = dn.func.run(
                functional.bind(tensors), tensors.state, steps=5, callbacks=callbacks
            )
            actual, auxiliary = dn.func.run(
                kernel.bind(tensors), tensors.state, steps=5, callbacks=callbacks
            )
        _close(actual, expected)
        _close(auxiliary["callbacks"], reference["callbacks"])
        assert auxiliary["callbacks"]["statistics"]["carry"]["frames"].item() == 5
        assert kernel.execution_report().requested_execution == "while"
        assert kernel.execution_report().steps == 1
