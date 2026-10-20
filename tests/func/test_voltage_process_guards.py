"""Pure voltage hooks fail closed without mutating their source Population.

These admission/guard probes are intentionally eager: compiler coverage lives
in test_voltage_processes, rather than compiling every invalid program.
"""

from __future__ import annotations

import random

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import VoltageProcess
from dendra.models.mod import pas
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)

pytestmark = pytest.mark.cpu
DT = 0.05
_HIDDEN_TENSOR = torch.tensor(1.0)
_HIDDEN_CONFIGURATION = {"offset": 1.0}


class _PureCounter(VoltageProcess):
    VoltageProcess.GLOBAL(offset=0.5)
    VoltageProcess.CARRY("counter")

    def voltage_values(self, v, dt, values):
        return v + self.offset + values["counter"], {"counter": values["counter"] + dt}


class _RebindCarry(_PureCounter):
    def voltage_values(self, v, dt, values):
        self.counter = self.counter + dt
        return v, {}


class _MutateCarry(_PureCounter):
    def voltage_values(self, v, dt, values):
        values["counter"].add_(dt)
        return v, {}


class _MutateVoltage(_PureCounter):
    def voltage_values(self, v, dt, values):
        v.add_(self.offset)
        return v, {}


class _MutateParameter(_PureCounter):
    def voltage_values(self, v, dt, values):
        self.offset.add_(1)
        return v, {}


class _MutatePythonState(_PureCounter):
    calls = 0

    def voltage_values(self, v, dt, values):
        self.calls += 1
        return v, {}


class _TorchRandom(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v + torch.rand_like(v), {}


class _PythonRandom(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v + random.random(), {}


class _HiddenTensor(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v + _HIDDEN_TENSOR, {}


class _HiddenConfiguration(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v + _HIDDEN_CONFIGURATION["offset"], {}


class _LegacyReset(_PureCounter):
    def update_v(self, v):
        self.counter = self.counter + self.dt
        return v + self.offset


class _InvalidTuple(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v


class _InvalidMapping(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v, []


class _UnknownCarry(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v, {"undeclared": torch.zeros_like(v)}


class _InvalidVoltageShape(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v[..., 0], {}


class _InvalidVoltageDtype(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v.to(torch.int64), {}


class _InvalidCarryShape(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v, {"counter": values["counter"][..., 0]}


class _InvalidCarryDtype(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v, {"counter": values["counter"].to(torch.int64)}


class _HelperVoltage(_PureCounter):
    factor = 1.0

    def _shift(self, v):
        return v + self.factor * self.offset

    def voltage_values(self, v, dt, values):
        return self._shift(v), {"counter": values["counter"] + dt}


class _RegisteredCarryRead(_PureCounter):
    def voltage_values(self, v, dt, values):
        return v + self.offset + self.counter, {"counter": values["counter"] + dt}


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _model(mechanism=_PureCounter):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(N=1, C=2, dtype=torch.float64)
        model.insert(pas)
        model.insert(mechanism)
        model.initialize()
        model.train()
    return model


@pytest.mark.parametrize(
    "mechanism",
    (
        _RebindCarry,
        _MutateCarry,
        _MutateVoltage,
        _MutateParameter,
        _MutatePythonState,
        _TorchRandom,
        _PythonRandom,
        _HiddenTensor,
        _HiddenConfiguration,
    ),
)
def test_impure_voltage_hooks_are_rejected_without_source_changes(
    mechanism, cpu_threads
):
    model = _model(mechanism)
    source = _module_snapshot(model)
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, source)
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    assert random.getstate() == python_rng
    assert model.mech.mechanisms[mechanism.__name__].__dict__.get("calls", 0) == 0


@pytest.mark.parametrize(
    "mechanism",
    (
        _InvalidTuple,
        _InvalidMapping,
        _UnknownCarry,
        _InvalidVoltageShape,
        _InvalidVoltageDtype,
        _InvalidCarryShape,
        _InvalidCarryDtype,
    ),
)
def test_voltage_hook_output_schemas_are_validated_without_source_changes(
    mechanism, cpu_threads
):
    model = _model(mechanism)
    source = _module_snapshot(model)
    with pytest.raises(
        (dn.func.FunctionalizationError, TypeError, ValueError, KeyError),
        match="voltage_values|voltage|counter|carry|undeclared",
    ):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, source)


def test_legacy_voltage_override_remains_imperative_only(cpu_threads):
    model = _model(_LegacyReset)
    process = model.mech.mechanisms["_LegacyReset"]
    process._configure_timestep(DT)
    before = process.counter.clone()
    result = process.update_v(model.v)
    torch.testing.assert_close(result, model.v + process.offset)
    torch.testing.assert_close(process.counter, before + DT)
    source = _module_snapshot(model)
    with pytest.raises(dn.func.FunctionalizationError, match="update_v|adapter"):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, source)


def _changed_voltage(self, v, dt, values):
    return v + 100, {}


@pytest.mark.parametrize("change", ("hook", "helper", "configuration", "schema"))
def test_voltage_program_changes_invalidate_existing_functional_plan(
    change, monkeypatch, cpu_threads
):
    model = _model(_HelperVoltage)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    process = model.mech.mechanisms["_HelperVoltage"]
    if change == "hook":
        monkeypatch.setattr(_HelperVoltage, "voltage_values", _changed_voltage)
    elif change == "helper":
        monkeypatch.setattr(_HelperVoltage, "_shift", _changed_voltage)
    elif change == "configuration":
        monkeypatch.setattr(_HelperVoltage, "factor", 2.0)
    else:
        process._carry = (*process._carry, "another")
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Ss]tructur|changed|[Ss]chema"
    ):
        functional.prepare(tensors.parameters, tensors.constants)
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Ss]tructur|changed|[Ss]chema"
    ):
        functional.step(tensors.parameters, prepared, tensors.state)


def test_voltage_parameter_replacements_remain_live_differentiable_inputs(cpu_threads):
    model = _model()
    source = _module_snapshot(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    baseline, _ = functional.step(tensors.parameters, prepared, tensors.state)
    key = next(name for name in tensors.parameters if name.endswith(".offset_param"))
    replacement = (tensors.parameters[key].detach() + 0.7).requires_grad_()
    parameters = {**tensors.parameters, key: replacement}
    prepared = functional.prepare(parameters, tensors.constants)
    actual, _ = functional.step(parameters, prepared, tensors.state)
    assert not torch.allclose(actual["integrator"]["v"], baseline["integrator"]["v"])
    gradient = torch.autograd.grad(actual["integrator"]["v"].sum(), replacement)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    _assert_module_snapshot(model, source)


def test_registered_voltage_carry_reads_use_the_explicit_functional_state(cpu_threads):
    model = _model(_RegisteredCarryRead)
    source = _module_snapshot(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    baseline, _ = functional.step(tensors.parameters, prepared, tensors.state)
    state = torch.utils._pytree.tree_map(torch.clone, tensors.state)
    state["mechanism_buffers"]["_RegisteredCarryRead"]["counter"].fill_(2)
    changed, _ = functional.step(tensors.parameters, prepared, state)
    assert not torch.allclose(changed["integrator"]["v"], baseline["integrator"]["v"])
    torch.testing.assert_close(
        changed["mechanism_buffers"]["_RegisteredCarryRead"]["counter"],
        state["mechanism_buffers"]["_RegisteredCarryRead"]["counter"] + DT,
    )
    _assert_module_snapshot(model, source)


def test_voltage_output_validation_resets_after_module_dtype_conversion(cpu_threads):
    model = _model()
    process = model.mech.mechanisms["_PureCounter"]
    process.update_v(model.v)
    assert process._voltage_schema_validated
    process.to(dtype=torch.float32)
    assert not process._voltage_schema_validated
    process.update_v(model.v.to(torch.float32))
    assert process._voltage_schema_validated
