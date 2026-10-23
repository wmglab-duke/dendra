"""Stage, phase, and gradient contracts for shared explicit Runge-Kutta kernels."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.integrators.explicit import _euler, _rk2, _rk4

from .test_integrator_explicit_implicit import CableModel


class _StageCurrent(torch.nn.Module):
    def __init__(self, *, record=True):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float64))
        self.record = record
        self.phases = []
        self.stage_voltages = []
        self.accepted_frame = None
        self._last_frame = ()

    def _evaluate(self, v):
        ionic = self.gain * (v + 63.0) ** 3
        return ionic + 0.01 * (v + 65.0), (ionic,)

    def _record_current(self, v):
        if self.record:
            self.phases.append("current")
            self.stage_voltages.append(v)

    def advance_pre_current(self, v, dt, temp):
        if self.record:
            self.phases.append("pre")

    def advance_post_current(self, v, dt, temp, frame):
        if self.record:
            self.phases.append("post")
            self.accepted_frame = frame

    def iexp(self, v):
        self._record_current(v)
        current, self._last_frame = self._evaluate(v)
        return current

    def capture_ion_current_frame(self):
        return self._last_frame

    def _evaluate_explicit_current_frame(self, v):
        self._record_current(v)
        return self._evaluate(v)

    def update_v(self, v):
        raise AssertionError("Runge-Kutta does not apply VoltageProcess hooks")

    def i(self, v):
        raise AssertionError("Runge-Kutta does not evaluate implicit conductances")


@pytest.mark.parametrize("cls", [_rk2, _rk4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1,), (2, 3, 4)])
@pytest.mark.parametrize("training", [False, True])
def test_shared_runge_kutta_preserves_native_stages_frames_and_gradients(
    cls, dtype, shape, training
):
    model = CableModel(shape=shape).to(dtype=dtype)
    native = cls(model, _StageCurrent(), imem=True).to(dtype=dtype)
    native.initialize(model, 0.00001)
    native.ssd.weight = torch.nn.Parameter(
        torch.tensor([[[1.0, -2.0, 1.1], [0.9, -2.0, 1.0]]], dtype=dtype)
    )
    native.train(training)
    pure = copy.deepcopy(native)
    values = (
        torch.linspace(-66.0, -64.0, model.v.numel(), dtype=dtype).reshape(shape),
        torch.linspace(-0.1, 0.1, shape[-1], dtype=dtype),
        torch.full((shape[-1],), 1.0e-10, dtype=dtype),
        torch.tensor(0.00001, dtype=dtype),
    )
    native_inputs = tuple(value.clone().requires_grad_() for value in values)
    pure_inputs = tuple(value.clone().requires_grad_() for value in values)
    v, ve, intra, dt = native_inputs
    expected = native._step(v, ve, native.area_c, dt, 37.0, native.cm_c, intra)
    v, ve, intra, dt = pure_inputs
    actual = pure._functional_step(v, dt, 37.0, ve, intra, call_local_currents=True)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0.0, atol=0.0)
    stages = 2 if cls is _rk2 else 4
    for owner in (native, pure):
        assert owner.mech.phases == ["pre", *["current"] * stages, "post"]
        assert len(owner.mech.stage_voltages) == stages
        torch.testing.assert_close(owner.mech.stage_voltages[0], values[0])
        ionic_frames = tuple(
            owner.mech._evaluate(voltage)[1][0] for voltage in owner.mech.stage_voltages
        )
        if cls is _rk2:
            accepted = ionic_frames[1]
        else:
            accepted = sum(
                weight * frame
                for frame, weight in zip(
                    ionic_frames, (1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0)
                )
            )
        torch.testing.assert_close(
            owner.mech.accepted_frame[0], accepted, rtol=0.0, atol=0.0
        )
    for result, reference in zip(
        pure.mech.stage_voltages, native.mech.stage_voltages, strict=True
    ):
        torch.testing.assert_close(result, reference, rtol=0.0, atol=0.0)
    native_loss = (
        expected[0].square().sum()
        + expected[1].sum()
        + native.mech.accepted_frame[0].sum()
    )
    pure_loss = (
        actual[0].square().sum() + actual[1].sum() + pure.mech.accepted_frame[0].sum()
    )
    expected_gradients = torch.autograd.grad(
        native_loss, (*native_inputs, native.ssd.weight, native.mech.gain)
    )
    actual_gradients = torch.autograd.grad(
        pure_loss, (*pure_inputs, pure.ssd.weight, pure.mech.gain)
    )
    for result, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, reference, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("factory,cls", [(dn.rk2, _rk2), (dn.rk4, _rk4)])
def test_runge_kutta_advertises_shared_workspace_and_raw_geometry(factory, cls):
    model = CableModel(shape=(2, 4))
    model.cm_scale.fill_(7.0)
    model.rhoa_scale.fill_(3.0)
    model.area_scale.fill_(5.0)
    integrator = factory()(model, _StageCurrent()).to(dtype=torch.float64)
    integrator.initialize(model, 0.01)
    spec = integrator._functional_spec()
    assert spec.implementation is cls
    assert spec.operator_kind == "scalar_explicit_path"
    assert spec.workspace_schema == _euler._PREPARED_WORKSPACE_SCHEMA
    assert "_functional_step" in spec.critical_methods
    if cls is _rk4:
        assert "_combine_ion_current_frames" in spec.critical_methods
    expected = _euler._prepare_workspace(
        0.01, dx=model.dx, diam=model.diam, cm=model.cm, rhoa=model.rhoa
    )
    for name, reference in expected.items():
        torch.testing.assert_close(
            getattr(integrator, name), reference, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("cls,stages", [(_rk2, 2), (_rk4, 4)])
def test_native_runge_kutta_retains_custom_frk_dispatch(cls, stages):
    class CustomFRK(cls):
        def FRK(self, v, ve, area, cm, ra, intra=None):
            self.calls += 1
            slope, ionic = super().FRK(v, ve, area, cm, ra, intra)
            return 0.5 * slope, ionic

    model = CableModel(shape=(2, 4))
    integrator = CustomFRK(model, _StageCurrent(), imem=True).to(dtype=torch.float64)
    integrator.calls = 0
    integrator.initialize(model, 0.00001)
    result, _ = integrator._step(
        model.v,
        integrator.ve_zero,
        integrator.area_c,
        0.00001,
        model.celsius,
        integrator.cm_c,
    )
    assert integrator.calls == stages
    assert torch.isfinite(result).all()


class _PureRungeKutta(torch.nn.Module):
    def __init__(self, cls):
        super().__init__()
        model = CableModel(shape=(2, 4))
        self.integrator = cls(model, _StageCurrent(record=False), imem=True).to(
            dtype=torch.float64
        )
        self.integrator.initialize(model, 0.00001)
        self.integrator.ssd.weight.requires_grad_()

    def forward(self, voltage, ve, intra):
        return self.integrator._functional_step(
            voltage, 0.00001, 37.0, ve, intra, call_local_currents=True
        )


@pytest.mark.parametrize("cls", [_rk2, _rk4])
def test_shared_runge_kutta_fullgraph_compilation_preserves_gradients(cls):
    eager = _PureRungeKutta(cls)
    compiled_source = copy.deepcopy(eager)
    compiled = torch.compile(compiled_source, backend="aot_eager", fullgraph=True)
    values = (
        torch.linspace(-66.0, -64.0, 8, dtype=torch.float64).reshape(2, 4),
        torch.linspace(-0.1, 0.1, 4, dtype=torch.float64),
        torch.full((4,), 1.0e-10, dtype=torch.float64),
    )
    eager_inputs = tuple(value.clone().requires_grad_() for value in values)
    compiled_inputs = tuple(value.clone().requires_grad_() for value in values)
    expected = eager(*eager_inputs)
    actual = compiled(*compiled_inputs)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0.0, atol=0.0)
    expected_gradients = torch.autograd.grad(
        expected[0].square().sum() + expected[1].sum(),
        (*eager_inputs, *eager.parameters()),
    )
    actual_gradients = torch.autograd.grad(
        actual[0].square().sum() + actual[1].sum(),
        (*compiled_inputs, *compiled_source.parameters()),
    )
    for result, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, reference, rtol=2.0e-14, atol=2.0e-14)
