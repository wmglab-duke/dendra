"""Skip-voltage execution keeps actual current frames and voltage ownership."""

from __future__ import annotations

import pytest
import torch

from dendra.models.integrators.implicit import _bwd_euler_sc_skip
from tests.test_integrator_explicit_implicit import CableModel, LinearMechanism


class _CallLocalFrames(LinearMechanism):
    def __init__(self, shift):
        super().__init__(shift=shift)
        self.phases = []
        self.accepted = None

    def update_v(self, v):
        self.phases.append("voltage")
        return super().update_v(v)

    def advance_pre_current(self, v, dt, temp):
        self.phases.append("pre")
        self.pre_voltage = v

    def _evaluate_membrane_frame(self, v, *, call_local, require_current):
        assert call_local and require_current
        self.phases.append("current")
        # Deliberately different current and linearization coefficients expose
        # any attempted voltage solve or correction of the accepted ion frame.
        current = 0.1 * v.square()
        conductance = torch.full_like(v, 1.0e6)
        source = torch.full_like(v, -1.0e9)
        return conductance, source, current, (0.25 * current,), (conductance,)

    def i(self, v):
        raise AssertionError("call-local execution must not publish scratch currents")

    def capture_ion_current_frame(self):
        raise AssertionError("call-local execution must retain its returned frame")

    def advance_post_current(self, v, dt, temp, ion_current_frame):
        self.phases.append("post")
        self.post_voltage = v
        self.accepted = ion_current_frame


def test_scnv_functional_state_does_not_change_imperative_voltage_ownership():
    model = CableModel(shape=(1, 2))
    integrator = _bwd_euler_sc_skip(model, LinearMechanism())
    spec = integrator._functional_spec()

    assert integrator.v_vars == []
    assert integrator._FUNCTIONAL_STATE_NAMES == ("v",)
    assert spec.operator_kind == "scalar_point_skip"
    assert spec.implementation is _bwd_euler_sc_skip
    assert spec.workspace_schema == (("cmdt", "parameter"), ("area", "geometry"))


def test_scnv_prepared_workspace_preserves_independent_storage_and_gradients():
    model = CableModel(shape=(1, 2))
    integrator = _bwd_euler_sc_skip(model, LinearMechanism())
    cm = torch.tensor([[1.2, 2.0]], dtype=torch.float64, requires_grad=True)
    area = torch.tensor([[0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    dt = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

    workspace = integrator._derive_prepared_workspace(dt, cm=cm, area=area)
    assert workspace["area"].data_ptr() != area.data_ptr()
    gradients = torch.autograd.grad(
        workspace["cmdt"].sum() + workspace["area"].sum(), (cm, area, dt)
    )
    torch.testing.assert_close(gradients[0], torch.full_like(cm, 1.0e-3 / dt.detach()))
    torch.testing.assert_close(gradients[1], torch.ones_like(area))
    torch.testing.assert_close(
        gradients[2], -1.0e-3 * cm.detach().sum() / dt.detach().square()
    )


@pytest.mark.parametrize("imem", (False, True))
def test_scnv_call_local_actual_currents_keep_phase_order_and_capacitive_gradient(imem):
    model = CableModel(shape=(1, 2), v_init=-5.0)
    voltage = torch.tensor([[-5.0, -2.0]], dtype=torch.float64, requires_grad=True)
    shift = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
    handler = _CallLocalFrames(shift)
    integrator = _bwd_euler_sc_skip(model, handler, imem=imem)
    integrator.initialize(model, 0.1)

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("scnv must not solve membrane voltage")

    new_voltage, membrane = integrator._step(
        voltage,
        0.1,
        model.celsius,
        ve=torch.full_like(voltage, torch.nan),
        intra=torch.full_like(voltage, torch.nan),
        solver=unexpected_solver,
        call_local_currents=True,
    )
    expected_voltage = voltage + shift
    expected_current = 0.1 * expected_voltage.square()
    assert handler.phases == ["voltage", "pre", "current", "post"]
    torch.testing.assert_close(new_voltage, expected_voltage)
    torch.testing.assert_close(handler.pre_voltage, expected_voltage)
    torch.testing.assert_close(handler.post_voltage, expected_voltage)
    torch.testing.assert_close(handler.accepted[0], 0.25 * expected_current)

    loss = new_voltage.sum()
    expected_v_gradient = torch.ones_like(voltage)
    expected_shift_gradient = voltage.new_tensor(voltage.numel())
    if imem:
        expected_membrane = (
            integrator.cmdt * shift + expected_current
        ) * integrator.area
        torch.testing.assert_close(membrane, expected_membrane)
        loss = loss + membrane.sum()
        expected_v_gradient = (
            expected_v_gradient + 0.2 * expected_voltage * integrator.area
        )
        expected_shift_gradient = (
            expected_shift_gradient
            + ((integrator.cmdt + 0.2 * expected_voltage) * integrator.area).sum()
        )
    else:
        assert membrane is None
    voltage_gradient, shift_gradient = torch.autograd.grad(loss, (voltage, shift))
    torch.testing.assert_close(voltage_gradient, expected_v_gradient)
    torch.testing.assert_close(shift_gradient, expected_shift_gradient)
