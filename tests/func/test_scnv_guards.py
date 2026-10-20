"""Voltage-skip carry declarations remain explicit and source guarded."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.func._population import _resolve_functional_integrator_spec
from dendra.models.integrators.implicit import _bwd_euler_sc_skip
from dendra.models.mod import pas
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)

DT = 0.05
pytestmark = pytest.mark.cpu


class _AddVoltage(torch.nn.Module):
    def forward(self, voltage, increment):
        return (voltage + increment,)


class _CustomSkip(_bwd_euler_sc_skip):
    pass


def _model(*, integrator=None, transform=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True):
        model = dn.SingleCompartment(
            N=1,
            C=2,
            dtype=torch.float64,
            integrator=dn.scnv() if integrator is None else integrator,
        )
        model.insert(pas)
        if transform:
            model.register_pre_initialize_transform(
                "shift",
                _AddVoltage(),
                reads=("state.integrator.v",),
                writes=("state.integrator.v",),
                inputs={"increment": torch.tensor(1.0, dtype=torch.float64)},
            )
        model.initialize()
    return model


def test_scnv_pre_initialization_transform_can_read_and_replace_exposed_voltage():
    model = _model(transform=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    source = _module_snapshot(model)
    increment = model.v.new_tensor(3.0, requires_grad=True)
    inputs = tensors.initialization._replace(
        transforms={"pre.shift.increment": increment}
    )
    initialized = functional.initialize(tensors.parameters, tensors.constants, inputs)
    voltage = initialized.state["integrator"]["v"]
    torch.testing.assert_close(voltage, inputs.v_init + increment)
    (gradient,) = torch.autograd.grad(voltage.sum(), (increment,))
    torch.testing.assert_close(gradient, increment.new_tensor(model.v.numel()))
    assert model.integrator.v_vars == []
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize(
    "hook",
    ("_FUNCTIONAL_STATE_NAMES", "_prepare_workspace", "_step", "v_vars"),
)
def test_scnv_rejects_instance_contract_overrides_without_source_changes(hook):
    model = _model()
    if hook == "_FUNCTIONAL_STATE_NAMES":
        replacement = ("v",)
    elif hook == "v_vars":
        replacement = ["v"]
    else:

        def replacement(*args, **kwargs):
            return None

    setattr(model.integrator, hook, replacement)
    source = _module_snapshot(model)
    with pytest.raises(
        dn.func.FunctionalizationError, match="functionalization requires"
    ):
        dn.func.make_functional(model, dt=DT)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("declaration", (("v", "other"), (), ["v"]))
def test_scnv_state_declaration_must_match_canonical_contract(monkeypatch, declaration):
    model = _model()
    monkeypatch.setattr(_bwd_euler_sc_skip, "_FUNCTIONAL_STATE_NAMES", declaration)
    assert (
        _resolve_functional_integrator_spec(model.integrator, "scalar_point_skip")
        is None
    )
    with pytest.raises(
        dn.func.FunctionalizationError, match="Integrator|functionalization requires"
    ):
        dn.func.make_functional(model, dt=DT)


def test_scnv_state_declaration_changes_invalidate_existing_plan(monkeypatch):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    monkeypatch.setattr(_bwd_euler_sc_skip, "_FUNCTIONAL_STATE_NAMES", ("v", "other"))
    with pytest.raises(
        dn.func.FunctionalizationError, match="structure|changed|compatible"
    ):
        functional.prepare(tensors.parameters, tensors.constants)


def test_empty_solved_voltage_does_not_admit_unrelated_integrators():
    model = _model(integrator=dn.bwd_euler_sc())
    model.integrator.v_vars = []
    with pytest.raises(dn.func.FunctionalizationError, match="tensor-state names"):
        dn.func.make_functional(model, dt=DT)


def test_scnv_custom_subclass_remains_outside_canonical_contract():
    model = _model(integrator=_CustomSkip)
    with pytest.raises(
        dn.func.FunctionalizationError, match="functionalization requires"
    ):
        dn.func.make_functional(model, dt=DT)


def test_scnv_malformed_commit_is_atomic_for_exposed_voltage_and_carry():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    source = _module_snapshot(model)
    malformed = {
        **tensors.state,
        "integrator": {**tensors.state["integrator"], "v": model.v[..., :1]},
    }
    with pytest.raises((dn.func.FunctionalizationError, ValueError), match="shape"):
        functional.commit_state_(model, malformed)
    _assert_module_snapshot(model, source)
