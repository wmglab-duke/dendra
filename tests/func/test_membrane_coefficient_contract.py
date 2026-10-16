"""Direct coefficients are executable structure, not silently mutable weights."""

from types import MethodType

import pytest
import torch

import dendra as dn
from dendra.models.mod import hh


def _model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(hh)
        model.initialize()
    return model


@pytest.mark.parametrize(
    "hook",
    [
        "_evaluate_membrane_frame",
        "_can_evaluate_direct_membrane_frame",
        "_direct_membrane_dtype_compatible",
        "_evaluate_direct_membrane_coefficients",
        "_gather_assigned_current_fields",
    ],
)
def test_functionalization_rejects_handler_assembly_override(hook):
    model = _model()
    setattr(model.mech, hook, lambda *args, **kwargs: None)
    with pytest.raises(dn.func.FunctionalizationError, match="current-frame evaluator"):
        dn.func.make_functional(model, dt=0.01)


@pytest.mark.parametrize("hook", ["i_affine_terms", "i_with_g", "i"])
def test_coefficient_method_replacement_invalidates_existing_functional_plan(hook):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=0.01)
    mechanism = model.mech.mechanisms["hh"]
    original = getattr(mechanism, hook)

    def replacement(self, v):
        del self
        return original(v)

    setattr(mechanism, hook, MethodType(replacement, mechanism))
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)


def test_coefficient_code_replacement_changes_functional_structure(monkeypatch):
    from dendra.func._population import _structure_signature

    model = _model()
    before = _structure_signature(model)
    coefficient = model.mech.mechanisms["hh"].i_affine_terms.__func__

    def replacement(self, v):
        return torch.zeros_like(v), torch.zeros_like(v)

    monkeypatch.setattr(coefficient, "__code__", replacement.__code__)
    assert _structure_signature(model) != before


@pytest.mark.parametrize("hook", ["i_affine_terms"])
def test_declared_coefficient_override_before_capture_is_used_and_checked(hook):
    model = _model()
    mechanism = model.mech.mechanisms["hh"]
    original = getattr(mechanism, hook)

    def replacement(self, v):
        del self
        return original(v)

    setattr(mechanism, hook, MethodType(replacement, mechanism))
    with dn.ctx(MEMBRANE_ASSEMBLY="validate"):
        assert model.integrator._membrane_frame(model.v, call_local=True)[2] is None
    functional, tensors = dn.func.make_functional(model, dt=0.01)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    final, _ = functional.step(tensors.parameters, prepared, tensors.state)
    assert torch.isfinite(final["integrator"]["v"]).all()
