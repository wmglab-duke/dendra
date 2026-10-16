"""Single-current HH preserves independent three-current physics and autograd."""

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism as M
from dendra.models.mod.HH import hh, mhn

from .test_hh_rate_configuration import (
    CHANNEL,
    GATES,
    close,
    membrane_trace_and_gradients,
    model_with,
)


class ThreeCurrentRateHH(M):
    """Independent current arithmetic; shares only the unchanged gate solver."""

    M.STATE_BUNDLE(mhn)
    M.GLOBAL(gnabar=0.12, gkbar=0.036, gl=0.0003, ena=50.0, ek=-77.0, el=-54.3)
    M.NONSPECIFIC_CURRENT("il", "ina", "ik")

    def il(self, v):
        return self.gl * (v - self.el)

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)

    def ik(self, v):
        return self.gkbar * self.n**4 * (v - self.ek)


class ScaledHH(hh):
    M.AFFINE("i")

    def i_affine_terms(self, v):
        g, j = super().i_affine_terms(v)
        return 2.0 * g, 2.0 * j


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [6.3, 23.0, 36.0])
@pytest.mark.parametrize("kind", ["sc", "dhs"])
@pytest.mark.parametrize("assembly", ["off", "auto", "validate"])
def test_all_voltage_and_gate_samples_and_all_parameter_gradients(
    dtype, temperature, kind, assembly
):
    actual, actual_gradients = membrane_trace_and_gradients(
        hh, temperature, dtype, kind, assembly=assembly, imem=False
    )
    expected, expected_gradients = membrane_trace_and_gradients(
        ThreeCurrentRateHH, temperature, dtype, kind, assembly="off", imem=False
    )
    if dtype == torch.float32:
        # Same strict bounds as the independent pre-rate-form HH regression.
        torch.testing.assert_close(actual[:, 0], expected[:, 0], rtol=0, atol=3e-4)
        close(actual[:, 1:], expected[:, 1:], dtype)
    else:
        close(actual, expected, dtype)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        close(actual_gradient, expected_gradient, dtype, gradient=True)


def test_one_registered_current_and_generated_current_conductance_adapter():
    model = model_with(hh, 23.0, torch.float64, "sc", imem=False)
    mechanism = next(iter(model.mech.mechanisms.values()))
    assert tuple(mechanism._currents["nonspecific"]) == ("i",)
    assert not hasattr(mechanism, "_direct_affine_terms")
    assert not hasattr(mechanism, "_direct_affine_coefficient_fields")
    for name in ("ina", "ik", "il"):
        assert not hasattr(hh, name)
        assert not hasattr(mechanism, name)
    for name in CHANNEL:
        assert getattr(mechanism, name).shape == ()
        assert getattr(mechanism, f"{name}_param").shape == ()
    for name in GATES:
        assert getattr(mechanism, name).shape == model.v.shape
    voltage = torch.linspace(-80, 20, model.v.numel(), dtype=model.v.dtype).reshape(
        model.v.shape
    )
    conductance, source = mechanism.i_affine_terms(voltage)
    current, paired_conductance = mechanism.i_with_g(voltage)
    expected = (
        ThreeCurrentRateHH.ina(mechanism, voltage)
        + ThreeCurrentRateHH.ik(mechanism, voltage)
        + ThreeCurrentRateHH.il(mechanism, voltage)
    )
    assert conductance.shape == source.shape == voltage.shape
    torch.testing.assert_close(current, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(current, conductance * voltage - source, rtol=0, atol=0)
    torch.testing.assert_close(paired_conductance, conductance, rtol=0, atol=0)


def test_subclasses_override_total_hook():
    mechanisms = [
        next(iter(model_with(cls, 23.0, torch.float64, "sc").mech.mechanisms.values()))
        for cls in (hh, ScaledHH)
    ]
    reference, scaled = mechanisms
    voltage = torch.tensor([[-70.0, -55.0, 10.0]], dtype=torch.float64).expand(2, 3)
    torch.testing.assert_close(scaled.i(voltage), 2.0 * reference.i(voltage))


@pytest.mark.parametrize("assembly", ["off", "auto", "validate"])
def test_hh_assembly_selection_and_membrane_current_reporting(assembly):
    for imem in (False, True):
        model = model_with(hh, 23.0, torch.float64, "sc", assembly=assembly, imem=imem)
        with dn.ctx(JIT=0, MEMBRANE_ASSEMBLY=assembly):
            g, j, current, _, _ = model.integrator._membrane_frame(model.v)
        assert (current is None) == (assembly != "off" and not imem)
        assert torch.isfinite(g).all() and torch.isfinite(j).all()


def test_context_override_switches_assembly_on_an_existing_model():
    model = model_with(hh, 23.0, torch.float64, "sc", imem=False)
    frames = []
    for assembly in ("auto", "off", "validate", "auto"):
        with dn.ctx(JIT=0, MEMBRANE_ASSEMBLY=assembly):
            frame = model.integrator._membrane_frame(model.v)
        assert (frame[2] is None) == (assembly != "off")
        frames.append(frame)
    for frame in frames[1:]:
        torch.testing.assert_close(frame[0], frames[0][0], rtol=0, atol=0)
        torch.testing.assert_close(frame[1], frames[0][1], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("field", CHANNEL)
def test_direct_hook_implicit_broadcasting_and_nonempty_parameter_vmap(
    dtype, batch, field
):
    # GLOBAL coefficients are scalar; initialized gates have the local voltage
    # shape. Empty hidden lanes are covered at Dendra's functional execution
    # boundary, which owns the PyTorch scalar-batching compatibility handling.
    values = dict(
        zip(
            CHANNEL + GATES, (0.12, 0.036, 0.0003, 50.0, -77.0, -54.3, 0.19, 0.61, 0.32)
        )
    )
    shared = {name: torch.tensor(value, dtype=dtype) for name, value in values.items()}
    for gate in GATES:
        shared[gate] = shared[gate].expand(2, 3)
    voltage = torch.linspace(-70.0, 10.0, 6, dtype=dtype).reshape(2, 3)
    mapped = torch.linspace(0.95, 1.05, batch, dtype=dtype) * values[field]

    def coefficients(value):
        fields = dict(shared)
        fields[field] = value
        return torch.stack(hh.i_affine_terms(SimpleNamespace(**fields), voltage))

    actual = torch.vmap(coefficients)(mapped)
    assert actual.shape == (batch, 2, 2, 3)
    expected = torch.stack([coefficients(value) for value in mapped])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.func.grad(lambda value: coefficients(value).sum())
    actual_gradient = torch.vmap(gradient)(mapped)
    expected_gradient = torch.stack([gradient(value) for value in mapped])
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
