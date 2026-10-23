"""Numerical and ownership contracts for the shared explicit Euler core."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.integrators.explicit import _euler, _eulerv1
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion

from .test_integrator_explicit_implicit import CableModel, LinearMechanism


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1,), (2, 4), (2, 3, 4)])
def test_euler_pure_workspace_preserves_native_cylindrical_coefficients(dtype, shape):
    values = tuple(
        torch.linspace(lo, hi, shape[-1], dtype=dtype)
        .expand(shape)
        .clone()
        .requires_grad_()
        for lo, hi in ((2.0, 3.0), (1.0, 2.0), (0.8, 1.2), (80.0, 110.0))
    )
    dx, diam, cm, rhoa = values
    actual = _euler._prepare_workspace(0.01, dx=dx, diam=diam, cm=cm, rhoa=rhoa)
    length_cm = dx / 10000.0
    area = torch.pi * (diam / 10000.0) * length_cm
    capacitance = cm / 1000.0 * area
    resistance = rhoa * length_cm / (torch.pi * (diam / 20000.0) ** 2)
    expected = {
        "cm_inv": 1.0 / capacitance,
        "ra_inv": 1.0 / resistance,
        "cm_c": capacitance,
        "area_c": area,
        "ve_zero": torch.zeros_like(capacitance),
    }
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0.0, atol=0.0)
        assert value.shape == shape
        assert value.dtype == dtype
    loss = sum(value.sum() for value in actual.values())
    gradients = torch.autograd.grad(loss, values)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0


@pytest.mark.parametrize("factory", [dn.euler, dn.rk1, dn.eulerv1])
def test_euler_native_initialize_uses_shared_workspace_and_raw_geometry(factory):
    model = CableModel(shape=(2, 4), diam=torch.tensor([2.0, 3.0, 4.0, 5.0]))
    model.cm_scale.fill_(7.0)
    model.rhoa_scale.fill_(3.0)
    model.area_scale.fill_(5.0)
    integrator = factory()(model, LinearMechanism()).to(dtype=torch.float64)
    integrator.initialize(model, 0.01)
    expected = integrator._prepare_workspace(
        0.01, dx=model.dx, diam=model.diam, cm=model.cm, rhoa=model.rhoa
    )
    assert integrator._functional_spec().operator_kind == "scalar_explicit_path"
    for name, value in expected.items():
        torch.testing.assert_close(getattr(integrator, name), value, rtol=0.0, atol=0.0)


class _PhasedCurrent(LinearMechanism):
    def __init__(self):
        super().__init__(g=0.017, e=-63.0, shift=1000.0)
        self.phases = []

    def update_v(self, v):
        raise AssertionError("Euler does not apply VoltageProcess hooks")

    def advance_pre_current(self, v, dt, temp):
        self.phases.append(("pre", v.detach().clone()))

    def advance_post_current(self, v, dt, temp, frame):
        self.phases.append(("post", v.detach().clone(), frame))

    def iexp(self, v):
        self.phases.append(("explicit",))
        return self._current(v)

    def capture_ion_current_frame(self):
        return (torch.tensor(0.25, dtype=torch.float64),)

    def _evaluate_explicit_current_frame(self, v):
        return self.iexp(v), self.capture_ion_current_frame()

    def i(self, v):
        raise AssertionError("Euler does not evaluate implicit conductance hooks")


@pytest.mark.parametrize("cls", [_euler, _eulerv1])
@pytest.mark.parametrize("compartments", [1, 4])
@pytest.mark.parametrize("training", [False, True])
def test_euler_functional_adapter_preserves_native_phases_stencil_and_gradients(
    cls, compartments, training
):
    model = CableModel(shape=(2, compartments), v_init=-65.0)
    native = cls(model, _PhasedCurrent(), imem=True).to(dtype=torch.float64)
    native.initialize(model, 0.00001)
    native.ssd.weight = torch.nn.Parameter(
        torch.tensor([[[1.0, -2.0, 1.1], [0.9, -2.0, 1.0]]], dtype=torch.float64)
    )
    native.train(training)
    pure = copy.deepcopy(native)
    v = torch.linspace(-66.0, -64.0, 2 * compartments, dtype=torch.float64).reshape(
        2, compartments
    )
    ve = torch.linspace(-0.1, 0.1, compartments, dtype=torch.float64)
    intra = torch.full((compartments,), 1.0e-10, dtype=torch.float64)
    native_inputs = tuple(value.clone().requires_grad_() for value in (v, ve, intra))
    pure_inputs = tuple(value.clone().requires_grad_() for value in (v, ve, intra))
    native_v, native_ve, native_intra = native_inputs
    expected = native._step(
        native_v,
        native_ve,
        native.area_c,
        0.00001,
        37.0,
        native.cm_c,
        native_intra,
    )
    pure_v, pure_ve, pure_intra = pure_inputs
    actual = pure._functional_step(
        pure_v,
        0.00001,
        37.0,
        pure_ve,
        pure_intra,
        call_local_currents=True,
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0.0, atol=0.0)
    native_gradients = torch.autograd.grad(
        expected[0].square().sum() + expected[1].sum(),
        (*native_inputs, native.ssd.weight),
    )
    pure_gradients = torch.autograd.grad(
        actual[0].square().sum() + actual[1].sum(),
        (*pure_inputs, pure.ssd.weight),
    )
    for actual_gradient, expected_gradient in zip(
        pure_gradients, native_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=0.0, atol=0.0
        )
    for owner in (native, pure):
        assert tuple(phase[0] for phase in owner.mech.phases) == (
            "pre",
            "explicit",
            "post",
        )
        torch.testing.assert_close(owner.mech.phases[0][1], v)
        torch.testing.assert_close(owner.mech.phases[-1][1], v)
        torch.testing.assert_close(
            owner.mech.phases[-1][2][0], torch.tensor(0.25, dtype=torch.float64)
        )


class _ExplicitCubic(Mechanism):
    Mechanism.RANGE(g=0.25)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.EXPLICIT("i")

    def i(self, v):
        return self.g * v**3

    def i_with_conductance(self, v):
        raise AssertionError("explicit current evaluation requested a conductance")


class _ExplicitIon(Mechanism):
    Mechanism.RANGE(g=0.125)
    Mechanism.USEION("na", write=["ina"])
    Mechanism.EXPLICIT("ina")

    def ina(self, v):
        return self.g * (v + 3.0)

    def ina_with_conductance(self, v):
        raise AssertionError("explicit current evaluation requested a conductance")


def _explicit_handler():
    shape = (2, 4)
    dtype = torch.float64
    field = torch.ones(shape, dtype=dtype)
    mechanisms = {}
    for cls, name, key in (
        (_ExplicitCubic, "dense", None),
        (_ExplicitCubic, "restricted", torch.tensor([1, 1, 6])),
        (_ExplicitIon, "ion", torch.tensor([0, 4, 7])),
    ):
        local_shape = shape if key is None else (key.numel(),)
        mechanisms[name] = cls(
            name, field * 34.0, field, local_shape, local_shape, key=key, dtype=dtype
        )
    ion = Ion("na", shape, einit=0, eadvance=0).to(dtype=dtype)
    mechanisms["ion"].register_ion(ion)
    handler = MechanismHandler(
        field * 34.0,
        field,
        mechanisms,
        ions={"na": ion},
        currents={"i": {"dense": ["i"], "restricted": ["i"]}, "ina": {"ion": ["ina"]}},
    ).to(dtype=dtype)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(shape, dtype=dtype))
    return handler


def test_explicit_current_frame_matches_native_duplicates_and_preserves_scratch():
    handler = _explicit_handler()
    voltage = torch.linspace(-2.0, 2.0, 8, dtype=torch.float64).reshape(2, 4)
    original = tuple(
        (id(value), value._version, value.clone()) for value in handler._buf_i
    )
    actual, frame = handler._evaluate_explicit_current_frame(voltage)
    for value, (identity, version, saved) in zip(handler._buf_i, original, strict=True):
        assert id(value) == identity and value._version == version
        torch.testing.assert_close(value, saved, rtol=0.0, atol=0.0)
    expected = handler.iexp(voltage)
    native_frame = handler.capture_ion_current_frame()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(frame[0], native_frame[0], rtol=0.0, atol=0.0)
    ionic = torch.zeros_like(voltage).flatten()
    keys = torch.tensor([0, 4, 7])
    ionic = ionic.index_add(0, keys, 0.125 * (voltage.flatten()[keys] + 3.0))
    torch.testing.assert_close(frame[0], ionic.reshape_as(voltage), rtol=0.0, atol=0.0)


class _ExplicitFrame(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.handler = _explicit_handler()

    def forward(self, voltage):
        return self.handler._evaluate_explicit_current_frame(voltage)


@pytest.mark.parametrize("lanes", [0, 3])
def test_explicit_current_frames_support_parameter_only_vmap_with_duplicate_support(
    lanes,
):
    module = _ExplicitFrame()
    voltage = torch.linspace(-2.0, 2.0, 8, dtype=torch.float64).reshape(2, 4)
    original = module.handler.mechanisms["dense"].g
    gains = original.expand((lanes,) + original.shape).clone().requires_grad_()

    def evaluate(gain):
        return torch.func.functional_call(
            module, {"handler.mechanisms.dense.g": gain}, (voltage,)
        )

    actual, frame = torch.vmap(evaluate)(gains)
    assert actual.shape == (lanes, 2, 4)
    assert frame[0].shape == (2, 4) or frame[0].shape == (lanes, 2, 4)
    if lanes:
        references = tuple(evaluate(gain)[0] for gain in gains)
        torch.testing.assert_close(actual, torch.stack(references), rtol=0.0, atol=0.0)
    gradient = torch.autograd.grad(actual.sum(), gains)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.shape == gains.shape


def test_shared_euler_workspace_and_explicit_frame_compile_with_gradients():
    module = _ExplicitFrame()
    shape = (2, 4)

    def evaluate(voltage, diam, dx, cm, rhoa):
        workspace = _euler._prepare_workspace(0.01, dx=dx, diam=diam, cm=cm, rhoa=rhoa)
        current, frame = module(voltage)
        return torch.cat(
            (
                current.flatten(),
                frame[0].flatten(),
                *(value.flatten() for value in workspace.values()),
            )
        )

    values = (
        torch.linspace(-2.0, 2.0, 8, dtype=torch.float64).reshape(shape),
        torch.linspace(1.0, 2.0, 8, dtype=torch.float64).reshape(shape),
        torch.linspace(2.0, 3.0, 8, dtype=torch.float64).reshape(shape),
        torch.linspace(0.8, 1.2, 8, dtype=torch.float64).reshape(shape),
        torch.linspace(80.0, 110.0, 8, dtype=torch.float64).reshape(shape),
    )
    eager_values = tuple(value.clone().requires_grad_() for value in values)
    compiled_values = tuple(value.clone().requires_grad_() for value in values)
    expected = evaluate(*eager_values)
    actual = torch.compile(evaluate, backend="aot_eager", fullgraph=True)(
        *compiled_values
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    expected_gradients = torch.autograd.grad(expected.sum(), eager_values)
    actual_gradients = torch.autograd.grad(actual.sum(), compiled_values)
    for gradient, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, reference, rtol=2.0e-14, atol=2.0e-14)
