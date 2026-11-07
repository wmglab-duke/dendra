"""Shared Dufort--Frankel coefficients and call-local native kernels."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.integrators.explicit import (
    _dufort_frankel,
    _dufort_frankel_homogeneous,
    _reflect_pad_last,
)
from dendra.models.mod import pas

from .test_integrator_explicit_implicit import CableModel, LinearMechanism


@pytest.mark.parametrize("cls", [_dufort_frankel, _dufort_frankel_homogeneous])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1,), (2, 4), (2, 3, 4)])
def test_dufort_workspace_retains_native_coefficients_and_geometry_gradients(
    cls, dtype, shape
):
    dx, diam, cm, rhoa = values = tuple(
        torch.linspace(lo, hi, shape[-1], dtype=dtype)
        .expand(shape)
        .clone()
        .requires_grad_()
        for lo, hi in ((2.0, 3.0), (1.0, 2.0), (0.8, 1.2), (80.0, 110.0))
    )
    dt = torch.tensor(0.001, dtype=dtype, requires_grad=True)
    actual = cls._prepare_workspace(dt, dx=dx, diam=diam, cm=cm, rhoa=rhoa)
    area = torch.pi * (diam / 10000) * (dx / 10000)
    cap = cm / 1000 * area
    resistance = rhoa * (dx / 10000) / (torch.pi * (diam / 20000) ** 2)
    s1 = 2 * dt / cap
    expected = {
        "area": area,
        "s1": s1,
        "s3": area * s1,
        "ve_zero": torch.zeros_like(cm),
    }
    if cls is _dufort_frankel_homogeneous:
        s2 = torch.zeros_like(s1) if shape[-1] == 1 else s1 / resistance
        expected.update(s2=s2, s4=1 + s2)
    else:
        flat = resistance.reshape(-1, shape[-1])
        padded = _reflect_pad_last(flat, (1, 1))
        left = (1 / (0.5 * (padded[:, :-2] + padded[:, 1:-1]))).reshape(shape)
        right = (1 / (0.5 * (padded[:, 1:-1] + padded[:, 2:]))).reshape(shape)
        if shape[-1] == 1:
            left, right = torch.zeros_like(left), torch.zeros_like(right)
        expected.update(
            c_left=s1 * left, c_right=s1 * right, c_axial=0.5 * (s1 * left + s1 * right)
        )
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert actual[name].shape == shape
        assert actual[name].dtype == dtype
    gradients = torch.autograd.grad(
        sum(value.sum() for value in actual.values()), (*values, dt), allow_unused=True
    )
    for index, gradient in enumerate(gradients):
        if shape[-1] == 1 and index == 3:
            assert gradient is None
            continue
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0


@pytest.mark.parametrize("cls", [_dufort_frankel, _dufort_frankel_homogeneous])
def test_dufort_native_initialization_uses_shared_raw_cylindrical_workspace(cls):
    model = CableModel(shape=(2, 4), diam=torch.tensor([2.0, 3.0, 4.0, 5.0]))
    model.cm_scale.fill_(7)
    model.rhoa_scale.fill_(3)
    model.area_scale.fill_(5)
    integrator = cls(model, LinearMechanism()).to(dtype=torch.float64)
    integrator.initialize(model, 0.01)
    assert integrator._functional_spec().operator_kind == "scalar_dufort_path"
    expected = cls._prepare_workspace(
        0.01, dx=model.dx, diam=model.diam, cm=model.cm, rhoa=model.rhoa
    )
    for name, value in expected.items():
        torch.testing.assert_close(getattr(integrator, name), value, rtol=0, atol=0)
    assert not model._df_history_valid
    model._df_history_valid.fill_(True)
    integrator.initialize(model, 0.01)
    assert model._df_history_valid
    integrator.initialize(model, 0.02)
    assert not model._df_history_valid


class _PhasedCurrent(LinearMechanism):
    def __init__(self):
        super().__init__(g=0.017, e=-63.0, shift=1000.0)
        self.phases = []

    def update_v(self, _v):
        raise AssertionError("Dufort does not apply VoltageProcess hooks")

    def advance_pre_current(self, v, dt, temp):
        self.phases.append(("pre", v.detach().clone()))

    def advance_post_current(self, v, dt, temp, frame):
        self.phases.append(("post", v.detach().clone(), frame))

    def iexp(self, v):
        self.phases.append(("explicit",))
        return self._current(v)

    def idf(self, v, v_prev):
        self.phases.append(("dufort",))
        return self._current(0.5 * v_prev), self._g(v)

    def capture_ion_current_frame(self):
        return (torch.tensor(0.25, dtype=torch.float64),)

    def capture_ion_conductance_frame(self):
        return (torch.tensor(0.017, dtype=torch.float64),)

    def _evaluate_explicit_current_frame(self, v):
        return self.iexp(v), self.capture_ion_current_frame()

    def _evaluate_dufort_current_frame(self, v, v_prev):
        current, conductance = self.idf(v, v_prev)
        return (
            current,
            conductance,
            self.capture_ion_current_frame(),
            self.capture_ion_conductance_frame(),
        )


@pytest.mark.parametrize("kind", ["heterogeneous", "direct", "convolution"])
@pytest.mark.parametrize("compartments", [1, 4])
@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("dynamic_dispatch", [False, True])
def test_dufort_call_local_kernel_preserves_native_phases_values_and_gradients(
    kind, compartments, startup, dynamic_dispatch
):
    model = CableModel(shape=(2, compartments), v_init=-65)
    cls = _dufort_frankel if kind == "heterogeneous" else _dufort_frankel_homogeneous
    kwargs = {} if kind == "heterogeneous" else {"conv": kind == "convolution"}
    native = cls(model, _PhasedCurrent(), beta=0.8, imem=True, **kwargs).to(
        dtype=torch.float64
    )
    native.initialize(model, 0.00001)
    pure = copy.deepcopy(native)
    v = torch.linspace(-66, -64, 2 * compartments, dtype=torch.float64).reshape(
        2, compartments
    )
    v_prev = v - 0.1
    ve = torch.linspace(-0.1, 0.1, compartments, dtype=torch.float64)
    intra = torch.full((compartments,), 1e-10, dtype=torch.float64)

    def run(integrator, inputs, call_local):
        v, vp, ve, intra = inputs
        if call_local:
            valid = torch.tensor(not startup)
            history_dt = torch.tensor(torch.nan, dtype=v.dtype)

            def current_dispatch(v, v_prev, use_history):
                del use_history
                if startup:
                    current, frame = integrator._initial_current_frame(
                        v, call_local_currents=True
                    )
                    return (
                        current,
                        torch.zeros_like(current),
                        frame,
                        tuple(torch.zeros_like(value) for value in frame),
                    )
                return integrator._dufort_current_frame(
                    v, v_prev, call_local_currents=True
                )

            result = integrator._functional_step(
                v,
                0.00001,
                37.0,
                ve,
                intra,
                history=(vp, valid, history_dt),
                use_history=valid if dynamic_dispatch else not startup,
                current_dispatch=current_dispatch if dynamic_dispatch else None,
            )
            assert result[2].dtype == torch.bool
            assert result[2]
            torch.testing.assert_close(result[3], torch.tensor(0.00001, dtype=v.dtype))
            return result[0], result[1], result[4]
        if kind == "heterogeneous":
            args = (
                (v, ve, 0.00001, 37.0, intra)
                if startup
                else (v, vp, ve, 0.00001, 37.0, intra)
            )
        elif startup:
            args = (
                v,
                ve,
                integrator.s1,
                integrator.s2,
                integrator.area,
                0.00001,
                37.0,
                intra,
            )
        else:
            args = (
                v,
                vp,
                ve,
                integrator.s1,
                integrator.s2,
                integrator.s3,
                integrator.s4,
                integrator.area,
                0.00001,
                37.0,
                intra,
            )
        method = (
            "_step_initial"
            if startup
            else "_step_conv"
            if kind == "convolution"
            else "_step"
        )
        return getattr(integrator, method)(*args, call_local_currents=call_local)

    native_inputs = tuple(
        value.clone().requires_grad_() for value in (v, v_prev, ve, intra)
    )
    pure_inputs = tuple(
        value.clone().requires_grad_() for value in (v, v_prev, ve, intra)
    )
    expected = run(native, native_inputs, False)
    actual = run(pure, pure_inputs, True)
    for reference, result in zip(expected, actual, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for inputs, outputs in ((native_inputs, expected), (pure_inputs, actual)):
        loss = outputs[0].square().sum() + outputs[2].square().sum()
        gradients = torch.autograd.grad(loss, inputs, allow_unused=True)
        if inputs is native_inputs:
            native_gradients = gradients
        else:
            for actual_gradient, expected_gradient in zip(
                gradients, native_gradients, strict=True
            ):
                if actual_gradient is None:
                    assert expected_gradient is None
                elif expected_gradient is None:
                    assert not actual_gradient.any()
                else:
                    torch.testing.assert_close(
                        actual_gradient, expected_gradient, rtol=0, atol=0
                    )
    assert [phase[0] for phase in pure.mech.phases] == [
        "pre",
        "explicit" if startup else "dufort",
        "post",
    ]
    torch.testing.assert_close(
        pure.mech.phases[-1][2][0],
        native.mech.phases[-1][2][0].expand_as(pure.mech.phases[-1][2][0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("cls", [_dufort_frankel, _dufort_frankel_homogeneous])
@pytest.mark.parametrize("lanes", [0, 3])
def test_dufort_workspace_supports_hidden_geometry_batches(cls, lanes):
    inputs = tuple(
        torch.full((lanes, 2, 4), value, dtype=torch.float64)
        for value in (2.0, 1.0, 0.8, 80.0)
    )

    def coefficients(dx, diam, cm, rhoa):
        return cls._prepare_workspace(0.001, dx=dx, diam=diam, cm=cm, rhoa=rhoa)

    actual = torch.vmap(coefficients)(*inputs)
    expected = coefficients(*inputs)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("cls", [_dufort_frankel, _dufort_frankel_homogeneous])
def test_dufort_workspace_compiles_with_geometry_and_timestep_gradients(cls):
    values = tuple(
        torch.full((2, 4), value, dtype=torch.float64, requires_grad=True)
        for value in (2.0, 1.0, 0.8, 80.0)
    )
    dt = torch.tensor(0.001, dtype=torch.float64, requires_grad=True)

    def loss(dx, diam, cm, rhoa, dt):
        workspace = cls._prepare_workspace(dt, dx=dx, diam=diam, cm=cm, rhoa=rhoa)
        return sum(value.square().mean() for value in workspace.values())

    expected = loss(*values, dt)
    actual = torch.compile(loss, backend="aot_eager", fullgraph=True)(*values, dt)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual, (*values, dt))
    expected_gradients = torch.autograd.grad(expected, (*values, dt))
    for result, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["heterogeneous", "direct", "convolution"])
@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("lanes", [0, 3])
@pytest.mark.parametrize("dynamic_dispatch", [False, True])
def test_dufort_pure_history_phases_support_hidden_voltage_batches(
    kind, startup, lanes, dynamic_dispatch
):
    model = CableModel(shape=(2, 4), v_init=-65)
    cls = _dufort_frankel if kind == "heterogeneous" else _dufort_frankel_homogeneous
    kwargs = {} if kind == "heterogeneous" else {"conv": kind == "convolution"}
    integrator = cls(model, _PhasedCurrent(), beta=0.8, imem=True, **kwargs).to(
        dtype=torch.float64
    )
    integrator.initialize(model, 0.00001)
    voltage = torch.full((lanes, 2, 4), -65.0, dtype=torch.float64)

    def phase(v):
        def current_dispatch(v, previous, use_history):
            del use_history
            if startup:
                current, frame = integrator._initial_current_frame(
                    v, call_local_currents=True
                )
                return (
                    current,
                    torch.zeros_like(current),
                    frame,
                    tuple(torch.zeros_like(value) for value in frame),
                )
            return integrator._dufort_current_frame(
                v, previous, call_local_currents=True
            )

        return integrator._functional_step(
            v,
            0.00001,
            37.0,
            history=(
                v - 0.1,
                torch.tensor(not startup),
                torch.tensor(0.00001, dtype=v.dtype),
            ),
            use_history=torch.tensor(not startup) if dynamic_dispatch else not startup,
            current_dispatch=current_dispatch if dynamic_dispatch else None,
        )[0]

    actual = torch.vmap(phase)(voltage)
    if lanes:
        expected = torch.stack([phase(row) for row in voltage])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == voltage.shape


@pytest.mark.parametrize("cls", [_dufort_frankel, _dufort_frankel_homogeneous])
def test_dufort_pure_history_configuration_handles_batched_mixed_timesteps(cls):
    valid = torch.tensor([True, True, False])
    history_dt = torch.tensor([0.001, 0.002, torch.nan], dtype=torch.float64)
    timestep = torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64)
    configured, metadata = torch.vmap(cls._prepare_history)(valid, history_dt, timestep)
    torch.testing.assert_close(configured, torch.tensor([True, False, False]))
    torch.testing.assert_close(metadata, timestep)
    compiled = torch.compile(cls._prepare_history, backend="aot_eager", fullgraph=True)
    actual = compiled(valid, history_dt, 0.001)
    torch.testing.assert_close(actual[0], torch.zeros_like(valid))
    torch.testing.assert_close(actual[1], timestep)


@pytest.mark.parametrize("kind", ["heterogeneous", "direct", "convolution"])
def test_dufort_tensor_starter_does_not_read_invalid_nan_history(kind):
    model = CableModel(shape=(2, 4), v_init=-65)
    cls = _dufort_frankel if kind == "heterogeneous" else _dufort_frankel_homogeneous
    kwargs = {} if kind == "heterogeneous" else {"conv": kind == "convolution"}
    integrator = cls(model, _PhasedCurrent(), beta=0.8, imem=True, **kwargs).to(
        dtype=torch.float64
    )
    integrator.initialize(model, 0.00001)
    v = model.v.clone().requires_grad_()
    previous = torch.full_like(v, torch.nan, requires_grad=True)

    def current_dispatch(v, safe_previous, use_history):
        assert torch.isfinite(safe_previous).all()
        assert not use_history
        current, frame = integrator._initial_current_frame(v, call_local_currents=True)
        return (
            current,
            torch.zeros_like(current),
            frame,
            tuple(torch.zeros_like(value) for value in frame),
        )

    result = integrator._functional_step(
        v,
        0.00001,
        37.0,
        history=(previous, torch.tensor(False), v.new_tensor(torch.nan)),
        use_history=torch.tensor(False),
        current_dispatch=current_dispatch,
    )
    assert torch.isfinite(result[0]).all()
    assert torch.isfinite(result[4]).all()
    gradients = torch.autograd.grad(
        result[0].square().sum() + result[4].square().sum(), (v, previous)
    )
    assert torch.isfinite(gradients[0]).all()
    assert torch.equal(gradients[1], torch.zeros_like(previous))
    assert [phase[0] for phase in integrator.mech.phases] == ["pre", "explicit", "post"]


@pytest.mark.parametrize("method", ["df", "dfh"])
@pytest.mark.parametrize("network", [False, True])
def test_dufort_legacy_state_dict_load_preserves_native_history(method, network):
    dt = 0.01
    with dn.ctx(JIT=0, DTYPE=torch.float64, DEVICE="cpu"), torch.no_grad():

        def make():
            population = dn.Population(
                N=1,
                C=4,
                integrator=getattr(dn, method)(imem=True),
                v_init=torch.tensor([-72.0, -66.0, -61.0, -75.0]),
                cm=1.0,
                rhoa=100.0,
                dtype=torch.float64,
            )
            population.diam.fill_(1.0)
            population.dx.fill_(200.0)
            population.insert(pas, g=1.0e-4, e=-62.0)
            owner = dn.Network({"cable": population}) if network else population
            owner.initialize(dt)
            population.integrator._initialize(population, dt, force=True)
            return owner, population

        source_owner, source = make()
        target_owner, target = make()
        for _ in range(2):
            source.integrator.step(source, dt)

        legacy = copy.deepcopy(source_owner.state_dict())
        # The zero extracellular field was unregistered in earlier DF models.
        # Keep that ordinary checkpoint schema while deriving it as a buffer.
        assert not any(name.endswith("integrator.ve_zero") for name in legacy)
        loaded = target_owner.load_state_dict(legacy, strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        assert "ve_zero" in target.integrator._buffers
        assert not target.integrator.initialized
        for name in source.integrator.v_vars:
            torch.testing.assert_close(getattr(target, name), getattr(source, name))

        # Ordinary loading invalidates coefficients; rebuilding at the same dt
        # must preserve the restored two-level history and its next native solve.
        for population in (source, target):
            population.integrator._initialize(population, dt, force=True)
            population.integrator.step(population, dt)
        for name in (*source.integrator.v_vars, "i_membrane"):
            torch.testing.assert_close(
                getattr(target, name), getattr(source, name), rtol=0.0, atol=0.0
            )
