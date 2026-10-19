"""True rate-form HH against an independent pre-change inf/tau fixture."""

from __future__ import annotations

import inspect

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism as M
from dendra.models.mechanisms import State as S
from dendra.models.mechanisms.ops import exp, vtrap
from dendra.models.mod.HH import hh, mhn


class LegacyMhn(S):
    """Frozen pre-rate-form state declaration and authored arithmetic."""

    S.STATE("m", "h", "n")
    S.DERIVATIVE("m' = (minf-m)/mtau", "h' = (hinf-h)/htau", "n' = (ninf-n)/ntau")
    S.ASSIGNED("minf", "mtau", "hinf", "htau", "ninf", "ntau")
    S.DERIVED_BUFFER("q10")
    S.GLOBAL(am1=0.1, am2=4.0, ah1=0.07, ah2=1.0, an1=0.01, an2=0.125)

    def derive_buffers(self):
        return {"q10": 3.0 ** ((self.celsius - 6.3) / 10.0)}

    def assigned_values(self, v, values):
        del values
        alpha_m = self.am1 * vtrap(-(v + 40), 10)
        beta_m = self.am2 * exp(-(v + 65) / 18)
        total_m = alpha_m + beta_m
        alpha_h = self.ah1 * exp(-(v + 65) / 20)
        beta_h = self.ah2 / (exp(-(v + 35) / 10) + 1)
        total_h = alpha_h + beta_h
        alpha_n = self.an1 * vtrap(-(v + 55), 10)
        beta_n = self.an2 * exp(-(v + 65) / 80)
        total_n = alpha_n + beta_n
        return {
            "minf": alpha_m / total_m,
            "mtau": 1 / (self.q10 * total_m),
            "hinf": alpha_h / total_h,
            "htau": 1 / (self.q10 * total_h),
            "ninf": alpha_n / total_n,
            "ntau": 1 / (self.q10 * total_n),
        }

    def state_defaults(self, v, values):
        assigned = self.assigned_values(v, values)
        return {gate: assigned[gate + "inf"] for gate in ("m", "h", "n")}


class LegacyHH(M):
    M.STATE_BUNDLE(LegacyMhn)
    M.GLOBAL(gnabar=0.12, gkbar=0.036, gl=0.0003, ena=50.0, ek=-77.0, el=-54.3)
    M.NONSPECIFIC_CURRENT("il", "ina", "ik")

    def il(self, v):
        return self.gl * (v - self.el)

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)

    def ik(self, v):
        return self.gkbar * self.n**4 * (v - self.ek)


GATES = ("m", "h", "n")
KINETIC = ("am1", "am2", "ah1", "ah2", "an1", "an2")
CHANNEL = ("gnabar", "gkbar", "gl", "ena", "ek", "el")


def gate_state(cls, temperature, dtype, shape=(7,)):
    with dn.ctx(JIT=0):
        state = cls(
            torch.tensor(temperature, dtype=dtype),
            torch.ones(shape, dtype=dtype),
            None,
            shape,
            shape,
        ).to(dtype=dtype)
    state.q10 = state.derive_buffers()["q10"]
    return state


def gate_voltages(dtype):
    # Exact removable singularities plus both neighboring directions.
    return torch.tensor(
        [-80.0, -55.0001, -55.0, -54.9999, -40.0001, -40.0, -39.9999], dtype=dtype
    )


def close(actual, expected, dtype, *, gradient=False):
    if dtype == torch.float64:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-10 if gradient else 2e-12,
            atol=2e-11 if gradient else 2e-12,
        )
    else:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=3e-4 if gradient else 5e-6,
            atol=2e-5 if gradient else 3e-5,
        )


def test_rate_declarations_have_no_compatibility_state_or_override():
    assert mhn._assigned == ("m_a", "m_b", "h_a", "h_b", "n_a", "n_b")
    assert mhn._derivative == {
        "m' = m_a - m_b * m",
        "h' = h_a - h_b * h",
        "n' = n_a - n_b * n",
    }
    assert "advance" not in mhn.__dict__
    source = inspect.getsource(mhn.assigned_values)
    assert all(
        name not in source for name in ("minf", "mtau", "hinf", "htau", "ninf", "ntau")
    )
    state = gate_state(mhn, 23.0, torch.float64)
    solver_source = state._solve.__source__
    for gate in GATES:
        # CNEXP delegates the authored affine rate to its shared exact helper.
        assert (
            f"_{gate} = _linear_exact_step({gate}, -{gate} * {gate}_b + {gate}_a, "
            f"{gate}_a, -{gate}_b, dt)"
        ) in solver_source
        assert f"{gate}tau" not in solver_source


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [6.3, 23.0, 36.0])
def test_coefficients_initial_equilibrium_and_temperature_scaling(dtype, temperature):
    new, legacy = (gate_state(cls, temperature, dtype) for cls in (mhn, LegacyMhn))
    voltage = gate_voltages(dtype)
    actual, expected = (
        new.assigned_values(voltage, {}),
        legacy.assigned_values(voltage, {}),
    )
    initial, legacy_initial = (
        new.state_defaults(voltage, {}),
        legacy.state_defaults(voltage, {}),
    )
    for gate in GATES:
        assert torch.isfinite(actual[gate + "_a"]).all()
        assert torch.isfinite(actual[gate + "_b"]).all()
        close(
            actual[gate + "_a"], expected[gate + "inf"] / expected[gate + "tau"], dtype
        )
        close(actual[gate + "_b"], 1 / expected[gate + "tau"], dtype)
        torch.testing.assert_close(initial[gate], legacy_initial[gate], rtol=0, atol=0)
    reference = gate_state(mhn, 6.3, dtype).state_defaults(voltage, {})
    for gate in GATES:
        torch.testing.assert_close(initial[gate], reference[gate], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [6.3, 23.0, 36.0])
def test_multistep_generated_gate_solver_matches_legacy(dtype, temperature):
    new, legacy = (gate_state(cls, temperature, dtype) for cls in (mhn, LegacyMhn))
    voltage = gate_voltages(dtype)
    a = {
        gate: torch.full_like(voltage, value)
        for gate, value in zip(GATES, (0.19, 0.61, 0.32))
    }
    b = dict(a)
    for step in range(64):
        drive = voltage + 0.1 * step
        a, b = new.advance(drive, 0.005, a), legacy.advance(drive, 0.005, b)
        for gate in GATES:
            close(a[gate], b[gate], dtype)


@pytest.mark.parametrize("temperature", [6.3, 23.0, 36.0])
def test_gate_voltage_kinetic_parameter_and_temperature_gradients(temperature):
    dtype = torch.float64
    voltage = gate_voltages(dtype).requires_grad_()
    parameters = torch.tensor(
        [0.1, 4.0, 0.07, 1.0, 0.01, 0.125], dtype=dtype, requires_grad=True
    )
    celsius = torch.tensor(temperature, dtype=dtype, requires_grad=True)
    results = []
    for cls in (mhn, LegacyMhn):
        state = gate_state(cls, temperature, dtype)
        for name, value in zip(KINETIC, parameters.unbind()):
            setattr(state, name, value)
        state.celsius = celsius
        state.q10 = state.derive_buffers()["q10"]
        values = {
            gate: torch.full_like(voltage, value)
            for gate, value in zip(GATES, (0.19, 0.61, 0.32))
        }
        samples = []
        for step in range(16):
            values = state.advance(voltage + 0.1 * step, 0.005, values)
            samples.append(torch.stack([values[gate] for gate in GATES]))
        trace = torch.stack(samples)
        weights = (
            torch.arange(1, trace.numel() + 1, dtype=dtype).reshape(trace.shape)
            / trace.numel()
        )
        gradients = torch.autograd.grad(
            (trace * weights).mean(), (voltage, parameters, celsius), retain_graph=True
        )
        assert all(
            torch.isfinite(g).all() and torch.count_nonzero(g) == g.numel()
            for g in gradients
        )
        results.append((trace, gradients))
    close(results[0][0], results[1][0], dtype)
    for actual, expected in zip(results[0][1], results[1][1]):
        close(actual, expected, dtype, gradient=True)


def model_with(cls, temperature, dtype, kind, *, assembly="auto", imem=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, MEMBRANE_ASSEMBLY=assembly, IMEM=int(imem)):
        if kind == "sc":
            model = dn.SingleCompartment(
                N=2,
                C=3,
                dtype=dtype,
                celsius=temperature,
                integrator=dn.bwd_euler_sc(imem=imem),
            )
        else:
            graph = nx.DiGraph()
            for node in range(3):
                graph.add_node(
                    node,
                    L=10.0,
                    diam=2.0,
                    Ra=100.0,
                    cm=1.0,
                    area=20.0,
                    volume=10.0,
                    volume_i=10.0,
                    volume_o=0.0,
                    x=float(node),
                    y=0.0,
                    z=0.0,
                )
            graph.add_edge(0, 1, R_ohm=8e7)
            graph.add_edge(0, 2, R_ohm=9e7)
            model = dn.Tree.from_graph(
                graph,
                N=2,
                dtype=dtype,
                celsius=temperature,
                integrator=dn.dhs(threads=1, imem=imem),
            )
        model.insert(cls)
        model.initialize()
        model.train()
        model.integrator.initialize(model, 0.005)
    return model


def membrane_trace_and_gradients(
    cls, temperature, dtype, kind, *, assembly="auto", imem=True
):
    model = model_with(cls, temperature, dtype, kind, assembly=assembly, imem=imem)
    mechanism = next(iter(model.mech.mechanisms.values()))
    v0 = (
        torch.tensor([-65.0, -55.0, -40.0], dtype=dtype)
        .expand(model.shape)
        .clone()
        .requires_grad_()
    )
    channels = torch.tensor(
        [0.12, 0.036, 0.0003, 50.0, -77.0, -54.3], dtype=dtype, requires_grad=True
    )
    for name, value in zip(CHANNEL, channels.unbind()):
        setattr(mechanism, name, value)
    # Gates are independent of V0, so the tested initial-voltage derivative
    # does not silently include equilibrium reinitialization.
    for gate, value in zip(GATES, (0.19, 0.61, 0.32)):
        setattr(mechanism, gate, torch.full_like(v0, value))
    stimulus = (
        torch.linspace(0.003, 0.009, 64, dtype=dtype)
        .reshape(64, 1, 1)
        .expand(64, *model.shape)
        .clone()
        .requires_grad_()
    )
    voltage, samples = v0, []
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, MEMBRANE_ASSEMBLY=assembly):
        for step in range(64):
            voltage, _ = model.integrator._step(
                voltage, 0.005, model.celsius, intra=stimulus[step] * model.area
            )
            samples.append(
                torch.stack([voltage, *(getattr(mechanism, gate) for gate in GATES)])
            )
    trace = torch.stack(samples)
    loss = trace[:, 0].square().mean() / 1000.0
    gradients = torch.autograd.grad(loss, (stimulus, v0, channels))
    assert all(
        torch.isfinite(g).all() and torch.count_nonzero(g) == g.numel()
        for g in gradients
    )
    return trace, gradients


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [6.3, 23.0, 36.0])
@pytest.mark.parametrize("kind", ["sc", "dhs"])
def test_full_voltage_gate_traces_and_stimulus_initial_voltage_channel_gradients(
    dtype, temperature, kind
):
    actual, actual_grads = membrane_trace_and_gradients(hh, temperature, dtype, kind)
    expected, expected_grads = membrane_trace_and_gradients(
        LegacyHH, temperature, dtype, kind
    )
    if dtype == torch.float32:
        # Reordered gate arithmetic accumulates sub-microvolt rounding in the
        # coupled membrane trajectory; bound voltage separately from gates.
        torch.testing.assert_close(actual[:, 0], expected[:, 0], rtol=0, atol=3e-4)
        close(actual[:, 1:], expected[:, 1:], dtype)
    else:
        close(actual, expected, dtype)
    for a, b in zip(actual_grads, expected_grads):
        close(a, b, dtype, gradient=True)
