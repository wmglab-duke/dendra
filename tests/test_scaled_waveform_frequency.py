"""Ordinary waveforms and simulations accept frequency parametrization."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from dendra.units import nA

DTYPE = torch.float64
T_REF = 3.5

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class FrequencyFromCycles(torch.nn.Module):
    """Store cycles over a fixed interval; expose the physical frequency."""

    def __init__(self, t_ref):
        super().__init__()
        self.t_ref = float(t_ref)
        if not math.isfinite(self.t_ref) or self.t_ref <= 0:
            raise ValueError("t_ref must be finite and positive")

    def forward(self, cycles):
        return cycles / self.t_ref

    def right_inverse(self, frequency):
        return frequency * self.t_ref


def _tensor(value):
    return torch.tensor(value, dtype=DTYPE)


def _wave(kind, *, batched=False, amp_scale=1.0):
    frequency = _tensor([[0.7, 1.1], [0.9, 1.3]] if batched else 0.7)
    amplitude = _tensor([[1.0, 0.5], [0.7, 0.2]] if batched else 1.0)
    phase = _tensor([[0.2, -0.1], [0.1, 0.3]] if batched else 0.2)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        return getattr(dn, kind)(
            amp=amplitude * amp_scale,
            freq=frequency,
            phase=phase,
            delay=_tensor(0.0),
            off=_tensor(torch.inf),
            off_after=_tensor(torch.inf),
            tau=_tensor(0.01),
        )


def _scale(wave):
    register_parametrization(wave, "freq", FrequencyFromCycles(T_REF))
    return wave.parametrizations.freq.original


def _loss(values):
    weights = torch.linspace(0.5, 1.5, values.numel(), dtype=DTYPE).reshape_as(values)
    return (weights * values.square()).mean()


@pytest.mark.parametrize("kind", ["sin", "cos"])
@pytest.mark.parametrize("batched", [False, True], ids=["scalar", "batched-multitone"])
def test_frequency_parametrization_preserves_waveform_and_gradient_chain(kind, batched):
    physical = _wave(kind, batched=batched)
    scaled = _wave(kind, batched=batched)
    times = torch.linspace(-0.05, 0.7, 23, dtype=DTYPE)
    initial_frequency = scaled.freq.detach().clone()
    before = scaled(times).detach().clone()
    cycles = _scale(scaled)

    torch.testing.assert_close(cycles, initial_frequency * T_REF, rtol=0, atol=0)
    assert any(parameter is cycles for parameter in scaled.parameters())
    assert cycles.is_leaf and cycles.requires_grad
    actual, expected = scaled(times), physical(times)
    torch.testing.assert_close(actual, before, rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-14)
    assert actual.shape == ((2, times.numel()) if batched else times.shape)

    # Include all oscillator parameters: infinite default off times must not
    # contaminate otherwise finite gradients through the shared gate.
    scaled_parameters = dict(scaled.named_parameters())
    physical_parameters = dict(physical.named_parameters())
    scaled_gradients = dict(
        zip(
            scaled_parameters,
            torch.autograd.grad(_loss(actual), tuple(scaled_parameters.values())),
            strict=True,
        )
    )
    physical_gradients = dict(
        zip(
            physical_parameters,
            torch.autograd.grad(_loss(expected), tuple(physical_parameters.values())),
            strict=True,
        )
    )
    for gradient in (*scaled_gradients.values(), *physical_gradients.values()):
        assert torch.isfinite(gradient).all()
    frequency_gradient = physical_gradients["freq"]
    assert torch.count_nonzero(frequency_gradient) == frequency_gradient.numel()
    torch.testing.assert_close(
        scaled_gradients["parametrizations.freq.original"],
        frequency_gradient / T_REF,
        rtol=2e-12,
        atol=2e-14,
    )


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_frequency_parametrization_supports_repeated_optimizer_updates(kind):
    scaled = _wave(kind, batched=True)
    physical = _wave(kind, batched=True)
    cycles = _scale(scaled)
    optimizer = torch.optim.Adam([cycles], lr=0.01)
    times = torch.linspace(0.0, 0.7, 19, dtype=DTYPE)

    for _ in range(3):
        previous = cycles.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        _loss(scaled(times)).backward()
        assert cycles.grad is not None and torch.isfinite(cycles.grad).all()
        assert torch.count_nonzero(cycles.grad) == cycles.numel()
        optimizer.step()
        assert not torch.equal(cycles, previous)
        assert torch.isfinite(cycles).all()
        with torch.no_grad():
            physical.freq.copy_(cycles / T_REF)
        torch.testing.assert_close(
            scaled(times), physical(times), rtol=2e-13, atol=2e-14
        )


def _cable(wave):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=_tensor(1e-4), e=_tensor(-65.0))
        model[:, 2].inject(wave)
        model.initialize()
        model.train()
    return model


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_injected_scaled_frequency_matches_ordinary_cable_run_gradient(kind):
    physical = _wave(kind, amp_scale=0.2 * nA)
    scaled = _wave(kind, amp_scale=0.2 * nA)
    cycles = _scale(scaled)
    reference, actual = _cable(physical), _cable(scaled)

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        reference.run(tstop=0.08, dt=0.01)
        actual.run(tstop=0.08, dt=0.01)
    torch.testing.assert_close(actual.v, reference.v, rtol=2e-12, atol=2e-12)
    assert not torch.equal(actual.v, torch.full_like(actual.v, -65.0))
    expected_gradient = torch.autograd.grad(reference.v.square().mean(), physical.freq)[
        0
    ]
    actual_gradient = torch.autograd.grad(actual.v.square().mean(), cycles)[0]
    assert torch.isfinite(expected_gradient).all() and expected_gradient.abs() > 0
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(
        actual_gradient, expected_gradient / T_REF, rtol=2e-10, atol=2e-12
    )


def test_compiled_scaled_waveform_preserves_direct_eager_gradients():
    wave = _wave("sin", batched=True)
    cycles = _scale(wave)
    times = torch.linspace(0.0, 0.7, 19, dtype=DTYPE)
    expected = wave(times)
    expected_gradient = torch.autograd.grad(_loss(expected), cycles)[0]
    compiled = torch.compile(wave, backend="aot_eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(times)
        actual_gradient = torch.autograd.grad(_loss(actual), cycles)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-14)
    assert torch.isfinite(actual_gradient).all()
    assert torch.count_nonzero(actual_gradient) == cycles.numel()
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-12, atol=2e-14
    )
