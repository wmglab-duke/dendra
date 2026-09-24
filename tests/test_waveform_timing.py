"""Hard oscillator edges retain the intended timing surrogate derivatives."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from dendra.units import nA

DTYPE = torch.float64
TIMING_NAMES = ("delay", "off", "off_after")

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _tensor(value):
    return torch.tensor(value, dtype=DTYPE)


def _wave(kind, *, components=False, constraint="absolute", amp_scale=1.0):
    # Binary-exact timing values make the minimum's tie intentional.
    if components:
        amplitude, frequency, phase = [1.2, 0.8, 1.1], [0.6, 0.7, 0.8], [0.2, 0.3, -0.1]
        delay = [0.125, 0.125, 0.125]
        off, off_after = [0.625, 0.875, 0.625], [0.75, 0.5, 0.5]
    else:
        amplitude, frequency, phase, delay = 1.2, 0.7, 0.2, 0.125
        off, off_after = {
            "absolute": (0.625, 0.75),
            "relative": (0.875, 0.5),
            "tie": (0.625, 0.5),
        }[constraint]
    with dn.ctx(JIT=0, REQUIRE_GRAD=0):
        wave = getattr(dn, kind)(
            amp=_tensor(amplitude) * amp_scale,
            freq=_tensor(frequency),
            phase=_tensor(phase),
            delay=_tensor(delay),
            off=_tensor(off),
            off_after=_tensor(off_after),
            tau=_tensor(0.025),
        )
    wave.requires_grad_(False)
    for name in TIMING_NAMES:
        getattr(wave, name).requires_grad_(True)
    assert all(getattr(wave, name).requires_grad for name in TIMING_NAMES)
    return wave


def _times():
    return _tensor([0.075, 0.1, 0.125, 0.15, 0.25, 0.4, 0.575, 0.6, 0.625, 0.65, 0.7])


def _weights(times):
    return torch.linspace(0.3, 1.3, times.numel(), dtype=DTYPE)


def _component_reference(wave, kind, times):
    """Differentiate the STE formula analytically, without autograd or Dendra gates."""
    amplitude = wave.amp.detach().unsqueeze(-1)
    frequency = wave.freq.detach().unsqueeze(-1)
    delay = wave.delay.detach().unsqueeze(-1)
    end = torch.minimum(
        wave.off.detach(), wave.off_after.detach() + wave.delay.detach()
    )
    end = end.unsqueeze(-1)
    tau = wave.tau.detach()
    argument = 2 * torch.pi * frequency * (
        times - delay
    ) + wave.phase.detach().unsqueeze(-1)
    oscillation = torch.sin(argument) if kind == "sin" else torch.cos(argument)
    phase_delay = (
        -2
        * torch.pi
        * frequency
        * (torch.cos(argument) if kind == "sin" else -torch.sin(argument))
    )
    hard = ((times >= delay) & (times < end)).to(DTYPE)
    start_soft = torch.sigmoid((times - delay) / tau)
    end_soft = torch.sigmoid((end - times) / tau)
    gate_start = -start_soft * (1 - start_soft) * end_soft / tau
    gate_end = start_soft * end_soft * (1 - end_soft) / tau
    # The three components have absolute, relative, and tied end constraints.
    absolute_share = _tensor([1.0, 0.0, 0.5]).unsqueeze(-1)
    relative_share = _tensor([0.0, 1.0, 0.5]).unsqueeze(-1)
    local_derivatives = torch.stack(
        (
            amplitude
            * (
                oscillation * (gate_start + relative_share * gate_end)
                + hard * phase_delay
            ),
            amplitude * oscillation * gate_end * absolute_share,
            amplitude * oscillation * gate_end * relative_share,
        )
    )
    forward = (amplitude * hard * oscillation).sum(0)
    gradients = (local_derivatives * _weights(times)).sum(-1)
    return forward, gradients


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_multitone_timing_matches_analytic_ste_with_active_inactive_and_tied_ends(kind):
    wave, times = _wave(kind, components=True), _times()
    expected, expected_gradients = _component_reference(wave, kind, times)
    actual = wave(times)
    gradients = torch.stack(
        torch.autograd.grad(
            (actual * _weights(times)).sum(),
            tuple(getattr(wave, name) for name in TIMING_NAMES),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Neither sigmoid tail may alter the abrupt forward waveform.
    assert torch.count_nonzero(actual[times < 0.125]) == 0
    assert torch.count_nonzero(actual[times >= 0.625]) == 0
    assert torch.isfinite(gradients).all()
    torch.testing.assert_close(gradients, expected_gradients, rtol=2e-12, atol=2e-12)
    assert gradients[1, 1] == 0  # inactive absolute off
    assert gradients[2, 0] == 0  # inactive relative off_after
    assert gradients[1, 2] != 0
    torch.testing.assert_close(gradients[1, 2], gradients[2, 2], rtol=0, atol=0)


def _run_timing_case(kind, constraint):
    wave = _wave(kind, constraint=constraint, amp_scale=0.2 * nA)
    with dn.ctx(JIT=0, REQUIRE_GRAD=0):
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
        model.run(tstop=1.0, dt=0.025)
    timing = tuple(getattr(wave, name) for name in TIMING_NAMES)
    assert all(parameter.requires_grad for parameter in timing)
    gradients = torch.stack(torch.autograd.grad(model.v.square().mean(), timing))
    assert torch.isfinite(gradients).all()
    return model.v.detach(), gradients.detach()


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_ordinary_cable_run_preserves_timing_chain_rule_and_minimum_tie(kind):
    voltage, absolute = _run_timing_case(kind, "absolute")
    relative_voltage, relative = _run_timing_case(kind, "relative")
    tie_voltage, tie = _run_timing_case(kind, "tie")
    # Every constraint choice ends the same physical input at exactly 0.625 ms.
    torch.testing.assert_close(relative_voltage, voltage, rtol=0, atol=0)
    torch.testing.assert_close(tie_voltage, voltage, rtol=0, atol=0)
    assert absolute[0] != 0 and absolute[1] != 0
    assert absolute[2] == 0 and relative[1] == 0
    torch.testing.assert_close(relative[2], absolute[1], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        relative[0], absolute[0] + absolute[1], rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(
        tie[1:], absolute[1].expand(2) / 2, rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(
        tie[0], absolute[0] + absolute[1] / 2, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_timing_jacobians_and_compiled_loss_preserve_analytic_surrogate(kind):
    wave, times = _wave(kind, components=True), _times()
    _expected_values, expected_gradient = _component_reference(wave, kind, times)
    timing = torch.stack(tuple(getattr(wave, name).detach() for name in TIMING_NAMES))

    def loss(values):
        replacements = dict(zip(TIMING_NAMES, values.unbind(0), strict=True))
        signal = torch.func.functional_call(wave, replacements, (times,))
        return (signal * _weights(times)).sum()

    reverse = torch.func.jacrev(loss)(timing)
    forward = torch.func.jacfwd(loss)(timing)
    torch.testing.assert_close(reverse, expected_gradient, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(forward, expected_gradient, rtol=2e-12, atol=2e-12)
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    timing.requires_grad_()
    with torch_compiler_warning_context():
        actual_loss = compiled(timing)
        actual_gradient = torch.autograd.grad(actual_loss, timing)[0]
    torch.testing.assert_close(actual_loss, loss(timing), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-12, atol=2e-12
    )


def test_repeat_outer_timing_gradients_agree_in_ordinary_and_functional_simulations():
    def build():
        with dn.ctx(JIT=0, REQUIRE_GRAD=0):
            # A constant child has no timing derivatives of its own. Any delay
            # or off gradient therefore comes from the repaired outer gate.
            wave = dn.constant(value=_tensor(0.2 * nA)).repeat(
                _tensor(2.0),
                delay=_tensor(0.125),
                off=_tensor(0.625),
                tau=_tensor(0.025),
            )
            wave.requires_grad_(False)
            wave.delay.requires_grad_(True)
            wave.off.requires_grad_(True)
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
        return model, wave

    # An exact binary timestep keeps both scheduling interfaces on precisely
    # the same clock grid, including the abrupt start and stop boundaries.
    dt, steps = 0.03125, 32
    imperative, imperative_wave = build()
    source, source_wave = build()
    functional, tensors = dn.func.make_functional(source, dt=dt)
    parameters = dict(tensors.parameters)
    timing_names = []
    for name in ("delay", "off"):
        matches = [
            key
            for key in parameters
            if key.startswith("stimulation.intra.") and key.endswith(f".{name}")
        ]
        assert len(matches) == 1
        key = matches[0]
        parameters[key] = parameters[key].detach().clone().requires_grad_()
        torch.testing.assert_close(
            parameters[key], getattr(source_wave, name), rtol=0, atol=0
        )
        timing_names.append(key)
    prepared = functional.prepare(parameters, tensors.constants)
    final, _ = dn.func.run(
        functional.bind(parameters, prepared), tensors.state, steps=steps
    )
    with dn.ctx(JIT=0, REQUIRE_GRAD=0):
        imperative.run(tstop=steps * dt, dt=dt)

    torch.testing.assert_close(
        final["integrator"]["v"], imperative.v, rtol=2e-12, atol=2e-12
    )
    ordinary_gradients = torch.stack(
        torch.autograd.grad(
            imperative.v.square().mean(), (imperative_wave.delay, imperative_wave.off)
        )
    )
    functional_gradients = torch.stack(
        torch.autograd.grad(
            final["integrator"]["v"].square().mean(),
            tuple(parameters[key] for key in timing_names),
        )
    )
    assert torch.isfinite(ordinary_gradients).all()
    assert torch.count_nonzero(ordinary_gradients) == 2
    assert torch.isfinite(functional_gradients).all()
    torch.testing.assert_close(
        functional_gradients, ordinary_gradients, rtol=2e-10, atol=2e-12
    )
    torch.testing.assert_close(
        source.v, torch.full_like(source.v, -65.0), rtol=0, atol=0
    )
    assert source.t.item() == 0
