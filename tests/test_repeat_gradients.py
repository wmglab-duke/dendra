"""Repeat windows keep abrupt values and sigmoid timing derivatives."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.stim.waveform import Waveform

DTYPE = torch.float64


class _NonfiniteOutsideWindow(Waveform):
    def __init__(self, output_dtype):
        super().__init__()
        self.output_dtype = output_dtype

    def fn(self, t):
        finite = torch.ones_like(t, dtype=self.output_dtype)
        return torch.where(t < 0, torch.nan, torch.where(t > 0.5, torch.inf, finite))


def _tensor(value):
    return torch.as_tensor(value, dtype=DTYPE)


def _constant_repeat(freq=2.0, *, finite_off=True):
    with dn.ctx(REQUIRE_GRAD=1):
        child = dn.constant(value=_tensor(1.3))
        options = {"off": _tensor(0.8)} if finite_off else {}
        return child.repeat(
            _tensor(freq), delay=_tensor(0.2), tau=_tensor(0.05), **options
        ).to(dtype=DTYPE)


def _parameters(waveform, *, clone=False):
    return {
        name: value.detach().clone().requires_grad_() if clone else value
        for name, value in waveform.named_parameters()
    }


def _reference_gate(t, delay, off, tau, *, finite_off=True):
    # The forward is deliberately hard. Compare derivatives with its stated
    # sigmoid surrogate, rather than finite differences of that hard signal.
    tau = tau.clamp(min=1e-6)
    soft = torch.sigmoid((t - delay) / tau)
    hard = t >= delay
    if finite_off:
        soft = soft * torch.sigmoid((off - t) / tau)
        hard = hard & (t < off)
    return hard.to(t.dtype) + (soft - soft.detach())


def _constant_reference(t, parameters, *, finite_off):
    gate = _reference_gate(
        t,
        parameters["delay"],
        parameters["off"],
        parameters["tau"],
        finite_off=finite_off,
    )
    return (parameters["waveform.value"] * gate).expand(
        (*parameters["freq"].shape, *t.shape)
    )


def _gradients(value, parameters, weights, *, require=()):
    gradients = torch.autograd.grad(
        (value * weights).sum(), tuple(parameters.values()), allow_unused=True
    )
    result = {}
    for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True):
        if name in require:
            assert gradient is not None, f"{name} has no derivative path"
        result[name] = torch.zeros_like(parameter) if gradient is None else gradient
    return result


def _assert_finite_close(actual, expected):
    observed, observed_spec = pytree.tree_flatten(actual)
    reference, reference_spec = pytree.tree_flatten(expected)
    assert observed_spec == reference_spec
    for actual_value, expected_value in zip(observed, reference, strict=True):
        assert torch.isfinite(actual_value).all()
        assert torch.isfinite(expected_value).all()
        torch.testing.assert_close(actual_value, expected_value, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("freq", [2.0, [1.0, 2.0, 3.0], [[1.0, 2.0], [3.0, 4.0]]])
@pytest.mark.parametrize("finite_off", [False, True])
def test_constant_repeat_isolates_window_gradients_and_preserves_sweep_shapes(
    freq, finite_off
):
    waveform = _constant_repeat(freq, finite_off=finite_off)
    parameters = _parameters(waveform)
    assert set(parameters) == {"freq", "delay", "off", "tau", "waveform.value"}
    reference_parameters = _parameters(waveform, clone=True)
    t = _tensor([0.16, 0.2, 0.24, 0.45, 0.76, 0.8, 0.84])
    actual = waveform(t)
    reference = _constant_reference(t, reference_parameters, finite_off=finite_off)
    assert actual.shape == (*torch.as_tensor(freq).shape, t.numel())
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # Onset is included and a finite off boundary is excluded exactly.
    assert torch.all(actual[..., 1] == parameters["waveform.value"])
    if finite_off:
        assert torch.all(actual[..., -2:] == 0)
    weights = _tensor([0.2, -0.3, 0.9, 0.4, 1.2, -0.5, 0.6]).expand(actual.shape)
    observed = _gradients(actual, parameters, weights, require=("delay", "off", "tau"))
    expected = _gradients(reference, reference_parameters, weights)
    _assert_finite_close(observed, expected)
    assert observed["delay"].abs() > 1e-4
    assert observed["tau"].abs() > 1e-4
    assert torch.count_nonzero(observed["freq"]) == 0
    if finite_off:
        assert observed["off"].abs() > 1e-4
    else:
        assert observed["off"] == 0


def test_mixed_finite_and_infinite_repeat_windows_keep_independent_gradients():
    with dn.ctx(REQUIRE_GRAD=1):
        waveform = (
            dn.constant(value=_tensor(1.3))
            .repeat(
                _tensor([1.2, 2.0, 3.4]),
                delay=_tensor([0.12, 0.2, 0.28]),
                off=_tensor([torch.inf, 0.62, 0.82]),
                tau=_tensor([0.03, 0.05, 0.08]),
            )
            .to(dtype=DTYPE)
        )
    parameters = _parameters(waveform)
    reference_parameters = _parameters(waveform, clone=True)
    t = _tensor([0.1, 0.2, 0.3, 0.58, 0.62, 0.75, 0.82, 0.86])
    reference = torch.stack(
        [
            _constant_reference(
                t,
                {
                    name: value if value.ndim == 0 else value[index]
                    for name, value in reference_parameters.items()
                },
                finite_off=index != 0,
            )
            for index in range(3)
        ]
    )
    actual = waveform(t)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    weights = torch.arange(1, 25, dtype=DTYPE).reshape(3, 8) / 10
    observed = _gradients(actual, parameters, weights, require=("delay", "off", "tau"))
    _assert_finite_close(observed, _gradients(reference, reference_parameters, weights))
    assert observed["off"][0] == 0
    assert torch.all(observed["off"][1:].abs() > 1e-4)


def test_repeated_pulse_preserves_child_and_wrapper_timing_gradients():
    with dn.ctx(REQUIRE_GRAD=1):
        waveform = dn.mono_rect(
            amp=_tensor(1.4), delay=_tensor(0.03), pw=_tensor(0.18), tau=_tensor(0.04)
        ).repeat(_tensor(2.0), delay=_tensor(0.1), off=_tensor(0.72), tau=_tensor(0.06))
    parameters = _parameters(waveform)
    reference_parameters = _parameters(waveform, clone=True)
    t = _tensor([0.08, 0.1, 0.12, 0.15, 0.21, 0.29, 0.59, 0.63, 0.69, 0.72, 0.75])
    p = reference_parameters
    phase = torch.fmod(t - p["delay"], 1.0 / p["freq"])
    outer = _reference_gate(t, p["delay"], p["off"], p["tau"])
    child = _reference_gate(
        phase,
        p["waveform.delay"],
        p["waveform.delay"] + p["waveform.pw"],
        p["waveform.tau"],
    )
    reference = outer * p["waveform.amp"] * child
    actual = waveform(t)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    weights = torch.arange(1, t.numel() + 1, dtype=DTYPE) / 10
    observed = _gradients(actual, parameters, weights, require=tuple(parameters))
    _assert_finite_close(observed, _gradients(reference, reference_parameters, weights))
    for name in (
        "freq",
        "delay",
        "off",
        "tau",
        "waveform.delay",
        "waveform.pw",
        "waveform.tau",
    ):
        assert observed[name].abs() > 1e-4


@pytest.mark.parametrize("finite_off", [False, True])
@torch_compiler_warning_context()
def test_repeat_window_supports_forward_and_reverse_jacobians(finite_off):
    waveform = _constant_repeat(finite_off=finite_off)
    parameters = _parameters(waveform, clone=True)
    t = _tensor([0.16, 0.2, 0.24, 0.76, 0.8, 0.84])

    def actual(p):
        return torch.func.functional_call(waveform, p, (t,))

    def reference(p):
        return _constant_reference(t, p, finite_off=finite_off)

    expected = torch.func.jacrev(reference)(parameters)
    _assert_finite_close(torch.func.jacrev(actual)(parameters), expected)
    _assert_finite_close(torch.func.jacfwd(actual)(parameters), expected)


@torch_compiler_warning_context()
def test_fullgraph_compiled_repeat_preserves_window_gradients():
    waveform = _constant_repeat(finite_off=False)
    parameters = _parameters(waveform)
    reference_parameters = _parameters(waveform, clone=True)
    t = _tensor([0.16, 0.2, 0.24, 0.76, 0.8, 0.84])
    compiled = torch.compile(waveform, backend="aot_eager", fullgraph=True)
    actual = compiled(t)
    reference = _constant_reference(t, reference_parameters, finite_off=False)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    weights = _tensor([0.2, -0.3, 0.9, 1.2, -0.5, 0.6])
    observed = _gradients(actual, parameters, weights, require=("delay", "off", "tau"))
    _assert_finite_close(observed, _gradients(reference, reference_parameters, weights))


def test_repeat_default_temperature_and_legacy_positional_controls():
    with dn.ctx(REQUIRE_GRAD=1):
        child = dn.constant(value=1.0)
        waveform = child.repeat(2.0, 0.2, 0.8).to(dtype=DTYPE)
    assert waveform.tau.item() == pytest.approx(0.01)
    assert waveform.tau.dtype == DTYPE
    assert waveform.tau.requires_grad
    with pytest.raises(TypeError):
        child.repeat(2.0, 0.2, 0.8, 0.05)


@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float64])
def test_repeat_keeps_masked_nonfinite_child_values_zero_and_preserves_dtype(
    output_dtype,
):
    with dn.ctx(REQUIRE_GRAD=1):
        child = _NonfiniteOutsideWindow(output_dtype)
        waveform = child.repeat(
            _tensor(1.0), delay=_tensor(0.2), off=_tensor(0.5), tau=_tensor(0.05)
        )
    t = _tensor([-0.1, 0.2, 0.4, 0.5, 0.8])
    actual = waveform(t)
    assert actual.dtype == output_dtype
    torch.testing.assert_close(
        actual, torch.tensor([0, 1, 1, 0, 0], dtype=output_dtype), rtol=0, atol=0
    )
    observed = _gradients(
        actual,
        _parameters(waveform),
        torch.ones_like(actual),
        require=("delay", "off", "tau"),
    )
    assert all(torch.isfinite(value).all() for value in observed.values())


def test_repeat_window_retains_mixed_derivatives_with_child_amplitude():
    waveform = _constant_repeat(finite_off=True)
    t = _tensor([0.72, 0.8, 0.86])
    weights = _tensor([0.3, 0.7, 1.1])
    loss = (waveform(t) * weights).sum()
    off_gradient, amplitude_gradient = torch.autograd.grad(
        loss, (waveform.off, waveform.waveform.value), create_graph=True
    )
    off_then_amplitude = torch.autograd.grad(
        off_gradient, waveform.waveform.value, retain_graph=True
    )[0]
    amplitude_then_off = torch.autograd.grad(amplitude_gradient, waveform.off)[0]
    onset = torch.sigmoid((t - waveform.delay) / waveform.tau)
    ending = torch.sigmoid((waveform.off - t) / waveform.tau)
    expected = (weights * onset * ending * (1 - ending) / waveform.tau).sum()
    assert expected.abs() > 0.1
    _assert_finite_close(off_then_amplitude, expected)
    _assert_finite_close(amplitude_then_off, expected)
