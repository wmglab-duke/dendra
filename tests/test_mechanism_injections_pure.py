"""Waveform injection numerics share pure and native committing paths."""

import copy
import math

import pytest
import torch
from torch.func import functional_call, jacfwd, jacrev, jvp, vmap

from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._injections import expand_injection_value


class _LinearWaveform(torch.nn.Module):
    def __init__(self, amplitude, slope=0.0):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.as_tensor(amplitude).clone())
        self.slope = torch.nn.Parameter(torch.as_tensor(slope).clone())

    def forward(self, t):
        return self.amplitude.unsqueeze(-1) + self.slope.unsqueeze(-1) * t


class _Evaluation(torch.nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def forward(self, v, t):
        return self.owner.injection_values(v, t=t)


def _mechanism(dtype=torch.float64):
    shape = (2, 3)
    return Mechanism(
        "injections",
        torch.full(shape, 34.0, dtype=dtype),
        torch.ones(shape, dtype=dtype),
        shape,
        shape,
    )


def _register(mechanism, amplitude, slope=0.0, **kwargs):
    waveform = _LinearWaveform(amplitude, slope).to(mechanism.diam)
    assert mechanism.register_waveform_injection(
        waveform,
        index=(slice(None), 1),
        model_shape=(2, 3),
        **kwargs,
    )
    return waveform


def _assert_source_unchanged(mechanism, values, identities):
    assert set(mechanism.state_dict()) == set(values)
    for name, value in mechanism.state_dict().items():
        torch.testing.assert_close(value, values[name], rtol=0, atol=0)
    assert {name: id(value) for name, value in mechanism.named_buffers()} == identities


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3), (0,)])
@pytest.mark.parametrize("payload", ["scalar", "local", "suffix", "selected"])
def test_pure_current_padding_matches_native_without_changing_inputs(
    dtype, batch_shape, payload
):
    mechanism = _mechanism(dtype)
    amplitude = {
        "scalar": 2.0,
        "local": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "suffix": [1.0, 2.0, 3.0],
        "selected": [2.0, 5.0],
    }[payload]
    _register(mechanism, amplitude, slope=0.5, scale=3.0)
    v = torch.full((*batch_shape, 2, 3), -60.0, dtype=dtype)
    t = torch.tensor(0.4, dtype=dtype)
    source = copy.deepcopy(mechanism.state_dict())
    identities = {name: id(value) for name, value in mechanism.named_buffers()}
    v_before, t_before = v.clone(), t.clone()

    actual = mechanism.injection_values(v, t=t)
    expected = torch.zeros_like(v)
    if payload == "scalar":
        expected[..., :, 1] = 6.6
    elif payload == "suffix":
        expected[..., :, 1] = 6.6
    else:
        expected[..., 0, 1] = 6.6
        expected[..., 1, 1] = 15.6
    torch.testing.assert_close(actual, expected)
    _assert_source_unchanged(mechanism, source, identities)
    torch.testing.assert_close(v, v_before, rtol=0, atol=0)
    torch.testing.assert_close(t, t_before, rtol=0, atol=0)

    native = mechanism.evaluate_injections(v, t=t)
    assert mechanism.i_inj is native
    torch.testing.assert_close(native, actual, rtol=0, atol=0)


def test_explicit_batch_selected_payload_has_independent_currents():
    mechanism = _mechanism()
    _register(mechanism, [[1.0, 2.0], [3.0, 4.0]], scale=2.0)
    v = torch.zeros(2, 2, 3, dtype=torch.float64)
    result = mechanism.injection_values(v, t=0.0)
    expected = torch.zeros_like(v)
    expected[..., 1] = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_named_currents_sum_only_matching_registrations_and_reuse_shared_waveform():
    mechanism = _mechanism()
    waveform = _register(mechanism, [1.0, 2.0], current_name="i_a", scale=2.0)
    assert mechanism.register_waveform_injection(
        waveform,
        index=(slice(None), 1),
        model_shape=(2, 3),
        current_name="i_a",
        scale=-0.5,
    )
    _register(mechanism, 7.0, current_name="i_b")
    assert mechanism.injected_waveforms[0] is mechanism.injected_waveforms[1]
    v = torch.zeros(2, 3, dtype=torch.float64)
    result = mechanism.injection_values(v, t=0.0, current_name="i_a")
    expected = torch.zeros_like(v)
    expected[:, 1] = torch.tensor([1.5, 3.0])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(result.sum(), waveform.amplitude)
    torch.testing.assert_close(gradient, torch.full_like(gradient, 1.5))
    torch.testing.assert_close(
        mechanism.injection_values(v, t=0.0, current_name="missing"),
        torch.zeros_like(v),
    )


def test_scale_registration_copies_but_explicit_scale_and_clock_keep_gradients():
    mechanism = _mechanism()
    scale = torch.tensor([[2.0], [3.0]], dtype=torch.float64, requires_grad=True)
    waveform = _register(mechanism, [1.2, -0.4], [0.8, 1.1], scale=scale)
    assert not mechanism._injection_scale_0.requires_grad
    assert mechanism._injection_scale_0.data_ptr() != scale.data_ptr()
    evaluation = _Evaluation(mechanism)
    v = torch.zeros(2, 3, dtype=torch.float64)
    t = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    current = functional_call(evaluation, {"owner._injection_scale_0": scale}, (v, t))
    grad_amplitude, grad_slope, grad_scale, grad_t = torch.autograd.grad(
        current.sum(), (waveform.amplitude, waveform.slope, scale, t)
    )
    torch.testing.assert_close(grad_amplitude, scale.squeeze(-1))
    torch.testing.assert_close(grad_slope, t * scale.squeeze(-1))
    torch.testing.assert_close(
        grad_scale, (waveform.amplitude + t * waveform.slope).unsqueeze(-1)
    )
    torch.testing.assert_close(grad_t, (waveform.slope * scale.squeeze(-1)).sum())
    torch.testing.assert_close(
        mechanism._injection_scale_0, torch.tensor([[2.0], [3.0]], dtype=torch.float64)
    )


def test_explicit_time_overrides_native_clock_and_native_default_assigns_current():
    mechanism = _mechanism()
    _register(mechanism, 1.0, 2.0)
    mechanism.t = torch.tensor(3.0, dtype=torch.float64)
    native_default = mechanism.evaluate_injections()
    expected = torch.zeros(2, 3, dtype=torch.float64)
    expected[:, 1] = 7.0
    torch.testing.assert_close(native_default, expected)
    pure = mechanism.injection_values(torch.zeros_like(expected), t=0.5)
    expected[:, 1] = 2.0
    torch.testing.assert_close(pure, expected)
    assert mechanism.i_inj is native_default
    with pytest.raises(TypeError, match="t"):
        mechanism.injection_values(torch.zeros_like(expected))
    with pytest.raises(ValueError, match="one timestep"):
        mechanism.injection_values(torch.zeros_like(expected), t=torch.arange(2.0))


def test_full_local_registration_without_population_frame_and_clear():
    mechanism = _mechanism()
    waveform = _LinearWaveform(2.0).to(mechanism.diam)
    assert mechanism.register_waveform_injection(waveform)
    v = torch.zeros(3, 2, 3, dtype=torch.float64)
    torch.testing.assert_close(mechanism.injection_values(v, t=0.0), v + 2.0)
    mechanism.clear_injections()
    before = mechanism.i_inj
    torch.testing.assert_close(
        mechanism.injection_values(v, t=0.0), torch.zeros_like(v)
    )
    assert mechanism.i_inj is before
    assert not mechanism._injection_specs
    assert not list(mechanism.injected_waveforms)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_fullgraph_compile_preserves_clock_parameter_and_selected_scale_gradients(
    backend,
):
    mechanism = _mechanism()
    waveform = _register(mechanism, [1.0, 2.0], [0.5, 0.8], scale=[[2.0], [3.0]])
    evaluation = _Evaluation(mechanism)
    compiled = torch.compile(evaluation, backend=backend, fullgraph=True)
    v = torch.zeros(2, 2, 3, dtype=torch.float64)
    t = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    actual = compiled(v, t)
    expected = evaluation(v, t)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), (t, waveform.amplitude))
    expected_gradient = torch.autograd.grad(expected.sum(), (t, waveform.amplitude))
    for observed, reference in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(observed, reference)


def test_pure_current_supports_vmap_empty_vmap_jacobians_and_jvp():
    mechanism = _mechanism()
    _register(mechanism, [1.0, 2.0], [0.5, 0.8])
    v = torch.zeros(2, 3, dtype=torch.float64)

    def evaluate(t):
        return mechanism.injection_values(v, t=t)

    times = torch.tensor([0.0, 0.2, 0.7], dtype=torch.float64)
    torch.testing.assert_close(
        vmap(evaluate)(times), torch.stack([evaluate(t) for t in times])
    )
    assert vmap(evaluate)(times[:0]).shape == (0, 2, 3)
    t = times[1]
    torch.testing.assert_close(jacfwd(evaluate)(t), jacrev(evaluate)(t))
    _, tangent = jvp(evaluate, (t,), (torch.ones_like(t),))
    torch.testing.assert_close(tangent, jacrev(evaluate)(t))


@pytest.mark.parametrize("batch_shape", [(), (0,), (2, 0)])
def test_padding_zero_selected_and_empty_batch_remain_pure(batch_shape):
    mask = torch.zeros(2, 3, dtype=torch.bool)
    value = torch.empty(0, dtype=torch.float64)
    out = torch.zeros((*batch_shape, 2, 3), dtype=torch.float64)
    result = expand_injection_value(
        value,
        mask,
        out,
        selected_index=torch.empty(0, dtype=torch.long),
        n_local_selected=0,
    )
    assert result.shape == out.shape
    torch.testing.assert_close(result, torch.zeros_like(out))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_pure_current_preserves_device_and_gradients():
    mechanism = _mechanism(torch.float32).cuda()
    waveform = _register(mechanism, [1.0, 2.0], [0.5, 0.8])
    v = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    t = torch.tensor(0.4, device="cuda", requires_grad=True)
    output = mechanism.injection_values(v, t=t)
    assert output.device == v.device
    gradient = torch.autograd.grad(output.sum(), (t, waveform.amplitude))
    assert all(torch.isfinite(value).all() for value in gradient)


def _legacy_reference(waveform, t, scale, mask, out):
    """Independent oracle for native scale/time ordering and selection padding."""
    value = waveform(torch.atleast_1d(t)) * scale
    if value.ndim and value.shape[-1] == 1:
        value = value.squeeze(-1)
    expanded_mask = mask.expand_as(out)
    try:
        return value.expand_as(out) * expanded_mask
    except RuntimeError:
        if value.numel() == int(expanded_mask.sum().item()):
            selected = value.reshape(-1)
        else:
            count = int(mask.sum().item())
            batch_shape = out.shape[: out.ndim - mask.ndim]
            selected = value.expand((*batch_shape, count)).reshape(-1)
        return torch.zeros_like(out).masked_scatter(expanded_mask, selected)


@pytest.mark.parametrize("payload_shape", [(), (2,), (2, 3)])
def test_time_last_scale_broadcast_keeps_legacy_order(payload_shape):
    mechanism = _mechanism()
    amplitude = torch.arange(1, 1 + math.prod(payload_shape), dtype=torch.float64)
    amplitude = amplitude.reshape(payload_shape)
    scale_shape = (*payload_shape, 1)
    scale = torch.arange(2, amplitude.numel() + 2, dtype=torch.float64).reshape(
        scale_shape
    )
    waveform = _register(mechanism, amplitude, slope=0.5, scale=scale)
    v = torch.zeros(2, 3, dtype=torch.float64)
    t = torch.tensor([0.4], dtype=torch.float64)
    expected = _legacy_reference(waveform, t, scale, mechanism._injection_mask_0, v)
    torch.testing.assert_close(mechanism.injection_values(v, t=t), expected)


def test_selected_parameter_forward_and_reverse_transforms_agree():
    mechanism = _mechanism()
    _register(mechanism, [1.0, 2.0], [0.5, 0.8])
    evaluation = _Evaluation(mechanism)
    v = torch.zeros(2, 3, dtype=torch.float64)
    t = torch.tensor(0.4, dtype=torch.float64)

    def evaluate(amplitude):
        return functional_call(
            evaluation,
            {"owner.injected_waveforms.0.amplitude": amplitude},
            (v, t),
        )

    amplitude = torch.tensor([1.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(jacfwd(evaluate)(amplitude), jacrev(evaluate)(amplitude))
    amplitudes = torch.stack((amplitude, amplitude * 2))
    torch.testing.assert_close(
        vmap(evaluate)(amplitudes),
        torch.stack([evaluate(value) for value in amplitudes]),
    )
    assert vmap(evaluate)(amplitudes[:0]).shape == (0, 2, 3)


@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 3)])
def test_registered_broadcast_mask_accepts_flattened_selected_payload(shape):
    mechanism = Mechanism(
        "broadcast_injections",
        torch.full((1, 3), 34.0, dtype=torch.float64),
        torch.ones(1, 3, dtype=torch.float64),
        (1, 3),
        (1, 3),
    )
    count = math.prod(shape[:-1]) * 2
    amplitude = torch.arange(1, count + 1, dtype=torch.float64)
    waveform = _LinearWaveform(amplitude).to(mechanism.diam)
    assert mechanism.register_waveform_injection(
        waveform, index=(slice(None), [0, 2]), model_shape=(1, 3)
    )
    v = torch.zeros(shape, dtype=torch.float64)
    t = torch.tensor(0.0, dtype=torch.float64)
    expected = _legacy_reference(
        waveform, t, mechanism._injection_scale_0, mechanism._injection_mask_0, v
    )
    torch.testing.assert_close(mechanism.injection_values(v, t=t), expected)
    compiled = torch.compile(
        _Evaluation(mechanism), backend="aot_eager", fullgraph=True
    )
    actual = compiled(v, t)
    torch.testing.assert_close(actual, expected)
    (gradient,) = torch.autograd.grad(actual.sum(), waveform.amplitude)
    torch.testing.assert_close(gradient, torch.ones_like(gradient))


class _CustomPaddingMechanism(Mechanism):
    def _expand_injection_value(self, value, mask, out, **kwargs):
        return Mechanism._expand_injection_value(self, value, mask, out, **kwargs) + 1.0


def test_native_adapter_retains_subclass_padding_override():
    shape = (2, 3)
    mechanism = _CustomPaddingMechanism(
        "custom_padding",
        torch.full(shape, 34.0, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
        shape,
        shape,
    )
    _register(mechanism, 2.0)
    v = torch.zeros(shape, dtype=torch.float64)
    expected = torch.ones_like(v)
    expected[:, 1] = 3.0
    torch.testing.assert_close(mechanism.evaluate_injections(v, t=0.0), expected)
