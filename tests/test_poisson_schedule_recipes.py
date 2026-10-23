"""Poisson recipes share native scalar ordering and explicit detached draws."""

import io

import pytest
import torch

import dendra as dn
from dendra.models.stim.waveform._poisson import (
    evaluate_poisson_schedule,
    poisson_schedule_from_draws,
    sample_poisson_draws,
    sample_poisson_draws_from_state,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    rng = torch.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    try:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64):
            yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_rng_state(rng)


def _legacy_schedule(*, interval, n, start, noise, off, generator, dtype):
    """Independent reference for the original Python scalar implementation."""
    times = []
    current = start
    count = 0
    while current < off and (n is None or count < n):
        times.append(current)
        if noise == 0.0:
            delta = interval
        else:
            uniform = torch.rand((), generator=generator, device=generator.device)
            exponential = -uniform.log() * interval
            delta = (1.0 - noise) * interval + noise * exponential.item()
        current += delta
        count += 1
    return torch.tensor(times or [torch.inf], dtype=dtype), count


@pytest.mark.parametrize("draw_dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("output_dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("noise", (0.0, 0.2, 1.0))
@pytest.mark.parametrize(
    "n,start,off", ((0, 0.15, 4.0), (1, 0.15, 4.0), (17, 0.15, 1.3), (None, 0.15, 4.0))
)
def test_shared_schedule_matches_original_values_and_exact_scalar_rng_consumption(
    draw_dtype, output_dtype, noise, n, start, off
):
    torch.set_default_dtype(draw_dtype)
    reference_generator = torch.Generator().manual_seed(12345)
    generator = torch.Generator().manual_seed(12345)
    spec = dict(interval=0.31, n=n, start=start, noise=noise, off=off)
    expected, count = _legacy_schedule(
        **spec, generator=reference_generator, dtype=output_dtype
    )
    draws, actual_count = sample_poisson_draws(**spec, generator=generator)
    result = poisson_schedule_from_draws(
        draws,
        interval=spec["interval"],
        start=start,
        noise=noise,
        off=off,
        dtype=output_dtype,
    )
    assert actual_count == count
    assert draws.shape == (count if n is None else n,)
    assert draws.dtype == draw_dtype
    torch.testing.assert_close(result[: max(count, 1)], expected, rtol=0, atol=0)
    assert torch.isinf(result[max(count, 1) :]).all()
    assert torch.equal(generator.get_state(), reference_generator.get_state())


@pytest.mark.parametrize("noise", (0.0, 0.3, 1.0))
@pytest.mark.parametrize("n,start,off", ((0, 0.0, 2.0), (4, 2.0, 1.0), (7, 0.0, 1.0)))
def test_native_waveform_preserves_dynamic_schedule_shape(noise, n, start, off):
    generator = torch.Generator().manual_seed(924)
    reference_generator = torch.Generator().manual_seed(924)
    spec = dict(interval=0.3, n=n, start=start, noise=noise, off=off)
    waveform = (
        dn.mono_rect().to(dtype=torch.float64).poisson(**spec, generator=generator)
    )
    expected, _ = _legacy_schedule(
        **spec, generator=reference_generator, dtype=torch.float64
    )
    torch.testing.assert_close(waveform._spike_times, expected, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), reference_generator.get_state())
    waveform.reshape_for_intra()
    waveform.regenerate_schedule_()
    expected, _ = _legacy_schedule(
        **spec, generator=reference_generator, dtype=torch.float64
    )
    torch.testing.assert_close(
        waveform._spike_times, expected.reshape(-1, 1, 1), rtol=0, atol=0
    )
    assert torch.equal(generator.get_state(), reference_generator.get_state())


def test_state_adapter_keeps_global_rng_and_input_frame_unchanged_and_continues():
    state = torch.Generator().manual_seed(1729).get_state().clone()
    original_state = state.clone()
    global_state = torch.get_rng_state().clone()
    spec = dict(interval=0.13, n=19, start=0.0, noise=0.8, off=0.7)
    first, count, following = sample_poisson_draws_from_state(state, **spec)
    second, second_count, final = sample_poisson_draws_from_state(following, **spec)
    generator = torch.Generator()
    generator.set_state(state)
    expected_first, expected_count = sample_poisson_draws(**spec, generator=generator)
    expected_second, expected_second_count = sample_poisson_draws(
        **spec, generator=generator
    )
    assert torch.equal(first, expected_first)
    assert torch.equal(second, expected_second)
    assert count == expected_count
    assert second_count == expected_second_count
    assert torch.equal(final, generator.get_state())
    assert torch.equal(state, original_state)
    assert torch.equal(global_state, torch.get_rng_state())


def test_poisson_draws_and_onsets_are_detached_but_waveform_gradients_survive():
    draws = torch.tensor([0.8, 0.4, 0.1, 0.5], requires_grad=True)
    schedule = poisson_schedule_from_draws(
        draws, interval=0.3, start=0.1, noise=0.5, off=0.7, dtype=torch.float64
    )
    assert not schedule.requires_grad
    pulse = dn.mono_rect(amp=0.4, delay=0.03, pw=0.12).to(dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 67, dtype=torch.float64)
    result = evaluate_poisson_schedule(pulse.fn, t, schedule)
    expected = sum(pulse(t - onset) for onset in schedule if torch.isfinite(onset))
    parameters = tuple(pulse.parameters())
    actual_gradients = torch.autograd.grad(result.square().sum(), parameters)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert draws.grad is None


@pytest.mark.parametrize("factory", (lambda: dn.constant(value=2.0), dn.sin))
def test_empty_schedules_and_padded_slots_contribute_zero(factory):
    waveform = factory().poisson(interval=0.2, n=3, start=2.0, off=1.0)
    t = torch.linspace(0.0, 1.0, 13, dtype=torch.float64)
    result = waveform(t)
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
    assert torch.isfinite(result).all()
    result.sum().backward()
    for parameter in waveform.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) == 0


def test_padded_constant_recipe_counts_only_finite_onsets():
    schedule = poisson_schedule_from_draws(
        torch.full((7,), 0.5),
        interval=0.3,
        start=0.0,
        noise=0.0,
        off=0.7,
        dtype=torch.float64,
    )
    pulse = dn.constant(value=2.0)
    result = evaluate_poisson_schedule(pulse.fn, torch.arange(4.0), schedule)
    torch.testing.assert_close(result, torch.full_like(result, 6.0))


def test_native_pickle_continuation_preserves_private_generator_and_schedule():
    waveform = dn.mono_rect(amp=0.3).poisson(
        interval=0.3,
        n=7,
        off=1.1,
        randomize_every_call=True,
        generator=torch.Generator().manual_seed(2109),
    )
    t = torch.linspace(0.0, 2.0, 31)
    waveform(t)
    payload = io.BytesIO()
    torch.save(waveform, payload)
    payload.seek(0)
    restored = torch.load(payload, weights_only=False)
    torch.testing.assert_close(restored._spike_times, waveform._spike_times)
    for _ in range(3):
        torch.testing.assert_close(restored(t), waveform(t), rtol=0, atol=0)
        assert torch.equal(
            restored.generator.get_state(), waveform.generator.get_state()
        )


def test_pure_schedule_and_evaluation_compile_as_a_full_graph():
    def recipe(draws, amplitude, t):
        schedule = poisson_schedule_from_draws(
            draws,
            interval=0.3,
            start=0.0,
            noise=0.5,
            off=0.7,
            dtype=t.dtype,
            device=t.device,
        )
        return evaluate_poisson_schedule(
            lambda time: amplitude * torch.exp(-time.square()), t, schedule
        )

    draws = torch.tensor([0.8, 0.4, 0.2, 0.7])
    amplitude = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    expected = recipe(draws, amplitude, t)
    result = torch.compile(recipe, backend="aot_eager", fullgraph=True)(
        draws, amplitude, t
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), amplitude)[0],
        torch.autograd.grad(expected.sum(), amplitude)[0],
        rtol=0,
        atol=0,
    )
