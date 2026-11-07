"""Invocation schedules use evaluation-local scratch and shared waveform math."""

import pytest
import torch
from torch._higher_order_ops.scan import scan
from torch.func import functional_call

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.stim.waveform._poisson import (
    POISSON_INVOCATION_POSITION,
    POISSON_INVOCATION_STACK,
    configure_poisson_invocations,
    evaluate_poisson_schedule,
    poisson_invocation_event_count,
    poisson_invocation_schedule,
    poisson_schedule_from_draws,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous_threads)


def test_invocation_selection_keeps_inputs_owned_and_saved_indices_independent():
    schedules = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    position = torch.zeros((1,), dtype=torch.int64)
    first, following = poisson_invocation_schedule(schedules, position)
    assert torch.equal(position, torch.tensor([0]))
    position.copy_(following)
    second, following = poisson_invocation_schedule(schedules, position)
    position.copy_(following)
    assert torch.equal(position, torch.tensor([2]))
    gradient = torch.autograd.grad((first * second).sum(), schedules)[0]
    torch.testing.assert_close(gradient, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))


@pytest.mark.parametrize(
    "draws,interval,start,noise,off,expected",
    (
        ([0.0, 0.5, 0.5], 1.0, 0.0, 1.0, torch.inf, 1),
        ([0.5, 0.0, 0.5], 1.0, 0.0, 1.0, torch.inf, 2),
        ([0.5, 0.5, 0.0], 1.0, 0.0, 1.0, torch.inf, 3),
        ([0.5, 0.5, 0.5], 1.0e308, 1.0e308, 0.0, torch.inf, 1),
        ([0.5, 0.5, 0.5], 0.3, 0.0, 0.0, 0.5, 2),
        ([0.5, 0.5, 0.5], 0.3, 2.0, 0.0, 1.0, 1),
        ([], 0.3, 0.0, 0.5, torch.inf, 1),
    ),
)
def test_native_invocation_event_count_uses_clock_termination_and_empty_sentinel(
    draws, interval, start, noise, off, expected
):
    uniforms = torch.tensor(draws, requires_grad=True)
    before = uniforms.detach().clone()
    actual = poisson_invocation_event_count(
        uniforms, interval=interval, start=start, noise=noise, off=off
    )
    assert actual.shape == ()
    assert actual.dtype == torch.int64
    assert actual.item() == expected
    assert not actual.requires_grad
    assert torch.equal(uniforms.detach(), before)


def test_invocation_count_ignores_waveform_dtype_overflow_after_native_clock():
    draws = torch.full((3,), 0.5)
    recipe = dict(interval=1.0e40, start=1.0e40, noise=0.0, off=torch.inf)
    waveform_schedule = poisson_schedule_from_draws(
        draws, **recipe, dtype=torch.float32
    )
    assert torch.isinf(waveform_schedule).all()
    actual = poisson_invocation_event_count(draws, **recipe)
    assert actual.item() == 3


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_invocation_count_assertion_rejects_changed_child_count(backend):
    def checked(draws):
        count = poisson_invocation_event_count(
            draws, interval=0.3, start=0.0, noise=0.7, off=torch.inf
        )
        torch._assert_async(
            count == 3, "Poisson draws changed the nested invocation count"
        )
        return draws.detach().clone()

    compiled = torch.compile(checked, backend=backend, fullgraph=True)
    valid = torch.tensor([0.5, 0.6, 0.7])
    torch.testing.assert_close(compiled(valid), valid, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="changed the nested invocation count"):
        compiled(torch.tensor([0.0, 0.6, 0.7]))


@pytest.mark.parametrize("lanes", (0, 2))
def test_invocation_count_supports_mapped_and_empty_draw_batches(lanes):
    draws = torch.full((lanes, 3), 0.5)
    if lanes:
        draws[0, 0] = 0.0
    actual = torch.vmap(
        lambda value: poisson_invocation_event_count(
            value, interval=0.3, start=0.0, noise=0.7, off=torch.inf
        )
    )(draws)
    assert actual.shape == (lanes,)
    assert torch.equal(actual, torch.tensor([1, 3] if lanes else [], dtype=torch.int64))


def _expression():
    waveform = dn.mono_rect(amp=0.4, pw=0.21).poisson(
        interval=0.3, n=3, generator=torch.Generator().manual_seed(83)
    )
    configure_poisson_invocations(waveform)
    expression = waveform + waveform
    parameter_name = "waveforms.0.waveform.amp"
    prefix = "waveforms.0."

    def evaluate(schedules, amplitude, times):
        mapping = {
            parameter_name: amplitude,
            f"{prefix}{POISSON_INVOCATION_STACK}": schedules,
            f"{prefix}{POISSON_INVOCATION_POSITION}": torch.zeros(
                (1,), dtype=torch.int64, device=schedules.device
            ),
        }
        return functional_call(
            expression,
            mapping,
            (times,),
            tie_weights=False,
        )

    return waveform, evaluate


@pytest.mark.parametrize("backend", (None, "aot_eager"))
def test_aliased_waveform_invocations_replay_without_retaining_scratch(backend):
    waveform, evaluate = _expression()
    schedules = torch.tensor([[0.0, 0.3, torch.inf], [0.12, 0.37, 0.62]])
    times = torch.linspace(0.0, 0.9, 19)
    amplitude = waveform.waveform.amp
    expected = sum(
        evaluate_poisson_schedule(waveform.waveform.fn, times, schedule)
        for schedule in schedules
    )
    expected_gradient = torch.autograd.grad(expected.square().sum(), amplitude)[0]
    before_schedule = waveform._spike_times.clone()
    before_rng = waveform.generator.get_state().clone()
    if backend is not None:
        evaluate = torch.compile(evaluate, backend=backend, fullgraph=True)
    for _ in range(2):
        actual = evaluate(schedules, amplitude, times)
        actual_gradient = torch.autograd.grad(actual.square().sum(), amplitude)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        assert getattr(waveform, POISSON_INVOCATION_STACK).shape[0] == 0
        assert torch.equal(
            getattr(waveform, POISSON_INVOCATION_POSITION), torch.tensor([0])
        )
        assert torch.equal(waveform._spike_times, before_schedule)
        assert torch.equal(waveform.generator.get_state(), before_rng)
    assert all("_poisson_invocation_" not in name for name in waveform.state_dict())


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_local_invocation_positions_reset_inside_compiled_scan_and_backward(backend):
    waveform, evaluate = _expression()
    times = torch.linspace(0.0, 0.9, 19)
    schedules = torch.tensor(
        [
            [[0.0, 0.3, torch.inf], [0.12, 0.37, 0.62]],
            [[0.04, 0.21, 0.68], [0.14, 0.45, torch.inf]],
        ]
    )
    amplitude = waveform.waveform.amp

    def rollout(draws, amp):
        def step(carry, current):
            value = evaluate(current, amp, times).square().sum()
            return (carry + value).clone(), value.clone()

        return scan(step, amp.new_zeros(()), draws)

    expected = sum(
        evaluate(value, amplitude, times).square().sum() for value in schedules
    )
    actual, emitted = torch.compile(rollout, backend=backend, fullgraph=True)(
        schedules, amplitude
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, emitted.sum(), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, amplitude)[0],
        torch.autograd.grad(expected, amplitude)[0],
        rtol=0,
        atol=0,
    )
    assert getattr(waveform, POISSON_INVOCATION_STACK).shape[0] == 0
    assert torch.count_nonzero(getattr(waveform, POISSON_INVOCATION_POSITION)) == 0


@pytest.mark.parametrize("n,start,off", ((5, 0.0, 0.5), (0, 0.0, 2.0), (5, 2.0, 1.0)))
def test_static_native_event_slots_preserve_child_calls_and_empty_sentinel(
    monkeypatch, n, start, off
):
    child = dn.constant(value=2.0)
    waveform = child.poisson(interval=0.3, n=n, start=start, off=off, noise=0.0)
    calls = []
    original = child.fn

    def observed(times):
        calls.append(times)
        return original(times)

    monkeypatch.setattr(child, "fn", observed)
    times = torch.linspace(0.0, 1.0, 7)
    expected = waveform(times)
    native_slots = len(calls)
    assert native_slots == waveform._spike_times.shape[0]
    padded = waveform._spike_times.new_full((max(n, 1),), torch.inf)
    padded[:native_slots] = waveform._spike_times
    waveform._spike_times = padded
    configure_poisson_invocations(waveform, event_slots=native_slots)
    calls.clear()
    actual = waveform(times)
    assert len(calls) == native_slots
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
