"""Bound waveform grids retain native invocation precision across loop engines."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import reset_torch_compiler, torch_compiler_warning_context
from dendra.func._stimulation_clock import _sampling_time_tape
from dendra.models.mod import pas
from dendra.units import nA

pytestmark = pytest.mark.cpu
DT = 0.025
STEPS = 16


@pytest.fixture(autouse=True)
def _compiler_scope():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        reset_torch_compiler()
        torch.set_num_threads(previous)


def _population(dtype, mode, *, device="cpu", training=False):
    with dn.ctx(JIT=0, DTYPE=dtype, DEVICE=device, REQUIRE_GRAD=1, IMEM=False):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=0.001, e=-65.0)
        pulse = dn.mono_rect(
            amp=0.01 * nA if mode == "intra" else 1.0,
            delay=0.0 if dtype == torch.float32 else 0.1,
            pw=0.20000001788139343 if dtype == torch.float32 else 0.2,
        )
        if mode == "intra":
            model[..., 0].inject(pulse)
            extra = None
        else:
            extra = (
                torch.tensor([[0.0, 0.3, 0.7, -0.2, -0.1]], dtype=dtype, device=device),
                pulse,
            )
        model.train(training)
        model.initialize()
        if dtype == torch.float32:
            # A nonzero origin and an adjacent Float32 endpoint expose both
            # accumulation rounding and premature timestep downcasting.
            model.run(steps=3, dt=DT, extra=extra)
    return model, extra


def _native(model, extra, steps, **kwargs):
    recorder = dn.Recorder(("v", "t"))
    model.run(steps=steps, dt=DT, extra=extra, callbacks=[recorder], **kwargs)
    return {name: torch.from_numpy(recorder.numpy(name)).clone() for name in ("v", "t")}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("execution", ["scan", "while"])
@pytest.mark.parametrize("mode", ["intra", "bound_extra", "runtime_extra"])
def test_population_hard_pulse_matches_native_grid_across_chunks_and_tail(
    dtype,
    execution,
    mode,
):
    native, native_extra = _population(dtype, mode)
    model, extra = _population(dtype, mode)
    expected = _native(native, native_extra, STEPS)
    runtime = mode == "runtime_extra"
    program = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        chunk_steps=7,
        execution=execution,
        extra=dn.VarInput() if runtime else extra,
        callbacks={"trace": dn.Recorder(("v", "t"))},
        backend="eager",
    )
    initial = model.v.clone()
    state, auxiliary = program(**({"extra": extra} if runtime else {}))
    actual = auxiliary["callbacks"]["trace"]
    tolerance = 3e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        actual["v"], expected["v"], rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(actual["t"], expected["t"], rtol=0, atol=0)
    torch.testing.assert_close(state["clock"]["t"], native.t, rtol=0, atol=0)
    torch.testing.assert_close(model.v, initial, rtol=0, atol=0)
    assert actual["v"].shape[0] == STEPS + 1


def test_sampling_tape_keeps_native_float32_forward_and_timing_gradients():
    model, _ = _population(torch.float32, "intra")
    functional, _ = dn.func.make_functional(model, dt=DT)
    origin = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    dt = torch.tensor(DT, dtype=torch.float32, requires_grad=True)
    actual = _sampling_time_tape(functional, origin, 7, 9, DT, clock_dt=dt)
    offsets = torch.arange(7, 16, dtype=torch.float64)
    expected = (origin.detach().double() + offsets * DT).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    origin_gradient, dt_gradient = torch.autograd.grad(actual.sum(), (origin, dt))
    torch.testing.assert_close(origin_gradient, origin.new_tensor(9), rtol=0, atol=0)
    torch.testing.assert_close(
        dt_gradient, dt.new_tensor(float(offsets.sum())), rtol=0, atol=0
    )

    primal, tangent = torch.func.jvp(
        lambda start, step: _sampling_time_tape(
            functional,
            start,
            7,
            9,
            DT,
            clock_dt=step,
        ),
        (origin.detach(), dt.detach()),
        (torch.ones_like(origin), torch.ones_like(dt)),
    )
    torch.testing.assert_close(primal, expected, rtol=0, atol=0)
    torch.testing.assert_close(tangent, (1 + offsets).float(), rtol=0, atol=0)


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_population_continuation_restarts_stimulation_origin_without_clock_reset(
    execution,
):
    native, _ = _population(torch.float64, "intra")
    model, _ = _population(torch.float64, "intra")
    program = dn.compile(
        model,
        dt=DT,
        steps=9,
        chunk_steps=4,
        execution=execution,
        callbacks={"trace": dn.Recorder(("v", "t"))},
        backend="eager",
    )
    expected_first = _native(native, None, 9)
    _, first = program()
    torch.testing.assert_close(
        first["callbacks"]["trace"]["v"],
        expected_first["v"],
        rtol=1e-12,
        atol=1e-12,
    )
    continuation = program.restore_continuation(
        program.export_continuation(first["continuation"])
    )
    expected_second = _native(native, None, 7)
    state, second = program(continuation, steps=7)
    for name in ("v", "t"):
        torch.testing.assert_close(
            second["callbacks"]["trace"][name],
            expected_second[name][1:],
            rtol=1e-12,
            atol=1e-12,
        )
    torch.testing.assert_close(state["clock"]["t"], native.t, rtol=0, atol=0)


@pytest.mark.parametrize("execution", ["scan", "while"])
@pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "replay"])
def test_population_mixed_poisson_bank_preserves_each_native_sampling_policy(
    execution,
    explicit,
):
    native, _ = _population(torch.float64, "intra")
    model, _ = _population(torch.float64, "intra")
    pulses = []
    for population in (native, model):
        with dn.ctx(JIT=0, DTYPE=torch.float64, REQUIRE_GRAD=1, IMEM=False):
            # The zero-amplitude bank still consumes real RNG. Its presence
            # selects native accepted-clock sampling only for explicit replay.
            pulse = dn.mono_rect(amp=0.0, pw=0.04).poisson(
                interval=0.06,
                n=5,
                start=-0.03,
                noise=0.7,
                randomize_every_call=True,
                generator=torch.Generator().manual_seed(17),
            )
            population[..., -1].inject(pulse)
            pulses.append(pulse)
    program = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution=execution,
        callbacks={"trace": dn.Recorder(("v", "t"))},
        backend="eager",
    )
    source_rng = pulses[1].generator.get_state().clone()
    sampled = program.functional.sample_runtime_draws(
        program.initial_state, steps=STEPS
    )
    kwargs = {"draws": sampled} if explicit else {}
    expected = _native(native, None, STEPS, **kwargs)
    state, auxiliary = program(**kwargs)
    for name in ("v", "t"):
        torch.testing.assert_close(
            auxiliary["callbacks"]["trace"][name],
            expected[name],
            rtol=1e-12,
            atol=1e-12,
        )
    native_rng = pulses[0].generator.get_state()
    assert not torch.equal(native_rng, source_rng)
    rng_frames = [
        value
        for value in torch.utils._pytree.tree_leaves(state)
        if torch.is_tensor(value)
        and value.dtype == torch.uint8
        and value.shape == native_rng.shape
    ]
    expected_rng = [
        value[-1]
        for value in torch.utils._pytree.tree_leaves(sampled.rng)
        if torch.is_tensor(value)
        and value.dtype == torch.uint8
        and value.shape == (STEPS, *native_rng.shape)
    ]
    assert expected_rng
    assert all(
        any(torch.equal(value, expected) for value in rng_frames)
        for expected in expected_rng
    )
    if explicit:
        assert any(torch.equal(value, native_rng) for value in rng_frames)
    torch.testing.assert_close(
        pulses[1].generator.get_state(), source_rng, rtol=0, atol=0
    )
