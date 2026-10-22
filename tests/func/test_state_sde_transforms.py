"""State SDE paths preserve transforms and real compiled execution."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_state_sde_execution import DT, _HeunLeak, _OULeak, _population


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _case(mechanism):
    functional, tensors = dn.func.make_functional(_population(mechanism), dt=DT)
    tape = functional.sample_noise_draw_tape(tensors.state, 3)
    sigma_key = next(
        name for name in tensors.parameters if name.endswith("sigma_param")
    )
    draw_key = next(iter(tape.draws))
    initial = tensors.state["mechanisms"]["sde"]["x"]

    def trajectory(sigma, initial, dt, draws):
        parameters = {**tensors.parameters, sigma_key: sigma}
        constants = {**tensors.constants, "dt": dt}
        state = {
            **tensors.state,
            "mechanisms": {
                **tensors.state["mechanisms"],
                "sde": {**tensors.state["mechanisms"]["sde"], "x": initial},
            },
        }
        packet = tape._replace(draws={**tape.draws, draw_key: draws})
        prepared = functional.prepare(parameters, constants)
        final, _ = functional.rollout(
            parameters,
            prepared,
            state,
            steps=3,
            inputs=dn.func.RolloutInput(noise=packet),
        )
        return torch.cat(
            (
                final["integrator"]["v"].flatten(),
                final["mechanisms"]["sde"]["x"].flatten(),
            )
        )

    return trajectory, (
        tensors.parameters[sigma_key].detach(),
        initial,
        tensors.constants["dt"],
        tape.draws[draw_key],
    )


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
def test_sde_jacrev_jacfwd_and_detached_draw_jvp(mechanism):
    trajectory, operands = _case(mechanism)
    reverse = torch.func.jacrev(trajectory, argnums=(0, 1, 2))(*operands)
    forward = torch.func.jacfwd(trajectory, argnums=(0, 1, 2))(*operands)
    torch.testing.assert_close(reverse, forward, rtol=2e-10, atol=2e-11)
    assert all(value.abs().sum() > 0 for value in reverse)
    _, tangent = torch.func.jvp(
        trajectory,
        operands,
        tuple(
            torch.ones_like(value) if index == 3 else torch.zeros_like(value)
            for index, value in enumerate(operands)
        ),
    )
    torch.testing.assert_close(tangent, torch.zeros_like(tangent), rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
def test_sde_vmap_and_empty_vmap_match_independent_paths(mechanism):
    trajectory, operands = _case(mechanism)
    batched = tuple(
        torch.stack((value, value + (0.01 if index != 2 else 0.02)))
        for index, value in enumerate(operands)
    )
    expected = torch.stack(
        [trajectory(*(value[index] for value in batched)) for index in range(2)]
    )
    actual = torch.vmap(trajectory)(*batched)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    empty = torch.vmap(trajectory)(*(value[:0] for value in batched))
    assert empty.shape == (0, *expected.shape[1:])


@pytest.mark.cpu
@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_inductor_sde_recording_and_continuation_match_native(mechanism, execution):
    source, native = _population(mechanism), _population(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        execution=execution,
        backend="inductor",
        callbacks={"trace": dn.func.Recorder(("v", "sde.x"))},
    )
    state, first = runner()
    continued, second = runner(state, callback_state=first["callbacks"].state)
    voltages, states = [native.v.clone()], [native.mech.sde.x.clone()]
    with torch.no_grad():
        for _ in range(8):
            native.step(DT)
            voltages.append(native.v.clone())
            states.append(native.mech.sde.x.clone())
    for name, expected in (("v", voltages), ("sde.x", states)):
        actual = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        torch.testing.assert_close(
            actual, torch.stack(expected), rtol=2e-11, atol=2e-12
        )
    torch.testing.assert_close(
        continued, runner.functional.extract(native).state, rtol=2e-11, atol=2e-12
    )
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_sde_tapes_and_checkpoint_suffix_match_native(mechanism, dtype):
    with dn.ctx(DEVICE="cuda", DTYPE=dtype):
        source, native = _population(mechanism), _population(mechanism)
    runner = dn.compile(source, dt=DT, steps=4, backend="aot_eager")
    prefix, _ = runner()
    final, _ = runner(prefix)
    with torch.no_grad():
        for _ in range(8):
            native.step(DT)
    tolerance = 5e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(
        final, runner.functional.extract(native).state, rtol=tolerance, atol=tolerance
    )
