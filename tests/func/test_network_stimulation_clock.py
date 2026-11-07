"""Network bank timestamps match native run invocation grids at hard edges."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import reset_torch_compiler, torch_compiler_warning_context
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


def _network(dtype, mode):
    with dn.ctx(JIT=0, DTYPE=dtype, DEVICE="cpu", REQUIRE_GRAD=1, IMEM=False):
        cell = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        cell.insert(pas, g=0.001, e=-65.0)
        # After three accepted steps the native float32 origin is rounded
        # before grid construction. These exact endpoints distinguish its
        # local invocation grid from the Network's global solver clock.
        width = 0.22500000894069672 if dtype == torch.float32 else 0.22500000000000003
        pulse = dn.mono_rect(
            amp=0.01 * nA if mode == "intra" else 1.0,
            delay=0.0,
            pw=torch.tensor(width, dtype=dtype),
        )
        if mode == "intra":
            cell[..., 0].inject(pulse)
            extra = None
        else:
            field = torch.tensor([[0.0, 0.3, 0.7, -0.2, -0.1]], dtype=dtype)
            extra = {"cell": (field, pulse)}
        network = dn.Network({"cell": cell})
        network.eval()
        network.initialize(DT)
        network.run(steps=3, extra=extra)
        return network, extra


def _native(network, extra, steps, *, draws=None):
    recorder = dn.Recorder(("v",))
    network.run(
        steps=steps,
        extra=extra,
        callbacks=[dn.callbacks.PopulationCallback("cell", recorder)],
        **({"draws": draws} if draws is not None else {}),
    )
    return torch.from_numpy(recorder.numpy("v")).clone()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("execution", ["scan", "while"])
@pytest.mark.parametrize("mode", ["intra", "bound_extra", "runtime_extra"])
def test_network_banks_match_native_invocation_grid_with_chunks_and_tail(
    dtype,
    execution,
    mode,
):
    native, native_extra = _network(dtype, mode)
    source, extra = _network(dtype, mode)
    expected = _native(native, native_extra, STEPS)
    runtime = mode == "runtime_extra"
    program = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        chunk_steps=7,
        execution=execution,
        extra={"cell": dn.VarInput()} if runtime else extra,
        callbacks={"trace": dn.Recorder(("cell.v", "t"))},
        backend="eager",
    )
    initial = source.cell.v.clone()
    state, auxiliary = program(**({"extra": extra} if runtime else {}))
    actual = auxiliary["callbacks"]["trace"]["cell.v"]
    tolerance = 3e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(source.cell.v, initial, rtol=0, atol=0)
    torch.testing.assert_close(state["clock"]["t"], native.t, rtol=0, atol=0)
    assert actual.shape[0] == STEPS + 1


def test_network_continuation_restarts_only_the_stimulation_invocation_origin():
    native, _ = _network(torch.float64, "intra")
    source, _ = _network(torch.float64, "intra")
    program = dn.compile(
        source,
        dt=DT,
        steps=9,
        chunk_steps=4,
        callbacks={"trace": dn.Recorder(("cell.v", "t"))},
        backend="eager",
    )
    expected_first = _native(native, None, 9)
    _, first = program()
    torch.testing.assert_close(
        first["callbacks"]["trace"]["cell.v"],
        expected_first,
        rtol=1e-12,
        atol=1e-12,
    )
    continuation = program.restore_continuation(
        program.export_continuation(first["continuation"])
    )
    expected_second = _native(native, None, 7)
    state, second = program(continuation, steps=7)
    torch.testing.assert_close(
        second["callbacks"]["trace"]["cell.v"],
        expected_second[1:],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(state["clock"]["t"], native.t, rtol=0, atol=0)


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_network_explicit_poisson_replay_keeps_mixed_bank_accepted_clock(execution):
    native, _ = _network(torch.float64, "intra")
    source, _ = _network(torch.float64, "intra")
    pulses = []
    for network in (native, source):
        with dn.ctx(JIT=0, DTYPE=torch.float64, REQUIRE_GRAD=1, IMEM=False):
            # Zero amplitude isolates the deterministic pulse-edge assertion,
            # while this second bank still consumes genuine invocation RNG.
            pulse = dn.mono_rect(amp=0.0, pw=0.04).poisson(
                interval=0.06,
                n=5,
                start=-0.03,
                noise=0.7,
                randomize_every_call=True,
                generator=torch.Generator().manual_seed(17),
            )
            network.cell[..., -1].inject(pulse)
            pulses.append(pulse)
    program = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        chunk_steps=7,
        execution=execution,
        callbacks={"trace": dn.Recorder(("cell.v", "t"))},
        backend="eager",
    )
    source_rng = pulses[1].generator.get_state().clone()
    draws = program.functional.sample_runtime_draws(program.initial_state, steps=STEPS)
    expected = _native(native, None, STEPS, draws=draws)
    state, auxiliary = program(draws=draws)
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"]["cell.v"],
        expected,
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
    assert any(torch.equal(value, native_rng) for value in rng_frames)
    torch.testing.assert_close(
        pulses[1].generator.get_state(), source_rng, rtol=0, atol=0
    )
