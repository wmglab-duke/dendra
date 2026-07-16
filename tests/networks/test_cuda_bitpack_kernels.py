"""Real-GPU correctness oracles for NetCon bitpacked history kernels."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn
from dendra.models.networks import netcon_bitpack_ops as native_ops
from dendra.models.networks import netcon_bitpack_ops_triton as triton_ops

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DT = 0.1
BITS_PER_WORD = 63
WEIGHTS = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
DELAYS = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])


def _backend_ops(name: str):
    if name == "native":
        if not native_ops.is_available():
            pytest.skip(
                "native C++/CUDA bitpack extension unavailable: "
                f"{native_ops.build_info()}"
            )
        return native_ops
    if name == "triton":
        if not triton_ops.is_available():
            pytest.skip(
                f"Triton bitpack kernels unavailable: {triton_ops.last_error()!r}"
            )
        return triton_ops
    raise AssertionError(f"unknown backend {name!r}")


def _pack_words(spikes: torch.Tensor) -> torch.Tensor:
    spikes = spikes.to(device="cpu", dtype=torch.bool).reshape(-1)
    n_words = (spikes.numel() + BITS_PER_WORD - 1) // BITS_PER_WORD
    words = [0] * n_words
    for source in torch.nonzero(spikes, as_tuple=False).flatten().tolist():
        words[source // BITS_PER_WORD] |= 1 << (source % BITS_PER_WORD)
    return torch.tensor(words, dtype=torch.int64)


def _spike_pattern(n_source: int, row: int = 0) -> torch.Tensor:
    source = torch.arange(n_source)
    spikes = ((source * 3 + row * 5) % 11) == 0
    for boundary in (0, 62, 63, 64, 125, 126, n_source - 1):
        if 0 <= boundary < n_source:
            spikes[boundary] = True
    return spikes


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("n_source", [0, 1, 63, 64, 65, 127])
def test_cuda_pack_source_spikes_matches_signed_int64_word_oracle(backend, n_source):
    ops = _backend_ops(backend)
    spikes_cpu = _spike_pattern(n_source)
    n_words = (n_source + BITS_PER_WORD - 1) // BITS_PER_WORD
    slot = 2
    history = torch.full((4, n_words), -1, dtype=torch.int64, device="cuda")
    current = torch.tensor([slot], dtype=torch.int64, device="cuda")

    ops.pack_source_spikes(spikes_cpu.cuda(), history, current)
    torch.cuda.synchronize()

    expected = torch.full((4, n_words), -1, dtype=torch.int64)
    expected[slot] = _pack_words(spikes_cpu)
    assert torch.equal(history.cpu(), expected)


def _delivery_case(dtype: torch.dtype, *, uniform_delay: int | None = None):
    depth, n_source, n_conn, n_post = 7, 130, 521, 23
    history_cpu = torch.stack(
        [_pack_words(_spike_pattern(n_source, row)) for row in range(depth)]
    )
    connection = torch.arange(n_conn, dtype=torch.int64)
    source = (connection * 17 + 5) % n_source
    word_idx = source // BITS_PER_WORD
    bit_mask = torch.tensor(
        [1 << int(bit) for bit in (source % BITS_PER_WORD)], dtype=torch.int64
    )
    post_idx = (connection * 7 + connection // 3) % n_post
    if uniform_delay is None:
        delay_steps = 1 + (connection * 5 + 2) % (depth - 1)
    else:
        delay_steps = torch.full_like(connection, int(uniform_delay))
    weights = torch.linspace(-2.25, 3.5, n_conn, dtype=dtype)
    current = torch.tensor([2], dtype=torch.int64)

    rows = (current - delay_steps).remainder(depth)
    words = history_cpu[rows, word_idx]
    active = torch.bitwise_and(words, bit_mask) != 0
    expected = torch.zeros(n_post, dtype=dtype)
    expected.index_add_(0, post_idx, weights * active.to(dtype))

    cuda = tuple(
        tensor.cuda()
        for tensor in (
            history_cpu,
            current,
            delay_steps,
            word_idx,
            bit_mask,
            post_idx,
            weights,
        )
    )
    return (*cuda, torch.zeros(n_post, dtype=dtype, device="cuda"), expected)


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_mixed_delay_delivery_matches_cpu_collision_oracle(backend, dtype):
    ops = _backend_ops(backend)
    (
        history,
        current,
        delay_steps,
        word_idx,
        bit_mask,
        post_idx,
        weights,
        actual,
        expected,
    ) = _delivery_case(dtype)

    ops.build_delivery(
        history,
        current,
        delay_steps,
        word_idx,
        bit_mask,
        post_idx,
        weights,
        actual,
    )
    torch.cuda.synchronize()

    if dtype == torch.float32:
        rtol, atol = 2.0e-6, 2.0e-6
    else:
        rtol, atol = 2.0e-13, 2.0e-13
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_native_uniform_delay_delivery_matches_cpu_collision_oracle(dtype):
    ops = _backend_ops("native")
    delay_step = 3
    (
        history,
        current,
        _delay_steps,
        word_idx,
        bit_mask,
        post_idx,
        weights,
        actual,
        expected,
    ) = _delivery_case(dtype, uniform_delay=delay_step)

    ops.build_delivery_uniform(
        history,
        current,
        delay_step,
        word_idx,
        bit_mask,
        post_idx,
        weights,
        actual,
    )
    torch.cuda.synchronize()

    if dtype == torch.float32:
        rtol, atol = 2.0e-6, 2.0e-6
    else:
        rtol, atol = 2.0e-13, 2.0e-13
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


def _fan_network(backend: str, dtype: torch.dtype):
    post = dn.Population(N=2, C=1, v_init=-65.0, dtype=dtype, device="cuda")
    post.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(N=3, dtype=dtype, device="cuda")
    net = dn.Network(
        {"post": post},
        netstim=stim,
        track_netcon_events=False,
        netcon_delay_backend=backend,
    )
    net.connect_dense(
        net.netstim[:],
        post[:],
        post.mech.syn,
        threshold=None,
        weight=WEIGHTS.to(dtype=dtype, device="cuda"),
        delay=DELAYS.to(dtype=dtype, device="cuda"),
    )
    net.build(DT)
    net.init_synapses()
    return net, next(iter(net.synapses.values()))


def _run_tape(net, netcon, tape):
    trace = []
    for spikes in tape:
        net.netstim.spikes.copy_(
            torch.as_tensor(spikes, dtype=torch.bool, device="cuda")
        )
        netcon.advance()
        trace.append(netcon.syn.g.detach().flatten().cpu())
    torch.cuda.synchronize()
    return torch.stack(trace)


def _reference_trace(tape, dtype):
    pre_idx = [0, 0, 1, 1, 2, 2]
    post_idx = [0, 1, 0, 1, 0, 1]
    delay_steps = [1, 2, 3, 1, 2, 3]
    weights = WEIGHTS.to(dtype=dtype)
    pending = {}
    state = torch.zeros(2, dtype=dtype)
    trace = []
    for step, spikes in enumerate(tape):
        state = state + pending.pop(step, torch.zeros_like(state))
        trace.append(state.clone())
        for edge, (source, target, delay) in enumerate(
            zip(pre_idx, post_idx, delay_steps)
        ):
            if spikes[source]:
                due = pending.setdefault(step + delay, torch.zeros_like(state))
                due[target] += weights[edge]
    return torch.stack(trace)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_netcon_native_bitpack_matches_dense_and_completes_kernel_calls(
    monkeypatch, dtype
):
    _backend_ops("native")
    tape = (
        (True, True, False),
        (False, False, True),
        (True, False, True),
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (False, False, False),
        (False, False, False),
        (True, True, True),
        (False, False, False),
        (False, False, False),
        (False, False, False),
    )

    dense_net, dense_con = _fan_network("dense", dtype)
    expected = _run_tape(dense_net, dense_con, tape)

    calls = {"pack": 0, "delivery": 0}
    original_pack = native_ops.pack_source_spikes
    original_delivery = native_ops.build_delivery

    def counted_pack(*args, **kwargs):
        result = original_pack(*args, **kwargs)
        calls["pack"] += 1
        return result

    def counted_delivery(*args, **kwargs):
        result = original_delivery(*args, **kwargs)
        calls["delivery"] += 1
        return result

    monkeypatch.setattr(native_ops, "pack_source_spikes", counted_pack)
    monkeypatch.setattr(native_ops, "build_delivery", counted_delivery)

    bitpack_net, bitpack_con = _fan_network("bitpacked_history", dtype)
    actual = _run_tape(bitpack_net, bitpack_con, tape)
    reference = _reference_trace(tape, dtype)

    assert calls == {"pack": len(tape), "delivery": len(tape)}
    assert native_ops.build_info()["loaded"] is True
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
