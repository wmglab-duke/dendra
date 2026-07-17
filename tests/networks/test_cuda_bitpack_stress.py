"""Seeded stress, stream, and boundary checks for GPU bitpack kernels."""

from __future__ import annotations

import pytest
import torch

from dendra.models.networks import netcon_bitpack_ops as native_ops
from dendra.models.networks import netcon_bitpack_ops_triton as triton_ops

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.stochastic,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

BITS_PER_WORD = 63


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
    words = [0] * ((spikes.numel() + BITS_PER_WORD - 1) // BITS_PER_WORD)
    for source in torch.nonzero(spikes, as_tuple=False).flatten().tolist():
        words[source // BITS_PER_WORD] |= 1 << (source % BITS_PER_WORD)
    return torch.tensor(words, dtype=torch.int64)


def _seeded_large_delivery_case(seed: int, dtype: torch.dtype):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    depth, n_source, n_conn, n_post = 11, 191, 4097, 17
    history_cpu = torch.stack(
        [
            _pack_words(torch.rand(n_source, generator=generator) < (0.18 + 0.02 * row))
            for row in range(depth)
        ]
    )
    source = torch.randint(n_source, (n_conn,), dtype=torch.int64, generator=generator)
    delay_steps = torch.randint(
        1, depth, (n_conn,), dtype=torch.int64, generator=generator
    )
    word_idx = source // BITS_PER_WORD
    bit_mask = torch.tensor(
        [1 << int(bit) for bit in source % BITS_PER_WORD],
        dtype=torch.int64,
    )
    post_idx = torch.randint(n_post, (n_conn,), dtype=torch.int64, generator=generator)
    weights = (
        torch.randn(n_conn, dtype=torch.float64, generator=generator)
        * torch.logspace(-2.0, 1.0, n_conn, dtype=torch.float64)
    ).to(dtype)
    current = torch.tensor([seed % depth], dtype=torch.int64)

    rows = (current - delay_steps).remainder(depth)
    active = torch.bitwise_and(history_cpu[rows, word_idx], bit_mask) != 0
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
    actual = torch.zeros(n_post, dtype=dtype, device="cuda")
    return (*cuda, actual, expected)


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seeded_large_collision_delivery_matches_cpu_oracle(backend, dtype):
    ops = _backend_ops(backend)
    for seed in (701, 709, 719):
        *inputs, actual, expected = _seeded_large_delivery_case(seed, dtype)
        for _ in range(4):
            actual.zero_()
            ops.build_delivery(*inputs, actual)
            torch.cuda.synchronize()
            assert torch.isfinite(actual).all()
            if dtype == torch.float32:
                rtol, atol = 4.0e-5, 5.0e-5
            else:
                rtol, atol = 4.0e-12, 4.0e-12
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_pack_source_spikes_honors_nondefault_stream(backend):
    ops = _backend_ops(backend)
    source = torch.arange(127)
    patterns = (
        ((source * 3 + 5) % 11 == 0),
        ((source * 7 + 2) % 13 == 0),
    )
    histories = [
        torch.full((5, 3), -1, dtype=torch.int64, device="cuda") for _ in patterns
    ]
    currents = [
        torch.tensor([slot], dtype=torch.int64, device="cuda") for slot in (1, 4)
    ]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    default_stream = torch.cuda.current_stream()

    for stream, pattern, history, current in zip(
        streams, patterns, histories, currents
    ):
        stream.wait_stream(default_stream)
        with torch.cuda.stream(stream):
            ops.pack_source_spikes(pattern.cuda(), history, current)
    torch.cuda.synchronize()

    for pattern, history, current in zip(patterns, histories, currents):
        expected = torch.full((5, 3), -1, dtype=torch.int64)
        expected[int(current.item())] = _pack_words(pattern)
        assert torch.equal(history.cpu(), expected)


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_delivery_honors_nondefault_stream(backend):
    ops = _backend_ops(backend)
    cases = (
        _seeded_large_delivery_case(743, torch.float64),
        _seeded_large_delivery_case(751, torch.float64),
    )
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    default_stream = torch.cuda.current_stream()

    for stream, case in zip(streams, cases):
        *inputs, actual, _expected = case
        stream.wait_stream(default_stream)
        with torch.cuda.stream(stream):
            ops.build_delivery(*inputs, actual)
    torch.cuda.synchronize()

    for case in cases:
        *_inputs, actual, expected = case
        torch.testing.assert_close(actual.cpu(), expected, rtol=4.0e-12, atol=4.0e-12)


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_empty_delivery_is_a_noop(backend):
    ops = _backend_ops(backend)
    history = torch.zeros((3, 2), dtype=torch.int64, device="cuda")
    current = torch.tensor([1], dtype=torch.int64, device="cuda")
    empty_index = torch.empty(0, dtype=torch.int64, device="cuda")
    empty_weight = torch.empty(0, dtype=torch.float64, device="cuda")
    actual = torch.linspace(1.0, 2.0, 5, dtype=torch.float64, device="cuda")
    expected = actual.clone()

    ops.build_delivery(
        history,
        current,
        empty_index,
        empty_index,
        empty_index,
        empty_index,
        empty_weight,
        actual,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


def _uniform_case(delay_step: int):
    depth, n_source, n_conn, n_post = 7, 130, 521, 23
    history_cpu = torch.stack(
        [
            _pack_words((torch.arange(n_source) * 3 + row * 5) % 11 == 0)
            for row in range(depth)
        ]
    )
    connection = torch.arange(n_conn, dtype=torch.int64)
    source = (connection * 17 + 5) % n_source
    word_idx = source // BITS_PER_WORD
    bit_mask = torch.tensor(
        [1 << int(bit) for bit in source % BITS_PER_WORD],
        dtype=torch.int64,
    )
    post_idx = (connection * 7 + connection // 3) % n_post
    weights = torch.linspace(-2.25, 3.5, n_conn, dtype=torch.float64)
    current = torch.tensor([2], dtype=torch.int64)
    rows = (current - delay_step).remainder(depth)
    active = torch.bitwise_and(history_cpu[rows, word_idx], bit_mask) != 0
    expected = torch.zeros(n_post, dtype=torch.float64)
    expected.index_add_(0, post_idx, weights * active.to(torch.float64))
    cuda = tuple(
        tensor.cuda()
        for tensor in (
            history_cpu,
            current,
            word_idx,
            bit_mask,
            post_idx,
            weights,
        )
    )
    return (*cuda, torch.zeros(n_post, dtype=torch.float64, device="cuda"), expected)


@pytest.mark.parametrize("delay_step", [0, 1, 7, 10])
def test_native_uniform_delay_normalization_boundaries(delay_step):
    ops = _backend_ops("native")
    depth = 7
    effective = max(1, delay_step)
    if effective >= depth:
        effective %= depth
    (
        history,
        current,
        word_idx,
        bit_mask,
        post_idx,
        weights,
        actual,
        expected,
    ) = _uniform_case(effective)
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
    torch.testing.assert_close(actual.cpu(), expected, rtol=2.0e-13, atol=2.0e-13)
