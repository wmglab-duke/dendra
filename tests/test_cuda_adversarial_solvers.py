"""Seeded adversarial dense-oracle checks across CUDA solver boundaries."""

from __future__ import annotations

import pytest
import torch

from dendra.models.integrators.tree import build_dhs_layers, build_morphology
from dendra.models.integrators.triton import (
    TRITON_AVAILABLE,
    dhs_bt_solve_cuda,
    dhs_solve_cuda,
    pcr_solve_cuda_t,
    solve_bt_spd_cuda,
    solve_bt_spd_cuda_consume,
    thomas_solve_cuda_bt,
    thomas_solve_cuda_t,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.stochastic,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not TRITON_AVAILABLE,
        reason="CUDA and Triton are required",
    ),
]


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _randn(shape, generator):
    return torch.randn(shape, dtype=torch.float64, generator=generator)


def _strided_cuda(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = value.to(device="cuda", dtype=dtype)
    if value.shape[-1] == 0:
        return value
    storage = torch.empty(
        (*value.shape[:-1], 2 * value.shape[-1]),
        dtype=dtype,
        device="cuda",
    )
    view = storage[..., ::2]
    view.copy_(value)
    assert not view.is_contiguous()
    return view


def _cuda_value(
    value: torch.Tensor, dtype: torch.dtype, *, noncontiguous: bool
) -> torch.Tensor:
    if noncontiguous:
        return _strided_cuda(value, dtype)
    return value.to(device="cuda", dtype=dtype)


def _dense_tridiagonal(lower, main, upper):
    matrix = torch.diag_embed(main)
    matrix.diagonal(offset=-1, dim1=-2, dim2=-1).copy_(lower)
    matrix.diagonal(offset=1, dim1=-2, dim2=-1).copy_(upper)
    return matrix


def _dense_block_tridiagonal(lower, main, upper):
    batch, size, block, _ = main.shape
    matrix = torch.zeros((batch, size * block, size * block), dtype=main.dtype)
    for node in range(size):
        current = slice(block * node, block * (node + 1))
        matrix[:, current, current] = main[:, node]
        if node == size - 1:
            continue
        following = slice(block * (node + 1), block * (node + 2))
        matrix[:, following, current] = torch.diag_embed(lower[:, node])
        matrix[:, current, following] = torch.diag_embed(upper[:, node])
    return matrix


def _dense_tree(main, edge, parent):
    matrix = torch.diag_embed(main + edge)
    for child, parent_node in enumerate(parent.tolist()):
        if parent_node < 0:
            continue
        conductance = edge[:, child]
        matrix[:, parent_node, parent_node] += conductance
        matrix[:, child, parent_node] -= conductance
        matrix[:, parent_node, child] -= conductance
    return matrix


def _dense_block_tree(main, edge, parent):
    batch, size, block, _ = main.shape
    matrix = torch.zeros(
        (batch, size * block, size * block), dtype=main.dtype, device=main.device
    )
    for node in range(size):
        current = slice(block * node, block * (node + 1))
        matrix[:, current, current] = main[:, node] + torch.diag_embed(edge[:, node])
        parent_node = int(parent[node])
        if parent_node < 0:
            continue
        parent_slice = slice(block * parent_node, block * (parent_node + 1))
        coupling = torch.diag_embed(edge[:, node])
        matrix[:, parent_slice, parent_slice] += coupling
        matrix[:, current, parent_slice] -= coupling
        matrix[:, parent_slice, current] -= coupling
    return matrix


def _assert_dense_solution(actual, matrix, rhs, dtype, *, label):
    expected = torch.linalg.solve(
        matrix.to(dtype=torch.float64),
        rhs.to(dtype=torch.float64).unsqueeze(-1),
    ).squeeze(-1)
    actual_cpu = actual.detach().cpu().to(dtype=torch.float64)
    if dtype == torch.float32:
        rtol, atol, residual_limit = 4.0e-4, 6.0e-5, 3.0e-5
    else:
        rtol, atol, residual_limit = 5.0e-11, 5.0e-12, 2.0e-12
    torch.testing.assert_close(
        actual_cpu.reshape_as(expected), expected, rtol=rtol, atol=atol
    )

    residual = (
        matrix.to(dtype=torch.float64) @ actual_cpu.reshape_as(rhs).unsqueeze(-1)
    ).squeeze(-1) - rhs.to(dtype=torch.float64)
    scale = (
        matrix.abs().to(dtype=torch.float64)
        @ actual_cpu.reshape_as(rhs).abs().unsqueeze(-1)
    ).squeeze(-1) + rhs.abs().to(dtype=torch.float64)
    backward_error = float((residual.abs() / scale.clamp_min(1.0e-30)).max())
    assert backward_error <= residual_limit, (
        f"{label} backward error {backward_error:.6g} exceeded {residual_limit:.6g}"
    )


_TRIDIAGONAL_CASES = (
    (101, 1, 1, False),
    (103, 3, 2, True),
    (107, 2, 7, False),
    (109, 4, 33, True),
    (113, 2, 65, False),
)


@pytest.mark.parametrize(
    "solver",
    [thomas_solve_cuda_t, pcr_solve_cuda_t],
    ids=["thomas", "pcr"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seeded_tridiagonal_cuda_matches_cpu_dense_across_boundaries(solver, dtype):
    for seed, batch, size, noncontiguous in _TRIDIAGONAL_CASES:
        generator = _generator(seed)
        lower = 0.35 * _randn((batch, size - 1), generator)
        upper = 0.35 * _randn((batch, size - 1), generator)
        row_load = torch.zeros((batch, size), dtype=torch.float64)
        row_load[:, 1:] += lower.abs()
        row_load[:, :-1] += upper.abs()
        main = (
            row_load
            + 0.35
            + 1.75 * torch.rand((batch, size), dtype=torch.float64, generator=generator)
        )
        rhs = _randn((batch, size), generator)

        lower_gpu = _cuda_value(lower, dtype, noncontiguous=noncontiguous)
        main_gpu = _cuda_value(main, dtype, noncontiguous=noncontiguous)
        upper_gpu = _cuda_value(upper, dtype, noncontiguous=noncontiguous)
        rhs_gpu = _cuda_value(rhs, dtype, noncontiguous=noncontiguous)
        before = tuple(
            value.clone() for value in (lower_gpu, main_gpu, upper_gpu, rhs_gpu)
        )

        actual = solver(lower_gpu, main_gpu, upper_gpu, rhs_gpu)
        torch.cuda.synchronize()
        assert all(
            torch.equal(value, original)
            for value, original in zip(
                (lower_gpu, main_gpu, upper_gpu, rhs_gpu), before
            )
        )
        matrix = _dense_tridiagonal(
            lower_gpu.cpu().double(),
            main_gpu.cpu().double(),
            upper_gpu.cpu().double(),
        )
        _assert_dense_solution(
            actual,
            matrix,
            rhs_gpu.cpu().double(),
            dtype,
            label=f"{solver.__name__} seed={seed} B={batch} K={size}",
        )


def _parent_array(size: int, style: str, generator) -> torch.Tensor:
    parent = torch.empty(size, dtype=torch.int64)
    parent[0] = -1
    for node in range(1, size):
        if style == "chain":
            parent[node] = node - 1
        elif style == "star":
            parent[node] = 0
        elif style == "random":
            parent[node] = torch.randint(
                node, (), dtype=torch.int64, generator=generator
            )
        else:
            raise AssertionError(f"unknown topology style {style!r}")
    return parent


_TREE_CASES = (
    (211, 1, 1, 1, "chain"),
    (223, 3, 2, 1, "chain"),
    (227, 5, 17, 2, "star"),
    (229, 33, 33, 4, "star"),
    (233, 5, 19, 8, "random"),
    (239, 3, 47, 16, "random"),
    (241, 2, 65, 32, "random"),
)

_BLOCK_TREE_CASES = (
    (1211, 1, 1, 1, "chain"),
    (1223, 3, 2, 1, "chain"),
    (1227, 5, 9, 2, "star"),
    (1229, 33, 9, 4, "star"),
    (1233, 5, 9, 8, "random"),
    (1239, 3, 9, 16, "random"),
    (1241, 2, 9, 32, "random"),
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seeded_scalar_tree_cuda_matches_cpu_dense_across_warp_plans(dtype):
    for seed, batch, size, threads, style in _TREE_CASES:
        generator = _generator(seed)
        parent = _parent_array(size, style, generator)
        _, _, depth = build_morphology(parent.tolist())
        order, layer_ptr = build_dhs_layers(depth, threads)
        main = 0.5 + 3.0 * torch.rand(
            (batch, size), dtype=torch.float64, generator=generator
        )
        edge = 0.4 * torch.rand((batch, size), dtype=torch.float64, generator=generator)
        edge[:, 0] = 0.0
        rhs = _randn((batch, size), generator)
        noncontiguous = seed % 2 == 1 and size > 1

        main_gpu = _cuda_value(main, dtype, noncontiguous=noncontiguous)
        edge_gpu = _cuda_value(edge, dtype, noncontiguous=noncontiguous)
        rhs_gpu = _cuda_value(rhs, dtype, noncontiguous=noncontiguous)
        actual = dhs_solve_cuda(
            main_gpu,
            edge_gpu,
            rhs_gpu,
            parent.cuda(),
            order.to(device="cuda", dtype=torch.int64),
            layer_ptr.to(device="cuda", dtype=torch.int64),
            threads=threads,
        )
        torch.cuda.synchronize()

        matrix = _dense_tree(
            main_gpu.cpu().double(),
            edge_gpu.cpu().double(),
            parent,
        )
        _assert_dense_solution(
            actual,
            matrix,
            rhs_gpu.cpu().double(),
            dtype,
            label=(
                f"dhs seed={seed} style={style} B={batch} K={size} threads={threads}"
            ),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seeded_block_tree_cuda_matches_cpu_dense_across_warp_plans(dtype):
    for seed, batch, size, threads, style in _BLOCK_TREE_CASES:
        generator = _generator(seed + 1000)
        parent = _parent_array(size, style, generator)
        _, _, depth = build_morphology(parent.tolist())
        order, layer_ptr = build_dhs_layers(depth, threads)
        main = 0.06 * _randn((batch, size, 3, 3), generator)
        diagonal = main.diagonal(dim1=-2, dim2=-1)
        diagonal.copy_(
            2.0
            + main.abs().sum(-1)
            + torch.rand(
                (batch, size, 3),
                dtype=torch.float64,
                generator=generator,
            )
        )
        edge = 0.16 * torch.rand(
            (batch, size, 3), dtype=torch.float64, generator=generator
        )
        edge[:, 0] = 0.0
        rhs = _randn((batch, size, 3), generator)
        noncontiguous = seed % 2 == 1 and size > 1

        main_gpu = _cuda_value(main, dtype, noncontiguous=noncontiguous)
        edge_gpu = _cuda_value(edge, dtype, noncontiguous=noncontiguous)
        rhs_gpu = _cuda_value(rhs, dtype, noncontiguous=noncontiguous)
        actual = dhs_bt_solve_cuda(
            main_gpu,
            edge_gpu,
            rhs_gpu,
            parent.cuda(),
            order.to(device="cuda", dtype=torch.int64),
            layer_ptr.to(device="cuda", dtype=torch.int64),
            threads=threads,
        )
        torch.cuda.synchronize()

        matrix = _dense_block_tree(
            main_gpu.cpu().double(),
            edge_gpu.cpu().double(),
            parent,
        )
        _assert_dense_solution(
            actual.reshape(batch, -1),
            matrix,
            rhs_gpu.cpu().double().reshape(batch, -1),
            dtype,
            label=(
                f"dhs-bt seed={seed} style={style} B={batch} K={size} threads={threads}"
            ),
        )


def test_seeded_block_tree_cuda_gradients_match_dense_sibling_reductions():
    generator = _generator(313)
    batch, size, threads = 2, 9, 4
    parent = _parent_array(size, "random", generator)
    _, _, depth = build_morphology(parent.tolist())
    order, layer_ptr = build_dhs_layers(depth, threads)
    main = 0.04 * _randn((batch, size, 3, 3), generator)
    main.diagonal(dim1=-2, dim2=-1).copy_(2.5 + main.abs().sum(-1))
    edge = 0.12 * torch.rand((batch, size, 3), dtype=torch.float64, generator=generator)
    edge[:, 0] = 0.0
    rhs = _randn((batch, size, 3), generator)

    main_gpu = main.cuda().requires_grad_()
    edge_gpu = edge.cuda().requires_grad_()
    rhs_gpu = rhs.cuda().requires_grad_()
    actual = dhs_bt_solve_cuda(
        main_gpu,
        edge_gpu,
        rhs_gpu,
        parent.cuda(),
        order.to(device="cuda", dtype=torch.int64),
        layer_ptr.to(device="cuda", dtype=torch.int64),
        threads=threads,
    )
    matrix = _dense_block_tree(main_gpu, edge_gpu, parent)
    expected = torch.linalg.solve(matrix, rhs_gpu.reshape(batch, -1, 1)).reshape_as(
        rhs_gpu
    )
    weights = torch.linspace(
        0.4, 1.6, actual.numel(), dtype=torch.float64, device="cuda"
    ).reshape_as(actual)
    actual_grad = torch.autograd.grad(
        (actual.square() * weights).sum(),
        (main_gpu, edge_gpu, rhs_gpu),
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        (expected.square() * weights).sum(),
        (main_gpu, edge_gpu, rhs_gpu),
    )

    torch.testing.assert_close(actual, expected, rtol=4.0e-10, atol=4.0e-11)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=3.0e-8, atol=3.0e-9)


def _block_tridiagonal_case(seed, batch, size):
    generator = _generator(seed)
    lower = 0.08 * _randn((batch, size - 1, 3), generator)
    upper = 0.08 * _randn((batch, size - 1, 3), generator)
    main = 0.05 * _randn((batch, size, 3, 3), generator)
    main.diagonal(dim1=-2, dim2=-1).copy_(2.0 + main.abs().sum(-1))
    rhs = _randn((batch, size, 3), generator)
    return lower, main, upper, rhs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seeded_block_thomas_cuda_matches_cpu_dense_above_fibre_block(dtype):
    for seed, batch, size in ((401, 1, 1), (409, 3, 5), (419, 35, 17)):
        lower, main, upper, rhs = _block_tridiagonal_case(seed, batch, size)
        noncontiguous = size == 5
        lower_gpu = _cuda_value(lower, dtype, noncontiguous=noncontiguous)
        main_gpu = _cuda_value(main, dtype, noncontiguous=noncontiguous)
        upper_gpu = _cuda_value(upper, dtype, noncontiguous=noncontiguous)
        rhs_gpu = _cuda_value(rhs, dtype, noncontiguous=noncontiguous)
        actual = thomas_solve_cuda_bt(lower_gpu, main_gpu, upper_gpu, rhs_gpu)
        torch.cuda.synchronize()
        matrix = _dense_block_tridiagonal(
            lower_gpu.cpu().double(),
            main_gpu.cpu().double(),
            upper_gpu.cpu().double(),
        )
        _assert_dense_solution(
            actual.reshape(batch, -1),
            matrix,
            rhs_gpu.cpu().double().reshape(batch, -1),
            dtype,
            label=f"block-thomas seed={seed} B={batch} K={size}",
        )


def _spd_case(seed, batch, size):
    generator = _generator(seed)
    bands = 0.045 * _randn((batch, size - 1, 3), generator)
    factor = 0.08 * _randn((batch, size, 3, 3), generator)
    main = factor @ factor.transpose(-1, -2)
    band_load = torch.zeros((batch, size, 3), dtype=torch.float64)
    band_load[:, :-1] += bands.abs()
    band_load[:, 1:] += bands.abs()
    diagonal_scale = torch.tensor([0.25, 1.0, 12.0], dtype=torch.float64)
    main.diagonal(dim1=-2, dim2=-1).add_(band_load + diagonal_scale)
    rhs = _randn((batch, size, 3), generator)
    return bands, main, bands.clone(), rhs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seeded_spd_consume_matches_public_and_cpu_dense(dtype):
    batch, size = 35, 7
    lower, main, upper, rhs = _spd_case(503, batch, size)
    lower_gpu = lower.to(device="cuda", dtype=dtype)
    main_gpu = main.to(device="cuda", dtype=dtype)
    upper_gpu = upper.to(device="cuda", dtype=dtype)
    rhs_gpu = rhs.to(device="cuda", dtype=dtype)

    public = solve_bt_spd_cuda(lower_gpu, main_gpu, upper_gpu, rhs_gpu)
    main_work = main_gpu.clone()
    rhs_work = rhs_gpu.clone()
    consumed = solve_bt_spd_cuda_consume(lower_gpu, main_work, upper_gpu, rhs_work)
    torch.cuda.synchronize()

    assert not torch.equal(main_work, main_gpu)
    assert not torch.equal(rhs_work, rhs_gpu)
    torch.testing.assert_close(consumed, public, rtol=0.0, atol=0.0)
    matrix = _dense_block_tridiagonal(
        lower_gpu.cpu().double(),
        main_gpu.cpu().double(),
        upper_gpu.cpu().double(),
    )
    _assert_dense_solution(
        public.reshape(batch, -1),
        matrix,
        rhs_gpu.cpu().double().reshape(batch, -1),
        dtype,
        label=f"block-spd consume B={batch} K={size}",
    )
