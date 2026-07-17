"""Independent CPU/GPU and dense-oracle checks for CUDA solver kernels."""

from __future__ import annotations

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import (
    DENDRA_SOLVERS_AVAILABLE,
    build_dhs_layers,
    build_morphology,
)
from dendra.models.integrators.triton import (
    TRITON_AVAILABLE,
    dhs_bt_solve_cuda,
    dhs_solve_cuda,
    pcr_solve_cuda_t,
    solve_bt_spd_cuda,
    thomas_solve_cuda_bt,
    thomas_solve_cuda_t,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not TRITON_AVAILABLE,
        reason="CUDA and Triton are required",
    ),
]

_PARENT = torch.tensor([-1, 0, 0, 1, 1, 2, 5], dtype=torch.int64)


def _tree_inputs(dtype: torch.dtype, *, batch: int = 5):
    size = int(_PARENT.numel())
    d_mem = torch.linspace(1.3, 2.6, batch * size, dtype=dtype).reshape(batch, size)
    a_geom = torch.linspace(0.015, 0.095, batch * size, dtype=dtype).reshape(
        batch, size
    )
    a_geom[:, 0] = 0.0
    rhs = torch.linspace(-1.7, 2.1, batch * size, dtype=dtype).reshape(batch, size)
    return d_mem, a_geom, rhs


def _dense_tree_solve(d_mem, a_geom, rhs):
    matrix = torch.diag_embed(d_mem + a_geom)
    for child, parent in enumerate(_PARENT.tolist()):
        if parent < 0:
            continue
        edge = a_geom[:, child]
        matrix[:, parent, parent] += edge
        matrix[:, child, parent] -= edge
        matrix[:, parent, child] -= edge
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)


def _dhs_plan(threads: int, device: torch.device | str):
    _, _, depth = build_morphology(_PARENT.tolist())
    order, layer_ptr = build_dhs_layers(depth, threads)
    return (
        _PARENT.to(device=device),
        order.to(device=device, dtype=torch.int64),
        layer_ptr.to(device=device, dtype=torch.int64),
    )


@pytest.mark.parametrize("threads", [1, 8, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_dhs_matches_cpu_backend_and_dense_oracle(dtype, threads):
    d_mem, a_geom, rhs = _tree_inputs(dtype)
    cpu_result = None
    if DENDRA_SOLVERS_AVAILABLE:
        parent_cpu, order_cpu, layer_ptr_cpu = _dhs_plan(threads, "cpu")
        cpu_result = torch.ops.dendra_solvers.dhs_solve(
            d_mem.clone(),
            a_geom.clone(),
            rhs.clone(),
            parent_cpu,
            order_cpu,
            layer_ptr_cpu,
        )

    device = torch.device("cuda")
    parent_gpu, order_gpu, layer_ptr_gpu = _dhs_plan(threads, device)
    gpu_result = dhs_solve_cuda(
        d_mem.to(device),
        a_geom.to(device),
        rhs.to(device),
        parent_gpu,
        order_gpu,
        layer_ptr_gpu,
        threads=threads,
    )
    dense_result = _dense_tree_solve(
        d_mem.to(device), a_geom.to(device), rhs.to(device)
    )

    if dtype == torch.float32:
        rtol, atol = 5.0e-5, 5.0e-6
    else:
        rtol, atol = 3.0e-12, 3.0e-12
    if cpu_result is not None:
        torch.testing.assert_close(gpu_result.cpu(), cpu_result, rtol=rtol, atol=atol)
    torch.testing.assert_close(gpu_result, dense_result, rtol=rtol, atol=atol)


def test_scalar_dhs_cuda_gradients_match_dense_oracle():
    device = torch.device("cuda")
    d_mem, a_geom, rhs = [
        value.to(device).requires_grad_()
        for value in _tree_inputs(torch.float64, batch=3)
    ]
    parent, order, layer_ptr = _dhs_plan(8, device)

    actual = dhs_solve_cuda(
        d_mem,
        a_geom,
        rhs,
        parent,
        order,
        layer_ptr,
        threads=8,
    )
    expected = _dense_tree_solve(d_mem, a_geom, rhs)
    weights = torch.linspace(0.4, 1.6, actual.numel(), device=device).reshape_as(actual)
    actual_grad = torch.autograd.grad(
        (actual.square() * weights).sum(),
        (d_mem, a_geom, rhs),
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        (expected.square() * weights).sum(), (d_mem, a_geom, rhs)
    )

    torch.testing.assert_close(actual, expected, rtol=3.0e-12, atol=3.0e-12)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=3.0e-10, atol=3.0e-10)


def _dense_block_solve(lower, main, upper, rhs):
    batch, size, block, _ = main.shape
    dense = torch.zeros(
        (batch, size * block, size * block),
        dtype=main.dtype,
        device=main.device,
    )
    for node in range(size):
        current = slice(node * block, (node + 1) * block)
        dense[:, current, current] = main[:, node]
        if node == size - 1:
            continue
        following = slice((node + 1) * block, (node + 2) * block)
        dense[:, following, current] = torch.diag_embed(lower[:, node])
        dense[:, current, following] = torch.diag_embed(upper[:, node])
    return torch.linalg.solve(dense, rhs.reshape(batch, -1, 1)).reshape_as(rhs)


def _spd_inputs(dtype: torch.dtype, size: int, *, noncontiguous: bool):
    device = torch.device("cuda")
    batch = 3
    seed = torch.linspace(
        -0.18,
        0.22,
        batch * size * 9,
        dtype=dtype,
        device=device,
    ).reshape(batch, size, 3, 3)
    main_value = seed @ seed.transpose(-1, -2)
    main_value = main_value + 3.0 * torch.eye(3, dtype=dtype, device=device)

    band_size = batch * max(size - 1, 0) * 3
    bands = torch.linspace(
        -0.045,
        0.055,
        band_size,
        dtype=dtype,
        device=device,
    ).reshape(batch, size - 1, 3)
    lower = bands.clone()
    upper = bands.clone()
    rhs_value = torch.linspace(
        -1.3,
        1.9,
        batch * size * 3,
        dtype=dtype,
        device=device,
    ).reshape(batch, size, 3)

    if noncontiguous:
        main = torch.empty((batch, size, 3, 6), dtype=dtype, device=device)[..., ::2]
        rhs = torch.empty((batch, size, 6), dtype=dtype, device=device)[..., ::2]
        main.copy_(main_value)
        rhs.copy_(rhs_value)
    else:
        main = main_value
        rhs = rhs_value
    return lower, main, upper, rhs


@pytest.mark.parametrize(
    "dtype,size,noncontiguous",
    [
        (torch.float32, 1, False),
        (torch.float32, 5, True),
        (torch.float64, 1, True),
        (torch.float64, 5, False),
    ],
)
def test_spd_block_solver_matches_dense_oracle_and_preserves_inputs(
    dtype, size, noncontiguous
):
    lower, main, upper, rhs = _spd_inputs(dtype, size, noncontiguous=noncontiguous)
    main_before = main.clone()
    rhs_before = rhs.clone()

    actual = solve_bt_spd_cuda(lower, main, upper, rhs)
    expected = _dense_block_solve(lower, main, upper, rhs)

    assert torch.equal(main, main_before)
    assert torch.equal(rhs, rhs_before)
    if dtype == torch.float32:
        rtol, atol = 4.0e-5, 4.0e-6
    else:
        rtol, atol = 4.0e-11, 4.0e-12
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_spd_block_solver_gradients_match_symmetric_dense_parameterization():
    device = torch.device("cuda")
    dtype = torch.float64
    batch, size = 2, 4
    band_seed = (
        torch.linspace(
            -0.035,
            0.045,
            batch * (size - 1) * 3,
            dtype=dtype,
            device=device,
        )
        .reshape(batch, size - 1, 3)
        .requires_grad_()
    )
    main_seed = (
        torch.linspace(
            -0.12,
            0.16,
            batch * size * 9,
            dtype=dtype,
            device=device,
        )
        .reshape(batch, size, 3, 3)
        .requires_grad_()
    )
    rhs = (
        torch.linspace(
            -1.1,
            1.5,
            batch * size * 3,
            dtype=dtype,
            device=device,
        )
        .reshape(batch, size, 3)
        .requires_grad_()
    )
    main = 0.5 * (main_seed + main_seed.transpose(-1, -2))
    main = main + 3.5 * torch.eye(3, dtype=dtype, device=device)

    actual = solve_bt_spd_cuda(band_seed, main, band_seed, rhs)
    expected = _dense_block_solve(band_seed, main, band_seed, rhs)
    weights = torch.linspace(0.5, 1.4, actual.numel(), device=device).reshape_as(actual)
    actual_grad = torch.autograd.grad(
        (actual.square() * weights).sum(),
        (band_seed, main_seed, rhs),
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        (expected.square() * weights).sum(), (band_seed, main_seed, rhs)
    )

    torch.testing.assert_close(actual, expected, rtol=4.0e-11, atol=4.0e-12)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=8.0e-10, atol=8.0e-11)


def _compile_solver_case(name: str, dtype: torch.dtype):
    device = torch.device("cuda")
    batch, size = 2, 7
    if name == "block_tree":
        seed = torch.linspace(
            -0.09,
            0.12,
            batch * size * 9,
            dtype=dtype,
            device=device,
        ).reshape(batch, size, 3, 3)
        main = seed.clone()
        main.diagonal(dim1=-2, dim2=-1).copy_(2.7 + seed.abs().sum(-1))
        edge = torch.linspace(
            0.01,
            0.16,
            batch * size * 3,
            dtype=dtype,
            device=device,
        ).reshape(batch, size, 3)
        edge[:, 0] = 0.0
        rhs = torch.linspace(
            -1.2,
            1.6,
            batch * size * 3,
            dtype=dtype,
            device=device,
        ).reshape(batch, size, 3)
        parent, order, layer_ptr = _dhs_plan(8, device)

        def solve(diagonal, axial, source):
            return dhs_bt_solve_cuda(
                diagonal,
                axial,
                source,
                parent,
                order,
                layer_ptr,
                threads=8,
            )

        return solve, (main, edge, rhs)

    if name in ("thomas", "pcr"):
        lower = torch.linspace(
            0.02, 0.08, batch * (size - 1), dtype=dtype, device=device
        ).reshape(batch, size - 1)
        upper = torch.linspace(
            0.03, 0.09, batch * (size - 1), dtype=dtype, device=device
        ).reshape(batch, size - 1)
        main = torch.full((batch, size), 3.0, dtype=dtype, device=device)
        rhs = torch.linspace(
            -1.0, 1.0, batch * size, dtype=dtype, device=device
        ).reshape(batch, size)
        if name == "thomas":

            def solve(a, b, c, d):
                return thomas_solve_cuda_t(a, b, c, d)

        else:

            def solve(a, b, c, d):
                return pcr_solve_cuda_t(a, b, c, d)

        return solve, (lower, main, upper, rhs)

    if name == "dhs":
        d_mem, a_geom, rhs = [
            value.to(device) for value in _tree_inputs(dtype, batch=batch)
        ]
        parent, order, layer_ptr = _dhs_plan(8, device)

        def solve(diagonal, axial, source):
            return dhs_solve_cuda(
                diagonal,
                axial,
                source,
                parent,
                order,
                layer_ptr,
                threads=8,
            )

        return solve, (d_mem, a_geom, rhs)

    lower, main, upper, rhs = _spd_inputs(dtype, size, noncontiguous=False)
    if name == "block_thomas":

        def solve(lower_arg, main_arg, upper_arg, rhs_arg):
            return thomas_solve_cuda_bt(lower_arg, main_arg, upper_arg, rhs_arg)

    elif name == "block_spd":

        def solve(lower_arg, main_arg, upper_arg, rhs_arg):
            return solve_bt_spd_cuda(lower_arg, main_arg, upper_arg, rhs_arg)

    else:
        raise AssertionError(f"unknown solver {name!r}")
    return solve, (lower, main, upper, rhs)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "solver",
    ["thomas", "pcr", "dhs", "block_tree", "block_thomas", "block_spd"],
)
def test_cuda_solver_fullgraph_compile_matches_eager_and_repeats(solver, dtype):
    solve, inputs = _compile_solver_case(solver, dtype)
    with torch_compiler_warning_context():
        eager = solve(*inputs)
        compiled = torch.compile(solve, backend="inductor", fullgraph=True)
        first = compiled(*inputs)
        second = compiled(*inputs)

    torch.cuda.synchronize()
    torch.testing.assert_close(first, eager, rtol=0.0, atol=0.0)
    assert torch.equal(first, second)


@pytest.mark.slow
@pytest.mark.parametrize(
    "solver",
    ["thomas", "pcr", "dhs", "block_tree", "block_thomas", "block_spd"],
)
def test_cuda_solver_fullgraph_compile_gradients_match_eager(solver):
    solve, inputs = _compile_solver_case(solver, torch.float64)
    eager_inputs = tuple(
        value.detach().clone().requires_grad_(torch.is_floating_point(value))
        for value in inputs
    )
    compiled_inputs = tuple(
        value.detach().clone().requires_grad_(torch.is_floating_point(value))
        for value in inputs
    )
    with torch_compiler_warning_context():
        eager = solve(*eager_inputs)
        compiled = torch.compile(solve, backend="inductor", fullgraph=True)
        actual = compiled(*compiled_inputs)
        weights = torch.linspace(
            0.5, 1.3, eager.numel(), dtype=eager.dtype, device=eager.device
        ).reshape_as(eager)
        eager_grad = torch.autograd.grad(
            (eager.square() * weights).sum(),
            [value for value in eager_inputs if value.requires_grad],
        )
        actual_grad = torch.autograd.grad(
            (actual.square() * weights).sum(),
            [value for value in compiled_inputs if value.requires_grad],
        )

    torch.testing.assert_close(actual, eager, rtol=0.0, atol=0.0)
    for got, want in zip(actual_grad, eager_grad):
        torch.testing.assert_close(got, want, rtol=2.0e-12, atol=2.0e-12)
