"""Scalar CUDA DHS precision against independent dense CPU systems."""

from __future__ import annotations

import pytest
import torch

from dendra.models.integrators import _cuda_functional as facade
from tests.func.test_cuda_functional_reference import _tree_matrix

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _solver(implementation, parents):
    from dendra.models.integrators.triton.dhs_kernel import (
        DHSSolveStable,
        dhs_solve_cuda,
    )

    # These chain fixtures have one node per layer, children before parents.
    parent = torch.tensor(parents, dtype=torch.int64, device="cuda")
    order = torch.arange(len(parents) - 1, -1, -1, device="cuda")
    layers = torch.arange(len(parents) + 1, device="cuda")
    if implementation == "single":
        return lambda main, edge, rhs: DHSSolveStable.apply(
            main, edge, rhs, parent, order, layers, 32
        )
    if implementation == "packed":
        return lambda main, edge, rhs: dhs_solve_cuda(
            main, edge, rhs, parent, order, layers, threads=32
        )
    return lambda main, edge, rhs: facade.dhs_solve(
        main, edge, rhs, parent, order, layers, threads=32
    )


def _dense(main, edge, rhs, parents):
    matrix = _tree_matrix(main, edge, parents)
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("implementation", ("single", "packed"))
def test_float32_large_axial_conductance_preserves_uniform_solution(implementation):
    """A passive uniform equilibrium remains finite across a stiff edge."""
    parents = (-1, 0)
    main = torch.ones(1, 2, dtype=torch.float32)
    edge = torch.tensor([[0.0, float(2**26)]], dtype=torch.float32)
    rhs = main.clone()
    expected = _dense(main.double(), edge.double(), rhs.double(), parents)
    torch.testing.assert_close(
        expected, torch.ones_like(expected), rtol=1e-8, atol=1e-8
    )

    actual = _solver(implementation, parents)(
        *(value.cuda() for value in (main, edge, rhs))
    )
    assert actual.dtype == torch.float32
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-6, atol=2e-7)


def _regular_case(signed):
    if signed:
        # A nonsingular indefinite system with a negative internal diagonal.
        # The root's edge coefficient still contributes to its own diagonal.
        main = [[2.0, -0.5, 1.0], [2.4, -0.6, 1.3]]
        edge = [[0.2, 0.2, 0.1], [0.3, 0.15, 0.2]]
    else:
        main = [[1.0, 2.0, 1.5], [1.2, 1.8, 2.4]]
        edge = [[0.2, 0.3, 0.4], [0.15, 0.4, 0.2]]
    rhs = [[0.7, -1.2, 0.3], [-0.2, 0.9, 1.1]]
    return tuple(
        torch.tensor(value, dtype=torch.float32) for value in (main, edge, rhs)
    )


@pytest.mark.parametrize("implementation", ("single", "packed", "facade"))
@pytest.mark.parametrize("signed", (False, True), ids=("spd", "indefinite"))
def test_float32_scalar_schur_preserves_dense_primal_and_gradients(
    implementation, signed
):
    parents = (-1, 0, 1)
    values = _regular_case(signed)
    reference = tuple(value.double().requires_grad_() for value in values)
    expected = _dense(*reference, parents)
    weights = torch.tensor([[0.3, -0.7, 0.9], [-0.6, 0.8, 0.2]], dtype=torch.float64)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), reference)

    arguments = tuple(value.cuda().requires_grad_() for value in values)
    actual = _solver(implementation, parents)(*arguments)
    gradients = torch.autograd.grad((actual * weights.cuda().float()).sum(), arguments)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-5, atol=1e-6)
    for actual_gradient, expected_gradient in zip(
        gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient.cpu().double(), expected_gradient, rtol=2e-5, atol=2e-6
        )
    # The independently trainable root placeholder has the same diagonal
    # derivative as the root's membrane diagonal, rather than being ignored.
    torch.testing.assert_close(gradients[0][:, 0], gradients[1][:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("signed", (False, True), ids=("spd", "indefinite"))
def test_float32_scalar_schur_jvp_matches_dense_cpu(signed):
    parents = (-1, 0, 1)
    values = _regular_case(signed)
    directions = tuple(
        torch.linspace(
            -0.2 + index * 0.03, 0.3 + index * 0.02, value.numel()
        ).reshape_as(value)
        for index, value in enumerate(values)
    )
    expected, expected_tangent = torch.func.jvp(
        lambda *args: _dense(*args, parents),
        tuple(value.double() for value in values),
        tuple(value.double() for value in directions),
    )
    actual, tangent = torch.func.jvp(
        _solver("facade", parents),
        tuple(value.cuda() for value in values),
        tuple(value.cuda() for value in directions),
    )
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        tangent.cpu().double(), expected_tangent, rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize("implementation", ("imperative", "facade"))
def test_float32_multi_schur_preserves_stiff_equilibrium_and_partial_warp(
    implementation,
):
    from dendra.models.integrators.triton.dhs_kernel_multi import dhs_solve_multi_cuda

    # Two stiff two-node neurons fill one warp; a singleton occupies half of
    # the next warp. Poisoned padding must never enter either active system.
    main = torch.tensor(
        [
            [1, 1, float("nan"), float("nan")],
            [1, 1, float("nan"), float("nan")],
            [2, float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    edge = torch.tensor(
        [
            [0, 2**26, float("nan"), float("nan")],
            [0, 2**26, float("nan"), float("nan")],
            [0.25, float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    rhs = torch.tensor(
        [
            [1, 1, float("nan"), float("nan")],
            [0.5, 0.5, float("nan"), float("nan")],
            [3, float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    metadata = tuple(
        torch.tensor(value, dtype=torch.int64, device="cuda")
        for value in (
            (-1, 0, -1),
            (1, 0, 0),
            (0, 1, 2, 0, 1),
            (0, 2),
            (0, 2),
            (0, 3),
            (2, 1),
            (0, 2),
            (2, 1),
        )
    )
    solve = (
        dhs_solve_multi_cuda
        if implementation == "imperative"
        else facade.dhs_multi_solve
    )
    actual = solve(
        *(value.cuda() for value in (main, edge, rhs)),
        *metadata,
        K_stride=4,
        L_max=2,
        threads=16,
        grid_x=2,
    )
    assert actual.dtype == torch.float32
    for row, parents in enumerate(((-1, 0), (-1, 0), (-1,))):
        width = len(parents)
        expected = _dense(
            main[row : row + 1, :width].double(),
            edge[row : row + 1, :width].double(),
            rhs[row : row + 1, :width].double(),
            parents,
        )
        torch.testing.assert_close(
            actual[row : row + 1, :width].cpu().double(), expected, rtol=2e-6, atol=2e-7
        )
        if implementation == "facade":
            torch.testing.assert_close(
                actual[row, width:],
                torch.zeros_like(actual[row, width:]),
                rtol=0,
                atol=0,
            )
