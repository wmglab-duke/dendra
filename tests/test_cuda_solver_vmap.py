"""Transform batching contracts for Dendra's CUDA linear solvers."""

from __future__ import annotations

import warnings

import pytest
import torch

from dendra.models.integrators.triton import (
    TRITON_AVAILABLE,
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


def _tridiagonal_inputs(*, requires_grad: bool):
    device = torch.device("cuda")
    dtype = torch.float64
    batch, size = 2, 7
    lower = torch.linspace(
        -0.12, 0.08, batch * (size - 1), device=device, dtype=dtype
    ).reshape(batch, size - 1)
    upper = torch.linspace(
        0.09, -0.07, batch * (size - 1), device=device, dtype=dtype
    ).reshape(batch, size - 1)
    main = torch.full((batch, size), 2.5, device=device, dtype=dtype)
    rhs = torch.linspace(-1.1, 1.4, batch * size, device=device, dtype=dtype).reshape(
        batch, size
    )
    return tuple(
        value.requires_grad_(requires_grad) for value in (lower, main, upper, rhs)
    )


def _block_inputs(*, spd: bool, requires_grad: bool):
    device = torch.device("cuda")
    dtype = torch.float64
    batch, size = 2, 5
    band = torch.linspace(
        -0.045,
        0.055,
        batch * (size - 1) * 3,
        device=device,
        dtype=dtype,
    ).reshape(batch, size - 1, 3)
    lower = band.clone()
    upper = band.clone() if spd else -0.7 * band.flip(-1)

    seed = torch.linspace(
        -0.14,
        0.18,
        batch * size * 9,
        device=device,
        dtype=dtype,
    ).reshape(batch, size, 3, 3)
    if spd:
        main = seed @ seed.transpose(-1, -2)
        main = main + 3.0 * torch.eye(3, device=device, dtype=dtype)
    else:
        main = seed.clone()
        main.diagonal(dim1=-2, dim2=-1).add_(3.0)
    rhs = torch.linspace(
        -1.3,
        1.7,
        batch * size * 3,
        device=device,
        dtype=dtype,
    ).reshape(batch, size, 3)
    return tuple(
        value.requires_grad_(requires_grad) for value in (lower, main, upper, rhs)
    )


def _solver_case(name: str, *, requires_grad: bool):
    if name == "thomas":
        return thomas_solve_cuda_t, _tridiagonal_inputs(requires_grad=requires_grad)
    if name == "pcr":
        return pcr_solve_cuda_t, _tridiagonal_inputs(requires_grad=requires_grad)
    if name == "block":
        return thomas_solve_cuda_bt, _block_inputs(
            spd=False, requires_grad=requires_grad
        )
    if name == "block_spd":
        return solve_bt_spd_cuda, _block_inputs(spd=True, requires_grad=requires_grad)
    raise AssertionError(f"unknown solver case {name!r}")


def _without_vmap_fallback_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch._C._debug_only_display_vmap_fallback_warnings(True)
        try:
            result = call()
            torch.cuda.synchronize()
        finally:
            torch._C._debug_only_display_vmap_fallback_warnings(False)

    fallback = [
        warning
        for warning in caught
        if "batching rule" in str(warning.message).lower()
        or "vmap fallback" in str(warning.message).lower()
    ]
    assert not fallback, "\n".join(str(warning.message) for warning in fallback)
    return result


def test_block_cuda_solver_vmap_flattens_transform_and_solver_batches():
    solver, coefficients = _solver_case("block", requires_grad=False)
    transform_batch = 3
    rhs = coefficients[-1]
    rhs_batch = torch.stack([rhs + 0.13 * offset for offset in range(transform_batch)])

    actual = _without_vmap_fallback_warnings(
        lambda: torch.vmap(solver, in_dims=(None, None, None, 0))(
            *coefficients[:-1], rhs_batch
        )
    )
    expected = torch.stack(
        [solver(*coefficients[:-1], rhs_value) for rhs_value in rhs_batch]
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize("name", ["thomas", "pcr", "block", "block_spd"])
def test_cuda_solver_vmap_vjp_matches_explicit_vjp_rows(name):
    solver, inputs = _solver_case(name, requires_grad=True)
    output = solver(*inputs)
    transform_batch = 3
    seeds = torch.linspace(
        -0.9,
        1.2,
        transform_batch * output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape(transform_batch, *output.shape)

    def vjp(seed):
        return torch.autograd.grad(output, inputs, grad_outputs=seed, retain_graph=True)

    actual = _without_vmap_fallback_warnings(lambda: torch.vmap(vjp)(seeds))
    expected_rows = [
        torch.autograd.grad(
            output,
            inputs,
            grad_outputs=seed,
            retain_graph=True,
        )
        for seed in seeds
    ]
    expected = tuple(
        torch.stack([row[input_index] for row in expected_rows])
        for input_index in range(len(inputs))
    )

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=4.0e-10, atol=4.0e-11)


@pytest.mark.parametrize("name", ["thomas", "pcr", "block", "block_spd"])
def test_cuda_solver_vmap_vjp_accepts_an_empty_cotangent_batch(name):
    solver, inputs = _solver_case(name, requires_grad=True)
    output = solver(*inputs)
    seeds = output.new_empty((0, *output.shape))

    def vjp(seed):
        return torch.autograd.grad(
            output,
            inputs,
            grad_outputs=seed,
            retain_graph=True,
        )

    actual = _without_vmap_fallback_warnings(lambda: torch.vmap(vjp)(seeds))
    for gradient, input_tensor in zip(actual, inputs, strict=True):
        assert gradient.shape == (0, *input_tensor.shape)
        assert gradient.numel() == 0


@pytest.mark.parametrize("name", ["thomas", "pcr", "block", "block_spd"])
def test_cuda_solver_legacy_batched_vjp_remains_correct(name):
    solver, inputs = _solver_case(name, requires_grad=True)
    output = solver(*inputs)
    transform_batch = 3
    seeds = torch.linspace(
        -0.8,
        1.1,
        transform_batch * output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape(transform_batch, *output.shape)

    # ``is_grads_batched`` still uses PyTorch's legacy batching key. Custom
    # dispatcher ops keep this path correct through its per-row fallback; the
    # modern ``torch.vmap(vjp)`` path above is the fused, warning-free API.
    actual = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=seeds,
        is_grads_batched=True,
        retain_graph=True,
    )
    expected_rows = [
        torch.autograd.grad(
            output,
            inputs,
            grad_outputs=seed,
            retain_graph=True,
        )
        for seed in seeds
    ]
    expected = tuple(
        torch.stack([row[input_index] for row in expected_rows])
        for input_index in range(len(inputs))
    )

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=4.0e-10, atol=4.0e-11)
