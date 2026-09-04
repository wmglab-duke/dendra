"""Transform-batching contracts for the CUDA tree-family solvers."""

from __future__ import annotations

import warnings

import pytest
import torch

from dendra.models.integrators.triton import (
    TRITON_AVAILABLE,
    dhs_bt_solve_cuda,
    dhs_solve_cuda,
    dhs_solve_multi_cuda,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not TRITON_AVAILABLE,
        reason="CUDA and Triton are required",
    ),
]


def _without_vmap_fallback(call):
    previous = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch._C._debug_only_display_vmap_fallback_warnings(True)
        try:
            result = call()
        finally:
            torch._C._debug_only_display_vmap_fallback_warnings(previous)
    fallback = [
        str(item.message)
        for item in caught
        if "batching rule not implemented" in str(item.message).lower()
        or "performance drop" in str(item.message).lower()
    ]
    assert fallback == []
    return result


def _explicit_vjps(output, inputs, seeds):
    rows = [
        torch.autograd.grad(
            output,
            inputs,
            grad_outputs=seed,
            retain_graph=True,
        )
        for seed in seeds
    ]
    return tuple(torch.stack(parts) for parts in zip(*rows, strict=True))


def _assert_vjps_match(output, inputs, seeds):
    expected = _explicit_vjps(output, inputs, seeds)
    actual = _without_vmap_fallback(
        lambda: torch.vmap(
            lambda seed: torch.autograd.grad(
                output,
                inputs,
                grad_outputs=seed,
                retain_graph=True,
            )
        )(seeds)
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=4.0e-10, atol=4.0e-10)

    # ``is_grads_batched`` uses PyTorch's legacy vmap path. Its dispatcher
    # fallback may warn, but it must remain correct while the modern path above
    # uses the explicit fused batching rules.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = torch.autograd.grad(
            output,
            inputs,
            grad_outputs=seeds,
            is_grads_batched=True,
            retain_graph=True,
        )
    for got, want in zip(legacy, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=4.0e-10, atol=4.0e-10)


def _assert_empty_vjp_batch(output, inputs):
    seeds = output.new_empty((0, *output.shape))

    def vjp(seed):
        return torch.autograd.grad(
            output,
            inputs,
            grad_outputs=seed,
            retain_graph=True,
        )

    actual = _without_vmap_fallback(lambda: torch.vmap(vjp)(seeds))
    for gradient, input_tensor in zip(actual, inputs, strict=True):
        assert gradient.shape == (0, *input_tensor.shape)
        assert gradient.numel() == 0


def _tree_case(*, batch=3):
    device = torch.device("cuda")
    dtype = torch.float64
    parent = torch.tensor([-1, 0, 0, 2], device=device)
    order = torch.tensor([1, 3, 2, 0], device=device)
    layer_ptr = torch.tensor([0, 2, 3, 4], device=device)
    d_mem = torch.linspace(1.5, 2.4, batch * 4, device=device, dtype=dtype).reshape(
        batch, 4
    )
    a_geom = torch.linspace(0.0, 0.16, batch * 4, device=device, dtype=dtype).reshape(
        batch, 4
    )
    a_geom[:, 0] = 0.0
    rhs = torch.linspace(-1.2, 1.7, batch * 4, device=device, dtype=dtype).reshape(
        batch, 4
    )
    return d_mem, a_geom, rhs, parent, order, layer_ptr


@pytest.mark.parametrize("implementation", ["packed", "stable"])
@pytest.mark.parametrize("batch", [1, 3])
def test_scalar_tree_vjp_is_fused_and_correct(implementation, batch):
    from dendra.models.integrators.triton.dhs_kernel import DHSSolveStable

    values = _tree_case(batch=batch)
    d_mem, a_geom, rhs, parent, order, layer_ptr = values
    if implementation == "packed":

        def solve(d, a, r):
            return dhs_solve_cuda(d, a, r, parent, order, layer_ptr, threads=4)

    else:

        def solve(d, a, r):
            return DHSSolveStable.apply(
                d.contiguous(),
                a.contiguous(),
                r.contiguous(),
                parent,
                order,
                layer_ptr,
                4,
            )

    inputs = tuple(value.detach().requires_grad_() for value in (d_mem, a_geom, rhs))
    output = solve(*inputs)
    seeds = torch.linspace(
        -0.8,
        1.1,
        5 * output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape(5, *output.shape)
    _assert_vjps_match(output, inputs, seeds)


def test_tree_vjp_rejects_nested_vmap_before_raw_triton_launch():
    d_mem, a_geom, rhs, parent, order, layer_ptr = _tree_case(batch=1)
    inputs = tuple(value.detach().requires_grad_() for value in (d_mem, a_geom, rhs))
    output = dhs_solve_cuda(
        *inputs,
        parent,
        order,
        layer_ptr,
        threads=4,
    )
    seeds = torch.randn(
        2,
        3,
        *output.shape,
        device=output.device,
        dtype=output.dtype,
    )

    def vjp(seed):
        return torch.autograd.grad(
            output,
            inputs,
            grad_outputs=seed,
            retain_graph=True,
        )

    with pytest.raises(RuntimeError, match="does not support nested vmap"):
        torch.vmap(torch.vmap(vjp))(seeds)


@pytest.mark.parametrize("implementation", ["packed", "stable"])
def test_scalar_tree_vjp_accepts_an_empty_cotangent_batch(implementation):
    from dendra.models.integrators.triton.dhs_kernel import DHSSolveStable

    d_mem, a_geom, rhs, parent, order, layer_ptr = _tree_case(batch=1)
    inputs = tuple(value.detach().requires_grad_() for value in (d_mem, a_geom, rhs))
    if implementation == "packed":
        output = dhs_solve_cuda(
            *inputs,
            parent,
            order,
            layer_ptr,
            threads=4,
        )
    else:
        output = DHSSolveStable.apply(
            *inputs,
            parent,
            order,
            layer_ptr,
            4,
        )
    _assert_empty_vjp_batch(output, inputs)


def _block_tree_case(batch):
    d_mem, a_geom, rhs_scalar, parent, order, layer_ptr = _tree_case(batch=batch)
    eye = torch.eye(3, device=d_mem.device, dtype=d_mem.dtype)
    main = d_mem[..., None, None] * eye
    coupling = torch.tensor(
        [[0.0, 0.025, -0.015], [0.025, 0.0, 0.02], [-0.015, 0.02, 0.0]],
        device=d_mem.device,
        dtype=d_mem.dtype,
    )
    main = main + coupling
    edge = a_geom[..., None] * torch.tensor(
        [0.8, 1.0, 1.2], device=d_mem.device, dtype=d_mem.dtype
    )
    rhs = rhs_scalar[..., None] * torch.tensor(
        [1.0, -0.4, 0.7], device=d_mem.device, dtype=d_mem.dtype
    )
    return main, edge, rhs, parent, order, layer_ptr


@pytest.mark.parametrize("batch", [1, 2])
def test_block_tree_vjp_is_fused_and_correct(batch):
    main, edge, rhs, parent, order, layer_ptr = _block_tree_case(batch)

    def solve(m, e, r):
        return dhs_bt_solve_cuda(m, e, r, parent, order, layer_ptr, threads=4)

    inputs = tuple(value.detach().requires_grad_() for value in (main, edge, rhs))
    output = solve(*inputs)
    seeds = torch.linspace(
        -1.0,
        0.9,
        4 * output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape(4, *output.shape)
    _assert_vjps_match(output, inputs, seeds)


def test_block_tree_vjp_accepts_an_empty_cotangent_batch():
    main, edge, rhs, parent, order, layer_ptr = _block_tree_case(batch=1)
    inputs = tuple(value.detach().requires_grad_() for value in (main, edge, rhs))
    output = dhs_bt_solve_cuda(
        *inputs,
        parent,
        order,
        layer_ptr,
        threads=4,
    )
    _assert_empty_vjp_batch(output, inputs)


def _multi_tree_case(batch):
    device = torch.device("cuda")
    dtype = torch.float64
    width = 4
    d_mem = torch.linspace(1.4, 2.5, batch * width, device=device, dtype=dtype).reshape(
        batch, width
    )
    a_geom = torch.linspace(
        0.0, 0.14, batch * width, device=device, dtype=dtype
    ).reshape(batch, width)
    a_geom[:, 0] = 0.0
    rhs = torch.linspace(-1.3, 1.6, batch * width, device=device, dtype=dtype).reshape(
        batch, width
    )

    if batch == 1:
        topology = (
            torch.tensor([-1, 0, 1, 2], device=device),
            torch.tensor([3, 2, 1, 0], device=device),
            torch.tensor([0, 1, 2, 3, 4], device=device),
        )
        plan = (
            torch.tensor([0], device=device),
            torch.tensor([0], device=device),
            torch.tensor([0], device=device),
            torch.tensor([4], device=device),
            torch.tensor([0], device=device),
            torch.tensor([1], device=device),
        )
    else:
        # Rows 0-1 use a chain; row 2 uses a branched four-node tree.
        topology = (
            torch.tensor([-1, 0, 1, 2, -1, 0, 0, 2], device=device),
            torch.tensor([3, 2, 1, 0, 1, 3, 2, 0], device=device),
            torch.tensor([0, 1, 2, 3, 4, 0, 2, 3, 4], device=device),
        )
        plan = (
            torch.tensor([0, 4], device=device),
            torch.tensor([0, 4], device=device),
            torch.tensor([0, 5], device=device),
            torch.tensor([4, 3], device=device),
            torch.tensor([0, 2], device=device),
            torch.tensor([2, 1], device=device),
        )
    return d_mem, a_geom, rhs, topology, plan


@pytest.mark.parametrize("batch", [1, 3])
def test_multi_tree_vjp_repeats_the_warp_plan_correctly(batch):
    d_mem, a_geom, rhs, topology, plan = _multi_tree_case(batch)

    def solve(d, a, r):
        return dhs_solve_multi_cuda(
            d,
            a,
            r,
            *topology,
            *plan,
            K_stride=4,
            L_max=4,
            threads=4,
            grid_x=plan[0].numel(),
        )

    inputs = tuple(value.detach().requires_grad_() for value in (d_mem, a_geom, rhs))
    output = solve(*inputs)
    seeds = torch.linspace(
        -0.7,
        1.2,
        5 * output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape(5, *output.shape)
    _assert_vjps_match(output, inputs, seeds)


def test_multi_tree_vjp_accepts_an_empty_cotangent_batch():
    d_mem, a_geom, rhs, topology, plan = _multi_tree_case(batch=1)
    inputs = tuple(value.detach().requires_grad_() for value in (d_mem, a_geom, rhs))
    output = dhs_solve_multi_cuda(
        *inputs,
        *topology,
        *plan,
        K_stride=4,
        L_max=4,
        threads=4,
        grid_x=plan[0].numel(),
    )
    _assert_empty_vjp_batch(output, inputs)
