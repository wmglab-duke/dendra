"""Transform-compatible facades over the existing CUDA solver kernels.

Imperative solvers deliberately retain their existing launch paths. Functional
execution uses split-context autograd Functions: all changing numerical inputs
are explicit, backward reuses the same solve, and JVP applies ``A dx = db-dA x``.
Only the raw, nonmutating dispatcher boundaries import Triton, so importing
Dendra on a CPU-only installation never acquires a CUDA dependency.
"""

from __future__ import annotations

import torch

from .triton._contracts import (
    flatten_vmap_solver_batch,
    restore_vmap_solver_batch,
    validate_block_tridiagonal,
    validate_tree,
    validate_tree_block,
    validate_tree_multi,
    validate_tridiagonal,
)


def _launch_tridiagonal(a, b, c, rhs, block, pcr):
    if block:
        from .triton.bt_kernel import thomas_bt3_solve

        return thomas_bt3_solve(a, b, c, rhs)
    if pcr:
        from .triton.pcr_kernel_thomas import _pcr_solve_cuda_t_adjoint

        return _pcr_solve_cuda_t_adjoint(a, b, c, rhs)
    from .triton.t_kernel_thomas import _thomas_solve_cuda_t_adjoint

    return _thomas_solve_cuda_t_adjoint(a, b, c, rhs)


def _launch_tree(main, edge, rhs, parent, order, layers, threads, block):
    if block:
        from .triton.dhs_kernel_bt import _dhs_bt3_adjoint

        # This existing boundary solves the transpose; feed transposed node
        # blocks to obtain the ordinary forward system. Edge blocks are diagonal.
        return _dhs_bt3_adjoint(
            main.transpose(-1, -2).contiguous(),
            edge,
            rhs,
            parent,
            order,
            layers,
            threads,
        )
    from .triton.dhs_kernel import _dhs_packed_adjoint

    # Scalar DHS is symmetric. Its adjoint operator is exactly the forward solve.
    return _dhs_packed_adjoint(main, edge, rhs, parent, order, layers, threads)


def _launch_multi(main, edge, rhs, *plan):
    from .triton.dhs_kernel_multi import _dhs_multi_adjoint

    return _dhs_multi_adjoint(main, edge, rhs, *plan)


@torch.library.custom_op("dendra_functional_cuda::tridiagonal", mutates_args=())
def _raw_tridiagonal(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    rhs: torch.Tensor,
    block: bool,
    pcr: bool,
) -> torch.Tensor:
    """Keep the solve and its AD contract intact through no-grad FX capture."""
    return _launch_tridiagonal(a, b, c, rhs, block, pcr)


@_raw_tridiagonal.register_fake
def _fake_tridiagonal(a, b, c, rhs, block, pcr):
    return rhs.new_empty(rhs.shape)


@torch.library.custom_op("dendra_functional_cuda::tree", mutates_args=())
def _raw_tree(
    main: torch.Tensor,
    edge: torch.Tensor,
    rhs: torch.Tensor,
    parent: torch.Tensor,
    order: torch.Tensor,
    layers: torch.Tensor,
    threads: int,
    block: bool,
) -> torch.Tensor:
    # The reused raw adjoint launchers assume stride-one topology, whereas
    # the public contract permits arbitrary same-valued one-dimensional views.
    return _launch_tree(
        main,
        edge,
        rhs,
        parent.contiguous(),
        order.contiguous(),
        layers.contiguous(),
        threads,
        block,
    )


@_raw_tree.register_fake
def _fake_tree(main, edge, rhs, parent, order, layers, threads, block):
    return rhs.new_empty(rhs.shape)


@torch.library.custom_op("dendra_functional_cuda::multi", mutates_args=())
def _raw_multi(
    main: torch.Tensor,
    edge: torch.Tensor,
    rhs: torch.Tensor,
    p_cat: torch.Tensor,
    order: torch.Tensor,
    layers: torch.Tensor,
    p_off: torch.Tensor,
    o_off: torch.Tensor,
    l_off: torch.Tensor,
    levels: torch.Tensor,
    row_base: torch.Tensor,
    row_count: torch.Tensor,
    width: int,
    max_levels: int,
    threads: int,
    grid: int,
) -> torch.Tensor:
    plan = tuple(
        value.contiguous()
        for value in (
            p_cat,
            order,
            layers,
            p_off,
            o_off,
            l_off,
            levels,
            row_base,
            row_count,
        )
    )
    output = _launch_multi(main, edge, rhs, *plan, width, max_levels, threads, grid)
    _parent, active = _multi_parent_plane(rhs, plan)
    # Legacy packed launches do not write padding. Mask it inside the opaque
    # boundary too, because scan captures that boundary without its outer
    # autograd Function and must retain the very same primal/derivative contract.
    return torch.where(active, output, torch.zeros_like(output))


@_raw_multi.register_fake
def _fake_multi(main, edge, rhs, *metadata):
    return rhs.new_empty(rhs.shape)


def _save(ctx, *values):
    ctx.save_for_backward(*values)
    ctx.save_for_forward(*values)


def _require_shared(in_dims):
    if any(value is not None for value in in_dims):
        raise ValueError("CUDA functional solver topology must be shared across vmap")


def _empty_solution(flattened):
    # Keep a differentiable zero-sized result without launching a zero-grid
    # kernel, including when only a coefficient (not the RHS) is differentiated.
    result = flattened[-1].clone()
    for value in flattened[:-1]:
        result = result + value.sum() * 0.0
    return result


class _TridiagonalSolve(torch.autograd.Function):
    @staticmethod
    def forward(a, b, c, rhs, block, pcr):
        return _raw_tridiagonal(a, b, c, rhs, block, pcr)

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b, c, _rhs, ctx.block, ctx.pcr = inputs
        _save(ctx, a, b, c, output)

    @staticmethod
    def backward(ctx, grad_x):
        a, b, c, x = ctx.saved_tensors
        main = b.transpose(-1, -2).contiguous() if ctx.block else b
        g = _apply_tridiagonal(c, main, a, grad_x, ctx.block, ctx.pcr)
        axis = -2 if ctx.block else -1
        if ctx.block:
            grad_main = -(g.unsqueeze(-1) * x.unsqueeze(-2))
        else:
            grad_main = -(g * x)
        return (
            -(
                g.narrow(axis, 1, g.shape[axis] - 1)
                * x.narrow(axis, 0, x.shape[axis] - 1)
            ),
            grad_main,
            -(
                g.narrow(axis, 0, g.shape[axis] - 1)
                * x.narrow(axis, 1, x.shape[axis] - 1)
            ),
            g,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, a_t, b_t, c_t, rhs_t, _block_t, _pcr_t):
        a, b, c, x = ctx.saved_tensors
        tangent = torch.zeros_like(x) if rhs_t is None else rhs_t
        axis = -2 if ctx.block else -1
        size = x.shape[axis]
        pad = torch.zeros_like(x.narrow(axis, 0, 1))
        if b_t is not None:
            tangent = tangent - (
                (b_t @ x.unsqueeze(-1)).squeeze(-1) if ctx.block else b_t * x
            )
        if a_t is not None:
            tangent = tangent - torch.cat(
                (pad, a_t * x.narrow(axis, 0, size - 1)), dim=axis
            )
        if c_t is not None:
            tangent = tangent - torch.cat(
                (c_t * x.narrow(axis, 1, size - 1), pad), dim=axis
            )
        return _apply_tridiagonal(a, b, c, tangent, ctx.block, ctx.pcr)

    @staticmethod
    def vmap(info, in_dims, a, b, c, rhs, block, pcr):
        _require_shared(in_dims[4:])
        flat, batch = flatten_vmap_solver_batch(info, in_dims[:4], a, b, c, rhs)
        result = (
            _empty_solution(flat)
            if info.batch_size == 0
            else _apply_tridiagonal(*flat, block, pcr)
        )
        return restore_vmap_solver_batch(result, info.batch_size, batch), 0


def _tree_edge_gradient(x, g, parent, *, block):
    axis = -2 if block else -1
    safe = parent.clamp_min(0).to(torch.long)
    root = (parent < 0).reshape((1, -1, 1) if block else (1, -1))
    return torch.where(
        root,
        -(g * x),
        -((g - g.index_select(axis, safe)) * (x - x.index_select(axis, safe))),
    )


def _tree_tangent_rhs(main_t, edge_t, rhs_t, x, parent, *, block):
    tangent = torch.zeros_like(x) if rhs_t is None else rhs_t
    if main_t is not None:
        tangent = tangent - (
            (main_t @ x.unsqueeze(-1)).squeeze(-1) if block else main_t * x
        )
    if edge_t is None:
        return tangent
    axis = -2 if block else -1
    shape = (1, -1, 1) if block else (1, -1)
    safe = parent.clamp_min(0).to(torch.long)
    root = (parent < 0).reshape(shape)
    edge_action = torch.where(
        root, torch.zeros_like(x), edge_t * (x - x.index_select(axis, safe))
    )
    tangent = tangent - edge_action
    tangent = tangent.scatter_add(axis, safe.reshape(shape).expand_as(x), edge_action)
    return tangent - torch.where(root, edge_t * x, torch.zeros_like(x))


class _TreeSolve(torch.autograd.Function):
    @staticmethod
    def forward(main, edge, rhs, parent, order, layers, threads, block):
        return _raw_tree(main, edge, rhs, parent, order, layers, threads, block)

    @staticmethod
    def setup_context(ctx, inputs, output):
        main, edge, _rhs, parent, order, layers, ctx.threads, ctx.block = inputs
        _save(ctx, main, edge, output, parent, order, layers)

    @staticmethod
    def backward(ctx, grad_x):
        main, edge, x, parent, order, layers = ctx.saved_tensors
        adjoint_main = main.transpose(-1, -2).contiguous() if ctx.block else main
        g = _apply_tree(
            adjoint_main, edge, grad_x, parent, order, layers, ctx.threads, ctx.block
        )
        grad_main = -(g.unsqueeze(-1) * x.unsqueeze(-2)) if ctx.block else -(g * x)
        return (
            grad_main,
            _tree_edge_gradient(x, g, parent, block=ctx.block),
            g,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, main_t, edge_t, rhs_t, _p_t, _o_t, _l_t, _threads_t, _block_t):
        main, edge, x, parent, order, layers = ctx.saved_tensors
        tangent = _tree_tangent_rhs(main_t, edge_t, rhs_t, x, parent, block=ctx.block)
        return _apply_tree(
            main, edge, tangent, parent, order, layers, ctx.threads, ctx.block
        )

    @staticmethod
    def vmap(info, in_dims, main, edge, rhs, parent, order, layers, threads, block):
        _require_shared(in_dims[3:])
        flat, batch = flatten_vmap_solver_batch(info, in_dims[:3], main, edge, rhs)
        result = (
            _empty_solution(flat)
            if info.batch_size == 0
            else _apply_tree(*flat, parent, order, layers, threads, block)
        )
        return restore_vmap_solver_batch(result, info.batch_size, batch), 0


def _multi_parent_plane(template, plan):
    """Expand row-local parents with tensor ops, never a host topology read."""
    p_cat, _order, layer_ptr, p_off, _o_off, l_off, levels, row_base, row_count = plan
    rows, width = template.shape
    row = torch.arange(rows, device=template.device).unsqueeze(1)
    bases = row_base.to(torch.long).unsqueeze(0)
    counts = row_count.to(torch.long).unsqueeze(0)
    owner = ((row >= bases) & (row < bases + counts)).to(torch.long).argmax(dim=1)
    offsets = p_off.to(torch.long).index_select(0, owner)
    ends = l_off.to(torch.long).index_select(0, owner) + levels.to(
        torch.long
    ).index_select(0, owner)
    sizes = layer_ptr.to(torch.long).index_select(0, ends)
    column = torch.arange(width, device=template.device).unsqueeze(0)
    active = column < sizes.unsqueeze(1)
    safe_column = torch.minimum(column, (sizes - 1).clamp_min(0).unsqueeze(1))
    parent = p_cat.to(torch.long)[offsets.unsqueeze(1) + safe_column]
    return torch.where(active, parent, -1), active


def _repeat_multi_plan(lanes, batch, plan):
    p_cat, order, layers, p_off, o_off, l_off, levels, row_base, row_count = plan
    offsets = torch.arange(lanes, device=row_base.device, dtype=row_base.dtype) * batch
    return (
        p_cat,
        order,
        layers,
        p_off.repeat(lanes),
        o_off.repeat(lanes),
        l_off.repeat(lanes),
        levels.repeat(lanes),
        (row_base.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1),
        row_count.repeat(lanes),
    )


class _MultiSolve(torch.autograd.Function):
    @staticmethod
    def forward(main, edge, rhs, *metadata):
        return _raw_multi(main, edge, rhs, *metadata)

    @staticmethod
    def setup_context(ctx, inputs, output):
        main, edge, _rhs, *metadata = inputs
        ctx.config = metadata[9:]
        _save(ctx, main, edge, output, *metadata[:9])

    @staticmethod
    def backward(ctx, grad_x):
        main, edge, x, *plan = ctx.saved_tensors
        g = _apply_multi(main, edge, grad_x, *plan, *ctx.config)
        parent, active = _multi_parent_plane(x, plan)
        safe = parent.clamp_min(0)
        edge_grad = torch.where(
            parent < 0,
            -(g * x),
            -((g - g.gather(1, safe)) * (x - x.gather(1, safe))),
        )
        zero = torch.zeros_like(g)
        return (
            torch.where(active, -(g * x), zero),
            torch.where(active, edge_grad, zero),
            torch.where(active, g, zero),
            *((None,) * 13),
        )

    @staticmethod
    def jvp(ctx, main_t, edge_t, rhs_t, *_metadata_t):
        main, edge, x, *plan = ctx.saved_tensors
        parent, active = _multi_parent_plane(x, plan)
        tangent = torch.zeros_like(x) if rhs_t is None else rhs_t
        if main_t is not None:
            tangent = tangent - main_t * x
        if edge_t is not None:
            safe = parent.clamp_min(0)
            root = (parent < 0) & active
            action = torch.where(
                active & ~root,
                edge_t * (x - x.gather(1, safe)),
                torch.zeros_like(x),
            )
            tangent = (tangent - action).scatter_add(1, safe, action)
            tangent = tangent - torch.where(root, edge_t * x, torch.zeros_like(x))
        tangent = torch.where(active, tangent, torch.zeros_like(tangent))
        return _apply_multi(main, edge, tangent, *plan, *ctx.config)

    @staticmethod
    def vmap(info, in_dims, main, edge, rhs, *metadata):
        _require_shared(in_dims[3:])
        flat, batch = flatten_vmap_solver_batch(info, in_dims[:3], main, edge, rhs)
        if info.batch_size == 0:
            result = _empty_solution(flat)
        else:
            plan = _repeat_multi_plan(info.batch_size, batch, metadata[:9])
            width, levels, threads, grid = metadata[9:]
            result = _apply_multi(
                *flat, *plan, width, levels, threads, grid * info.batch_size
            )
        return restore_vmap_solver_batch(result, info.batch_size, batch), 0


# ``make_fx`` intentionally erases Python autograd.Function wrappers during
# scan capture. Keep ordinary reverse AD and batched reverse AD attached to
# the surviving opaque solver nodes as well as the direct Function entrypoint.
# Both routes share the exact same context layout and derivative formulas.
_raw_tridiagonal.register_autograd(
    _TridiagonalSolve.backward, setup_context=_TridiagonalSolve.setup_context
)
_raw_tridiagonal.register_vmap(_TridiagonalSolve.vmap)
_raw_tree.register_autograd(_TreeSolve.backward, setup_context=_TreeSolve.setup_context)
_raw_tree.register_vmap(_TreeSolve.vmap)
_raw_multi.register_autograd(
    _MultiSolve.backward, setup_context=_MultiSolve.setup_context
)
_raw_multi.register_vmap(_MultiSolve.vmap)


@torch.compiler.allow_in_graph
def _apply_tridiagonal(a, b, c, rhs, block, pcr):
    """Leave custom JVP handling to AOTAutograd rather than Dynamo inlining."""
    return _TridiagonalSolve.apply(a, b, c, rhs, block, pcr)


@torch.compiler.allow_in_graph
def _apply_tree(main, edge, rhs, parent, order, layers, threads, block):
    return _TreeSolve.apply(main, edge, rhs, parent, order, layers, threads, block)


@torch.compiler.allow_in_graph
def _apply_multi(main, edge, rhs, *metadata):
    return _MultiSolve.apply(main, edge, rhs, *metadata)


def thomas_solve(a, b, c, rhs):
    validate_tridiagonal(a, b, c, rhs)
    return _apply_tridiagonal(a, b, c, rhs, False, False)


def pcr_solve(a, b, c, rhs):
    validate_tridiagonal(a, b, c, rhs)
    return _apply_tridiagonal(a, b, c, rhs, False, True)


def block_thomas_solve(lower, main, upper, rhs):
    validate_block_tridiagonal(lower, main, upper, rhs)
    return _apply_tridiagonal(lower, main, upper, rhs, True, False)


def dhs_solve(main, edge, rhs, parent, order, layers, *, threads):
    validate_tree(main, edge, rhs, parent, order, layers, threads)
    return _apply_tree(main, edge, rhs, parent, order, layers, threads, False)


def dhs_bt_solve(main, edge, rhs, parent, order, layers, *, threads):
    validate_tree_block(main, edge, rhs, parent, order, layers, threads)
    return _apply_tree(main, edge, rhs, parent, order, layers, threads, True)


def dhs_multi_solve(main, edge, rhs, *plan, K_stride, L_max, threads, grid_x):
    validate_tree_multi(main, edge, rhs, *plan, K_stride, L_max, threads, grid_x)
    return _apply_multi(main, edge, rhs, *plan, K_stride, L_max, threads, grid_x)
