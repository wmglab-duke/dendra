"""Fused CUDA elimination and recovery for planned scalar tree systems."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._contracts import flatten_vmap_solver_batch, restore_vmap_solver_batch

_BLOCK_SIZE = 64


@triton.jit
def _coefficient(original, work, slot, active, size: tl.constexpr):
    initial = slot < size
    v_initial = tl.load(original + slot, active & initial, other=0)
    v_reduced = tl.load(work + slot - size, active & ~initial, other=0)
    return tl.where(initial, v_initial, v_reduced)


@triton.jit
def _schur_product(left, right, pivot):
    # Dividing the smaller factor first can lose a representable shunt or
    # bridge. Keeping it outside the quotient also avoids subtractive Schur
    # cancellation under strong axial coupling.
    left_is_smaller = tl.abs(left) < tl.abs(right)
    small = tl.where(left_is_smaller, left, right)
    large = tl.where(left_is_smaller, right, left)
    return small * (large / pivot)


@triton.jit
def _update(
    d,
    g,
    rhs,
    d_work,
    g_work,
    rhs_work,
    updates,
    incidence,
    index,
    active,
    size: tl.constexpr,
    update_count: tl.constexpr,
    incidence_count: tl.constexpr,
):
    source = tl.load(updates + index, active, other=0)
    offset = tl.load(updates + update_count + index, active, other=0)
    count = tl.load(updates + 2 * update_count + index, active, other=0)
    diagonal = _coefficient(d, d_work, source, active, size)
    residual = _coefficient(rhs, rhs_work, source, active, size)
    for step in range(tl.max(count, 0)):
        record = offset + step
        valid = step < count
        eliminated = tl.load(incidence + record, valid, other=0)
        near_slot = tl.load(incidence + incidence_count + record, valid, other=0)
        far_slot = tl.load(incidence + 2 * incidence_count + record, valid, other=-1)
        destination = tl.load(incidence + 3 * incidence_count + record, valid, other=-1)
        shunt = _coefficient(d, d_work, eliminated, valid, size)
        value = _coefficient(rhs, rhs_work, eliminated, valid, size)
        near = _coefficient(g, g_work, near_slot, valid, size)
        far = _coefficient(g, g_work, far_slot, valid & (far_slot >= 0), size)
        pivot = tl.where(valid, shunt + (near + far), 1)
        diagonal += _schur_product(near, shunt, pivot)
        residual += (near / pivot) * value
        tl.store(
            g_work + destination - size,
            _schur_product(near, far, pivot),
            valid & (destination >= size),
        )
    tl.store(d_work + index, diagonal, active)
    tl.store(rhs_work + index, residual, active)


@triton.jit
def _recover(
    d,
    g,
    rhs,
    d_work,
    g_work,
    rhs_work,
    out,
    recovery,
    index,
    active,
    size: tl.constexpr,
):
    stride: tl.constexpr = size - 1
    node = tl.load(recovery + index, active, other=0)
    source = tl.load(recovery + stride + index, active, other=0)
    ancestor = tl.load(recovery + 2 * stride + index, active, other=0)
    child = tl.load(recovery + 3 * stride + index, active, other=-1)
    near_slot = tl.load(recovery + 4 * stride + index, active, other=0)
    far_slot = tl.load(recovery + 5 * stride + index, active, other=-1)
    shunt = _coefficient(d, d_work, source, active, size)
    value = _coefficient(rhs, rhs_work, source, active, size)
    near = _coefficient(g, g_work, near_slot, active, size)
    far = _coefficient(g, g_work, far_slot, active & (child >= 0), size)
    ancestor_value = tl.load(out + ancestor, active, other=0)
    child_value = tl.load(out + child, active & (child >= 0), other=0)
    pivot = tl.where(active, shunt + (near + far), 1)
    tl.store(
        out + node, (value + near * ancestor_value + far * child_value) / pivot, active
    )


@triton.jit
def _solve_contracted_tree(
    d,
    g,
    rhs,
    d_work,
    g_work,
    rhs_work,
    out,
    phases,
    updates,
    incidence,
    recovery,
    size: tl.constexpr,
    update_count: tl.constexpr,
    incidence_count: tl.constexpr,
    phase_count: tl.constexpr,
    root: tl.constexpr,
    root_state: tl.constexpr,
    block: tl.constexpr,
):
    lane = tl.arange(0, block)
    for phase in range(phase_count):
        start = tl.load(phases + phase)
        end = tl.load(phases + phase_count + phase)
        for tile in range(start, end, block):
            index = tile + lane
            _update(
                d,
                g,
                rhs,
                d_work,
                g_work,
                rhs_work,
                updates,
                incidence,
                index,
                index < end,
                size,
                update_count,
                incidence_count,
            )
        tl.debug_barrier()

    shunt = _coefficient(d, d_work, root_state, True, size)
    residual = _coefficient(rhs, rhs_work, root_state, True, size)
    tl.store(out + root, residual / (shunt + tl.load(g + root)))
    tl.debug_barrier()

    for reverse in range(phase_count):
        phase = phase_count - 1 - reverse
        start = tl.load(phases + 2 * phase_count + phase)
        end = tl.load(phases + 3 * phase_count + phase)
        for tile in range(start, end, block):
            index = tile + lane
            _recover(
                d,
                g,
                rhs,
                d_work,
                g_work,
                rhs_work,
                out,
                recovery,
                index,
                index < end,
                size,
            )
        tl.debug_barrier()


@triton.jit
def _fine_kernel(
    d,
    g,
    rhs,
    work,
    out,
    phases,
    updates,
    incidence,
    recovery,
    size: tl.constexpr,
    update_count: tl.constexpr,
    edge_count: tl.constexpr,
    incidence_count: tl.constexpr,
    phase_count: tl.constexpr,
    root: tl.constexpr,
    root_state: tl.constexpr,
    scratch_size: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offset = row * size
    d_work = work + row * scratch_size
    rhs_work = d_work + update_count
    g_work = rhs_work + update_count
    _solve_contracted_tree(
        d + offset,
        g + offset,
        rhs + offset,
        d_work,
        g_work,
        rhs_work,
        out + offset,
        phases,
        updates,
        incidence,
        recovery,
        size,
        update_count,
        incidence_count,
        phase_count,
        root,
        root_state,
        block,
    )


@triton.jit
def _single_node(d, g, rhs, out, batch, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    valid = row < batch
    pivot = tl.load(d + row, valid, other=1) + tl.load(g + row, valid, other=0)
    tl.store(out + row, tl.load(rhs + row, valid, other=0) / pivot, valid)


@triton.jit
def _condense_segment_tile(
    d,
    g,
    rhs,
    segments,
    interior,
    saved_rhs,
    pivots,
    bridges,
    summaries,
    segment,
    segment_count: tl.constexpr,
    block: tl.constexpr,
):
    active = segment < segment_count
    child = tl.load(segments + segment_count + segment, active, other=0)
    start = tl.load(segments + 2 * segment_count + segment, active, other=0)
    end = tl.load(segments + 3 * segment_count + segment, active, other=0)
    first = tl.load(interior + start, active & (start < end), other=0)
    near = tl.load(g + tl.where(start < end, first, child), active, other=0)
    dtype: tl.constexpr = d.dtype.element_ty
    left_diagonal = tl.full((block,), 0, dtype)
    left_rhs = tl.full((block,), 0, dtype)
    carry_diagonal = tl.full((block,), 0, dtype)
    carry_rhs = tl.full((block,), 0, dtype)
    for step in range(tl.max(end - start, 0)):
        position = start + step
        valid = active & (position < end)
        node = tl.load(interior + position, valid, other=0)
        more = position + 1 < end
        successor = tl.load(interior + position + 1, valid & more, other=0)
        far = tl.load(g + tl.where(more, successor, child), valid, other=0)
        shunt = tl.load(d + node, valid, other=0) + carry_diagonal
        value = tl.load(rhs + node, valid, other=0) + carry_rhs
        pivot = tl.where(valid, shunt + (near + far), 1)
        tl.store(saved_rhs + position, value, valid)
        tl.store(pivots + position, pivot, valid)
        tl.store(bridges + position, near, valid)
        left_diagonal += tl.where(valid, _schur_product(near, shunt, pivot), 0)
        left_rhs += tl.where(valid, (near / pivot) * value, 0)
        carry_diagonal = tl.where(
            valid, _schur_product(far, shunt, pivot), carry_diagonal
        )
        carry_rhs = tl.where(valid, (far / pivot) * value, carry_rhs)
        near = tl.where(valid, _schur_product(near, far, pivot), near)
    tl.store(summaries + segment, left_diagonal, active)
    tl.store(summaries + segment_count + segment, left_rhs, active)
    tl.store(summaries + 2 * segment_count + segment, carry_diagonal, active)
    tl.store(summaries + 3 * segment_count + segment, carry_rhs, active)
    tl.store(summaries + 4 * segment_count + segment, near, active)


@triton.jit
def _gather_segment_tile(
    d,
    g,
    rhs,
    retained,
    endpoint_incidence,
    summaries,
    d_reduced,
    g_reduced,
    rhs_reduced,
    index,
    retained_count: tl.constexpr,
    segment_count: tl.constexpr,
):
    active = index < retained_count
    node = tl.load(retained + index, active, other=0)
    incoming = tl.load(retained + retained_count + index, active, other=-1)
    start = tl.load(retained + 2 * retained_count + index, active, other=0)
    count = tl.load(retained + 3 * retained_count + index, active, other=0)
    diagonal = tl.load(d + node, active, other=0)
    value = tl.load(rhs + node, active, other=0)
    for step in range(tl.max(count, 0)):
        incident = start + step
        valid = step < count
        segment = tl.load(endpoint_incidence + incident, valid, other=0)
        side = tl.load(
            endpoint_incidence + 2 * segment_count + incident, valid, other=0
        )
        entry = summaries + 2 * side * segment_count + segment
        diagonal += tl.load(entry, valid, other=0)
        value += tl.load(entry + segment_count, valid, other=0)
    original_edge = tl.load(g + node, active & (incoming < 0), other=0)
    bridge = tl.load(
        summaries + 4 * segment_count + incoming, active & (incoming >= 0), other=0
    )
    tl.store(d_reduced + index, diagonal, active)
    tl.store(rhs_reduced + index, value, active)
    tl.store(g_reduced + index, tl.where(incoming < 0, original_edge, bridge), active)


@triton.jit
def _recover_segment_tile(
    g,
    out,
    segments,
    interior,
    saved_rhs,
    pivots,
    bridges,
    segment,
    segment_count: tl.constexpr,
):
    active = segment < segment_count
    ancestor = tl.load(segments + segment, active, other=0)
    successor = tl.load(segments + segment_count + segment, active, other=0)
    start = tl.load(segments + 2 * segment_count + segment, active, other=0)
    end = tl.load(segments + 3 * segment_count + segment, active, other=0)
    ancestor_value = tl.load(out + ancestor, active, other=0)
    successor_value = tl.load(out + successor, active, other=0)
    for step in range(tl.max(end - start, 0)):
        position = end - 1 - step
        valid = active & (position >= start)
        node = tl.load(interior + position, valid, other=0)
        value = tl.load(saved_rhs + position, valid, other=0)
        pivot = tl.load(pivots + position, valid, other=1)
        near = tl.load(bridges + position, valid, other=0)
        far = tl.load(g + successor, valid, other=0)
        recovered = (value + near * ancestor_value + far * successor_value) / pivot
        tl.store(out + node, recovered, valid)
        successor = tl.where(valid, node, successor)
        successor_value = tl.where(valid, recovered, successor_value)


@triton.jit
def _segment_kernel(
    d,
    g,
    rhs,
    work,
    out,
    segments,
    interior,
    retained,
    endpoint_incidence,
    phases,
    updates,
    incidence,
    recovery,
    size: tl.constexpr,
    retained_count: tl.constexpr,
    segment_count: tl.constexpr,
    interior_count: tl.constexpr,
    update_count: tl.constexpr,
    edge_count: tl.constexpr,
    incidence_count: tl.constexpr,
    phase_count: tl.constexpr,
    root: tl.constexpr,
    root_state: tl.constexpr,
    scratch_size: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offset = row * size
    d, g, rhs, out = d + offset, g + offset, rhs + offset, out + offset
    saved_rhs = work + row * scratch_size
    pivots = saved_rhs + interior_count
    bridges = pivots + interior_count
    d_reduced = bridges + interior_count
    g_reduced = d_reduced + retained_count
    rhs_reduced = g_reduced + retained_count
    summaries = rhs_reduced + retained_count

    lane = tl.arange(0, block)
    for tile in range(0, segment_count, block):
        _condense_segment_tile(
            d,
            g,
            rhs,
            segments,
            interior,
            saved_rhs,
            pivots,
            bridges,
            summaries,
            tile + lane,
            segment_count,
            block,
        )
    tl.debug_barrier()
    for tile in range(0, retained_count, block):
        _gather_segment_tile(
            d,
            g,
            rhs,
            retained,
            endpoint_incidence,
            summaries,
            d_reduced,
            g_reduced,
            rhs_reduced,
            tile + lane,
            retained_count,
            segment_count,
        )
    tl.debug_barrier()

    # Segment summaries are no longer needed after the gather.
    d_work = summaries
    rhs_work = d_work + update_count
    g_work = rhs_work + update_count
    x_reduced = g_work + edge_count
    _solve_contracted_tree(
        d_reduced,
        g_reduced,
        rhs_reduced,
        d_work,
        g_work,
        rhs_work,
        x_reduced,
        phases,
        updates,
        incidence,
        recovery,
        retained_count,
        update_count,
        incidence_count,
        phase_count,
        root,
        root_state,
        block,
    )

    for tile in range(0, retained_count, block):
        index = tile + lane
        valid = index < retained_count
        node = tl.load(retained + index, valid, other=0)
        tl.store(out + node, tl.load(x_reduced + index, valid, other=0), valid)
    tl.debug_barrier()
    for tile in range(0, segment_count, block):
        _recover_segment_tile(
            g,
            out,
            segments,
            interior,
            saved_rhs,
            pivots,
            bridges,
            tile + lane,
            segment_count,
        )


def _launch(d, g, rhs, tables, shape):
    """Submit one CUDA kernel with invocation-owned scratch and output storage."""
    (
        size,
        retained,
        segments,
        interiors,
        updates,
        edges,
        incidences,
        phases,
        root,
        root_state,
    ) = shape
    batch = d.shape[0]
    out = torch.empty_like(rhs, memory_format=torch.contiguous_format)
    if batch == 0:
        return out
    d, g, rhs = (value.contiguous() for value in (d, g, rhs))
    options = {
        "block": _BLOCK_SIZE,
        "num_warps": _BLOCK_SIZE // 32,
        "enable_fp_fusion": False,
    }
    if size == 1:
        _single_node[(triton.cdiv(batch, _BLOCK_SIZE),)](
            d, g, rhs, out, batch, **options
        )
    elif segments:
        scratch_size = 3 * (retained + interiors) + max(
            5 * segments, 2 * updates + edges + retained
        )
        work = rhs.new_empty((batch, scratch_size))
        _segment_kernel[(batch,)](
            d,
            g,
            rhs,
            work,
            out,
            *tables,
            size,
            retained,
            segments,
            interiors,
            updates,
            edges,
            incidences,
            phases,
            root,
            root_state,
            scratch_size,
            **options,
        )
    else:
        scratch_size = max(1, 2 * updates + edges)
        work = rhs.new_empty((batch, scratch_size))
        _fine_kernel[(batch,)](
            d,
            g,
            rhs,
            work,
            out,
            *tables[-4:],
            size,
            updates,
            edges,
            incidences,
            phases,
            root,
            root_state,
            scratch_size,
            **options,
        )
    return out


@torch.library.custom_op(
    "dendra_triton::tree_contraction", mutates_args=(), device_types="cuda"
)
def _numeric_solve(
    d: torch.Tensor,
    g: torch.Tensor,
    rhs: torch.Tensor,
    parent: torch.Tensor,
    is_root: torch.Tensor,
    segments: torch.Tensor,
    interiors: torch.Tensor,
    retained: torch.Tensor,
    endpoint_incidence: torch.Tensor,
    phases: torch.Tensor,
    updates: torch.Tensor,
    incidence: torch.Tensor,
    recovery: torch.Tensor,
    shape: list[int],
) -> torch.Tensor:
    tables = (
        segments,
        interiors,
        retained,
        endpoint_incidence,
        phases,
        updates,
        incidence,
        recovery,
    )
    return _launch(d, g, rhs, tables, shape)


@_numeric_solve.register_fake
def _fake_solve(d, g, rhs, *metadata):
    return torch.empty_like(rhs, memory_format=torch.contiguous_format)


def _gradients(adjoint, x, parent, is_root, needs_input_grad):
    need_d, need_g, need_rhs = needs_input_grad[:3]
    conductance = None
    if need_g:
        x_drop = torch.where(is_root, x, x - x.index_select(-1, parent))
        adjoint_drop = torch.where(
            is_root, adjoint, adjoint - adjoint.index_select(-1, parent)
        )
        conductance = -(x_drop * adjoint_drop)
    return (
        -(adjoint * x) if need_d else None,
        conductance,
        adjoint if need_rhs else None,
    )


def _saved_operands(inputs, output):
    d, g, _, parent, is_root, *launch = inputs
    return (d, g, output, parent, is_root, *launch[:-1]), launch[-1]


def _setup_numeric_context(ctx, inputs, output):
    saved, ctx.shape = _saved_operands(inputs, output)
    ctx.save_for_backward(*saved)


def _numeric_backward(ctx, cotangent):
    d, g, x, parent, is_root, *tables = ctx.saved_tensors
    adjoint = _numeric_solve(d, g, cotangent, parent, is_root, *tables, ctx.shape)
    return (
        *_gradients(adjoint, x, parent, is_root, ctx.needs_input_grad),
        *((None,) * (len(tables) + 3)),
    )


_numeric_solve.register_autograd(
    _numeric_backward, setup_context=_setup_numeric_context
)


def _vmap(solve, info, in_dims, d, g, rhs, *launch):
    # The last operand is a list of shape constants. Functorch can mirror it
    # with a list of None entries instead of a single None.
    if any(dimension is not None for dimension in in_dims[3:-1]):
        raise ValueError("tree contraction topology must be shared under vmap")
    flat, batch = flatten_vmap_solver_batch(info, in_dims[:3], d, g, rhs)
    output = solve(*flat, *launch)
    return restore_vmap_solver_batch(output, info.batch_size, batch), 0


@_numeric_solve.register_vmap
def _numeric_vmap(info, in_dims, d, g, rhs, *launch):
    return _vmap(_numeric_solve, info, in_dims, d, g, rhs, *launch)


def prepare(plan, device):
    """Upload the immutable descriptors before CUDA execution or graph capture."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA tree contraction requires a CUDA device")
    return plan.prepare(device)


def _prepared_arguments(plan, device):
    topology = plan._plan
    tables = topology._devices.get(device)
    if tables is None:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "prepare the CUDA tree contraction plan before compilation"
            )
        prepare(plan, device)
        tables = topology._devices[device]

    if plan.max_segment_interior_nodes:
        skeleton = topology.skeleton
        retained = topology.retained_count
        segments = topology.segment_count
        interiors = topology.interior_count
    else:
        skeleton = topology
        retained = plan.size
        segments = interiors = 0
        # These four operands are unused by the fine implementation, which
        # shares the same dispatcher schema with segmented contraction.
        tables = (tables[0], tables[0], tables[0], tables[0], *tables)

    shape = (
        plan.size,
        retained,
        segments,
        interiors,
        skeleton.state_slots,
        skeleton.edge_slots,
        tables[-2].shape[1],
        len(skeleton.phases),
        skeleton.root,
        skeleton.root_state,
    )
    return tables, shape


class _TreeContraction(torch.autograd.Function):
    @staticmethod
    def forward(d, g, rhs, *launch):
        return _numeric_solve(d, g, rhs, *launch)

    @staticmethod
    def setup_context(ctx, inputs, output):
        saved, ctx.shape = _saved_operands(inputs, output)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    def backward(ctx, cotangent):
        d, g, x, parent, is_root, *tables = ctx.saved_tensors
        adjoint = _apply(d, g, cotangent, parent, is_root, *tables, ctx.shape)
        return (
            *_gradients(adjoint, x, parent, is_root, ctx.needs_input_grad),
            *((None,) * (len(tables) + 3)),
        )

    @staticmethod
    def jvp(ctx, d_tangent, g_tangent, rhs_tangent, *_metadata_tangent):
        d, g, x, parent, is_root, *tables = ctx.saved_tensors
        tangent = torch.zeros_like(x) if rhs_tangent is None else rhs_tangent
        if d_tangent is not None:
            tangent = tangent - d_tangent * x
        if g_tangent is not None:
            flow = torch.where(is_root, 0, g_tangent * (x - x.index_select(-1, parent)))
            tangent = (tangent - flow).scatter_add(
                -1, parent.unsqueeze(0).expand_as(flow), flow
            )
            tangent = tangent - torch.where(is_root, g_tangent * x, 0)
        return _apply(d, g, tangent, parent, is_root, *tables, ctx.shape)

    @staticmethod
    def vmap(info, in_dims, d, g, rhs, *launch):
        return _vmap(_apply, info, in_dims, d, g, rhs, *launch)


@torch.compiler.allow_in_graph
def _apply_in_graph(d, g, rhs, *launch):
    return _TreeContraction.apply(d, g, rhs, *launch)


def _apply(d, g, rhs, *launch):
    if torch.compiler.is_compiling():
        return _apply_in_graph(d, g, rhs, *launch)
    return _TreeContraction.apply(d, g, rhs, *launch)


def solve(d, g, rhs, plan):
    """Solve a scalar Hines right-hand side in float32 or float64 on CUDA."""
    if d.ndim != 2 or d.shape[1] != plan.size:
        raise ValueError(
            "tree contraction inputs must have shape (B, K) matching the plan"
        )
    if d.dtype not in (torch.float32, torch.float64):
        raise ValueError("CUDA tree contraction requires float32 or float64")
    for value in (d, g, rhs):
        if (
            value.shape != d.shape
            or value.device != d.device
            or value.device.type != "cuda"
            or value.dtype != d.dtype
            or value.layout != torch.strided
        ):
            raise ValueError(
                "tree contraction inputs must be same-shaped strided CUDA tensors with matching dtypes"
            )
    tables, shape = _prepared_arguments(plan, d.device)
    return _apply(d, g, rhs, *plan._gradient_indices(d.device), *tables, list(shape))
