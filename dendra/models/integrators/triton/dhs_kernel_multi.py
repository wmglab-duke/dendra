# ==============================================================================
# Multi-morph, warp-homogeneous DHS (single launch)
# ==============================================================================

import torch
import triton
import triton.language as tl


@triton.jit
def _multi_dhs_kernel_warp_hom(
    D_ptr,  # (B_total, K_STRIDE)   main diag   (SOLVER order; row-padded)
    A_ptr,  # (B_total, K_STRIDE)   axial g     (SOLVER order; row-padded)
    B_ptr,  # (B_total, K_STRIDE)   RHS         (SOLVER order; row-padded)
    V_ptr,  # (B_total, K_STRIDE)   output V    (SOLVER order; row-padded)
    # ---- concatenated per-morph topology tables (no batch dimension) ----
    P_cat,  # int32/int64[sum K_g]      parents in local [0..K_g-1], root = -1
    ORDER_cat,  # int32/int64[sum K_g]      elimination order (children→parents)
    LAYER_PTR_cat,  # int32/int64[sum (L_g+1)]  layer start indices (per morph)
    # ---- per-warp plan: GUARANTEES one morphology per warp ----
    WARP_P_OFF,  # int64[W]  base into P_cat for this warp’s morphology
    WARP_ORDER_OFF,  # int64[W]  base into ORDER_cat
    WARP_LPTR_OFF,  # int64[W]  base into LAYER_PTR_cat
    WARP_L,  # int32[W]  number of layers L_g for this morph
    WARP_ROW_BASE,  # int64[W]  first row (neuron) index this warp covers
    WARP_ROW_COUNT,  # int32[W]  how many neurons (≤ NEURONS_PER_WARP) are valid
    # ---- compile-time constants ----
    K_STRIDE: tl.constexpr,  # row pitch (max K across morphs)
    L_MAX: tl.constexpr,  # max layers across morphs (loop bound)
    K_THREADS: tl.constexpr,  # lanes per neuron (≤ 32)
    WARP_SIZE: tl.constexpr,  # 32
):  # pragma: no cover
    # ---- warp/lane bookkeeping ----
    warp_id = tl.program_id(0)
    lane_abs = tl.arange(0, WARP_SIZE)  # 0..31
    n_in_warp = lane_abs // K_THREADS  # sub-warp id (which neuron in this warp)
    lane_loc = lane_abs % K_THREADS  # 0..K_THREADS-1 within a layer chunk

    # ---- per-warp morphology bases ----
    P = P_cat + tl.load(WARP_P_OFF + warp_id)
    ORDER = ORDER_cat + tl.load(WARP_ORDER_OFF + warp_id)
    LPTR = LAYER_PTR_cat + tl.load(WARP_LPTR_OFF + warp_id)
    L_g = tl.load(WARP_L + warp_id)  # runtime int for this morph

    # ---- rows (neurons) covered by this warp ----
    row0 = tl.load(WARP_ROW_BASE + warp_id)  # first row
    row_count = tl.load(WARP_ROW_COUNT + warp_id)  # ≤ NEURONS_PER_WARP
    valid_neuron = n_in_warp < row_count
    row_idx = row0 + n_in_warp
    row_safe = tl.where(valid_neuron, row_idx, 0)

    # Per-lane base pointers to row-padded planes
    D = D_ptr + row_safe * K_STRIDE
    A = A_ptr + row_safe * K_STRIDE
    B = B_ptr + row_safe * K_STRIDE
    V = V_ptr + row_safe * K_STRIDE

    # ==========================
    # 1) Forward elimination
    # ==========================
    for layer in range(0, L_MAX):
        # Only active for layers < L_g
        active_layer = layer < L_g

        # Masked loads are safe even if layer >= L_g
        s = tl.load(LPTR + layer, mask=active_layer, other=0)
        e = tl.load(LPTR + layer + 1, mask=active_layer, other=0)

        offset = s + lane_loc
        in_range = lane_loc < (e - s)
        m = in_range & valid_neuron & active_layer

        idx = tl.load(ORDER + offset, mask=m, other=0)  # local [0..K_g-1]
        parent = tl.load(P + idx, mask=m, other=-1)
        g_i = tl.load(A + idx, mask=m, other=0.0)

        # “just-in-time” diagonal finalize
        old_d = tl.atomic_add(D + idx, g_i, mask=m)
        d_i = old_d + g_i

        has_parent = m & (parent >= 0)
        p_safe = tl.where(has_parent, parent, 0)

        # accumulate into parent’s diagonal
        tl.atomic_add(D + p_safe, g_i, mask=has_parent)

        # elimination update
        b_i = tl.load(B + idx, mask=m, other=0.0)
        fac = -g_i / d_i
        tl.atomic_add(D + p_safe, fac * g_i, mask=has_parent)
        tl.atomic_add(B + p_safe, -fac * b_i, mask=has_parent)

    tl.debug_barrier()  # finish all forward atomics before back-sub

    # ==========================
    # 2) Back substitution
    # ==========================
    for layer in range(L_MAX - 1, -1, -1):
        active_layer = layer < L_g

        s = tl.load(LPTR + layer, mask=active_layer, other=0)
        e = tl.load(LPTR + layer + 1, mask=active_layer, other=0)

        offset = s + lane_loc
        in_range = lane_loc < (e - s)
        m = in_range & valid_neuron & active_layer

        idx = tl.load(ORDER + offset, mask=m, other=0)
        parent = tl.load(P + idx, mask=m, other=-1)

        has_parent = m & (parent >= 0)
        p_safe = tl.where(has_parent, parent, 0)

        g_i = tl.load(A + idx, mask=m, other=0.0)
        d_e = tl.load(D + idx, mask=m, other=1.0)
        b_e = tl.load(B + idx, mask=m, other=0.0)
        v_par = tl.load(V + p_safe, mask=has_parent, other=0.0)

        v_i = (b_e + g_i * v_par) / d_e
        tl.store(V + idx, v_i, mask=m)


# ------------------------------------------------------------------------------
# Autograd bridge (forward & backward mirrors single-morph)
# ------------------------------------------------------------------------------


# ============================
# Backward: grad_a kernel
# ============================


@triton.jit
def _multi_dhs_grad_a_kernel(
    V_ptr,  # (B_total, K_STRIDE)  solver order
    G_ptr,  # (B_total, K_STRIDE)  solver order (adjoint solution)
    Agrad_ptr,  # (B_total, K_STRIDE)  OUT: grad wrt a_geom (solver order)
    P_cat,
    ORDER_cat,
    LAYER_PTR_cat,  # concatenated topology
    WARP_P_OFF,
    WARP_ORDER_OFF,
    WARP_LPTR_OFF,
    WARP_L,
    WARP_ROW_BASE,
    WARP_ROW_COUNT,  # per-warp plan
    K_STRIDE: tl.constexpr,
    L_MAX: tl.constexpr,
    K_THREADS: tl.constexpr,
    WARP_SIZE: tl.constexpr,
):  # pragma: no cover
    warp_id = tl.program_id(0)
    lane_abs = tl.arange(0, WARP_SIZE)
    n_in_warp = lane_abs // K_THREADS
    lane_loc = lane_abs % K_THREADS

    # per-warp morphology
    P = P_cat + tl.load(WARP_P_OFF + warp_id)
    ORDER = ORDER_cat + tl.load(WARP_ORDER_OFF + warp_id)
    LPTR = LAYER_PTR_cat + tl.load(WARP_LPTR_OFF + warp_id)
    L_g = tl.load(WARP_L + warp_id)

    # rows covered by this warp
    row0 = tl.load(WARP_ROW_BASE + warp_id)
    row_count = tl.load(WARP_ROW_COUNT + warp_id)
    valid_row = n_in_warp < row_count
    row_idx = row0 + n_in_warp
    row_safe = tl.where(valid_row, row_idx, 0)

    V = V_ptr + row_safe * K_STRIDE
    G = G_ptr + row_safe * K_STRIDE
    dA = Agrad_ptr + row_safe * K_STRIDE

    # per-layer sweep (read-only topology; write dA)
    for layer in range(0, L_MAX):
        active_layer = layer < L_g

        s = tl.load(LPTR + layer, mask=active_layer, other=0)
        e = tl.load(LPTR + layer + 1, mask=active_layer, other=0)

        offset = s + lane_loc
        in_range = lane_loc < (e - s)
        m = in_range & valid_row & active_layer

        idx = tl.load(ORDER + offset, mask=m, other=0)
        par = tl.load(P + idx, mask=m, other=-1)

        has_par = m & (par >= 0)
        p_safe = tl.where(has_par, par, 0)

        v_i = tl.load(V + idx, mask=m, other=0.0)
        g_i = tl.load(G + idx, mask=m, other=0.0)

        v_p = tl.load(V + p_safe, mask=has_par, other=0.0)
        g_p = tl.load(G + p_safe, mask=has_par, other=0.0)

        grad_nonroot = -((g_i - g_p) * (v_i - v_p))
        grad_root = -(g_i * v_i)

        grad = tl.where(has_par, grad_nonroot, grad_root)
        tl.store(dA + idx, grad, mask=m)


class DHSSolveMultiPacked(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        d_mem,
        a_geom,
        b,  # (B_total, K_stride), solver order
        P_cat,
        ORDER_cat,
        LAYER_PTR_cat,
        WARP_P_OFF,
        WARP_ORDER_OFF,
        WARP_LPTR_OFF,
        WARP_L,
        WARP_ROW_BASE,
        WARP_ROW_COUNT,
        K_stride: int,
        L_max: int,
        threads: int,
        grid_x: int = None,
    ):
        assert d_mem.shape == a_geom.shape == b.shape
        V_out = torch.empty_like(b)

        if grid_x is None:
            grid_x = WARP_ROW_BASE.numel()

        _multi_dhs_kernel_warp_hom[(grid_x,)](
            d_mem.clone(),
            a_geom,
            b.clone(),
            V_out,
            P_cat,
            ORDER_cat,
            LAYER_PTR_cat,
            WARP_P_OFF,
            WARP_ORDER_OFF,
            WARP_LPTR_OFF,
            WARP_L,
            WARP_ROW_BASE,
            WARP_ROW_COUNT,
            K_STRIDE=K_stride,
            L_MAX=L_max,
            K_THREADS=threads,
            WARP_SIZE=32,
            num_warps=1,
            num_stages=4,
        )

        # Save ONLY dynamic things as "saved tensors"
        ctx.save_for_backward(d_mem, a_geom, V_out)

        # Stash static plan/topology as plain attributes (tiny ref cost, no copies)
        ctx.P_cat = P_cat
        ctx.ORDER_cat = ORDER_cat
        ctx.LAYER_PTR_cat = LAYER_PTR_cat
        ctx.WARP_P_OFF = WARP_P_OFF
        ctx.WARP_ORDER_OFF = WARP_ORDER_OFF
        ctx.WARP_LPTR_OFF = WARP_LPTR_OFF
        ctx.WARP_L = WARP_L
        ctx.WARP_ROW_BASE = WARP_ROW_BASE
        ctx.WARP_ROW_COUNT = WARP_ROW_COUNT

        ctx.K_stride = K_stride
        ctx.L_max = L_max
        ctx.threads = threads
        ctx.grid_x = grid_x
        return V_out

    @staticmethod
    def backward(ctx, grad_out):
        d_mem, a_geom, V_out = ctx.saved_tensors

        # Recover static tensors from ctx attrs (no save_for_backward used)
        P_cat = ctx.P_cat
        ORDER_cat = ctx.ORDER_cat
        LAYER_PTR_cat = ctx.LAYER_PTR_cat
        WARP_P_OFF = ctx.WARP_P_OFF
        WARP_ORDER_OFF = ctx.WARP_ORDER_OFF
        WARP_LPTR_OFF = ctx.WARP_LPTR_OFF
        WARP_L = ctx.WARP_L
        WARP_ROW_BASE = ctx.WARP_ROW_BASE
        WARP_ROW_COUNT = ctx.WARP_ROW_COUNT

        K_stride, L_max, threads, grid_x = (
            ctx.K_stride,
            ctx.L_max,
            ctx.threads,
            ctx.grid_x,
        )

        # 1) adjoint solve: g = A^{-1} * grad_out   (solver order, row-padded)
        g = torch.empty_like(grad_out)
        _multi_dhs_kernel_warp_hom[(grid_x,)](
            d_mem.clone(),
            a_geom,
            grad_out.clone(),
            g,
            P_cat,
            ORDER_cat,
            LAYER_PTR_cat,
            WARP_P_OFF,
            WARP_ORDER_OFF,
            WARP_LPTR_OFF,
            WARP_L,
            WARP_ROW_BASE,
            WARP_ROW_COUNT,
            K_STRIDE=K_stride,
            L_MAX=L_max,
            K_THREADS=threads,
            WARP_SIZE=32,
            num_warps=1,
            num_stages=4,
        )

        # 2) grads wrt inputs (all in solver order)
        grad_b = g
        grad_d = -(g * V_out)

        grad_a = torch.zeros_like(a_geom)
        _multi_dhs_grad_a_kernel[(grid_x,)](
            V_out,
            g,
            grad_a,
            P_cat,
            ORDER_cat,
            LAYER_PTR_cat,
            WARP_P_OFF,
            WARP_ORDER_OFF,
            WARP_LPTR_OFF,
            WARP_L,
            WARP_ROW_BASE,
            WARP_ROW_COUNT,
            K_STRIDE=K_stride,
            L_MAX=L_max,
            K_THREADS=threads,
            WARP_SIZE=32,
            num_warps=1,
            num_stages=2,
        )

        # Return grads for each forward input (tensors only)
        return (
            grad_d,  # d_mem
            grad_a,  # a_geom
            grad_b,  # b
            None,
            None,
            None,  # P_cat, ORDER_cat, LAYER_PTR_cat
            None,
            None,
            None,
            None,  # WARP_*
            None,
            None,  # WARP_ROW_BASE, WARP_ROW_COUNT
            None,
            None,
            None,
            None,  # K_stride, L_max, threads, grid_x
        )


def dhs_solve_multi_cuda(
    d_mem,
    a_geom,
    b,
    P_cat,
    ORDER_cat,
    LAYER_PTR_cat,
    WARP_P_OFF,
    WARP_ORDER_OFF,
    WARP_LPTR_OFF,
    WARP_L,
    WARP_ROW_BASE,
    WARP_ROW_COUNT,
    K_stride: int,
    L_max: int,
    threads: int,
    grid_x: int = None,
):
    """
    Public entry for multi-morph DHS. Matches the signature used by `_dhs_multi.step(...)`.
    """
    return DHSSolveMultiPacked.apply(
        d_mem,
        a_geom,
        b,
        P_cat,
        ORDER_cat,
        LAYER_PTR_cat,
        WARP_P_OFF,
        WARP_ORDER_OFF,
        WARP_LPTR_OFF,
        WARP_L,
        WARP_ROW_BASE,
        WARP_ROW_COUNT,
        K_stride,
        L_max,
        threads,
        grid_x,
    )
