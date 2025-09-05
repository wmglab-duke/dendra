# ==============================================================================
# Multi-morph, warp-homogeneous DHS (single launch)
# ==============================================================================

import torch
import triton
import triton.language as tl


@triton.jit
def _multi_dhs_kernel_warp_hom(
    D_ptr,  # (B_total, K_STRIDE)   main diag  (SOLVER order; row-padded)
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
# Autograd bridge (forward implemented; backward TODO — mirror single-morph)
# ------------------------------------------------------------------------------


class DHSSolveMultiPacked(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        d_mem,
        a_geom,
        b,  # (B_total, K_stride) planes (SOLVER order; row-padded)
        P_cat,
        ORDER_cat,
        LAYER_PTR_cat,  # concatenated topology
        WARP_P_OFF,
        WARP_ORDER_OFF,
        WARP_LPTR_OFF,
        WARP_L,
        WARP_ROW_BASE,
        WARP_ROW_COUNT,  # per-warp plan
        K_stride: int,
        L_max: int,
        threads: int,
        grid_x: int = None,
    ):
        """
        Runs the warp-homogeneous multi-morph DHS solve in one launch.
        All inputs are assumed already prepared by `_dhs_multi.initialize(...)`
        and `_dhs_multi.step(...)` (i.e., in SOLVER order and row-padded).
        """
        assert d_mem.shape == a_geom.shape == b.shape, "d/a/b planes must share shape"
        B_total, Kp = d_mem.shape
        assert Kp == K_stride

        V_out = torch.empty_like(b)

        if grid_x is None:
            grid_x = WARP_ROW_BASE.numel()

        _multi_dhs_kernel_warp_hom[(grid_x,)](
            d_mem.clone(),  # cloned: kernel overwrites D and B
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

        # save for (future) backward
        ctx.save_for_backward(
            d_mem,
            a_geom,
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
        )
        ctx.K_stride = K_stride
        ctx.L_max = L_max
        ctx.threads = threads
        ctx.grid_x = grid_x
        return V_out

    @staticmethod
    def backward(ctx, grad_out):
        # You can mirror your single-morph adjoint:
        # 1) run the same kernel with RHS = grad_out to get g
        # 2) grad_b = g
        # 3) grad_d = -(g * V_out)
        # 4) grad_a via parent gathers in solver order.
        # Leaving unimplemented here to keep scope focused on the forward kernel.
        raise NotImplementedError("DHSSolveMultiPacked backward not implemented yet.")


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
