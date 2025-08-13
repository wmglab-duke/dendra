import math

import torch
import triton
import triton.language as tl


# ============================================================
# Small 3x3 inverse (row-major inputs/outputs)
# ============================================================
@triton.jit
def _inv3x3(a0, a1, a2, a3, a4, a5, a6, a7, a8):  # pragma: no cover
    det = a0 * (a4 * a8 - a5 * a7) - a1 * (a3 * a8 - a5 * a6) + a2 * (a3 * a7 - a4 * a6)
    invd = 1.0 / det
    return (
        (a4 * a8 - a5 * a7) * invd,
        (a2 * a7 - a1 * a8) * invd,
        (a1 * a5 - a2 * a4) * invd,
        (a5 * a6 - a3 * a8) * invd,
        (a0 * a8 - a2 * a6) * invd,
        (a2 * a3 - a0 * a5) * invd,
        (a3 * a7 - a4 * a6) * invd,
        (a1 * a6 - a0 * a7) * invd,
        (a0 * a4 - a1 * a3) * invd,
    )


# ============================================================
# Packed DHS for 3×3 block systems (vi, ve0, ve1)
# ============================================================
@triton.jit
def _dhs_bt3_kernel_packed(
    D_ptr,  # (B, K, 9)   – block diagonals (row-major 3x3)
    G_ptr,  # (B, K, 3)   – edge diag conductances to parent per component
    B_ptr,  # (B, K, 3)   – RHS
    X_ptr,  # (B, K, 3)   – output solution
    MINV_ptr,  # (B, K, 9)   – store (A_ii)^(-1) for back-sub
    P_ptr,  # (K,)        – parent indices
    ORDER_ptr,  # (K,)        – elimination order (children→parent)
    LAYER_PTR_ptr,  # (L+1,)      – layer starts
    B_total: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    K_THREADS: tl.constexpr,
    WARP_SIZE: tl.constexpr,  # 32
):  # pragma: no cover
    NEURONS_PER_WARP = WARP_SIZE // K_THREADS

    lane_abs = tl.arange(0, WARP_SIZE)  # 0…31
    n_in_warp = lane_abs // K_THREADS  # sub‑neuron id within warp
    lane_local = lane_abs % K_THREADS  # lane within sub‑neuron

    b_global = tl.program_id(0) * NEURONS_PER_WARP + n_in_warp
    valid_neuron = b_global < B_total
    b_safe = tl.where(valid_neuron, b_global, 0)

    # Base pointers for this lane's neuron
    D = D_ptr + b_safe * (K * 9)
    G = G_ptr + b_safe * (K * 3)
    BB = B_ptr + b_safe * (K * 3)
    XX = X_ptr + b_safe * (K * 3)
    MINV = MINV_ptr + b_safe * (K * 9)

    P = P_ptr
    ORDER = ORDER_ptr
    LAYER_PTR = LAYER_PTR_ptr

    # ---------------------------
    # 1) Forward elimination
    # ---------------------------
    for layer in range(0, L):
        s = tl.load(LAYER_PTR + layer)
        e = tl.load(LAYER_PTR + layer + 1)

        offset = s + lane_local
        in_rng = lane_local < (e - s)
        m = in_rng & valid_neuron

        idx = tl.load(ORDER + offset, mask=m, other=0)
        parent = tl.load(P + idx, mask=m, other=-1)

        # Child pointers
        D_ch = D + idx * 9
        G_ch = G + idx * 3
        B_ch = BB + idx * 3

        # Load child's block diagonal (row-major)
        a0 = tl.load(D_ch + 0, mask=m, other=0.0)
        a1 = tl.load(D_ch + 1, mask=m, other=0.0)
        a2 = tl.load(D_ch + 2, mask=m, other=0.0)
        a3 = tl.load(D_ch + 3, mask=m, other=0.0)
        a4 = tl.load(D_ch + 4, mask=m, other=0.0)
        a5 = tl.load(D_ch + 5, mask=m, other=0.0)
        a6 = tl.load(D_ch + 6, mask=m, other=0.0)
        a7 = tl.load(D_ch + 7, mask=m, other=0.0)
        a8 = tl.load(D_ch + 8, mask=m, other=0.0)

        # Edge conductance vector to parent (diag entries)
        g0 = tl.load(G_ch + 0, mask=m, other=0.0)
        g1 = tl.load(G_ch + 1, mask=m, other=0.0)
        g2 = tl.load(G_ch + 2, mask=m, other=0.0)

        # "Just in time" degree add on child block diag
        a0 = a0 + g0
        a4 = a4 + g1
        a8 = a8 + g2

        # Store the *eliminated* child block back to D (used later in back-sub)
        tl.store(D_ch + 0, a0, mask=m)
        tl.store(D_ch + 1, a1, mask=m)
        tl.store(D_ch + 2, a2, mask=m)
        tl.store(D_ch + 3, a3, mask=m)
        tl.store(D_ch + 4, a4, mask=m)
        tl.store(D_ch + 5, a5, mask=m)
        tl.store(D_ch + 6, a6, mask=m)
        tl.store(D_ch + 7, a7, mask=m)
        tl.store(D_ch + 8, a8, mask=m)

        # Invert child's block once
        i0, i1, i2, i3, i4, i5, i6, i7, i8 = _inv3x3(a0, a1, a2, a3, a4, a5, a6, a7, a8)

        # Save inverse for back-sub
        MINV_ch = MINV + idx * 9
        tl.store(MINV_ch + 0, i0, mask=m)
        tl.store(MINV_ch + 1, i1, mask=m)
        tl.store(MINV_ch + 2, i2, mask=m)
        tl.store(MINV_ch + 3, i3, mask=m)
        tl.store(MINV_ch + 4, i4, mask=m)
        tl.store(MINV_ch + 5, i5, mask=m)
        tl.store(MINV_ch + 6, i6, mask=m)
        tl.store(MINV_ch + 7, i7, mask=m)
        tl.store(MINV_ch + 8, i8, mask=m)

        # Parent pointers / masks
        valid_parent = m & (parent >= 0)
        p_safe = tl.where(valid_parent, parent, 0)

        D_pa = D + p_safe * 9
        B_pa = BB + p_safe * 3

        # 1a) Degree add on parent's block diagonal (diag += g)
        tl.atomic_add(D_pa + 0, g0, mask=valid_parent)  # (0,0)
        tl.atomic_add(D_pa + 4, g1, mask=valid_parent)  # (1,1)
        tl.atomic_add(D_pa + 8, g2, mask=valid_parent)  # (2,2)

        # 1b) Schur complement on parent's block:  A_pp += - diag(g) * inv(A_ii) * diag(g)
        s00 = g0 * i0 * g0
        s01 = g0 * i1 * g1
        s02 = g0 * i2 * g2
        s10 = g1 * i3 * g0
        s11 = g1 * i4 * g1
        s12 = g1 * i5 * g2
        s20 = g2 * i6 * g0
        s21 = g2 * i7 * g1
        s22 = g2 * i8 * g2

        tl.atomic_add(D_pa + 0, -s00, mask=valid_parent)
        tl.atomic_add(D_pa + 1, -s01, mask=valid_parent)
        tl.atomic_add(D_pa + 2, -s02, mask=valid_parent)
        tl.atomic_add(D_pa + 3, -s10, mask=valid_parent)
        tl.atomic_add(D_pa + 4, -s11, mask=valid_parent)
        tl.atomic_add(D_pa + 5, -s12, mask=valid_parent)
        tl.atomic_add(D_pa + 6, -s20, mask=valid_parent)
        tl.atomic_add(D_pa + 7, -s21, mask=valid_parent)
        tl.atomic_add(D_pa + 8, -s22, mask=valid_parent)

        # 1c) RHS update on parent:  b_p += diag(g) * inv(A_ii) * b_i
        b0 = tl.load(B_ch + 0, mask=m, other=0.0)
        b1 = tl.load(B_ch + 1, mask=m, other=0.0)
        b2 = tl.load(B_ch + 2, mask=m, other=0.0)

        # w = inv(A_ii) @ b_i
        w0 = i0 * b0 + i1 * b1 + i2 * b2
        w1 = i3 * b0 + i4 * b1 + i5 * b2
        w2 = i6 * b0 + i7 * b1 + i8 * b2

        tl.atomic_add(B_pa + 0, g0 * w0, mask=valid_parent)
        tl.atomic_add(B_pa + 1, g1 * w1, mask=valid_parent)
        tl.atomic_add(B_pa + 2, g2 * w2, mask=valid_parent)

    tl.debug_barrier()

    # ---------------------------
    # 2) Back substitution
    # ---------------------------
    for layer in range(L - 1, -1, -1):
        s = tl.load(LAYER_PTR + layer)
        e = tl.load(LAYER_PTR + layer + 1)

        offset = s + lane_local
        in_rng = lane_local < (e - s)
        m = in_rng & valid_neuron

        idx = tl.load(ORDER + offset, mask=m, other=0)
        parent = tl.load(P + idx, mask=m, other=-1)

        # Pointers
        D_ch = D + idx * 9
        MINV_ch = MINV + idx * 9
        G_ch = G + idx * 3
        B_ch = BB + idx * 3
        X_ch = XX + idx * 3

        # Load eliminated D (not strictly needed if we rely on MINV), and MINV
        i0 = tl.load(MINV_ch + 0, mask=m, other=1.0)
        i1 = tl.load(MINV_ch + 1, mask=m, other=0.0)
        i2 = tl.load(MINV_ch + 2, mask=m, other=0.0)
        i3 = tl.load(MINV_ch + 3, mask=m, other=0.0)
        i4 = tl.load(MINV_ch + 4, mask=m, other=1.0)
        i5 = tl.load(MINV_ch + 5, mask=m, other=0.0)
        i6 = tl.load(MINV_ch + 6, mask=m, other=0.0)
        i7 = tl.load(MINV_ch + 7, mask=m, other=0.0)
        i8 = tl.load(MINV_ch + 8, mask=m, other=1.0)

        # Parent voltage
        valid_parent = m & (parent >= 0)
        p_safe = tl.where(valid_parent, parent, 0)
        X_pa = XX + p_safe * 3

        vp0 = tl.load(X_pa + 0, mask=valid_parent, other=0.0)
        vp1 = tl.load(X_pa + 1, mask=valid_parent, other=0.0)
        vp2 = tl.load(X_pa + 2, mask=valid_parent, other=0.0)

        # Local data
        g0 = tl.load(G_ch + 0, mask=m, other=0.0)
        g1 = tl.load(G_ch + 1, mask=m, other=0.0)
        g2 = tl.load(G_ch + 2, mask=m, other=0.0)

        b0 = tl.load(B_ch + 0, mask=m, other=0.0)
        b1 = tl.load(B_ch + 1, mask=m, other=0.0)
        b2 = tl.load(B_ch + 2, mask=m, other=0.0)

        # rhs = b_i + diag(g) * v_parent
        r0 = b0 + g0 * vp0
        r1 = b1 + g1 * vp1
        r2 = b2 + g2 * vp2

        # x_i = inv(A_ii_elim) @ rhs
        x0 = i0 * r0 + i1 * r1 + i2 * r2
        x1 = i3 * r0 + i4 * r1 + i5 * r2
        x2 = i6 * r0 + i7 * r1 + i8 * r2

        tl.store(X_ch + 0, x0, mask=m)
        tl.store(X_ch + 1, x1, mask=m)
        tl.store(X_ch + 2, x2, mask=m)


class DHSBTSolve3(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, D_blocks, G_vec, b, parent_idx, order, layer_ptr, threads: int
    ):  # pragma: no cover
        B, K, m, n = D_blocks.shape
        assert m == 3 and n == 3, "DHSBTSolve3 only supports 3x3 blocks (M=3)."
        L = layer_ptr.numel() - 1

        X = torch.empty_like(b)
        MINV = torch.empty(B, K, 9, device=b.device, dtype=b.dtype)

        NEURONS_PER_WARP = 32 // threads
        grid_x = math.ceil(B / NEURONS_PER_WARP)

        _dhs_bt3_kernel_packed[(grid_x,)](
            D_blocks.reshape(B, -1),
            G_vec.reshape(B, -1),
            b.reshape(B, -1),
            X.reshape(B, -1),
            MINV.reshape(B, -1),
            parent_idx,
            order,
            layer_ptr,
            B_total=B,
            K=K,
            L=L,
            K_THREADS=threads,
            WARP_SIZE=32,
            num_warps=1,
            num_stages=4,
        )

        ctx.save_for_backward(D_blocks, G_vec, X, parent_idx, order, layer_ptr)
        ctx.threads = threads
        return X

    @staticmethod
    def backward(ctx, grad_out):  # pragma: no cover
        D_blocks, G_vec, X, parent_idx, order, layer_ptr = ctx.saved_tensors
        threads = ctx.threads

        B, K, _, _ = D_blocks.shape
        L = layer_ptr.numel() - 1

        # 1) Adjoint solve: A^T g = grad_out
        g = torch.empty_like(grad_out)
        MINV_tmp = torch.empty(B, K, 9, device=g.device, dtype=g.dtype)

        NEURONS_PER_WARP = 32 // threads
        grid_x = math.ceil(B / NEURONS_PER_WARP)

        _dhs_bt3_kernel_packed[(grid_x,)](
            D_blocks.reshape(B, -1).clone(),  # rebuild identical matrix
            G_vec.reshape(B, -1),  # parameters (edge diags)
            grad_out.reshape(B, -1).clone(),  # RHS
            g.reshape(B, -1),  # out
            MINV_tmp.reshape(B, -1),  # scratch
            parent_idx,
            order,
            layer_ptr,
            B_total=B,
            K=K,
            L=L,
            K_THREADS=threads,
            WARP_SIZE=32,
            num_warps=1,
            num_stages=4,
        )

        # 2) Gradients wrt original inputs
        grad_b = g
        # dL/dD_i = - g_i x_i^T   (outer product, per-node 3x3)
        grad_D = -(g.unsqueeze(-1) * X.unsqueeze(-2))

        # dL/dG per component follows the scalar formula, applied component-wise
        is_root = parent_idx < 0
        parent_clamped = parent_idx.clamp_min(0)
        # Gather parent slices (B,K,3)
        X_parent = X.gather(
            1, parent_clamped.view(1, -1, 1).expand(X.shape[0], -1, X.shape[2])
        )
        g_parent = g.gather(
            1, parent_clamped.view(1, -1, 1).expand(g.shape[0], -1, g.shape[2])
        )

        grad_G = -(g - g_parent) * (X - X_parent)
        # For roots: only diagonal contribution remains
        grad_G[:, is_root, :] = -(g[:, is_root, :] * X[:, is_root, :])

        return grad_D, grad_G, grad_b, None, None, None, None


def dhs_bt_solve_cuda(D_blocks, G_vec, b, parent_idx, order, layer_ptr, threads=16):
    """
    Public wrapper: solve A x = b on a tree, where each node has a 3-vector unknown
    (vi, ve0, ve1).

    Parameters
    ----------
    D_blocks  : (B, K, 3, 3)  block-diagonal per node (radial + time terms). *No axial degree terms*.
    G_vec     : (B, K, 3)     edge conductances to parent for each component (0:vi, 1:ve0, 2:ve1).
                               Must be zero for roots.
    b         : (B, K, 3)     right-hand side per node
    parent_idx: (K,) int64    parent indices (-1 for root)
    order     : (K,) int64    elimination order (children→parent)
    layer_ptr : (L+1,) int64  layer boundaries
    threads   : int           lanes per neuron (≤32)
    """
    return DHSBTSolve3.apply(D_blocks, G_vec, b, parent_idx, order, layer_ptr, threads)
