import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------
# 3×3 analytic inverse ------------------------------------------------
# ---------------------------------------------------------------------
@triton.jit
def inv3x3(a0, a1, a2, a3, a4, a5, a6, a7, a8):
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


# ---------------------------------------------------------------------
# one warp handles up to 32 fibres, K and B arbitrary -----------------
# ---------------------------------------------------------------------
BLOCK_FIBRES = 32  # fibres handled per warp (== warp size)


@triton.jit
def thomas_bt3_kernel(
    L_ptr, M_ptr, U_ptr, D_ptr, X_ptr, Minv_ptr, B, K: tl.constexpr, BLOCK: tl.constexpr
):
    # ------- map lanes → global fibre IDs ---------------------
    wid = tl.program_id(0)  # warp / CTA id
    lane_id = tl.arange(0, BLOCK)  # [0, …, 31]
    fid = wid * BLOCK + lane_id  # global fibre indices
    mask = fid < B  # some lanes may be inactive

    # stride = (#blocks −1)·3  or  K·{3,9}
    L = L_ptr + fid * (K - 1) * 3
    M = M_ptr + fid * K * 9
    U = U_ptr + fid * (K - 1) * 3
    D = D_ptr + fid * K * 3
    X = X_ptr + fid * K * 3
    Minv = Minv_ptr + fid * K * 9

    # ---------------- forward elimination ---------------------
    # ---- first main block
    a0 = tl.load(M + 0, mask=mask)
    a1 = tl.load(M + 1, mask=mask)
    a2 = tl.load(M + 2, mask=mask)
    a3 = tl.load(M + 3, mask=mask)
    a4 = tl.load(M + 4, mask=mask)
    a5 = tl.load(M + 5, mask=mask)
    a6 = tl.load(M + 6, mask=mask)
    a7 = tl.load(M + 7, mask=mask)
    a8 = tl.load(M + 8, mask=mask)

    i0, i1, i2, i3, i4, i5, i6, i7, i8 = inv3x3(a0, a1, a2, a3, a4, a5, a6, a7, a8)

    tl.store(Minv + 0, i0, mask=mask)
    tl.store(Minv + 1, i1, mask=mask)
    tl.store(Minv + 2, i2, mask=mask)
    tl.store(Minv + 3, i3, mask=mask)
    tl.store(Minv + 4, i4, mask=mask)
    tl.store(Minv + 5, i5, mask=mask)
    tl.store(Minv + 6, i6, mask=mask)
    tl.store(Minv + 7, i7, mask=mask)
    tl.store(Minv + 8, i8, mask=mask)

    for k in range(1, K):
        # lower diag
        u0 = tl.load(U + (k - 1) * 3 + 0, mask=mask)
        u1 = tl.load(U + (k - 1) * 3 + 1, mask=mask)
        u2 = tl.load(U + (k - 1) * 3 + 2, mask=mask)

        w_u0 = i0 * u0
        w_u1 = i1 * u1
        w_u2 = i2 * u2
        w_u3 = i3 * u0
        w_u4 = i4 * u1
        w_u5 = i5 * u2
        w_u6 = i6 * u0
        w_u7 = i7 * u1
        w_u8 = i8 * u2

        # Update M_k = M_k - diag(L_k) * W_U
        # This is now a row-scaling of W_U.
        l0 = tl.load(L + (k - 1) * 3 + 0, mask=mask)
        l1 = tl.load(L + (k - 1) * 3 + 1, mask=mask)
        l2 = tl.load(L + (k - 1) * 3 + 2, mask=mask)

        base = M + k * 9
        a0 = tl.load(base + 0, mask=mask) - l0 * w_u0
        a1 = tl.load(base + 1, mask=mask) - l0 * w_u1
        a2 = tl.load(base + 2, mask=mask) - l0 * w_u2
        a3 = tl.load(base + 3, mask=mask) - l1 * w_u3
        a4 = tl.load(base + 4, mask=mask) - l1 * w_u4
        a5 = tl.load(base + 5, mask=mask) - l1 * w_u5
        a6 = tl.load(base + 6, mask=mask) - l2 * w_u6
        a7 = tl.load(base + 7, mask=mask) - l2 * w_u7
        a8 = tl.load(base + 8, mask=mask) - l2 * w_u8

        # 3. Update D_k = D_k - diag(L_k) * (M'_{k-1})⁻¹ * D'_{k-1}
        dp0 = tl.load(D + (k - 1) * 3 + 0, mask=mask)
        dp1 = tl.load(D + (k - 1) * 3 + 1, mask=mask)
        dp2 = tl.load(D + (k - 1) * 3 + 2, mask=mask)

        # Calculate W_d = (M'_{k-1})⁻¹ * D'_{k-1}
        wd0 = i0 * dp0 + i1 * dp1 + i2 * dp2
        wd1 = i3 * dp0 + i4 * dp1 + i5 * dp2
        wd2 = i6 * dp0 + i7 * dp1 + i8 * dp2

        dk0 = tl.load(D + k * 3 + 0, mask=mask) - l0 * wd0
        dk1 = tl.load(D + k * 3 + 1, mask=mask) - l1 * wd1
        dk2 = tl.load(D + k * 3 + 2, mask=mask) - l2 * wd2
        tl.store(D + k * 3 + 0, dk0, mask=mask)
        tl.store(D + k * 3 + 1, dk1, mask=mask)
        tl.store(D + k * 3 + 2, dk2, mask=mask)

        # invert A_k for next loop
        i0, i1, i2, i3, i4, i5, i6, i7, i8 = inv3x3(a0, a1, a2, a3, a4, a5, a6, a7, a8)

        base_inv = Minv + k * 9
        tl.store(base_inv + 0, i0, mask=mask)
        tl.store(base_inv + 1, i1, mask=mask)
        tl.store(base_inv + 2, i2, mask=mask)
        tl.store(base_inv + 3, i3, mask=mask)
        tl.store(base_inv + 4, i4, mask=mask)
        tl.store(base_inv + 5, i5, mask=mask)
        tl.store(base_inv + 6, i6, mask=mask)
        tl.store(base_inv + 7, i7, mask=mask)
        tl.store(base_inv + 8, i8, mask=mask)

    # ---------------- backward substitution -------------------------
    # last block
    d0 = tl.load(D + (K - 1) * 3 + 0, mask=mask)
    d1 = tl.load(D + (K - 1) * 3 + 1, mask=mask)
    d2 = tl.load(D + (K - 1) * 3 + 2, mask=mask)
    x0 = i0 * d0 + i1 * d1 + i2 * d2
    x1 = i3 * d0 + i4 * d1 + i5 * d2
    x2 = i6 * d0 + i7 * d1 + i8 * d2
    tl.store(X + (K - 1) * 3 + 0, x0, mask=mask)
    tl.store(X + (K - 1) * 3 + 1, x1, mask=mask)
    tl.store(X + (K - 1) * 3 + 2, x2, mask=mask)

    for k in range(K - 2, -1, -1):
        # D_k -= U_k * X_{k+1}
        u0 = tl.load(U + k * 3 + 0, mask=mask)
        u1 = tl.load(U + k * 3 + 1, mask=mask)
        u2 = tl.load(U + k * 3 + 2, mask=mask)

        dk0 = tl.load(D + k * 3 + 0, mask=mask) - u0 * x0
        dk1 = tl.load(D + k * 3 + 1, mask=mask) - u1 * x1
        dk2 = tl.load(D + k * 3 + 2, mask=mask) - u2 * x2

        base_inv = Minv + k * 9
        i0 = tl.load(base_inv + 0, mask=mask)
        i1 = tl.load(base_inv + 1, mask=mask)
        i2 = tl.load(base_inv + 2, mask=mask)
        i3 = tl.load(base_inv + 3, mask=mask)
        i4 = tl.load(base_inv + 4, mask=mask)
        i5 = tl.load(base_inv + 5, mask=mask)
        i6 = tl.load(base_inv + 6, mask=mask)
        i7 = tl.load(base_inv + 7, mask=mask)
        i8 = tl.load(base_inv + 8, mask=mask)

        # solve for x_k
        x0 = i0 * dk0 + i1 * dk1 + i2 * dk2
        x1 = i3 * dk0 + i4 * dk1 + i5 * dk2
        x2 = i6 * dk0 + i7 * dk1 + i8 * dk2
        tl.store(X + k * 3 + 0, x0, mask=mask)
        tl.store(X + k * 3 + 1, x1, mask=mask)
        tl.store(X + k * 3 + 2, x2, mask=mask)


# ---------------------------------------------------------------------
# python launcher -----------------------------------------------------
# ---------------------------------------------------------------------
def _thomas_triton(lower, main, upper, rhs):
    """
    lower : (B, K-1, 3)
    main  : (B, K,   3,3)
    upper : (B, K-1, 3)
    rhs   : (B, K,   3)
    """
    B, K = rhs.shape[:2]
    out = torch.empty_like(rhs)

    # main_c = main.clone()
    rhs_c = rhs.clone()
    minv_c = torch.empty(B, K, 9, device=main.device, dtype=main.dtype)

    grid = ((B + BLOCK_FIBRES - 1) // BLOCK_FIBRES,)

    thomas_bt3_kernel[grid](  # “one warp solves up to 32 fibres”
        lower.reshape(B, -1),
        main.reshape(B, -1),
        upper.reshape(B, -1),
        rhs_c.reshape(B, -1),
        out.reshape(B, -1),
        minv_c.reshape(B, -1),
        B,
        K=K,
        BLOCK=BLOCK_FIBRES,
        num_warps=1,
        num_stages=4,
    )

    return out


# ---------------------------------------------------------------------
# autograd wrapper ----------------------------------------------------
# ---------------------------------------------------------------------
class ThomasSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lower, main, upper, rhs):
        """
        lower : (B, K-1, 3)
        main  : (B, K,   3,3)
        upper : (B, K-1, 3)
        rhs   : (B, K,   3)
        returns x       : (B, K,   3)
        """
        x = _thomas_triton(lower, main, upper, rhs)
        # save for backward
        ctx.save_for_backward(lower, main, upper, x)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out = ∂L/∂x  (same shape as x)
        returns gradients w.r.t. (lower, main, upper, rhs)
        """
        lower, main, upper, x = ctx.saved_tensors
        B, K = x.shape[:2]

        # ------ 1. adjoint solve:  Aᵀ g = grad_out -------------------
        main_T = main.transpose(-1, -2).contiguous()  # (B,K,3,3)
        # swap lower <-> upper
        g = _thomas_triton(
            upper,  # acts as new lower
            main_T,
            lower,  # acts as new upper
            grad_out,
        )

        # ------ 2. compute parameter gradients -----------------------
        # rhs
        grad_rhs = g

        # main blocks (B,K,3,3)
        grad_main = -(g.unsqueeze(-1) * x.unsqueeze(-2))

        # upper & lower bands
        grad_upper = -(g[:, :-1] * x[:, 1:])  # (B,K-1,3)
        grad_lower = -(g[:, 1:] * x[:, :-1])  # (B,K-1,3)

        return grad_lower, grad_main, grad_upper, grad_rhs


# convenience function -------------------------------------------------
def thomas_solve_cuda_bt(lower, main, upper, rhs):
    """differentiable wrapper"""
    return ThomasSolve.apply(lower, main, upper, rhs)
