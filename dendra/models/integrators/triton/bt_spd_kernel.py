from typing import Tuple

import triton
import triton.language as tl

BLOCK_FIBRES = 32  # fibres per warp, as before


@triton.jit
def _chol3x3_raw(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
):
    cond0 = a0 > 0
    l00 = tl.sqrt(tl.where(cond0, a0, 1.0))
    l10 = a1 / l00
    l20 = a2 / l00

    s11 = a4 - l10 * l10
    cond1 = s11 > 0
    l11 = tl.sqrt(tl.where(cond1, s11, 1.0))

    l21 = (a5 - l10 * l20) / l11

    s22 = a8 - l20 * l20 - l21 * l21
    cond2 = s22 > 0
    l22 = tl.sqrt(tl.where(cond2, s22, 1.0))

    succ = cond0 & cond1 & cond2

    L0 = l00
    L1 = 0.0
    L2 = 0.0

    L3 = l10
    L4 = l11
    L5 = 0.0

    L6 = l20
    L7 = l21
    L8 = l22

    return L0, L1, L2, L3, L4, L5, L6, L7, L8, succ


@triton.jit
def _try_chol_with_bump3x3(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    eps_rel,
    eps_abs,
):
    a01 = 0.5 * (a1 + a3)
    a02 = 0.5 * (a2 + a6)
    a12 = 0.5 * (a5 + a7)

    a1 = a01
    a3 = a01
    a2 = a02
    a6 = a02
    a5 = a12
    a7 = a12

    L0a, L1a, L2a, L3a, L4a, L5a, L6a, L7a, L8a, succ1 = _chol3x3_raw(
        a0, a1, a2, a3, a4, a5, a6, a7, a8
    )

    tr = tl.abs(a0) + tl.abs(a4) + tl.abs(a8)

    r0 = tl.abs(a1) + tl.abs(a2)
    r1 = tl.abs(a3) + tl.abs(a5)
    r2 = tl.abs(a6) + tl.abs(a7)
    b0 = a0 - r0
    b1 = a4 - r1
    b2 = a8 - r2
    gmin = tl.minimum(b0, tl.minimum(b1, b2))

    margin = tl.maximum(eps_abs, eps_rel * tr)
    tau = tl.where(gmin > 0, margin, -gmin + margin)

    a0b = a0 + tau
    a4b = a4 + tau
    a8b = a8 + tau

    L0b, L1b, L2b, L3b, L4b, L5b, L6b, L7b, L8b, succ2 = _chol3x3_raw(
        a0b, a1, a2, a3, a4b, a5, a6, a7, a8b
    )

    tau2 = 10.0 * margin
    a0c = a0b + tau2
    a4c = a4b + tau2
    a8c = a8b + tau2

    L0c, L1c, L2c, L3c, L4c, L5c, L6c, L7c, L8c, succ3 = _chol3x3_raw(
        a0c, a1, a2, a3, a4c, a5, a6, a7, a8c
    )

    L0 = tl.where(succ1, L0a, tl.where(succ2, L0b, L0c))
    L1 = tl.where(succ1, L1a, tl.where(succ2, L1b, L1c))
    L2 = tl.where(succ1, L2a, tl.where(succ2, L2b, L2c))
    L3 = tl.where(succ1, L3a, tl.where(succ2, L3b, L3c))
    L4 = tl.where(succ1, L4a, tl.where(succ2, L4b, L4c))
    L5 = tl.where(succ1, L5a, tl.where(succ2, L5b, L5c))
    L6 = tl.where(succ1, L6a, tl.where(succ2, L6b, L6c))
    L7 = tl.where(succ1, L7a, tl.where(succ2, L7b, L7c))
    L8 = tl.where(succ1, L8a, tl.where(succ2, L8b, L8c))

    return L0, L1, L2, L3, L4, L5, L6, L7, L8


@triton.jit
def _chol_solve3(
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    b0,
    b1,
    b2,
):
    y0 = b0 / L0
    y1 = (b1 - L3 * y0) / L4
    y2 = (b2 - L6 * y0 - L7 * y1) / L8

    x2 = y2 / L8
    x1 = (y1 - L7 * x2) / L4
    x0 = (y0 - L3 * x1 - L6 * x2) / L0

    return x0, x1, x2


@triton.jit
def thomas_bt3_spd_fwd_kernel(
    L_ptr,  # (B, K-1, 3)
    M_ptr,  # (B, K,   9)  main 3x3 blocks, overwritten with L_k
    U_ptr,  # (B, K-1, 3)
    D_ptr,  # (B, K,   3)  RHS, updated in-place
    X_ptr,  # (B, K,   3)  solution
    CHOL_ptr,  # (B, K,   9)  cache of L_k
    B,
    eps_rel,
    eps_abs,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):  # pragma: no cover
    wid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    fid = wid * BLOCK + lane
    mask = fid < B

    Lband = L_ptr + fid * (K - 1) * 3
    M = M_ptr + fid * K * 9
    Uband = U_ptr + fid * (K - 1) * 3
    D = D_ptr + fid * K * 3
    X = X_ptr + fid * K * 3
    CHOL = CHOL_ptr + fid * K * 9

    # k = 0: factor first block
    a0 = tl.load(M + 0, mask=mask)
    a1 = tl.load(M + 1, mask=mask)
    a2 = tl.load(M + 2, mask=mask)
    a3 = tl.load(M + 3, mask=mask)
    a4 = tl.load(M + 4, mask=mask)
    a5 = tl.load(M + 5, mask=mask)
    a6 = tl.load(M + 6, mask=mask)
    a7 = tl.load(M + 7, mask=mask)
    a8 = tl.load(M + 8, mask=mask)

    L0, L1, L2, L3, L4, L5, L6, L7, L8 = _try_chol_with_bump3x3(
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
        a8,
        eps_rel,
        eps_abs,
    )

    # store L_0 in main and chol cache
    tl.store(M + 0, L0, mask=mask)
    tl.store(M + 1, L1, mask=mask)
    tl.store(M + 2, L2, mask=mask)
    tl.store(M + 3, L3, mask=mask)
    tl.store(M + 4, L4, mask=mask)
    tl.store(M + 5, L5, mask=mask)
    tl.store(M + 6, L6, mask=mask)
    tl.store(M + 7, L7, mask=mask)
    tl.store(M + 8, L8, mask=mask)

    tl.store(CHOL + 0, L0, mask=mask)
    tl.store(CHOL + 1, L1, mask=mask)
    tl.store(CHOL + 2, L2, mask=mask)
    tl.store(CHOL + 3, L3, mask=mask)
    tl.store(CHOL + 4, L4, mask=mask)
    tl.store(CHOL + 5, L5, mask=mask)
    tl.store(CHOL + 6, L6, mask=mask)
    tl.store(CHOL + 7, L7, mask=mask)
    tl.store(CHOL + 8, L8, mask=mask)

    for k in range(1, K):
        u0 = tl.load(Uband + (k - 1) * 3 + 0, mask=mask)
        u1 = tl.load(Uband + (k - 1) * 3 + 1, mask=mask)
        u2 = tl.load(Uband + (k - 1) * 3 + 2, mask=mask)

        c00, c01, c02 = _chol_solve3(L0, L1, L2, L3, L4, L5, L6, L7, L8, u0, 0.0, 0.0)
        c10, c11, c12 = _chol_solve3(L0, L1, L2, L3, L4, L5, L6, L7, L8, 0.0, u1, 0.0)
        c20, c21, c22 = _chol_solve3(L0, L1, L2, L3, L4, L5, L6, L7, L8, 0.0, 0.0, u2)

        lo0 = tl.load(Lband + (k - 1) * 3 + 0, mask=mask)
        lo1 = tl.load(Lband + (k - 1) * 3 + 1, mask=mask)
        lo2 = tl.load(Lband + (k - 1) * 3 + 2, mask=mask)

        base = M + k * 9
        a0 = tl.load(base + 0, mask=mask) - lo0 * c00
        a1 = tl.load(base + 1, mask=mask) - lo0 * c10
        a2 = tl.load(base + 2, mask=mask) - lo0 * c20
        a3 = tl.load(base + 3, mask=mask) - lo1 * c01
        a4 = tl.load(base + 4, mask=mask) - lo1 * c11
        a5 = tl.load(base + 5, mask=mask) - lo1 * c21
        a6 = tl.load(base + 6, mask=mask) - lo2 * c02
        a7 = tl.load(base + 7, mask=mask) - lo2 * c12
        a8 = tl.load(base + 8, mask=mask) - lo2 * c22

        d_prev0 = tl.load(D + (k - 1) * 3 + 0, mask=mask)
        d_prev1 = tl.load(D + (k - 1) * 3 + 1, mask=mask)
        d_prev2 = tl.load(D + (k - 1) * 3 + 2, mask=mask)
        t0, t1, t2 = _chol_solve3(
            L0, L1, L2, L3, L4, L5, L6, L7, L8, d_prev0, d_prev1, d_prev2
        )

        dk0 = tl.load(D + k * 3 + 0, mask=mask) - lo0 * t0
        dk1 = tl.load(D + k * 3 + 1, mask=mask) - lo1 * t1
        dk2 = tl.load(D + k * 3 + 2, mask=mask) - lo2 * t2
        tl.store(D + k * 3 + 0, dk0, mask=mask)
        tl.store(D + k * 3 + 1, dk1, mask=mask)
        tl.store(D + k * 3 + 2, dk2, mask=mask)

        L0, L1, L2, L3, L4, L5, L6, L7, L8 = _try_chol_with_bump3x3(
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            eps_rel,
            eps_abs,
        )

        tl.store(base + 0, L0, mask=mask)
        tl.store(base + 1, L1, mask=mask)
        tl.store(base + 2, L2, mask=mask)
        tl.store(base + 3, L3, mask=mask)
        tl.store(base + 4, L4, mask=mask)
        tl.store(base + 5, L5, mask=mask)
        tl.store(base + 6, L6, mask=mask)
        tl.store(base + 7, L7, mask=mask)
        tl.store(base + 8, L8, mask=mask)

        base_chol = CHOL + k * 9
        tl.store(base_chol + 0, L0, mask=mask)
        tl.store(base_chol + 1, L1, mask=mask)
        tl.store(base_chol + 2, L2, mask=mask)
        tl.store(base_chol + 3, L3, mask=mask)
        tl.store(base_chol + 4, L4, mask=mask)
        tl.store(base_chol + 5, L5, mask=mask)
        tl.store(base_chol + 6, L6, mask=mask)
        tl.store(base_chol + 7, L7, mask=mask)
        tl.store(base_chol + 8, L8, mask=mask)

    # Backward substitution using in-place L_k
    baseL = M + (K - 1) * 9
    L0 = tl.load(baseL + 0, mask=mask)
    L1 = tl.load(baseL + 1, mask=mask)
    L2 = tl.load(baseL + 2, mask=mask)
    L3 = tl.load(baseL + 3, mask=mask)
    L4 = tl.load(baseL + 4, mask=mask)
    L5 = tl.load(baseL + 5, mask=mask)
    L6 = tl.load(baseL + 6, mask=mask)
    L7 = tl.load(baseL + 7, mask=mask)
    L8 = tl.load(baseL + 8, mask=mask)

    d0 = tl.load(D + (K - 1) * 3 + 0, mask=mask)
    d1 = tl.load(D + (K - 1) * 3 + 1, mask=mask)
    d2 = tl.load(D + (K - 1) * 3 + 2, mask=mask)

    x0, x1, x2 = _chol_solve3(L0, L1, L2, L3, L4, L5, L6, L7, L8, d0, d1, d2)
    tl.store(X + (K - 1) * 3 + 0, x0, mask=mask)
    tl.store(X + (K - 1) * 3 + 1, x1, mask=mask)
    tl.store(X + (K - 1) * 3 + 2, x2, mask=mask)

    for k in range(K - 2, -1, -1):
        u0 = tl.load(Uband + k * 3 + 0, mask=mask)
        u1 = tl.load(Uband + k * 3 + 1, mask=mask)
        u2 = tl.load(Uband + k * 3 + 2, mask=mask)

        dk0 = tl.load(D + k * 3 + 0, mask=mask) - u0 * x0
        dk1 = tl.load(D + k * 3 + 1, mask=mask) - u1 * x1
        dk2 = tl.load(D + k * 3 + 2, mask=mask) - u2 * x2

        baseL = M + k * 9
        L0 = tl.load(baseL + 0, mask=mask)
        L1 = tl.load(baseL + 1, mask=mask)
        L2 = tl.load(baseL + 2, mask=mask)
        L3 = tl.load(baseL + 3, mask=mask)
        L4 = tl.load(baseL + 4, mask=mask)
        L5 = tl.load(baseL + 5, mask=mask)
        L6 = tl.load(baseL + 6, mask=mask)
        L7 = tl.load(baseL + 7, mask=mask)
        L8 = tl.load(baseL + 8, mask=mask)

        x0, x1, x2 = _chol_solve3(L0, L1, L2, L3, L4, L5, L6, L7, L8, dk0, dk1, dk2)
        tl.store(X + k * 3 + 0, x0, mask=mask)
        tl.store(X + k * 3 + 1, x1, mask=mask)
        tl.store(X + k * 3 + 2, x2, mask=mask)


@triton.jit
def thomas_bt3_spd_solve_with_chol_kernel(
    L_ptr,  # (B, K-1, 3)
    U_ptr,  # (B, K-1, 3)
    CHOL_ptr,  # (B, K,   9)  cached L_k
    D_ptr,  # (B, K,   3)  RHS
    X_ptr,  # (B, K,   3)  solution
    B,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):  # pragma: no cover
    wid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    fid = wid * BLOCK + lane
    mask = fid < B

    Lband = L_ptr + fid * (K - 1) * 3
    Uband = U_ptr + fid * (K - 1) * 3
    CHOL = CHOL_ptr + fid * K * 9
    D = D_ptr + fid * K * 3
    X = X_ptr + fid * K * 3

    # Make a working copy of RHS
    d0 = tl.load(D + 0, mask=mask)
    d1 = tl.load(D + 1, mask=mask)
    d2 = tl.load(D + 2, mask=mask)
    # store back; we'll keep overwriting D as working buffer
    tl.store(D + 0, d0, mask=mask)
    tl.store(D + 1, d1, mask=mask)
    tl.store(D + 2, d2, mask=mask)

    # Forward sweep: incorporate lower diag bands
    for k in range(1, K):
        baseLchol = CHOL + (k - 1) * 9
        L0 = tl.load(baseLchol + 0, mask=mask)
        L1 = tl.load(baseLchol + 1, mask=mask)
        L2 = tl.load(baseLchol + 2, mask=mask)
        L3 = tl.load(baseLchol + 3, mask=mask)
        L4 = tl.load(baseLchol + 4, mask=mask)
        L5 = tl.load(baseLchol + 5, mask=mask)
        L6 = tl.load(baseLchol + 6, mask=mask)
        L7 = tl.load(baseLchol + 7, mask=mask)
        L8 = tl.load(baseLchol + 8, mask=mask)

        dprev0 = tl.load(D + (k - 1) * 3 + 0, mask=mask)
        dprev1 = tl.load(D + (k - 1) * 3 + 1, mask=mask)
        dprev2 = tl.load(D + (k - 1) * 3 + 2, mask=mask)
        t0, t1, t2 = _chol_solve3(
            L0, L1, L2, L3, L4, L5, L6, L7, L8, dprev0, dprev1, dprev2
        )

        lo0 = tl.load(Lband + (k - 1) * 3 + 0, mask=mask)
        lo1 = tl.load(Lband + (k - 1) * 3 + 1, mask=mask)
        lo2 = tl.load(Lband + (k - 1) * 3 + 2, mask=mask)

        dk0 = tl.load(D + k * 3 + 0, mask=mask) - lo0 * t0
        dk1 = tl.load(D + k * 3 + 1, mask=mask) - lo1 * t1
        dk2 = tl.load(D + k * 3 + 2, mask=mask) - lo2 * t2
        tl.store(D + k * 3 + 0, dk0, mask=mask)
        tl.store(D + k * 3 + 1, dk1, mask=mask)
        tl.store(D + k * 3 + 2, dk2, mask=mask)

    # Backward substitution with cached L_k
    baseLchol = CHOL + (K - 1) * 9
    L0 = tl.load(baseLchol + 0, mask=mask)
    L1 = tl.load(baseLchol + 1, mask=mask)
    L2 = tl.load(baseLchol + 2, mask=mask)
    L3 = tl.load(baseLchol + 3, mask=mask)
    L4 = tl.load(baseLchol + 4, mask=mask)
    L5 = tl.load(baseLchol + 5, mask=mask)
    L6 = tl.load(baseLchol + 6, mask=mask)
    L7 = tl.load(baseLchol + 7, mask=mask)
    L8 = tl.load(baseLchol + 8, mask=mask)

    d0 = tl.load(D + (K - 1) * 3 + 0, mask=mask)
    d1 = tl.load(D + (K - 1) * 3 + 1, mask=mask)
    d2 = tl.load(D + (K - 1) * 3 + 2, mask=mask)

    x0, x1, x2 = _chol_solve3(L0, L1, L2, L3, L4, L5, L6, L7, L8, d0, d1, d2)
    tl.store(X + (K - 1) * 3 + 0, x0, mask=mask)
    tl.store(X + (K - 1) * 3 + 1, x1, mask=mask)
    tl.store(X + (K - 1) * 3 + 2, x2, mask=mask)

    for k in range(K - 2, -1, -1):
        u0 = tl.load(Uband + k * 3 + 0, mask=mask)
        u1 = tl.load(Uband + k * 3 + 1, mask=mask)
        u2 = tl.load(Uband + k * 3 + 2, mask=mask)

        dk0 = tl.load(D + k * 3 + 0, mask=mask) - u0 * x0
        dk1 = tl.load(D + k * 3 + 1, mask=mask) - u1 * x1
        dk2 = tl.load(D + k * 3 + 2, mask=mask) - u2 * x2

        baseLchol = CHOL + k * 9
        L0 = tl.load(baseLchol + 0, mask=mask)
        L1 = tl.load(baseLchol + 1, mask=mask)
        L2 = tl.load(baseLchol + 2, mask=mask)
        L3 = tl.load(baseLchol + 3, mask=mask)
        L4 = tl.load(baseLchol + 4, mask=mask)
        L5 = tl.load(baseLchol + 5, mask=mask)
        L6 = tl.load(baseLchol + 6, mask=mask)
        L7 = tl.load(baseLchol + 7, mask=mask)
        L8 = tl.load(baseLchol + 8, mask=mask)

        x0, x1, x2 = _chol_solve3(L0, L1, L2, L3, L4, L5, L6, L7, L8, dk0, dk1, dk2)
        tl.store(X + k * 3 + 0, x0, mask=mask)
        tl.store(X + k * 3 + 1, x1, mask=mask)
        tl.store(X + k * 3 + 2, x2, mask=mask)


import torch  # noqa: E402
from torch.library import triton_op, wrap_triton  # noqa: E402


# ------------------ low-level fwd op: (x, chol) ----------------------
@triton_op("dendra_triton::solve_bt_spd_fwd_impl", mutates_args={})
def solve_bt_spd_fwd_impl(
    lower: torch.Tensor,  # (B, K-1, 3)
    main: torch.Tensor,  # (B, K,   3,3)
    upper: torch.Tensor,  # (B, K-1, 3)
    rhs: torch.Tensor,  # (B, K,   3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K = rhs.shape[:2]

    lower = lower.contiguous()
    main = main.contiguous()
    upper = upper.contiguous()
    rhs = rhs.contiguous()

    x = torch.empty_like(rhs)
    chol = torch.empty(B, K, 9, device=rhs.device, dtype=rhs.dtype)

    if rhs.dtype == torch.float64:
        eps_rel, eps_abs = 1e-12, 1e-30
    else:
        eps_rel, eps_abs = 1e-6, 1e-18

    grid = ((B + BLOCK_FIBRES - 1) // BLOCK_FIBRES,)

    wrap_triton(thomas_bt3_spd_fwd_kernel)[grid](
        lower.reshape(B, -1),
        main.reshape(B, -1),
        upper.reshape(B, -1),
        rhs.reshape(B, -1),
        x.reshape(B, -1),
        chol.reshape(B, -1),
        B,
        eps_rel,
        eps_abs,
        K=K,
        BLOCK=BLOCK_FIBRES,
        num_warps=1,
        num_stages=4,
    )
    return x, chol


@solve_bt_spd_fwd_impl.register_fake
def _(lower, main, upper, rhs):
    B, K = rhs.shape[:2]
    x = rhs.new_empty(rhs.shape)
    chol = rhs.new_empty((B, K, 9))
    return x, chol


# ------------------ low-level solve-with-chol op ---------------------
@triton_op("dendra_triton::solve_bt_spd_solve_with_chol_impl", mutates_args={})
def solve_bt_spd_solve_with_chol_impl(
    lower: torch.Tensor,  # (B, K-1, 3)
    upper: torch.Tensor,  # (B, K-1, 3)
    chol: torch.Tensor,  # (B, K,   9)
    rhs: torch.Tensor,  # (B, K,   3)
) -> torch.Tensor:
    B, K = rhs.shape[:2]

    lower = lower.contiguous()
    upper = upper.contiguous()
    chol = chol.contiguous()
    rhs = rhs.contiguous()

    x = torch.empty_like(rhs)

    grid = ((B + BLOCK_FIBRES - 1) // BLOCK_FIBRES,)

    wrap_triton(thomas_bt3_spd_solve_with_chol_kernel)[grid](
        lower.reshape(B, -1),
        upper.reshape(B, -1),
        chol.reshape(B, -1),
        rhs.reshape(B, -1),
        x.reshape(B, -1),
        B,
        K=K,
        BLOCK=BLOCK_FIBRES,
        num_warps=1,
        num_stages=4,
    )
    return x


@solve_bt_spd_solve_with_chol_impl.register_fake
def _(lower, upper, chol, rhs):
    return rhs.new_empty(rhs.shape)


def _bt_spd_setup_context(ctx, inputs, output):
    # inputs is the tuple (lower, main, upper, rhs)
    lower, main, upper, rhs = inputs

    # output is the *tuple* returned by solve_bt_spd_fwd_impl: (x, chol)
    x, chol = output

    # stash what we need for backward
    ctx.save_for_backward(lower, main, upper, x, chol)


def _bt_spd_backward(ctx, grad_x, grad_chol_ignored):
    lower, main, upper, x, chol = ctx.saved_tensors

    # 1) Solve A g = grad_x using cached Cholesky factors
    op_solve = torch.ops.dendra_triton.solve_bt_spd_solve_with_chol_impl
    g = op_solve.default(lower, upper, chol, grad_x)

    # 2) Gradients w.r.t. inputs
    grad_rhs = g
    grad_main = -(g.unsqueeze(-1) * x.unsqueeze(-2))  # (B,K,3,3)
    grad_upper = -(g[:, :-1, :] * x[:, 1:, :])  # (B,K-1,3)
    grad_lower = -(g[:, 1:, :] * x[:, :-1, :])  # (B,K-1,3)

    return grad_lower, grad_main, grad_upper, grad_rhs


solve_bt_spd_fwd_impl.register_autograd(
    _bt_spd_backward,
    setup_context=_bt_spd_setup_context,
)


def solve_bt_spd_cuda(lower, main, upper, rhs):
    """
    Public-facing SPD block-Thomas solver.

    Same signature as before:
        lower : (B, K-1, 3)
        main  : (B, K,   3,3)
        upper : (B, K-1, 3)
        rhs   : (B, K,   3)

    Returns:
        x     : (B, K,   3)

    Internally calls dendra_triton::solve_bt_spd_fwd_impl, which
    computes both x and cached Cholesky factors chol and reuses chol
    in backward via solve_bt_spd_solve_with_chol_impl.
    """
    x, _chol = torch.ops.dendra_triton.solve_bt_spd_fwd_impl(lower, main, upper, rhs)
    return x
