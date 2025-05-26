from typing import Tuple

import torch
import torch.nn.functional as F

from axonml.models.mechanisms.compilers import MechCompiler, ImplicitCompiler
from axonml.models.mechanisms.handler.builders import ImplicitHandlerBuilder
from axonml.helpers import IMEM

from .core import Integrator, SCIntegrator


@torch.jit.script
def thomas_tridiag_solve(a, b, c, d):
    # assume a, b, c, d are already contiguous float32
    cp = torch.zeros_like(b)  # (B, K)
    cp[:, :-1] = c  # copy super-diag, last column stays 0

    dp = d.clone()

    # first row
    inv = 1.0 / b[:, 0]
    cp[:, 0] *= inv
    dp[:, 0] *= inv

    # forward sweep
    K = b.size(1)
    for i in range(1, K):
        inv = 1.0 / (b[:, i] - a[:, i - 1] * cp[:, i - 1])
        cp[:, i] *= inv
        dp[:, i] = (dp[:, i] - a[:, i - 1] * dp[:, i - 1]) * inv

    # back substitution (reuse dp as x)
    for i in range(K - 2, -1, -1):
        dp[:, i] -= cp[:, i] * dp[:, i + 1]

    return dp  # (B,K)


@torch.jit.script
def _pad_diagonals(
    a: torch.Tensor, c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert (B,K-1) lower/upper diagonals to (B,K) 0-padded form."""
    B, Km1 = a.shape
    K = Km1 + 1
    a_pad = torch.zeros(B, K, dtype=a.dtype, device=a.device)
    c_pad = torch.zeros(B, K, dtype=c.dtype, device=c.device)
    a_pad[:, 1:] = a
    c_pad[:, :-1] = c
    return a_pad, c_pad


@torch.jit.script
def pcr_tridiag_solve(
    a_in: torch.Tensor, b_in: torch.Tensor, c_in: torch.Tensor, d_in: torch.Tensor
) -> torch.Tensor:
    """
    Parallel cyclic reduction.
    a_in, c_in : (B, K-1)   lower / upper
    b_in, d_in : (B, K)
    """
    B, K = b_in.shape

    # --- pad a, c to shape (B, K) ---
    a = torch.zeros(B, K, dtype=b_in.dtype, device=b_in.device)
    a[:, 1:] = a_in
    c = torch.zeros_like(a)
    c[:, :-1] = c_in
    b = b_in.clone()
    d = d_in.clone()

    stride = 1
    while stride < K:
        left = stride
        right = K - stride  # exclusive upper bound
        if left >= right:  # no rows with both neighbours remain
            break

        idx = torch.arange(left, right, device=b.device)  # always valid now

        # gather neighbours
        a_i, b_i, c_i, d_i = a[:, idx], b[:, idx], c[:, idx], d[:, idx]
        a_l, b_l, c_l, d_l = (
            a[:, idx - stride],
            b[:, idx - stride],
            c[:, idx - stride],
            d[:, idx - stride],
        )
        a_r, b_r, c_r, d_r = (
            a[:, idx + stride],
            b[:, idx + stride],
            c[:, idx + stride],
            d[:, idx + stride],
        )

        alpha = a_i / b_l
        gamma = c_i / b_r

        b[:, idx] = b_i - c_l * alpha - a_r * gamma
        d[:, idx] = d_i - d_l * alpha - d_r * gamma
        a[:, idx] = -a_l * alpha
        c[:, idx] = -c_r * gamma

        stride <<= 1

    # now a[:,1:], c[:,:-1] are guaranteed zero → purely diagonal system
    return d / b


def inv2(mat):                     # mat : (...,2,2)
    a,b = mat[...,0,0], mat[...,0,1]
    c,d = mat[...,1,0], mat[...,1,1]
    det = a*d - b*c
    out = torch.stack([ d, -b, -c, a], dim=-1).reshape_as(mat)
    return out / det[...,None,None]


def inv3(mat: torch.Tensor) -> torch.Tensor:
    """
    Fast inverse of a batch of 3×3 matrices.
    mat shape (...,3,3) – works on any device / dtype
    """

    # ─── entries ──────────────────────────────────────────────────────────
    a, b, c = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    d, e, f = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    g, h, i = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

    # ─── cofactors Cij (sign-included) ────────────────────────────────────
    C00 =   e * i - f * h
    C01 = -(d * i - f * g)
    C02 =   d * h - e * g

    C10 = -(b * i - c * h)
    C11 =   a * i - c * g
    C12 = -(a * h - b * g)

    C20 =   b * f - c * e
    C21 = -(a * f - c * d)
    C22 =   a * e - b * d

    cof = torch.stack([C00, C01, C02,
                       C10, C11, C12,
                       C20, C21, C22], dim=-1).reshape_as(mat)   # (...,3,3)

    # ─── determinant (dot first row with *its* cofactors) ────────────────
    det = a * C00 + b * C01 + c * C02           # <-- fixed line

    # ─── inverse = adj(det) / det ────────────────────────────────────────
    adj = cof.transpose(-1, -2)
    return adj / det[..., None, None]


@torch.jit.script
def pcr_block_tridiag_solve_2(
    a_in: torch.Tensor, b_in: torch.Tensor, c_in: torch.Tensor, d_in: torch.Tensor
) -> torch.Tensor:
    """
    Parallel Cyclic Reduction for block tridiagonal systems.
    Solves (A_in_i * x_i-1) + (B_in_i * x_i) + (C_in_i * x_i+1) = D_in_i for x_i.

    Args:
        a_in (torch.Tensor): Lower block diagonal. Shape: (Batch, K, M, M).
                             a_in[:, k] is the block A_{k+1}.
        b_in (torch.Tensor): Main block diagonal. Shape: (Batch, K, M, M).
                             b_in[:, k] is the block B_k.
        c_in (torch.Tensor): Upper block diagonal. Shape: (Batch, K, M, M).
                             c_in[:, k] is the block C_k.
        d_in (torch.Tensor): Right-hand side. Shape: (Batch, K, M) or (Batch, K, M, 1).
                             d_in[:, k] is the vector D_k.

    Returns:
        torch.Tensor: Solution x. Shape: (Batch, K, M).
    """
    B, K, M, _ = b_in.shape

    a = torch.zeros(B, K, M, M, dtype=b_in.dtype, device=b_in.device)
    c = torch.zeros_like(a)

    a[:, 1:] = a_in
    c[:, :-1] = c_in

    b = b_in.clone()
    d = d_in.clone()

    stride = 1
    while stride < K:
        left = stride
        right = K - stride  # exclusive upper bound
        if left >= right:  # no rows with both neighbours remain
            break

        # Indices of the equations we are updating in this pass
        # These equations must have valid left (i-stride) and right (i+stride) neighbors
        idx = torch.arange(left, right, device=b.device)

        # Gather current coefficients for equations 'i' (idx)
        a_i = a[:, idx]  # Shape: (B, num_active_rows, M, M)
        b_i = b[:, idx]  # Shape: (B, num_active_rows, M, M)
        c_i = c[:, idx]  # Shape: (B, num_active_rows, M, M)
        d_i = d[:, idx]  # Shape: (B, num_active_rows, M)

        # Gather coefficients from left neighbors 'l' (idx - stride)
        a_l = a[:, idx - stride]
        b_l = b[:, idx - stride]
        c_l = c[:, idx - stride]
        d_l = d[:, idx - stride] # Shape: (B, num_active_rows, M)

        # Gather coefficients from right neighbors 'r' (idx + stride)
        a_r = a[:, idx + stride]
        b_r = b[:, idx + stride]
        c_r = c[:, idx + stride]
        d_r = d[:, idx + stride] # Shape: (B, num_active_rows, M)

        # Compute reduction factors (block matrices)
        # Alpha_i = A_i * B_l^{-1}
        # Gamma_i = C_i * B_r^{-1}
        alpha = a_i @ inv2(b_l) # Shape: (B, num_active_rows, M, M)
        gamma = c_i @ inv2(b_r) # Shape: (B, num_active_rows, M, M)

        b[:, idx] = b_i - (alpha @ c_l) - (gamma @ a_r)
        
        d_l_unsqueezed = d_l.unsqueeze(-1) # Shape: (B, num_active_rows, M, 1)
        d_r_unsqueezed = d_r.unsqueeze(-1) # Shape: (B, num_active_rows, M, 1)
        
        d[:, idx] = (
            d_i - (alpha @ d_l_unsqueezed).squeeze(-1) - (gamma @ d_r_unsqueezed).squeeze(-1)
        )
        
        # A_new_i = -Alpha_i * A_l
        a[:, idx] = -(alpha @ a_l)
        
        # C_new_i = -Gamma_i * C_r
        c[:, idx] = -(gamma @ c_r)

        stride <<= 1 # Double the stride: stride = stride * 2

    # After reduction, a and c are effectively zero (or store negligible values),
    # and the system B_final * x = D_final is block-diagonal.
    # Solve B_k * x_k = D_k for each k.
    # d has shape (B, K, M), b has shape (B, K, M, M)
    # We need to solve Bx = D -> x = B^{-1}D
    # torch.linalg.solve expects D to be (..., M, Nrhs), so unsqueeze d
    solution, _ = torch.linalg.solve_ex(b, d.unsqueeze(-1)) # Shape: (B, K, M, 1)
    return solution.squeeze(-1) # Shape: (B, K, M)


@torch.jit.script
def pcr_block_tridiag_solve(
    a_in: torch.Tensor, b_in: torch.Tensor, c_in: torch.Tensor, d_in: torch.Tensor
) -> torch.Tensor:
    """
    Parallel Cyclic Reduction for block tridiagonal systems.
    Solves (A_in_i * x_i-1) + (B_in_i * x_i) + (C_in_i * x_i+1) = D_in_i for x_i.

    Args:
        a_in (torch.Tensor): Lower block diagonal. Shape: (Batch, K, M, M).
                             a_in[:, k] is the block A_{k+1}.
        b_in (torch.Tensor): Main block diagonal. Shape: (Batch, K, M, M).
                             b_in[:, k] is the block B_k.
        c_in (torch.Tensor): Upper block diagonal. Shape: (Batch, K, M, M).
                             c_in[:, k] is the block C_k.
        d_in (torch.Tensor): Right-hand side. Shape: (Batch, K, M) or (Batch, K, M, 1).
                             d_in[:, k] is the vector D_k.

    Returns:
        torch.Tensor: Solution x. Shape: (Batch, K, M).
    """
    B, K, M, _ = b_in.shape

    a = torch.zeros(B, K, M, M, dtype=b_in.dtype, device=b_in.device)
    c = torch.zeros_like(a)

    a[:, 1:] = a_in
    c[:, :-1] = c_in

    b = b_in.clone()
    d = d_in.clone()

    stride = 1
    while stride < K:
        left = stride
        right = K - stride  # exclusive upper bound
        if left >= right:  # no rows with both neighbours remain
            break

        # Indices of the equations we are updating in this pass
        # These equations must have valid left (i-stride) and right (i+stride) neighbors
        idx = torch.arange(left, right, device=b.device)

        # Gather current coefficients for equations 'i' (idx)
        a_i = a[:, idx]  # Shape: (B, num_active_rows, M, M)
        b_i = b[:, idx]  # Shape: (B, num_active_rows, M, M)
        c_i = c[:, idx]  # Shape: (B, num_active_rows, M, M)
        d_i = d[:, idx]  # Shape: (B, num_active_rows, M)

        # Gather coefficients from left neighbors 'l' (idx - stride)
        a_l = a[:, idx - stride]
        b_l = b[:, idx - stride]
        c_l = c[:, idx - stride]
        d_l = d[:, idx - stride] # Shape: (B, num_active_rows, M)

        # Gather coefficients from right neighbors 'r' (idx + stride)
        a_r = a[:, idx + stride]
        b_r = b[:, idx + stride]
        c_r = c[:, idx + stride]
        d_r = d[:, idx + stride] # Shape: (B, num_active_rows, M)

        # Compute reduction factors (block matrices)
        # Alpha_i = A_i * B_l^{-1}
        # Gamma_i = C_i * B_r^{-1}
        alpha = a_i @ inv3(b_l) # Shape: (B, num_active_rows, M, M)
        gamma = c_i @ inv3(b_r) # Shape: (B, num_active_rows, M, M)

        b[:, idx] = b_i - (alpha @ c_l) - (gamma @ a_r)
        
        d_l_unsqueezed = d_l.unsqueeze(-1) # Shape: (B, num_active_rows, M, 1)
        d_r_unsqueezed = d_r.unsqueeze(-1) # Shape: (B, num_active_rows, M, 1)
        
        d[:, idx] = (
            d_i - (alpha @ d_l_unsqueezed).squeeze(-1) - (gamma @ d_r_unsqueezed).squeeze(-1)
        )
        
        # A_new_i = -Alpha_i * A_l
        a[:, idx] = -(alpha @ a_l)
        
        # C_new_i = -Gamma_i * C_r
        c[:, idx] = -(gamma @ c_r)

        stride <<= 1 # Double the stride: stride = stride * 2

    # After reduction, a and c are effectively zero (or store negligible values),
    # and the system B_final * x = D_final is block-diagonal.
    # Solve B_k * x_k = D_k for each k.
    # d has shape (B, K, M), b has shape (B, K, M, M)
    # We need to solve Bx = D -> x = B^{-1}D
    # torch.linalg.solve expects D to be (..., M, Nrhs), so unsqueeze d
    solution, _ = torch.linalg.solve_ex(b, d.unsqueeze(-1)) # Shape: (B, K, M, 1)
    return solution.squeeze(-1) # Shape: (B, K, M)


def assemble_rhs_no_intra(
    vc: torch.Tensor,           # (B,K,M)   voltages at tᶰ
    g_cm: torch.Tensor,         # (B,K)     C_m / dt            (membrane cap)
    gcap: torch.Tensor,         # (B,K,M-1) x_c[j] / dt         (all shell caps)
    g_rad: torch.Tensor,        # (B,K)     xg_out + xcout/dt   (outermost interface)
    ve: torch.Tensor,           # (B,K)     extcell potential
    ires: torch.Tensor,         # (B,K)     residual current
    irev: torch.Tensor,         # (B,K)     reversal potential
) -> torch.Tensor:
    """
    Assemble RHS d for a block-tridiagonal fibre with M unknowns per axial node.

    Returns
    -------
    d : (B,K,M)
    """
    _, _, M = vc.shape
    d = torch.zeros_like(vc)                # (B,K,M)

    # ------------------------------------------------------------------
    # 1) membrane capacitor between v_i (0) and v_e0 (1)
    # ------------------------------------------------------------------
    dv_m = (vc[:, :, 0] - vc[:, :, 1])        # (B,K)
    d[:, :, 0] +=  g_cm * dv_m
    d[:, :, 1] += -g_cm * dv_m

    # residual current on v_i row
    d[:, :, 0] += -irev + ires

    # ------------------------------------------------------------------
    # 2) capacitors between consecutive shells (ve_j ↔ ve_{j+1})
    #    for j = 0 … M-3   (inner interfaces)
    # ------------------------------------------------------------------
    if M > 2:
        gcap_inner = gcap[:, :, :-1]                        # (B,K,M-2)
        dv_shell   = (vc[:, :, 1:-1] - vc[:, :, 2:]) # (B,K,M-2)

        d[:, :, 1:-1] += -gcap_inner * dv_shell    # row ve_j
        d[:, :, 2:  ] +=  gcap_inner * dv_shell    # row ve_{j+1}

    # ------------------------------------------------------------------
    # 3) outermost interface  ve_{M-2}  ↔  bath (ve)
    # ------------------------------------------------------------------
    gcap_out = gcap[:, :, -1]                   # (B,K)   last capacitor
    dv_out   = (vc[:, :, -1] - ve)       # (B,K)

    # capacitor explicit piece
    d[:, :, -1] += -gcap_out * dv_out

    # constant current from bath (conductance + C/dt part)
    d[:, :, -1] +=  g_rad * ve        #  (xg + xcout/dt) * Vbath

    return d


def thomas_block_tridiag_solve(
    a: torch.Tensor,   # (B, K-1, M, M)   lower  blocks  A_{k}
    b: torch.Tensor,   # (B, K  , M, M)   main   blocks  B_{k}
    c: torch.Tensor,   # (B, K-1, M, M)   upper  blocks  C_{k}
    d: torch.Tensor    # (B, K  , M)      right-hand side D_{k}
) -> torch.Tensor:
    """
    Block Thomas algorithm  (serial sweep, O(K M³)).

    Solves  A_k x_{k-1} + B_k x_k + C_k x_{k+1} = D_k   for k = 0…K-1.
    Boundary blocks:  A_0 and C_{K-1} are unused / can be zero.

    Shapes
    ------
    * a : (B,K-1,M,M)  — A_1 … A_{K-1}
    * b : (B,K  ,M,M)
    * c : (B,K-1,M,M)  — C_0 … C_{K-2}
    * d : (B,K  ,M)    — D_0 … D_{K-1}

    Returns
    -------
    x : (B,K,M)
    """
    B, K, M, _ = b.shape
    device, dtype = b.device, b.dtype

    # work copies (we'll overwrite in-place)
    bb = b.clone()                 # (B,K,M,M)
    dd = d.clone()                 # (B,K,M)

    e = torch.eye(M, device=device, dtype=dtype)

    # -------- forward sweep ------------------------------------------
    for k in range(1, K):
        # inv of previous main block
        inv_prev, _ = torch.linalg.solve_ex(bb[:, k-1], e)
        # compute multiplier  G_k = A_k · B_{k-1}^{-1}
        Gk = torch.matmul(a[:, k-1], inv_prev)            # (B,M,M)

        # update current main diagonal   B_k ← B_k - G_k · C_{k-1}
        bb[:, k] = bb[:, k] - torch.matmul(Gk, c[:, k-1])

        # update RHS                     D_k ← D_k - G_k · D_{k-1}
        dd[:, k] = dd[:, k] - torch.matmul(Gk, dd[:, k-1].unsqueeze(-1)).squeeze(-1)

    # -------- backward substitution ----------------------------------
    x = torch.zeros(B, K, M, device=device, dtype=dtype)

    # last block
    x[:, -1], _ = torch.linalg.solve_ex(bb[:, -1], dd[:, -1])

    for k in range(K-2, -1, -1):
        rhs = dd[:, k] - torch.matmul(c[:, k], x[:, k+1].unsqueeze(-1)).squeeze(-1)
        x[:, k], _ = torch.linalg.solve_ex(bb[:, k], rhs)

    return x


@torch.jit.script
def block_thomas_solve(
    lower: torch.Tensor,    # (B, K-1, 3, 3)   A_i
    main:  torch.Tensor,    # (B, K,   3, 3)   B_i   (will be overwritten)
    upper: torch.Tensor,    # (B, K-1, 3, 3)   C_i
    rhs:   torch.Tensor     # (B, K,   3)
) -> torch.Tensor:          # returns y^{n+1}   (B, K, 3)
    B, K, _, _ = main.shape
    x = torch.empty((B, K, 3), dtype=rhs.dtype, device=rhs.device)

    # ---------- forward elimination ----------
    for i in range(1, K):
        # W_i = A_i @ inv(B_{i-1})
        inv_prev, info = torch.linalg.inv_ex(main[:, i-1])
        w = lower[:, i-1].matmul(inv_prev)          # (B,3,3)

        # B_i ← B_i - W_i @ C_{i-1}
        main[:, i] = main[:, i] - w.matmul(upper[:, i-1])

        # d_i ← d_i - W_i @ d_{i-1}
        rhs[:, i] = rhs[:, i] - torch.einsum('bij,bj->bi', w, rhs[:, i-1])

    # ---------- back substitution ----------
    x[:, -1], info = torch.linalg.solve_ex(main[:, -1], rhs[:, -1])
    for i in range(K-2, -1, -1):
        rhs[:, i] = rhs[:, i] - torch.einsum('bij,bj->bi',
                                             upper[:, i], x[:, i+1])
        x[:, i], info = torch.linalg.solve_ex(main[:, i], rhs[:, i])

    return x


@torch.jit.script
def A_mv(v, diag, g_left, g_right):  # v shape (B, K), mV
    out = diag * v
    out[:, :-1] += g_right * v[:, 1:]
    out[:, 1:] += g_left * v[:, :-1]
    return out


@torch.jit.script
def arnoldi(
    v0: torch.Tensor,
    m: int,
    diag: torch.Tensor,
    g_left: torch.Tensor,
    g_right: torch.Tensor,
    V_buf: torch.Tensor,
    H_buf: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    V = V_buf.narrow(2, 0, m).zero_()  # view without new alloc
    H = H_buf.narrow(1, 0, m).narrow(2, 0, m).zero_()

    beta = torch.linalg.norm(v0, dim=1)  # (B,)
    V[:, :, 0] = v0 / beta[:, None]

    for j in range(m):
        w = diag * V[:, :, j]
        w[:, :-1] += g_right * V[:, 1:, j]
        w[:, 1:] += g_left * V[:, :-1, j]

        Vj = V[:, :, : j + 1]  # (B,K,j+1)
        coef = torch.einsum("bkj,bk->bj", Vj, w)  # (B,j+1)
        w -= torch.einsum("bj,bkj->bk", coef, Vj)  # update
        H[:, : j + 1, j] = coef

        if j + 1 < m:
            hnorm_sq = (w * w).sum(dim=1)
            hnorm = torch.sqrt(hnorm_sq + 1e-30)  # avoid 0
            H[:, j + 1, j] = hnorm
            V[:, :, j + 1] = w / hnorm[:, None]

    return V, H, beta


@torch.jit.script
def lanczos(
    v0: torch.Tensor,  # (B, K)
    m: int,
    diag: torch.Tensor,  # (B, K)
    g_left: torch.Tensor,  # (B, K-1)
    g_right: torch.Tensor,  # (B, K-1)
    V_buf: torch.Tensor,  # (B, K, M_max)
    T_buf: torch.Tensor,  # (B, M_max, M_max)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    V = V_buf.narrow(2, 0, m).zero_()
    T = T_buf.narrow(1, 0, m).narrow(2, 0, m).zero_()

    # β₀ = ∥v0∥, and v₁ = v0/β₀
    b0 = torch.linalg.norm(v0, dim=1)
    V[:, :, 0] = v0 / b0[:, None]
    prev_v = torch.zeros_like(v0)  # just store last v_j

    for j in range(m):
        vj = V[:, :, j]  # (B,K)
        # w = A·vj
        w = diag * vj
        w[:, :-1] += g_right * vj[:, 1:]
        w[:, 1:] += g_left * vj[:, :-1]

        if j == 0:
            # First step: subtract α₀ v₁
            a = (vj * w).sum(dim=1)  # (B,)
            w = w - a[:, None] * vj
        else:
            # Two-term orthogonalization in one fused einsum
            V2 = torch.stack([vj, prev_v], dim=2)  # (B,K,2)
            coeff = torch.einsum("bkj,bk->bj", V2, w)  # (B,2)
            a = coeff[:, 0]  # α_j = <vj,w>
            # subtract α_j vj + β_{j-1} prev_v
            w = w - torch.einsum("bj,bkj->bk", coeff, V2)

        # recompute β_j = ∥w∥
        b = torch.linalg.norm(w, dim=1)  # (B,)

        # fill T and next basis vector
        T[:, j, j] = a
        if j + 1 < m:
            T[:, j, j + 1] = b
            T[:, j + 1, j] = b
            V[:, :, j + 1] = w / b[:, None]
            prev_v = vj

    return V, T, b0


@torch.jit.script
def expm_krylov_arnoldi(v, h, m: int, diag, g_left, g_right, V, H):
    V, H, beta = arnoldi(v, m, diag, g_left, g_right, V, H)
    expH = torch.matrix_exp(h * H)  # (B,m,m)
    y = expH[..., 0] * beta.unsqueeze(1)  # (B,m,1)
    return torch.einsum("bkm,bm->bk", V, y)


@torch.jit.script
def expm_krylov_lanczos(v, h, m: int, diag, g_left, g_right, V, H):
    V, H, beta = lanczos(v, m, diag, g_left, g_right, V, H)
    expH = torch.matrix_exp(h * H)  # (B,m,m)
    y = expH[..., 0] * beta.unsqueeze(1)  # (B,m,1)
    return torch.einsum("bkm,bm->bk", V, y)


@torch.jit.script
def phi1_krylov_arnoldi(
    v, h, m: int, diag, g_left, g_right, V_buf, H_buf, eye_m, tol: float = 1e-12
) -> torch.Tensor:
    V, H, beta = arnoldi(v, m, diag, g_left, g_right, V_buf, H_buf)

    H_scaled = h * H

    expH = torch.matrix_exp(H_scaled)  # (B,m,m)
    rhs = (expH - eye_m)[..., :, 0]  # (B,m)

    phi, _ = torch.linalg.solve_ex(H_scaled, rhs)  # (B,m)

    # project back : V · φ · βe₁
    y = phi * beta.unsqueeze(1)  # (B,m,1)
    return torch.einsum("bkm, bm -> bk", V, y)  # (B,K)


@torch.jit.script
def phi1_krylov_arnoldi_g(
    v, h, m: int, diag, g_left, g_right, V_buf, H_buf, eye_m, tol: float = 1e-12
) -> torch.Tensor:
    V, H, beta = arnoldi(v, m, diag, g_left, g_right, V_buf, H_buf)

    H_scaled = h * H

    expH = torch.matrix_exp(H_scaled)  # (B,m,m)
    rhs = (expH - eye_m)[..., :, 0]  # (B,m)

    # LU solve: φ1 = (hH)^{-1}(expH - I)  —— but guard tiny H
    small = H.abs().sum(dim=(1, 2)) < tol  # (B,)

    # allocate output
    phi = torch.empty_like(rhs)

    # 1) tiny blocks  ——  use series I + ½H
    if small.any():
        phi[small] = eye_m + 0.5 * H[small, :, 0]

    # 2) regular blocks —— single solve
    if (~small).any():
        phi[~small], _ = torch.linalg.solve_ex(H_scaled[~small], rhs[~small])

    # project back : V · φ · βe₁
    y = phi * beta.unsqueeze(1)  # (B,m,1)
    return torch.einsum("bkm, bm -> bk", V, y)  # (B,K)


@torch.jit.script
def phi1_krylov_lanczos(
    v, h, m: int, diag, g_left, g_right, V_buf, H_buf, eye_m, tol: float = 1e-12
) -> torch.Tensor:
    V, H, beta = lanczos(v, m, diag, g_left, g_right, V_buf, H_buf)

    H_scaled = h * H

    expH = torch.matrix_exp(H_scaled)  # (B,m,m)
    rhs = (expH - eye_m)[..., :, 0].contiguous()  # (B,m)

    phi, _ = torch.linalg.solve_ex(H_scaled, rhs)

    # project back : V · φ · βe₁
    y = phi * beta.unsqueeze(1)  # (B,m,1)
    return torch.einsum("bkm, bm -> bk", V, y)  # (B,K)


@torch.jit.script
def phi1_krylov_lanczos_g(
    v, h, m: int, diag, g_left, g_right, V_buf, H_buf, eye_m, tol: float = 1e-12
) -> torch.Tensor:
    V, H, beta = lanczos(v, m, diag, g_left, g_right, V_buf, H_buf)

    H_scaled = h * H

    expH = torch.matrix_exp(H_scaled)  # (B,m,m)
    rhs = (expH - eye_m)[..., :, 0].contiguous()  # (B,m)

    # LU solve: φ1 = (hH)^{-1}(expH - I)  —— but guard tiny H
    small = H.abs().sum(dim=(1, 2)) < tol  # (B,)

    # allocate output once
    phi = torch.empty_like(rhs)

    # 1) tiny blocks  ——  use series I + ½H
    if small.any():
        phi[small] = eye_m + 0.5 * H[small, :, 0]

    # 2) regular blocks —— single solve
    if (~small).any():
        phi[~small], _ = torch.linalg.solve_ex(H_scaled[~small], rhs[~small])

    # project back : V · φ · βe₁
    y = phi * beta.unsqueeze(1)  # (B,m,1)
    return torch.einsum("bkm, bm -> bk", V, y)  # (B,K)


def build_block_matrix(
    g_i: torch.Tensor,                  # (B, K-1)   axial conductance, axoplasm
    g_axial: torch.Tensor,              # (B, K-1, M-1) axial conductances, all shells
    c_rad_m: torch.Tensor,              # (B, K)     C_m / dt   (NO g_m)
    g_rad_shell: torch.Tensor = None,   # (B, K, M-1) xg[j] + xc[j]/dt   (shell j-1 ↔ j)
) -> torch.Tensor:
    """
    Assemble block-diagonal slices B_k for a cable with vi + (M-1) extracellular shells.

    Parameters
    ----------
    g_i : (B, K-1)             axial conductance of the axoplasm.
    g_axial : (B, K-1, M-1)    axial conductances of every extracellular shell.
    c_rad_m : (B, K)           capacitor term C_m / dt (no leak yet).
    g_rad_shell : (B, K, M-1), optional
        Radial conductance between shell j-1 and shell j ( 0 ≤ j ≤ M-2 ).
        If `None`, zeros are assumed (no radial leaks / capacitors).

    Returns
    -------
    B : (B, K, M, M)          block matrices ready for the solver.
                              They already contain C_m/dt but *not* g_m.
    """
    B, Km1 = g_i.shape
    K = Km1 + 1      
    M = g_axial.shape[2] + 1      # 1 intracellular + (M-1) shells

    print(f"Building block matrix for {B} batches, {K} compartments, {M} shells")

    device, dtype = g_i.device, g_i.dtype
    zeros_BK   = lambda: torch.zeros(B, K,      device=device, dtype=dtype)
    zeros_BKM1 = lambda: torch.zeros(B, K, M-1, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 1.  left / right axial conductances for every node
    # ------------------------------------------------------------------
    giL = zeros_BK();  giR = zeros_BK()
    giL[:, 1:]  = g_i
    giR[:, :-1] = g_i

    geL = zeros_BKM1(); geR = zeros_BKM1()
    geL[:, 1:, :]  = g_axial
    geR[:, :-1, :] = g_axial

    assert (giL[:,0] == 0).all() and (giR[:,-1] == 0).all()
    assert (geL[:,0,:] == 0).all() and (geR[:,-1,:] == 0).all()

    # ------------------------------------------------------------------
    # 2.  radial conductances between shells (if any)
    # ------------------------------------------------------------------
    if g_rad_shell is None:
        g_rad_shell = zeros_BKM1()           # default: no shell-to-shell coupling

    # inward radial array:  concat c_rad_m for vi↔ve0 as "index 0"
    # shape (B, K, M-1)  where entry 0 is vi↔ve0, entries 1… for shells
    g_inward = torch.cat([c_rad_m.unsqueeze(-1), g_rad_shell], dim=-1)

    # ------------------------------------------------------------------
    # 3.  allocate block matrix  (B, K, M, M)
    # ------------------------------------------------------------------
    Bmat = torch.zeros(B, K, M, M, device=device, dtype=dtype)

    # ---------------- diagonals ---------------------------------------
    # (0) intracellular node
    Bmat[..., 0, 0] = giL + giR + g_inward[..., 0]

    # (1 … M-1) shell diagonals
    axial_in  = geL + geR                      # (B, K, M-1)
    outward   = torch.zeros_like(axial_in)     # default 0

    if M > 2:
        # outward radial for shell j is inward radial of shell j+1
        outward[..., :-1] = g_inward[..., 2:]  # shift left by 1

        diag_shells = axial_in + g_inward[..., 1:] + outward   # (B, K, M-1)

        # write them without using .diagonal -------------------------------
        idx = torch.arange(1, M, device=device)   # [1, 2, …, M-1]
        Bmat[..., idx, idx] = diag_shells          # shape matches (B,K,M-1)

    else:  # ----- single extracellular shell (ve0) --------------------
        # axial_in, g_inward[...,1] are both (B,K,1) → squeeze last dim
        Bmat[..., 1, 1] = (axial_in.squeeze(-1)
                        + g_inward[..., 1].squeeze(-1))

    # ---------------- off-diagonals (symmetric) -----------------------
    Bmat[..., 0, 1] = Bmat[..., 1, 0] = -g_inward[..., 0]  # vi ↔ ve0

    if M > 2:
        # shell j-1 ↔ shell j   for j = 1 … M-2
        g_r = g_inward[..., 1:]                     # (B,K,M-1)
        Bmat[..., idx[:-1], idx[1:]] = -g_r[..., :-1]
        Bmat[..., idx[1:], idx[:-1]] = -g_r[..., :-1]

    return Bmat


class _krylov_etd1(Integrator):
    """
    IMEX ETD1 method using Krylov subspace for the matrix exponential.
    """

    compiler = ImplicitCompiler
    builder = ImplicitHandlerBuilder
    is_df = False

    def __init__(
        self, model, mech, m: int = 4, method="arnoldi", guard=False, imem=None
    ):
        super().__init__(model, mech, imem)
        self.m = m

        B = model.n_ax
        K = model.n_comp
        self.B = B
        self.K = K

        if method == "arnoldi":
            self._expm = expm_krylov_arnoldi
            if guard:
                self._phi1 = phi1_krylov_arnoldi_g
            else:
                self._phi1 = phi1_krylov_arnoldi
        elif method == "lanczos":
            self._expm = expm_krylov_lanczos
            if guard:
                self._phi1 = phi1_krylov_lanczos_g
            else:
                self._phi1 = phi1_krylov_lanczos
        else:
            raise ValueError(f"Unknown method: {method}")

        self.register_buffer("kernel", torch.tensor([1.0, -2.0, 1.0]).view(1, 1, 3))
        self.register_buffer("diag", torch.tensor(0.0))
        self.register_buffer("g_ax", torch.tensor(0.0))
        self.register_buffer("cm_inv", torch.tensor(0.0))
        self.register_buffer("scale", torch.tensor(0.0))
        self.register_buffer("V_buf", torch.zeros(B, K, m))
        self.register_buffer("H_buf", torch.zeros(B, m, m))
        self.register_buffer("eye_m", torch.eye(m))

    def initialize(self, model, dt):
        B, K = model.n_ax, model.n_comp
        radius_cm = 1e-4 * model.diam.unsqueeze(1) / 2.0
        dx_cm = 1e-4 * model.dx

        # surface area
        A_mem = 2 * torch.pi * radius_cm * dx_cm
        cm = 1e-6 * model.cm
        cm = cm * A_mem  # F

        g_ax = torch.pi * radius_cm**2 / (model.rhoa * dx_cm)
        g_ax_over_Cm = g_ax / cm

        self.g_ax = g_ax_over_Cm

        diag = torch.zeros(B, K, device=model.device())
        diag[:, :-1] -= self.g_ax  # − g_right
        diag[:, 1:] -= self.g_ax  # − g_left

        self.diag = diag
        self.cm_inv = 1.0 / cm
        self.scale = A_mem * self.cm_inv

    def step(self, model, ve, dt, t_ind):
        model.v = self._step_no_intra(model.v, ve, dt, model.temp_c)

    def step_intra(self, model, ve, intra, dt, t_ind):
        model.v = self._step_intra(model.v, ve, dt, model.temp_c, intra)

    @torch.jit.script_method
    def _step_no_intra(self, v, ve, dt, temp):
        dt_s = dt * 1e-3
        self.mech.advance(v, dt, temp)
        f_n = (self.mech.irev().squeeze(1) - self.mech.i(v).squeeze(1)) * self.scale
        gtot = self.mech.gtot(v).squeeze(1) * self.scale
        S = F.conv1d(ve, self.kernel, padding=1).squeeze(1)
        S[:, 0] = ve[:, 0, 1] - ve[:, 0, 0]  # fix boundary left
        S[:, -1] = ve[:, 0, -2] - ve[:, 0, -1]  # fix boundary right
        S *= self.g_ax
        f_n = f_n + S
        diag = self.diag - gtot
        v_lin = self._expm(
            v.squeeze(1),
            dt_s,
            self.m,
            diag,
            self.g_ax,
            self.g_ax,
            self.V_buf,
            self.H_buf,
        )
        v_nl = dt_s * self._phi1(
            f_n,
            dt_s,
            self.m,
            diag,
            self.g_ax,
            self.g_ax,
            self.V_buf,
            self.H_buf,
            self.eye_m,
        )
        return (v_lin + v_nl).unsqueeze(1)

    @torch.jit.script_method
    def _step_intra(self, v, ve, dt, temp, intra):
        dt_s = dt * 1e-3
        self.mech.advance(v, dt, temp)
        f_n = (self.mech.irev().squeeze(1) - self.mech.i(v).squeeze(1)) * self.scale
        gtot = self.mech.gtot(v).squeeze(1) * self.scale
        S = F.conv1d(ve, self.kernel, padding=1).squeeze(1)
        S[:, 0] = ve[:, 0, 1] - ve[:, 0, 0]  # fix boundary left
        S[:, -1] = ve[:, 0, -2] - ve[:, 0, -1]  # fix boundary right
        S *= self.g_ax
        f_n = f_n + S - intra.squeeze(1)
        diag = self.diag - gtot
        v_lin = self._expm(
            v.squeeze(1),
            dt_s,
            self.m,
            diag,
            self.g_ax,
            self.g_ax,
            self.V_buf,
            self.H_buf,
        )
        v_nl = dt_s * self._phi1(
            f_n,
            dt_s,
            self.m,
            diag,
            self.g_ax,
            self.g_ax,
            self.V_buf,
            self.H_buf,
            self.eye_m,
        )
        return (v_lin + v_nl).unsqueeze(1)

    def detach(self, model):
        model.v = model.v.detach()
        if self.imem:
            model.i_membrane.detach_()
        self.mech.detach()

    def init_v(self, model):
        model.v[:] = model.v_init
        model.v = model.v.detach()
        if self.imem:
            model.i_membrane[:] = 0.0
            model.i_membrane.detach_()


class _bwd_euler_sc(SCIntegrator):
    """
    Implicit Euler method.
    """

    compiler = ImplicitCompiler
    builder = ImplicitHandlerBuilder
    is_df = False

    def __init__(self, model, mech, imem=None, N=1, P=1, C=1):
        if model.n_comp != 1:
            raise ValueError(
                "Backward Euler currently only supports single compartment models."
            )
        super().__init__(model, mech, imem, N, P, C)

    @torch.jit.ignore
    def initialize(self, model, dt):
        self.cmdt = (1e-6 * model.cm) / (1e-3 * dt)

    def step(self, model, dt, t_ind):
        model.v = self._step_no_intra(model.v, dt, model.temp_c)

    def step_intra(self, model, intra, dt, t_ind):
        model.v = self._step_intra(model.v, dt, model.temp_c, intra)

    @torch.jit.script_method
    def _step_no_intra(self, v, dt, temp):
        self.mech.advance(v, dt, temp)
        self.mech.i(v)
        i = self.mech.irev()
        gtot = self.mech.gtot(v)
        return (self.cmdt * v + i) / (self.cmdt + gtot)

    @torch.jit.script_method
    def _step_intra(self, v, dt, temp, intra):
        i = -self.mech.i(v) + self.mech.irev() + intra
        gtot = self.mech.gtot(v)
        self.mech.advance(v, dt, temp)
        return (self.cmdt * v + i) / (self.cmdt + gtot)


class _bwd_euler_ub(Integrator):
    """
    Implicit Euler method.
    """

    compiler = ImplicitCompiler
    builder = ImplicitHandlerBuilder
    is_df = False

    def __init__(self, model, mech, method="pcr", **kw):
        super().__init__(model, mech, **kw)
        B, K = model.n_ax, model.n_comp
        self.register_buffer("kernel", torch.tensor([1.0, -2.0, 1.0]).view(1, 1, 3))
        # Buffers for diffusive diag, axonal conductance, membrane scale
        self.register_buffer("diag_base", torch.zeros(B, K))
        self.register_buffer("g_ax", torch.zeros(B, K))
        self.register_buffer("cm_inv", torch.zeros(B, K))
        self.register_buffer("scale", torch.zeros(B, K))

        if method == "pcr":
            self._solve = pcr_tridiag_solve
        elif method == "thomas":
            self._solve = thomas_tridiag_solve
        else:
            raise ValueError(f"Unknown method: {method}")

    def initialize(self, model, dt):
        B, K = model.n_ax, model.n_comp
        # same geometry & membrane setup as ETD1
        radius_cm = 1e-4 * model.diam.unsqueeze(1) / 2.0
        dx_cm = 1e-4 * model.dx
        A_mem = 2 * torch.pi * radius_cm * dx_cm
        cm = 1e-6 * model.cm * A_mem
        g_ax = torch.pi * radius_cm**2 / (model.rhoa * dx_cm)
        g_ax_over_Cm = g_ax / cm
        # diffusive base diag entries
        diag = torch.zeros(B, K, device=model.device())
        diag[:, :-1] -= g_ax_over_Cm
        diag[:, 1:] -= g_ax_over_Cm
        self.diag_base.copy_(diag)
        self.g_ax.copy_(g_ax_over_Cm)
        self.cm_inv.copy_(1.0 / cm)
        self.scale.copy_(A_mem * self.cm_inv)

    def step(self, model, ve, dt, t_ind):
        model.v = self._step_no_intra(model.v, ve, dt, model.temp_c)

    def step_intra(self, model, ve, intra, dt, t_ind):
        model.v = self._step_intra(model.v, ve, dt, model.temp_c, intra)

    @torch.jit.script_method
    def _step_no_intra(self, v, ve, dt, temp):
        dt_s = dt * 1e-3
        # advance gating
        self.mech.advance(v, dt, temp)

        # nonlinear residual currents
        i_res = self.mech.i(v).squeeze(1)  # (B,K)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v).squeeze(1) * self.scale  # (B,K)
        irev = self.mech.irev().squeeze(1)  # (B,K)

        # diffusive extracellular coupling
        S = torch.nn.functional.conv1d(ve, self.kernel, padding=1).squeeze(1)
        S[:, 0] = ve[:, 0, 1] - ve[:, 0, 0]
        S[:, -1] = ve[:, 0, -2] - ve[:, 0, -1]
        S = S * self.g_ax

        # form RHS: v_n + dt*(linear_reversal + S - residual)
        f_n = irev * self.scale + S - i_res * self.scale
        RHS = v.squeeze(1) + dt_s * f_n

        # build tridiagonal system M v_{n+1} = RHS
        A_diag = self.diag_base - gtot
        main = 1.0 - dt_s * A_diag
        lower = -dt_s * self.g_ax[:, 1:]
        upper = -dt_s * self.g_ax[:, :-1]

        # a: (B, K-1), b: (B, K), c: (B, K-1), d: (B, K)
        inv_b = 1.0 / main  # shape (B, K)

        # scale the three diagonals
        a_s = lower * inv_b[:, 1:]  # each row i: divide a_i by b_i
        c_s = upper * inv_b[:, :-1]  # divide c_i by b_i
        b_s = torch.ones_like(main)

        # scale RHS
        d_s = RHS * inv_b  # divide each equation by its pivot b_i

        # solve tridiagonal system
        v_np1 = self._solve(a_s, b_s, c_s, d_s)  # (B, K)
        return v_np1.unsqueeze(1)

    @torch.jit.script_method
    def _step_intra(self, v, ve, dt, temp, intra):
        dt_s = dt * 1e-3
        # advance gating
        self.mech.advance(v, dt, temp)

        # nonlinear residual currents
        i_res = self.mech.i(v).squeeze(1)  # (B,K)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v).squeeze(1) * self.scale  # (B,K)
        irev = self.mech.irev().squeeze(1)  # (B,K)

        # diffusive extracellular coupling
        S = torch.nn.functional.conv1d(ve, self.kernel, padding=1).squeeze(1)
        S[:, 0] = ve[:, 0, 1] - ve[:, 0, 0]
        S[:, -1] = ve[:, 0, -2] - ve[:, 0, -1]
        S = S * self.g_ax

        # form RHS: v_n + dt*(linear_reversal + S - residual)
        f_n = irev * self.scale + S - i_res * self.scale - intra.squeeze(1)
        RHS = v.squeeze(1) + dt_s * f_n

        # build tridiagonal system M v_{n+1} = RHS
        A_diag = self.diag_base - gtot
        main = 1.0 - dt_s * A_diag
        lower = -dt_s * self.g_ax[:, 1:]
        upper = -dt_s * self.g_ax[:, :-1]

        # a: (B, K-1), b: (B, K), c: (B, K-1), d: (B, K)
        inv_b = 1.0 / main  # shape (B, K)

        # scale the three diagonals
        a_s = lower * inv_b[:, 1:]  # each row i: divide a_i by b_i
        c_s = upper * inv_b[:, :-1]  # divide c_i by b_i
        b_s = torch.ones_like(main)

        # scale RHS
        d_s = RHS * inv_b  # divide each equation by its pivot b_i

        # solve tridiagonal system
        v_np1 = self._solve(a_s, b_s, c_s, d_s)  # (B, K)
        return v_np1.unsqueeze(1)

    def detach(self, model):
        model.v = model.v.detach()
        if self.imem:
            model.i_membrane.detach_()
        self.mech.detach()

    def init_v(self, model):
        model.v[:] = model.v_init
        model.v = model.v.detach()
        if self.imem:
            model.i_membrane[:] = 0.0
            model.i_membrane.detach_()


class _bwd_euler_bt(torch.jit.ScriptModule):
    """
    Implicit Euler method for block tridiagonal system.
    """

    compiler = ImplicitCompiler
    builder = ImplicitHandlerBuilder
    is_df = False

    def __init__(self, model, mech, method="pcr", **kwargs):
        super().__init__()
        self.mech = mech

        B, K, M = model.n_ax, model.n_comp, model.n_layers
        M = M + 1
        self.B = B
        self.K = K
        self.M = M

        self.register_buffer("upper", torch.zeros(B, K-1, M, M))
        self.register_buffer("lower", torch.zeros(B, K-1, M, M))
        self.register_buffer("maind", torch.zeros(B, K,   M, M))
        self.register_buffer("area", torch.zeros(B, K))

        self.register_buffer("cm_dt", torch.zeros(B, K))
        self.register_buffer("xc_dt", torch.zeros(B, K))
        self.register_buffer("xg", torch.zeros(B, K, M))

        self.register_buffer("i_membrane", torch.zeros(1))
        
        model.register_buffer("vc", torch.zeros(B, K, M))
        model.vc[..., 0] = model.v_init

        self._solve = thomas_block_tridiag_solve
        
    @classmethod
    def shape(cls, n_ax, n_comp):
        return (n_ax, n_comp)

    def init_v(self, model):
        model.vc[:] = 0.0
        model.vc[..., 0] = model.v_init
        model.vc = model.vc.detach()

    def detach(self, model):
        model.vc = model.vc.detach()
        self.mech.detach()

    def initialize(self, model, dt):
        dt = dt * 1e-3  # convert ms to seconds
        B, K, M = self.B, self.K, self.M

        L = model.L * 1e-4                              # cm
        diam = model.diam * 1e-4                        # cm
        radius = 0.5 * diam                             # cm

        ri = model.rhoa * L / (torch.pi * radius**2)    # Ohm
        ri = 0.5 * (ri[:, :-1] + ri[:, 1:])             # (B, K-1)
        gi = 1 / ri                                     # S
        gi = F.pad(gi, (1, 1))                          # (B, K+1)        

        raxial = model.xraxial * L.unsqueeze(-1) * 1e6         # Ohm
        raxial = 0.5 * (raxial[:, :-1, :] + raxial[:, 1:, :])
        gaxial = 1 / raxial                                    # S
        zeros = torch.zeros_like(gaxial[:, :1])      # same dtype/device, length 1 along dim=1
        gaxial = torch.cat([zeros, gaxial, zeros], dim=1)

        gx0, gx1 = gaxial.unbind(-1)  # (B, K+1)

        area = torch.pi * diam * L

        cm = model.cm * 1e-6 * area                     # F
        cm_dt = cm / dt                                 # F/s
        xc = model.xc * 1e-6 * area.unsqueeze(-1)       # F
        xc_dt = xc / dt                                 # F/s
        xc0, xc1 = xc_dt.unbind(-1)                     # (B, K)
        xg = model.xg * area.unsqueeze(-1)              # S
        xg0, xg1 = xg.unbind(-1)                        # (B, K)

        main  = torch.zeros((B, K,   M, M), device=model.device(), dtype=model.dtype())
        lower = torch.zeros((B, K-1, M, M), device=model.device(), dtype=model.dtype())
        upper = torch.zeros((B, K-1, M, M), device=model.device(), dtype=model.dtype())

        for i in range(K):
            # axial conductances to neighbours; 0 at sealed ends
            ga_L  = gi[:, i]     if i > 0     else 0.0
            ga_R  = gi[:, i+1]   if i < K-1   else 0.0
            gx0_L = gx0[:, i]    if i > 0     else 0.0
            gx0_R = gx0[:, i+1]  if i < K-1   else 0.0
            gx1_L = gx1[:, i]    if i > 0     else 0.0
            gx1_R = gx1[:, i+1]  if i < K-1   else 0.0

            # convenient aliases for per-node radial parameters
            cm_i            = cm_dt[:, i]
            xc0_i, xg0_i    = xc0[:, i],   xg0[:, i]
            xc1_i, xg1_i    = xc1[:, i],   xg1[:, i]

            # ── MAIN block B_i (B×3×3) ───────────────────────────────────────────
            Bi = main[:, i]

            # row 0 – v
            Bi[:, 0, 0] = cm_i + ga_L + ga_R
            Bi[:, 0, 1] = -(cm_i)

            # row 1 – ve0
            Bi[:, 1, 0] = -(cm_i)
            Bi[:, 1, 1] = (cm_i + xc0_i) + xg0_i + gx0_L + gx0_R
            Bi[:, 1, 2] = -(xc0_i + xg0_i)

            # row 2 – ve1
            Bi[:, 2, 1] = -(xc0_i + xg0_i)
            Bi[:, 2, 2] = (xc0_i + xc1_i) + xg0_i + xg1_i + gx1_L + gx1_R

            # ── off-diagonal blocks (purely diagonal) ────────────────────────────
            if i > 0:
                lower[:, i-1, 0, 0] = -ga_L
                lower[:, i-1, 1, 1] = -gx0_L
                lower[:, i-1, 2, 2] = -gx1_L
            if i < K-1:
                upper[:, i,   0, 0] = -ga_R
                upper[:, i,   1, 1] = -gx0_R
                upper[:, i,   2, 2] = -gx1_R

        self.area = area
        self.cm_dt = cm_dt
        self.xc_dt = xc_dt
        self.xg = xg

        self.maind = main
        self.lower = lower
        self.upper = upper
        
    def step(self, model, ve, dt, t_ind):
        model.vc = self._step_no_intra(model.vc, ve, dt, model.temp_c)

    @torch.jit.script_method
    def _step_no_intra(self, vc, ve, dt, temp):
        ve = ve.squeeze(1)

        vi, ve0, ve1 = vc.unbind(-1)
        xc0, xc1 = self.xc_dt.unbind(-1)
        xg1 = self.xg[..., 1]

        v = vc[..., 0] - vc[..., 1]

        # advance gating
        self.mech.advance(v, dt, temp)

        ires = self.mech.i(v)

        # nonlinear residual currents
        ires = ires * self.area # (B, K)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v).squeeze(0) * self.area
        irev = self.mech.irev().squeeze(0)  * self.area # (B, K)

        B = self.maind.clone()  # (B, K, M, M)
        B[..., 0, 0] += gtot
        B[..., 1, 1] += gtot
        B[..., 0, 1] -= gtot
        B[..., 1, 0] -= gtot

        D = assemble_rhs(
            self.B,
            self.K,
            self.M,
            self.cm_dt,
            vi,
            ve0,
            ve1,
            xc0,
            xc1,
            xg1,
            ve,
            ires,
            irev,
        )

        # solve tridiagonal system
        vc = self._solve(self.lower, B, self.upper, D)  # (B, K)
        return vc

def assemble_rhs(
    B:int, K:int, M:int, cm_dt, vi, ve0, ve1, xc0, xc1, xg1, e_ext, ires, irev, 
):
    rhs = torch.zeros(B, K, M, device=vi.device, dtype=vi.dtype)
    rhs[..., 0] = cm_dt * (vi - ve0) + irev - ires
    rhs[..., 1] = -cm_dt * vi + (cm_dt + xc0) * ve0 - xc0 * ve1 - irev + ires
    rhs[..., 2] = -xc0 * ve0 + (xc0 + xc1) * ve1 + xg1 * e_ext
    return rhs

from torch import Tensor

@torch.jit.script
def right_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # solves  A · X = B   for X,  without forming A⁻¹
    return torch.linalg.solve_ex(A, B)[0]


def neighbour_product_m1(off: Tensor, x: Tensor) -> Tensor:
    out = torch.zeros_like(x)             # left neighbour exists for i ≥ 1
    out[:, 1:] = torch.einsum('bkmn,bkn->bkm', off[:, 1:], x[:, :-1])
    return out

def neighbour_product_p1(off: Tensor, x: Tensor) -> Tensor:
    out = torch.zeros_like(x)             # right neighbour exists for i ≤ K-2                     # right neighbour exists for i ≤ K-2
    out[:, :-1] = torch.einsum('bkmn,bkn->bkm', off[:, :-1], x[:, 1:])
    return out


@torch.jit.script
def pcr_block_tridiag_stable(
    A_in: torch.Tensor,    # (B,K-1,3,3)  lower
    B_in: torch.Tensor,    # (B,K,3,3)  main
    C_in: torch.Tensor,    # (B,K-1,3,3)  upper
    D_in: torch.Tensor     # (B,K,3)    rhs
) -> torch.Tensor:         # --------   solution  (B,K,3)
    
    B,K,M,_ = B_in.shape

    A_full = torch.zeros_like(B_in)        # (B, K, 3, 3)
    C_full = torch.zeros_like(B_in)        # (B, K, 3, 3)
    A_full[:, 1:]  = A_in                  # A₁ … A_{K-1}
    C_full[:, :-1] = C_in                  # C₀ … C_{K-2}
    
    A = torch.zeros(B, K, M, M, dtype=B_in.dtype, device=B_in.device)
    C = torch.zeros_like(A)
    A[:,1:] = A_in
    C[:,:-1] = C_in

    # ---------- 1. diagonal scaling -----------------------------------------
    # scale rows so that the main block is the identity:  X_i = B_i⁻¹
    X = torch.linalg.inv(B_in)          # (B,K,3,3)
    A[:, 1:] = torch.matmul(A[:, 1:],  X[:, :-1])  # A_i   · B_{i-1}⁻¹   (i ≥ 1)
    C[:, :-1] = torch.matmul(C[:, :-1], X[:, 1:])  # C_i   · B_{i+1}⁻¹   (i ≤ K-2)

    # rows i = 0 and i = K-1 keep A=0 or C=0 because of the sealed ends
    D = torch.einsum('bkmn,bkn->bkm', X, D_in)     # B⁻¹ · D
    B = torch.eye(3, device=B_in.device, dtype=B_in.dtype).expand_as(B_in).clone()

    # ---------- 2. cyclic reduction -----------------------------------------
    stride = 1
    while stride < K:
        if stride >= K - stride:  # no rows with both neighbours remain
            break
        idx = torch.arange(stride, K-stride, device=B_in.device)

        A_i, C_i = A[:,idx], C[:,idx]
        A_L, C_L = A[:,idx-stride], C[:,idx-stride]
        A_R, C_R = A[:,idx+stride], C[:,idx+stride]
        D_i      = D[:,idx]
        D_L      = D[:,idx-stride]
        D_R      = D[:,idx+stride]

        # alpha = A_i          (since B_L == I after scaling)
        # gamma = C_i          (since B_R == I)
        alpha, gamma = A_i, C_i

        B[:,idx]  = B[:,idx] - alpha @ C_L - gamma @ A_R
        delta_L = (alpha @ D_L.unsqueeze(-1)).squeeze(-1)
        delta_R = (gamma @ D_R.unsqueeze(-1)).squeeze(-1)
        D[:, idx]  = D_i - delta_L - delta_R
        A[:,idx]  = -alpha @ A_L
        C[:,idx]  = -gamma @ C_R

        stride <<= 1

    # ---------- 3. solve the decoupled 3×3 blocks ----------------------------
    X_sol = right_solve(B, D.unsqueeze(-1)).squeeze(-1)     # (B,K,3)

    # left end (i = 0)
    rhs0 = D_in[:, 0] - torch.einsum('bmn,bn->bm', C_full[:, 0], X_sol[:, 1])
    X_sol[:, 0] = torch.linalg.solve(B_in[:, 0], rhs0)

    # right end (i = K-1)
    rhsL = D_in[:, -1] - torch.einsum('bmn,bn->bm', A_full[:, -1], X_sol[:, -2])
    X_sol[:, -1] = torch.linalg.solve(B_in[:, -1], rhsL)

    return X_sol