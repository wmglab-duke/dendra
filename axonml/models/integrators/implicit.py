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
