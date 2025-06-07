from typing import Tuple
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import axonml_solvers
    AXONML_SOLVERS_AVAILABLE = True
except ImportError:
    AXONML_SOLVERS_AVAILABLE = False

from axonml.models.mechanisms.compilers import MechCompiler, ImplicitCompiler
from axonml.models.mechanisms.handler.builders import ImplicitHandlerBuilder
from axonml.helpers import IMEM

from .core import Integrator, SCIntegrator


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


def A_mv(v, diag, g_left, g_right):  # v shape (B, K), mV
    out = diag * v
    out[:, :-1] += g_right * v[:, 1:]
    out[:, 1:] += g_left * v[:, :-1]
    return out


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


def expm_krylov_arnoldi(v, h, m: int, diag, g_left, g_right, V, H):
    V, H, beta = arnoldi(v, m, diag, g_left, g_right, V, H)
    expH = torch.matrix_exp(h * H)  # (B,m,m)
    y = expH[..., 0] * beta.unsqueeze(1)  # (B,m,1)
    return torch.einsum("bkm,bm->bk", V, y)


def expm_krylov_lanczos(v, h, m: int, diag, g_left, g_right, V, H):
    V, H, beta = lanczos(v, m, diag, g_left, g_right, V, H)
    expH = torch.matrix_exp(h * H)  # (B,m,m)
    y = expH[..., 0] * beta.unsqueeze(1)  # (B,m,1)
    return torch.einsum("bkm,bm->bk", V, y)


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
        radius_cm = 1e-4 * model.diam / 2.0
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

    def initialize(self, model, dt):
        self.cmdt = (1e-6 * model.cm) / (1e-3 * dt)

    def step(self, model, dt, t_ind):
        model.v = self._step_no_intra(model.v, dt, model.temp_c)

    def step_intra(self, model, intra, dt, t_ind):
        model.v = self._step_intra(model.v, dt, model.temp_c, intra)

    def _step_no_intra(self, v, dt, temp):
        self.mech.advance(v, dt, temp)
        self.mech.i(v)
        i = self.mech.irev()
        gtot = self.mech.gtot(v)
        return (self.cmdt * v + i) / (self.cmdt + gtot)

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

    def __init__(self, model, mech, method="thomas", **kw):
        if not AXONML_SOLVERS_AVAILABLE:
            warnings.warn(
                "axonml_solvers not available, using PCR solver for BWD Euler."
            )
            method = "pcr"
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
            self._solve = torch.ops.axonml_solvers.thomas_solve_t
        else:
            raise ValueError(f"Unknown method: {method}")

    def initialize(self, model, dt):
        B, K = model.n_ax, model.n_comp
        # same geometry & membrane setup as ETD1
        radius_cm = 1e-4 * model.diam / 2.0
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

    def _step_no_intra(self, v, ve, dt, temp):
        dt_s = dt * 1e-3
        # advance gating
        self.mech.advance(v, dt, temp)

        # nonlinear residual currents
        i_res = self.mech.i(v)  # (B,K)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v) * self.scale  # (B,K)
        irev = self.mech.irev()  # (B,K)

        # diffusive extracellular coupling
        S = torch.nn.functional.conv1d(ve.unsqueeze(1), self.kernel, padding=1).squeeze(1)
        S[:, 0] = ve[:, 1] - ve[:, 0]
        S[:, -1] = ve[:, -2] - ve[:, -1]
        S = S * self.g_ax

        # form RHS: v_n + dt*(linear_reversal + S - residual)
        f_n = irev * self.scale + S - i_res * self.scale
        RHS = v + dt_s * f_n

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
        return v_np1

    def _step_intra(self, v, ve, dt, temp, intra):
        dt_s = dt * 1e-3
        # advance gating
        self.mech.advance(v, dt, temp)

        # nonlinear residual currents
        i_res = self.mech.i(v)  # (B,K)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v) * self.scale  # (B,K)
        irev = self.mech.irev()  # (B,K)

        # diffusive extracellular coupling
        S = torch.nn.functional.conv1d(ve.unsqueeze(1), self.kernel, padding=1).squeeze(1)
        S[:, 0] = ve[:, 1] - ve[:, 0]
        S[:, -1] = ve[:, -2] - ve[:, -1]
        S = S * self.g_ax

        # form RHS: v_n + dt*(linear_reversal + S - residual)
        f_n = (irev - ires) * self.scale + S + intra.squeeze(1) * self.cm_inv
        RHS = v + dt_s * f_n

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
        return v_np1

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


class _bwd_euler_bt(torch.nn.Module):
    """
    Implicit Euler method for block tridiagonal system.
    """

    compiler = ImplicitCompiler
    builder = ImplicitHandlerBuilder
    is_df = False

    def __init__(self, model, mech, method="warp", **kwargs):
        if not AXONML_SOLVERS_AVAILABLE:
            raise ImportError(
                "The axonml_solvers extension is not available. "
                "Please install the axonml_solvers package."
            )
        super().__init__()
        self.mech = mech

        B, K, M = model.n_ax, model.n_comp, model.n_layers
        M = M + 1
        self.B = B
        self.K = K
        self.M = M

        self.register_buffer("upper", torch.zeros(B, K-1, M, M))
        self.register_buffer("lower", torch.zeros(B, K-1, M))
        self.register_buffer("maind", torch.zeros(B, K,   M))
        self.register_buffer("area",  torch.zeros(B, K))

        self.register_buffer("cm_dt", torch.zeros(B, K))
        self.register_buffer("xc_dt", torch.zeros(B, K))
        self.register_buffer("c_rad", torch.zeros(B, K, M))
        self.register_buffer("xg",    torch.zeros(B, K, M))

        self.register_buffer("i_membrane", torch.zeros(1))
        
        model.register_buffer("v",  torch.zeros(B, K))
        model.register_buffer("vc", torch.zeros(B, K, M))
        model.vc[..., 0] = model.v_init
        model.v[:] = model.v_init

        if method == "thread":
            self._solve = torch.ops.axonml_solvers.solve_bt
        elif method == "warp":
            self._solve = torch.ops.axonml_solvers.solve_bt_warp
        else:
            raise ValueError(f"Unknown method: {method}")
        
    @classmethod
    def shape(cls, n_ax, n_comp):
        return (n_ax, n_comp)

    def init_v(self, model):
        model.vc[:] = 0.0
        model.vc[..., 0] = model.v_init
        model.v[:] = model.v_init
        model.vc = model.vc.detach()
        model.v = model.v.detach()

    def detach(self, model):
        model.vc = model.vc.detach()
        model.v = model.v.detach()
        self.mech.detach()

    def initialize(self, model, dt):
        """
        Generic MxM block initialisation (M >= 3).

        Unknown ordering per compartment
            0  : intracellular v
            1  : ve[0]            (innermost shell)
            ...
            M-1: ve[M-2]          (outermost shell)

        The outermost (Dirichlet) bath is *not* part of the unknowns.
        """
        # ------------------------------------------------------------------
        # Geometry-dependent scalars
        # ------------------------------------------------------------------
        dt = dt * 1e-3                           # ms → s
        B, K, M = self.B, self.K, self.M
        dev, dtyp = model.device(), model.dtype()

        L     = model.L     * 1e-4          # μm → cm
        diam  = model.diam  * 1e-4          # μm → cm
        radius = 0.5 * diam                 # cm
        area   = torch.pi * diam * L        # cm² for each segment

        # ------------------------------------------------------------------
        # Axial conductances (left/right padding → K+1)
        # ------------------------------------------------------------------
        ri   = model.rhoa * L / (torch.pi * radius**2)      # Ω
        ri   = 0.5 * (ri[:, :-1] + ri[:, 1:])               # (B,K-1)
        gi   = 1.0 / ri                                     # S
        gi   = F.pad(gi, (1, 1))                            # (B,K+1)

        raxial  = model.xraxial * L.unsqueeze(-1) * 1e6     # Ω
        raxial  = 0.5 * (raxial[:, :-1, :] + raxial[:, 1:, :])
        gaxial  = 1.0 / raxial                              # S, (B,K-1,M-1)
        zeros_G = torch.zeros((B, 1, M - 1),
                            device=dev, dtype=dtyp)
        gaxial  = torch.cat([zeros_G, gaxial, zeros_G], dim=1)  # (B,K+1,M-1)

        # convenience slices for later
        gi_L  = gi[:, :-1]          # (B,K)
        gi_R  = gi[:, 1:]
        gx_L  = gaxial[:, :-1, :]   # (B,K,M-1)
        gx_R  = gaxial[:, 1:,  :]

        # ------------------------------------------------------------------
        # Radial (membrane + shell) elements
        # ------------------------------------------------------------------
        cm_dt = model.cm * 1e-6 * area / dt                 # F/s, (B,K)

        xc_dt = model.xc * 1e-6 * area.unsqueeze(-1) / dt   # F/s, (B,K,M-1)
        xg    = model.xg * area.unsqueeze(-1)               # S  , (B,K,M-1)

        # ------------------------------------------------------------------
        # Allocate blocks
        # ------------------------------------------------------------------
        main  = torch.zeros((B, K,   M, M), device=dev, dtype=dtyp)
        lower = torch.zeros((B, K-1, M), device=dev, dtype=dtyp)
        upper = torch.zeros((B, K-1, M), device=dev, dtype=dtyp)

        zeros_B = torch.zeros(B, device=dev, dtype=dtyp)    # utility vector

        # ------------------------------------------------------------------
        # Build each compartment block
        # ------------------------------------------------------------------
        for i in range(K):
            # axial conductances to neighbours (0 at sealed ends)
            ga_L = gi_L[:, i] if i > 0     else zeros_B
            ga_R = gi_R[:, i] if i < K-1   else zeros_B

            for s in range(M):                     # row/col in M×M block
                # ---------- diagonal element -----------------------------------
                if s == 0:                          # vi
                    diag = cm_dt[:, i] + ga_L + ga_R
                elif s == 1:                        # ve[0]  (membrane + first shell)
                    xc_out = xc_dt[:, i, 0]
                    xg_out = xg[:,    i, 0]
                    gs_L   = gx_L[:,  i, 0] if i > 0   else zeros_B
                    gs_R   = gx_R[:,  i, 0] if i < K-1 else zeros_B
                    diag   = cm_dt[:, i] + xc_out + xg_out + gs_L + gs_R
                else:                               # ve[s-1],  s ≥ 2
                    # inward coupling is index (s-2), outward is index (s-1)
                    xc_in  = xc_dt[:, i, s-2]
                    xg_in  = xg[:,    i, s-2]
                    xc_out = xc_dt[:, i, s-1]
                    xg_out = xg[:,    i, s-1]
                    gs_L   = gx_L[:,  i, s-1] if i > 0   else zeros_B
                    gs_R   = gx_R[:,  i, s-1] if i < K-1 else zeros_B
                    diag   = xc_in + xc_out + xg_in + xg_out + gs_L + gs_R

                main[:, i, s, s] = diag

                # ---------- radial off-diagonal (coupling to s+1) ---------------
                if s < M - 1:
                    if s == 0:
                        coup = -cm_dt[:, i]                       # vi ↔ ve0
                    else:
                        coup = -(xc_dt[:, i, s-1] + xg[:, i, s-1])  # ve[s-1] ↔ ve[s]
                    main[:, i, s,   s+1] = coup
                    main[:, i, s+1, s  ] = coup   # symmetry

                # ---------- axial off-diagonal blocks ---------------------------
                if i > 0:
                    if s == 0:
                        lower[:, i-1, 0] = -ga_L
                    else:
                        lower[:, i-1, s] = -gx_L[:, i, s-1]
                if i < K - 1:
                    if s == 0:
                        upper[:, i, 0] = -ga_R
                    else:
                        upper[:, i, s] = -gx_R[:, i, s-1]

        # ------------------------------------------------------------------
        # Store for use in the time-stepping routine
        # ------------------------------------------------------------------
        self.area  = area
        self.cm_dt = cm_dt
        self.xc_dt = xc_dt
        self.xg    = xg
        self.c_rad = torch.cat([cm_dt.unsqueeze(-1), xc_dt], dim=-1)

        self.maind = main
        self.lower = lower
        self.upper = upper
            
    def step(self, model, ve, dt, t_ind):
        model.vc, model.v = self._step_no_intra(model.vc, model.v, ve, dt, model.temp_c)

    def step_intra(self, model, ve, intra, dt, t_ind):
        model.vc, model.v = self._step_intra(model.vc, model.v, ve, intra, dt, model.temp_c)

    def _step_no_intra(self, vc, v, ve, dt, temp):

        xg = self.xg[..., -1]

        # advance gating
        self.mech.advance(v, dt, temp)

        ires = self.mech.i(v)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v) * self.area
        irev = self.mech.irev()

        d = (irev - ires) * self.area

        B = self.maind.clone()  # (B, K, M, M)
        B[..., 0, 0] += gtot
        B[..., 1, 1] += gtot
        B[..., 0, 1] -= gtot
        B[..., 1, 0] -= gtot

        D = assemble_rhs(
            vc,
            self.c_rad,
            d,
            xg,
            ve
        )

        # solve tridiagonal system
        vc = self._solve(self.lower, B, self.upper, D)  # (B, K)
        v = vc[..., 0] - vc[..., 1]  # vi = v - ve0
        return vc, v

    def _step_intra(self, vc, v, ve, intra, dt, temp):

        xg = self.xg[..., -1]

        # advance gating
        self.mech.advance(v, dt, temp)

        ires = self.mech.i(v)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v) * self.area
        irev = self.mech.irev()

        d = (irev - ires) * self.area + intra.squeeze(1)

        B = self.maind.clone()
        B[..., 0, 0] += gtot
        B[..., 1, 1] += gtot
        B[..., 0, 1] -= gtot
        B[..., 1, 0] -= gtot

        D = assemble_rhs(
            vc,
            self.c_rad,
            d,
            xg,
            ve
        )

        # solve tridiagonal system
        vc = self._solve(self.lower, B, self.upper, D)
        v = vc[..., 0] - vc[..., 1]
        return vc, v


def assemble_rhs(
    v_prev, c_rad, d, xg, e_ext
):
    rhs = torch.zeros_like(v_prev)

    v_c = c_rad[:, :, :-1] * (v_prev[:, :, :-1] - v_prev[:, :, 1:])

    rhs[:, :, :-1] += v_c
    rhs[:, :,  1:] -= v_c

    rhs[:, :, 0] += d
    rhs[:, :, 1] -= d

    rhs[:, :, -1] += xg * e_ext + c_rad[:, :, -1] * v_prev[:, :, -1]

    return rhs