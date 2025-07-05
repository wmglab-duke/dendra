from typing import Tuple

import torch
import torch.nn.functional as F

from axonml.models.mechanisms.handler.builders import ImplicitHandlerBuilder
from axonml.helpers import IMEM

from .core import Integrator


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

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._step(model.v, dt, model.temp_c, ve, intra)

    def _step(self, v, dt, temp, ve=None, intra=None):
        dt_s = dt * 1e-3
        self.mech.advance(v, dt, temp)

        ires = self.mech.i(v)
        irev = self.mech.irev()
        gtot = self.mech.gtot(v) * self.scale

        f_n = (irev - ires) * self.scale

        if ve is not None:
            S = F.conv1d(ve.unsqueeze(1), self.kernel, padding=1).squeeze(1)
            S[:, 0] = ve[:, 1] - ve[:, 0]  # fix boundary left
            S[:, -1] = ve[:, -2] - ve[:, -1]  # fix boundary right

            S *= self.g_ax
            f_n = f_n + S

        if intra is not None:
            f_n = f_n + intra
        
        diag = self.diag - gtot
        v_lin = self._expm(
            v,
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
        return (v_lin + v_nl)

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