import logging
import math
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import axonml_solvers  # noqa: F401

    AXONML_SOLVERS_AVAILABLE = True
except ImportError:
    AXONML_SOLVERS_AVAILABLE = False

from .core import Integrator, MultiIntegrator
from .tridiag import pcr_solve_t
from .triton import thomas_solve_cuda_bt, thomas_solve_cuda_t


class _bwd_euler_sc(Integrator):
    """
    Implicit Euler method.
    """

    def __init__(self, model, mech, imem=None):
        super().__init__(model, mech, imem)
        self.register_buffer("cmdt", torch.tensor(0.0))

    def initialize(self, model, dt):
        self.cmdt = (1e-6 * model.cm) / (1e-3 * dt)
        self.area = 2 * math.pi * (1e-4 * model.diam / 2.0) * (1e-4 * model.dx)  # cm²

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._solve(model.v, dt, model.celsius, intra)

    def _solve(self, v, dt, temp, intra=None):
        # apply voltage processes
        v = self.mech.update_v(v, dt)
        self.mech.advance(v, dt, temp)
        itot, gtot = self.mech.i(v)

        denom = self.cmdt + gtot

        v_new = v

        # Adjust for the ionic current part
        ionic_update = itot / denom
        v_new = v_new - ionic_update

        # Adjust for the external injected current, if any
        if intra is not None:
            # We can fuse the division with the area into the update.
            external_update = (intra / self.area) / denom
            v_new = v_new + external_update

        return v_new


class _bwd_euler_sc_multi(MultiIntegrator, _bwd_euler_sc):
    def __init__(self, model, mech, imem=None, write_back=True):
        super().__init__(model, mech, imem, write_back)

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._solve(model.v, dt, model.celsius, intra)
        self._write_back(model)


class _bwd_euler_ub(Integrator):
    """
    Implicit Euler method.
    """

    def __init__(self, model, mech, method="thomas", **kw):
        super().__init__(model, mech, **kw)
        self.method = method
        B, K = model.n_ax, model.n_comp

        # Buffers for diffusive diag, axonal conductance, membrane scale
        self.register_buffer("diag_base", torch.zeros(B, K))
        self.register_buffer("g_ax", torch.zeros(B, K))
        self.register_buffer("cm_inv", torch.zeros(B, K))
        self.register_buffer("scale", torch.zeros(B, K))
        self.register_buffer("lower", torch.zeros(B, K - 1))
        self.register_buffer("upper", torch.zeros(B, K - 1))
        self.register_buffer("g_edge_Cinv", torch.zeros(B, K - 1))

        if method == "pcr":
            self._solve = pcr_solve_t
        elif method == "thomas":
            self._solve = thomas_solve_cuda_t
        else:
            raise ValueError(f"Unknown method: {method}")

    def initialize(self, model, dt):
        if self.method == "pcr":
            self._solve = pcr_solve_t
        elif self.method == "thomas":
            if model.device().type == "cuda":
                self._solve = thomas_solve_cuda_t
            elif model.device().type == "cpu":
                if AXONML_SOLVERS_AVAILABLE:
                    self._solve = torch.ops.axonml_solvers.thomas_solve_t
                else:
                    warnings.warn(
                        "Using `bwd_euler_ub` solver on CPU without axonml_solvers installed. "
                        "Falling back to PCR solver."
                    )
                    self._solve = pcr_solve_t

        B, K = model.np, model.nc
        dt_s = dt * 1e-3  # s

        # ── geometry (all element-wise) ──────────────────────────────
        radius_cm = 1e-4 * model.diam / 2.0  # µm → cm   (B,K)
        dx_cm = 1e-4 * model.dx  # µm → cm   (B,K)
        area_cm2 = 2 * torch.pi * radius_cm * dx_cm  # cm²

        Cm = 1e-6 * model.cm * area_cm2  # F   (B,K)
        Cm_inv = 1.0 / Cm  # 1/F

        # segment axial resistance  (Ω cm)
        Ra_seg = model.rhoa * dx_cm / (torch.pi * radius_cm**2)  # (B,K)

        # ── edge axial conductance between centres i ↔ i+1 ──────────
        # harmonic mean:   g_edge = 2 / (Ra_i + Ra_{i+1})
        g_edge = 2.0 / (Ra_seg[:, :-1] + Ra_seg[:, 1:])  # (B,K-1)

        # convert to   g / C    (1/s)   for each adjoining cell
        g_left = g_edge / Cm[:, :-1]  # affects row i     (B,K-1)
        g_right = g_edge / Cm[:, 1:]  # affects row i+1   (B,K-1)

        g_edge_Cinv = g_edge / Cm[:, :-1]  # (B, K-1)   1/s
        self.g_edge_Cinv = g_edge_Cinv

        # ── fill solver buffers ─────────────────────────────────────
        # diagonal of the diffusive operator (base part, no ion channels yet)
        diag = torch.zeros(B, K, device=model.device())
        diag[:, :-1] -= g_left
        diag[:, 1:] -= g_right
        self.diag_base = diag

        # time-scaled banded matrix (Thomas / DHS will overwrite main diag later)
        self.lower = -dt_s * g_left  # (B,K-1)
        self.upper = -dt_s * g_right  # (B,K-1)

        # misc pre-computed factors used elsewhere
        self.cm_inv = Cm_inv  # (B,K)
        self.scale = area_cm2 * Cm_inv  # A·s / C == 1, but keep for code reuse

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._step(model.v, dt, model.celsius, ve, intra)

    def _step(self, v, dt, temp, ve=None, intra=None) -> Tensor:
        dt_s = dt * 1e-3

        v = self.mech.update_v(v, dt)  # apply voltage processes

        self.mech.advance(v, dt, temp)

        itot, gtot = self.mech.i(v)  # (B,K)

        # f_n = (irev - i_res) * self.scale
        f_n = (gtot * v - itot) * self.scale  # (B,K)

        if ve is not None:
            # diffusive extracellular coupling
            flux = self.g_edge_Cinv * (ve[:, 1:] - ve[:, :-1])  # (B, K-1)
            S = torch.zeros_like(ve)  # (B, K)
            S[:, 1:-1] = -flux[:, :-1] + flux[:, 1:]
            S[:, 0] = -flux[:, 0]
            S[:, -1] = flux[:, -1]

            # form RHS: v_n + dt*(linear_reversal + S - residual)
            f_n = f_n + S

        if intra is not None:
            f_n = f_n + intra * self.cm_inv

        RHS = v + dt_s * f_n

        # build tridiagonal system M v_{n+1} = RHS
        A_diag = self.diag_base - gtot * self.scale
        main = 1.0 - dt_s * A_diag

        # a: (B, K-1), b: (B, K), c: (B, K-1), d: (B, K)
        # inv_b = 1.0 / main  # shape (B, K)

        # scale the three diagonals
        a_s = self.lower
        c_s = self.upper
        b_s = main

        # scale RHS
        d_s = RHS  # divide each equation by its pivot b_i

        # solve tridiagonal system
        v_np1 = self._solve(a_s, b_s, c_s, d_s)  # (B, K)
        # advance gating
        return v_np1


class _bwd_euler_bt(Integrator):
    """
    Implicit Euler method for block tridiagonal system.
    """

    v_vars = ["v", "vc"]

    def __init__(self, model, mech, imem=None, method="triton", **kwargs):
        if not AXONML_SOLVERS_AVAILABLE:
            logging.warning(
                "Only CUDA-based solvers available, using triton Thomas solver. "
                "CPU models will not work. Install axonml_solvers for CPU support."
            )
        super().__init__(model, mech, imem)

        self.mech = mech

        B, K, M = np.prod(model.shape[:-1]), model.n_comp, model.n_layers + 1
        self.B = B
        self.K = K
        self.M = M

        self.register_buffer("upper", torch.zeros(B, K - 1, M))
        self.register_buffer("lower", torch.zeros(B, K - 1, M))
        self.register_buffer("maind", torch.zeros(B, K, M, M))
        self.register_buffer("area", torch.zeros(B, K))

        self.register_buffer("cm_dt", torch.zeros(B, K))
        self.register_buffer("xc_dt", torch.zeros(B, K))
        self.register_buffer("c_rad", torch.zeros(B, K, M))
        self.register_buffer("xg", torch.zeros(B, K, M))

        self.register_buffer("i_membrane", torch.zeros(1))

        model.register_buffer("vc", torch.zeros(B, K, M))

        model.vc[..., 0] = model.v_init
        model.v[:] = model.v_init

        self.initialized = False
        self.dt = None

    @classmethod
    def shape(cls, np, nc):
        return (np, nc)

    def init_v(self, model):
        model.vc.zero_()
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
        if model.device().type == "cpu":
            if not AXONML_SOLVERS_AVAILABLE:
                raise RuntimeError(
                    "CPU models require axonml_solvers to be installed for implicit integration."
                )
            self._solve = torch.ops.axonml_solvers.solve_bt
        elif model.device().type == "cuda":
            self._solve = thomas_solve_cuda_bt

        # ------------------------------------------------------------------
        # Geometry-dependent scalars
        # ------------------------------------------------------------------
        dt = dt * 1e-3  # ms → s
        B, K, M = self.B, self.K, self.M
        dev, dtyp = model.device(), model.dtype()

        L = model.dx.expand(B, K) * 1e-4  # μm → cm
        diam = model.diam.expand(B, K) * 1e-4  # μm → cm
        radius = 0.5 * diam  # cm
        area = torch.pi * diam * L  # cm² for each segment

        # ------------------------------------------------------------------
        # Axial conductances (left/right padding → K+1)
        # ------------------------------------------------------------------
        ri = model.rhoa * L / (torch.pi * radius**2)  # Ω
        ri = 0.5 * (ri[:, :-1] + ri[:, 1:])  # (B,K-1)
        gi = 1.0 / ri  # S
        gi = F.pad(gi, (1, 1))  # (B,K+1)

        raxial = model.xraxial.expand(B, K, 2) * L.unsqueeze(-1) * 1e6  # Ω
        raxial = 0.5 * (raxial[:, :-1, :] + raxial[:, 1:, :])
        gaxial = 1.0 / raxial  # S, (B,K-1,M-1)
        zeros_G = torch.zeros((B, 1, M - 1), device=dev, dtype=dtyp)
        gaxial = torch.cat([zeros_G, gaxial, zeros_G], dim=1)  # (B,K+1,M-1)

        # convenience slices for later
        gi_L = gi[:, :-1]  # (B,K)
        gi_R = gi[:, 1:]
        gx_L = gaxial[:, :-1, :]  # (B,K,M-1)
        gx_R = gaxial[:, 1:, :]

        # ------------------------------------------------------------------
        # Radial (membrane + shell) elements
        # ------------------------------------------------------------------
        area_cm2 = model.area
        cm_dt = model.cm * 1e-6 * area_cm2 / dt  # F/s, (B,K)

        xc_dt = model.xc * 1e-6 * area_cm2.unsqueeze(-1) / dt  # F/s, (B,K,M-1)
        xg = model.xg * area_cm2.unsqueeze(-1)  # S  , (B,K,M-1)

        # ------------------------------------------------------------------
        # Allocate blocks
        # ------------------------------------------------------------------
        main = torch.zeros((B, K, M, M), device=dev, dtype=dtyp)
        lower = torch.zeros((B, K - 1, M), device=dev, dtype=dtyp)
        upper = torch.zeros((B, K - 1, M), device=dev, dtype=dtyp)

        zeros_B = torch.zeros(B, device=dev, dtype=dtyp)  # utility vector

        # ------------------------------------------------------------------
        # Build each compartment block
        # ------------------------------------------------------------------
        for i in range(K):
            # axial conductances to neighbours (0 at sealed ends)
            ga_L = gi_L[:, i] if i > 0 else zeros_B
            ga_R = gi_R[:, i] if i < K - 1 else zeros_B

            for s in range(M):  # row/col in M×M block
                # ---------- diagonal element -----------------------------------
                if s == 0:  # vi
                    diag = cm_dt[:, i] + ga_L + ga_R
                elif s == 1:  # ve[0]  (membrane + first shell)
                    xc_out = xc_dt[:, i, 0]
                    xg_out = xg[:, i, 0]
                    gs_L = gx_L[:, i, 0] if i > 0 else zeros_B
                    gs_R = gx_R[:, i, 0] if i < K - 1 else zeros_B
                    diag = cm_dt[:, i] + xc_out + xg_out + gs_L + gs_R
                else:  # ve[s-1],  s ≥ 2
                    # inward coupling is index (s-2), outward is index (s-1)
                    xc_in = xc_dt[:, i, s - 2]
                    xg_in = xg[:, i, s - 2]
                    xc_out = xc_dt[:, i, s - 1]
                    xg_out = xg[:, i, s - 1]
                    gs_L = gx_L[:, i, s - 1] if i > 0 else zeros_B
                    gs_R = gx_R[:, i, s - 1] if i < K - 1 else zeros_B
                    diag = xc_in + xc_out + xg_in + xg_out + gs_L + gs_R

                main[:, i, s, s] = diag

                # ---------- radial off-diagonal (coupling to s+1) ---------------
                if s < M - 1:
                    if s == 0:
                        coup = -cm_dt[:, i]  # vi ↔ ve0
                    else:
                        coup = -(
                            xc_dt[:, i, s - 1] + xg[:, i, s - 1]
                        )  # ve[s-1] ↔ ve[s]
                    main[:, i, s, s + 1] = coup
                    main[:, i, s + 1, s] = coup  # symmetry

                # ---------- axial off-diagonal blocks ---------------------------
                if i > 0:
                    if s == 0:
                        lower[:, i - 1, 0] = -ga_L
                    else:
                        lower[:, i - 1, s] = -gx_L[:, i, s - 1]
                if i < K - 1:
                    if s == 0:
                        upper[:, i, 0] = -ga_R
                    else:
                        upper[:, i, s] = -gx_R[:, i, s - 1]

        # ------------------------------------------------------------------
        # Store for use in the time-stepping routine
        # ------------------------------------------------------------------
        self.area = area
        self.cm_dt = cm_dt
        self.xc_dt = xc_dt
        self.xg = xg
        self.c_rad = torch.cat([cm_dt.unsqueeze(-1), xc_dt], dim=-1)

        self.maind = main
        self.lower = lower
        self.upper = upper

        self.base_shape = tuple(list(model.shape) + [self.M])

    def step(self, model, dt, ve=None, intra=None):
        model.vc, model.v = self._step(
            model.vc.view(-1, self.K, 3), model.v, dt, model.celsius, ve, intra
        )

    def _step(self, vc, v, dt, temp, ve=None, intra=None) -> Tuple[Tensor, Tensor]:
        xg = self.xg[..., -1]

        # apply voltage processes
        v = self.mech.update_v(v, dt)

        # advance gating
        self.mech.advance(v, dt, temp)

        itot, gtot = self.mech.i(v)

        # linearized ionic conductances & reversal
        gtot = gtot.view(-1, self.K) * self.area

        itot = itot.view(-1, self.K) * self.area  # (B, K)

        d = gtot * v.view(-1, self.K) - itot

        if intra is not None:
            d = d + intra

        B = self.maind.clone()  # (B, K, M, M)
        B[..., 0, 0] += gtot
        B[..., 1, 1] += gtot
        B[..., 0, 1] -= gtot
        B[..., 1, 0] -= gtot

        D = assemble_rhs(vc, self.c_rad, d, xg, ve)

        # solve tridiagonal system
        vc = self._solve(self.lower, B, self.upper, D).reshape(
            self.base_shape
        )  # (B, K)
        v = vc[..., 0] - vc[..., 1]  # v = vi - ve0
        return vc, v


def assemble_rhs(v_prev, c_rad, d, xg, e_ext):
    rhs = torch.zeros_like(v_prev)

    v_c = c_rad[..., :-1] * (v_prev[..., :-1] - v_prev[..., 1:])

    rhs[..., :-1] += v_c
    rhs[..., 1:] -= v_c

    rhs[..., 0] += d
    rhs[..., 1] -= d

    if e_ext is not None:
        rhs[..., -1] += xg * e_ext + c_rad[..., -1] * v_prev[..., -1]
    else:
        rhs[..., -1] += c_rad[..., -1] * v_prev[..., -1]

    return rhs
