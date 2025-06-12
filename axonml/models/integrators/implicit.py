from typing import Tuple
import warnings
import math

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
from .tridiag import pcr_tridiag_solve


class _bwd_euler_sc(SCIntegrator):
    """
    Implicit Euler method.
    """

    compiler = ImplicitCompiler
    builder = ImplicitHandlerBuilder
    is_df = False

    def __init__(self, model, mech, imem=None, N=1, P=1, C=1):
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
                "axonml_solvers not available, using Python PCR solver for BWD Euler."
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
        self.register_buffer("lower", torch.zeros(B, K-1))
        self.register_buffer("upper", torch.zeros(B, K-1))

        if method == "pcr":
            self._solve = pcr_tridiag_solve
        elif method == "thomas":
            self._solve = torch.ops.axonml_solvers.thomas_solve_t
        else:
            raise ValueError(f"Unknown method: {method}")

    def initialize(self, model, dt):
        B, K = model.n_ax, model.n_comp
        dt_s = dt * 1e-3
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
        self.lower.copy_(-dt_s * self.g_ax[:, :-1])  # (B, K-1)
        self.upper.copy_(-dt_s * self.g_ax[:, 1:])  # (B, K-1)

    def step(self, model, dt, ve=None, intra=None):
        model.v = self._step(model.v, dt, model.temp_c, ve, intra)

    def _step(self, v, dt, temp, ve=None, intra=None) -> Tensor:
        dt_s = dt * 1e-3
        # advance gating
        self.mech.advance(v, dt, temp)

        # nonlinear residual currents
        i_res = self.mech.i(v)  # (B,K)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v) * self.scale  # (B,K)
        irev = self.mech.irev()  # (B,K)

        f_n = (irev - i_res) * self.scale

        if ve is not None:
            # diffusive extracellular coupling
            S = torch.nn.functional.conv1d(ve.unsqueeze(1), self.kernel, padding=1).squeeze(1)
            S[:, 0] = ve[:, 1] - ve[:, 0]
            S[:, -1] = ve[:, -2] - ve[:, -1]
            S = S * self.g_ax

            # form RHS: v_n + dt*(linear_reversal + S - residual)
            f_n = f_n + S

        if intra is not None:
            f_n = f_n + intra * self.cm_inv

        RHS = v + dt_s * f_n

        # build tridiagonal system M v_{n+1} = RHS
        A_diag = self.diag_base - gtot
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

        self.register_buffer("upper", torch.zeros(B, K-1, M))
        self.register_buffer("lower", torch.zeros(B, K-1, M))
        self.register_buffer("maind", torch.zeros(B, K,   M, M))
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
            
    def step(self, model, dt, ve=None, intra=None):
        model.vc, model.v = self._step(model.vc, model.v, dt, model.temp_c, ve, intra)

    def _step(self, vc, v, dt, temp, ve=None, intra=None) -> Tuple[Tensor, Tensor]:

        xg = self.xg[..., -1]

        # advance gating
        self.mech.advance(v, dt, temp)

        ires = self.mech.i(v)

        # linearized ionic conductances & reversal
        gtot = self.mech.gtot(v) * self.area
        irev = self.mech.irev()

        d = (irev - ires) * self.area

        if intra is not None:
            d = d + intra

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


def assemble_rhs(
    v_prev, c_rad, d, xg, e_ext
):
    rhs = torch.zeros_like(v_prev)

    v_c = c_rad[:, :, :-1] * (v_prev[:, :, :-1] - v_prev[:, :, 1:])

    rhs[:, :, :-1] += v_c
    rhs[:, :,  1:] -= v_c

    rhs[:, :, 0] += d
    rhs[:, :, 1] -= d

    if e_ext is not None:
        rhs[:, :, -1] += xg * e_ext + c_rad[:, :, -1] * v_prev[:, :, -1]
    else:
        rhs[:, :, -1] += c_rad[:, :, -1] * v_prev[:, :, -1]

    return rhs