import logging
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import dendra_solvers  # noqa: F401

    DENDRA_SOLVERS_AVAILABLE = True
except ImportError:
    DENDRA_SOLVERS_AVAILABLE = False

from .cable import unbranched_edge_conductance
from .core import (
    Integrator,
    MultiIntegrator,
    _as_solve_block,
    _as_solve_matrix,
    _expanded_v_init,
    _flatten_to_solve,
    _model_solve_shape,
    ensure_model_buffer,
)
from .tridiag import pcr_solve_t
from .tridiag.block import block_pcr_solve_t
from .triton import (
    pcr_solve_cuda_t,
    solve_bt_spd_cuda_consume_unchecked,
    thomas_solve_cuda_bt,
    thomas_solve_cuda_t,
)


class _bwd_euler_sc(Integrator):
    r"""
    Single-compartment implicit (backward) Euler integrator.

    Advances the membrane voltage with a fully implicit Euler step on the
    ionic current term:

    .. math::
       v^{n+1} = v^{n} - \frac{\Delta t}{C_m} I_\mathrm{ion}(v^{n+1})

    which is linearized through the mechanism's ``i`` call
    (returning total current and conductance), yielding a closed-form update
    per compartment without solving a spatial system.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    def __init__(self, model, mech, imem=None):
        super().__init__(model, mech, imem)
        self.register_buffer("cmdt", torch.tensor(0.0))
        self.register_buffer("area", torch.tensor(0.0))

    def initialize(self, model, dt):
        # model.cm: uF/cm²
        # dt: ms
        self.cmdt = (1e-6 * model.cm * model.cm_scale) / (1e-3 * dt)
        self.area = model.area * model.area_scale

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_solve", model.v, dt, model.celsius, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _solve(self, v, dt, temp, intra=None):
        # apply voltage processes
        v = self.mech.update_v(v)
        self._advance_pre_current(v, dt, temp)
        itot, gtot = self.mech.i(v)
        ion_current_frame = self._capture_ion_current_frame()
        ion_conductance_frame = self._capture_ion_conductance_frame()

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

        i_membrane = None
        if self.imem:
            # capacitance term (mA/cm^2) using cmdt = Cm/dt_s per area
            i_cap = self.cmdt * (v_new - v)  # mA/cm^2
            i_ion = itot + gtot * (v_new - v)  # mA/cm^2
            i_mem_dens = i_cap + i_ion
            i_membrane = i_mem_dens * self.area  # mA

        accepted_frame = self._linearize_ion_current_frame(
            ion_current_frame,
            ion_conductance_frame,
            v_new - v,
        )
        self._advance_post_current(v, dt, temp, accepted_frame)

        return v_new, i_membrane


class _bwd_euler_sc_skip(Integrator):
    r"""
    Single-compartment implicit Euler that skips solving the voltage.
    Updates to voltage by VoltageProcesses are still applied.

    Applies the same implicit Euler step as :class:`_bwd_euler_sc` but only
    evaluates ionic currents when stepping. Useful when the voltage is updated
    externally (e.g., via a closed-form solution within a Mechanism) but ionic
    currents still need to be advanced and evaluated. Consequently,
    ``intra`` is not a voltage-solve input for this integrator. When ``imem``
    is enabled, the imposed voltage transition is still included in the
    capacitive part of the shared membrane-current contract.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    v_vars = []

    def __init__(self, model, mech, imem=None):
        super().__init__(model, mech, imem)
        self.register_buffer("cmdt", torch.tensor(0.0))
        self.register_buffer("area", torch.tensor(0.0))

    def initialize(self, model, dt):
        self.cmdt = (1e-6 * model.cm * model.cm_scale) / (1e-3 * dt)
        self.area = model.area * model.area_scale

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_solve", model.v, dt, model.celsius, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _solve(self, v, dt, temp, intra=None):
        # apply voltage processes
        # Keep the pre-process voltage independent from the returned tensor.
        # VoltageProcess implementations are documented as out-of-place, but
        # protecting this value also makes the capacitive term robust to a
        # custom process that aliases or mutates its input.
        v_old = v.clone() if self.imem else None
        v_new = self.mech.update_v(v)
        self._advance_pre_current(v_new, dt, temp)
        # No voltage solve/linearization follows, so ``itot`` is the exact
        # mechanism current evaluated at the externally imposed voltage.
        itot, _ = self.mech.i(v_new)
        ion_current_frame = self._capture_ion_current_frame()

        i_membrane = None
        if self.imem:
            i_cap = self.cmdt * (v_new - v_old)
            i_membrane = (i_cap + itot) * self.area
        self._advance_post_current(v_new, dt, temp, ion_current_frame)
        return v_new, i_membrane


class _bwd_euler_sc_multi(MultiIntegrator, _bwd_euler_sc):
    def __init__(self, model, mech, imem=None, write_back=True):
        super().__init__(model, mech, imem, write_back)

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_solve", model.v, dt, model.celsius, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane
        self._write_back(model)


class _bwd_euler_ub(Integrator):
    r"""
    Implicit Euler for a single-layer, unbranched cable morphology.

    Builds a tridiagonal system from the cable diffusion operator and ionic
    linearization, then solves it each step with a selectable solver backend.
    Supports optional gradient-clipped Thomas kernels on CUDA.

    Parameters
    ----------
    method : {"thomas", "inv", "spd", "pcr"}, optional
        Solver backend: Thomas (default; ``"inv"`` maps to ``"thomas"``),
        SPD tridiagonal (CPU only), or PCR (CPU/CUDA fallback).
    clip_scale_backward : float, optional
        Enable gradient clipping in the CUDA Thomas kernel by this scale factor
        (only applicable when ``method="thomas"``). Default None disables.
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    supports_unbranched_cable = True

    def __init__(
        self, model, mech, method: str = "inv", clip_scale_backward=None, **kw
    ):
        super().__init__(model, mech, **kw)
        if method == "inv":
            method = "thomas"
        self.method = method.lower()

        # GC-variant (gradient clipping inside Thomas kernel) only makes sense for method="thomas"
        self.use_gc_variant = False
        if clip_scale_backward is not None:
            if self.method != "thomas":
                warnings.warn(
                    f"clip_scale_backward is only supported for method='thomas'; "
                    f"ignoring for method='{self.method}'."
                )
            else:
                self.register_buffer("clip_scale", torch.tensor(clip_scale_backward))
                self.use_gc_variant = True

        self._last_bands: Tuple[Tensor, Tensor, Tensor] = None
        B, K = _model_solve_shape(model)
        self.B = B
        self.K = K
        self.base_shape = tuple(model.shape)

        # Buffers for diffusive diag, axonal conductance, membrane scale
        self.register_buffer("diag_base", torch.zeros(B, K))
        self.register_buffer("g_ax", torch.zeros(B, K))
        self.register_buffer("cm_inv", torch.zeros(B, K))
        self.register_buffer("scale", torch.zeros(B, K))
        self.register_buffer("lower", torch.zeros(B, K - 1))
        self.register_buffer("upper", torch.zeros(B, K - 1))
        self.register_buffer("g_edge_Cinv", torch.zeros(B, K - 1))
        self.register_buffer("g_edge_Cinv_right", torch.zeros(B, K - 1))

    def _select_solver(self, model):
        dev = model.device().type  # "cpu" or "cuda"

        # -------- explicit PCR --------
        if self.method == "pcr":
            if dev == "cuda":
                self._solve = pcr_solve_cuda_t
            else:
                self._solve = pcr_solve_t
            return

        # -------- SPD option (CPU only) --------
        if self.method == "spd":
            if dev == "cuda":
                warnings.warn(
                    "method='spd' is not implemented on CUDA; falling back to method='thomas' (CUDA)."
                )
                self.method = "thomas"  # fall through
            elif dev == "cpu":
                # We need dendra_solvers + solve_tri_spd
                try:
                    from dendra_solvers import solve_tri_spd
                except ImportError:
                    warnings.warn(
                        "method='spd' requested on CPU but solve_tri_spd / dendra_solvers "
                        "is not available; falling back to method='thomas' (CPU) or PCR."
                    )
                    if DENDRA_SOLVERS_AVAILABLE:
                        self._solve = torch.ops.dendra_solvers.thomas_solve_t
                    else:
                        self._solve = pcr_solve_t
                    return
                else:
                    # SPD solver is usable
                    self._solve = solve_tri_spd
                    return

        # -------- default / THOMAS path --------
        if self.method != "thomas":
            warnings.warn(
                f"Unknown or unsupported solver method '{self.method}', "
                f"falling back to method='thomas'."
            )
            self.method = "thomas"

        if dev == "cuda":
            # CUDA: always use CUDA Thomas implementation
            self._solve = thomas_solve_cuda_t
        elif dev == "cpu":
            if DENDRA_SOLVERS_AVAILABLE:
                if self.use_gc_variant:
                    self._solve = torch.ops.dendra_solvers.thomas_solve_t_gc
                else:
                    self._solve = torch.ops.dendra_solvers.thomas_solve_t
            else:
                warnings.warn(
                    "Using `bwd_euler_ub` on CPU without dendra_solvers installed. "
                    "Falling back to PCR solver."
                )
                self._solve = pcr_solve_t

    def initialize(self, model, dt):
        """
        Compute geometry-dependent coefficients and select tridiagonal solver
        based on `self.method` and device.
        """
        # Select solver first (depends on device and method)
        self._select_solver(model)

        B, K = _model_solve_shape(model)
        self.B, self.K = B, K
        dt_s = dt * 1e-3  # s

        cm = _as_solve_matrix(model.cm, model) * _as_solve_matrix(model.cm_scale, model)

        # ── geometry (all element-wise) ──────────────────────────────
        area_cm2 = _as_solve_matrix(model.area, model) * _as_solve_matrix(
            model.area_scale, model
        )

        Cm = 1e-6 * cm * area_cm2  # F   (B,K)
        Cm_inv = 1.0 / Cm  # 1/F

        # ── edge axial conductance between centres i ↔ i+1 ──────────
        # Native Cable uses exact compiled edge resistance; conventional Axon
        # models retain the established half-cylinder reconstruction.
        g_edge = unbranched_edge_conductance(model)

        # convert to   g / C    (1/s)   for each adjoining cell
        g_left = g_edge / Cm[:, :-1]  # affects row i     (B,K-1)
        g_right = g_edge / Cm[:, 1:]  # affects row i+1   (B,K-1)

        self.g_edge_Cinv = g_left
        self.g_edge_Cinv_right = g_right

        # ── fill solver buffers ─────────────────────────────────────
        # diagonal of the diffusive operator (base part, no ion channels yet)
        diag = torch.zeros(B, K, device=model.device(), dtype=model.dtype())
        diag[:, :-1] -= g_left
        diag[:, 1:] -= g_right
        self.diag_base = diag

        # time-scaled banded matrix (Thomas / SPD will overwrite main diag later)
        # For edge i <-> i+1, the upper entry belongs to row i and is
        # normalized by C_i; the lower entry belongs to row i+1 and is
        # normalized by C_{i+1}.
        self.lower = -dt_s * g_right  # (B,K-1)  subdiag
        self.upper = -dt_s * g_left  # (B,K-1)  superdiag

        # misc pre-computed factors used elsewhere
        self.cm_inv = Cm_inv  # (B,K)
        self.scale = area_cm2 * Cm_inv  # 1 / c_m (inverse specific capacitance, cm^2/F)

        self.base_shape = tuple(model.shape)

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_step", model.v, dt, model.celsius, ve, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _step(self, v, dt, temp, ve=None, intra=None) -> Tensor:
        dt_s = dt * 1e-3

        v = self.mech.update_v(v)  # apply voltage processes
        self._advance_pre_current(v, dt, temp)

        itot, gtot = self.mech.i(v)  # public voltage shape
        ion_current_frame = self._capture_ion_current_frame()
        ion_conductance_frame = self._capture_ion_conductance_frame()

        v_flat = _flatten_to_solve(v, self.K)
        itot = _flatten_to_solve(itot, self.K)
        gtot = _flatten_to_solve(gtot, self.K)

        # f_n = (irev - i_res) * self.scale
        f_n = (gtot * v_flat - itot) * self.scale  # (B,K)

        if ve is not None:
            # diffusive extracellular coupling
            ve_flat = _flatten_to_solve(ve, self.K, self.base_shape)
            delta_ve = ve_flat[..., 1:] - ve_flat[..., :-1]
            S = torch.zeros_like(ve_flat)  # (B, K)
            # A varying extracellular field enters through A @ ve.  Each side
            # of an edge must use the capacitance of its own compartment.
            S[..., :-1] += self.g_edge_Cinv * delta_ve
            S[..., 1:] -= self.g_edge_Cinv_right * delta_ve
            f_n = f_n + S

        if intra is not None:
            f_n = f_n + _flatten_to_solve(intra, self.K, self.base_shape) * self.cm_inv

        RHS = v_flat + dt_s * f_n

        # build tridiagonal system M v_{n+1} = RHS
        A_diag = self.diag_base - gtot * self.scale
        main = 1.0 - dt_s * A_diag  # (B,K)

        # diagonals
        a_s = self.lower  # (B,K-1)
        c_s = self.upper  # (B,K-1)
        b_s = main  # (B,K)

        if not self.jit:
            self._last_bands = (a_s, b_s, c_s)

        d_s = RHS

        # solve tridiagonal system
        if self.K == 1:
            # Tridiagonal extension kernels require at least one off-diagonal.
            # A one-compartment Cable is the same scalar implicit system and
            # has the exact closed-form solution below.
            v_np1 = d_s / b_s
        elif self.use_gc_variant:
            # GC variant only valid for Thomas solver
            v_np1 = self._solve(a_s, b_s, c_s, d_s, self.clip_scale)
        else:
            v_np1 = self._solve(a_s, b_s, c_s, d_s)

        i_membrane = None

        if self.imem:
            area = self.scale / self.cm_inv  # cm^2 (segment area)
            Cm = 1.0 / self.cm_inv  # F (segment capacitance)
            Cdt = Cm / dt_s  # A/V (capacitive 'conductance')
            g_abs = gtot * area  # S (ionic conductance per segment)
            i_abs = itot * area  # mA (ionic current per segment)
            dmem = Cdt + g_abs  # A/V
            i_membrane = (dmem * (v_np1 - v_flat) + i_abs).reshape(self.base_shape)

        v_new = v_np1.reshape(self.base_shape)
        accepted_frame = self._linearize_ion_current_frame(
            ion_current_frame,
            ion_conductance_frame,
            v_new - v,
        )
        self._advance_post_current(v, dt, temp, accepted_frame)

        return v_new, i_membrane


class _bwd_euler_bt(Integrator):
    r"""
    Implicit Euler for block tridiagonal systems (multi-layer extracellular).

    Solves a block-tridiagonal linear system per step that couples the
    intracellular voltage and extracellular shell variables. The current block
    solver path is specialized to 3 unknowns per compartment. The SPD backend
    uses consuming workspace solvers: the per-step block matrix and RHS buffers
    constructed in ``_step`` are disposable and may be overwritten by the solver.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    method : {"spd", "inv"}, optional
        Block solver backend. ``"inv"`` selects a Thomas-like block solver,
        ``"spd"`` uses an SPD block solver where available. Default "inv".
    **kwargs
        Forwarded to base integrator; reserved for future solver options.

    Notes
    -----
    When ``imem`` is enabled, public ``model.i_membrane`` is the absolute
    transmembrane current in mA: compartment area times the outward-current
    convention ``Cm * (v_new - v_old) / dt + I_ion(v_new)`` (with the same
    per-step current linearization used by the voltage solve). It excludes
    axial cable currents and applied intracellular stimulus as explicit terms;
    those drives affect ``i_membrane`` only through the solved voltage.
    """

    v_vars = ["v", "vc"]

    def __init__(self, model, mech, imem=None, method="inv", **kwargs):
        if not DENDRA_SOLVERS_AVAILABLE:
            logging.warning(
                "Native CPU block solvers are unavailable. CUDA will use Triton "
                "and MPS will use pure-PyTorch block PCR; install dendra_solvers "
                "for CPU support."
            )
        super().__init__(model, mech, imem)

        valid_methods = ["spd", "inv", "thomas"]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method: {method}, must be one of {valid_methods}"
            )
        if method == "inv":
            method = "thomas"
        self.method = method

        self.mech = mech

        B, K = _model_solve_shape(model)
        M = model.n_layers + 1
        if M != 3:
            raise ValueError(
                "_bwd_euler_bt currently requires exactly 3 unknowns per "
                f"compartment (model.n_layers + 1 == 3); got {M}."
            )
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

        ensure_model_buffer(model, "vc", tuple(model.shape) + (M,))

        v0 = _expanded_v_init(model)
        model.vc[..., 0] = v0
        model.v[:] = v0

        self.initialized = False
        self.dt = None

    @classmethod
    def shape(cls, np, nc):
        return (np, nc)

    def init_v(self, model):
        v0 = _expanded_v_init(model).clone().detach().contiguous()
        ensure_model_buffer(model, "vc", tuple(model.shape) + (self.M,))
        model.vc.zero_()
        model.vc[..., 0] = v0
        model.v = v0.clone()
        model.vc = model.vc.detach()
        model.v = model.v.detach()
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()

    def detach(self, model):
        model.vc = model.vc.detach()
        model.v = model.v.detach()
        if self.imem:
            model.i_membrane = model.i_membrane.detach()
        self.mech.detach()

    def _select_solver(self, model):
        dev = model.device().type

        # Apple MPS cannot load either the CPU extension or CUDA/Triton kernels.
        # Use a vectorized, differentiable pure-Torch solver for both public
        # methods. The method still controls CPU/CUDA dispatch exactly as before.
        if dev == "mps":
            self._solve = block_pcr_solve_t
            return

        if self.method == "spd":
            if dev == "cuda":
                # _step constructs fresh B_work/D_work buffers, so use the
                # unchecked consuming path to avoid duplicate per-step validation.
                self._solve = solve_bt_spd_cuda_consume_unchecked
            elif dev == "cpu":
                try:
                    from dendra_solvers import solve_bt_spd_consume_unchecked
                except ImportError as err:
                    raise ImportError(
                        "_bwd_euler_bt(method='spd') requires dendra_solvers for CPU execution. "
                        "Use CUDA or install dendra_solvers."
                    ) from err
                else:
                    # _step constructs fresh B_work/D_work buffers, so use the
                    # unchecked consuming path to avoid duplicate per-step validation.
                    self._solve = solve_bt_spd_consume_unchecked
        else:  # method == "thomas"
            if dev == "cuda":
                self._solve = thomas_solve_cuda_bt
            elif dev == "cpu":
                if DENDRA_SOLVERS_AVAILABLE:
                    self._solve = torch.ops.dendra_solvers.solve_bt
                else:
                    raise ImportError(
                        "_bwd_euler_bt(method='thomas') requires dendra_solvers for CPU execution. "
                        "Use CUDA or install dendra_solvers."
                    )

    def initialize(self, model, dt):
        """
        3x3 block initialisation.

        Unknown ordering per compartment
            0: intracellular v
            1: ve[0]            (innermost shell)
            2: ve[1]            (outermost unknown shell)

        The outermost (Dirichlet) bath is *not* part of the unknowns.
        """
        self._select_solver(model)
        self._refresh_solver_shape(model, block_dim=self.M)
        if tuple(model.vc.shape) != tuple(model.shape) + (self.M,):
            vc_new = torch.zeros(
                *model.shape, self.M, device=model.device(), dtype=model.dtype()
            )
            old = model.vc.reshape(-1, self.M)
            new = vc_new.reshape(-1, self.M)
            n = min(old.shape[0], new.shape[0])
            new[:n].copy_(old[:n].to(device=model.device(), dtype=model.dtype()))
            model.vc = vc_new

        # ------------------------------------------------------------------
        # Geometry-dependent scalars
        # ------------------------------------------------------------------
        dt = dt * 1e-3  # ms → s
        B, K, M = self.B, self.K, self.M
        dev, dtyp = model.device(), model.dtype()

        L = _as_solve_matrix(model.dx, model) * 1e-4  # μm → cm
        diam = _as_solve_matrix(model.diam, model) * 1e-4  # μm → cm
        radius = 0.5 * diam  # cm
        area = torch.pi * diam * L  # cm² for each segment

        # ------------------------------------------------------------------
        # Axial conductances (left/right padding → K+1)
        # ------------------------------------------------------------------
        ri = _as_solve_matrix(model.rhoa, model) * L / (torch.pi * radius**2)  # Ω
        ri = 0.5 * (ri[:, :-1] + ri[:, 1:])  # (B,K-1)
        gi = 1.0 / ri  # S
        gi = F.pad(gi, (1, 1))  # (B,K+1)

        raxial = (
            _as_solve_block(model.xraxial, model, (M - 1,)) * L.unsqueeze(-1) * 1e6
        )  # Ω
        raxial = 0.5 * (raxial[:, :-1, :] + raxial[:, 1:, :])
        self.register_buffer("raxial", raxial[..., 0])
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
        area_cm2 = area  # cm²
        cm_dt = _as_solve_matrix(model.cm, model) * 1e-6 * area_cm2 / dt

        xc_dt = (
            _as_solve_block(model.xc, model, (M - 1,))
            * 1e-6
            * area_cm2.unsqueeze(-1)
            / dt
        )  # F/s, (B,K,M-1)
        xg = _as_solve_block(model.xg, model, (M - 1,)) * area_cm2.unsqueeze(-1)

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
        vc_new, v_new, i_membrane = self._call_kernel(
            "_step",
            model.vc.reshape(-1, self.K, self.M),
            model.v,
            dt,
            model.celsius,
            ve,
            intra,
        )
        model.vc = vc_new
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _step(self, vc, v, dt, temp, ve=None, intra=None) -> Tuple[Tensor, Tensor]:
        xg = self.xg[..., -1]

        # apply voltage processes
        v_state = self.mech.update_v(v)

        # advance gating
        self._advance_pre_current(v_state, dt, temp)

        itot, gtot = self.mech.i(v_state)
        ion_current_frame = self._capture_ion_current_frame()
        ion_conductance_frame = self._capture_ion_conductance_frame()

        # linearized ionic conductances & reversal
        v_flat = _flatten_to_solve(v_state, self.K)
        gtot = _flatten_to_solve(gtot, self.K) * self.area

        itot = _flatten_to_solve(itot, self.K) * self.area  # (B, K)

        d = gtot * v_flat - itot

        if intra is not None:
            d = d + _flatten_to_solve(intra, self.K, self.base_shape[:-1])

        # Disposable solver workspaces. The SPD consuming solvers may overwrite
        # both buffers during factorization/elimination; neither is read after
        # the solve.
        B_work = self.maind.clone()  # (B, K, M, M)
        B_work[..., 0, 0] += gtot
        B_work[..., 1, 1] += gtot
        B_work[..., 0, 1] -= gtot
        B_work[..., 1, 0] -= gtot

        ve_flat = (
            _flatten_to_solve(ve, self.K, self.base_shape[:-1])
            if ve is not None
            else None
        )
        D_work = assemble_rhs(vc, self.c_rad, d, xg, ve_flat)

        # solve block-tridiagonal system
        vc_new = self._solve(self.lower, B_work, self.upper, D_work).reshape(
            self.base_shape
        )  # (model.shape)
        v_new = vc_new[..., 0] - vc_new[..., 1]  # v = vi - ve0

        i_membrane = None

        if self.imem:
            vprev_mem = vc[..., 0] - vc[..., 1]
            delta_v = _flatten_to_solve(v_new, self.K) - vprev_mem.reshape(-1, self.K)
            i_membrane = (self.cm_dt + gtot) * delta_v + itot
            # ``i_membrane`` follows NEURON's extracellular convention exactly:
            # area * (Cm * (v_new - v_old) / dt + I_ion(v_new)), using Dendra's
            # outward-current sign and the solve's current linearization.  The
            # result is absolute mA. Applied intracellular stimulus and axial
            # currents affect it only through ``v_new``; neither is itself a
            # transmembrane current term.
            i_membrane = i_membrane.reshape_as(v_new)

        accepted_frame = self._linearize_ion_current_frame(
            ion_current_frame,
            ion_conductance_frame,
            v_new - v_state,
        )
        self._advance_post_current(v_state, dt, temp, accepted_frame)

        return vc_new, v_new, i_membrane


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


def assemble_rhs_into(
    out: torch.Tensor,
    v_prev: torch.Tensor,
    c_rad: torch.Tensor,
    d: torch.Tensor,
    xg: torch.Tensor,
    e_ext: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    In-place variant of assemble_rhs. Fills `out` (shape = v_prev.shape = [B,K,3]).
    """
    out.zero_()

    # v_c on radial edges
    v_c = c_rad[..., :-1] * (v_prev[..., :-1] - v_prev[..., 1:])
    out[..., :-1] += v_c
    out[..., 1:] -= v_c

    # membrane coupling vi<->ve0:
    out[..., 0] += d
    out[..., 1] -= d

    # outermost shell / boundary term
    if e_ext is not None:
        out[..., -1] += xg * e_ext + c_rad[..., -1] * v_prev[..., -1]
    else:
        out[..., -1] += c_rad[..., -1] * v_prev[..., -1]
    return out
