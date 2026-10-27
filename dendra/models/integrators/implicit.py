import logging
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import dendra_solvers

    DENDRA_SOLVERS_AVAILABLE = True
except ImportError:
    dendra_solvers = None
    DENDRA_SOLVERS_AVAILABLE = False

from ._cuda_functional import pcr_solve as _functional_pcr_solve
from ._cuda_functional import thomas_solve as _functional_thomas_solve
from ._native import _mps_solver_available
from .cable import (
    _layered_edge_conductance,
    unbranched_edge_conductance,
)
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
from .tridiag import pcr_solve_parallel_t, pcr_solve_t
from .tridiag.block import block_pcr_solve_t
from .triton import (
    pcr_solve_cuda_t,
    solve_bt_spd_cuda_consume_unchecked,
    thomas_solve_cuda_bt,
    thomas_solve_cuda_t,
)

# Older solver packages expose identically named reverse-only SPD wrappers.
# Admit only the framework-owned pure facades that declare the transform
# contract, and retain their identities when validating a configured solver.
_FUNCTIONAL_SCALAR_SPD = getattr(dendra_solvers, "solve_tri_spd", None)
if not getattr(_FUNCTIONAL_SCALAR_SPD, "_DENDRA_FUNC_COMPATIBLE", False):
    _FUNCTIONAL_SCALAR_SPD = None
_FUNCTIONAL_BLOCK_SPD = getattr(dendra_solvers, "solve_bt_spd", None)
if not getattr(_FUNCTIONAL_BLOCK_SPD, "_DENDRA_FUNC_COMPATIBLE", False):
    _FUNCTIONAL_BLOCK_SPD = None
_NATIVE_BLOCK_SPD_CONSUME = getattr(
    dendra_solvers, "solve_bt_spd_consume_unchecked", None
)

# Only exact, framework-owned Thomas/PCR solves are authorized to receive a
# voltage-increment RHS. A custom callable keeps its historical absolute-RHS
# contract, even when it shares a standard solver's name or wrapped metadata.
# Capture identities once, so a later monkeypatch cannot authorize itself.
_BWD_EULER_INCREMENT_SOLVERS = tuple(
    solve
    for solve in (
        pcr_solve_t,
        pcr_solve_parallel_t,
        pcr_solve_cuda_t,
        thomas_solve_cuda_t,
        _functional_pcr_solve,
        _functional_thomas_solve,
        (
            getattr(torch.ops.dendra_solvers, "thomas_solve_t", None)
            if DENDRA_SOLVERS_AVAILABLE
            else None
        ),
        (
            getattr(torch.ops.dendra_solvers, "pcr_solve_t", None)
            if DENDRA_SOLVERS_AVAILABLE
            else None
        ),
        getattr(dendra_solvers, "thomas_solve_t", None),
        getattr(dendra_solvers, "pcr_solve_t", None),
    )
    if callable(solve)
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

    # Integrator-owned tensor workspace contract. Topology adapters provide
    # effective physical tensors in their natural broadcast layouts; the
    # single-compartment solver deliberately retains those layouts rather than
    # expanding them to the full voltage shape.
    _PREPARED_WORKSPACE_SCHEMA = (
        ("cmdt", "parameter"),
        ("area", "geometry"),
    )
    _FUNCTIONAL_OPERATOR_KIND = "scalar_point"
    _FUNCTIONAL_CRITICAL_METHODS = ()

    def __init__(self, model, mech, imem=None):
        super().__init__(model, mech, imem)
        self.register_buffer("cmdt", torch.tensor(0.0))
        self.register_buffer("area", torch.tensor(0.0))

    @staticmethod
    def _prepare_workspace(dt, *, cm, area):
        """Purely derive the single-compartment implicit-Euler workspace.

        ``cm`` is effective specific capacitance in uF/cm^2 and ``area`` is
        effective membrane area in cm^2. Both inputs retain their natural
        broadcast layouts. The returned tensors own independent storage while
        preserving autograd edges to every explicit input.
        """
        if torch.is_tensor(dt):
            # Retain a logical singleton axis until the scalar is combined
            # with parameter tensors. This preserves hidden vmap lanes,
            # including a zero-sized lane batch.
            dt_s = dt.reshape(1) * 1.0e-3
        else:
            dt_s = dt * 1.0e-3

        return {
            "cmdt": (1.0e-6 * cm) / dt_s,
            "area": area.clone(memory_format=torch.preserve_format),
        }

    def initialize(self, model, dt):
        workspace = self._derive_prepared_workspace(
            dt,
            cm=model.cm * model.cm_scale,
            area=model.area * model.area_scale,
        )
        self._install_prepared_workspace(workspace)

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_step", model.v, dt, model.celsius, ve, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _voltage_update(
        self,
        v,
        dt,
        itot,
        gtot,
        ve=None,
        intra=None,
        *,
        solver=None,
        membrane_source=None,
    ):
        """Return the shared closed-form single-compartment voltage update."""
        del dt, ve, solver

        denom = self.cmdt + gtot
        if itot is None:
            residual = membrane_source - gtot * v
        else:
            residual = -itot
        if intra is not None:
            residual = residual + intra / self.area
        # Cancel the capacitive old-voltage term analytically. Combining the
        # physical currents first also avoids two separately rounded voltage
        # updates when membrane and applied currents nearly balance.
        v_new = v + residual / denom

        i_membrane = None
        if self.imem:
            i_cap = self.cmdt * (v_new - v)
            i_ion = itot + gtot * (v_new - v)
            i_membrane = (i_cap + i_ion) * self.area
        return v_new, i_membrane

    _DIRECT_MEMBRANE_VOLTAGE_UPDATE = _voltage_update


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
    # Imperative integration leaves voltage ownership to mechanisms. The
    # functional transition still carries the externally imposed model voltage
    # explicitly so recording, continuation and capacitive current retain it.
    _FUNCTIONAL_STATE_NAMES = ("v",)
    _PREPARED_WORKSPACE_SCHEMA = _bwd_euler_sc._PREPARED_WORKSPACE_SCHEMA
    _FUNCTIONAL_OPERATOR_KIND = "scalar_point_skip"
    _FUNCTIONAL_CRITICAL_METHODS = ()
    _prepare_workspace = staticmethod(_bwd_euler_sc._prepare_workspace)

    def __init__(self, model, mech, imem=None):
        super().__init__(model, mech, imem)
        self.register_buffer("cmdt", torch.tensor(0.0))
        self.register_buffer("area", torch.tensor(0.0))

    def initialize(self, model, dt):
        workspace = self._derive_prepared_workspace(
            dt,
            cm=model.cm * model.cm_scale,
            area=model.area * model.area_scale,
        )
        self._install_prepared_workspace(workspace)

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_step", model.v, dt, model.celsius, ve, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _step(
        self,
        v,
        dt,
        temp,
        ve=None,
        intra=None,
        *,
        solver=None,
        call_local_currents=False,
    ):
        del ve, intra, solver
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
        if call_local_currents:
            _, _, itot, ion_current_frame, _ = self._membrane_frame(
                v_new, call_local=True, require_current=True
            )
        else:
            itot, _ = self.mech.i(v_new)
            ion_current_frame = self._capture_ion_current_frame()

        i_membrane = None
        if self.imem:
            i_cap = self.cmdt * (v_new - v_old)
            i_membrane = (i_cap + itot) * self.area
        self._advance_post_current(v_new, dt, temp, ion_current_frame)
        return v_new, i_membrane


class _bwd_euler_sc_multi(MultiIntegrator, _bwd_euler_sc):
    """Packed independent-compartment variant of implicit Euler.

    The numerical workspace and tensor transition are intentionally inherited
    from :class:`_bwd_euler_sc`: concatenation changes ownership/writeback, not
    the per-compartment equation.  A distinct operator kind lets functional
    lowering admit the packed runtime without mistaking it for an ordinary
    :class:`SingleCompartment` topology.
    """

    _FUNCTIONAL_OPERATOR_KIND = "scalar_multi_point"
    _FUNCTIONAL_CRITICAL_METHODS = ()

    def __init__(self, model, mech, imem=None, write_back=True):
        super().__init__(model, mech, imem, write_back)

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_step", model.v, dt, model.celsius, ve, intra
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

    Parameters
    ----------
    method : {"thomas", "inv", "spd", "pcr"}, optional
        Solver backend: Thomas (default; ``"inv"`` maps to ``"thomas"``),
        SPD tridiagonal (CPU only), or PCR (CPU/CUDA fallback).
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    supports_unbranched_cable = True
    # Integrator-owned tensor workspace contract. Topology adapters provide
    # effective physical tensors in solve layout; this schema describes only
    # the numerical state derived from them. ``node`` and ``edge`` are static
    # shape roles consumed by functional lowering, not public API names.
    _PREPARED_WORKSPACE_SCHEMA = (
        ("diag_base", "node"),
        ("lower", "edge"),
        ("upper", "edge"),
        ("g_edge_Cinv", "edge"),
        ("g_edge_Cinv_right", "edge"),
        ("cm_inv", "node"),
        ("scale", "node"),
    )
    _FUNCTIONAL_OPERATOR_KIND = "scalar_path"
    _FUNCTIONAL_CRITICAL_METHODS = ("_select_solver",)

    def __init__(self, model, mech, method: str = "inv", **kw):
        super().__init__(model, mech, **kw)
        if method == "inv":
            method = "thomas"
        self.method = method.lower()

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

    def _select_solver(self, device):
        dev = torch.device(device).type

        # -------- explicit PCR --------
        if self.method == "pcr":
            if dev == "cuda":
                self._solve = pcr_solve_cuda_t
            elif dev == "mps":
                self._solve = (
                    torch.ops.dendra_solvers.pcr_solve_t
                    if DENDRA_SOLVERS_AVAILABLE
                    and _mps_solver_available(dendra_solvers)
                    else pcr_solve_parallel_t
                )
            else:
                self._solve = pcr_solve_t
            return

        # -------- SPD option (CPU only) --------
        if self.method == "spd":
            if dev == "mps":
                raise NotImplementedError(
                    "method='spd' is currently unsupported on MPS; "
                    "select method='thomas' or method='pcr'."
                )
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
        elif dev == "mps":
            self._solve = (
                torch.ops.dendra_solvers.thomas_solve_t
                if DENDRA_SOLVERS_AVAILABLE and _mps_solver_available(dendra_solvers)
                else pcr_solve_parallel_t
            )
        elif dev == "cpu":
            if DENDRA_SOLVERS_AVAILABLE:
                self._solve = torch.ops.dendra_solvers.thomas_solve_t
            else:
                warnings.warn(
                    "Using `bwd_euler_ub` on CPU without dendra-solvers installed. "
                    "Falling back to PCR solver. Install the native package with "
                    "`python -m pip install --upgrade "
                    '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
                )
                self._solve = pcr_solve_t

    def _functional_solver(self):
        """Return the transform-compatible facade for the selected UB solver."""
        solver = self._solve
        if solver is not None and solver is _FUNCTIONAL_SCALAR_SPD:
            return solver
        if solver is not None and solver is thomas_solve_cuda_t:
            from ._cuda_functional import thomas_solve

            return thomas_solve
        if solver is not None and solver is pcr_solve_cuda_t:
            from ._cuda_functional import pcr_solve

            return pcr_solve
        solver_name = getattr(solver, "__name__", None)
        solver_module = getattr(solver, "__module__", None)

        if solver_module == "torch._ops.dendra_solvers":
            facade = (
                None
                if dendra_solvers is None
                else getattr(dendra_solvers, solver_name, None)
            )
            if callable(facade) and solver_name in {"thomas_solve_t", "pcr_solve_t"}:
                return facade
            raise RuntimeError(
                "The selected native CPU solver requires a dendra-solvers build "
                "that exports torch.func-compatible tridiagonal facades. Upgrade "
                "it with `python -m pip install --upgrade "
                '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
            )

        if solver_name in {"pcr_solve_t", "pcr_solve_parallel_t"} and solver_module == (
            "dendra.models.integrators.tridiag.pcr"
        ):
            return solver

        raise RuntimeError(
            "The selected bwd_euler_ub solver is not transform-compatible. "
            "Use a dendra-solvers build with pure SPD facades for method='spd', "
            "or select method='thomas' or method='pcr'."
        )

    @staticmethod
    def _prepare_workspace(dt, *, cm, area, edge_conductance):
        """Purely derive the unbranched implicit-Euler tensor workspace.

        Parameters are already normalized to the solver's ``(B, K)`` layout:
        ``cm`` is effective specific capacitance in uF/cm^2, ``area`` is
        effective membrane area in cm^2, and ``edge_conductance`` is the
        centre-to-centre axial conductance in siemens. The method neither reads
        nor mutates module state, so imperative and functional execution can
        share the exact same numerical preparation.
        """
        if torch.is_tensor(dt):
            # Retain a logical singleton axis until the scalar is combined
            # with spatial tensors. This preserves hidden vmap lanes,
            # including a zero-sized lane batch.
            dt_s = dt.reshape(1) * 1.0e-3
        else:
            dt_s = dt * 1.0e-3

        capacitance = 1.0e-6 * cm * area
        cm_inv = capacitance.reciprocal()

        g_left = edge_conductance / capacitance[..., :-1]
        g_right = edge_conductance / capacitance[..., 1:]
        zeros = torch.zeros_like(capacitance[..., :1])
        diag_base = torch.cat((-g_left, zeros), dim=-1) + torch.cat(
            (zeros, -g_right),
            dim=-1,
        )

        return {
            "diag_base": diag_base,
            "lower": -dt_s * g_right,
            "upper": -dt_s * g_left,
            "g_edge_Cinv": g_left,
            "g_edge_Cinv_right": g_right,
            "cm_inv": cm_inv,
            "scale": area * cm_inv,
        }

    def initialize(self, model, dt):
        """
        Compute geometry-dependent coefficients and select tridiagonal solver
        based on `self.method` and device.
        """
        # Select solver first (depends on device and method)
        self._select_solver(model.device())

        B, K = _model_solve_shape(model)
        self.B, self.K = B, K
        cm = _as_solve_matrix(model.cm, model) * _as_solve_matrix(model.cm_scale, model)

        # ── geometry (all element-wise) ──────────────────────────────
        area_cm2 = _as_solve_matrix(model.area, model) * _as_solve_matrix(
            model.area_scale, model
        )

        # ── edge axial conductance between centres i ↔ i+1 ──────────
        # Native Cable uses exact compiled edge resistance; conventional Axon
        # models retain the established half-cylinder reconstruction.
        g_edge = unbranched_edge_conductance(model)

        # Build every value before committing any of them. The pure builder is
        # also the functional backend's preparation primitive.
        workspace = self._derive_prepared_workspace(
            dt,
            cm=cm,
            area=area_cm2,
            edge_conductance=g_edge,
        )
        self._install_prepared_workspace(workspace)

        self.base_shape = tuple(model.shape)

    def step(self, model, dt, ve=None, intra=None):
        v_new, i_membrane = self._call_kernel(
            "_step", model.v, dt, model.celsius, ve, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _voltage_update(
        self,
        v,
        dt,
        itot,
        gtot,
        ve=None,
        intra=None,
        *,
        solver=None,
        membrane_source=None,
    ):
        """Return the shared unbranched implicit voltage update."""
        # Retain a logical singleton axis until dt is combined with spatial
        # tensors. Scalar-only binary batching otherwise selects lane zero for
        # an empty outer vmap.
        if torch.is_tensor(dt):
            dt_s = dt.reshape(1) * 1.0e-3
        else:
            dt_s = dt * 1.0e-3

        v_flat = _flatten_to_solve(v, self.K, self.base_shape)
        gtot = _flatten_to_solve(gtot, self.K, self.base_shape)
        solve = self._solve if solver is None else solver
        # The SPD facade can regularize pivots, so it is excluded from the
        # exact-solve list.
        # Unsupported/custom solvers retain the absolute-voltage RHS contract.
        increment = any(solve is standard for standard in _BWD_EULER_INCREMENT_SOLVERS)
        if itot is None:
            source = _flatten_to_solve(membrane_source, self.K, self.base_shape)
            f_n = (source - gtot * v_flat if increment else source) * self.scale
        else:
            itot = _flatten_to_solve(itot, self.K, self.base_shape)
            f_n = (-itot if increment else gtot * v_flat - itot) * self.scale

        if increment and self.K > 1:
            delta_v = v_flat[..., 1:] - v_flat[..., :-1]
            left = self.g_edge_Cinv * delta_v
            right = self.g_edge_Cinv_right * delta_v
            # Retain the distinct row-capacitance normalization at both ends
            # of each physical edge. This matrix is generally nonsymmetric.
            axial = torch.cat(
                (left[..., :1], left[..., 1:] - right[..., :-1], -right[..., -1:]),
                dim=-1,
            )
            f_n = f_n + axial

        if ve is not None:
            ve_flat = _flatten_to_solve(ve, self.K, self.base_shape)
            delta_ve = ve_flat[..., 1:] - ve_flat[..., :-1]
            if self.K > 1:
                left = self.g_edge_Cinv * delta_ve
                right = self.g_edge_Cinv_right * delta_ve
                # Prepared conductances may carry vmap lanes that ``ve`` does
                # not. Assemble out of place so those lanes can broadcast.
                extracellular = torch.cat(
                    (
                        left[..., :1],
                        left[..., 1:] - right[..., :-1],
                        -right[..., -1:],
                    ),
                    dim=-1,
                )
                f_n = f_n + extracellular

        if intra is not None:
            f_n = f_n + (
                _flatten_to_solve(intra, self.K, self.base_shape) * self.cm_inv
            )

        rhs = dt_s * f_n if increment else v_flat + dt_s * f_n
        main = 1.0 - dt_s * (self.diag_base - gtot * self.scale)

        if self.K == 1:
            v_np1 = rhs / main
        else:
            v_np1 = solve(self.lower, main, self.upper, rhs)

        if increment:
            v_np1 = v_flat + v_np1

        i_membrane = None
        if self.imem:
            area = self.scale / self.cm_inv
            capacitance = 1.0 / self.cm_inv
            capacitive_conductance = capacitance / dt_s
            ionic_conductance = gtot * area
            ionic_current = itot * area
            membrane_conductance = capacitive_conductance + ionic_conductance
            i_membrane = (
                membrane_conductance * (v_np1 - v_flat) + ionic_current
            ).reshape(self.base_shape)

        return v_np1.reshape(self.base_shape), i_membrane

    _DIRECT_MEMBRANE_VOLTAGE_UPDATE = _voltage_update


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
    _PREPARED_WORKSPACE_SCHEMA = (
        ("lower", "block_edge"),
        ("upper", "block_edge"),
        ("maind", "block_matrix"),
        ("area", "node"),
        ("cm_dt", "node"),
        ("xc_dt", "shell_node"),
        ("c_rad", "block_node"),
        ("xg", "shell_node"),
    )
    _FUNCTIONAL_OPERATOR_KIND = "block_path"
    _FUNCTIONAL_CRITICAL_METHODS = ("_select_solver",)

    def __init__(self, model, mech, imem=None, method="inv", **kwargs):
        if not DENDRA_SOLVERS_AVAILABLE:
            logging.warning(
                "Native CPU block solvers are unavailable. CUDA will use Triton "
                "and MPS will use pure-PyTorch block PCR; install dendra-solvers "
                "for CPU support with `python -m pip install --upgrade "
                '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
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

    def _functional_solver(self):
        """Return a transform-compatible block solver facade.

        Native consuming SPD execution and this nonmutating facade share the
        same factorization and cached-Cholesky adjoint kernels.
        """
        if (
            self._solve is not None
            and self._solve is solve_bt_spd_cuda_consume_unchecked
        ):
            from ._cuda_functional import block_spd_solve

            return block_spd_solve
        if (
            self._solve is not None
            and self._solve is _NATIVE_BLOCK_SPD_CONSUME
            and _FUNCTIONAL_BLOCK_SPD is not None
        ):
            return _FUNCTIONAL_BLOCK_SPD
        if self._solve is not None and self._solve is thomas_solve_cuda_bt:
            from ._cuda_functional import block_thomas_solve

            return block_thomas_solve
        if self._solve is block_pcr_solve_t:
            return block_pcr_solve_t
        if (
            DENDRA_SOLVERS_AVAILABLE
            and self._solve is torch.ops.dendra_solvers.solve_bt
        ):
            native = getattr(dendra_solvers, "solve_bt", None)
            return native if callable(native) else block_pcr_solve_t
        raise RuntimeError(
            "The selected block solver is not transform-compatible. "
            "Use a dendra-solvers build with pure SPD facades for method='spd', "
            "or the standard CPU or CUDA method='thomas' implementation."
        )

    @staticmethod
    def _prepare_workspace(
        dt,
        *,
        cm,
        area,
        intracellular_edge_conductance,
        extracellular_edge_conductance,
        xc,
        xg,
    ):
        """Purely derive the two-layer block implicit-Euler workspace.

        All tensors arrive in flattened solve layout. ``cm``/``xc`` are in
        microfarads per square centimetre, ``xg`` is in siemens per square
        centimetre, ``area`` is in square centimetres, and both edge inputs are
        conductances in siemens.
        """
        if torch.is_tensor(dt):
            dt_s = dt.reshape(1) * 1.0e-3
        else:
            dt_s = dt * 1.0e-3

        cm_dt = cm * 1.0e-6 * area / dt_s
        xc_dt = xc * 1.0e-6 * area.unsqueeze(-1) / dt_s
        radial_g = xg * area.unsqueeze(-1)

        zero_node = torch.zeros_like(cm_dt[..., :1])
        gi_left = torch.cat((zero_node, intracellular_edge_conductance), dim=-1)
        gi_right = torch.cat((intracellular_edge_conductance, zero_node), dim=-1)

        # Derive the sealed-end row from a node-shaped tensor. For K == 1 the
        # edge tensor is empty, so slicing its first edge would incorrectly
        # retain a zero-length compartment axis.
        zero_shell = torch.zeros_like(xc_dt[..., :1, :])
        gx_left = torch.cat(
            (zero_shell, extracellular_edge_conductance),
            dim=-2,
        )
        gx_right = torch.cat(
            (extracellular_edge_conductance, zero_shell),
            dim=-2,
        )

        shell_0 = xc_dt[..., 0] + radial_g[..., 0]
        diagonal_0 = cm_dt + gi_left + gi_right
        diagonal_1 = cm_dt + shell_0 + gx_left[..., 0] + gx_right[..., 0]
        diagonal_2 = (
            shell_0
            + xc_dt[..., 1]
            + radial_g[..., 1]
            + gx_left[..., 1]
            + gx_right[..., 1]
        )
        zero = torch.zeros_like(cm_dt)
        main = torch.stack(
            (
                torch.stack((diagonal_0, -cm_dt, zero), dim=-1),
                torch.stack((-cm_dt, diagonal_1, -shell_0), dim=-1),
                torch.stack((zero, -shell_0, diagonal_2), dim=-1),
            ),
            dim=-2,
        )
        off_diagonal = -torch.cat(
            (
                intracellular_edge_conductance.unsqueeze(-1),
                extracellular_edge_conductance,
            ),
            dim=-1,
        )

        return {
            "lower": off_diagonal.clone(memory_format=torch.preserve_format),
            "upper": off_diagonal.clone(memory_format=torch.preserve_format),
            "maind": main,
            "area": area.clone(memory_format=torch.preserve_format),
            "cm_dt": cm_dt,
            "xc_dt": xc_dt,
            "c_rad": torch.cat((cm_dt.unsqueeze(-1), xc_dt), dim=-1),
            "xg": radial_g,
        }

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

    def _select_solver(self, device):
        dev = torch.device(device).type

        if dev == "mps":
            if self.method == "spd":
                self._solve = block_pcr_solve_t
                return
            self._solve = (
                torch.ops.dendra_solvers.solve_bt
                if DENDRA_SOLVERS_AVAILABLE and _mps_solver_available(dendra_solvers)
                else block_pcr_solve_t
            )
            return

        if self.method == "spd":
            if dev == "cuda":
                # _step constructs fresh B_work/D_work buffers, so use the
                # unchecked consuming path to avoid duplicate per-step validation.
                self._solve = solve_bt_spd_cuda_consume_unchecked
            elif dev == "cpu":
                if not DENDRA_SOLVERS_AVAILABLE:
                    raise ImportError(
                        "_bwd_euler_bt(method='spd') requires the dendra-solvers "
                        "package for CPU execution. Use CUDA or run "
                        "`python -m pip install --upgrade "
                        '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
                    )
                try:
                    from dendra_solvers import solve_bt_spd_consume_unchecked
                except ImportError as err:
                    raise ImportError(
                        "_bwd_euler_bt(method='spd') requires the dendra-solvers "
                        "package for CPU execution. Use CUDA or run "
                        "`python -m pip install --upgrade "
                        '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
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
                        "_bwd_euler_bt(method='thomas') requires the dendra-solvers "
                        "package for CPU execution. Use CUDA or run "
                        "`python -m pip install --upgrade "
                        '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.'
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
        self._select_solver(model.device())
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

        cm = _as_solve_matrix(model.cm, model) * _as_solve_matrix(
            model.cm_scale,
            model,
        )
        area = _as_solve_matrix(model.area, model) * _as_solve_matrix(
            model.area_scale,
            model,
        )
        dx = _as_solve_matrix(model.dx, model)
        extracellular_edge_conductance = _layered_edge_conductance(
            _as_solve_block(model.xraxial, model, (self.M - 1,)),
            dx,
        )
        workspace = self._derive_prepared_workspace(
            dt,
            cm=cm,
            area=area,
            intracellular_edge_conductance=unbranched_edge_conductance(model),
            extracellular_edge_conductance=extracellular_edge_conductance,
            xc=_as_solve_block(model.xc, model, (self.M - 1,)),
            xg=_as_solve_block(model.xg, model, (self.M - 1,)),
        )
        self._install_prepared_workspace(workspace)
        self.base_shape = tuple(model.shape) + (self.M,)

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

    def _step(
        self,
        vc,
        v,
        dt,
        temp,
        ve=None,
        intra=None,
        *,
        solver=None,
        call_local_currents=False,
    ) -> Tuple[Tensor, Tensor]:
        # apply voltage processes
        v_state = self.mech.update_v(v)

        # advance gating
        self._advance_pre_current(v_state, dt, temp)

        (
            gtot,
            membrane_source,
            itot,
            ion_current_frame,
            ion_conductance_frame,
        ) = self._membrane_frame(v_state, call_local=call_local_currents)

        # linearized ionic conductances & reversal
        public_voltage_shape = self.base_shape[:-1]
        v_flat = _flatten_to_solve(v_state, self.K, public_voltage_shape)
        gtot = _flatten_to_solve(gtot, self.K, public_voltage_shape)
        v_prev_membrane = vc[..., 0] - vc[..., 1]

        if itot is None:
            # Capacitance uses vc, not the separately carried membrane v.
            # Passive direct coefficients can be independent of v entirely.
            # Retain the legacy cancellation's zero AD edge to that input so
            # differentiating both state fields still returns zero, not unused.
            d = (
                _flatten_to_solve(membrane_source, self.K, public_voltage_shape)
                + 0.0 * v_flat
                - gtot * v_prev_membrane
            ) * self.area
        else:
            itot = _flatten_to_solve(itot, self.K, public_voltage_shape)
            d = (gtot * (v_flat - v_prev_membrane) - itot) * self.area
            itot = itot * self.area
        gtot = gtot * self.area

        if intra is not None:
            d = d + _flatten_to_solve(intra, self.K, self.base_shape[:-1])

        # Assemble out of place so hidden torch.func lanes on current parameters
        # can broadcast over the shared prepared block matrix. The result is a
        # disposable workspace for consuming native SPD solvers as well.
        zero = torch.zeros_like(gtot)
        ionic_blocks = torch.stack(
            (
                torch.stack((gtot, -gtot, zero), dim=-1),
                torch.stack((-gtot, gtot, zero), dim=-1),
                torch.stack((zero, zero, zero), dim=-1),
            ),
            dim=-2,
        )
        B_work = self.maind + ionic_blocks

        ve_flat = (
            _flatten_to_solve(ve, self.K, self.base_shape[:-1])
            if ve is not None
            else None
        )
        # Solve for a voltage increment. Forming rhs - A @ vc numerically
        # would cancel large capacitance terms; their exact cancellation leaves
        # only the physical membrane, shell, bath and axial current residuals.
        D_work = _block_voltage_residual(vc, d, self.xg, ve_flat)
        axial_flux = -self.lower * (vc[..., 1:, :] - vc[..., :-1, :])
        sealed_end = torch.zeros_like(vc[..., :1, :])
        D_work = (
            D_work
            + torch.cat((axial_flux, sealed_end), dim=-2)
            - torch.cat((sealed_end, axial_flux), dim=-2)
        )

        # solve block-tridiagonal system
        solve = self._solve if solver is None else solver
        vc_new = (vc + solve(self.lower, B_work, self.upper, D_work)).reshape(
            self.base_shape
        )
        v_new = vc_new[..., 0] - vc_new[..., 1]  # v = vi - ve0

        i_membrane = None

        if self.imem:
            vprev_mem = vc[..., 0] - vc[..., 1]
            delta_v = _flatten_to_solve(
                v_new,
                self.K,
                public_voltage_shape,
            ) - vprev_mem.reshape(-1, self.K)
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


def _block_voltage_residual(v_prev, membrane_residual, xg, e_ext):
    """Assemble physical currents for a three-component voltage increment."""
    zero = torch.zeros_like(membrane_residual)
    shell_flux = xg[..., 0] * (v_prev[..., 1] - v_prev[..., 2])
    bath_voltage = zero if e_ext is None else e_ext
    bath_flux = xg[..., 1] * (bath_voltage - v_prev[..., 2])
    membrane = torch.stack((membrane_residual, -membrane_residual, zero), dim=-1)
    radial = torch.stack((zero, -shell_flux, shell_flux + bath_flux), dim=-1)
    return membrane + radial


def assemble_rhs(v_prev, c_rad, d, xg, e_ext):
    v_c = c_rad[..., :-1] * (v_prev[..., :-1] - v_prev[..., 1:])
    radial = torch.cat(
        (
            v_c[..., :1],
            v_c[..., 1:] - v_c[..., :-1],
            -v_c[..., -1:],
        ),
        dim=-1,
    )
    zero = torch.zeros_like(d)
    membrane = torch.stack((d, -d, zero), dim=-1)
    boundary = c_rad[..., -1] * v_prev[..., -1]
    if e_ext is not None:
        boundary = boundary + xg * e_ext
    bath = torch.stack((zero, zero, boundary), dim=-1)
    return radial + membrane + bath


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
