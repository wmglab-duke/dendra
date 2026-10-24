from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from .core import (
    Integrator,
    _broadcast_to_shape,
    _expanded_v_init,
    _flatten_to_solve,
)


def _reflect_pad_last(x: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
    """Reflect-pad the final axis, including axes shorter than the padding.

    ``torch.nn.functional.pad(..., mode="reflect")`` requires each padding
    width to be smaller than the input axis.  A one-compartment cable therefore
    failed even though its sealed-boundary extension is well defined.  Building
    the reflected index map directly also handles wide smoothing stencils on
    very short cables while preserving autograd.
    """
    left, right = (int(padding[0]), int(padding[1]))
    size = int(x.shape[-1])
    if size < 1:
        raise ValueError("cannot reflect-pad an empty final dimension")
    if left == 0 and right == 0:
        return x
    if size == 1:
        return x.expand(*x.shape[:-1], left + 1 + right)

    positions = torch.arange(-left, size + right, device=x.device, dtype=torch.long)
    period = 2 * (size - 1)
    positions = torch.remainder(positions, period)
    indices = torch.where(positions < size, positions, period - positions)
    return x.index_select(-1, indices)


@torch.compiler.assume_constant_result
def _has_empty_stencil_vmap_axis(value):
    """Inspect hidden transform shapes without reading numerical tensor data."""
    functorch = torch._C._functorch
    while functorch.is_functorch_wrapped_tensor(value):
        is_batched = functorch.is_batchedtensor(value)
        batch_axis = functorch.maybe_get_bdim(value)
        value = functorch.get_unwrapped(value)
        if is_batched and value.shape[batch_axis] == 0:
            return True
    return False


def _tensor_stencil_conv1d(conv, x, weight, padding):
    """Evaluate a small stencil without external convolution buffer aliases.

    Single-row reflection and zero-sized hidden input or weight batches use
    this equivalent tensor expression. Other evaluations retain the ordinary
    convolution kernel. No numerical tensor values are inspected or unwrapped.
    """
    if conv.padding_mode == "reflect":
        x = _reflect_pad_last(x, padding)
    elif conv.padding_mode == "zeros":
        x = F.pad(x, padding)
    else:
        x = F.pad(x, padding, mode=conv.padding_mode)
    dilation = int(conv.dilation[0])
    stride = int(conv.stride[0])
    width = (int(weight.shape[-1]) - 1) * dilation + 1
    count = (x.shape[-1] - width) // stride + 1
    # Static tap slices keep the empty-lane reverse path in ordinary tensor
    # operations; unfold_backward has no zero-batch vmap rule.
    windows = torch.stack(
        [
            x[..., offset : offset + count * stride : stride]
            for offset in range(0, width, dilation)
        ],
        dim=-1,
    )
    result = (windows.unsqueeze(1) * weight.unsqueeze(0).unsqueeze(-2)).sum(
        dim=(-1, -3)
    )
    if conv.bias is not None:
        result = result + conv.bias.reshape(1, -1, 1)
    return result


def _conv1d_forward(
    conv: torch.nn.Conv1d, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Run Conv1d with robust reflection padding for short final axes."""
    padding = tuple(int(value) for value in conv._reversed_padding_repeated_twice)
    # Single-row channel views can look contiguous to Inductor's reuse
    # planner while RK stages still need aliases of their stack storage.
    if conv.groups == 1 and (
        (conv.padding_mode == "reflect" and x.shape[0] == 1)
        or (
            torch._C._are_functorch_transforms_active()
            and (
                _has_empty_stencil_vmap_axis(x) or _has_empty_stencil_vmap_axis(weight)
            )
        )
    ):
        return _tensor_stencil_conv1d(conv, x, weight, padding)
    if conv.padding_mode == "reflect" and (
        padding[0] >= x.shape[-1] or padding[1] >= x.shape[-1]
    ):
        x = _reflect_pad_last(x, padding)
        return F.conv1d(
            x,
            weight,
            conv.bias,
            conv.stride,
            0,
            conv.dilation,
            conv.groups,
        )
    return conv._conv_forward(x, weight, conv.bias)


def _conv_last(conv: torch.nn.Conv1d, *xs: torch.Tensor) -> torch.Tensor:
    """Apply a Conv1d stencil along the final dimension of arbitrary-shaped tensors."""
    ref = xs[0]
    K = ref.shape[-1]
    x = torch.stack(
        [_flatten_to_solve(t, K, tuple(ref.shape)) for t in xs],
        dim=1,
    )
    return conv(x).squeeze(1).reshape_as(ref)


def _filter_last(filter_: torch.nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """Apply a single-channel Conv1d filter along the final axis."""
    xf = _flatten_to_solve(x, target_shape=tuple(x.shape)).unsqueeze(1)
    return _conv1d_forward(filter_, xf, filter_.weight).squeeze(1).reshape_as(x)


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return _conv1d_forward(self, x, weight_)


class _euler(Integrator):
    r"""
    Explicit forward Euler integrator for membrane voltage PDEs.

    Uses a single-stage explicit update

    .. math::
       v^{n+1} = v^{n} + \Delta t \, f(v^{n}, t^{n})

    where :math:`f` combines ionic currents and axial diffusion terms. The
    operator relies on a symmetric convolution stencil to approximate the
    spatial second derivative of the cable equation.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    _PREPARED_WORKSPACE_SCHEMA = (
        ("cm_inv", "node"),
        ("ra_inv", "node"),
        ("cm_c", "node"),
        ("area_c", "node"),
        ("ve_zero", "node"),
    )
    _FUNCTIONAL_OPERATOR_KIND = "scalar_explicit_path"
    _FUNCTIONAL_CRITICAL_METHODS = ("_functional_step", "_explicit_slope", "FRK")

    def __init__(self, model, mech, imem=None):
        super().__init__(model, mech, imem)

        self.register_buffer("cm_inv", torch.tensor(0.0))
        self.register_buffer("ra_inv", torch.tensor(0.0))
        self.register_buffer("ve_zero", torch.tensor(0.0))

        self.register_buffer("cm_c", torch.tensor(0.0))
        self.register_buffer("area_c", torch.tensor(0.0))

        weight = [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]]
        nc = 2
        nw = 3

        self.ssd = SymmetricConv1D(
            nc, 1, nw, bias=False, padding="same", padding_mode="reflect"
        )
        self.ssd.weight.data = torch.tensor(weight).reshape(1, nc, nw)
        for p in self.ssd.parameters():
            p.requires_grad = False

    @staticmethod
    def _prepare_workspace(dt, *, dx, diam, cm, rhoa):
        """Derive the explicit cable workspace from effective model tensors.

        Inputs use the public voltage shape and native units: micrometres for
        ``dx`` and ``diam``, uF/cm^2 for ``cm``, and ohm cm for ``rhoa``. The
        explicit integrators use each compartment's cylindrical resistance and
        area, including the materialized node geometry of myelinated models.
        The timestep does not enter these coefficients. Preparation preserves
        gradients and never reads or mutates an integrator.
        """
        del dt
        dx = dx / 10000.0  # cm
        area = torch.pi * (diam / 10000.0) * dx  # cm^2
        cm = cm / 1000.0 * area
        ra = (rhoa * dx) / (torch.pi * (diam / 20000.0) ** 2)
        return {
            "cm_inv": 1.0 / cm,
            "ra_inv": 1.0 / ra,
            "cm_c": cm,
            "area_c": area,
            "ve_zero": torch.zeros_like(cm),
        }

    def initialize(self, model, dt) -> None:
        workspace = self._derive_prepared_workspace(
            dt,
            dx=_broadcast_to_shape(model.dx, model.shape),
            diam=_broadcast_to_shape(model.diam, model.shape),
            cm=_broadcast_to_shape(model.cm, model.shape),
            rhoa=_broadcast_to_shape(model.rhoa, model.shape),
        )
        self._install_prepared_workspace(workspace)

    def _explicit_slope(self, v, ve, area, cm, ra, intra, *, call_local_currents):
        """Evaluate one explicit stage with shared native/current arithmetic."""
        d2v = _conv_last(self.ssd, v, ve)
        if call_local_currents:
            evaluate = getattr(self.mech, "_evaluate_explicit_current_frame", None)
            if evaluate is None:
                raise RuntimeError(
                    "call-local explicit current evaluation requires a "
                    "MechanismHandler with _evaluate_explicit_current_frame()"
                )
            current, current_frame = evaluate(v)
        else:
            current = self.mech.iexp(v)
            current_frame = None
        i_ion = current * area
        i_drive = i_ion
        if intra is not None:
            intra = _broadcast_to_shape(intra, tuple(v.shape))
            i_drive = i_drive - intra
        return cm * ((ra * d2v) - i_drive), i_ion, current_frame

    def FRK(self, v, ve, area, cm, ra, intra=None):
        slope, ionic, _frame = self._explicit_slope(
            v, ve, area, cm, ra, intra, call_local_currents=False
        )
        return slope, ionic

    def step(self, model, dt, ve=None, intra=None):
        if ve is None:
            ve = self.ve_zero
        v_new, i_membrane = self._call_kernel(
            "_step", model.v, ve, self.area_c, dt, model.celsius, self.cm_c, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _functional_step(
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
        """Route the common functional signature to the native explicit kernel."""
        del solver
        if ve is None:
            ve = self.ve_zero
        return self._step(
            v,
            ve,
            self.area_c,
            dt,
            temp,
            self.cm_c,
            intra,
            call_local_currents=call_local_currents,
        )

    def _step(
        self, v, ve, area, dt, temp, cm, intra=None, *, call_local_currents=False
    ):
        self._advance_pre_current(v, dt, temp)
        if call_local_currents:
            K1, i_ion, accepted_frame = self._explicit_slope(
                v,
                ve,
                area,
                self.cm_inv,
                self.ra_inv,
                intra,
                call_local_currents=True,
            )
        else:
            K1, i_ion = self.FRK(v, ve, area, self.cm_inv, self.ra_inv, intra)
            accepted_frame = self._capture_ion_current_frame()
        v_n = v + K1 * dt
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            i_membrane = i_cap + i_ion
        return v_n, i_membrane


class _eulerv1(_euler):
    r"""
    Single-stage forward Euler integrator for scalar cable models.

    Uses the same voltage update, cylindrical coefficients, and symmetric
    spatial stencil as :class:`_euler`.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    _FUNCTIONAL_OPERATOR_KIND = "scalar_explicit_path"
    _step = _euler._step


_rk1 = _euler


class _rk2(_euler):
    r"""
    Two-stage explicit Runge-Kutta (midpoint) integrator.

    Computes

    .. math::
       k_1 = f(v^n, t^n), \quad
       k_2 = f\!\left(v^n + \tfrac{\Delta t}{2} k_1, t^n + \tfrac{\Delta t}{2}\right), \\
       v^{n+1} = v^n + \Delta t \, k_2

    providing second-order accuracy for the cable equation with explicit spatial
    diffusion and ionic currents.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    _FUNCTIONAL_OPERATOR_KIND = "scalar_explicit_path"

    def _step(
        self, v, ve, area, dt, temp, cm, intra=None, *, call_local_currents=False
    ):
        self._advance_pre_current(v, dt, temp)
        if call_local_currents:
            K1, _, _ = self._explicit_slope(
                v,
                ve,
                area,
                self.cm_inv,
                self.ra_inv,
                intra,
                call_local_currents=True,
            )
            K2, i2, accepted_frame = self._explicit_slope(
                v + K1 * dt / 2.0,
                ve,
                area,
                self.cm_inv,
                self.ra_inv,
                intra,
                call_local_currents=True,
            )
        else:
            K1, _ = self.FRK(v, ve, area, self.cm_inv, self.ra_inv, intra)
            K2, i2 = self.FRK(
                v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv, intra
            )
            accepted_frame = self._capture_ion_current_frame()
        v_n = v + K2 * dt
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            i_membrane = i_cap + i2
        return v_n, i_membrane


class _rk4(_euler):
    r"""
    Classical four-stage Runge-Kutta integrator.

    Evaluates four slopes (:math:`k_1..k_4`) at staged voltage predictions and
    combines them with the standard RK4 weights to yield fourth-order accuracy
    for the explicit cable equation update.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    _FUNCTIONAL_OPERATOR_KIND = "scalar_explicit_path"
    _FUNCTIONAL_CRITICAL_METHODS = (
        *_euler._FUNCTIONAL_CRITICAL_METHODS,
        "_combine_ion_current_frames",
    )

    def _step(
        self, v, ve, area, dt, temp, cm, intra=None, *, call_local_currents=False
    ):
        self._advance_pre_current(v, dt, temp)
        if call_local_currents:
            K1, i1, frame1 = self._explicit_slope(
                v,
                ve,
                area,
                self.cm_inv,
                self.ra_inv,
                intra,
                call_local_currents=True,
            )
            K2, i2, frame2 = self._explicit_slope(
                v + K1 * dt / 2.0,
                ve,
                area,
                self.cm_inv,
                self.ra_inv,
                intra,
                call_local_currents=True,
            )
            K3, i3, frame3 = self._explicit_slope(
                v + K2 * dt / 2.0,
                ve,
                area,
                self.cm_inv,
                self.ra_inv,
                intra,
                call_local_currents=True,
            )
            K4, i4, frame4 = self._explicit_slope(
                v + K3 * dt,
                ve,
                area,
                self.cm_inv,
                self.ra_inv,
                intra,
                call_local_currents=True,
            )
        else:
            K1, i1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv, intra)
            frame1 = self._capture_ion_current_frame()
            K2, i2 = self.FRK(
                v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv, intra
            )
            frame2 = self._capture_ion_current_frame()
            K3, i3 = self.FRK(
                v + K2 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv, intra
            )
            frame3 = self._capture_ion_current_frame()
            K4, i4 = self.FRK(v + K3 * dt, ve, area, self.cm_inv, self.ra_inv, intra)
            frame4 = self._capture_ion_current_frame()
        v_n = v + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6.0
        accepted_frame = self._combine_ion_current_frames(
            (frame1, frame2, frame3, frame4),
            (1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0),
        )
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            i_rk4 = (i1 + 2 * i2 + 2 * i3 + i4) / 6.0
            i_membrane = i_cap + i_rk4
        return v_n, i_membrane


def ssd_df(v_c, v_p, v_e):
    K = v_c.shape[-1]
    vc = _flatten_to_solve(v_c, K, tuple(v_c.shape))
    vp = _flatten_to_solve(v_p, K, tuple(v_c.shape))
    ve = _flatten_to_solve(v_e, K, tuple(v_c.shape))
    vc_p = _reflect_pad_last(vc, (1, 1))
    ve_p = _reflect_pad_last(ve, (1, 1))
    ret = vc_p[:, :-2] + vc_p[:, 2:] - vp + ve_p[:, 2:] + ve_p[:, :-2] - 2 * ve
    return ret.reshape_as(v_c)


def ssd_df_no_ve(v_c, v_p):
    K = v_c.shape[-1]
    vc = _flatten_to_solve(v_c, K, tuple(v_c.shape))
    vp = _flatten_to_solve(v_p, K, tuple(v_c.shape))
    vc_p = _reflect_pad_last(vc, (1, 1))
    return (vc_p[:, :-2] + vc_p[:, 2:] - vp).reshape_as(v_c)


class _dufort_frankel_homogeneous(Integrator):
    r"""
    Dufort-Frankel explicit integrator for homogeneous morphologies.

    Uses the Dufort-Frankel scheme on the cable diffusion term to achieve
    unconditional stability for the linear part. Currents proven or declared
    affine in voltage are centered exactly between the stored time levels;
    other currents are evaluated explicitly with zero Dufort conductance. A
    one-``dt`` forward-Euler step constructs the missing history level after
    initialization or a timestep change. Optionally applies spatial smoothing
    after each voltage update. Can compute with either convolution-based or direct
    finite-difference stencils.

    Parameters
    ----------
    beta : float, optional
        Exponential smoothing factor (1.0 disables smoothing). Default 1.0.
    smooth_every : int, optional
        Retained configuration metadata. Spatial smoothing is applied each
        timestep when ``beta < 1``. Default 100.
    conv : bool, optional
        If True, use a fixed convolution kernel for the spatial stencil;
        otherwise use direct finite differences. Default False.
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    __constants__ = ["beta", "smoothing", "smooth_every", "imem"]
    v_vars = ["v", "v_prev", "_df_history_valid", "_df_history_dt"]
    _FUNCTIONAL_OPERATOR_KIND = "scalar_dufort_path"
    _FUNCTIONAL_STATE_NAMES = tuple(v_vars)
    _FUNCTIONAL_CRITICAL_METHODS = (
        "_functional_step",
        "_functional_tensor_step",
        "_prepare_history",
        "_step_initial",
        "_step_conv",
        "_initial_current_frame",
        "_dufort_current_frame",
        "_initial_spatial",
        "_initial_voltage",
        "_history_spatial",
        "_history_voltage",
    )
    _PREPARED_WORKSPACE_SCHEMA = (
        ("area", "node"),
        ("s1", "node"),
        ("s2", "node"),
        ("s3", "node"),
        ("s4", "node"),
        ("ve_zero", "node"),
    )

    def __init__(self, model, mech, beta=1.0, smooth_every=100, conv=False, imem=None):
        super().__init__(model, mech, imem)

        model.register_buffer("v_prev", _expanded_v_init(model).clone().detach())
        model.register_buffer(
            "_df_history_valid",
            torch.zeros((), dtype=torch.bool, device=model.v.device),
        )
        model.register_buffer(
            "_df_history_dt",
            torch.full((), torch.nan, dtype=model.v.dtype, device=model.v.device),
        )

        self.register_buffer("area", torch.tensor(0.0))
        self.register_buffer("s1", torch.tensor(0.0))
        self.register_buffer("s2", torch.tensor(0.0))
        self.register_buffer("s3", torch.tensor(0.0))
        self.register_buffer("s4", torch.tensor(0.0))
        self.register_buffer("ve_zero", torch.tensor(0.0), persistent=False)

        self.smooth_every = smooth_every
        self.beta = beta
        self.f64 = False
        self.smoothing = bool((1 - beta))
        self.conv = conv

        if self.smoothing:
            self.filter = torch.nn.Conv1d(
                1, 1, 5, padding=2, bias=False, padding_mode="reflect"
            )
            self.filter.weight.data = torch.tensor(
                [0.25, 0.25, 0, 0.25, 0.25], dtype=torch.float
            ).reshape(1, 1, 5)
            for p in self.filter.parameters():
                p.requires_grad = False

        weight = [
            [1.0, +0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, -2.0, 1.0],
        ]
        nc = 3
        nw = 3
        self.ssd = SymmetricConv1D(
            nc, 1, nw, bias=False, padding="same", padding_mode="reflect"
        )
        self.ssd.weight.data = torch.tensor(weight).reshape(1, nc, nw)
        for p in self.ssd.parameters():
            p.requires_grad = False

    @staticmethod
    def _prepare_workspace(dt, *, dx, diam, cm, rhoa):
        """Derive the native homogeneous two-level cable coefficients."""
        area = torch.pi * (diam / 10000) * (dx / 10000)
        cm = (cm / 1e3) * area
        dx = dx / 10000
        radii = diam / 20000
        ra = (rhoa * dx) / (torch.pi * radii**2)
        s1 = 2 * dt / cm
        s2 = s1 / ra
        if cm.shape[-1] == 1:
            # A sealed one-compartment cable has no axial edge. Reflection is
            # useful for boundary stencils on longer cables, but treating the
            # lone compartment as its own neighbour creates spurious temporal
            # coupling in the two-level Dufort update.
            s2 = torch.zeros_like(s2)
        return {
            "area": area,
            "s1": s1,
            "s2": s2,
            "s3": area * s1,
            "s4": 1 + s2,
            "ve_zero": torch.zeros_like(cm),
        }

    def initialize(self, model, dt) -> None:
        workspace = self._derive_prepared_workspace(
            dt,
            dx=_broadcast_to_shape(model.dx, model.shape),
            diam=_broadcast_to_shape(model.diam, model.shape),
            cm=_broadcast_to_shape(model.cm, model.shape),
            rhoa=_broadcast_to_shape(model.rhoa, model.shape),
        )
        self._install_prepared_workspace(workspace)
        self.f64 = model.dtype() == torch.float64

        # Forced coefficient rebuilds in training do not invalidate a same-dt
        # history. A genuine timestep change does, including after restoring a
        # checkpoint into a fresh integrator instance.
        model._df_history_valid, model._df_history_dt = self._prepare_history(
            model._df_history_valid, model._df_history_dt, dt
        )

    @staticmethod
    def _prepare_history(valid, history_dt, dt):
        """Preserve same-timestep history and invalidate a changed timestep."""
        dt_value = torch.as_tensor(dt, dtype=history_dt.dtype, device=history_dt.device)
        # Give scalar metadata a visible axis for empty hidden vmap lanes.
        same_dt = torch.all(
            history_dt.reshape(history_dt.numel()) == dt_value.reshape(1)
        )
        valid = (valid.reshape(valid.numel()) & same_dt.reshape(1)).reshape(valid.shape)
        history_dt = (
            torch.zeros_like(history_dt).reshape(history_dt.numel())
            + dt_value.reshape(1)
        ).reshape(history_dt.shape)
        return valid, history_dt

    def step(self, model, dt, ve=None, intra=None):
        if ve is None:
            if self.conv:
                ve = self.ve_zero
        if bool(torch.all(model._df_history_valid).item()):
            v_new, v_prev_new, i_membrane = self._call_kernel(
                "_step_conv" if self.conv else "_step",
                model.v,
                model.v_prev,
                ve,
                self.s1,
                self.s2,
                self.s3,
                self.s4,
                self.area,
                dt,
                model.celsius,
                intra,
            )
        else:
            v_new, v_prev_new, i_membrane = self._call_kernel(
                "_step_initial",
                model.v,
                ve,
                self.s1,
                self.s2,
                self.area,
                dt,
                model.celsius,
                intra,
            )
        model.v = v_new
        model.v_prev = v_prev_new
        model._df_history_valid = torch.ones_like(model._df_history_valid)
        if self.imem:
            model.i_membrane = i_membrane

    def _initial_current_frame(self, v, *, call_local_currents):
        if call_local_currents:
            return self.mech._evaluate_explicit_current_frame(v)
        current = self.mech.iexp(v)
        return current, self._capture_ion_current_frame()

    def _dufort_current_frame(self, v, v_prev, *, call_local_currents):
        if call_local_currents:
            return self.mech._evaluate_dufort_current_frame(v, v_prev)
        current, conductance = self.mech.idf(v, v_prev)
        return (
            current,
            conductance,
            self._capture_ion_current_frame(),
            self._capture_ion_conductance_frame(),
        )

    def _functional_step(
        self,
        v,
        dt,
        temp,
        ve=None,
        intra=None,
        *,
        history,
        use_history,
        solver=None,
        call_local_currents=True,
        current_dispatch=None,
    ):
        """Evaluate one selected history phase with the native numerical kernel.

        ``current_dispatch`` selects the current phase with an isolated pure
        tensor packet. Mechanism advances and smoothing execute once around
        that packet. The static phase adapter remains available when no packet
        dispatcher is needed. History never requires a host tensor read.
        """
        del solver
        v_prev, valid, history_dt = history
        if current_dispatch is not None:
            v_new, v_prev_new, i_membrane = self._functional_tensor_step(
                v, v_prev, dt, temp, ve, intra, use_history, current_dispatch
            )
        elif use_history:
            if self.conv and ve is None:
                ve = self.ve_zero
            step = self._step_conv if self.conv else self._step
            v_new, v_prev_new, i_membrane = step(
                v,
                v_prev,
                ve,
                self.s1,
                self.s2,
                self.s3,
                self.s4,
                self.area,
                dt,
                temp,
                intra,
                call_local_currents=call_local_currents,
            )
        else:
            v_new, v_prev_new, i_membrane = self._step_initial(
                v,
                ve,
                self.s1,
                self.s2,
                self.area,
                dt,
                temp,
                intra,
                call_local_currents=call_local_currents,
            )
        return (
            v_new,
            v_prev_new,
            torch.ones_like(valid),
            torch.zeros_like(history_dt)
            + torch.as_tensor(dt, dtype=history_dt.dtype, device=history_dt.device),
            i_membrane,
        )

    def _functional_tensor_step(
        self, v, v_prev, dt, temp, ve, intra, use_history, current_dispatch
    ):
        # Invalid checkpoints may contain an arbitrary unused history level.
        # Keep its NaNs out of the inactive numerical branch and its backward.
        v_prev = torch.where(use_history, v_prev, v)
        initial_spatial = self._initial_spatial(v, ve)
        history_spatial = self._history_spatial(v, v_prev, ve)
        self._advance_pre_current(v, dt, temp)
        current, conductance, i_frame, g_frame = current_dispatch(
            v, v_prev, use_history
        )
        initial_voltage = self._initial_voltage(v, initial_spatial, current, intra)
        history_voltage = self._history_voltage(
            v, v_prev, history_spatial, current, conductance, intra
        )
        unsmoothed = torch.where(use_history, history_voltage, initial_voltage)
        accepted_frame = self._linearize_ion_current_frame(
            i_frame, g_frame, unsmoothed, conductance_scale=0.5
        )
        i_mem = torch.where(
            use_history, current + 0.5 * conductance * unsmoothed, current
        )
        v_new = unsmoothed
        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = torch.where(
                use_history,
                (v_new - v_prev) / self.s1,
                2.0 * (v_new - v) / self.s1,
            )
            i_membrane = i_cap + i_mem * self.area
        return v_new, v, i_membrane

    def _initial_spatial(self, v, ve, *, s2=None):
        del s2
        if ve is None:
            return ssd_df_no_ve(v, v) - v
        return ssd_df(v, v, ve) - v

    def _initial_voltage(
        self, v, spatial, current, intra, *, s1=None, s2=None, area=None
    ):
        s1 = self.s1 if s1 is None else s1
        s2 = self.s2 if s2 is None else s2
        area = self.area if area is None else area
        i_drive = current * area
        if intra is not None:
            i_drive = i_drive - _broadcast_to_shape(intra, tuple(v.shape))
        # The prepared coefficients contain 2*dt; the starter spans one dt.
        return v + 0.5 * s2 * spatial - 0.5 * s1 * i_drive

    def _history_spatial(self, v, v_prev, ve, *, s2=None, conv=None):
        s2 = self.s2 if s2 is None else s2
        conv = self.conv if conv is None else conv
        if conv:
            ve = self.ve_zero if ve is None else ve
            return s2 * _conv_last(self.ssd, v, v_prev, ve)
        if ve is None:
            return s2 * ssd_df_no_ve(v, v_prev)
        return s2 * ssd_df(v, v_prev, ve)

    def _history_voltage(
        self,
        v,
        v_prev,
        spatial,
        current,
        conductance,
        intra,
        *,
        s1=None,
        s3=None,
        s4=None,
        area=None,
    ):
        s1 = self.s1 if s1 is None else s1
        s3 = self.s3 if s3 is None else s3
        s4 = self.s4 if s4 is None else s4
        area = self.area if area is None else area
        i_drive = current * area
        if intra is not None:
            i_drive = i_drive - _broadcast_to_shape(intra, tuple(v.shape))
        return (v_prev + spatial - s1 * i_drive) / (s4 + 0.5 * conductance * s3)

    def _step_initial(
        self, v, ve, s1, s2, area, dt, temp, intra=None, *, call_local_currents=False
    ):
        """Construct the missing previous time level with one Euler step."""
        spatial = self._initial_spatial(v, ve, s2=s2)
        self._advance_pre_current(v, dt, temp)
        i_mem, accepted_frame = self._initial_current_frame(
            v, call_local_currents=call_local_currents
        )
        v_new = self._initial_voltage(v, spatial, i_mem, intra, s1=s1, s2=s2, area=area)
        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = 2.0 * (v_new - v) / s1
            i_membrane = i_cap + i_mem * area
        return v_new, v, i_membrane

    def _step(
        self,
        v,
        v_prev,
        ve,
        s1,
        s2,
        s3,
        s4,
        area,
        dt,
        temp,
        intra=None,
        *,
        call_local_currents=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        spatial = self._history_spatial(v, v_prev, ve, s2=s2, conv=False)
        self._advance_pre_current(v, dt, temp)
        i_ion, gtot, i_frame, g_frame = self._dufort_current_frame(
            v, v_prev, call_local_currents=call_local_currents
        )
        v_new = self._history_voltage(
            v, v_prev, spatial, i_ion, gtot, intra, s1=s1, s3=s3, s4=s4, area=area
        )
        accepted_frame = self._linearize_ion_current_frame(
            i_frame, g_frame, v_new, conductance_scale=0.5
        )
        i_mem = i_ion + 0.5 * gtot * v_new
        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = (v_new - v_prev) / s1
            i_membrane = i_cap + i_mem * area
        return v_new, v, i_membrane

    def _step_conv(
        self,
        v,
        v_prev,
        ve,
        s1,
        s2,
        s3,
        s4,
        area,
        dt,
        temp,
        intra=None,
        *,
        call_local_currents=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        spatial = self._history_spatial(v, v_prev, ve, s2=s2, conv=True)
        self._advance_pre_current(v, dt, temp)
        i_ion, gtot, i_frame, g_frame = self._dufort_current_frame(
            v, v_prev, call_local_currents=call_local_currents
        )
        v_new = self._history_voltage(
            v, v_prev, spatial, i_ion, gtot, intra, s1=s1, s3=s3, s4=s4, area=area
        )
        accepted_frame = self._linearize_ion_current_frame(
            i_frame, g_frame, v_new, conductance_scale=0.5
        )
        i_mem = i_ion + 0.5 * gtot * v_new
        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = (v_new - v_prev) / s1
            i_membrane = i_cap + i_mem * area
        return v_new, v, i_membrane

    def init_v(self, model):
        v0 = _expanded_v_init(model).clone().detach().contiguous()
        model.v = v0.clone()
        model.v_prev = v0.clone()
        model._df_history_valid = torch.zeros_like(model._df_history_valid)
        model._df_history_dt = torch.full_like(model._df_history_dt, torch.nan)
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()


class _dufort_frankel(Integrator):
    r"""
    Dufort-Frankel explicit integrator for heterogeneous morphologies.

    Extends the Dufort-Frankel scheme to non-uniform cable geometries by
    computing compartment-specific axial conductances. The method maintains
    unconditional stability for the linear diffusion term. Currents proven or
    declared affine in voltage are centered exactly between the stored time
    levels; other currents are evaluated explicitly with zero Dufort
    conductance. A one-``dt`` forward-Euler step constructs the missing history
    level after initialization or a timestep change. Voltage smoothing remains
    optional.

    Parameters
    ----------
    beta : float, optional
        Exponential smoothing factor (1.0 disables smoothing). Default 1.0.
    smooth_every : int, optional
        Retained configuration metadata. Spatial smoothing is applied each
        timestep when ``beta < 1``. Default 100.
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    __constants__ = ["beta", "smoothing", "smooth_every", "imem"]
    v_vars = ["v", "v_prev", "_df_history_valid", "_df_history_dt"]
    _FUNCTIONAL_OPERATOR_KIND = "scalar_dufort_path"
    _FUNCTIONAL_STATE_NAMES = tuple(v_vars)
    _FUNCTIONAL_CRITICAL_METHODS = (
        "_functional_step",
        "_functional_tensor_step",
        "_prepare_history",
        "_step_initial",
        "_initial_current_frame",
        "_dufort_current_frame",
        "_initial_spatial",
        "_initial_voltage",
        "_history_spatial",
        "_history_voltage",
    )
    _initial_current_frame = _dufort_frankel_homogeneous._initial_current_frame
    _dufort_current_frame = _dufort_frankel_homogeneous._dufort_current_frame
    _prepare_history = staticmethod(_dufort_frankel_homogeneous._prepare_history)
    _functional_tensor_step = _dufort_frankel_homogeneous._functional_tensor_step
    _PREPARED_WORKSPACE_SCHEMA = (
        ("area", "node"),
        ("s1", "node"),
        ("s3", "node"),
        ("c_axial", "node"),
        ("c_left", "node"),
        ("c_right", "node"),
        ("ve_zero", "node"),
    )

    def __init__(self, model, mech, beta=1.0, smooth_every=100, imem=None):
        super().__init__(model, mech, imem)

        model.register_buffer("v_prev", _expanded_v_init(model).clone().detach())
        model.register_buffer(
            "_df_history_valid",
            torch.zeros((), dtype=torch.bool, device=model.v.device),
        )
        model.register_buffer(
            "_df_history_dt",
            torch.full((), torch.nan, dtype=model.v.dtype, device=model.v.device),
        )

        self.register_buffer("area", torch.tensor(0.0))
        self.register_buffer("s1", torch.tensor(0.0))
        self.register_buffer("s3", torch.tensor(0.0))

        # These coefficients will hold the heterogeneous coupling factors.
        self.register_buffer("c_axial", torch.tensor(0.0))
        self.register_buffer("c_left", torch.tensor(0.0))
        self.register_buffer("c_right", torch.tensor(0.0))
        self.register_buffer("ve_zero", torch.tensor(0.0), persistent=False)

        self.smooth_every = smooth_every
        self.beta = beta
        self.f64 = False
        self.smoothing = bool((1 - beta))

        if self.smoothing:
            self.filter = torch.nn.Conv1d(
                1, 1, 5, padding=2, bias=False, padding_mode="reflect"
            )
            self.filter.weight.data = torch.tensor(
                [0.25, 0.25, 0, 0.25, 0.25], dtype=torch.float
            ).reshape(1, 1, 5)
            for p in self.filter.parameters():
                p.requires_grad = False

    @staticmethod
    def _prepare_workspace(dt, *, dx, diam, cm, rhoa):
        """Derive the native heterogeneous two-level cable coefficients."""
        # Units are critical here. We'll work in [V, A, s, F, Ohm, cm].
        area = torch.pi * (diam / 10000.0) * (dx / 10000.0)
        cm_total = (cm / 1000.0) * area

        dx_cm = dx / 10000.0
        radii_cm = diam / 20000.0
        ra_comp = (rhoa * dx_cm) / (torch.pi * radii_cm**2)
        ra_flat = _flatten_to_solve(ra_comp, target_shape=tuple(ra_comp.shape))
        ra_padded = _reflect_pad_last(ra_flat, (1, 1))
        r_left = 0.5 * (ra_padded[:, :-2] + ra_padded[:, 1:-1])
        r_right = 0.5 * (ra_padded[:, 1:-1] + ra_padded[:, 2:])

        g_left = (1.0 / r_left).reshape(cm.shape)
        g_right = (1.0 / r_right).reshape(cm.shape)
        if cm.shape[-1] == 1:
            g_left = torch.zeros_like(g_left)
            g_right = torch.zeros_like(g_right)

        s1 = (2 * dt) / cm_total
        c_left = s1 * g_left
        c_right = s1 * g_right
        return {
            "area": area,
            "s1": s1,
            "s3": area * s1,
            "c_left": c_left,
            "c_right": c_right,
            "c_axial": 0.5 * (c_left + c_right),
            "ve_zero": torch.zeros_like(cm),
        }

    def initialize(self, model, dt) -> None:
        workspace = self._derive_prepared_workspace(
            dt,
            dx=_broadcast_to_shape(model.dx, model.shape),
            diam=_broadcast_to_shape(model.diam, model.shape),
            cm=_broadcast_to_shape(model.cm, model.shape),
            rhoa=_broadcast_to_shape(model.rhoa, model.shape),
        )
        self._install_prepared_workspace(workspace)
        self.f64 = model.dtype() == torch.float64

        self.initialized = True
        model._df_history_valid, model._df_history_dt = self._prepare_history(
            model._df_history_valid, model._df_history_dt, dt
        )

    def step(self, model, dt, ve=None, intra=None):
        if bool(torch.all(model._df_history_valid).item()):
            v_new, v_prev_new, i_membrane = self._call_kernel(
                "_step",
                model.v,
                model.v_prev,
                ve,
                dt,
                model.celsius,
                intra,
            )
        else:
            v_new, v_prev_new, i_membrane = self._call_kernel(
                "_step_initial",
                model.v,
                ve,
                dt,
                model.celsius,
                intra,
            )
        model.v = v_new
        model.v_prev = v_prev_new
        model._df_history_valid = torch.ones_like(model._df_history_valid)
        if self.imem:
            model.i_membrane = i_membrane

    def _functional_step(
        self,
        v,
        dt,
        temp,
        ve=None,
        intra=None,
        *,
        history,
        use_history,
        solver=None,
        call_local_currents=True,
        current_dispatch=None,
    ):
        """Adapt an isolated pure history phase to the native shared kernel."""
        del solver
        v_prev, valid, history_dt = history
        if current_dispatch is not None:
            v_new, v_prev_new, i_membrane = self._functional_tensor_step(
                v, v_prev, dt, temp, ve, intra, use_history, current_dispatch
            )
        elif use_history:
            v_new, v_prev_new, i_membrane = self._step(
                v,
                v_prev,
                ve,
                dt,
                temp,
                intra,
                call_local_currents=call_local_currents,
            )
        else:
            v_new, v_prev_new, i_membrane = self._step_initial(
                v,
                ve,
                dt,
                temp,
                intra,
                call_local_currents=call_local_currents,
            )
        return (
            v_new,
            v_prev_new,
            torch.ones_like(valid),
            torch.zeros_like(history_dt)
            + torch.as_tensor(dt, dtype=history_dt.dtype, device=history_dt.device),
            i_membrane,
        )

    def _initial_spatial(self, v, ve):
        K = v.shape[-1]
        v_flat = _flatten_to_solve(v, K, tuple(v.shape))
        c_left = _flatten_to_solve(self.c_left, K, tuple(v.shape))
        c_right = _flatten_to_solve(self.c_right, K, tuple(v.shape))
        v_padded = _reflect_pad_last(v_flat, (1, 1))
        delta_axial = 0.5 * c_left * (v_padded[:, :-2] - v_flat)
        delta_axial = delta_axial + 0.5 * c_right * (v_padded[:, 2:] - v_flat)
        if ve is not None:
            ve_flat = _flatten_to_solve(ve, K, tuple(v.shape))
            ve_padded = _reflect_pad_last(ve_flat, (1, 1))
            delta_axial = delta_axial + 0.5 * c_left * (ve_padded[:, :-2] - ve_flat)
            delta_axial = delta_axial + 0.5 * c_right * (ve_padded[:, 2:] - ve_flat)
        return delta_axial

    def _initial_voltage(self, v, spatial, current, intra):
        K = v.shape[-1]
        v_flat = _flatten_to_solve(v, K, tuple(v.shape))
        area = _flatten_to_solve(self.area, K, tuple(v.shape))
        s1 = _flatten_to_solve(self.s1, K, tuple(v.shape))
        i_ion_stim = _flatten_to_solve(current, K, tuple(v.shape)) * area
        if intra is not None:
            i_ion_stim = i_ion_stim - _flatten_to_solve(intra, K, tuple(v.shape))
        return (v_flat + spatial - 0.5 * s1 * i_ion_stim).reshape_as(v)

    def _history_spatial(self, v, v_prev, ve):
        K = v.shape[-1]
        v_flat = _flatten_to_solve(v, K, tuple(v.shape))
        v_prev_flat = _flatten_to_solve(v_prev, K, tuple(v.shape))
        c_left = _flatten_to_solve(self.c_left, K, tuple(v.shape))
        c_right = _flatten_to_solve(self.c_right, K, tuple(v.shape))
        c_axial = _flatten_to_solve(self.c_axial, K, tuple(v.shape))
        v_padded = _reflect_pad_last(v_flat, (1, 1))
        num_v_prev = v_prev_flat * (1 - c_axial)
        num_axial_v = c_left * v_padded[:, :-2] + c_right * v_padded[:, 2:]
        if ve is not None:
            ve_flat = _flatten_to_solve(ve, K, tuple(v.shape))
            ve_padded = _reflect_pad_last(ve_flat, (1, 1))
            num_axial_ve = c_left * (ve_padded[:, :-2] - ve_flat) + c_right * (
                ve_padded[:, 2:] - ve_flat
            )
            return num_v_prev + num_axial_v + num_axial_ve
        return num_v_prev + num_axial_v

    def _history_voltage(self, v, v_prev, spatial, current, conductance, intra):
        del v_prev
        K = v.shape[-1]
        area = _flatten_to_solve(self.area, K, tuple(v.shape))
        s1 = _flatten_to_solve(self.s1, K, tuple(v.shape))
        s3 = _flatten_to_solve(self.s3, K, tuple(v.shape))
        c_axial = _flatten_to_solve(self.c_axial, K, tuple(v.shape))
        i_ion_stim = _flatten_to_solve(current, K, tuple(v.shape)) * area
        if intra is not None:
            i_ion_stim = i_ion_stim - _flatten_to_solve(intra, K, tuple(v.shape))
        num_ion = -s1 * i_ion_stim
        numerator = spatial + num_ion
        denominator = (
            1 + c_axial + (0.5 * s3 * _flatten_to_solve(conductance, K, tuple(v.shape)))
        )
        return (numerator / denominator).reshape_as(v)

    def _step_initial(self, v, ve, dt, temp, intra=None, *, call_local_currents=False):
        """Construct the missing previous time level with one Euler step."""
        spatial = self._initial_spatial(v, ve)
        self._advance_pre_current(v, dt, temp)
        i_mem, accepted_frame = self._initial_current_frame(
            v, call_local_currents=call_local_currents
        )
        v_new = self._initial_voltage(v, spatial, i_mem, intra)
        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = 2.0 * (v_new - v) / self.s1
            i_membrane = i_cap + i_mem * self.area
        return v_new, v, i_membrane

    def _step(
        self,
        v,
        v_prev,
        ve,
        dt,
        temp,
        intra=None,
        *,
        call_local_currents=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        spatial = self._history_spatial(v, v_prev, ve)
        self._advance_pre_current(v, dt, temp)
        i_ion, gtot, i_frame, g_frame = self._dufort_current_frame(
            v, v_prev, call_local_currents=call_local_currents
        )
        v_new = self._history_voltage(v, v_prev, spatial, i_ion, gtot, intra)
        accepted_frame = self._linearize_ion_current_frame(
            i_frame, g_frame, v_new, conductance_scale=0.5
        )
        i_mem = i_ion + 0.5 * gtot * v_new
        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )
        self._advance_post_current(v, dt, temp, accepted_frame)
        i_membrane = None
        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            i_membrane = i_cap + i_mem * self.area
        return v_new, v, i_membrane

    # The rest of the class methods (init_v, detach) do not need changes.
    def init_v(self, model):
        v0 = _expanded_v_init(model).clone().detach().contiguous()
        model.v = v0.clone()
        model.v_prev = v0.clone()
        model._df_history_valid = torch.zeros_like(model._df_history_valid)
        model._df_history_dt = torch.full_like(model._df_history_dt, torch.nan)
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()
