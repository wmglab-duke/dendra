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
        return F.pad(x, (left, right), mode="replicate")

    positions = torch.arange(-left, size + right, device=x.device, dtype=torch.long)
    period = 2 * (size - 1)
    positions = torch.remainder(positions, period)
    indices = torch.where(positions < size, positions, period - positions)
    return x.index_select(-1, indices)


def _conv1d_forward(
    conv: torch.nn.Conv1d, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Run Conv1d with robust reflection padding for short final axes."""
    padding = tuple(int(value) for value in conv._reversed_padding_repeated_twice)
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
    xf = _flatten_to_solve(x).unsqueeze(1)
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

    def initialize(self, model, dt) -> None:
        dx = _broadcast_to_shape(model.dx, model.shape) / 10000.0  # cm
        diam = _broadcast_to_shape(model.diam, model.shape)
        area = torch.pi * (diam / 10000.0) * dx  # cm^2
        cm = _broadcast_to_shape(model.cm, model.shape) / 1000.0 * area
        ra = (_broadcast_to_shape(model.rhoa, model.shape) * dx) / (
            torch.pi * (diam / 20000.0) ** 2
        )
        self.cm_inv = 1.0 / cm
        self.ra_inv = 1.0 / ra
        self.cm_c = cm
        self.area_c = area
        self.ve_zero = torch.zeros_like(model.v)

    def FRK(self, v, ve, area, cm, ra, intra=None):
        d2v = _conv_last(self.ssd, v, ve)
        if intra is None:
            i_ion = self.mech.iexp(v) * area
            return cm * ((ra * d2v) - i_ion), i_ion
        else:
            intra = _broadcast_to_shape(intra, tuple(v.shape))
            i_ion = self.mech.iexp(v) * area - intra
            return cm * ((ra * d2v) - i_ion), i_ion

    def step(self, model, dt, ve=None, intra=None):
        if ve is None:
            ve = self.ve_zero
        v_new, i_membrane = self._call_kernel(
            "_step", model.v, ve, self.area_c, dt, model.celsius, self.cm_c, intra
        )
        model.v = v_new
        if self.imem:
            model.i_membrane = i_membrane

    def _step(self, v, ve, area, dt, temp, cm, intra):
        self.mech.advance(v, dt, temp)
        K1, i_ion = self.FRK(v, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + K1 * dt
        i_membrane = None
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            i_membrane = i_cap + i_ion
        return v_n, i_membrane


class _eulerv1(_euler):
    r"""
    Forward Euler integrator with a first-order Dufort-Frankel-style correction.

    Uses the same explicit stencil as :class:`_euler` but applies the FRK update
    to the current voltage rather than the stored inverse capacitance/axial
    resistances, providing a slightly more stable first-order method for stiff
    cable dynamics.

    Parameters
    ----------
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    def _step(self, v, ve, area, dt, temp, cm, intra=None):
        self.mech.advance(v, dt, temp)
        K1, i_ion = self.FRK(v, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + K1 * dt
        i_membrane = None
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            i_membrane = i_cap + i_ion
        return v_n, i_membrane


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

    def _step(self, v, ve, area, dt, temp, cm, intra=None):
        self.mech.advance(v, dt, temp)
        K1, _ = self.FRK(v, ve, area, self.cm_inv, self.ra_inv, intra)
        K2, i2 = self.FRK(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + K2 * dt
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

    def _step(self, v, ve, area, dt, temp, cm, intra=None):
        self.mech.advance(v, dt, temp)
        K1, i1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv, intra)
        K2, i2 = self.FRK(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv, intra)
        K3, i3 = self.FRK(v + K2 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv, intra)
        K4, i4 = self.FRK(v + K3 * dt, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6.0
        i_membrane = None
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            i_rk4 = (i1 + 2 * i2 + 2 * i3 + i4) / 6.0
            i_membrane = i_cap + i_rk4
        return v_n, i_membrane


def ssd_df(v_c, v_p, v_e):
    K = v_c.shape[-1]
    vc = _flatten_to_solve(v_c, K)
    vp = _flatten_to_solve(v_p, K)
    ve = _flatten_to_solve(v_e, K, tuple(v_c.shape))
    vc_p = _reflect_pad_last(vc, (1, 1))
    ve_p = _reflect_pad_last(ve, (1, 1))
    ret = vc_p[:, :-2] + vc_p[:, 2:] - vp + ve_p[:, 2:] + ve_p[:, :-2] - 2 * ve
    return ret.reshape_as(v_c)


def ssd_df_no_ve(v_c, v_p):
    K = v_c.shape[-1]
    vc = _flatten_to_solve(v_c, K)
    vp = _flatten_to_solve(v_p, K)
    vc_p = _reflect_pad_last(vc, (1, 1))
    return (vc_p[:, :-2] + vc_p[:, 2:] - vp).reshape_as(v_c)


def ssd_df_heterogeneous(v_c, v_p, v_e, g_left, g_right):
    """Calculates the spatial second derivative along the final axis."""
    K = v_c.shape[-1]
    vc = _flatten_to_solve(v_c, K)
    vp = _flatten_to_solve(v_p, K)
    ve = _flatten_to_solve(v_e, K, tuple(v_c.shape))
    gl = _flatten_to_solve(g_left, K)
    gr = _flatten_to_solve(g_right, K)
    vc_p = _reflect_pad_last(vc, (1, 1))
    ve_p = _reflect_pad_last(ve, (1, 1))
    g_sum = gl + gr
    d2v_c = gl * vc_p[:, :-2] + gr * vc_p[:, 2:] - g_sum * vp
    d2v_e = gl * ve_p[:, :-2] + gr * ve_p[:, 2:] - g_sum * ve
    return (d2v_c + d2v_e).reshape_as(v_c)


def ssd_df_heterogeneous_no_ve(v_c, v_p, g_left, g_right):
    """Calculates the spatial second derivative along the final axis without ve."""
    K = v_c.shape[-1]
    vc = _flatten_to_solve(v_c, K)
    vp = _flatten_to_solve(v_p, K)
    gl = _flatten_to_solve(g_left, K)
    gr = _flatten_to_solve(g_right, K)
    vc_p = _reflect_pad_last(vc, (1, 1))
    g_sum = gl + gr
    return (gl * vc_p[:, :-2] + gr * vc_p[:, 2:] - g_sum * vp).reshape_as(v_c)


class _dufort_frankel_homogeneous(Integrator):
    r"""
    Dufort-Frankel explicit integrator for homogeneous morphologies.

    Uses the Dufort-Frankel scheme on the cable diffusion term to achieve
    unconditional stability for the linear part while keeping ionic currents
    explicit. Optionally applies periodic spatial smoothing. Can compute with
    either convolution-based or direct finite-difference stencils.

    Parameters
    ----------
    beta : float, optional
        Exponential smoothing factor (1.0 disables smoothing). Default 1.0.
    smooth_every : int, optional
        Apply smoothing every ``smooth_every`` steps when ``beta < 1``.
        Default 100.
    conv : bool, optional
        If True, use a fixed convolution kernel for the spatial stencil;
        otherwise use direct finite differences. Default False.
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    __constants__ = ["beta", "smoothing", "smooth_every", "imem"]
    v_vars = ["v", "v_prev"]

    def __init__(self, model, mech, beta=1.0, smooth_every=100, conv=False, imem=None):
        super().__init__(model, mech, imem)

        model.register_buffer("v_prev", _expanded_v_init(model).clone().detach())

        self.register_buffer("area", torch.tensor(0.0))
        self.register_buffer("s1", torch.tensor(0.0))
        self.register_buffer("s2", torch.tensor(0.0))
        self.register_buffer("s3", torch.tensor(0.0))
        self.register_buffer("s4", torch.tensor(0.0))

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

    def initialize(self, model, dt) -> None:
        diam = _broadcast_to_shape(model.diam, model.shape)
        dx_raw = _broadcast_to_shape(model.dx, model.shape)
        rhoa = _broadcast_to_shape(model.rhoa, model.shape)
        cm_param = _broadcast_to_shape(model.cm, model.shape)
        self.area = torch.pi * (diam / 10000) * (dx_raw / 10000)
        cm = (cm_param / 1e3) * self.area
        dx = dx_raw / 10000
        radii = diam / 20000
        ra = (rhoa * dx) / (torch.pi * radii**2)
        self.s1 = 2 * dt / cm
        self.s2 = self.s1 / ra
        self.s3 = self.area * self.s1
        self.s4 = 1 + self.s2
        self.f64 = model.dtype() == torch.float64
        self.ve_zero = torch.zeros_like(model.v)
        if self.conv:
            self.method = self._step_conv
        else:
            self.method = self._step

    def step(self, model, dt, ve=None, intra=None):
        if ve is None:
            if self.conv:
                ve = self.ve_zero
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
        model.v = v_new
        model.v_prev = v_prev_new
        if self.imem:
            model.i_membrane = i_membrane

    def _step(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, intra=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ve is None:
            d2v = s2 * ssd_df_no_ve(v, v_prev)
        else:
            d2v = s2 * ssd_df(v, v_prev, ve)

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)

        if intra is not None:
            i_ion = i_ion * area - _broadcast_to_shape(intra, tuple(v.shape))
        else:
            i_ion = i_ion * area

        v_new = (v_prev + d2v - s1 * i_ion) / (s4 + 0.5 * gtot * s3)

        i_mem = self.mech.itot(v)

        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )

        i_membrane = None
        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            if intra is not None:
                i_membrane = (
                    i_cap + i_mem * area - _broadcast_to_shape(intra, tuple(v.shape))
                )
            else:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d2v = s2 * _conv_last(self.ssd, v, v_prev, ve)

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)

        if intra is not None:
            i_ion = i_ion * area - _broadcast_to_shape(intra, tuple(v.shape))
        else:
            i_ion = i_ion * area

        v_new = (v_prev + d2v - s1 * i_ion) / (s4 + 0.5 * gtot * s3)

        i_mem = self.mech.itot(v)

        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )

        i_membrane = None
        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            if intra is not None:
                i_membrane = (
                    i_cap + i_mem * area - _broadcast_to_shape(intra, tuple(v.shape))
                )
            else:
                i_membrane = i_cap + i_mem * area

        return v_new, v, i_membrane

    def init_v(self, model):
        v0 = _expanded_v_init(model).clone().detach().contiguous()
        model.v = v0.clone()
        model.v_prev = v0.clone()
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()


class _dufort_frankel(Integrator):
    r"""
    Dufort-Frankel explicit integrator for heterogeneous morphologies.

    Extends the Dufort-Frankel scheme to non-uniform cable geometries by
    computing compartment-specific axial conductances. The method maintains
    unconditional stability for the linear diffusion term while treating ionic
    currents explicitly and optionally smoothing voltages.

    Parameters
    ----------
    beta : float, optional
        Exponential smoothing factor (1.0 disables smoothing). Default 1.0.
    smooth_every : int, optional
        Apply smoothing every ``smooth_every`` steps when ``beta < 1``.
        Default 100.
    imem : bool or None, optional
        If truthy, accumulate membrane currents each step. Default None.
    """

    __constants__ = ["beta", "smoothing", "smooth_every", "imem"]
    v_vars = ["v", "v_prev"]

    def __init__(self, model, mech, beta=1.0, smooth_every=100, imem=None):
        super().__init__(model, mech, imem)

        model.register_buffer("v_prev", _expanded_v_init(model).clone().detach())

        self.register_buffer("area", torch.tensor(0.0))
        self.register_buffer("s1", torch.tensor(0.0))
        self.register_buffer("s3", torch.tensor(0.0))

        # These coefficients will hold the heterogeneous coupling factors.
        self.register_buffer("c_axial", torch.tensor(0.0))
        self.register_buffer("c_left", torch.tensor(0.0))
        self.register_buffer("c_right", torch.tensor(0.0))

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

    def initialize(self, model, dt) -> None:
        # Units are critical here. We'll work in [V, A, s, F, Ohm, cm].
        diam = _broadcast_to_shape(model.diam, model.shape)
        dx = _broadcast_to_shape(model.dx, model.shape)
        cm_param = _broadcast_to_shape(model.cm, model.shape)
        rhoa = _broadcast_to_shape(model.rhoa, model.shape)

        self.area = torch.pi * (diam / 10000.0) * (dx / 10000.0)
        cm_total = (cm_param / 1000.0) * self.area

        dx_cm = dx / 10000.0
        radii_cm = diam / 20000.0
        ra_comp = (rhoa * dx_cm) / (torch.pi * radii_cm**2)
        ra_flat = _flatten_to_solve(ra_comp)
        ra_padded = F.pad(ra_flat, (1, 1), "reflect")
        r_left = 0.5 * (ra_padded[:, :-2] + ra_padded[:, 1:-1])
        r_right = 0.5 * (ra_padded[:, 1:-1] + ra_padded[:, 2:])

        g_left = (1.0 / r_left).reshape(model.shape)
        g_right = (1.0 / r_right).reshape(model.shape)

        self.s1 = (2 * dt) / cm_total
        self.s3 = self.area * self.s1
        self.c_left = self.s1 * g_left
        self.c_right = self.s1 * g_right
        self.c_axial = 0.5 * (self.c_left + self.c_right)

        self.f64 = model.dtype() == torch.float64
        self.ve_zero = torch.zeros_like(model.v)

        self.initialized = True

    def step(self, model, dt, ve=None, intra=None):
        # The step logic is simplified as we no longer branch on `conv`
        v_new, v_prev_new, i_membrane = self._call_kernel(
            "_step",
            model.v,
            model.v_prev,
            ve,
            dt,
            model.celsius,
            intra,
        )
        model.v = v_new
        model.v_prev = v_prev_new
        if self.imem:
            model.i_membrane = i_membrane

    def _step(
        self,
        v,
        v_prev,
        ve,
        dt,
        temp,
        intra=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # The stimulus current `intra` is added to the numerator.
        K = v.shape[-1]
        v_flat = _flatten_to_solve(v, K)
        v_prev_flat = _flatten_to_solve(v_prev, K)
        c_left = _flatten_to_solve(self.c_left, K)
        c_right = _flatten_to_solve(self.c_right, K)
        c_axial = _flatten_to_solve(self.c_axial, K)

        v_padded = F.pad(v_flat, (1, 1), "reflect")
        num_v_prev = v_prev_flat * (1 - c_axial)
        num_axial_v = c_left * v_padded[:, :-2] + c_right * v_padded[:, 2:]

        if ve is not None:
            ve_flat = _flatten_to_solve(ve, K, tuple(v.shape))
            ve_padded = F.pad(ve_flat, (1, 1), "reflect")
            num_axial_ve = c_left * (ve_padded[:, :-2] - ve_flat) + c_right * (
                ve_padded[:, 2:] - ve_flat
            )
            num = num_v_prev + num_axial_v + num_axial_ve
        else:
            num = num_v_prev + num_axial_v

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)
        area = _flatten_to_solve(self.area, K)
        s1 = _flatten_to_solve(self.s1, K)
        s3 = _flatten_to_solve(self.s3, K)
        i_ion_stim = _flatten_to_solve(i_ion, K) * area
        if intra is not None:
            i_ion_stim = i_ion_stim - _flatten_to_solve(intra, K, tuple(v.shape))
        num_ion = -s1 * i_ion_stim

        numerator = num + num_ion
        denominator = 1 + c_axial + (0.5 * s3 * _flatten_to_solve(gtot, K))
        v_new = (numerator / denominator).reshape_as(v)

        i_mem = self.mech.itot(v)

        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * _filter_last(
                self.filter, v_new
            )

        i_membrane = None
        if self.imem:
            i_cap = (_flatten_to_solve(v_new, K) - v_prev_flat) / s1
            i_membrane = i_cap + _flatten_to_solve(i_mem, K) * area
            if intra is not None:
                i_membrane = i_membrane - _flatten_to_solve(intra, K, tuple(v.shape))
            i_membrane = i_membrane.reshape_as(v)

        return v_new, v, i_membrane

    # The rest of the class methods (init_v, detach) do not need changes.
    def init_v(self, model):
        v0 = _expanded_v_init(model).clone().detach().contiguous()
        model.v = v0.clone()
        model.v_prev = v0.clone()
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()
