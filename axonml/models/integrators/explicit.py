from typing import Tuple

import torch
from torch.nn import functional as F

from .core import Integrator


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return self._conv_forward(x, weight_, self.bias)


class _euler(Integrator):
    """
    Forward Euler integrator.
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
        dx = model.dx / 10000.0  # Convert to cm
        area = torch.pi * (model.diam / 10000.0) * dx  # Convert to cm^2
        cm = model.cm / 1000.0 * area  # Convert to F/cm^2
        ra = (model.rhoa * dx) / (
            torch.pi * (model.diameters[:, None] / 20000.0) ** 2
        )  # Convert to Ohm
        self.cm_inv = 1.0 / cm
        self.ra_inv = 1.0 / ra
        self.cm_c = cm
        self.area_c = area
        self.ve_zero = torch.zeros_like(model.v)

        self.dt = dt
        self.initialized = True

    def FRK(self, v, ve, area, cm, ra):
        x = torch.stack([v, ve], dim=1)
        d2v = self.ssd(x).squeeze(1)
        i_ion = self.mech.iexp(v)
        return cm * ((ra * d2v) - i_ion * area)

    def FRK_intra(self, v, ve, area, cm, ra, intra):
        x = torch.stack([v, ve], dim=1)
        d2v = self.ssd(x).squeeze(1)
        i_ion = self.mech.iexp(v)
        return cm * ((ra * d2v) - i_ion * area + intra)

    def step(self, model, dt, ve=None, intra=None):
        if ve is None:
            ve = self.ve_zero
        if intra is None:
            model.v = self._step_no_intra(
                model.v, ve, self.area_c, dt, model.celsius, self.cm_c
            )
        else:
            model.v = self._step_intra(
                model.v, ve, self.area_c, dt, model.celsius, self.cm_c, intra
            )

    def _step_no_intra(self, v, ve, area, dt, temp, cm):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + K1 * dt
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n

    def _step_intra(self, v, ve, area, dt, temp, cm, intra):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + K1 * dt
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n


class _eulerv1(_euler):
    """
    Euler integrator with first-order correction.
    """

    def _step_no_intra(self, v, ve, area, dt, temp, cm):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + K1 * dt
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n

    def _step_intra(self, v, ve, area, dt, temp, cm, intra):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + K1 * dt
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n


_rk1 = _euler


class _rk2(_euler):
    """
    Second-order Runge-Kutta integrator.
    """

    def _step_no_intra(self, v, ve, area, dt, temp, cm):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        K2 = self.FRK(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + K2 * dt
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n

    def _step_intra(self, v, ve, area, dt, temp, cm, intra):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        K2 = self.FRK_intra(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + K2 * dt
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n


class _rk4(_euler):
    """
    Fourth-order Runge-Kutta integrator.
    """

    def _step_no_intra(self, v, ve, area, dt, temp, cm):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        K2 = self.FRK(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K3 = self.FRK(v + K2 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K4 = self.FRK(v + K3 * dt, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6.0
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n

    def _step_intra(self, v, ve, area, dt, temp, cm, intra):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        K2 = self.FRK_intra(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K3 = self.FRK_intra(v + K2 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K4 = self.FRK_intra(v + K3 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6.0
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n


def ssd_df(v_c, v_p, v_e):
    v_c_p = F.pad(v_c, (1, 1), "reflect")
    v_e_p = F.pad(v_e, (1, 1), "reflect")

    ret = v_c_p[:, :-2] + v_c_p[:, 2:] - v_p + v_e_p[:, 2:] + v_e_p[:, :-2] - 2 * v_e

    return ret


def ssd_df_no_ve(v_c, v_p):
    v_c_p = F.pad(v_c, (1, 1), "reflect")

    ret = v_c_p[:, :-2] + v_c_p[:, 2:] - v_p

    return ret


def ssd_df_heterogeneous(v_c, v_p, v_e, g_left, g_right):
    """Calculates the spatial second derivative for heterogeneous morphologies."""
    v_c_p = F.pad(v_c, (1, 1), "reflect")
    v_e_p = F.pad(v_e, (1, 1), "reflect")
    g_sum = g_left + g_right

    # Dufort-Frankel for the membrane potential term
    d2v_c = g_left * v_c_p[:, :-2] + g_right * v_c_p[:, 2:] - g_sum * v_p
    # Standard central difference for the external potential term
    d2v_e = g_left * v_e_p[:, :-2] + g_right * v_e_p[:, 2:] - g_sum * v_e
    return d2v_c + d2v_e


def ssd_df_heterogeneous_no_ve(v_c, v_p, g_left, g_right):
    """Calculates the spatial second derivative for heterogeneous morphologies without ve."""
    v_c_p = F.pad(v_c, (1, 1), "reflect")
    g_sum = g_left + g_right
    d2v_c = g_left * v_c_p[:, :-2] + g_right * v_c_p[:, 2:] - g_sum * v_p
    return d2v_c


class _dufort_frankel_homogeneous(Integrator):
    """
    Dufort-Frankel integrator.
    """

    __constants__ = ["beta", "smoothing", "smooth_every", "imem"]
    v_vars = ["v", "v_prev"]

    def __init__(self, model, mech, beta=1.0, smooth_every=100, conv=False, imem=None):
        super().__init__(model, mech, imem)

        model.register_buffer("v_prev", torch.full((model.np, model.nc), model.v_init))

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
        self.area = torch.pi * (model.diam / 10000) * (model.dx / 10000)
        cm = (model.cm / 1e3) * self.area
        dx = model.dx / 10000
        radii = model.diam / 20000
        ra = (model.rhoa * dx) / (torch.pi * radii**2)
        self.s1 = 2 * dt / cm
        self.s2 = self.s1 / ra
        self.s3 = self.area * self.s1
        self.s4 = 1 + self.s2
        self.f64 = model.dtype() == torch.float64
        self.ve_zero = torch.zeros_like(model.v)
        if self.conv:
            self.method_intra = self._step_intra_conv
            self.method_no_intra = self._step_no_intra_conv
        else:
            self.method_intra = self._step_intra
            self.method_no_intra = self._step_no_intra

    def step(self, model, dt, ve=None, intra=None):
        if ve is None:
            if self.conv:
                ve = self.ve_zero
        if intra is None:
            model.v, model.v_prev = self.method_no_intra(
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
            )
        else:
            model.v, model.v_prev = self.method_intra(
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

    def _step_no_intra(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ve is None:
            d2v = ssd_df_no_ve(v, v_prev)
        else:
            d2v = ssd_df(v, v_prev, ve)

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)
        v_new = (v_prev + s2 * d2v - s3 * i_ion) / (s4 + s3 * 0.5 * gtot)

        self.mech.itot(v)

        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_no_intra_conv(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([v, v_prev, ve], dim=1)
        d2v = self.ssd(x).squeeze(1)

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)
        v_new = (v_prev + s2 * d2v - s3 * i_ion) / (s4 + s3 * 0.5 * gtot)

        self.mech.itot(v)

        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_intra(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, intra
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ve is None:
            d2v = s2 * ssd_df_no_ve(v, v_prev)
        else:
            d2v = s2 * ssd_df(v, v_prev, ve)

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)
        i_ion = i_ion * area - intra

        v_new = (v_prev + d2v - s1 * i_ion) / (s4 + 0.5 * gtot * s3)

        self.mech.itot(v)

        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_intra_conv(
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
        intra,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([v, v_prev, ve], dim=1)
        d2v = s2 * self.ssd(x).squeeze(1)

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)
        i_ion = i_ion * area - intra

        v_new = (v_prev + d2v - s1 * i_ion) / (s4 + 0.5 * gtot * s3)

        self.mech.itot(v)

        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def init_v(self, model):
        model.v = torch.full(
            model.v.shape, model.v_init, dtype=model.v.dtype, device=model.v.device
        ).detach()
        model.v_prev = torch.full(
            model.v_prev.shape,
            model.v_init,
            dtype=model.v_prev.dtype,
            device=model.v_prev.device,
        ).detach()
        if self.imem:
            model.i_membrane = torch.zeros(
                model.i_membrane.shape,
                dtype=model.i_membrane.dtype,
                device=model.i_membrane.device,
            ).detach()


class _dufort_frankel(Integrator):
    """
    Dufort-Frankel integrator for heterogeneous morphologies. (Corrected)
    """

    __constants__ = ["beta", "smoothing", "smooth_every", "imem"]
    v_vars = ["v", "v_prev"]

    def __init__(self, model, mech, beta=1.0, smooth_every=100, imem=None):
        super().__init__(model, mech, imem)

        model.register_buffer("v_prev", torch.full((model.np, model.nc), model.v_init))

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

        # Total area of each compartment [cm^2]
        self.area = torch.pi * (model.diam / 10000.0) * (model.dx / 10000.0)
        # Total capacitance of each compartment [F]
        cm_total = (model.cm / 1000.0) * self.area  # cm is in uF/cm^2 -> F/cm^2 -> F

        # Axial resistance of a FULL compartment [Ohm]
        dx_cm = model.dx / 10000.0
        radii_cm = model.diam / 20000.0
        # rhoa [Ohm-cm], dx [cm], radii [cm] -> ra [Ohm]
        ra_comp = (model.rhoa * dx_cm) / (torch.pi * radii_cm**2)

        # Resistance between compartment centers [Ohm]
        # This is the average of two half-compartment resistances.
        ra_padded = F.pad(ra_comp, (1, 1), "reflect")
        r_left = 0.5 * (ra_padded[:, :-2] + ra_padded[:, 1:-1])
        r_right = 0.5 * (ra_padded[:, 1:-1] + ra_padded[:, 2:])

        # Conductance between compartment centers [S, or 1/Ohm]
        g_left = 1.0 / r_left
        g_right = 1.0 / r_right

        # --- Define the core Dufort-Frankel coefficients ---
        # s1 is a coefficient used for scaling currents.
        self.s1 = (2 * dt) / cm_total
        self.s3 = self.area * self.s1  # Scales current density [uA/cm^2]

        # C_left/C_right are the coupling terms for neighboring voltages.
        self.c_left = self.s1 * g_left
        self.c_right = self.s1 * g_right
        # c_axial is the total axial coupling for the implicit terms.
        self.c_axial = 0.5 * (self.c_left + self.c_right)

        self.f64 = model.dtype() == torch.float64
        self.ve_zero = torch.zeros_like(model.v)

        self.initialized = True

    def step(self, model, dt, ve=None, intra=None):
        # The step logic is simplified as we no longer branch on `conv`
        if intra is None:
            model.v, model.v_prev = self._step_no_intra(
                model.v,
                model.v_prev,
                ve,
                dt,
                model.celsius,
            )
        else:
            model.v, model.v_prev = self._step_intra(
                model.v,
                model.v_prev,
                ve,
                dt,
                model.celsius,
                intra,
            )

    def _step_no_intra(
        self,
        v,
        v_prev,
        ve,
        dt,
        temp,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_padded = F.pad(v, (1, 1), "reflect")

        # --- Numerator Calculation ---
        # Contribution from previous voltage (v_prev)
        num_v_prev = v_prev * (1 - self.c_axial)

        # Contribution from neighboring voltages (v)
        num_axial_v = self.c_left * v_padded[:, :-2] + self.c_right * v_padded[:, 2:]

        # Contribution from external potential (ve) - this is a standard central difference
        if ve is not None:
            ve_padded = F.pad(ve, (1, 1), "reflect")
            num_axial_ve = self.c_left * (ve_padded[:, :-2] - ve) + self.c_right * (
                ve_padded[:, 2:] - ve
            )
            num = num_v_prev + num_axial_v + num_axial_ve
        else:
            num = num_v_prev + num_axial_v

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)
        num_ion = -self.s3 * i_ion

        numerator = num + num_ion

        # --- Denominator Calculation ---
        # Implicit contribution from axial current and total ionic conductance
        denominator = 1 + self.c_axial + (0.5 * self.s3 * gtot)

        v_new = numerator / denominator

        self.mech.itot(v)

        if self.smoothing:
            # unsqueeze/squeeze needed for Conv1d which expects (N, C, L)
            v_new = self.beta * v_new + (1 - self.beta) * self.filter(
                v_new.unsqueeze(1)
            ).squeeze(1)

        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * self.area

        return v_new, v

    def _step_intra(
        self,
        v,
        v_prev,
        ve,
        dt,
        temp,
        intra,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # This method is identical to _step_no_intra, with one change:
        # The stimulus current `intra` is added to the numerator.
        v_padded = F.pad(v, (1, 1), "reflect")

        num_v_prev = v_prev * (1 - self.c_axial)
        num_axial_v = self.c_left * v_padded[:, :-2] + self.c_right * v_padded[:, 2:]

        if ve is not None:
            ve_padded = F.pad(ve, (1, 1), "reflect")
            num_axial_ve = self.c_left * (ve_padded[:, :-2] - ve) + self.c_right * (
                ve_padded[:, 2:] - ve
            )
            num = num_v_prev + num_axial_v + num_axial_ve
        else:
            num = num_v_prev + num_axial_v

        self.mech.advance(v, dt, temp)
        i_ion, gtot = self.mech.idf(v, v_prev)
        i_ion_stim = i_ion * self.area - intra
        num_ion = (
            -self.s1 * i_ion_stim
        )  # Note: we use s1 here since i_ion_stim is in Amps

        numerator = num + num_ion

        denominator = 1 + self.c_axial + (0.5 * self.s3 * gtot)

        v_new = numerator / denominator

        self.mech.itot(v)

        if self.smoothing:
            v_new = self.beta * v_new + (1 - self.beta) * self.filter(
                v_new.unsqueeze(1)
            ).squeeze(1)

        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * self.area

        return v_new, v

    # The rest of the class methods (init_v, detach) do not need changes.
    def init_v(self, model):
        model.v = torch.full(
            model.v.shape, model.v_init, dtype=model.v.dtype, device=model.v.device
        )
        model.v.detach_()
        model.v_prev = torch.full(
            model.v_prev.shape,
            model.v_init,
            dtype=model.v_prev.dtype,
            device=model.v_prev.device,
        )
        model.v_prev.detach_()
        if self.imem:
            model.i_membrane = torch.zeros(
                model.i_membrane.shape,
                dtype=model.i_membrane.dtype,
                device=model.i_membrane.device,
            )
            model.i_membrane.detach_()
