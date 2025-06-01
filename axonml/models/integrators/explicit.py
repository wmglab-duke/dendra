from typing import Tuple

import torch
from torch.nn import functional as F

from axonml.models.mechanisms.compilers import (
    MechCompiler,
    DF_Compiler,
)
from axonml.helpers import IMEM

from .core import Integrator


@torch.jit.interface
class HandlerInterface:
    def initialize(self, v, v_init, temp) -> None:
        pass

    def advance(self, v, dt, temp) -> None:
        pass

    def detach(self) -> None:
        pass

    def i(self, v, v_prev) -> torch.Tensor:
        pass

    def itot(self, v) -> torch.Tensor:
        pass

    def gtot(self, v) -> torch.Tensor:
        pass


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return self._conv_forward(x, weight_, self.bias)


class _euler(Integrator):
    """
    Euler integrator.
    """

    compiler = MechCompiler
    builder = None
    is_df = False

    def __init__(self, model, mech, imem=None):
        super().__init__(model, mech, imem)

        self.register_buffer("cm_inv", torch.tensor(0.0))
        self.register_buffer("ra_inv", torch.tensor(0.0))

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
        self.cm_inv = 1.0 / model.cm_c
        self.ra_inv = 1.0 / model.ra_c

    def FRK(self, v, ve, area, cm, ra):
        x = torch.stack([v, ve], dim=1)
        d2v = self.ssd(x).squeeze(1)
        i_ion = self.mech.i(v, v) * area
        return cm * ((ra * d2v) - i_ion)

    def FRK_intra(self, v, ve, area, cm, ra, intra):
        x = torch.stack([v, ve], dim=1)
        d2v = self.ssd(x).squeeze(1)
        i_ion = self.mech.i(v, v) * area - intra
        return cm * ((ra * d2v) - i_ion)

    def step(self, model, ve, dt, t_ind):
        model.v = self._step_no_intra(
            model.v, ve, model.area_c, dt, model.temp_c, model.cm_c
        )

    def step_intra(self, model, ve, intra, dt, t_ind):
        model.v = self._step_intra(
            model.v, ve, model.area_c, dt, model.temp_c, model.cm_c, intra
        )

    def _step_no_intra(self, v, ve, area, dt, temp, cm):
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        self.mech.advance(v, dt, temp)
        v_n = v + K1 * dt
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n

    def _step_intra(self, v, ve, area, dt, temp, cm, intra):
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        self.mech.advance(v, dt, temp)
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
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        self.mech.advance(v, dt, temp)
        K2 = self.FRK(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + K2 * dt
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n

    def _step_intra(self, v, ve, area, dt, temp, cm, intra):
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        self.mech.advance(v, dt, temp)
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
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        self.mech.advance(v, dt, temp)
        K2 = self.FRK(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K3 = self.FRK(v + K2 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K4 = self.FRK(v + K3 * dt, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6.0
        if self.imem:
            i_cap = cm * (v_n - v) / dt
            self.i_membrane = i_cap + self.mech.imem * area
        return v_n

    def _step_intra(self, v, ve, area, dt, temp, cm, intra):
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        self.mech.advance(v, dt, temp)
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

    ret = (
        v_c_p[:, :-2]
        + v_c_p[:, 2:]
        - v_p
        + v_e_p[:, 2:]
        + v_e_p[:, :-2]
        - 2 * v_e
    )

    return ret


class _dufort_frankel(Integrator):
    """
    Dufort-Frankel integrator.
    """

    compiler = DF_Compiler
    builder = None
    is_df = True

    __constants__ = ["beta", "smoothing", "smooth_every", "imem"]

    def __init__(self, model, mech, beta=1.0, smooth_every=100, conv=False, imem=None):
        super().__init__(model, mech, imem)

        model.register_buffer(
            "v_prev", torch.full((model.n_ax, model.n_comp), model.v_init)
        )

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
        self.s1 = 2 * dt / model.cm_c
        self.s2 = self.s1 / model.ra_c
        self.s3 = model.area_c * self.s1
        self.s4 = 1 + self.s2
        self.f64 = model.dtype() == torch.float64
        if self.f64:
            if self.conv:
                self.method_intra = self._step_intra_64_conv
                self.method_no_intra = self._step_no_intra_64_conv
            else:
                self.method_intra = self._step_intra_64
                self.method_no_intra = self._step_no_intra_64
        else:
            if self.conv:
                self.method_intra = self._step_intra_conv
                self.method_no_intra = self._step_no_intra_conv
            else:
                self.method_intra = self._step_intra
                self.method_no_intra = self._step_no_intra

    def step(self, model, ve, dt, t_ind):
        model.v, model.v_prev = self.method_no_intra(
            model.v,
            model.v_prev,
            ve,
            self.s1,
            self.s2,
            self.s3,
            self.s4,
            model.area_c,
            dt,
            model.temp_c,
            t_ind,
        )

    def step_intra(self, model, ve, intra, dt, t_ind):
        model.v, model.v_prev = self.method_intra(
            model.v,
            model.v_prev,
            ve,
            self.s1,
            self.s2,
            self.s3,
            self.s4,
            model.area_c,
            dt,
            model.temp_c,
            intra,
            t_ind,
        )

    def _step_no_intra_64(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, t_ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d2v = ssd_df(v, v_prev, ve)

        i_ion = self.mech.i(v_prev, v)

        v_new = (v_prev + s2 * d2v - s3 * i_ion) / (s4 + s3 * self.mech.gtot(v))

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.smoothing:
            if (t_ind + 1) % self.smooth_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_no_intra_64_conv(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, t_ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([v, v_prev, ve], dim=1)
        d2v = self.ssd(x).squeeze(1)

        i_ion = self.mech.i(v_prev, v)

        v_new = (v_prev + s2 * d2v - s3 * i_ion) / (s4 + s3 * self.mech.gtot(v))

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.smoothing:
            if (t_ind + 1) % self.smooth_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_no_intra(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, t_ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d2v = ssd_df(v, v_prev, ve)

        i_ion = self.mech.i(v_prev, v)

        v_new = (v_prev + s2 * d2v - s3 * i_ion) / (s4 + s3 * self.mech.gtot(v))

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.smoothing:
            if (t_ind + 1) % self.smooth_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_no_intra_conv(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, t_ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([v, v_prev, ve], dim=1)
        d2v = self.ssd(x).squeeze(1)

        i_ion = self.mech.i(v_prev, v)

        v_new = (v_prev + s2 * d2v - s3 * i_ion) / (s4 + s3 * self.mech.gtot(v))

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.smoothing:
            if (t_ind + 1) % self.smooth_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_intra(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, intra, t_ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d2v = s2 * ssd_df(v, v_prev, ve)

        i_ion = self.mech.i(v_prev, v) * area - intra

        v_new = (v_prev + d2v - s1 * i_ion) / (s4 + self.mech.gtot(v) * s3)

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.smoothing:
            if (t_ind + 1) % self.smooth_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_intra_conv(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, intra, t_ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([v, v_prev, ve], dim=1)
        d2v = s2 * self.ssd(x).squeeze(1)

        i_ion = self.mech.i(v_prev, v) * area - intra

        v_new = (v_prev + d2v - s1 * i_ion) / (s4 + self.mech.gtot(v) * s3)

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.smoothing:
            if (t_ind + 1) % self.smooth_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_intra_64(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, intra, t_ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d2v = s2 * ssd_df(v, v_prev, ve)

        i_ion = self.mech.i(v_prev, v) * area - intra

        v_new = (v_prev + d2v - s1 * i_ion) / (s4 + self.mech.gtot(v) * s3)

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.smoothing:
            if (t_ind + 1) % self.smooth_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def _step_intra_64_conv(
        self, v, v_prev, ve, s1, s2, s3, s4, area, dt, temp, intra, t_ind: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([v, v_prev, ve], dim=1)
        d2v = s2 * self.ssd(x).squeeze(1)

        i_ion = self.mech.i(v_prev, v) * area - intra

        v_new = (v_prev + d2v - s1 * i_ion) / (s4 + self.mech.gtot(v) * s3)

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.smoothing:
            if (t_ind + 1) % self.smooth_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        if self.imem:
            i_cap = (v_new - v_prev) / self.s1
            self.i_membrane = i_cap + self.mech.imem * area

        return v_new, v

    def init_v(self, model):
        model.v[:] = model.v_init
        model.v.detach_()
        model.v_prev[:] = model.v_init
        model.v_prev.detach_()
        if self.imem:
            self.i_membrane[:] = 0.0
            self.i_membrane.detach_()

    def detach(self, model):
        model.v_prev.detach_()
        model.v.detach_()
        self.mech.detach()
        if self.imem:
            self.i_membrane.detach_()
