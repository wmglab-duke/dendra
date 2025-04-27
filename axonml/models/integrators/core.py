from functools import partial, partialmethod
from typing import Any, Type, Tuple, Callable

import torch

from axonml.models.mechanisms.compilers import (
    MechCompiler, DF_Compiler, ImplicitCompiler
)
from axonml.helpers import IMEM
from axonml.models.interfaces import AxonInterface


def partial_class(cls: Type[Any], /, *args, **kwargs) -> Type[Any]:
    """
    Return a subclass of *cls* whose __init__ is pre-filled with *args/kwargs*.
    Because it is a real subclass, all class attributes, methods,
    and isinstance checks continue to behave as expected.
    """
    class _Partial(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    _Partial.__name__ = f"{cls.__name__}Partial"
    _Partial.__qualname__ = _Partial.__name__
    return _Partial


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


class Integrator(torch.jit.ScriptModule):
    """
    Base class for all integrators.
    """

    __constants__ = {"imem"}

    def __init__(self, model, mech, imem=None):
        super().__init__()
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        model.register_buffer("v", torch.full((model.n_ax, 1, model.n_comp), model.v_init))
        if self.imem:
            model.register_buffer("i_membrane", torch.zeros((model.n_ax, 1, model.n_comp)))
        self.mech = mech

    def init_v(self, model):
        model.v[:] = model.v_init
        model.v.detach_()
        if self.imem:
            model.i_membrane[:] = 0.0
            model.i_membrane.detach_()

    def detach(self, model):
        model.v.detach_()
        if self.imem:
            model.i_membrane.detach_()
        self.mech.detach()
    

class _euler(Integrator):
    """
    Euler integrator.
    """

    compiler = MechCompiler
    is_df = False

    def __init__(
            self,
            model,
            mech,
            imem=None
    ):
        super().__init__(model, mech, imem)

        self.register_buffer("cm_inv", torch.tensor(0.0))
        self.register_buffer("ra_inv", torch.tensor(0.0))

        weight = [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]]
        nc = 2
        nw = 3

        self.ssd = SymmetricConv1D(
            nc, 1, nw, bias=False, padding='same', padding_mode='reflect'
        )
        self.ssd.weight.data = torch.tensor(weight).reshape(1, nc, nw)
        for p in self.ssd.parameters():
            p.requires_grad = False

    def initialize(self, model, dt) -> None:
        self.cm_inv = 1.0 / model.cm_c
        self.ra_inv = 1.0 / model.ra_c

    @torch.jit.script_method
    def FRK(self, v, ve, area, cm, ra):
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)
        i_ion = self.mech.i(v, v) * area
        return cm * ((ra * d2v) - i_ion)

    @torch.jit.script_method
    def FRK_intra(self, v, ve, area, cm, ra, intra):
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)
        i_ion = self.mech.i(v, v) * area - intra
        return cm * ((ra * d2v) - i_ion)
    
    def step(self, model, ve, dt, t_ind):
        v = model.v
        area = model.area_c
        temp = model.temp_c
        model.v = self._step_no_intra(v, ve, area, dt, temp)

    def step_intra(self, model, ve, intra, dt, t_ind):
        v = model.v
        area = model.area_c
        temp = model.temp_c
        model.v = self._step_intra(v, ve, area, dt, temp, intra)

    @torch.jit.script_method
    def _step_no_intra(self, v, ve, area, dt, temp):
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        self.mech.advance(v, dt, temp)
        v_n = v + K1 * dt
        return v_n
    
    @torch.jit.script_method
    def _step_no_intra_imem(self, v, ve, area, dt, temp, cm) -> Tuple[torch.Tensor, torch.Tensor]:
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        self.mech.advance(v, dt, temp)
        v_n = v + K1 * dt
        i_cap = cm * (v_n - v) / dt
        i_membrane = i_cap + self.mech.imem * area
        return v_n, i_membrane

    @torch.jit.script_method
    def _step_intra(self, v, ve, area, dt, temp, intra):
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        self.mech.advance(v, dt, temp)
        v_n = v + K1 * dt
        return v_n
    
    @torch.jit.script_method
    def _step_intra_imem(self, v, ve, area, dt, temp, cm, intra) -> Tuple[torch.Tensor, torch.Tensor]:    
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        self.mech.advance(v, dt, temp)
        v_n = v + K1 * dt
        i_cap = cm * (v_n - v) / dt
        i_membrane = i_cap + self.mech.imem * area
        return v_n, i_membrane


class _eulerv1(_euler):
    """
    Euler integrator with first-order correction.
    """

    @torch.jit.script_method
    def _step_no_intra(self, v, ve, area, dt, temp):
        self.mech.advance(v, dt, temp)
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + K1 * dt
        return v_n
    
    @torch.jit.script_method
    def step_no_intra_imem(self, v, ve, area, dt, temp, cm) -> Tuple[torch.Tensor, torch.Tensor]:
        self.mech.advance(v, dt, temp)
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + K1 * dt
        i_cap = cm * (v_n - v) / dt
        i_membrane = i_cap + self.mech.imem * area
        return v_n, i_membrane

    @torch.jit.script_method
    def _step_intra(self, v, ve, area, dt, temp, intra):
        self.mech.advance(v, dt, temp)        
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + K1 * dt
        return v_n
    
    @torch.jit.script_method
    def step_intra_imem(self, v, ve, area, dt, temp, cm, intra) -> Tuple[torch.Tensor, torch.Tensor]:    
        self.mech.advance(v, dt, temp)
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + K1 * dt
        i_cap = cm * (v_n - v) / dt
        i_membrane = i_cap + self.mech.imem * area
        return v_n, i_membrane


_rk1 = _euler


class _rk2(_euler):
    """
    Second-order Runge-Kutta integrator.
    """

    @torch.jit.script_method
    def _step_no_intra(self, v, ve, area, dt, temp) -> None:
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        self.mech.advance(v, dt, temp)
        K2 = self.FRK(v + K1 * dt, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + (K1 + K2) * dt / 2.0
        return v_n

    @torch.jit.script_method
    def _step_intra(self, v, ve, area, dt, temp, intra) -> None:
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        self.mech.advance(v, dt, temp)
        K2 = self.FRK_intra(v + K1 * dt, ve, area, self.cm_inv, self.ra_inv, intra)
        v_n = v + (K1 + K2) * dt / 2.0
        return v_n


class _rk4(_euler):
    """
    Fourth-order Runge-Kutta integrator.
    """

    @torch.jit.script_method
    def _step_no_intra(self, v, ve, area, dt, temp) -> None:
        K1 = self.FRK(v, ve, area, self.cm_inv, self.ra_inv)
        self.mech.advance(v, dt, temp)
        K2 = self.FRK(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K3 = self.FRK(v + K2 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K4 = self.FRK(v + K3 * dt, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6.0
        return v_n

    @torch.jit.script_method
    def _step_intra(self, v, ve, area, dt, temp, intra) -> None:
        K1 = self.FRK_intra(v, ve, area, self.cm_inv, self.ra_inv, intra)
        self.mech.advance(v, dt, temp)
        K2 = self.FRK_intra(v + K1 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K3 = self.FRK_intra(v + K2 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        K4 = self.FRK_intra(v + K3 * dt / 2.0, ve, area, self.cm_inv, self.ra_inv)
        v_n = v + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6.0
        return v_n
    

class _dufort_frankel(Integrator):
    """
    Dufort-Frankel integrator.
    """
    compiler = DF_Compiler
    is_df = True

    __constants__ = {"beta", "hd", "apply_smoothing_every"}

    def __init__(
            self,
            model,
            mech,
            beta=1.0,
            apply_smoothing_every=100,
            imem=None
    ):
        super().__init__(model, mech, imem)

        model.register_buffer("v_prev", torch.full((model.n_ax, 1, model.n_comp), model.v_init))

        self.register_buffer("s1", torch.tensor(0.0))
        self.register_buffer("s2", torch.tensor(0.0))

        self.apply_smoothing_every = apply_smoothing_every
        self.beta = beta
        self.hd = bool((1 - beta))  # hyper-diffusion
        if self.hd:
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

    def step(self, model, ve, dt, t_ind):
        v = model.v
        v_prev = model.v_prev
        area = model.area_c
        temp = model.temp_c
        model.v, model.v_prev = self._step_no_intra(v, v_prev, ve, area, dt, temp, t_ind)

    def step_intra(self, model, ve, intra, dt, t_ind):
        v = model.v
        v_prev = model.v_prev
        area = model.area_c
        temp = model.temp_c
        model.v, model.v_prev = self._step_intra(v, v_prev, ve, area, dt, temp, intra, t_ind)

    @torch.jit.script_method
    def _step_no_intra(self, v, v_prev, ve, area, dt, temp, t_ind:int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([v, v_prev, ve], dim=1)
        d2v = self.ssd(x)

        i_ion = self.mech.i(v, v_prev) * area

        v_new = (v_prev + self.s2 * d2v - self.s1 * i_ion) / (
            1 + self.s2 + self.s1 * self.mech.gtot(v) * area
        )

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.hd:
            if (t_ind + 1) % self.apply_smoothing_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        return v_new, v

    @torch.jit.script_method
    def _step_intra(self, v, v_prev, ve, area, dt, temp, intra, t_ind:int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([v, v_prev, ve], dim=1)
        d2v = self.ssd(x)

        i_ion = self.mech.i(v, v_prev) * area - intra

        v_new = (v_prev + self.s2 * d2v - self.s1 * i_ion) / (
            1 + self.s2 + self.s1 * self.mech.gtot(v) * area
        )

        self.mech.itot(v)
        self.mech.advance(v, dt, temp)

        if self.hd:
            if (t_ind + 1) % self.apply_smoothing_every == 0:
                v_new = self.beta * v_new + (1 - self.beta) * self.filter(v_new)

        return v_new, v

    def init_v(self, model):
        model.v[:] = model.v_init
        model.v.detach_()
        model.v_prev[:] = model.v_init
        model.v_prev.detach_()
        if self.imem:
            model.i_membrane[:] = 0.0
            model.i_membrane.detach_()

    def detach(self, model):
        model.v_prev.detach_()
        model.v.detach_()
        self.mech.detach()
        if self.imem:
            model.i_membrane.detach_()


euler = partial(partial_class, _euler)
rk1 = partial(partial_class, _rk1)
rk2 = partial(partial_class, _rk2)
rk4 = partial(partial_class, _rk4)
dufort_frankel = partial(partial_class, _dufort_frankel)
eulerv1 = partial(partial_class, _eulerv1)
