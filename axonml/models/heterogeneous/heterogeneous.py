import inspect
from typing import Tuple, List, Optional

import torch
from torch.nn import functional as F
from torch import Tensor

from tqdm.auto import tqdm

import axonml as ax
from axonml.models.stim.intrastim import IntraStim
from axonml.models.backend import Backend as A
from axonml.models.callbacks import CallbackList, Callback
from axonml.models.core import Axon
from axonml.helpers import ve_from_s_t


class Heterogeneous(Axon):
    def __init__(
        self,
        n_ax: int,
        n_comp: int,
        temp=37.0,
        v_init=-70.0,
        beta=0.0,
    ):
        diameters = torch.ones(n_ax)
        method = "dufort-frankel"
        super().__init__(diameters, n_comp, temp, v_init, method, beta)

    def _register_buffers(self, diameters):
        self.register_buffer("diam",   torch.empty(self.n_ax, 1, self.n_comp))
        self.register_buffer("node_l", torch.empty(self.n_ax, 1, self.n_comp))

        self.register_buffer("area_c", torch.empty(self.diam.shape))
        self.register_buffer("cm_c",   torch.empty(self.area_c.shape))
        self.register_buffer("ra_c",   torch.empty(self.n_ax, 1, self.n_comp - 1))

        self.register_buffer("v_init_c", torch.tensor(self.v_init))
        self.register_buffer("temp_c",   torch.tensor(self.temp))

    def x(self):
        node_l = torch.atleast_2d(self.node_l.squeeze())
        x = node_l.cumsum(dim=1) - node_l / 2
        return x - torch.sum(node_l, dim=1) / 2

    def ra_(self, diameters, node_l) -> torch.Tensor:
        radii = diameters / 20000
        inl = node_l / 10000
        c_ra = (self.rhoa * inl) / (torch.pi * (radii**2))
        return (c_ra[:, :, :-1] + c_ra[:, :, 1:]) / 2

    def area_(self, diameters, node_l) -> torch.Tensor:
        dx = node_l / 10000
        return torch.pi * (diameters / 10000) * dx

    def set_diam(self, diams):
        diams = torch.as_tensor(diams, dtype=self.dtype()).unsqueeze(1)
        self.diam[:] = diams
        self.calculate_geometric_params()

    def set_node_l(self, node_l):
        node_l = torch.as_tensor(node_l, dtype=self.dtype()).unsqueeze(1)
        self.node_l[:] = node_l
        self.calculate_geometric_params()

    def set_all(self, node_d, node_l):
        diams = torch.as_tensor(node_d, dtype=self.dtype()).unsqueeze(1)
        node_l = torch.as_tensor(node_l, dtype=self.dtype()).unsqueeze(1)
        self.diam[:] = diams
        self.node_l[:] = node_l
        self.calculate_geometric_params()

    def calculate_geometric_params(self):
        self.area_c[:] = self.area_(self.diam, self.node_l)
        self.cm_c[:] = self.cm_(self.area_c)
        self.ra_c[:] = self.ra_(self.diam, self.node_l)

    def run(
        self,
        ve: Tensor = None,
        space: Tensor = None,
        time: Tensor = None,
        dt: float = None,
        intra: Optional[IntraStim] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
        progressbar: bool = True,
        first: bool = True,
        multicontact: bool = False,
        longrunning: bool = False,
    ):
        with_intra = intra is not None
        if with_intra:
            if not isinstance(intra, IntraStim):
                raise ValueError("intra must be an instance of IntraStim")

        intra_only = False
        if ve is None and (space is None and time is None):
            if intra is None:
                raise ValueError(
                    "Either ve or ve_s and ve_t or intra must be provided."
                )
            intra_only = True
            ve_zero = torch.zeros_like(self.v)

        device = self.device()
        if ve is not None:
            ve = torch.as_tensor(ve, device=device)

        dt = dt if dt is not None else A.dt
        self.warn_about_dt(dt)
        self.dt = dt

        df = self.is_df

        method = getattr(self, f"step_no_intra_{self.method_conversion[self.method]}")
        method_intra = getattr(
            self, f"step_intra_{self.method_conversion[self.method]}"
        )

        with torch.set_grad_enabled(self.training):
            if ve is None and not intra_only:
                ve = ve_from_s_t(space, time, self.n_ax, self.device(), multicontact)

            if df:
                ve = 2 * ve

            if self.training:
                self.calculate_geometric_params()

            if (not self.initialized) or reinit:
                if "_steady_state" in self._caches:
                    self.restore("_steady_state")
                    self.t_ind = 0
                else:
                    self.init_v()
                    self.initialize(self.v, self.v_init_c, self.temp_c)
                    self.post_initialize()
                    self.t_ind = 0
                    self.initialized = True
                if with_intra:
                    intra.init(self)
            else:
                self.detach()

            if first:
                if callbacks:
                    for c in callbacks:
                        c.dt = dt

                if not isinstance(callbacks, CallbackList):
                    callbacks = CallbackList(callbacks)
                callbacks.pre_loop_hook(self)

            dt = torch.as_tensor(dt, device=device)

            rap = F.pad(self.ra_c, (1, 1), "reflect")

            if df:
                s = dt / self.cm_c
                phi_l = s / rap[:, :, :-1]
                phi_r = s / rap[:, :, 1:]
                s = 2 * s
                phi_sum = phi_l + phi_r
            else:
                cm_inv = 1 / self.cm_c
                ra_inv = 1 / rap

            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(total=ve.shape[0], desc=f"{self.t:.3f} ms")

            for i in range(ve.shape[0]):
                ve_ = ve[i] if not intra_only else ve_zero
                if df:
                    if with_intra:
                        self.v, self.v_prev = method_intra(
                            self.v,
                            self.v_prev,
                            ve_,
                            self.area_c,
                            s,
                            phi_l,
                            phi_r,
                            phi_sum,
                            dt,
                            self.temp_c,
                            intra(self.t_ind, self.v),
                        )
                    else:
                        self.v, self.v_prev = method(
                            self.v,
                            self.v_prev,
                            ve_,
                            self.area_c,
                            s,
                            phi_l,
                            phi_r,
                            phi_sum,
                            dt,
                            self.temp_c,
                        )
                else:
                    if with_intra:
                        self.v = method_intra(
                            self.v,
                            ve_,
                            self.area_c,
                            cm_inv,
                            ra_inv,
                            dt,
                            self.temp_c,
                            intra(self.t_ind, self.v),
                        )
                    else:
                        self.v = method(
                            self.v, ve_, self.area_c, cm_inv, ra_inv, dt, self.temp_c
                        )
                callbacks.post_step_hook(self)
                self.t_ind += 1

                if progressbar:
                    progressbar.update(1)
                    if self.t_ind % 100 == 0:
                        progressbar.set_description(f"{self.t:.1f} ms")

            if not longrunning:
                if progressbar:
                    progressbar.close()

    @torch.jit.script_method
    def FRK(self, v, ve, area, cm, ra) -> Tensor:
        v_p = F.pad(v, (1, 1), "reflect")
        v_e_p = F.pad(ve, (1, 1), "reflect")

        i_ion = self.mech.i(v, v) * area

        l = (v_p[:, :, :-2] - v + v_e_p[:, :, :-2] - ve) * ra[:, :, :-1]
        r = (v_p[:, :, 2:] - v + v_e_p[:, :, 2:] - ve) * ra[:, :, 1:]
        return cm * ((l + r) - i_ion)

    @torch.jit.script_method
    def FRK_intra(self, v, ve, area, cm, ra, intra) -> Tensor:
        v_p = F.pad(v, (1, 1), "reflect")
        v_e_p = F.pad(ve, (1, 1), "reflect")

        i_ion = self.mech.i(v, v) * area - intra

        l = (v_p[:, :, :-2] - v + v_e_p[:, :, :-2] - ve) * ra[:, :, :-1]
        r = (v_p[:, :, 2:] - v + v_e_p[:, :, 2:] - ve) * ra[:, :, 1:]
        return cm * ((l + r) - i_ion)

    @torch.jit.script_method
    def ssd_df(self, v_c, v_p, v_e, phi_l, phi_r):
        v_c_p = 2 * F.pad(v_c, (1, 1), "reflect")
        v_e_p = F.pad(v_e, (1, 1), "reflect")

        l = (v_c_p[:, :, :-2] - v_p + v_e_p[:, :, :-2] - v_e) * phi_l
        r = (v_c_p[:, :, 2:] - v_p + v_e_p[:, :, 2:] - v_e) * phi_r

        return l + r

    @torch.jit.script_method
    def step_no_intra_df(
        self, v, v_prev, ve, area, s, phi_l, phi_r, phi_sum, dt, temp
    ) -> Tuple[Tensor, Tensor]:
        # Calculate the ionic current
        i_ion = self.mech.i(v_prev, v) * area

        # Calculate the new voltage
        v_new = (v_prev + self.ssd_df(v, v_prev, ve, phi_l, phi_r) - s * i_ion) / (
            1 + phi_sum + s * self.mech.gtot(v) * area
        )

        # Calculate the total current
        self.mech.itot(v)

        # Advance the mech state
        self.mech.advance(v, dt, temp)

        if self.use_fast_imem:
            i_cap = self.cm_c * (v_new - v_prev) / (2 * dt)
            self.i_membrane[:] = i_cap + self.mech.imem * area

        if self.hd:
            v_new = v_new - self.beta * self.filter(v_new)

        return v_new, v

    @torch.jit.script_method
    def step_intra_df(
        self, v, v_prev, ve, area, s, phi_l, phi_r, phi_sum, dt, temp, intra
    ) -> Tuple[Tensor, Tensor]:
        # Calculate the ionic current
        i_ion = self.mech.i(v_prev, v) * area - intra

        # Calculate the new voltage
        v_new = (v_prev + self.ssd_df(v, v_prev, ve, phi_l, phi_r) - s * i_ion) / (
            1 + phi_sum + s * self.mech.gtot(v) * area
        )

        # Calculate the total current
        self.mech.itot(v)

        # Advance the mech state
        self.mech.advance(v, dt, temp)

        if self.use_fast_imem:
            i_cap = self.cm_c * (v_new - v_prev) / (2 * dt)
            self.i_membrane[:] = i_cap + self.mech.imem * area

        if self.hd:
            v_new = v_new - self.beta * self.filter(v_new)

        return v_new, v


class MyelinatedHeterogeneous(Axon):
    def __init__(
        self,
        n_ax: int,
        n_comp: int,
        temp=37.0,
        v_init=-70.0,
        method="dufort-frankel",
        beta=0.0,
    ):
        if method not in {"dufort-frankel", "df"}:
            raise ValueError(
                f"Method {method} is not supported for heterogeneous axons."
            )
        diameters = torch.ones(n_ax)
        super().__init__(diameters, n_comp, temp, v_init, method, beta)

    def _register_buffers(self, diameters):
        self.register_buffer("diam", torch.empty(self.n_ax, 1, self.n_comp))
        self.register_buffer("node_l", torch.empty(self.n_ax, 1, self.n_comp))
        self.register_buffer(
            "inter_node_length", torch.empty(self.n_ax, 1, self.n_comp - 1)
        )

        self.register_buffer(
            "inter_node_diam", torch.empty(self.n_ax, 1, self.n_comp - 1)
        )
        self.register_buffer("area_c", torch.empty(self.diam.shape))
        self.register_buffer("cm_c", torch.empty(self.area_c.shape))
        self.register_buffer("ra_c", torch.empty(self.inter_node_diam.shape))

        self.register_buffer("v_init_c", torch.tensor(self.v_init))
        self.register_buffer("temp_c", torch.tensor(self.temp))

    def ra_(self, inter_node_diam, inter_node_length) -> torch.Tensor:
        radii = inter_node_diam / 20000  # radius in cm
        inl = inter_node_length / 10000  # um -> cm
        return (self.rhoa * inl) / (torch.pi * (radii**2))

    def area_(self, diameters, node_l) -> torch.Tensor:
        dx = node_l / 10000  # um -> cm
        return torch.pi * (diameters / 10000) * dx

    def set_diam(self, diams):
        diams = torch.as_tensor(diams, dtype=self.dtype()).unsqueeze(1)
        self.diam[:] = diams
        self.calculate_geometric_params()

    def set_inter_node_diam(self, inter_node_diam):
        inter_node_diam = torch.as_tensor(
            inter_node_diam, dtype=self.dtype()
        ).unsqueeze(1)
        self.inter_node_diam[:] = inter_node_diam
        self.calculate_geometric_params()

    def set_inl(self, inl):
        inl = torch.as_tensor(inl, dtype=self.dtype()).unsqueeze(1)
        self.inter_node_length[:] = inl
        self.calculate_geometric_params()

    def set_node_l(self, node_l):
        node_l = torch.as_tensor(node_l, dtype=self.dtype()).unsqueeze(1)
        self.node_l[:] = node_l
        self.calculate_geometric_params()

    def set_all(self, node_d, ind, node_l, inl):
        diams = torch.as_tensor(node_d, dtype=self.dtype()).unsqueeze(1)
        inl = torch.as_tensor(inl, dtype=self.dtype()).unsqueeze(1)
        node_l = torch.as_tensor(node_l, dtype=self.dtype()).unsqueeze(1)
        inter_node_diam = torch.as_tensor(ind, dtype=self.dtype()).unsqueeze(1)
        self.diam[:] = diams
        self.inter_node_length[:] = inl
        self.node_l[:] = node_l
        self.inter_node_diam[:] = inter_node_diam
        self.calculate_geometric_params()

    def calculate_geometric_params(self):
        self.area_c[:] = self.area_(self.diam, self.node_l)
        self.cm_c[:] = self.cm_(self.area_c)
        self.ra_c[:] = self.ra_(self.inter_node_diam, self.inter_node_length)

    def run(
        self,
        ve: Tensor = None,
        space: Tensor = None,
        time: Tensor = None,
        dt: float = None,
        intra: Optional[IntraStim] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
        progressbar: bool = True,
        first: bool = True,
        multicontact: bool = False,
        longrunning: bool = False,
    ):
        with_intra = intra is not None
        if with_intra:
            if not isinstance(intra, IntraStim):
                raise ValueError("intra must be an instance of IntraStim")

        intra_only = False
        if ve is None and (space is None and time is None):
            if intra is None:
                raise ValueError(
                    "Either ve or ve_s and ve_t or intra must be provided."
                )
            intra_only = True
            ve_zero = torch.zeros_like(self.v)

        device = self.device()
        if ve is not None:
            ve = torch.as_tensor(ve, device=device)

        dt = dt if dt is not None else A.dt
        self.warn_about_dt(dt)
        self.dt = dt

        method = getattr(self, f"step_no_intra_df")
        method_intra = getattr(self, f"step_intra_df")

        with torch.set_grad_enabled(self.training):
            if ve is None and not intra_only:
                ve = ve_from_s_t(space, time, self.n_ax, self.device(), multicontact)

            ve = 2 * ve

            if self.training:
                self.calculate_geometric_params()

            if (not self.initialized) or reinit:
                if "_steady_state" in self._caches:
                    self.restore("_steady_state")
                    self.t_ind = 0
                else:
                    self.init_v()
                    self.initialize(self.v, self.v_init_c, self.temp_c)
                    self.post_initialize()
                    self.t_ind = 0
                    self.initialized = True
                if with_intra:
                    intra.init(self)
            else:
                self.detach()

            if first:
                if callbacks:
                    for c in callbacks:
                        c.dt = dt

                if not isinstance(callbacks, CallbackList):
                    callbacks = CallbackList(callbacks)
                callbacks.pre_loop_hook(self)

            dt = torch.as_tensor(dt, device=device)

            rap = F.pad(self.ra_c, (1, 1), "reflect")
            s = dt / self.cm_c
            phi_l = s / rap[:, :, :-1]
            phi_r = s / rap[:, :, 1:]
            s = 2 * s
            phi_sum = phi_l + phi_r

            if progressbar:
                if not isinstance(progressbar, tqdm):
                    progressbar = tqdm(total=ve.shape[0], desc=f"{self.t:.3f} ms")

            for i in range(ve.shape[0]):
                ve_ = ve[i] if not intra_only else ve_zero
                if with_intra:
                    self.v, self.v_prev = method_intra(
                        self.v,
                        self.v_prev,
                        ve_,
                        self.area_c,
                        s,
                        phi_l,
                        phi_r,
                        phi_sum,
                        dt,
                        self.temp_c,
                        intra(self.t_ind, self.v),
                    )
                else:
                    self.v, self.v_prev = method(
                        self.v,
                        self.v_prev,
                        ve_,
                        self.area_c,
                        s,
                        phi_l,
                        phi_r,
                        phi_sum,
                        dt,
                        self.temp_c,
                    )
                callbacks.post_step_hook(self)
                self.t_ind += 1

                if progressbar:
                    progressbar.update(1)
                    if self.t_ind % 100 == 0:
                        progressbar.set_description(f"{self.t:.1f} ms")

            if not longrunning:
                if progressbar:
                    progressbar.close()

    @torch.jit.script_method
    def ssd_df(self, v_c, v_p, v_e, phi_l, phi_r):
        v_c_p = 2 * F.pad(v_c, (1, 1), "reflect")
        v_e_p = F.pad(v_e, (1, 1), "reflect")

        l = (v_c_p[:, :, :-2] - v_p + v_e_p[:, :, :-2] - v_e) * phi_l
        r = (v_c_p[:, :, 2:] - v_p + v_e_p[:, :, 2:] - v_e) * phi_r

        return l + r

    @torch.jit.script_method
    def step_no_intra_df(
        self, v, v_prev, ve, area, s, phi_l, phi_r, phi_sum, dt, temp
    ) -> Tuple[Tensor, Tensor]:
        # Calculate the ionic current
        i_ion = self.mech.i(v_prev, v) * area

        # Calculate the new voltage
        v_new = (v_prev + self.ssd_df(v, v_prev, ve, phi_l, phi_r) - s * i_ion) / (
            1 + phi_sum + s * self.mech.gtot() * area
        )

        # Calculate the total current
        self.mech.itot(v)

        # Advance the mech state
        self.mech.advance(v, dt, temp)

        if self.use_fast_imem:
            i_cap = self.cm_c * (v_new - v_prev) / (2 * dt)
            self.i_membrane[:] = i_cap + self.mech.imem * area

        if self.hd:
            v_new = v_new - self.beta * self.filter(v_new)

        return v_new, v

    @torch.jit.script_method
    def step_intra_df(
        self, v, v_prev, ve, area, s, phi_l, phi_r, phi_sum, dt, temp, intra
    ) -> Tuple[Tensor, Tensor]:
        # Calculate the ionic current
        i_ion = self.mech.i(v_prev, v) * area - intra

        # Calculate the new voltage
        v_new = (v_prev + self.ssd_df(v, v_prev, ve, phi_l, phi_r) - s * i_ion) / (
            1 + phi_sum + s * self.mech.gtot() * area
        )

        # Calculate the total current
        self.mech.itot(v)

        # Advance the mech state
        self.mech.advance(v, dt, temp)

        if self.use_fast_imem:
            i_cap = self.cm_c * (v_new - v_prev) / (2 * dt)
            self.i_membrane[:] = i_cap + self.mech.imem * area

        if self.hd:
            v_new = v_new - self.beta * self.filter(v_new)

        return v_new, v

    @classmethod
    def from_geom(
        cls,
        node_d,
        node_l,
        inter_node_diam,
        inter_node_length,
        temp=None,
        v_init=None,
        method=None,
    ):
        if temp is None:
            temp = cls._get_init_defaults()["temp"]
        if v_init is None:
            v_init = cls._get_init_defaults()["v_init"]
        if method is None:
            method = cls._get_init_defaults()["method"]
        n_ax = len(node_d)
        n_comp = len(node_d[0])
        axon = cls(n_ax, n_comp, temp, v_init, method)
        axon.set_all(node_d, inter_node_diam, node_l, inter_node_length)
        return axon

    @classmethod
    def _get_init_defaults(cls):
        """Extract default values from __init__ signature."""
        signature = inspect.signature(cls.__init__)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty and k != "self"
        }


class ExtCell(Axon):
    def __init__(
        self,
        n_ax: int,
        n_comp: int,
        temp=37.0,
        v_init=-70.0,
        n_layers=2,
        integrator=None,
    ):
        if integrator is None:
            integrator = ax.bwd_euler_bt()
        self.n_layers = n_layers
        diameters = torch.ones(n_ax)
        super().__init__(diameters, n_comp, temp, v_init, integrator)

    def calculate_geometric_params(self):
        pass

    def _register_buffers(self, diameters):
        self.register_buffer("diam",        torch.full((self.n_ax, self.n_comp), 10.0))
        self.register_buffer("L",           torch.full((self.n_ax, self.n_comp), 10.0))

        self.register_buffer("xraxial",     torch.full((self.n_ax, self.n_comp, self.n_layers), 1e9))
        self.register_buffer("xc",          torch.full((self.n_ax, self.n_comp, self.n_layers), 0.0))
        self.register_buffer("xg",          torch.full((self.n_ax, self.n_comp, self.n_layers), 1e9))

        self.register_buffer("v_init_c",    torch.tensor(self.v_init))
        self.register_buffer("temp_c",      torch.tensor(self.temp))

    def x(self):
        node_l = torch.atleast_2d(self.L.squeeze())
        x = node_l.cumsum(dim=1) - node_l / 2
        return x - torch.sum(node_l, dim=1, keepdim=True) / 2
