from typing import Tuple

import torch
from torch.nn import functional as F
from torch import Tensor

from ..core import Axon


class Heterogeneous(Axon):
    def __init__(
        self,
        n_ax: int,
        n_node: int,
        temp=37.0,
        v_init=-70.0,
        method="dufort-frankel",
        pade=None,
    ):
        if method not in {"dufort-frankel", "df"}:
            raise ValueError(
                f"Method {method} is not supported for heterogeneous axons."
            )
        diameters = torch.ones(n_ax)
        super().__init__(diameters, n_node, temp, v_init, method, pade)

    def _register_buffers(self, diameters):
        self.register_buffer("diam", torch.empty(self.n_ax, 1, self.n_node))
        self.register_buffer("node_l", torch.empty(self.n_ax, 1, self.n_node))
        self.register_buffer(
            "inter_node_length", torch.empty(self.n_ax, 1, self.n_node - 1)
        )

        self.register_buffer(
            "inter_node_diam", torch.empty(self.n_ax, 1, self.n_node - 1)
        )
        self.register_buffer("area_c", torch.empty(self.diam.shape))
        self.register_buffer("cm_c", torch.empty(self.area_c.shape))
        self.register_buffer(
            "ra_c", torch.empty(self.inter_node_diam.shape)
        )

        self.register_buffer("v_init_c", torch.tensor(self.v_init))
        self.register_buffer("temp_c", torch.tensor(self.temp))

    def ra_(self, inter_node_diam, inter_node_length) -> torch.Tensor:
        radii = inter_node_diam / 20000  # radius in cm
        return (self.rhoa * inter_node_length) / (torch.pi * (radii**2))
    
    def area_(self, diameters, node_l) -> torch.Tensor:
        dx = node_l / 10000 # um -> cm
        return torch.pi * (diameters / 10000) * dx

    def set_diam(self, diams):
        diams = torch.as_tensor(diams, dtype=self.dtype()).unsqueeze(1)
        self.diam[:] = diams
        self.calculate_geometric_params()

    def set_inter_node_diam(self, inter_node_diam):
        inter_node_diam = torch.as_tensor(inter_node_diam, dtype=self.dtype()).unsqueeze(1)
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

    @torch.jit.script_method
    def ssd_df(self, v_c, v_p, v_e, s_2):
        v_c_p = F.pad(v_c, (1, 1), "reflect")
        v_e_p = F.pad(v_e, (1, 1), "reflect")
        s_2_p = F.pad(s_2, (1, 1), "reflect")

        v_p = 0.5 * v_p

        l = (v_c_p[:, :, :-2] - v_p + v_e_p[:, :, 2:] - v_e) * s_2_p[:, :, :-1]
        r = (v_c_p[:, :, 2:] - v_p + v_e_p[:, :, :-2] - v_e) * s_2_p[:, :, 1:]

        return l + r

    @torch.jit.script_method
    def step_no_intra_df(
        self, v, v_prev, ve, area, s, s2, dt, temp
    ) -> Tuple[Tensor, Tensor]:
        # Advance the mechanism
        self.mech.advance(v, dt, temp)

        # Calculate the ionic current
        i_ion = self.mech.i(v_prev) * area

        # Calculate the new voltage
        v_new = (v_prev + self.ssd_df(v, v_prev, ve, s2) - s * i_ion) / (
            1 + s2 + s * self.mech.gtot() * area
        )

        # Calculate the total current
        self.mech.itot(v)

        return v_new, v

    @torch.jit.script_method
    def step_intra_df(
        self, v, v_prev, ve, area, s, s2, dt, temp, intra
    ) -> Tuple[Tensor, Tensor]:
        # Advance the mechanism
        self.mech.advance(v, dt, temp)

        # Calculate the ionic current
        i_ion = self.mech.i(v_prev) * area - intra

        # Calculate the new voltage
        v_new = (v_prev + self.ssd_df(v, v_prev, ve, s2) - s * i_ion) / (
            1 + s2 + s * self.mech.gtot() * area
        )

        # Calculate the total current
        self.mech.itot(v)

        return v_new, v
