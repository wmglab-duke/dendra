import warnings

import numpy as np
import torch

import axonml as ax

from axonml.helpers import numpify
from axonml.models.heterogeneous import ExtCell
from axonml.models.heterogeneous.compartments import CompartmentID
from axonml.models.mod import (
    mrg_k, mrg_leak, mrg_naf, mrg_nap, pas
)
from axonml.models.declarations import PARAMETER
from axonml.models.parametric import Functional


ns = ["node", "mysa", "flut", "stin * 6", "flut", "mysa"]

# -- compartment diameters --
fd = lambda model: numpify(model.fd)
axonD = (
    lambda model: 0.553 * numpify(model.fd) - 0.024
)
nodeD = (
    lambda model: 0.321 * (0.553 * numpify(model.fd) - 0.024) + 0.37
)

# -- compartment lengths --
nodelength0 = lambda model: np.full_like(numpify(model.fd), 1.0)
paralength1 = lambda model: np.full_like(numpify(model.fd), 3.0)


def deltax(model):
    fd = numpify(model.fd)
    return -3.22 * fd**2 + 148 * fd - 128


def paralength2(model):
    fd = numpify(model.fd)
    return -0.171 * fd**2 + 6.48 * fd - 0.935


def interlength(model):
    return (
        deltax(model)
        - nodelength0(model)
        - (2 * paralength1(model))
        - (2 * paralength2(model))
    ) / 6


g_scale = lambda model: np.full_like(numpify(model.fd), 0.0001)
m_scale = lambda model: np.full_like(numpify(model.fd), 0.001)

xc = lambda model: 0.1 / (numpify(model.nl) * 2)
xc_node = lambda model: np.full_like(numpify(model.fd), 0.0)

xg = lambda model: 0.001 / (numpify(model.nl) * 2)
xg_node = lambda model: np.full_like(numpify(model.fd), 1e10)


def rpn0(model):
    rhoa = 0.7e6
    node_diam = nodeD(model)
    space_p1 = 0.002
    return (rhoa * 0.01) / (np.pi * ((((node_diam / 2) + space_p1) ** 2) - ((node_diam / 2) ** 2)))


def rpn1(model):
    rhoa = 0.7e6
    mysa_diam = nodeD(model)
    space_p1 = 0.002
    return (rhoa * 0.01) / (np.pi * ((((mysa_diam / 2) + space_p1) ** 2) - ((mysa_diam / 2) ** 2)))


def rpn2(model):
    rhoa = 0.7e6
    axon_diam = axonD(model)
    space_p2 = 0.004
    return (rhoa * 0.01) / (np.pi * ((((axon_diam / 2) + space_p2) ** 2) - ((axon_diam / 2) ** 2)))


def rpx(model):
    rhoa = 0.7e6
    axon_diam = axonD(model)
    space_i = 0.002
    return (rhoa * 0.01) / (np.pi * ((((axon_diam / 2) + space_i) ** 2) - ((axon_diam / 2) ** 2)))


node_d_funcs = {
    "node": nodeD,
    "flut": fd,
    "mysa": fd,
    "stin": fd,
}

node_l_funcs = {
    "node": nodelength0,
    "flut": paralength2,
    "mysa": paralength1,
    "stin": interlength,
}

secd_funcs = {
    "node": fd,
    "flut": axonD,
    "mysa": nodeD,
    "stin": axonD,
}

scale_funcs = {
    "node": g_scale,
    "flut": g_scale,
    "mysa": m_scale,
    "stin": g_scale,
}

xc_funcs = {
    'node': xc_node,
    'flut': xc,
    'mysa': xc,
    'stin': xc,
}

xg_funcs = {
    'node': xg_node,
    'flut': xg,
    'mysa': xg,
    'stin': xg,
}

xr_funcs = {
    "node": rpn0,
    "flut": rpn2,
    "mysa": rpn1,
    "stin": rpx,
}


class mrg_rhoa(Functional):
    def __init__(self, rhoa):
        self.rhoa = rhoa

    def fn(self, model):
        scale = 1 / (model.secd / model.fd[:, None]) ** 2
        return self.rhoa * scale


class mrg_cm(Functional):
    def __init__(self, cm):
        self.cm = cm

    def fn(self, model):
        cm = self.cm * model.secd / model.fd[:, None]
        node_locs = model.cid.locs(["node"])
        cm[:, node_locs] = self.cm
        return cm


class smolMRG(ExtCell):

    PARAMETER(rhoa=mrg_rhoa(70.0), cm=mrg_cm(2.0))

    def __init__(
        self,
        diameters=[2.0],
        n_node=501,
        temp=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        if torch.any((torch.as_tensor(diameters) > 5.7) | (torch.as_tensor(diameters) < 1.011)):
            warnings.warn(
                "Fiber diameter should not be <1.011um or >5.7 um for smolMRG. Use bigMRG for larger fibers."
            )

        cid = CompartmentID(ns, n_node - 1)
        n_ax = len(diameters)
        n_c = cid.nc()

        super().__init__(n_ax, n_c, temp, v_init, integrator=integrator)
        self.register_cid(cid)

        self.register_buffer(
            "fd", 
            torch.tensor(diameters, dtype=self.dtype())
        )
        self.register_buffer(
            "nl", 
            torch.clamp(torch.floor(17.4 * (0.553 * self.fd - 0.024) - 1.74), min=1)
        )
        self.register_buffer(
            "secd",
            torch.tensor(
                self.cid.build(secd_funcs, self), dtype=self.dtype()
            ),
        )
        self.register_buffer(
            "scale",
            torch.tensor(
                self.cid.build(scale_funcs, self), dtype=self.dtype()
            ),
        )

        self.diam[:] = torch.tensor(self.cid.build(node_d_funcs, self))
        self.dx[:]   = torch.tensor(self.cid.build(node_l_funcs, self))

        g       = self.scale * self.secd / self.fd[:, None]
        xc      = self.cid.build(xc_funcs, self)
        xg      = self.cid.build(xg_funcs, self)
        xraxial = self.cid.build(xr_funcs, self)

        self.xc[..., 0]      = torch.tensor(xc, dtype=self.dtype())
        self.xg[..., 0]      = torch.tensor(xg, dtype=self.dtype())
        self.xraxial[..., 0] = torch.tensor(xraxial, dtype=self.dtype())

        self.insert_at(
            ["flut", "mysa", "stin"], pas, g=g, e=self.v_init
        )
        self.insert_at("node", mrg_k, gkbar=0.115556)
        self.insert_at("node", mrg_leak)
        self.insert_at("node", mrg_naf, gnabar=2.333333)
        self.insert_at("node", mrg_nap)

    def c(self, *args):
        locs = self.cid.loc("node")
        n = len(locs)
        return [locs[round((n - 1) * arg)] for arg in args]

    def steady_state(self, dt=1.0, tstop=200):
        return super().steady_state(dt, tstop)