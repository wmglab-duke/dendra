import warnings

import numpy as np
import torch

from axonml.helpers import numpify, ctx
from ...heterogeneous import Heterogeneous
from ...heterogeneous.compartments import CompartmentID
from ...mod import mrg_k, mrg_leak, mrg_naf, mrg_nap, pas
from ...declarations import PARAMETER
from ...parametric import Functional


ns = ["node", "mysa", "flut", "stin * 6", "flut", "mysa"]

# -- compartment diameters --
fd = lambda model: numpify(model.fd)
axonD = lambda model: 0.553 * numpify(model.fd) - 0.024
nodeD = lambda model: 0.321 * (0.553 * numpify(model.fd) - 0.024) + 0.37

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


class mrg_rhoa(Functional):
    def __init__(self, rhoa):
        self.rhoa = rhoa

    def fn(self, model):
        scale = 1 / (model.secd / model.fd[:, None, None]) ** 2
        return self.rhoa * scale


class mrg_cm(Functional):
    def __init__(self, cm, node_cm, mysa_cm):
        self.cm = cm
        self.node_cm = node_cm
        self.mysa_cm = mysa_cm

    def fn(self, model):
        cm = self.cm / (model.nl * 2)[:, None]
        cm = cm.expand(model.n_ax, model.n_comp).unsqueeze(1).clone()
        node_locs = model.cid.locs(["node"])
        cm[:, :, node_locs] = self.node_cm
        mysa_locs = model.cid.locs(["mysa"])
        mysa_cm = self.mysa_cm / (model.nl * 2)[:, None]
        mysa_cm = mysa_cm.expand(model.n_ax, len(mysa_locs)).unsqueeze(1)
        cm[:, :, mysa_locs] = mysa_cm
        return cm


class g_mrg(Functional):
    def __init__(self, g):
        self.g = g

    def fn(self, model):
        g_myel = self.g / (model.nl * 2)[:, None, None]
        g_memb = model.scale * model.secd / model.fd[:, None, None]
        g = 1 / (1 / g_myel + 1 / g_memb)
        return g


class smolMRG(Heterogeneous):
    _dt_lim = 0.001

    PARAMETER(rhoa=mrg_rhoa(70.0), cm=mrg_cm(0.1, 2.0, 5.0))

    def __init__(
        self,
        diameters=[2.0],
        n_node=101,
        temp=37.0,
        v_init=-80.0,
    ):
        if torch.any(torch.as_tensor(diameters) < 1.02):
            warnings.warn("Fiber diameter should not be less than 1.02 um for smolMRG.")
        if torch.any(torch.as_tensor(diameters) > 5.7):
            warnings.warn(
                "Fiber diameter should not exceed 5.7 um for smolMRG. Use SMF instead."
            )

        cid = CompartmentID(ns, n_node - 1)
        n_ax = len(diameters)
        n_c = cid.nc()

        super().__init__(n_ax, n_c, temp, v_init)
        self.register_cid(cid)

        self.register_buffer("fd", torch.tensor(diameters, dtype=self.dtype()))
        self.register_buffer(
            "nl",
            torch.clamp(torch.floor(17.4 * (0.553 * self.fd - 0.024) - 1.74), min=1),
        )
        self.register_buffer(
            "secd",
            torch.tensor(
                self.cid.build(secd_funcs, self), dtype=self.dtype()
            ).unsqueeze(1),
        )
        self.register_buffer(
            "scale",
            torch.tensor(
                self.cid.build(scale_funcs, self), dtype=self.dtype()
            ).unsqueeze(1),
        )

        self.diam[:] = torch.tensor(self.cid.build(node_d_funcs, self))[:, None, :]
        self.node_l[:] = torch.tensor(self.cid.build(node_l_funcs, self))[:, None, :]

        self.calculate_geometric_params()

        with ctx(PADE=1):
            self.insert(pas, mask_out=self.cid.loc("node"), g=g_mrg(0.001), e=v_init)
            self.insert(mrg_k, mask_in=self.cid.loc("node"), gkbar=0.115556)
            self.insert(mrg_nap, mask_in=self.cid.loc("node"))
            self.insert(mrg_naf, mask_in=self.cid.loc("node"), gnabar=2.33333)
            self.insert(mrg_leak, mask_in=self.cid.loc("node"))

    def c(self, *args):
        locs = self.cid.loc("node")
        n = len(locs)
        return [locs[round((n - 1) * arg)] for arg in args]

    def steady_state(self, dt=1.0, maxiter=3000):
        return super().steady_state(dt, maxiter)
