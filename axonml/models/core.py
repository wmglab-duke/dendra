import math
from typing import List, Tuple, Optional, Dict

import torch
from torch import Tensor

from axonml import trained
from .callbacks import CallbackList, Callback
from .backend import Backend as A
from .mixins import Parameterized
from .mechanisms.core import Mechanism
from .mechanisms.declarations import PARAMETER
from .mechanisms.handler.handler import build_handler


@torch.jit.interface
class HandlerInterface:
    def i_intra(self, v, area, intra) -> torch.Tensor:
        pass

    def i_no_intra(self, v, area) -> torch.Tensor:
        pass

    def initialize(self, v, v_init) -> None:
        pass

    def advance(self, v, dt) -> None:
        pass

    def get(self, mech: str, state: str) -> torch.Tensor:
        pass


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        if self.training:
            weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        else:
            weight_ = self.weight
        return self._conv_forward(x, weight_, self.bias)


class Axon(Parameterized, torch.jit.ScriptModule):
    """Base 1D fiber class."""

    def __init__(self, temp=37.0, v_init=-80.0):
        super().__init__()
        self.temp = temp
        self.v_init = v_init
        self.mech: HandlerInterface = None

        self._m_list = []
        self._m_name = []
        self._m_curr = {}

        # solver stuff
        weight = [1.0, -2.0, 1.0]
        self.ssd = SymmetricConv1D(
            2, 1, 3, bias=False, padding="same", padding_mode="reflect"
        )
        self.ssd.weight.data = torch.tensor([weight, weight]).reshape(1, 2, 3)
        for p in self.ssd.parameters():
            p.requires_grad = False

        self.v = torch.tensor([v_init])
        self.initialized: bool = False

        # -- constants --
        self.pi = torch.nn.Parameter(torch.tensor(math.pi), requires_grad=False)
        self.eval()

    @torch.jit.export
    def n(self) -> int:
        return self.v.shape[0]

    def device(self):
        return self.ssd.weight.device

    def dtype(self):
        return self.ssd.weight.dtype

    def insert(self, mechanism: Mechanism, ic=None, **kwargs):
        m = mechanism(self.temp, ic=ic, **kwargs)
        self._m_list.append(m)
        self._m_name.append(mechanism.__name__)
        for k, v in m._currents.items():
            self._m_curr.setdefault(k, {}).update({mechanism.__name__: v})

    def build(self):
        self.mech = build_handler(self._m_list, self._m_name, self._m_curr)

    def area_(self, diameters):
        raise NotImplementedError()

    def ra_(self, diameters):
        raise NotImplementedError()

    def cm_(self, area):
        return self.cm * area

    def run(
        self,
        ve: Tensor,
        diameters: Tensor,
        dt: float = None,
        intra: Optional[Tensor] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
    ):
        with_intra = intra is not None
        with torch.set_grad_enabled(self.training):
            device = self.device()

            dt = dt if dt is not None else A.dt
            dt = torch.tensor(dt, device=device)

            if callbacks:
                for c in callbacks:
                    c.dt = dt

            ve = torch.as_tensor(ve, device=device)
            diameters = torch.as_tensor(diameters, device=device)
            area = self.area_(diameters)

            if (not self.initialized) or reinit:
                self.v = torch.full_like(ve[0], self.v_init)
                self.cm_c = self.cm_(area)
                self.ra_c = self.ra_(diameters)
                self.initialize()
                self.initialized = True

            callbacks = CallbackList(callbacks)
            callbacks.pre_loop_hook(self)

            for i in range(len(ve)):
                if with_intra:
                    self.v = self.step_intra(
                        self.v, ve[i], self.cm_c, self.ra_c, dt, area, intra[i]
                    )
                else:
                    self.v = self.step_no_intra(
                        self.v, ve[i], self.cm_c, self.ra_c, dt, area
                    )
                callbacks.post_step_hook(self)

    @torch.jit.script_method
    def initialize(self):
        v_init = torch.tensor(self.v_init, device=self.device(), dtype=self.dtype())
        self.mech.initialize(self.v, v_init)

    def dv(self, cm, ra, d2v, ion, dt) -> Tensor:
        """Calculate dv/dt

        Parameters
        ----------
        cm : torch.Tensor
            Node capacitance.
        ra : torch.Tensor
            Internodal resistance.
        d2v : torch.Tensor
            Spatial 2nd difference.
        ion : torch.Tensor
            Ionic current.
        dt : float
            Timestep.

        Returns
        -------
        Tensor
            dv/dt
        """
        cm = cm[:, None, None]
        ra = ra[:, None, None]
        dv = dt * (1 / cm) * (((1 / ra) * d2v) - ion)
        return dv

    @torch.jit.script_method
    def step_no_intra(
        self,
        v,
        ve,
        cm,
        ra,
        dt,
        area,
    ) -> Tensor:
        # -- 2nd diff --
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)

        # -- update gvs --
        self.mech.advance(v, dt)

        # -- calculate ionic current --
        i_ion = self.mech.i_no_intra(v, area)

        # -- update vm --
        dv = self.dv(cm, ra, d2v, i_ion, dt)
        v = v + dv

        return v

    @torch.jit.script_method
    def step_intra(self, v, ve, cm, ra, dt, area, intra) -> Tensor:
        # -- 2nd diff --
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)

        # -- update gvs --
        self.mech.advance(v, dt)

        # -- calculate ionic current --
        i_ion = self.mech.i_intra(v, area, intra)

        # -- update vm --
        dv = self.dv(cm, ra, d2v, i_ion, dt)
        v = v + dv

        return v

    @torch.jit.script_method
    def get_state(self, s: str) -> Tensor:
        if s == "v":
            return self.v
        mech, state = s.split(".")
        return self.mech.get(mech, state)

    def load(self, state_dict):
        if state_dict in trained:
            state_dict = torch.load(
                trained[state_dict], map_location=self.device(), weights_only=True
            )
        elif isinstance(state_dict, str):
            state_dict = torch.load(
                state_dict, map_location=self.device(), weights_only=True
            )
        self.load_state_dict(state_dict)
        return self

    def compile(self, nodes=16, axons=1):
        ve = torch.ones(1, axons, 1, nodes, device=self.device())
        d = 10 * torch.ones(axons, device=self.device())
        for _ in range(5):
            self.run(ve, d, reinit=True)
        self.initialized = False
        return self

    @torch.jit.ignore
    def all_states(self) -> List[str]:
        out = ["v"]
        for n in self._m_name:
            for s in getattr(self.m, n).states.keys():
                out.append(f"{n}.{s}")
        return out


class Unmyelinated(Axon):
    PARAMETER(
        {
            "membrane": {
                "cm": 1e-3,  # mF / cm2
                "rhoa": 35.4,  # ohm-cm
            }
        }
    )

    def __init__(self, dx=10.0, temp=37, v_init=-80):
        super().__init__(temp, v_init)
        self.dx: float = dx

    def area_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        return self.pi * (diameters / 10000) * dx

    def ra_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        radii = diameters / 20000
        return (self.rhoa * dx) / (self.pi * (radii**2))


class Myelinated(Axon):
    PARAMETER(
        {
            "axon_d": {
                "axond1": 0.0187623,
                "axond2": 4.787487e-01,
                "axond3": 1.203613e-01,
            },
            "node_d": {
                "noded1": 6.303781e-03,
                "noded2": 2.070544e-01,
                "noded3": 5.339006e-01,
            },
            "delta_x": {
                "deltax1": -8.215284e00,
                "deltax2": 2.724201e02,
                "deltax3": -7.802411e02,
            },
            "membrane": {
                "cm": 10e-3,
                "rhoa": 70.0,  # ohm-cm
            },
        }
    )

    def area_(self, diameters):
        lengths = torch.ones_like(diameters) / 10000
        return self.pi * self.nodeD(diameters) * lengths  # cm2

    def ra_(self, diameters):
        radii = diameters / 20000  # radius in cm
        rhoa = self.rhoa * self.rhoa_scale(diameters)
        return (rhoa * self.deltax(diameters)) / (self.pi * (radii**2))

    def rhoa_scale(self, diameters):
        return 1 / ((self.axonD(diameters) / diameters) ** 2)

    def axonD(self, diameters):
        axond = self.axond1 * diameters**2 + self.axond2 * diameters + self.axond3
        return axond

    def deltax(self, diameters):
        deltax = self.deltax1 * diameters**2 + self.deltax2 * diameters + self.deltax3
        return deltax / 10000

    def nodeD(self, diameters):
        noded = self.noded1 * diameters**2 + self.noded2 * diameters + self.noded3
        return noded / 10000
