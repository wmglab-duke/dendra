import math
from typing import List, Tuple, Optional

import torch
from torch import Tensor

from axonml import trained
from .callbacks import CallbackList, Callback
from .backend import Backend as A
from .mixins import Parameterized
from .mechanisms.core import Mechanism, MechanismInterface
from .mechanisms.declarations import PARAMETER


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
        self.mechanisms = torch.nn.ModuleDict()

        # solver stuff
        weight = [1.0, -2.0, 1.0]
        self.ssd = SymmetricConv1D(
            2, 1, 3, bias=False, padding="same", padding_mode="reflect"
        )
        self.ssd.weight.data = torch.tensor([weight, weight]).reshape(1, 2, 3)
        for p in self.ssd.parameters():
            p.requires_grad = False

        self.v = torch.tensor([self.v_init])
        self.initialized: bool = False

        # -- constants --
        self.pi = torch.nn.Parameter(torch.tensor(math.pi, requires_grad=False))
        self.eval()

    @torch.jit.export
    def n(self) -> int:
        return self.v.shape[0]

    def device(self):
        return self.ssd.weight.device

    def insert(self, mechanism: Mechanism, ic=None, **kwargs):
        v_init = torch.tensor(self.v_init, device=self.device())
        m = mechanism(self.temp, v_init, ic=ic, **kwargs)
        self.mechanisms[mechanism.__name__] = m

    def advance_mechanisms(self, v, dt):
        for _, mech in self.mechanisms.items():
            mech._advance(v, dt)

    def area_(self, diameters) -> torch.Tensor:
        """Membrane surface area of compartment."""
        pass

    def ra_(self, diameters) -> torch.Tensor:
        """Axial resistance of compartment."""
        pass

    def cm_(self, area) -> Tuple[Tensor, Tensor]:
        """Capacitance of compartment."""
        pass

    def run(
        self,
        ve: Tensor,
        diameters: Tensor,
        dt: float = None,
        intra: Optional[Tensor] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
    ):
        with torch.set_grad_enabled(self.training):
            dt = dt if dt is not None else A.dt
            dt = torch.tensor(dt, device=self.device())

            if callbacks:
                for c in callbacks:
                    c.dt = dt

            ve = torch.as_tensor(ve, device=self.device())
            diameters = torch.as_tensor(diameters, device=self.device())
            area = self.area_(diameters)

            if (not self.initialized) or reinit:
                self.v = torch.full_like(ve[0], self.v_init)
                self.cm_c = self.cm_(area)
                self.ra_c = self.ra_(diameters)
                self.init_buffers(self.v_init)
                self.init(self.v)
                self.initialized = True

            callbacks = CallbackList(callbacks)
            callbacks.pre_loop_hook(self)

            for i in range(len(ve)):
                self.v = self.step(
                    self.v, ve[i], self.cm_c, self.ra_c, dt, area, i, intra
                )
                callbacks.post_step_hook(self)

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
    def init_buffers(self, v_init: float) -> None:
        for _, m in self.mechanisms.items():
            m.init(torch.tensor(v_init, device=self.device()))

    @torch.jit.script_method
    def advance_vm(self, vm: Tensor, dv: Tensor) -> Tensor:
        vm = vm + dv
        return vm

    @torch.jit.script_method
    def i(self, v, area, idx: int, intra: Optional[Tensor] = None) -> Tensor:
        out = torch.tensor(0.0, device=self.device())
        for _, m in self.mechanisms.items():
            c = m.i(v)
            if c is not None:
                out = out + c
        out = out * area[:, None, None]
        if intra is not None:
            out -= intra[idx]
        return out

    @torch.jit.script_method
    def init(self, v) -> None:
        self.inflate(v)

    def inflate(self, v):
        for _, mech in self.mechanisms.items():
            mech.inflate(v)

    @torch.jit.script_method
    def step(
        self, v, ve, cm, ra, dt, area, i: int, intra: Optional[Tensor] = None
    ) -> Tensor:
        # -- 2nd diff --
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)

        # -- update gvs --
        self.advance_mechanisms(v, dt)

        # -- calculate ionic current --
        i_ion = self.i(v, area, i, intra)

        # -- update vm --
        dv = self.dv(cm, ra, d2v, i_ion, dt)
        v = self.advance_vm(v, dv)

        return v

    @torch.jit.script_method
    def get_state(self, s: str) -> Tensor:
        if s == "v":
            return self.v
        mech, state = s.split(".")
        m: MechanismInterface = self.mechanisms[mech]
        return m.get(state)

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
        return self

    @torch.jit.script_method
    def all_states(self) -> List[str]:
        out = ["v"]
        for n, m in self.mechanisms.items():
            for s in m.states.keys():
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

    def cm_(self, area) -> torch.Tensor:
        return self.cm * area

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

    def cm_(self, area):
        return self.cm * area

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
