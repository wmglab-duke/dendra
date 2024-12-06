import math
from typing import List, Tuple, Optional

import torch
from torch import Tensor

from axonml import trained
from .callbacks import CallbackList, Callback
from .backend import Backend as A
from .mixins import Parameterized
from .mechanisms.core import Mechanism
from .mechanisms.declarations import PARAMETER


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        return self._conv_forward(x, weight_, self.bias)
    

class Axon(torch.jit.ScriptModule, Parameterized):

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

        self.v = torch.tensor(self.v_init)
        self.initialized : bool = False

        # -- constants --
        self.pi = torch.nn.Parameter(torch.tensor(math.pi, requires_grad=False))
        self.eval()

    def insert(self, mechanism: Mechanism):
        m = mechanism(self.temp, self.v_init)
        self.mechanisms[mechanism.__name__] = m

    def inflate(self, x, area):
        for _, mech in self.mechanisms.items():
            mech.inflate(x, area)

    def advance_mechanisms(self, v, dt):
        for _, mech in self.mechanisms.items():
            mech._advance(v, dt)

    def ra(self, diameters) -> torch.Tensor:
        """Axial resistance of compartment."""
        pass

    def cm(self, area) -> Tuple[Tensor, Tensor]:
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
            area = self.area(diameters)

            if not self.initialized:
                self.v = torch.full_like(ve[0], self.v_init)
                self.init(self.v, diameters)
                self.initialized = True
            elif reinit:
                self.v = torch.full_like(ve[0], self.v_init)
                self.init_buffers(self.v_init)
                self.init(self.v, diameters)
                self.initialized = True

            cm = self.cm(area)
            ra = self.ra(diameters)

            for i in range(len(ve)):
                self.v = self.step(self.v, ve[i], cm, ra, dt, i, intra)


    def area(self, diameters) -> torch.Tensor:
        """Membrane surface area of compartment."""
        pass

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
        if self.fp32:
            return dv.float()
        return dv
    
    def init_buffers(self, v_init):
        for _, m in self.mechanisms.items():
            m.init(v_init)

    @torch.jit.script_method
    def advance_vm(self, vm: Tensor, dv: Tensor) -> Tensor:
        vm = vm + dv
        return vm

    @torch.jit.script_method
    def i(self, v, idx: int, intra: Optional[Tensor] = None) -> Tensor:
        out = torch.tensor(0.0, device=self.device())
        for _, m in self.mechanisms.items():
            out += m.i(v)
        if intra is not None:
            out -= intra[idx]
        return out

    @torch.jit.script_method
    def init(self, v, area) -> None:
        self.inflate(v, area)

    @torch.jit.script_method
    def step(self, v, ve, cm, ra, dt, i: int, intra: Optional[Tensor] = None) -> Tensor:
        # -- 2nd diff --
        x = torch.cat([v, ve], dim=1)
        d2v = self.ssd(x)

        # -- update gvs --
        self.advance_mechanisms(v, dt)

        # -- calculate ionic current --
        i_ion = self.i(v, i, intra)

        # -- update vm --
        dv = self.dv(cm, ra, d2v, i_ion, dt)
        v = self.advance_vm(v, dv)

        return v


class Unmyelinated(Axon):

    PARAMETER({
        "membrane": {
            "cm": 1e-3,     # mF / cm2
            "rhoa": 100.0,  # ohm-cm
        }
    })

    def __init__(self, dx=10.0, temp=37, v_init=-80):
        super().__init__(temp, v_init)
        self.dx : float = dx

    def area(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        return self.pi * (diameters / 10000) * dx
    
    def cm(self, area) -> torch.Tensor:
        return self.cm * area
    
    def ra(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        radii = diameters / 20000
        return (self.rhoa * dx) / (self.pi * (radii**2))