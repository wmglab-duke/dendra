import math
from typing import List, Tuple, Optional

import torch
import torch.jit as jit
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

        # -- constants --
        self.pi = torch.nn.Parameter(torch.tensor(math.pi, requires_grad=False))
        self.eval()

    def insert(self, mechanism: Mechanism):
        m = mechanism(self.temp, self.v_init)
        self.mechanisms[mechanism.__name__] = m

    def inflate(self, x, area):
        for _, mech in self.mechanisms.items():
            mech.inflate(x, area)

    def advance(self, v, dt):
        for _, mech in self.mechanisms.items():
            mech._advance(v, dt)

    def ra(self, diameters) -> torch.Tensor:
        """Axial resistance of compartment."""
        pass

    def area(self, diameters) -> torch.Tensor:
        """Membrane surface area of compartment."""
        pass

    @torch.jit.script_method
    def init(self, v, diameters):
        area = self.area(diameters)
        self.inflate(v, area)

    @torch.jit.script_method
    def step(self, v, dt):
        self.advance(v, dt)


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
    
    def ra(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        radii = diameters / 20000
        return (self.rhoa * dx) / (self.pi * (radii**2))