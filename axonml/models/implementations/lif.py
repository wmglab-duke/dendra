import torch

from ..core import Axon
from ..mod import pas, fire
from ..mechanisms import PARAMETER


class LIF(Axon):
    PARAMETER(cm=1.0, rhoa=100.0)
    def __init__(
            self, 
            diameters, 
            L=10.0, 
            temp=37, 
            v_init=-70, 
            g_pas=0.1, 
            threshold=-50, 
            method="dufort-frankel"
        ):
        n_comp = 1
        self.dx: float = L
        self.L: float = L
        super().__init__(diameters, n_comp, temp, v_init, method)
        self.insert(pas, e=v_init, g=g_pas)
        self.insert(fire, threshold=threshold, rest=v_init)

    def area_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        return torch.pi * (diameters / 10000) * dx

    def ra_(self, diameters) -> torch.Tensor:
        dx = torch.full_like(diameters, self.dx / 10000)
        radii = diameters / 20000
        return (self.rhoa * dx) / (torch.pi * (radii**2))