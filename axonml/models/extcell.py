import torch

from .integrators import bwd_euler_bt
from .core import Axon


class ExtCell(Axon):
    def __init__(
        self,
        diameters=[10.0],
        n_comp=101,
        celsius=37.0,
        v_init=-80.0,
        n_layers=2,
        integrator=None,
    ):
        if integrator is None:
            integrator = bwd_euler_bt()
        super().__init__(diameters, n_comp, celsius, v_init, integrator)
        self.n_layers = n_layers
        self._register_buffers()
        self.dx[:] = 10.0
        self.x[:] = self._x()

    def _register_buffers(self):
        self.register_buffer(
            "xraxial", torch.full((self.n_ax, self.n_comp, self.n_layers), 1e9)
        )
        self.register_buffer(
            "xc", torch.full((self.n_ax, self.n_comp, self.n_layers), 0.0)
        )
        self.register_buffer(
            "xg", torch.full((self.n_ax, self.n_comp, self.n_layers), 1e9)
        )

    def _x(self):
        dx = self.dx
        starts = torch.zeros_like(dx)
        starts[:, 1:] = torch.cumsum(dx, dim=1)[:, :-1]

        raw_midpoints = starts + dx / 2.0

        total_lengths = torch.sum(dx, dim=1, keepdim=True)
        offsets = (total_lengths / 2.0)

        centered_midpoints = raw_midpoints - offsets

        return centered_midpoints
