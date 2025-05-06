import torch
from axonml.helpers import IMEM


class Integrator(torch.jit.ScriptModule):
    """
    Base class for all integrators.
    """

    __constants__ = {"imem"}

    def __init__(self, model, mech, imem=None):
        super().__init__()
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        model.register_buffer(
            "v", torch.full((model.n_ax, 1, model.n_comp), model.v_init)
        )
        self.register_buffer("i_membrane", torch.zeros((model.n_ax, 1, model.n_comp)))
        self.mech = mech

    def init_v(self, model):
        model.v[:] = model.v_init
        model.v.detach_()
        if self.imem:
            model.i_membrane[:] = 0.0
            model.i_membrane.detach_()

    def detach(self, model):
        model.v.detach_()
        if self.imem:
            model.i_membrane.detach_()
        self.mech.detach()
