import torch
from axonml.helpers import IMEM

import inspect


def get_init_defaults(cls):
    signature = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }


class Integrator(torch.nn.Module):
    """
    Base class for all integrators.
    """

    __constants__ = {"imem"}

    def __init__(self, model, mech, imem=None):
        super().__init__()
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        self.mech = mech
        self.register_buffer("i_membrane", torch.zeros(model.np, model.nc))
        self.initialized = False
        self.dt = None

    @classmethod
    def shape(cls, np, nc):
        return (np, nc)

    def init_v(self, model):
        model.v = torch.full(model.v.shape, model.v_init, dtype=model.v.dtype, device=model.v.device)
        model.v.detach_()
        if self.imem:
            model.i_membrane = torch.zeros(model.i_membrane.shape, dtype=model.i_membrane.dtype, device=model.i_membrane.device)
            model.i_membrane.detach_()

    def detach(self, model):
        model.v.detach_()
        if self.imem:
            model.i_membrane.detach_()
        self.mech.detach()


class SCIntegrator(torch.nn.Module):
    def __init__(self, model, mech, imem=None):
        super().__init__()
        self.mech = mech
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        self.register_buffer("cmdt", torch.tensor(0.0))
        self.register_buffer("i_membrane", torch.zeros(model.np, model.nc))
        self.initialized = False
        self.dt = None

    @classmethod
    def shape(cls, np, nc):
        return (np, nc)

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
