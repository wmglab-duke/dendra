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
        model.register_buffer(
            "v", torch.full((model.n_ax, model.n_comp), model.v_init)
        )
        self.register_buffer("i_membrane", torch.zeros((model.n_ax, model.n_comp)))
        self.mech = mech

    @classmethod
    def shape(cls, n_ax, n_comp):
        return (n_ax, n_comp)

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


class SCIntegrator(torch.nn.Module):
    def __init__(self, model, mech, imem=None, N=1, P=1, C=1):
        super().__init__()
        self.mech = mech
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        self.register_buffer("cmdt", torch.tensor(0.0))
        self.register_buffer("i_membrane", torch.tensor(0.0))
        model.register_buffer("v", torch.full((N, P, C), model.v_init))

    @classmethod
    def shape(cls, n_ax, n_comp):
        defaults = get_init_defaults(cls)
        return (defaults["N"], defaults["P"], defaults["C"])

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
