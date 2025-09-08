import inspect

import torch

from axonml.helpers import IMEM, detach_vars


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
    v_vars = ["v"]

    def __init__(self, model, mech, imem=None):
        super().__init__()
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        self.mech = mech
        self.register_buffer("i_membrane", torch.zeros(model.shape))
        self.initialized = False
        self.dt = None
        self.shape = None

    def initialize(self, model, dt):
        raise NotImplementedError

    def needs_to_be_initialized(self, model, dt):
        return not self.initialized or self.dt != float(dt) or self.shape != model.shape

    def _initialize(self, model, dt):
        if self.needs_to_be_initialized(model, dt):
            self.dt = float(dt)
            self.shape = model.shape
            self.initialize(model, dt)
            self.initialized = True

    def init_v(self, model):
        model.v = torch.full(
            model.v.shape, model.v_init, dtype=model.v.dtype, device=model.v.device
        ).detach()
        if self.imem:
            model.i_membrane = torch.zeros(
                model.i_membrane.shape,
                dtype=model.i_membrane.dtype,
                device=model.i_membrane.device,
            ).detach()

    def detach(self, model):
        detach_vars(self, self.v_vars)
        for n, b in model.named_buffers():
            setattr(model, n, b.detach())
        if self.imem:
            model.i_membrane = model.i_membrane.detach()
        self.mech.detach()
