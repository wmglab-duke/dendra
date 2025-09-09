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


@torch.compile
def _write_back(model, split_at):
    # write v back to the constituent populations
    v_f = model.v
    splits = torch.tensor_split(v_f, split_at, dim=-1)
    for split, pop in zip(splits, model.populations.values()):
        pop.v = split.reshape_as(pop.v)


class MultiIntegrator(Integrator):
    def __init__(self, model, mech, imem=None, write_back=True):
        super().__init__(model, mech, imem)
        self.write_back = write_back
        self.split_at = None

    def _write_back(self, model):
        if self.write_back:
            _write_back(model, self.split_at)

    def _calc_splits(self, models):
        split_lengths = [m.numelc() for m in models]
        self.split_at = torch.cumsum(torch.tensor(split_lengths), dim=0)[:-1].tolist()

    def init_v(self, model):
        model.v = model.v_init.expand_as(model.v).clone().detach().contiguous()
        if self.write_back:
            self._calc_splits(model)
            _write_back(model, self.split_at)
        if self.imem:
            model.i_membrane = torch.zeros(
                model.i_membrane.shape,
                dtype=model.i_membrane.dtype,
                device=model.i_membrane.device,
            ).detach()
