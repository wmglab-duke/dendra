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
    r"""
    Base class for all integrators.

    Parameters
    ----------
    model : axonml.models.Population
        The population model to be integrated.
    mech : axonml.mechanisms.MechanismHandler
        The mechanism handler containing all mechanisms.
    imem : bool, optional
        Whether to use membrane current tracking. Default is None, which
        uses the global default defined by `axonml.helpers.IMEM`.

    Notes
    -----
    Subclasses should implement the :meth:`initialize` and :meth:`step` methods.
    These methods define how the integrator initializes its own state (e.g., pre-computes
    relevant constants) and advances the model state by one time step, respectively.

    """

    __constants__ = {"imem"}
    v_vars = ["v"]

    def __init__(self, model, mech, imem=None):
        super().__init__()
        imem = imem if imem is not None else IMEM
        self.imem = bool(imem)
        self.mech = mech
        self.initialized = False
        self.dt = None
        self.shape = None

    def initialize(self, model, dt):
        r"""Initializes the integrator state. Must be implemented by subclasses.

        Parameters
        ----------
        model : axonml.models.Population
            The population model to be integrated.
        dt : float
            The time step for integration.
        """
        raise NotImplementedError

    def step(self, model, dt, ve=None, intra=None):
        r"""Advances the model state by one time step. Must be implemented by subclasses.

        Parameters
        ----------
        model : axonml.models.Population
            The population model to be integrated.
        dt : float
            The time step for integration.
        ve : torch.Tensor, optional
            The extracellular potential at each compartment (in mV). Default is None.
        intra : torch.Tensor, optional
            The intracellular current at each compartment (in mA). Default is None.
        """
        raise NotImplementedError

    def needs_to_be_initialized(self, model, dt, force=False):
        if force:
            return True
        return not self.initialized or self.dt != float(dt) or self.shape != model.shape

    def _initialize(self, model, dt, force=False):
        if self.needs_to_be_initialized(model, dt, force):
            self.dt = float(dt)
            self.shape = model.shape
            for mech in self.mech.mechanisms.values():
                mech.set_dt(dt)
            self.initialize(model, dt)
            self.initialized = True

    def init_v(self, model):
        model.v = model.v.detach().clone().contiguous().copy_(model.v_init)
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()

    def detach(self, model):
        """Detach model state variables from the computation graph.

        Parameters
        ----------
        model : axonml.models.Population
            The model whose state variables are to be detached.
        """

        detach_vars(model, self.v_vars)
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
            model.i_membrane = torch.zeros_like(model.v).detach()
