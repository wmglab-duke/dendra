import inspect

import torch

from dendra.helpers import IMEM, detach_vars


def get_init_defaults(cls):
    signature = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }


def _expanded_v_init(model):
    """
    Return ``model.v_init`` expanded to ``model.v.shape``.

    Population implements ``expanded_v_init`` directly. The fallback here keeps
    integrators robust for Network/MultiIntegrator-like containers that expose
    only ``v`` and ``v_init``.
    """
    if hasattr(model, "expanded_v_init"):
        return model.expanded_v_init()

    target = model.v
    target_shape = tuple(target.shape)
    if len(target_shape) < 1:
        raise ValueError(
            f"model.v must have at least one dimension; got {target_shape}."
        )

    v0 = torch.as_tensor(model.v_init, device=target.device, dtype=target.dtype)

    if v0.ndim == 0 or v0.numel() == 1:
        return v0.reshape(()).expand_as(target)

    # For flattened Network state, model.nc may not exist; use the last voltage
    # dimension as the compartment/state dimension.
    n_comp = int(getattr(model, "nc", target_shape[-1]))
    if target_shape[-1] != n_comp:
        n_comp = int(target_shape[-1])

    if v0.ndim == 1:
        if v0.numel() != n_comp:
            raise ValueError(
                f"v_init has length {v0.numel()}, but the voltage state expects "
                f"length {n_comp}. Use a scalar or a vector matching model.nc/the "
                "last voltage dimension."
            )
        view_shape = (1,) * (target.ndim - 1) + (n_comp,)
        return v0.reshape(view_shape).expand_as(target)

    if tuple(v0.shape) == target_shape:
        return v0

    if target.ndim >= 2 and tuple(v0.shape) == tuple(target_shape[-2:]):
        view_shape = (1,) * (target.ndim - 2) + tuple(target_shape[-2:])
        return v0.reshape(view_shape).expand_as(target)

    if target.ndim >= 2 and tuple(v0.shape) == (1, target_shape[-1]):
        view_shape = (1,) * (target.ndim - 2) + (1, target_shape[-1])
        return v0.reshape(view_shape).expand_as(target)

    raise ValueError(
        "Unsupported v_init shape. Expected a scalar, a 1D vector matching "
        f"model.nc/the last voltage dimension ({n_comp}), the model core shape, "
        f"or full voltage shape {target_shape}; got shape {tuple(v0.shape)}."
    )


class Integrator(torch.nn.Module):
    r"""
    Base class for all integrators.

    Parameters
    ----------
    model : dendra.models.Population
        The population model to be integrated.
    mech : dendra.mechanisms.MechanismHandler
        The mechanism handler containing all mechanisms.
    imem : bool, optional
        Whether to use membrane current tracking. Default is None, which
        uses the global default defined by `dendra.helpers.IMEM`.

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
        model : dendra.models.Population
            The population model to be integrated.
        dt : float
            The time step for integration.
        """
        raise NotImplementedError

    def step(self, model, dt, ve=None, intra=None):
        r"""Advances the model state by one time step. Must be implemented by subclasses.

        Parameters
        ----------
        model : dendra.models.Population
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
        model.v = _expanded_v_init(model).clone().detach().contiguous()
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()

    def detach(self, model):
        """Detach model state variables from the computation graph.

        Parameters
        ----------
        model : dendra.models.Population
            The model whose state variables are to be detached.
        """

        detach_vars(model, self.v_vars)
        for n, b in model.named_buffers():
            setattr(model, n, b.detach())
        if self.imem:
            model.i_membrane = model.i_membrane.detach()
        self.mech.detach()

    def mutable_state_dict(self, model):
        dct = {}
        for var in self.v_vars:
            dct[var] = getattr(model, var)
        if self.imem:
            dct["i_membrane"] = model.i_membrane
        return dct

    def restore_mutable_state_dict(self, model, state_dict):
        for var in self.v_vars:
            setattr(model, var, state_dict[var])
        if self.imem:
            model.i_membrane = state_dict["i_membrane"]


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
        model.v = _expanded_v_init(model).clone().detach().contiguous()
        if self.write_back:
            self._calc_splits(model)
            _write_back(model, self.split_at)
        if self.imem:
            model.i_membrane = torch.zeros_like(model.v).detach()
