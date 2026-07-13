import torch

from ..mechanisms import PointProcess as M
from ..mechanisms.ops import *


class alphasynapse(M):
    """
    A synaptic mechanism that uses the alpha function to model synaptic conductance.
    The conductance `g` is calculated using the alpha function based on the time
    since the synapse was activated.

    Parameters
    ----------
    onset : float
        The time (in ms) when the synapse is activated. Default is 0.0 ms.
    tau : float
        The time constant (in ms) of the alpha function. Default is 0.1 ms.
    gmax : float
        The maximum conductance (in µS) of the synapse. Default is 0.0 µS.
    e : float
        The reversal potential (in mV) of the synapse. Default is 0.0 mV.

    Notes
    -----
    This is a :class:`~dendra.models.mechanisms.PointProcess`: ``gmax`` and
    ``g`` are numerical values in µS, while ``i(v)`` returns lumped,
    outward-positive current in nA. Dendra converts the current to mA/cm² using
    target compartment area during current assembly.
    """

    M.RANGE(onset=0.0, tau=0.1, gmax=0.0, e=0.0)
    M.NONSPECIFIC_CURRENT("i")

    @property
    def g(self):
        return self.gmax * alpha((self.t - self.onset) / self.tau)

    def i(self, v):
        return self.g * (v - self.e)


def alpha(x: torch.Tensor) -> torch.Tensor:
    """
    Efficient PyTorch implementation of the NEURON alpha function.

    alpha(x) = 0,                if x < 0 or x > 10
             = x * exp(1 - x),   if 0 <= x <= 10

    Args:
        x (torch.Tensor): A tensor of input values.

    Returns:
        torch.Tensor: The result of the alpha function applied element-wise.
    """
    # Calculate the value for the 'else' case: x * exp(1 - x)
    y_calc = x * torch.exp(1.0 - x)

    # Define the condition for the 'if' case.
    condition = (x < 0.0) | (x > 10.0)

    # If condition is true, take 0.0. Otherwise, take the calculated value.
    return torch.where(condition, 0.0, y_calc)
