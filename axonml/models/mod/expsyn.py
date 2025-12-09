import torch

from ..mechanisms import PointProcess as PP
from ..mechanisms import State as S
from ..mechanisms import Synapse as Syn
from ..mechanisms.ops import *


class g(S):
    S.STATE("g")
    S.RANGE(tau=0.1)
    S.DERIVATIVE("g' = -g / tau")

    def inf(self, v):
        return {"g": torch.zeros_like(v)}


class expsyn(PP, Syn):
    r"""
    A synaptic mechanism that uses a single-exponential function to model synaptic
    conductance.

    Parameters
    ----------
    e : float
        The reversal potential (in mV) of the synapse. Default is 0.0 mV.
    tau : float
        The time constant (in ms) of the synapse. Default is 0.1 ms.

    Notes
    -----
    Synaptic inputs are received as weights, which are added to the state variable
    :math:`g` to simulate synaptic activation. The state variable :math:`g` then
    decays exponentially with time constant :math:`\tau`.

    Optionally, the decay of :math:`g` can be written as

    .. math::

        g(t) = g_0 \exp\left(-\frac{t}{\tau}\right)

    where :math:`g_0` is the conductance immediately after synaptic activation.
    """

    PP.STATE(g)
    PP.RANGE(e=0.0)
    PP.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)

    def net_receive(self, weights, netcon):
        self.g = self.g + weights
