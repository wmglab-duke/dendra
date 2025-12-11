import torch

from ..mechanisms import PointProcess as PP
from ..mechanisms import State as S
from ..mechanisms import Synapse as Syn
from ..mechanisms.ops import exp, log


class A(S):
    S.STATE("A")
    S.RANGE(tau1=0.1)
    S.DERIVATIVE("A' = -A / tau1")

    def inf(self, v):
        return {"A": torch.zeros_like(v)}


class B(S):
    S.STATE("B")
    S.RANGE(tau2=10.0)
    S.DERIVATIVE("B' = -B / tau2")

    def inf(self, v):
        return {"B": torch.zeros_like(v)}


class exp2syn(PP, Syn):
    r"""
    A synaptic mechanism that uses a double-exponential function to model synaptic
    conductance. The conductance is modeled using two state variables ``A`` and
    ``B``, each following an exponential decay with different time constants.

    Parameters
    ----------
    e : float
        The reversal potential (in mV) of the synapse. Default is 0.0 mV.
    tau1 : float
        The rise time constant (in ms) of the synapse. Default is 0.1 ms.
    tau2 : float
        The decay time constant (in ms) of the synapse. Default is 10.0 ms.

    Notes
    -----
    The synaptic conductance :math:`g` is calculated as the difference between the
    two state variables :math:`B` and :math:`A`, scaled by the reversal potential
    :math:`e`. Synaptic inputs are received as weights, which are added to both
    state variables :math:`A` and :math:`B` to simulate synaptic activation.

    The factor for scaling the weights is calculated based on the time constants
    of the two states, following the formula

    .. math::

        \text{factor} =
        \frac{1}{-\exp\left(-\frac{t_p}{\tau_1}\right)
                + \exp\left(-\frac{t_p}{\tau_2}\right)}

    where :math:`t_p` is the time point at which the two exponentials are equal:

    .. math::

        t_p = \frac{\tau_1 \tau_2}{\tau_2 - \tau_1}
            \log\left(\frac{\tau_2}{\tau_1}\right)

    This ensures that the synaptic response is properly normalized based on the
    time constants.
    """

    PP.STATE(A, B)
    PP.RANGE(e=0.0)
    PP.ASSIGNED("factor")

    PP.NONSPECIFIC_CURRENT("i")

    def initial(self, v):
        tau1 = self.DE["A"].tau1
        tau2 = self.DE["B"].tau2
        tp = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
        factor = -exp(-tp / tau1) + exp(-tp / tau2)
        self.factor = 1 / factor

    def i(self, v):
        return (self.B - self.A) * (v - self.e)

    def net_receive(self, weights, netcon):
        weights = weights * self.factor
        self.A = self.A + weights
        self.B = self.B + weights
