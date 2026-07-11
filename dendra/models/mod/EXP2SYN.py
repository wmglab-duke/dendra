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

    As in NEURON's ``Exp2Syn``, the effective ratio ``tau1 / tau2`` is limited
    to ``[1e-9, 0.9999]`` during initialization. Equal, nearly equal, or
    reversed time constants therefore approach a normalized alpha-synapse
    response instead of producing a division by zero. Both time constants must
    be positive and finite; invalid values raise :class:`ValueError`. For low
    precision dtypes, the bounds are tightened to the nearest representable
    values strictly between zero and one.
    """

    PP.STATE(A, B)
    PP.RANGE(e=0.0)
    PP.BUFFER("factor")

    PP.NONSPECIFIC_CURRENT("i")

    def initial(self, v):
        state_a = self.DE["A"]
        tau1 = state_a.tau1
        tau2 = self.DE["B"].tau2
        valid = torch.isfinite(tau1) & torch.isfinite(tau2) & (tau1 > 0) & (tau2 > 0)
        if not bool(torch.all(valid).item()):
            raise ValueError("exp2syn tau1 and tau2 must be positive and finite.")

        ratio = tau1 / tau2
        zero = torch.zeros_like(ratio)
        one = torch.ones_like(ratio)
        lower = torch.maximum(
            torch.full_like(ratio, 1.0e-9), torch.nextafter(zero, one)
        )
        upper = torch.minimum(
            torch.full_like(ratio, 0.9999), torch.nextafter(one, zero)
        )
        ratio = torch.minimum(torch.maximum(ratio, lower), upper)

        # NEURON adjusts tau1 itself, so the A-state kinetics must use the same
        # effective value as the normalization factor.
        state_a.tau1 = ratio * tau2

        # At the peak, exp(-tp/tau1) is exactly ratio times
        # exp(-tp/tau2). This log-domain form is algebraically equivalent to
        # NEURON's tp expression but stays finite as ratio approaches one.
        log_denominator = torch.log1p(-ratio) + ratio / (1 - ratio) * log(ratio)
        self.factor = exp(-log_denominator)

    def i(self, v):
        return (self.B - self.A) * (v - self.e)

    def i_with_conductance(self, v):
        """Return the current and its exact voltage derivative."""
        conductance = self.B - self.A
        return conductance * (v - self.e), conductance

    def net_receive(self, weights, netcon):
        weights = weights * self.factor
        self.A = self.A + weights
        self.B = self.B + weights
