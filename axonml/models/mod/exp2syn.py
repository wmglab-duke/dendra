import torch

from ..mechanisms._mechanism import PointProcess as PP
from ..mechanisms._mechanism import Synapse as Syn
from ..mechanisms._state import State as S
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
