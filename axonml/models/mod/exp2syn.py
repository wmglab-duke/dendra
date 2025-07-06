from ..mechanisms._synapse import Synapse as Syn
from ..mechanisms._state import State as S
from ..mechanisms.ops import *


class A(S):
    S.STATE('A')
    S.PARAMETER(tau1=0.1)
    S.DERIVATIVE("A' = -A / tau1")

    def inf(self, v):
        return {'A':torch.zeros_like(v)}


class B(S):
    S.STATE('B')
    S.PARAMETER(tau2=10.0)
    S.DERIVATIVE("B' = -B / tau2")

    def inf(self, v):
        return {'B':torch.zeros_like(v)}


class exp2syn(Syn):
    Syn.STATE(A, B)
    Syn.PARAMETER(e=0)
    Syn.ASSIGNED("factor")

    Syn.NONSPECIFIC_CURRENT("i")

    def initial(self):
        tau1 = self.DE['A'].tau1
        tau2 = self.DE['B'].tau2
        tp = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
        factor = -exp(-tp / tau1) + exp(-tp / tau2)
        self.factor = 1 / factor

    def i(self, v):
        return (self.B - self.A) * (v - self.e)
    
    def net_receive(self, weights):
        weights = weights * self.factor
        self.A = self.A + weights
        self.B = self.B + weights
