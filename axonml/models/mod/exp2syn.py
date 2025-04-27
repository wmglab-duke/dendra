from ..mechanisms import *
from ..mechanisms.ops import *


class A(State):

    PARAMETER(tau1=0.1)
    DERIVATIVE("A' = -A / tau1")

    def inf(self, v):
        return torch.zeros_like(v)
    

class B(State):
    PARAMETER(tau2=10.0)
    DERIVATIVE("B' = -B / tau2")

    def inf(self, v):
        return torch.zeros_like(v)
    

class exp2syn(Mechanism):
    STATE(A, B)
    PARAMETER(e=0)
    ASSIGNED("factor")

    NONSPECIFIC_CURRENT("i")

    def initial(self):
        tau1 = self.DE['A'].tau1
        tau2 = self.DE['B'].tau2
        tp = (tau1 * tau2) / (tau1 - tau2) * log(tau1 / tau2)
        factor = -exp(tp / tau1) + exp(-tp / tau2)
        self.factor = 1 / factor

    def i(self, v):
        return (self.B - self.A) * (v - self.e)
    
    def net_receive(self, weights):
        weights = weights * self.factor
        self.A = self.A + weights
        self.B = self.B + weights
