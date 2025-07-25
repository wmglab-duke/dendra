from ..mechanisms._mechanism import PointProcess as PP, Synapse as Syn
from ..mechanisms._state import State as S
from ..mechanisms.ops import *


class g(S):
    S.STATE("g")
    S.RANGE(tau=0.1)
    S.DERIVATIVE("g' = -g / tau")

    def inf(self, v):
        return {"g": torch.zeros_like(v)}


class expsyn(PP, Syn):
    PP.STATE(g)
    PP.RANGE(e=0.0)
    PP.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)

    def net_receive(self, weights, netcon):
        self.g = self.g + weights
