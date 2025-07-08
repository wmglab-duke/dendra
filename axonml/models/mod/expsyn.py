from ..mechanisms._synapse import Synapse as Syn
from ..mechanisms._state import State as S
from ..mechanisms.ops import *


class g(S):
    S.STATE('g')
    S.PARAMETER(tau=0.1)
    S.DERIVATIVE("g' = -g / tau")

    def inf(self, v):
        return {'g':torch.zeros_like(v)}


class expsyn(Syn):

    Syn.STATE(g)
    Syn.PARAMETER(e=0)
    Syn.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
    
    def net_receive(self, weights):
        self.g = self.g + weights