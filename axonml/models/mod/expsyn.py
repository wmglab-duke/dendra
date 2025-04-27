from ..mechanisms import *
from ..mechanisms.ops import *


class g(State):

    PARAMETER(tau=0.1)
    DERIVATIVE("g' = -g / tau")

    def inf(self, v):
        return torch.zeros_like(v)
    

class expsyn(Mechanism):

    STATE(g)
    PARAMETER(e=0)
    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
    
    def net_receive(self, weights):
        self.g = self.g + weights