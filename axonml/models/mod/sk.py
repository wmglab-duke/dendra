from ..mechanisms import *
from ..mechanisms.ops import *


class n(State):
    USEQ10()

    DERIVATIVE("n' = (ninf - n) / taun")
    ASSIGNED("ninf", "taun")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        pca = log10(self.cai) - 3
        ninf = 1/(1+exp(-1*(pca + 6.4)/.12))
        taun = -1 * pca / self.q10()

    def inf(self, v):
        pca = log10(self.cai) - 3
        return 1/(1+exp(-1*(pca + 6.4)/.12))
    

class sk(Mechanism):
    STATE(n)
    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])
    USEION("ca", read=["cai"])

    def ik(self, v):
        return self.gbar * self.n * (v - self.ek)
