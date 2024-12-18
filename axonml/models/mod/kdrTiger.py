# Sheets 2007

from ..mechanisms import *
from ..mechanisms.ops import *


class n(State):
    USEQ10()

    PARAMETER(aq10=3.3, bq10=22.0, cq10=10.0, k1=15.4, vh=35.0)

    DERIVATIVE("n' = (ninf - n) / ntau")
    ASSIGNED("ninf", "ntau")

    def calc_q10(self):
        return 1.0 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def breakpoint(self, v):
        ninf = sigmoid((v + self.vh - 10.0) / self.k1)
        ntau_a = 0.16 + 0.8 * exp(-0.0267 * (v + 11.0))
        ntau_b = 1000 * (
            0.000688 + 1 / (exp((v + 75.2) / 6.5) + exp((v - 131.5) / -34.8))
        )
        ntau = self.q10() * torch.where(v < -31.0, ntau_b, ntau_a)

    def inf(self, v):
        return sigmoid((v + self.vh - 10.0) / self.k1)


class kdrTiger(Mechanism):
    STATE(n)

    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.n**4 * (v - self.ek)
