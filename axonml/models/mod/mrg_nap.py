# McIntyre, Richardson, Grill 2002

from ..mechanisms import *
from ..mechanisms.ops import expit, exprelr


class p(State):
    USEQ10()

    PARAMETER(
        ampA=0.01,
        ampB=27.0,
        ampC=10.2,
        bmpA=0.00025,
        bmpB=34.0,
        bmpC=10.0,
        pq10_1=2.2,
        bq10=20.0,
        cq10=10.0,
    )

    DERIVATIVE("p' = (pinf - p) / ptau", pade=True)
    ASSIGNED("pinf", "ptau")

    def calc_q10(self):
        return self.pq10_1 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        x = -(v + self.ampB)
        return self.q10() * self.ampA * exprelr(x, self.ampC)

    def beta(self, v):
        x = v + self.bmpB
        return self.q10() * self.bmpA * exprelr(x, self.bmpC)

    def breakpoint(self, v):
        amp = self.alpha(v)
        bmp = self.beta(v)
        ptau = 1 / (amp + bmp)
        pinf = amp * ptau


class mrg_nap(Mechanism):
    STATE(p)

    PARAMETER(gnapbar=0.01, ena=50.0)

    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.gnapbar * self.p**3 * (v - self.ena)