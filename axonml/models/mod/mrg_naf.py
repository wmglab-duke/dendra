# McIntyre, Richardson, Grill 2002 - NaF channel

from ..mechanisms import *
from ..mechanisms.ops import expit, exprelr


class m(State):
    USEQ10()

    PARAMETER(
        amA=1.86,
        amB=21.4,
        amC=10.3,
        bmA=0.086,
        bmB=25.7,
        bmC=9.16,
        aq10_1=2.2,
        bq10=20.0,
        cq10=10.0,
    )

    DERIVATIVE("m' = (minf - m) / mtau")
    ASSIGNED("minf", "mtau")

    def calc_q10(self):
        return self.aq10_1 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        x = -(v + self.amB)
        return self.q10() * self.amA * exprelr(x, self.amC)

    def beta(self, v):
        x = v + self.bmB
        return self.q10() * self.bmA * exprelr(x, self.bmC)

    def breakpoint(self, v):
        am = self.alpha(v)
        bm = self.beta(v)
        mtau = 1 / (am + bm)
        minf = am * mtau


class h(State):
    USEQ10()

    PARAMETER(
        ahA=0.062,
        ahB=114.0,
        ahC=11.0,
        bhA=2.3,
        bhB=31.8,
        bhC=13.4,
        aq10_2=2.9,
        bq10=20.0,
        cq10=10.0,
    )

    DERIVATIVE("h' = (hinf - h) / htau")
    ASSIGNED("hinf", "htau")

    def calc_q10(self):
        return self.aq10_2 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        x = v + self.ahB
        return self.q10() * self.ahA * exprelr(x, self.ahC)

    def beta(self, v):
        return self.q10() * self.bhA * expit((v + self.bhB) / self.bhC)

    def breakpoint(self, v):
        ah = self.alpha(v)
        bh = self.beta(v)
        htau = 1 / (ah + bh)
        hinf = ah * htau


class mrg_naf(Mechanism):
    STATE(m, h)

    PARAMETER(gnabar=3.0, ena=50.0)

    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)
