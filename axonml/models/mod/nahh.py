# From Traub & Miles "Neuronal networks of the hippocampus" (1991)
# Cummins et al. (2007), Sheets et al. (2007)

from ..mechanisms import *
from ..mechanisms.ops import exprelr, exp, expit


class m(State):
    USEQ10()

    PARAMETER(
        am1=0.32,
        am2=13.1,
        am3=4.0,
        bm1=0.28,
        bm2=40.1,
        bm3=5.0,
        aq10=3.0,
        bq10=30.0,
        cq10=10.0,
        mshift=-6.0,
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        v = v + 65.0 + self.mshift
        return self.q10() * self.am1 * exprelr(self.am2 - v, self.am3)

    def beta(self, v):
        v = v + 65.0 + self.mshift
        return self.q10() * self.bm1 * exprelr(v - self.bm2, self.bm3)

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        taum = 1 / (a + b)
        minf = a * taum


class h(State):
    USEQ10()

    PARAMETER(
        ah1=0.128,
        ah2=17.0,
        ah3=18.0,
        bh1=4.0,
        bh2=40.0,
        bh3=5.0,
        aq10=3.0,
        bq10=30.0,
        cq10=10.0,
        hshift=6.0,
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        v = v + 65.0 + self.hshift
        return self.q10() * self.ah1 * exp((self.ah2 - v) / self.ah3)

    def beta(self, v):
        v = v + 65.0 + self.hshift
        return self.q10() * self.bh1 * expit((v - self.bh2) / self.bh3)

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        tauh = 1 / (a + b)
        hinf = a * tauh


class nahh(Mechanism):
    STATE(m, h)

    PARAMETER(gnabar=0.3)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)
    
    def gtot(self):
        return self.gnabar * self.m**3 * self.h
