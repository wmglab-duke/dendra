from ..mechanisms import *
from ..mechanisms.ops import expit, exprelr, exp


class m(State):
    USEQ10()

    PARAMETER(amA=1.0, aq10=2.24659524757)

    DERIVATIVE("m' = (minf - m) / mtau")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 6.3) / 10)

    def alpha(self, v):
        return exprelr(2.5 - 0.1 * (v + 70), self.amA)

    def beta(self, v):
        return 4.0 * exp(-(v + 70.0) / 18.0)

    def export(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = a + b
        minf = a / s
        mtau = 1.0 / (self.q10() * s)
        return minf, mtau


class h(State):
    USEQ10()

    PARAMETER(aq10=2.24659524757)

    DERIVATIVE("h' = (hinf - h) / htau")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 6.3) / 10)

    def alpha(self, v):
        return 0.07 * exp(-(v + 70) / 20)

    def beta(self, v):
        return expit(3 - 0.1 * (v + 70))

    def export(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = a + b
        hinf = a / s
        htau = 1.0 / (self.q10() * s)
        return hinf, htau


class n(State):
    USEQ10()

    PARAMETER(anA=1.0, aq10=2.24659524757)

    DERIVATIVE("n' = (ninf - n) / ntau")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 6.3) / 10)

    def alpha(self, v):
        return 00.1 * exprelr(1.0 - 0.1 * (v + 70.0), self.anA)

    def beta(self, v):
        return 0.125 * exp(-(v + 70.0) / 80.0)

    def export(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = a + b
        ninf = a / s
        ntau = 1.0 / (self.q10() * s)
        return ninf, ntau


class rattay_aberham(Mechanism):
    STATE(m, h, n)

    PARAMETER(gnabar=0.12, gkbar=0.036, gl=0.0003, el=-59.4)

    USEION("na", read=["ena"], write=["ina"])
    USEION("k", read=["ek"], write=["ik"])
    NONSPECIFIC_CURRENT("il")

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)

    def ik(self, v):
        return self.gkbar * self.n**4 * (v - self.ek)

    def il(self, v):
        return self.gl * (v - self.el)
