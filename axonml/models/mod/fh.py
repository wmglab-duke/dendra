# Frankenhaeuser - Huxley channels for Xenopus


from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(
        aq10=3.0,
        bq10=20.0,
        cq10=10.0,
        aA=0.36,
        bA=22.0,
        cA=3.0,
        aB=0.4,
        bB=13.0,
        cB=20.0,
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return self.q10() * self.aA * exprelr(self.bA - v, self.cA)

    def beta(self, v):
        return self.q10() * self.aB * exprelr(v - self.bB, self.cB)

    def breakpoint(self, v):
        v = v + 70.0
        a = self.alpha(v)
        b = self.beta(v)
        taum = 1 / (a + b)
        minf = a * taum

    def inf(self, v):
        v = v + 70.0
        a = self.alpha(v)
        b = self.beta(v)
        return a / (a + b)


class h(State):
    USEQ10()

    PARAMETER(
        aq10=3.0,
        bq10=20.0,
        cq10=10.0,
        aA=0.1,
        bA=-10.0,
        cA=6.0,
        aB=4.5,
        bB=45.0,
        cB=10.0,
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return self.q10() * self.aA * exprelr(v - self.bA, self.cA)

    def beta(self, v):
        return self.q10() * self.aB / (exp((self.bB - v) / self.cB) + 1.0)

    def breakpoint(self, v):
        v = v + 70.0
        a = self.alpha(v)
        b = self.beta(v)
        tauh = 1 / (a + b)
        hinf = a * tauh

    def inf(self, v):
        v = v + 70.0
        a = self.alpha(v)
        b = self.beta(v)
        return a / (a + b)


class n(State):
    USEQ10()

    PARAMETER(
        aq10=3.0,
        bq10=20.0,
        cq10=10.0,
        aA=0.02,
        bA=35.0,
        cA=10.0,
        aB=0.05,
        bB=10.0,
        cB=10.0,
    )

    DERIVATIVE("n' = (ninf - n) / taun")
    ASSIGNED("ninf", "taun")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return self.q10() * self.aA * exprelr(self.bA - v, self.cA)

    def beta(self, v):
        return self.q10() * self.aB * exprelr(v - self.bB, self.cB)

    def breakpoint(self, v):
        v = v + 70.0
        a = self.alpha(v)
        b = self.beta(v)
        taun = 1 / (a + b)
        ninf = a * taun

    def inf(self, v):
        v = v + 70.0
        a = self.alpha(v)
        b = self.beta(v)
        return a / (a + b)


class p(State):
    USEQ10()

    PARAMETER(
        aq10=3.0,
        bq10=20.0,
        cq10=10.0,
        aA=0.006,
        bA=40.0,
        cA=10.0,
        aB=0.09,
        bB=-25.0,
        cB=20.0,
    )

    DERIVATIVE("p' = (pinf - p) / taup")
    ASSIGNED("pinf", "taup")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return self.q10() * self.aA * exprelr(self.bA - v, self.cA)

    def beta(self, v):
        return self.q10() * self.aB * exprelr(v - self.bB, self.cB)

    def breakpoint(self, v):
        v = v + 70.0
        a = self.alpha(v)
        b = self.beta(v)
        taup = 1 / (a + b)
        pinf = a * taup

    def inf(self, v):
        v = v + 70.0
        a = self.alpha(v)
        b = self.beta(v)
        return a / (a + b)


class fh(Mechanism):
    STATE(m, h, n, p)

    PARAMETER(
        pnabar=8e-3,
        ppbar=0.54e-3,
        pkbar=1.2e-3,
        gl=30.3e-3,
        el=-69.74,
        R=8.31441,
        FARADAY=96514,
    )

    USEION("k", read=["ki", "ko"], write=["ik"])
    USEION("na", read=["nai", "nao"], write=["ina"])

    NONSPECIFIC_CURRENT("il")

    def ina(self, v):
        z = 1e-3 * self.FARADAY * v / (self.R * (self.temp + 273.15))
        enao = self.nao * expinv(z)
        enai = self.nai * expinv(-z)
        ghkna = 1e-3 * self.FARADAY * (enai - enao)
        return ((self.pnabar * self.m**2 * self.h) + (self.ppbar * self.p**2)) * ghkna

    def ik(self, v):
        z = 1e-3 * self.FARADAY * v / (self.R * (self.temp + 273.15))
        eko = self.ko * expinv(z)
        eki = self.ki * expinv(-z)
        ghk = 1e-3 * self.FARADAY * (eki - eko)
        return self.pkbar * self.n**2 * ghk

    def il(self, v):
        return self.gl * (v - self.el)
