# transient ttx-sensitive Na+ current from Sheets et al. 2007

from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(
        aq10=2.5,
        bq10=21,
        cq10=10,
        A_am=15.5,
        B_am=-5.0,
        C_am=-12.08,
        A_bm=35.2,
        B_bm=72.7,
        C_bm=16.7,
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return self.A_am * sigmoid((-self.B_am - v) / self.C_am)

    def beta(self, v):
        return self.A_bm * sigmoid((-self.B_bm - v) / self.C_bm)

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        taum = self.q10() * s
        minf = a * s


class h(State):
    USEQ10()

    PARAMETER(
        aq10=2.5,
        bq10=21,
        cq10=10,
        A_ah=0.38685,
        B_ah=122.35,
        C_ah=15.29,
        A_bh=2.00283,
        B_bh=5.5266,
        C_bh=-12.70195,
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return self.A_ah * sigmoid((-self.B_ah - v) / self.C_ah)

    def beta(self, v):
        return -0.00283 + self.A_bh * sigmoid((-self.B_bh - v) / self.C_bh)

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        tauh = self.q10() * s
        hinf = a * s


class s(State):
    USEQ10()

    PARAMETER(
        aq10=2.5,
        bq10=21,
        cq10=10,
        A_as=0.00092,
        B_as=93.9,
        C_as=16.6,
        A_bs=-132.05,
        B_bs=-384.9,
        C_bs=28.5,
    )

    DERIVATIVE("s' = (sinf - s) / taus")
    ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return 0.00003 + self.A_as * sigmoid((-self.B_as - v) / self.C_as)

    def beta(self, v):
        return 132.05 + self.A_bs * sigmoid((-self.B_bs - v) / self.C_bs)

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        taus = self.q10() * s
        sinf = a * s


class nattxs(Mechanism):
    STATE(m, h, s)

    PARAMETER(gbar=0.0)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)
