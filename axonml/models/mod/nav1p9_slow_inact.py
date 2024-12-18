# NaV1.9 Na+ current from Herzog, Cummins, and Waxman 2001 p1353
# This current is also called the ttx-rp current (the tetrodotoxin resistent persistant current)


from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_am=1.032,
        B_am=6.99,
        C_am=-14.87115,
        A_bm=5.79,
        B_bm=130.4,
        C_bm=22.9,
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return self.A_am * sigmoid((-self.B_am - v) / self.C_am)

    def beta(self, v):
        return self.A_bm * sigmoid((-self.B_bm - v) / self.C_bm)

    def inf(self, v):
        return self.alpha(v) / (self.alpha(v) + self.beta(v))

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        minf = a * s
        taum = self.q10() * s


class h(State):
    USEQ10()

    PARAMETER(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_ah=0.06435,
        B_ah=73.26415,
        C_ah=3.71928,
        A_bh=0.13496,
        B_bh=10.27853,
        C_bh=-9.09334,
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return self.A_ah * sigmoid((-self.B_ah - v) / self.C_ah)

    def beta(self, v):
        return self.A_bh * sigmoid((-self.B_bh - v) / self.C_bh)

    def inf(self, v):
        return self.alpha(v) / (self.alpha(v) + self.beta(v))

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        hinf = a * s
        tauh = self.q10() * s


class s(State):
    USEQ10()

    PARAMETER(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_as=0.00000016,
        B_as=0.0,
        C_as=12.0,
        A_bs=0.0005,
        B_bs=32.0,
        C_bs=23.0,
    )

    DERIVATIVE("s' = (sinf - s) / taus")
    ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return self.A_as * sigmoid((v + self.B_as) / self.C_as)

    def beta(self, v):
        return self.A_bs * sigmoid((v + self.B_bs) / self.C_bs)

    def inf(self, v):
        return self.alpha(v) / (self.alpha(v) + self.beta(v))

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        sinf = a * s
        taus = self.q10() * s


class nav1p9_slow_inact(Mechanism):
    STATE(m, h, s)

    PARAMETER(gbar=0.0)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m * self.h * self.s * (v - self.ena)
