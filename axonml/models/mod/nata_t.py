# Colbert and Pan 2002


from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(
        aq10=2.3,
        bq10=21.0,
        cq10=10.0,
        mshift=-38.0,
        ma1=0.182,
        ma2=6.0,
        mb1=0.124,
        mb2=6.0,
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return (
            self.q10()
            * self.ma1
            * (v - self.mshift)
            / (1 - exp(-(v - self.mshift) / self.ma2))
        )

    def beta(self, v):
        return (
            self.q10()
            * self.mb1
            * (-v + self.mshift)
            / (1 - exp(-(-v + self.mshift) / self.mb2))
        )

    def breakpoint(self, v):
        v = torch.where(v == self.mshift, v + 0.0001, v)
        a = self.alpha(v)
        b = self.beta(v)
        taum = 1 / (a + b)
        minf = a * taum


class h(State):
    USEQ10()

    PARAMETER(
        aq10=2.3,
        bq10=21.0,
        cq10=10.0,
        hshift=-66.0,
        ha1=-0.015,
        ha2=6.0,
        hb1=-0.015,
        hb2=6.0,
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return (
            self.q10()
            * self.ha1
            * (v - self.hshift)
            / (1 - exp((v - self.hshift) / self.ha2))
        )

    def beta(self, v):
        return (
            self.q10()
            * self.hb1
            * (-v + self.hshift)
            / (1 - exp((-v + self.hshift) / self.hb2))
        )

    def breakpoint(self, v):
        v = torch.where(v == self.hshift, v + 0.0001, v)
        a = self.alpha(v)
        b = self.beta(v)
        tauh = 1 / (a + b)
        hinf = a * tauh


class nata_t(Mechanism):
    STATE(m, h)

    USEION("na", read=["ena"], write=["ina"])

    PARAMETER(gbar=0.00001)

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)
