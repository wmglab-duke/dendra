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
        mb2=6.0
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)
    
    def alpha(self, v):
        return self.ma1 * exprelr(self.mshift - v, self.ma2)
    
    def beta(self, v):
        return self.mb1 * exprelr(v - self.mshift, self.mb2)
    
    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        taum = 6.0 / (self.q10() * (a + b))
        minf = 1.0/(1+exp((v- -52.6)/-4.6))

    def inf(self, v):
        return 1.0/(1+exp((v- -52.6)/-4.6))
    

class h(State):

    USEQ10()

    PARAMETER(
        aq10 = 2.3,
        bq10 = 21.0,
        cq10 = 10.0,
        ha2 = 4.63,
        hb2 = 2.63
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def breakpoint(self, v):
        hinf = 1.0/(1+exp((v- -48.8)/10))
        a = 2.88e-6 * exprelr(v + 17.0, self.ha2)
        b = 6.94e-6 * exprelr(64.6 - v, self.hb2)
        tauh = 1 / (self.q10() * (a + b))

    def inf(self, v):
        return 1.0/(1+exp((v- -48.8)/10))
    

class nap_et2(Mechanism):
    STATE(m, h)

    USEION("na", read=["ena"], write=["ina"])

    PARAMETER(gbar=0.0001)

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)