# Sweeney Channel
# Sweeney channel fast sodium channel for myelinated axon
# set for a resting potential of -80 mV


from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    PARAMETER(amA=49, amB=126, amC=0.363, amD=5.3, bmA=56.2, bmB=4.17)

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def alpha(self, v):
        return (self.amB + self.amC * v) / (1 + exp(-(self.amA + v) / self.amD))

    def beta(self, v):
        return (
            (1 / exp((v + self.bmA) / self.bmB))
            * (self.amB + self.amC * v)
            / (1 + exp(-(self.amA + v) / self.amD))
        )

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        taum = 1 / (a + b)
        minf = a * taum

    def inf(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        return a / (a + b)


class h(State):
    PARAMETER(ahA=56, ahB=15.6, bhA=10, bhB=74.5, bhC=5.0)

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def alpha(self, v):
        return (
            (1 / exp((v + self.bhB) / self.bhC))
            * self.ahB
            / (1 + exp(-(v + self.ahA) / self.bhA))
        )

    def beta(self, v):
        return self.ahB / (1 + exp(-(v + self.ahA) / self.bhA))

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        tauh = 1 / (a + b)
        hinf = a * tauh

    def inf(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        return a / (a + b)


class sweeney(Mechanism):
    STATE(m, h)

    PARAMETER(gnabar=1.445, gl=0.128, el=-80.01, ena=35.64)

    USEION("na", write=["ina"])
    NONSPECIFIC_CURRENT("il")

    def ina(self, v):
        return self.gnabar * self.m**2 * self.h * (v - self.ena)

    def il(self, v):
        return self.gl * (v - self.el)
