from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def breakpoint(self, v):
        taum = 0.2 * 20.000 / (1 + exp(((v - (-46.560)) / (-44.140))))
        minf = 1 / (1 + exp(((v - (18.700)) / (-9.700))))

    def inf(self, v):
        return 1 / (1 + exp(((v - (18.700)) / (-9.700))))


class skv31(Mechanism):
    STATE(m)

    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m * (v - self.ek)
