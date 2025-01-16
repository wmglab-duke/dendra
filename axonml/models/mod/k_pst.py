# The persistent component of the K current
# Voltage-gated K+ channels in layer 5 neocortical pyramidal
# neurons from young rats: subtypes and gradients
# Korngreen and Sakmann, J. Physiology, 2000


from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(aq10=2.3, bq10=21.0, cq10=10.0)

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def breakpoint(self, v):
        v = v + 10.0
        q10 = self.q10()
        minf = 1.0 / (1.0 + exp(-(v + 1.0) / 12.0))
        mtau_o = (1.25 + 13 * exp(-v * 0.026)) / q10
        mtau_e = (1.25 + 175.03 * exp(-v * -0.026)) / q10
        taum = torch.where(v < -50.0, mtau_e, mtau_o)

    def inf(self, v):
        v = v + 10.0
        return 1.0 / (1.0 + exp(-(v + 1.0) / 12.0))


class h(State):
    USEQ10()

    PARAMETER(aq10=2.3, bq10=21.0, cq10=10.0)

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def breakpoint(self, v):
        v = v + 10.0
        q10 = self.q10()
        hinf = 1.0 / (1.0 + exp(-(v + 54.0) / -11.0))
        tauh = (360 + (1010 + 24 * (v + 55)) * exp(-((v + 75) / 48) ^ 2)) / q10

    def inf(self, v):
        v = v + 10.0
        return 1.0 / (1.0 + exp(-(v + 54.0) / -11.0))


class k_pst(Mechanism):
    STATE(m, h)
    USEION("k", read=["ek"], write=["ik"])

    PARAMETER(gbar=0.00001)

    def ik(self, v):
        return self.gbar * self.m**2 * self.h * (v - self.ek)
