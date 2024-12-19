# fast, TTX-sensitive current in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(
        Q10nafm=2.30,
        Q10TempA=22,
        Q10TempB=10,
        V0p5m=-31.62,
        S0p5m=6.98,
        A_taum=1.15,
        B_taum=0.06,
        C_taum=0.21,
        Vpm=-40.0,
    )

    DERIVATIVE("m'= (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.Q10nafm ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        taum = (
            self.q10() * self.A_taum * exp(-((self.B_taum) ** 2) * (v - self.Vpm) ** 2)
            + self.C_taum
        )
        minf = sigmoid((v - self.V0p5h) / self.S0p5h)

    def inf(self, v):
        return sigmoid((v - self.V0p5m) / self.S0p5m)


class h(State):
    USEQ10()

    PARAMETER(
        Q10nafh=1.50,
        Q10TempA=22,
        Q10TempB=10,
        V0p5h=-65.99,
        S0p5h=-5.97,
        A_tauh=18.0,
        B_tauh=0.043,
        C_tauh=1.35,
        Vph=-62.5,
    )

    DERIVATIVE("h'= (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.Q10nafh ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        tauh = (
            self.q10() * self.A_tauh * exp(-((self.B_tauh) ** 2) * (v - self.Vph) ** 2)
            + self.C_tauh
        )
        hinf = sigmoid((v - self.V0p5h) / self.S0p5h)

    def inf(self, v):
        return sigmoid((v - self.V0p5h) / self.S0p5h)


class naf97mean(Mechanism):
    STATE(m, h)

    PARAMETER(gbar=0.068967142)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)
