# fast, TTX-sensitive current in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(
        Q10nafm=2.30,
        Q10TempA=22.85,
        Q10TempB=10,
        shiftnaf=-17.5,
        V0p5m=41.35,
        S0p5m=-4.75,
        A_taum=0.75,
        B_taum=0.0635,
        C_taum=0.12,
        Vpm=-40.35,
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.Q10nafm ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        taum = self.q10() * (
            self.A_taum * exp(-((self.B_taum) ** 2) * (v - self.Vpm) ** 2) + self.C_taum
        )
        minf = sigmoid(-(v + self.V0p5m + self.shiftnaf) / self.S0p5m)

    def inf(self, v):
        return sigmoid(-(v + self.V0p5m + self.shiftnaf) / self.S0p5m)


class h(State):
    USEQ10()

    PARAMETER(
        Q10nafh=1.50,
        Q10TempA=22.85,
        Q10TempB=10,
        V0p5h=62.00,
        S0p5h=4.50,
        A_tauh=6.5,
        B_tauh=0.0295,
        C_tauh=0.55,
        Vph=-75.00,
        shiftnaf=-17.5,
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.Q10nafh ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        tauh = self.q10() * (
            self.A_tauh * exp(-((self.B_tauh) ** 2) * (v - self.Vph) ** 2) + self.C_tauh
        )
        hinf = sigmoid(-(v + self.V0p5h + self.shiftnaf) / self.S0p5h)

    def inf(self, v):
        return sigmoid(-(v + self.V0p5h + self.shiftnaf) / self.S0p5h)


class l(State):
    PARAMETER(V0p5l=40.0, S0p5l=1.5, A_taul=25.0, B_taul=4.5, C_taul=0.01, Vpl=-20.0)

    DERIVATIVE("l' = (linf - l) / taul")
    ASSIGNED("linf", "taul")

    def breakpoint(self, v):
        taul = (self.A_taul / (1.0 + exp((v + self.Vpl) / self.B_taul))) + self.C_taul
        linf = sigmoid(-(v + self.V0p5l) / self.S0p5l)

    def inf(self, v):
        return sigmoid(-(v + self.V0p5l) / self.S0p5l)


class naf(Mechanism):
    STATE(m, h, l)

    PARAMETER(gbar=0.068967142)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.l * (v - self.ena)
