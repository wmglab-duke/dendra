# slower, TTX-insensitive current in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(
        Q10nasm=2.30,
        Q10TempA=22,
        Q10TempB=10,
        V0p5m=-11.29,
        S0p5m=5.54,
        A_taum=1.45,
        B_taum=0.058,
        C_taum=0.26,
        Vpm=-14.5,
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.Q10nasm ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        taum = (
            self.q10() * self.A_taum * exp(-((self.B_taum) ** 2) * (v - self.Vpm) ** 2)
            + self.C_taum
        )
        minf = sigmoid((v - self.V0p5m) / self.S0p5m)

    def inf(self, v):
        return sigmoid((v - self.V0p5m) / self.S0p5m)


class h(State):
    USEQ10()

    PARAMETER(
        Q10nash=1.50,
        Q10TempA=22,
        Q10TempB=10,
        V0p5h=-31.00,
        S0p5h=-5.20,
        A_tauh=10.75,
        B_tauh=0.067,
        C_tauh=3.15,
        Vph=-13.5,
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.Q10nash ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        tauh = (
            self.q10() * self.A_tauh * exp(-((self.B_tauh) ** 2) * (v - self.Vph) ** 2)
            + self.C_tauh
        )
        hinf = sigmoid((v - self.V0p5h) / self.S0p5h)

    def inf(self, v):
        return sigmoid((v - self.V0p5h) / self.S0p5h)


class nas97mean(Mechanism):
    STATE(m, h)

    PARAMETER(gbar=0.001043349)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)
