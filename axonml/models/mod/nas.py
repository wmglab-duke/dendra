# slower, TTX-insensitive current in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(
        Q10nasm=2.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftnas=-20.0,
        V0p5m=20.35,
        S0p5m=-4.45,
        A_taum=1.50,
        B_taum=0.0595,
        C_taum=0.15,
        Vpm=-20.35,
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return self.Q10nasm ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        taum = self.q10() * (
            self.A_taum * exp(-((self.B_taum) ** 2) * (v - self.Vpm) ** 2) + self.C_taum
        )
        minf = sigmoid(-(v + self.V0p5m + self.shiftnas) / self.S0p5m)

    def inf(self, v):
        return sigmoid(-(v + self.V0p5m + self.shiftnas) / self.S0p5m)


class h(State):
    USEQ10()

    PARAMETER(
        Q10nash=1.50,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftnas=-20.0,
        V0p5h=18.00,
        S0p5h=4.50,
        A_tauh=4.95,
        B_tauh=0.0335,
        C_tauh=0.75,
        Vph=-20.00,
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return self.Q10nash ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        tauh = self.q10() * (
            self.A_tauh * exp(-((self.B_tauh) ** 2) * (v - self.Vph) ** 2) + self.C_tauh
        )
        hinf = sigmoid(-(v + self.V0p5h + self.shiftnas) / self.S0p5h)

    def inf(self, v):
        return sigmoid(-(v + self.V0p5h + self.shiftnas) / self.S0p5h)


class nas(Mechanism):
    STATE(m, h)

    PARAMETER(gbar=0.001043349)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)
