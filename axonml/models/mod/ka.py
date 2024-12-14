from ..mechanisms import *
from ..mechanisms.ops import *


class p(State):
    USEQ10()
    PARAMETER(
        shiftka=3.0,
        V0p5p=28.0,
        S0pp5p=-28.0,
        A_taup=5.0,
        B_taup=0.022,
        C_taup=2.5,
        Vpp=-65.0,
        q10ka=1.93,
        q10TempA=22.85,
        q10TempB=10.0,
    )

    DERIVATIVE("p' = (pinf - p) / taup")

    def calc_q10(self):
        return self.q10ka ** ((self.q10TempA - self.temp) / self.q10TempB)

    def export(self, v):
        taup = (
            self.q10() * self.A_taup * exp(-((self.B_taup) ** 2) * (v - self.Vpp) ** 2)
            + self.C_taup
        )
        pinf = 1.0 / (1.0 + exp((v + self.V0p5p + self.shiftka) / self.S0pp5p))
        return pinf, taup

    def inf(self, v):
        return 1.0 / (1.0 + exp((v + self.V0p5p + self.shiftka) / self.S0pp5p))


class q(State):
    USEQ10()
    PARAMETER(
        shiftka=3.0,
        V0p5q=58.0,
        S0p5q=7.9,
        A_tauq=100.0,
        B_tauq=0.035,
        C_tauq=10.5,
        Vpq=-30.0,
        q10ka=1.93,
        q10TempA=22.85,
        q10TempB=10.0,
    )

    DERIVATIVE("q' = (qinf - q) / tauq")

    def calc_q10(self):
        return self.q10ka ** ((self.q10TempA - self.temp) / self.q10TempB)

    def export(self, v):
        tauq = (
            self.q10() * self.A_tauq * exp(-((self.B_tauq) ** 2) * (v - self.Vpq) ** 2)
            + self.C_tauq
        )
        qinf = 1.0 / (1.0 + exp((v + self.V0p5q + self.shiftka) / self.S0p5q))
        return qinf, tauq

    def inf(self, v):
        return 1.0 / (1.0 + exp((v + self.V0p5q + self.shiftka) / self.S0p5q))


class ka(Mechanism):
    STATE(p, q)

    PARAMETER(gbar=0.000141471)

    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.p**2 * self.q * (v - self.ek)
