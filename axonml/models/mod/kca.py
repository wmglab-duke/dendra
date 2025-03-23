# calcium-activated potassium current in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class c(State):
    USEQ10()

    PARAMETER(
        Q10kcac=2.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        A_alphac=750.0,
        B_alphac=-10.0,
        C_alphac=12.0,
        A_betac=0.05,
        B_betac=-10.0,
        C_betac=-60.0,
    )

    DERIVATIVE("c' = (cinf - c) / tauc")
    ASSIGNED("cinf", "tauc")

    def calc_q10(self):
        return self.Q10kcac ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def alpha(self, v):
        return self.A_alphac * self.cai * safe_exp((v + self.B_alphac) / self.C_alphac)

    def beta(self, v):
        return self.A_betac * safe_exp((v + self.B_betac) / self.C_betac)

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        cinf = a * s
        tauc = self.q10() * 4.5 * s


class kca(Mechanism):
    STATE(c)

    PARAMETER(gbar=0.000141471)

    USEION("k", read=["ek"], write=["ik"])
    USEION("ca", read=["cai"])

    def ik(self, v):
        return self.gbar * self.c * (v - self.ek)
