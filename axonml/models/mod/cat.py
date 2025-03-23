# Cat is the Low threshold, transient Ca current in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class d(State):
    USEQ10()

    PARAMETER(
        Q10catd=1.90,
        Q10TempA=22.85,
        Q10TempB=10.0,
        V0p5d=54.00,
        S0p5d=-5.75,
        A_taud=22.0,
        B_taud=0.052,
        C_taud=2.5,
        Vpd=-68.0,
        shiftcat=-7.0,
    )

    DERIVATIVE("d' = (dinf - d) / taud")
    ASSIGNED("taud", "dinf")

    def calc_q10(self):
        return self.Q10catd ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        taud = (
            self.A_taud * exp(-((self.B_taud) ** 2) * (v - self.Vpd) ** 2) + self.C_taud
        )
        taud = self.q10() * taud
        dinf = sigmoid(-(v + self.V0p5d + self.shiftcat) / self.S0p5d)

    def inf(self, v):
        return sigmoid(-(v + self.V0p5d + self.shiftcat) / self.S0p5d)


class f(State):
    USEQ10()

    PARAMETER(
        Q10catf=2.20,
        Q10TempA=22.85,
        Q10TempB=10.0,
        V0p5f=68.00,
        S0p5f=6.0,
        A_tauf=103.0,
        B_tauf=0.050,
        C_tauf=12.5,
        Vpf=-58.0,
        shiftcat=-7.0,
    )

    DERIVATIVE("f' = (finf - f) / tauf")
    ASSIGNED("tauf", "finf")

    def calc_q10(self):
        return self.Q10catf ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        tauf = (
            self.A_tauf * exp(-((self.B_tauf) ** 2) * (v - self.Vpf) ** 2) + self.C_tauf
        )
        tauf = self.q10() * tauf
        finf = sigmoid(-(v + self.V0p5f + self.shiftcat) / self.S0p5f)

    def inf(self, v):
        return sigmoid(-(v + self.V0p5f + self.shiftcat) / self.S0p5f)


class cat(Mechanism):
    STATE(d, f)

    PARAMETER(gbar=1.23787e-05, R=8314.0, z=2, ecaoffset=78.7, F=96500)

    USEION("ca", read=["cao", "cai"], write=["ica"])

    def ica(self, v):
        ecat = (
            self.R
            * (self.temp + 273.15)
            / self.z
            / self.F
            * log((self.cao + 1e-9) / (self.cai + 1e-9))
        ) - self.ecaoffset
        return self.gbar * self.d * self.f * (v - ecat)

    def conductance(self, v):
        return self.gbar * self.d * self.f
