# can is the high threshold, long-lasting calcium current in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class d(State):
    USEQ10()

    PARAMETER(
        Q10can=4.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftcan=-7.0,
        V0p5d=20.0,
        S0p5d=-4.5,
        A_taud=3.25,
        B_taud=0.042,
        C_taud=0.395,
        Vpd=-31.0,
    )

    DERIVATIVE("d' = (dinf - d) / taud")
    ASSIGNED("dinf", "taud")

    def calc_q10(self):
        return self.Q10can ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        taud = (
            self.q10() * self.A_taud * exp(-((self.B_taud) ** 2) * (v - self.Vpd) ** 2)
            + self.C_taud
        )
        dinf = 1.0 / (1.0 + exp((v + self.V0p5d + self.shiftcan) / self.S0p5d))

    def inf(self, v):
        return 1.0 / (1.0 + exp((v + self.V0p5d + self.shiftcan) / self.S0p5d))


class f1(State):
    USEQ10()

    PARAMETER(
        Q10can=4.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftcan=-7.0,
        V0p5f1=20.0,
        S0p5f1=25.0,
        A_tauf1=33.5,
        B_tauf1=0.0395,
        C_tauf1=5.0,
        Vpf1=-30.0,
    )

    DERIVATIVE("f1' = (f1inf - f1) / tauf1")
    ASSIGNED("f1inf", "tauf1")

    def calc_q10(self):
        return self.Q10can ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        tauf1 = (
            self.q10()
            * self.A_tauf1
            * exp(-((self.B_tauf1) ** 2) * (v - self.Vpf1) ** 2)
            + self.C_tauf1
        )
        f1inf = 1.0 / (1.0 + exp((v + self.V0p5f1 + self.shiftcan) / self.S0p5f1))

    def inf(self, v):
        return 1.0 / (1.0 + exp((v + self.V0p5f1 + self.shiftcan) / self.S0p5f1))


class f2(State):
    USEQ10()

    PARAMETER(
        Q10can=4.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftcan=-7.0,
        V0p5f2=40.0,
        S0p5f2=10.0,
        A_tauf2=225.0,
        B_tauf2=0.0275,
        C_tauf2=75.0,
        Vpf2=-40.0,
    )

    DERIVATIVE("f2' = (f2inf - f2) / tauf2")
    ASSIGNED("f2inf", "tauf2")

    def calc_q10(self):
        return self.Q10can ** ((self.Q10TempA - self.temp) / self.Q10TempB)

    def breakpoint(self, v):
        tauf2 = (
            self.q10()
            * self.A_tauf2
            * exp(-((self.B_tauf2) ** 2) * (v - self.Vpf2) ** 2)
            + self.C_tauf2
        )
        f2inf = 1.0 / (1.0 + exp((v + self.V0p5f2 + self.shiftcan) / self.S0p5f2))

    def inf(self, v):
        return 1.0 / (1.0 + exp((v + self.V0p5f2 + self.shiftcan) / self.S0p5f2))


class can(Mechanism):
    STATE(d, f1, f2)

    PARAMETER(gbar=0.000106103, R=8314.0, z=2, ecaoffset=78.7, F=96500)

    USEION("ca", read=["cao", "cai"], write=["ica"])

    def ica(self, v):
        ecan = (
            self.R * ((self.temp + 273.15) / self.z / self.F * log(self.cao / self.cai))
            - self.ecaoffset
        )
        return self.gbar * self.d * (0.55 * self.f1 * 0.45 * self.f2) * (v - ecan)
