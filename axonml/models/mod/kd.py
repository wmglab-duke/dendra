from ..mechanisms import *
from ..mechanisms.ops import *


class n(State):
    USEQ10()

    PARAMETER(
        V0p5n=14.62,
        S0p5n=-18.38,
        A_alphan=0.001265,
        B_alphan=14.273,
        C_alphan=-10.0,
        A_betan=0.125,
        B_betan=55.0,
        C_betan=-2.5,
        q10kdn=1.40,
        q10TempA=22.85,
        Q10TempB=10,
        shiftkd=3.0,
    )

    DERIVATIVE("n' = (ninf - n) / ntau")

    def calc_q10(self):
        return self.q10kdn ** ((self.q10TempA - self.temp) / self.q10TempB)

    def alpha(self, v):
        x = v + self.B_alphan
        return -self.A_alphan * exprelr(x, self.C_alphan)

    def beta(self, v):
        return self.A_betan * exp((v + self.B_betan) / self.C_betan)

    def export(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        ntau = self.q10() * (1.0 + (1.0 / (a + b)))
        ninf = 1.0 / (1.0 + exp((v + self.V0p5n + self.shiftkd) / self.S0p5n))
        return ninf, ntau

    def inf(self, v):
        return 1.0 / (1.0 + exp((v + self.V0p5n + self.shiftkd) / self.S0p5n))


class kd(Mechanism):
    USEION(n)
    PARAMETER(gbar=0.000180376)
    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.n * (v - self.ek)
