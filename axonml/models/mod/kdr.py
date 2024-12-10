# Borg-Graham type KDR channel; Borg-Graham 1987

from ..mechanisms import *
from ..mechanisms.ops import exp


class l(State):
    USEQ10()

    PARAMETER(
        {
            "zetal": 2.0,
            "gml": 1.0,
            "vhalfl": -61.0,
            "a0l": 0.001,
            "aq10": 3.0,
            "bq10": 30.0,
            "cq10": 10.0,
        }
    )

    DERIVATIVE("l' = (linf - l) / taul")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return exp(
            1e-3
            * self.zetal
            * (v - self.vhalfl)
            * 9.648e4
            / (8.315 * (273.16 + self.temp))
        )

    def beta(self, v):
        return exp(
            1e-3
            * self.zetal
            * self.gml
            * (v - self.vhalfl)
            * 9.648e4
            / (8.315 * (273.16 + self.temp))
        )

    def export(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        al = 1 + a
        linf = 1 / al
        taul = b / (self.q10() * self.a0l * al)
        return linf, taul

    def inf(self, v):
        return 1 / (1 + self.alpha(v))


class n(State):
    USEQ10()

    PARAMETER(
        {
            "zetan": -5.0,
            "gmn": 0.4,
            "vhalfn": -32.0,
            "a0n": 0.03,
            "aq10": 3.0,
            "bq10": 30.0,
            "cq10": 10.0,
        }
    )

    DERIVATIVE("n' = (ninf - n) / taun")

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return exp(
            1e-3
            * self.zetan
            * (v - self.vhalfn)
            * 9.648e4
            / (8.315 * (273.16 + self.temp))
        )

    def beta(self, v):
        return exp(
            1e-3
            * self.zetan
            * self.gmn
            * (v - self.vhalfn)
            * 9.648e4
            / (8.315 * (273.16 + self.temp))
        )

    def export(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        an = 1 + a
        ninf = 1 / an
        taun = b / (self.q10() * self.a0n * an)
        return ninf, taun

    def inf(self, v):
        return 1 / (1 + self.alpha(v))


class kdr(Mechanism):
    STATE(l, n)

    PARAMETER({"gkbar": 0.003, "ek": -77.0})

    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        l = self.states["l"]
        n = self.states["n"]
        return self.gkbar * n**3 * l * (v - self.ek)
