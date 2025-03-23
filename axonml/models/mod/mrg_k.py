# McIntyre, Richardson, Grill 2002

from ..mechanisms import *
from ..mechanisms.ops import expit


class s(State):
    USEQ10()

    PARAMETER(
        asA=0.3,
        asB=-27.0,
        asC=-5.0,
        bsA=0.03,
        bsB=10.0,
        bsC=-1.0,
        aq10_3=3.0,
        bq10=36.0,
        cq10=10.0,
        vtraub=-80.0,
    )

    DERIVATIVE("s' = (sinf - s) / stau")
    ASSIGNED("sinf", "stau")

    def calc_q10(self):
        return self.aq10_3 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        b = self.q10() * self.asA * expit((self.vtraub - v - self.asB) / self.asC)
        return b

    def beta(self, v):
        b = self.q10() * self.bsA * expit((self.vtraub - v - self.bsB) / self.bsC)
        return b

    def breakpoint(self, v):
        as_ = self.alpha(v)
        bs = self.beta(v)
        stau = 1 / (as_ + bs)
        sinf = as_ * stau


class mrg_k(Mechanism):
    STATE(s)

    PARAMETER(gkbar=0.08, ek=-90.0)

    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.gkbar * self.s * (v - self.ek)
