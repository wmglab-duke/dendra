from ..mechanisms import Mechanism as M
from ..mechanisms import State as S
from ..mechanisms import *
from ..mechanisms.ops import exp, vtrap


class mhn(S):
    has_q10 = True

    S.STATE("m", "h", "n")
    S.DERIVATIVE(
        "m' = (minf - m) / mtau", "h' = (hinf - h) / htau", "n' = (ninf - n) / ntau"
    )
    S.ASSIGNED("minf", "mtau", "hinf", "htau", "ninf", "ntau")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 6.3) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        alpha_m = 0.1 * vtrap(-(v + 40), 10)
        beta_m = 4 * exp(-(v + 65) / 18)
        tot = alpha_m + beta_m
        mtau = 1 / (q10 * tot)
        minf = alpha_m / tot
        alpha_h = 0.07 * exp(-(v + 65) / 20)
        beta_h = 1 / (exp(-(v + 35) / 10) + 1)
        tot = alpha_h + beta_h
        htau = 1 / (q10 * tot)
        hinf = alpha_h / tot
        alpha_n = 0.01 * vtrap(-(v + 55), 10)
        beta_n = 0.125 * exp(-(v + 65) / 80)
        tot = alpha_n + beta_n
        ntau = 1 / (q10 * tot)
        ninf = alpha_n / tot
        return {
            "mtau": mtau,
            "minf": minf,
            "htau": htau,
            "hinf": hinf,
            "ntau": ntau,
            "ninf": ninf,
        }

    def inf(self, v):
        states = self.breakpoint(v)
        return {"m": states["minf"], "h": states["hinf"], "n": states["ninf"]}


class hh(M):
    M.STATE(mhn)
    M.GLOBAL(gnabar=0.12, gkbar=0.036, gl=0.0003, ena=50.0, ek=-77.0, el=-54.3)

    M.NONSPECIFIC_CURRENT("il", "ina", "ik")

    def il(self, v):
        return self.gl * (v - self.el)

    def ina(self, v):
        gna = self.gnabar * self.m**3 * self.h
        return gna * (v - self.ena)

    def ik(self, v):
        gk = self.gkbar * self.n**4
        return gk * (v - self.ek)
