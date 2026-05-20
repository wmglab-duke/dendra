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
    S.GLOBAL(am1=0.1, am2=4.0, ah1=0.07, ah2=1.0, an1=0.01, an2=0.125)

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 6.3) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        alpha_m = self.am1 * vtrap(-(v + 40), 10)
        beta_m = self.am2 * exp(-(v + 65) / 18)
        tot = alpha_m + beta_m
        mtau = 1 / (q10 * tot)
        minf = alpha_m / tot
        alpha_h = self.ah1 * exp(-(v + 65) / 20)
        beta_h = self.ah2 / (exp(-(v + 35) / 10) + 1)
        tot = alpha_h + beta_h
        htau = 1 / (q10 * tot)
        hinf = alpha_h / tot
        alpha_n = self.an1 * vtrap(-(v + 55), 10)
        beta_n = self.an2 * exp(-(v + 65) / 80)
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
        states = self.breakpoint(v, None)
        return {"m": states["minf"], "h": states["hinf"], "n": states["ninf"]}


class hh(M):
    """
    The Hodgkin-Huxley neuron model with sodium, potassium, and leak channels.
    This model includes the dynamics of the gating variables `m`, `h`, and `n`
    which represent the activation and inactivation of sodium and potassium channels.

    Parameters
    ----------
    gnabar : float
        Maximum sodium conductance (in mS/cm^2). Default is 0.12.
    gkbar : float
        Maximum potassium conductance (in mS/cm^2). Default is 0.036.
    gl : float
        Leak conductance (in mS/cm^2). Default is 0.0003.
    ena : float
        Sodium reversal potential (in mV). Default is 50.0.
    ek : float
        Potassium reversal potential (in mV). Default is -77.0.
    el : float
        Leak reversal potential (in mV). Default is -54.3.
    """

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
