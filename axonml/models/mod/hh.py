from ..mechanisms import *
from ..mechanisms._mechanism import Mechanism as M
from ..mechanisms._state import State as S
from ..mechanisms.ops import *


class m(S):
    has_q10 = True

    S.STATE('m')
    S.DERIVATIVE("m' = (minf - m) / mtau")
    S.ASSIGNED("minf", "mtau")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 6.3) / 10.0)

    def breakpoint(self, v):
        alpha = .1 * vtrap(-(v+40),10)
        beta =  4 * exp(-(v+65)/18)
        tot = alpha + beta
        mtau = 1/(self.q10() * tot)
        minf = alpha/tot
        return {"mtau": mtau, "minf": minf}
    
    def inf(self, v):
        return {'m': self.breakpoint(v)['minf']}


class h(S):
    has_q10 = True

    S.STATE('h')
    S.DERIVATIVE("h' = (hinf - h) / htau")
    S.ASSIGNED("hinf", "htau")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 6.3) / 10.0)

    def breakpoint(self, v):
        alpha = 0.07 * exp(-(v+65)/20)
        beta = 1/(exp(-(v+35)/10) + 1)
        tot = alpha + beta
        htau = 1/(self.q10() * tot)
        hinf = alpha/tot
        return {"htau": htau, "hinf": hinf}
    
    def inf(self, v):
        return {'h': self.breakpoint(v)['hinf']}


class n(S):
    has_q10 = True

    S.STATE('n')
    S.DERIVATIVE("n' = (ninf - n) / ntau")
    S.ASSIGNED("ninf", "ntau")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 6.3) / 10.0)

    def breakpoint(self, v):
        alpha = .01 * vtrap(-(v+55),10)
        beta = 0.125 * exp(-(v+65)/80)
        tot = alpha + beta
        ntau = 1/(self.q10() * tot)
        ninf = alpha/tot
        return {"ntau": ntau, "ninf": ninf}
    
    def inf(self, v):
        return {'n': self.breakpoint(v)['ninf']}


class hh(M):

    M.STATE(m, h, n)
    M.PARAMETER(gnabar=.12, gkbar=.036, gl=.0003, ena=50.0, ek=-77.0, el=-54.3)

    M.NONSPECIFIC_CURRENT("il", "ina", "ik")

    def il(self, v):
        return self.gl * (v - self.el)

    def ina(self, v):
        gna = self.gnabar * self.m**3 * self.h
        return gna * (v - self.ena)

    def ik(self, v):
        gk = self.gkbar * self.n**4
        return gk * (v - self.ek)