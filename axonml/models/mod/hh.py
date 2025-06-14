from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    DERIVATIVE("m' = (minf - m) / mtau")
    ASSIGNED("minf", "mtau")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 6.3) / 10.0)

    def breakpoint(self, v):
        alpha = .1 * vtrap(-(v+40),10)
        beta =  4 * exp(-(v+65)/18)
        tot = alpha + beta
        mtau = 1/(self.q10() * tot)
        minf = alpha/tot
    
    def inf(self, v):
        alpha = .1 * vtrap(-(v+40),10)
        beta =  4 * exp(-(v+65)/18)
        tot = alpha + beta
        return alpha / tot


class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / htau")
    ASSIGNED("hinf", "htau")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 6.3) / 10.0)

    def breakpoint(self, v):
        alpha = 0.07 * exp(-(v+65)/20)
        beta = 1/(exp(-(v+35)/10) + 1)
        tot = alpha + beta
        htau = 1/(self.q10() * tot)
        hinf = alpha/tot
    
    def inf(self, v):
        alpha = 0.07 * exp(-(v+65)/20)
        beta = 1/(exp(-(v+35)/10) + 1)
        tot = alpha + beta
        return alpha / tot


class n(State):
    USEQ10()

    DERIVATIVE("n' = (ninf - n) / ntau")
    ASSIGNED("ninf", "ntau")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 6.3) / 10.0)

    def breakpoint(self, v):
        alpha = .01 * vtrap(-(v+55),10)
        beta = 0.125 * exp(-(v+65)/80)
        tot = alpha + beta
        ntau = 1/(self.q10() * tot)
        ninf = alpha/tot
    
    def inf(self, v):
        alpha = .01 * vtrap(-(v+55),10)
        beta = 0.125 * exp(-(v+65)/80)
        tot = alpha + beta
        return alpha / tot


class hh(Mechanism):

    STATE(m, h, n)
    PARAMETER(gnabar=.12, gkbar=.036, gl=.0003, ena=50.0, ek=-77.0, el=-54.3)

    NONSPECIFIC_CURRENT("il", "ina", "ik")

    def il(self, v):
        return self.gl * (v - self.el)

    def ina(self, v):
        gna = self.gnabar * self.m**3 * self.h
        return gna * (v - self.ena)

    def ik(self, v):
        gk = self.gkbar * self.n**4
        return gk * (v - self.ek)