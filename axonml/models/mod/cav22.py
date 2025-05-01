from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)

    def breakpoint(self, v):
        minf = (1/(1 + exp(-1*(v + 4)/7.5)))**(1/3)
        taum = .1 + 0.5/(exp((v-3)/6.7)+exp(-1*(v+37)/13.5))
        taum = taum / self.q10()

    def inf(self, v):
        return (1/(1 + exp(-1*(v + 4)/7.5)))**(1/3)
    

class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)

    def breakpoint(self, v):
        hinf = 1/(1 + exp((v + 48)/7))
        tauh = 10/(exp((v-54)/23)+exp(-1*(v+150)/35))
        tauh = tauh / self.q10()

    def inf(self, v):
        return 1/(1 + exp((v + 48)/7))
    

class s(State):
    USEQ10()

    DERIVATIVE("s' = (sinf - s) / taus")
    ASSIGNED("sinf", "taus")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)

    def breakpoint(self, v):
        sinf = 1/(1 + exp((v + 81)/8.6))
        taus = 50 + 30/(exp((v-50)/26)+exp(-1*(v+150)/26))
        taus = taus / self.q10()

    def inf(self, v):
        return 1/(1 + exp((v + 81)/8.6))
    

class cav22(Mechanism):
    STATE(m, h, s)
    PARAMETER(gbar=0.0001)

    USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.eca)
