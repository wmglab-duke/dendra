from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        minf = (1/(1 + exp(-1*(v - 1.75)/10)))**(1/3)
        taum = 0.25 + 0.5/(exp((v - 15)/5) + exp(-1*(v + 27)/12))
        taum = taum / self.q10()

    def inf(self, v):
        return (1/(1 + exp(-1*(v - 1.75)/10)))**(1/3)
    

class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        hinf = 1/(1 + exp((v - 10)/8))
        tauh = 1.4/(exp((v - 145)/30) + exp(-1*(v + 150)/16.4))
        tauh = tauh / self.q10()

    def inf(self, v):
        return 1/(1 + exp((v - 10)/8))
    

class cav12(Mechanism):
    STATE(m, h)
    PARAMETER(gbar=0.0001)

    USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.eca)
