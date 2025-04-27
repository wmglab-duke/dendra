from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        q10 = self.q10()
        minf = (1/(1+exp(-1*(v+4)/7)))**(1/3)
        taum = 0.03 + .5/(exp((v)/12)+exp(-1*(v+29)/18))
        taum = taum / q10 / 2

    def inf(self, v):
        return (1/(1+exp(-1*(v+4)/7)))**(1/3)
    

class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        q10 = self.q10()
        hinf = 1/(1+exp((v+28)/4.56))
        tauh = 2.63+250/(exp((v+36)/7.7)+exp(-1*(v)/16.3)) + 1.4/(1+exp(-1*(v+0.6)/2.95))
        tauh = tauh / q10

    def inf(self, v):
        return 1/(1+exp((v+28)/4.56))
    

class s(State):
    USEQ10()

    DERIVATIVE("s' = (sinf - s) / taus")
    ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        q10 = self.q10()
        sinf = 1/(1+exp((v+50)/7.5))
        taus = 34/(exp((v+2)/16)+exp(-1*(v+108)/8)) + 160/(1+exp(-1*(v+110)/75))
        taus = taus / q10

    def inf(self, v):
        return 1/(1+exp((v+50)/7.5))
    

class newnav8(Mechanism):
    STATE(m, h, s)
    PARAMETER(gbar=0.0)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)