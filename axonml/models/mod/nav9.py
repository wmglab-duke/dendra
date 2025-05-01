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
        q10 = self.q10()
        minf = (1/(1+exp(-1*(v+51)/8.4)))**(1/3)
        taum = (0.3 + 1/(exp((v+1.3)/22)+exp(-1*(v+82)/10)))/2
        taum = taum / q10

    def inf(self, v):
        return (1/(1+exp(-1*(v+51)/8.4)))**(1/3)
    

class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        q10 = self.q10()
        hinf = 1/(1+exp((v+55)/11.5))
        tauh = 2 + 1/(exp((v-32)/14)+exp(-1*(v+157)/12))
        tauh = tauh / q10

    def inf(self, v):
        return 1/(1+exp((v+55)/11.5))
    

class s(State):
    USEQ10()

    DERIVATIVE("s' = (sinf - s) / taus")
    ASSIGNED("sinf", "taus")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        q10 = self.q10()
        sinf = 0.03 + 0.97/(1+exp((v+79)/6.87))
        taus = 1/(exp((v-195)/29)+exp(-1*(v+193)/12.4))+1165/(1+exp(-1*(v+40)/30))
        taus = taus / q10

    def inf(self, v):
        return 0.03 + 0.97/(1+exp((v+79)/6.87))
    

class nav9(Mechanism):
    STATE(m, h, s)
    PARAMETER(gbar=0.0)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)