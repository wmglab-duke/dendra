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
        minf = (1/(1+exp(-1*(v-24)/17)))**(1/3)
        taum = (1/(exp((v-39)/13)+exp(-1*(v+134)/47)))/2
        taum = taum / self.q10()

    def inf(self, v):
        return (1/(1+exp(-1*(v-24)/17)))**(1/3)
    

class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        hinf = 1/(1+exp((v+31)/12))
        tauh = 15 + 1/(exp((v-60)/15)+exp(-1*(v+300)/27))
        tauh = tauh / self.q10()

    def inf(self, v):
        return 1/(1+exp((v+31)/12))
    

class ka34(Mechanism):
    STATE(m, h)
    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ek)
