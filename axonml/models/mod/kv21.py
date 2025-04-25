from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        minf = (1/(1+exp((-1*(v - 1)/10))))**(1/3)
        taum = 2+1.2/(exp((v-15)/8.5)+exp(-1*(v+68)/15))
        taum = taum / self.q10()

    def inf(self, v):
        return (1/(1+exp((-1*(v - 1)/10))))**(1/3)
    

class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        hinf = 0.15 + .85/(1+exp(((v + 27)/7)))
        tauh = 45/(exp((v-10.2)/8)+exp(-1*(v+101)/8)) + 8200/(1+exp((-1*v/60)))
        tauh = tauh / self.q10()

    def inf(self, v):
        return 0.15 + .85/(1+exp(((v + 27)/7)))
    

class kv21(Mechanism):
    STATE(m, h)
    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ek)