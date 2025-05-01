from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 21.0) / 10.0)
    
    def breakpoint(self, v):
        minf = (1/(1+exp(-1*(v+25)/7)))**(1/3)
        taum = (0.07/(exp((v-11.4)/14)+exp(-1*(v+61)/9.4)) + .1/(1+exp(-1*(v+5.5)/8))) / self.q10() / 2

    def inf(self, v):
        return (1/(1+exp(-1*(v+25)/7)))**(1/3)
    

class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 21.0) / 10.0)
    
    def breakpoint(self, v):
        hinf = 1/(1+exp((v+79)/7))
        tauh = (1/(exp(-1*(v+131)/9.5) + exp((v+5.7)/12.4))) / self.q10()

    def inf(self, v):
        return 1/(1+exp((v+79)/7))
    

class nav7(Mechanism):
    STATE(m, h)
    PARAMETER(gbar=0.12)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)