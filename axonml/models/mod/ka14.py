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
        minf = (1/(1+exp(-1*(v+25)/12)))**(1/3)
        taum = (.6 + 2748/(exp((v+128)/14.5)+exp(-1*(v+10)/8))+1.7/(1+exp((v+8.5)/10.65)))/2
        taum = taum / self.q10()

    def inf(self, v):
        return (1/(1+exp(-1*(v+25)/12)))**(1/3)
    

class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        hinf = .073 + 0.924/(1+exp((v+47)/4.75)) 
        tauh = 35 + 11.22/(exp((v+21.4)/9.48)+exp(-1*(v+155.3)/16.4))
        tauh = tauh / self.q10()

    def inf(self, v):
        return .073 + 0.924/(1+exp((v+47)/4.75))
    

class s(State):
    USEQ10()

    DERIVATIVE("s' = (sinf - s) / taus")
    ASSIGNED("sinf", "taus")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        sinf = .415 + .576/(1+exp((v+44.5)/5.93)) + .103/(1+exp(-1*(v)/18.37))
        taus = 1000*(3.97/(exp((v+35.1)/9.97)+exp(-1*(v+83.3)/18))+2.5/(1+exp(-1*(v+27.3)/7.11)))
        taus = taus / self.q10()

    def inf(self, v):
        return .415 + .576/(1+exp((v+44.5)/5.93)) + .103/(1+exp(-1*(v)/18.37))
    

class ka14(Mechanism):
    STATE(m, h, s)
    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ek)
