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
        minf = (1/(1+exp((v+97)/7.35)))**(1/3)
        taum = 0.5/(exp((v-42)/11.8)+exp(-1*(v+498)/66.6))
        taum = taum / self.q10()

    def inf(self, v):
        return (1/(1+exp((v+97)/7.35)))**(1/3)


class n(State):
    USEQ10()

    DERIVATIVE("n' = (ninf - n) / taun")
    ASSIGNED("ninf", "taun")

    PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        ninf = (1/(1+exp((v+94)/8.9)))**(1/3)
        taun = 0.5/(exp((v+25)/4.1)+exp(-1*(v+356)/32))
        taun = taun / self.q10()

    def inf(self, v):
        return (1/(1+exp((v+94)/8.9)))**(1/3)
    

class hcn(Mechanism):
    STATE(m, n)
    PARAMETER(gbar=0.0001, ekna=-30.0)

    USEION("k", read=["ek"], write=["ik"])
    USEION("na", read=["ena"], write=["ina"])

    def ik(self, v):
        g = self.gbar * (0.25*self.n + 0.75*self.m)
        return (self.ekna-self.ena)*g*(v-self.ek)/(self.ek-self.ena)
    
    def ina(self, v):
        g = self.gbar * (0.25*self.n + 0.75*self.m)
        return (self.ekna-self.ek)*g*(v-self.ena)/(self.ena-self.ek)