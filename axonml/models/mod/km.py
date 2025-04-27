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
        minf = (1/(1+exp((-1*(v + 44)/6.4))))**(1/3)
        taum = 104/(exp((v-7)/20)+exp(-1*(v+32)/20)) + 30/(1+exp((-1*(v + 40)/80)))
        taum = taum / q10 / 2

    def inf(self, v):
        return (1/(1+exp((-1*(v + 44)/6.4))))**(1/3)
    

class n(State):
    USEQ10()

    DERIVATIVE("n' = (ninf - n) / taun")
    ASSIGNED("ninf", "taun")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)
    
    def breakpoint(self, v):
        q10 = self.q10()
        ninf = (1/(1+exp((-1*(v + 32)/9.2))))**(1/3)
        taun = 15 + 62/(exp((v-13)/20)+exp(-1*(v+90)/20)) + 50/(1+exp((-1*(v + 50)/8)))
        taun = taun / q10 / 2

    def inf(self, v):
        return (1/(1+exp((-1*(v + 32)/9.2))))**(1/3)
    

class km(Mechanism):
    STATE(m, n)
    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * (0.25 * self.m**3 + 0.75 * self.n**3) * (v - self.ek)