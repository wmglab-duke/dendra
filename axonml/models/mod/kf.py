# The steady state curves are collected from Winkelman 2005
# The time constant is from Gold 1996 and Safron 1996


from ..mechanisms import *
from ..mechanisms.ops import *


class h(State):
    USEQ10()

    PARAMETER(
        aq10=3.3, 
        bq10=23, 
        cq10=10, 
        vhh=-49.9, 
        kh=4.6, 
        shift=-15
    )

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def breakpoint(self, v):
        hinf = sigmoid((v - self.vhh + self.shift) / -self.kh)
        tauh = self.calc_q10() * (20 + 50 * exp(-((v+40)**2)/(2*40**2)))
        tauh = torch.where(tauh < 5, 5.0, tauh)

    def inf(self, v):
        return sigmoid((v - self.vhh + self.shift) / -self.kh)
    

class m(State):
    USEQ10()

    PARAMETER(
        aq10=3.3, 
        bq10=23, 
        cq10=10,
        vhm=-5.4,
        km=16.4
    )

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return 0.00395 * exp((v + 30) / 40)
    
    def beta(self, v):
        return 0.00395 * exp(-(v + 30) / 20)
    
    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        finf = sigmoid((v + 30) / 6)
        tauf = self.calc_q10() / (a + b)

    def inf(self, v):
        return sigmoid((v + 30) / 6)
    

class ks(Mechanism):
    STATE(s, f)

    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * (0.25 * self.s + 0.75 * self.f) * (v - self.ek)