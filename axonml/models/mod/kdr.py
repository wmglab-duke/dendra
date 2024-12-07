# Borg-Graham type KDR channel; Borg-Graham 1987

from ..mechanisms import (
    Mechanism, State, PARAMETER, STATE, USEQ10
)
from ..mechanisms.ops import exp


class l(State):

    USEQ10()

    PARAMETER({
        'zetal': 2.0,
        'gml': 1.0,
        'vhalfl': -61.0,
        'a0l': 0.001,
        'aq10': 3.0,
        'bq10': 30.0,
        'cq10': 10.0
    })

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return exp(1e-3 * self.zetal * (v - self.vhalfl) * 9.648e4 / (8.315 * (273.16 + self.temp)))
    
    def beta(self, v):
        return exp(1e-3 * self.zetal * self.gml * (v - self.vhalfl) * 9.648e4 / (8.315 * (273.16 + self.temp)))
    
    def advance(self, l, v, dt):
        al = self.alpha(v)
        bl = self.beta(v)
        al_ = 1 + al
        inf = 1 / al_
        l_tau_inv = (self.q10() * self.a0l * al_) / bl
        return self.cnexp(l, inf, l_tau_inv, dt)
    
    def inf(self, v):
        return 1 / (1 + self.alpha(v))
    

class n(State):

    USEQ10()

    PARAMETER({
        'zetan': -5.0,
        'gmn': 0.4,
        'vhalfn': -32.0,
        'a0n': 0.03,
        'aq10': 3.0,
        'bq10': 30.0,
        'cq10': 10.0
    })

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        return exp(1e-3 * self.zetan * (v - self.vhalfn) * 9.648e4 / (8.315 * (273.16 + self.temp)))
    
    def beta(self, v):
        return exp(1e-3 * self.zetan * self.gmn * (v - self.vhalfn) * 9.648e4 / (8.315 * (273.16 + self.temp)))
    
    def advance(self, n, v, dt):
        an = self.alpha(v)
        bn = self.beta(v)
        an_ = 1 + an
        inf = 1 / an_
        n_tau_inv = (self.q10() * self.a0n * an_) / bn
        return self.cnexp(n, inf, n_tau_inv, dt)

    def inf(self, v):
        return 1 / (1 + self.alpha(v))
    

class kdr(Mechanism):

    STATE(l, n)

    PARAMETER({
        'gkbar': 0.003,
        'ek': -77.0
    })

    def i(self, v):
        l = self.states['l']
        n = self.states['n']
        return self.gkbar * n**3 * l * (v - self.ek)