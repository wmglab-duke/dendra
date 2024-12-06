import torch

from ..mechanisms import (
    Mechanism, State, PARAMETER, STATE, INITIAL, USEQ10
)
from ..mechanisms.ops import expit, exprelr


class m(State):

    USEQ10()

    PARAMETER({
        "amA": 1.86,
        "amB": 21.4,
        "amC": 10.3,
        "bmA": 0.086,
        "bmB": 25.7,
        "bmC": 9.16,
        "aq10_1": 2.2,
        "bq10": 20.0,
        "cq10": 10.0
    })

    def calc_q10(self):
        return self.aq10_1 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        x = - (v + self.amB)
        return self.amA * exprelr(x, self.amC)
    
    def beta(self, v):
        x = (v + self.bmB)
        return self.bmA * exprelr(x, self.bmC)
    
    def advance(self, m, v, dt):
        am = self.alpha(v)
        bm = self.beta(v)
        m_tau_inv = am + bm
        m_inf = am / m_tau_inv
        return self.cnexp(m, m_inf, self.q10() * m_tau_inv, dt)


class p(State):

    USEQ10()

    PARAMETER({
        "ampA": 0.01,
        "ampB": 27.0,
        "ampC": 10.2,
        "bmpA": 0.00025,
        "bmpB": 34.0,
        "bmpC": 10.0,
        "pq10_1": 2.2,
        "bq10": 20.0,
        "cq10": 10.0
    })

    def calc_q10(self):
        return self.pq10_1 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        x = - (v + self.ampB)
        return self.ampA * exprelr(x, self.ampC)
    
    def beta(self, v):
        x = (v + self.bmpB)
        return self.bmpA * exprelr(x, self.bmpC)
    
    def advance(self, p, v, dt):
        amp = self.alpha(v)
        bmp = self.beta(v)
        p_tau_inv = amp + bmp
        p_inf = amp / p_tau_inv
        return self.cnexp(p, p_inf, self.q10() * p_tau_inv, dt)
    

class h(State):

    USEQ10()

    PARAMETER({
        "ahA": 0.062,
        "ahB": 114.0,
        "ahC": 11.0,
        "bhA": 2.3,
        "bhB": 31.8,
        "bhC": 13.4,
        "aq10_2": 2.9,
        "bq10": 20.0,
        "cq10": 10.0
    })

    def calc_q10(self):
        return self.aq10_2 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        x = (v + self.ahB)
        return self.ahA * exprelr(x, self.ahC)
    
    def beta(self, v):
        return self.bhA * expit((v + self.bhB) / self.bhC)
    
    def advance(self, h, v, dt):
        ah = self.alpha(v)
        bh = self.beta(v)
        h_tau_inv = ah + bh
        h_inf = ah / h_tau_inv
        return self.cnexp(h, h_inf, self.q10() * h_tau_inv, dt)
    

class s(State):

    USEQ10()

    PARAMETER({
        "asA": 0.3,
        "asB": -27.0,
        "asC": -5.0,
        "bsA": 0.03,
        "bsB": 10.0,
        "bsC": -1.0,
        "aq10_3": 3.0,
        "bq10": 36.0,
        "cq10": 10.0,
        "vtraub": -80.0
    })

    def calc_q10(self):
        return self.aq10_3 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        b = self.asA * expit((self.vtraub - v - self.asB) / self.asC)
        return b
    
    def beta(self, v):
        b = self.bsA * expit((self.vtraub - v - self.bsB) / self.bsC)
        return b
    
    def advance(self, s, v, dt):
        as_ = self.alpha(v)
        bs = self.beta(v)
        s_tau_inv = as_ + bs
        s_inf = as_ / s_tau_inv
        return self.cnexp(s, s_inf, self.q10() * s_tau_inv, dt)
    

class Axnode_Myel(Mechanism):
    
    STATE(m, p, h, s)
    
    PARAMETER({
        'gnabar': 3.0,
        'gnapbar': 0.01,
        'gkbar': 0.08,
        'gl': 0.007,
        'ena': 50.0,
        'ek': -90.0,
        'el': -90.0
    })
    
    INITIAL({
        'm': 0.0732093,
        'h': 0.62069505,
        'p': 0.20260409,
        's': 0.04302994
    })

    def i(self, v):

        # -- gating variables --
        m = self.states['m']
        h = self.states['h']
        p = self.states['p']
        s = self.states['s']

        # -- current --
        current = (
            (self.gnabar * m * m * m * h * (v - self.ena))
            + 
            (self.gnapbar * p * p * p * (v - self.ena))
            + 
            (self.gkbar * s * (v - self.ek))
            + 
            (self.gl * (v - self.el))
        )
        
        return current