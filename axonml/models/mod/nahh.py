# From Traub & Miles "Neuronal networks of the hippocampus" (1991)
# Cummins et al. (2007), Sheets et al. (2007)

from ..mechanisms import Mechanism, State, PARAMETER, STATE, INITIAL, USEQ10
from ..mechanisms.ops import exprelr, exp, expit


class m(State):
    USEQ10()

    PARAMETER(
        {
            "am1": 0.32,
            "am2": 13.1,
            "am3": 4.0,
            "bm1": 0.28,
            "bm2": 40.1,
            "bm3": 5.0,
            "aq10": 3.0,
            "bq10": 30.0,
            "cq10": 10.0,
            "mshift": -6.0,
        }
    )

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        v = v + 65.0 + self.mshift
        return self.am1 * exprelr(self.am2 - v, self.am3)

    def beta(self, v):
        v = v + 65.0 + self.mshift
        return self.bm1 * exprelr(v - self.bm2, self.bm3)

    def advance(self, m, v, dt):
        am = self.alpha(v)
        bm = self.beta(v)
        m_tau_inv = am + bm
        m_inf = am / m_tau_inv
        return self.cnexp(m, m_inf, self.q10() * m_tau_inv, dt)


class h(State):
    USEQ10()

    PARAMETER(
        {
            "ah1": 0.128,
            "ah2": 17.0,
            "ah3": 18.0,
            "bh1": 4.0,
            "bh2": 40.0,
            "bh3": 5.0,
            "aq10": 3.0,
            "bq10": 30.0,
            "cq10": 10.0,
            "hshift": 6.0,
        }
    )

    def calc_q10(self):
        return self.aq10 ** ((self.temp - self.bq10) / self.cq10)

    def alpha(self, v):
        v = v + 65.0 + self.hshift
        return self.ah1 * exp((self.ah2 - v) / self.ah3)

    def beta(self, v):
        v = v + 65.0 + self.hshift
        return self.bh1 * expit((v - self.bh2) / self.bh3)

    def advance(self, h, v, dt):
        ah = self.alpha(v)
        bh = self.beta(v)
        h_tau_inv = ah + bh
        h_inf = ah / h_tau_inv
        return self.cnexp(h, h_inf, self.q10() * h_tau_inv, dt)


class nahh(Mechanism):
    STATE(m, h)

    PARAMETER({"gnabar": 0.3, "ena": 50.0})

    def i(self, v):
        m = self.states["m"]
        h = self.states["h"]
        return self.gnabar * m**3 * h * (v - self.ena)
