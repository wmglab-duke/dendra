# h channel
# from Tom Andersson Sensitivity studies of voltage-dpendent conductance in neurons
# Tom has build his model on Kouranova 2008 Hyoerpolarization -activated
# cyclic nuleotide-gated channel mRNA and protein expression in large
# versus small diameter dorsal root ganglion neurons: correlation with
# hyperpolarization-activated current


from ..mechanisms import *
from ..mechanisms.ops import *


class s(State):
    USEQ10()

    PARAMETER(aq10=3.0, bq10=22.0, cq10=10.0)

    DERIVATIVE("s' = (sinf - s) / taus")
    ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def breakpoint(self, v):
        sinf = sigmoid(-(v + 87.2) / 9.7)
        taus_a = 300.0 + 542.0 * exp((v + 25.0) / -20.0)
        taus_b = 2500.0 + 100.0 * exp((v + 240.0) / 50.0)
        taus = self.q10() * torch.where(v < -70.0, taus_b, taus_a)

    def inf(self, v):
        return sigmoid(-(v + 87.2) / 9.7)


class f(State):
    USEQ10()

    PARAMETER(aq10=3.0, bq10=22.0, cq10=10.0)

    DERIVATIVE("f' = (finf - f) / tauf")
    ASSIGNED("finf", "tauf")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def breakpoint(self, v):
        finf = sigmoid(-(v + 87.2) / 9.7)
        tauf_a = 140.0 + 50.0 * exp((v + 25.0) / -20.0)
        tauf_b = 250.0 + 12.0 * exp((v + 240.0) / 50.0)
        tauf = self.q10() * torch.where(v < -70.0, tauf_b, tauf_a)

    def inf(self, v):
        return sigmoid(-(v + 87.2) / 9.7)


class h(Mechanism):
    STATE(s, f)

    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])
    USEION("na", read=["ena"], write=["ina"])

    ASSIGNED("g")

    def initial(self):
        self.g = self.gbar * (0.5 * self.s + 0.5 * self.f)

    def breakpoint(self, v):
        self.g = self.gbar * (0.5 * self.s + 0.5 * self.f)

    def ina(self, v):
        return 0.5 * self.g * (v - self.ena)

    def ik(self, v):
        return 0.5 * self.g * (v - self.ek)
