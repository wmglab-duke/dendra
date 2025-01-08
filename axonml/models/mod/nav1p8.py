# m and h are from Sheets, 2007
# s and u are from Delmas


from ..mechanisms import *
from ..mechanisms.ops import *


class m(State):
    USEQ10()

    PARAMETER(aq10=2.5, bq10=22.0, cq10=10.0)

    DERIVATIVE("m' = (minf - m) / taum")
    ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def breakpoint(self, v):
        a = 2.85 - 2.839 * sigmoid((1.159 - v) / 13.95)
        b = 7.6205 * sigmoid((-46.463 - v) / 8.8289)
        s = 1 / (a + b)
        taum = self.q10() * s
        minf = a * s

    def inf(self, v):
        a = 2.85 - 2.839 * sigmoid((1.159 - v) / 13.95)
        b = 7.6205 * sigmoid((-46.463 - v) / 8.8289)
        return a / (a + b)


class h(State):
    USEQ10()

    PARAMETER(aq10=2.5, bq10=22.0, cq10=10.0)

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def breakpoint(self, v):
        hinf = sigmoid((-32.2 - v) / 4.0)
        tauh = self.q10() * (1.218 + 42.043 * exp(-((v + 38.1) ** 2) / (2 * 15.19**2)))

    def inf(self, v):
        return sigmoid((-32.2 - v) / 4.0)


class s(State):
    USEQ10()

    PARAMETER(aq10=2.5, bq10=22.0, cq10=10.0)

    DERIVATIVE("s' = (sinf - s) / taus")
    ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return 0.001 * 5.4203 * sigmoid((-79.816 - v) / 16.269)

    def beta(self, v):
        return 0.001 * 5.0757 * sigmoid((v + 15.968) / 11.542)

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        taus = self.q10() / (a + b)
        sinf = sigmoid((-45.0 - v) / 8.0)

    def inf(self, v):
        return sigmoid((-45.0 - v) / 8.0)


class u(State):
    USEQ10()

    PARAMETER(aq10=2.5, bq10=22.0, cq10=10.0)

    DERIVATIVE("u' = (uinf - u) / tauu")
    ASSIGNED("uinf", "tauu")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.temp - self.bq10) / self.cq10))

    def alpha(self, v):
        return 0.0002 * 2.0434 * sigmoid((-67.499 - v) / 19.51)

    def beta(self, v):
        return 0.0002 * 1.9952 * sigmoid((v + 30.963) / 14.792)

    def breakpoint(self, v):
        a = self.alpha(v)
        b = self.beta(v)
        tauu = self.q10() / (a + b)
        uinf = sigmoid((-51.0 - v) / 8.0)

    def inf(self, v):
        return sigmoid((-51.0 - v) / 8.0)


class nav1p8(Mechanism):
    STATE(m, h, s, u)

    PARAMETER(gbar=0.0)

    USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * self.u * (v - self.ena)
