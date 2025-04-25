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
        pca = log10(self.cai) - 3
        v12 = -50.0*pca-232.0
        minf = 1/(1+exp(-1*(v - v12)/24))
        taum = 1/(exp((v+(58*pca)+303)/(3.2*pca))+exp(-1*(v+(107*pca)+453)/(6.8*pca)))+0.4
        taum = taum / q10

    def inf(self, v):
        pca = log10(self.cai) - 3
        v12 = -50.0*pca-232.0
        return 1/(1+exp(-1*(v - v12)/24))


class h(State):
    USEQ10()

    DERIVATIVE("h' = (hinf - h) / tauh")
    ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 3.0 ** ((self.temp - 22.0) / 10.0)

    def breakpoint(self, v):
        q10 = self.q10()
        pca = log10(self.cai) - 3
        vh12 =  -8*pca+35
        hinf= 1/(1+exp((v - vh12)/47))
        tauh = 1/(exp((v+(3*pca)+100)/(3*pca))+exp(-1*(v+(191*pca)+600)/(17*pca)))
        tauh = tauh / q10

    def inf(self, v):
        pca = log10(self.cai) - 3
        vh12 =  -8*pca+35
        return 1/(1+exp((v - vh12)/47))


class bk(Mechanism):
    STATE(m, h)
    PARAMETER(gbar=0.0001)

    USEION("k", read=["ek"], write=["ik"])
    USEION("ca", read=["cai"])

    def ik(self, v):
        return self.gbar * self.m * self.h * (v - self.ek)
