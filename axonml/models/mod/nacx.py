from ..mechanisms import *
from ..mechanisms.ops import *


class nacx(Mechanism):
    PARAMETER(gbar=316.0, F=96500, R=8314, knaca=36e-9, dnaca=0.0036)
    USEION("na", read=["nai", "nao"], write=["ina"])
    USEION("ca", read=["cai", "cao"], write=["ica"])

    ASSIGNED("inaca", "q10", "FRT")

    def breakpoint(self, v):
        dfcain = self.nai**3*self.cao*exp(0.5*v*self.FRT)
        dfcaout = self.nao**3*self.cai*exp(-0.5*v*self.FRT)
        s = 1 + self.dnaca*(self.cai*self.nao**3 + self.cao*self.nai**3)
        self.inaca = self.gbar*self.q10*self.knaca*(dfcain-dfcaout)/s

    def initial(self, v):
        T = 273 + self.temp
        self.q10.copy_((2.2*(T-296.0)+(310.0-T))/14.0)
        self.FRT = self.F/(self.R*T)
        dfcain = self.nai**3*self.cao*exp(0.5*v*self.FRT)
        dfcaout = self.nao**3*self.cai*exp(-0.5*v*self.FRT)
        s = 1 + self.dnaca*(self.cai*self.nao**3 + self.cao*self.nai**3)
        self.inaca = self.gbar*self.q10*self.knaca*(dfcain-dfcaout)/s

    def ina(self, v):
        return 3 * self.inaca
    
    def ica(self, v):
        return -2 * self.inaca