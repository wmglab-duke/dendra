# NaCaPump is the Sodium-Calcium Exchanger in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class nacapump(Mechanism):
    PARAMETER(
        KNaCa22=1.27324e-06,
        Q10NaCa=2.20,
        Q10TempA=22.85,
        Q10TempB=10,
        r=3,
        gamma=0.5,
        DNaCa=0.0036,
        F=96500,
        R=8.314,
    )

    USEION("ca", read=["cao", "cai"], write=["ica"])
    USEION("na", read=["nai", "nao"], write=["ina"])

    ASSIGNED("inca")

    def breakpoint(self, v):
        temp = self.temp + 273.15
        S = 1.0 + self.DNaCa * (self.cai * self.nao**3 + self.cao * self.nai**3)
        DFin = (
            self.nai**3
            * self.cao
            * exp(((self.r - 2) * self.gamma * v * self.F) / ((1000) * self.R * temp))
        )
        DFout = (
            self.nao**3
            * self.cai
            * exp(
                ((self.r - 2) * (self.gamma - 1) * v * self.F)
                / ((1000) * self.R * temp)
            )
        )
        self.inca = self.KNaCa * ((DFin - DFout) / S)

    def ina(self):
        return 3 * self.inca

    def ica(self):
        return -2 * self.inca
