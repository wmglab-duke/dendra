# Sodium-Potassium Pump in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class nakpumpSchild(Mechanism):
    PARAMETER(
        INaKmax22=0.009726135,
        Kmnai=5.46,
        Kmko=0.621,
        Q10NaK=1.16,
        Q10TempA=22.85,
        Q10TempB=10,
    )

    USEION("k", read=["ko"], write=["ik"])
    USEION("na", read=["nai"], write=["ina"])

    ASSIGNED("ink", "INaKmax")

    def initial(self, v):
        self.INaKmax = self.INaKmax22 * self.Q10NaK ** (
            (self.Q10TempA - self.temp) / self.Q10TempB
        )
        fnk = (v + 150.0) / (v + 200.0)
        self.ink = (
            self.INaKmax
            * fnk
            * ((self.nai / (self.nai + self.Kmnai)) ** 3)
            * ((self.ko / (self.ko + self.Kmko)) ** 2)
        )

    def breakpoint(self, v):
        fnk = (v + 150.0) / (v + 200.0)
        self.ink = (
            self.INaKmax
            * fnk
            * ((self.nai / (self.nai + self.Kmnai)) ** 3)
            * ((self.ko / (self.ko + self.Kmko)) ** 2)
        )

    def ina(self, v):
        return 3 * self.ink

    def ik(self, v):
        return -2 * self.ink
