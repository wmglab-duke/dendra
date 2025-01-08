# Sodium-dependent potassium current
# Paramaters according to Wang et al. 2003 (based on Bischoff et al. 1998)

from ..mechanisms import *
from ..mechanisms.ops import exp


class kna(Mechanism):
    PARAMETER(gbar=0.0001, pmax=0.37, nH=3.5, ec50=38.7)

    USEION("na", read=["nai"])
    USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        w = self.pmax / (1 + (self.ec50 / self.nai) ** self.nH)
        return self.gbar * w * (v - self.ek)
