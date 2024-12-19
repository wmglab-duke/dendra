# Calcium Pump in Schild 1994

from ..mechanisms import *
from ..mechanisms.ops import *


class capump(Mechanism):
    PARAMETER(
        ICaPmax22=0.000859437, KmCa=0.0005, Q10CaP=2.30, Q10TempA=22.0, Q10TempB=10.0
    )

    USEION("ca", read=["cai"], write=["ica"])

    ASSIGNED("IcaPmax")

    def initial(self):
        self.ICaPmax = self.ICaPmac22 * self.Q10CaP ** (
            (self.Q10TempA - self.temp) / self.Q10TempB
        )

    def ica(self, v):
        return self.ICaPmax * self.cai / (self.KmCa + self.cai)
