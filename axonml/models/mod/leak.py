from ..mechanisms import *
from ..mechanisms.ops import *


class leak(Mechanism):
    PARAMETER(gkleak=0.0, gnaleak=0.0)

    USEION("na", read=["ena"], write=["ina"])
    USEION("k", read=["ek"], write=["ik"])

    def ina(self, v):
        return self.gnaleak * (v - self.ena)

    def ik(self, v):
        return self.gkleak * (v - self.ek)
