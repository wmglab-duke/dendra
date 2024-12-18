# Chapman JB, Johnson EA, Kootsey JM. (1983)
# Electrical and Biochemical Properties of an Enzyme Model of the Sodium Pump
# J. Membrane Biol. 74, 139-153

from ..mechanisms import *
from ..mechanisms.ops import *


class nakpump(Mechanism):
    PARAMETER(smalla=0.0, b1=1.0)

    ASSIGNED("pump")
    USEION("na", read=["nai"], write=["ina"])
    USEION("k", read=["ko"], write=["ik"])

    def breakpoint(self, v):
        self.pump = (
            self.smalla
            / ((1.0 + self.b1 / self.ko) ** 2)
            * (
                1.62 / (1.0 + (6.7 / (self.nai + 8.0)) ** 3)
                + 1.0 / (1.0 + (67.6 / (self.nai + 8.0)) ** 3)
            )
        )

    def ina(self, v):
        return -1.5 * self.pump

    def ik(self, v):
        return self.pump
