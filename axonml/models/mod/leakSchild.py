from ..mechanisms import *
from ..mechanisms.ops import *


class leakSchild(Mechanism):
    PARAMETER(gbna=1.85681e-05, gbca=3.00626e-06, R=8314, z=2, ecaoffset=78.7, F=96500)
    USEION("na", read=["ena"], write=["ina"])
    USEION("ca", read=["cao", "cai"], write=["ica"])

    def ina(self, v):
        return self.gbna * (v - self.ena)

    def ica(self, v):
        ecaleak = (
            self.R * (self.temp + 273.15) / self.z / self.F * log(self.cao / self.cai)
        ) - self.ecaoffset
        return self.gbca * (v - ecaleak)
