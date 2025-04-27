from ..mechanisms import *
from ..mechanisms.ops import *


class extrapump(Mechanism):
    PARAMETER(pumpik=0.0, pumpina=0.0, pumpica=0.0)
    USEION("k", write=["ik"])
    USEION("na", write=["ina"])
    USEION("ca", write=["ica"])

    def ik(self, v):
        return self.pumpik

    def ina(self, v):
        return self.pumpina
    
    def ica(self, v):
        return self.pumpica
