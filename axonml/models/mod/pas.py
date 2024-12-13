from ..mechanisms import *


class pas(Mechanism):
    CONDUCTANCE({"g": 0.007})

    PARAMETER({"e": -70.0})

    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
