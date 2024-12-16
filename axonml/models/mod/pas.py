from ..mechanisms import *


class pas(Mechanism):
    PARAMETER(g=0.007, e=-70.0)

    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
