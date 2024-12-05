from ..mechanisms import (
    Mechanism, PARAMETER, CONDUCTANCE
)


class pas(Mechanism):

    CONDUCTANCE({
        'g': 0.007
    })

    PARAMETER({
        'e': -70.0
    })

    def i(self, v):
        g = self.conductances['g']
        return g * (v - self.e)