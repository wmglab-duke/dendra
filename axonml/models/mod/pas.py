from ..mechanisms import Mechanism, PARAMETER


class pas(Mechanism):
    PARAMETER({"g": 0.007, "e": -70.0})

    def i(self, v):
        return self.g * (v - self.e)
