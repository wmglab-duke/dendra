from ..mechanisms._mechanism import Mechanism as M


class pas(M):
    M.PARAMETER(g=0.001, e=-70.0)
    M.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
