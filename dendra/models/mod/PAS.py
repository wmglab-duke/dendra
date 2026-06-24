from ..mechanisms import Mechanism as M


class pas(M):
    """
    A passive membrane mechanism that models a leak conductance.

    Parameters
    ----------
    g : float
        The leak conductance (in mS/cm^2). Default is 0.001.
    e : float
        The leak reversal potential (in mV). Default is -70.0.
    """

    M.RANGE(g=0.001, e=-70.0)
    M.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
