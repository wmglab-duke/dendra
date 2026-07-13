from ..mechanisms import Mechanism as M


class pas(M):
    """
    A passive membrane mechanism that models a leak conductance.

    Parameters
    ----------
    g : float
        Leak conductance density in S/cm². Default is 0.001 S/cm²
        (equivalently 1 mS/cm²).
    e : float
        The leak reversal potential (in mV). Default is -70.0.

    Notes
    -----
    ``i(v)`` returns outward-positive current density in mA/cm².
    """

    M.RANGE(g=0.001, e=-70.0)
    M.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
