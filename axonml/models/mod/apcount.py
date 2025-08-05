import torch

from ..mechanisms import Mechanism as M


class apcount(M):
    """
    A mechanism that counts the number of action potentials (spikes).
    """

    M.RANGE(threshold=0.0)  # Threshold for spike detection
    M.ASSIGNED("_n", "active")

    def initial(self, v):
        self._n = torch.zeros_like(v, dtype=torch.float32)
        self.active = torch.zeros_like(v, dtype=torch.bool)

    def breakpoint(self, v):
        above_threshold = v > self.threshold
        spikes = above_threshold & ~self.active
        self._n += spikes.float()
        self.active = above_threshold

    @property
    def n(self):
        """
        Returns the number of action potentials (spikes) counted.
        """
        return self._n.int()
