import torch

from ..mechanisms import Mechanism as M


class apcount(M):
    """
    A mechanism that counts the number of action potentials (spikes).
    """

    M.RANGE(threshold=0.0)  # Threshold for spike detection
    M.ASSIGNED("n", "active")

    def initial(self, v):
        self.n = torch.zeros_like(v, dtype=torch.int)
        self.active = torch.zeros_like(v, dtype=torch.bool)

    def breakpoint(self, v):
        above_threshold = v > self.threshold
        spikes = above_threshold & ~self.active
        self.n += spikes.int()
        self.active = above_threshold
