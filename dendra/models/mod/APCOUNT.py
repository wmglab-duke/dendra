import torch

from ..mechanisms import Mechanism as M


class apcount(M):
    """
    A mechanism that counts the number of action potentials (spikes).
    The spike is detected when the membrane potential `v` crosses a specified threshold
    from below.

    Parameters
    ----------
    threshold : float
        The membrane potential threshold for spike detection. Default is 0.0 mV.
    """

    M.RANGE(threshold=0.0)  # Threshold for spike detection
    M.ASSIGNED("n", "active")

    def initial(self, v):
        self.n = torch.zeros_like(v, dtype=torch.float32)
        self.active = torch.zeros_like(v, dtype=torch.bool)

    def breakpoint(self, v):
        above_threshold = v > self.threshold
        spikes = above_threshold & ~self.active
        self.n += spikes.float()
        self.active = above_threshold
