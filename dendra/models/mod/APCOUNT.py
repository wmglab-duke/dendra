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
    M.BUFFER("n", "active")

    def initial(self, v):
        # Keep the hard counter at least float32 even for low-precision voltage
        # simulations.  float16 stops representing consecutive integers at
        # 2048 (bfloat16 at 256), which can silently drop later spikes.
        self.n = torch.zeros_like(v, dtype=torch.float32)
        # Starting above threshold is not an upward crossing.  Initialize the
        # hard latch from voltage so the first breakpoint cannot count a
        # spurious event, matching apcount_d and spikedetect semantics.
        self.active = v > self.threshold

    def breakpoint(self, v):
        above_threshold = v > self.threshold
        spikes = above_threshold & ~self.active
        self.n += spikes.to(dtype=self.n.dtype, device=self.n.device)
        self.active = above_threshold
