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
    M.CARRY("n", dtype=torch.float32)
    M.CARRY("active", dtype=torch.bool)

    def initial_values(self, v, values):
        # Keep the hard counter at least float32 even for low-precision voltage
        # simulations.  float16 stops representing consecutive integers at
        # 2048 (bfloat16 at 256), which can silently drop later spikes.
        # Starting above threshold is not an upward crossing.  Initialize the
        # hard latch from voltage so the first accepted transition cannot count a
        # spurious event, matching apcount_d and spikedetect semantics.
        return {
            "n": torch.zeros_like(v, dtype=torch.float32),
            "active": v > self.threshold,
        }

    def advance(self, v, dt, values):
        del dt
        above_threshold = v > self.threshold
        spikes = above_threshold & ~values["active"]
        return {
            "n": values["n"]
            + spikes.to(dtype=values["n"].dtype, device=values["n"].device),
            "active": above_threshold,
        }
