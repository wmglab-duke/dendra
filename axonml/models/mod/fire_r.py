import torch

from ..mechanisms import *


class fire_r(Mechanism):
    PARAMETER(threshold=-50.0, rest=-65.0, refractory=5.0, dt=0.01)
    ASSIGNED("is_refractory", "time_refractory", "spiked")

    def initial(self, v):
        self.is_refractory = torch.zeros_like(v, dtype=torch.bool)
        self.time_refractory = torch.zeros_like(v, dtype=v.dtype)

    def breakpoint(self, v):
        if v.dim() > 0:
            still_ref = self.is_refractory
            self.time_refractory[:] = torch.where(still_ref, self.time_refractory-self.dt, self.time_refractory)

            # cells whose timer expired leave refractory state
            recovered = still_ref & (self.time_refractory <= 0)
            self.is_refractory[:] = torch.where(recovered, False, self.is_refractory)

            # ---- 2. find new spikes (only in non-refractory cells) -------
            can_spike = ~self.is_refractory
            new_spike = can_spike & (v > self.threshold)

            # mark & start refractory for those
            self.is_refractory[:]   = torch.where(new_spike, True, self.is_refractory)
            self.time_refractory[:] = torch.where(new_spike, self.refractory, self.time_refractory)

            # ---- 3. force voltage to rest while refractory ---------------
            v[:] = torch.where(self.is_refractory, self.rest, v)