import torch

from ..mechanisms._mechanism import VoltageProcess


class fire_r(VoltageProcess):
    VoltageProcess.RANGE(threshold=-50.0, rest=-65.0, refractory=5.0)
    VoltageProcess.ASSIGNED("is_refractory", "time_refractory")

    def initial(self, v):
        self.is_refractory = torch.zeros_like(v, dtype=torch.bool)
        self.time_refractory = torch.zeros_like(v, dtype=v.dtype)

    def update_v(self, v, dt):
        still_ref = self.is_refractory
        time_refractory = self.time_refractory
        time_refractory = torch.where(still_ref, time_refractory-dt, time_refractory)

        # cells whose timer expired leave refractory state
        recovered = still_ref & (time_refractory <= 0)
        is_refractory = torch.where(recovered, False, still_ref)

        # ---- 2. find new spikes (only in non-refractory cells) -------
        can_spike = ~is_refractory
        new_spike = can_spike & (v > self.threshold)

        # mark & start refractory for those
        self.is_refractory   = torch.where(new_spike, True, is_refractory)
        self.time_refractory = torch.where(new_spike, self.refractory, time_refractory)

        # ---- 3. force voltage to rest while refractory ---------------
        return torch.where(self.is_refractory, self.rest, v)