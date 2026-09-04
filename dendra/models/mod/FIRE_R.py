import torch

from ..mechanisms import VoltageProcess


class fire_r(VoltageProcess):
    """A VoltageProcess that implements a leaky integrate-and-fire neuron model with refractory period.
    When the membrane potential `v` crosses the specified threshold, it is reset to the
    resting potential and enters a refractory period during which it cannot spike again.

    Parameters
    ----------
    threshold : float
        The membrane potential threshold for spike detection. Default is -50.0 mV.
    rest : float
        The resting membrane potential (in mV) to reset to after a spike. Default is -65.0 mV.
    refractory : float
        The duration (in ms) of the refractory period during which the neuron cannot spike again. Default is 5.0 ms.
    """

    VoltageProcess.RANGE(threshold=-50.0, rest=-65.0, refractory=5.0)
    VoltageProcess.CARRY("is_refractory", dtype=torch.bool)
    VoltageProcess.CARRY("time_refractory")

    def initial_values(self, v, values):
        return {
            "is_refractory": torch.zeros_like(v, dtype=torch.bool),
            "time_refractory": torch.zeros_like(v),
        }

    def update_v(self, v):
        still_ref = self.is_refractory
        time_refractory = self.time_refractory
        time_refractory = torch.where(
            still_ref, time_refractory - self.dt, time_refractory
        )

        # cells whose timer expired leave refractory state
        recovered = still_ref & (time_refractory <= 0)
        is_refractory = torch.where(recovered, False, still_ref)

        # ---- 2. find new spikes (only in non-refractory cells) -------
        can_spike = ~is_refractory
        new_spike = can_spike & (v > self.threshold)

        # mark & start refractory for those
        self.is_refractory = torch.where(new_spike, True, is_refractory)
        self.time_refractory = torch.where(new_spike, self.refractory, time_refractory)

        # ---- 3. force voltage to rest while refractory ---------------
        return torch.where(self.is_refractory, self.rest, v)
