import torch

from ..mechanisms._mechanism import VoltageProcess


class fire(VoltageProcess):
    VoltageProcess.RANGE(threshold=-50.0, rest=-65.0)

    def update_v(self, v, dt):
        return torch.where(v > self.threshold, self.rest, v)
