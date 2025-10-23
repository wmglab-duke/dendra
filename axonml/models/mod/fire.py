import torch

from ..mechanisms import VoltageProcess


class fire(VoltageProcess):
    VoltageProcess.RANGE(threshold=-50.0, rest=-65.0)

    def update_v(self, v):
        return torch.where(v > self.threshold, self.rest, v)
