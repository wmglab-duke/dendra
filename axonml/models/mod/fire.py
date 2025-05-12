import torch

from ..mechanisms import *


class fire(Mechanism):
    PARAMETER(threshold=-50.0, rest=-65.0)

    def breakpoint(self, v):
        if v.dim() > 0:
            v[:] = torch.where(v > self.threshold, self.rest, v)