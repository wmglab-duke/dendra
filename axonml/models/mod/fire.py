import torch

from ..mechanisms import *


class fire(Mechanism):
    PARAMETER(threshold=-50.0, rest=-70.0)

    def update(self, model):
        model.v = torch.where(model.v > self.threshold, self.rest, model.v)