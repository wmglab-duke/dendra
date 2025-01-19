import torch

from .core import Distribution
from axonml.models.mixins import to_param


class Uniform(Distribution):
    def __init__(self, low, high, seed=None, once=False):
        super().__init__(seed, once)
        self.low = to_param(low)
        self.high = to_param(high)

    def sample(self, n: int):
        return (
            torch.rand(n, device=self.device(), generator=self.rng)
            * (self.high - self.low)
            + self.low
        )


class PositiveUniform(Distribution):
    def __init__(self, low, high, seed=None, once=False):
        super().__init__(seed, once)
        self.low = to_param(low)
        self.high = to_param(high)

    def sample(self, n: int):
        return (
            torch.rand(n, device=self.device(), generator=self.device())
            * (self.high - self.low)
            + self.low
        ).abs()
