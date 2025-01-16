import torch

from .core import Distribution
from axonml.models.mixins import to_param


class Normal(Distribution):
    def __init__(self, mu, sigma, seed=None):
        super().__init__(seed)
        self.mu = to_param(mu)
        self.sigma = to_param(sigma)

    def sample(self, n: int):
        eps = torch.randn(n, device=self.device(), generator=self.rng)
        return self.mu + self.sigma * eps


class PositiveNormal(Distribution):
    def __init__(self, mu, sigma, seed=None):
        super().__init__(seed)
        self.mu = to_param(mu)
        self.sigma = to_param(sigma)

    def sample(self, n: int):
        eps = torch.randn(n, device=self.device(), generator=self.rng)
        return (self.mu + self.sigma * eps).abs()
