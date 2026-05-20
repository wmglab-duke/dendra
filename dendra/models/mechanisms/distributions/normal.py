import torch

from dendra.models.parametric import to_param

from .core import Distribution


class Normal(Distribution):
    def __init__(self, mu, sigma, seed=None, once=False):
        super().__init__(seed, once)
        self.mu = to_param(mu)
        self.sigma = to_param(sigma)

    def sample(self, n: int):
        eps = torch.randn(n, device=self.device(), generator=self.rng)
        return self.mu + self.sigma * eps


class PositiveNormal(Distribution):
    def __init__(self, mu, sigma, seed=None, once=False):
        super().__init__(seed, once)
        self.mu = to_param(mu)
        self.sigma = to_param(sigma)

    def sample(self, n: int):
        eps = torch.randn(n, device=self.device(), generator=self.rng)
        return (self.mu + self.sigma * eps).abs()
