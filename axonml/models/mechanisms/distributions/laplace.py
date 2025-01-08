import torch

from .core import Distribution
from axonml.models.mixins import to_param


class Laplace(Distribution):
    def __init__(self, mu, b, seed=None):
        super().__init__(seed)
        self.mu = to_param(mu)
        self.b = to_param(b)

    def sample(self, n: int):
        u = torch.rand(n, device=self.mu.device) - 0.5
        return self.mu - self.b * u.sign() * torch.log1p(-2 * u.abs())


class PositiveLaplace(Distribution):
    def __init__(self, mu, b, seed=None):
        super().__init__(seed)
        self.mu = to_param(mu)
        self.b = to_param(b)

    def sample(self, n: int):
        u = torch.rand(n, device=self.mu.device)
        return self.mu - self.b * torch.log1p(-2 * u)
