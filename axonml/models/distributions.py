import math

import torch
import torch.nn as nn
from torch.distributions import Normal as _Normal

from .parametric import to_param


class Distribution(nn.Module):
    """Abstract base class for re-parameterised distributions."""

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError


class Normal(Distribution):
    r"""Diagonal Normal with trainable $\mu$ and $\sigma$ confined to (0, ∞)."""

    def __init__(self, mean=0.0, std=1.0):
        super().__init__()

        self.mean = to_param(mean)
        self.std = to_param(std)

    # -------- utilities ---------------------------------------------------
    @property
    def _log_std(self):
        return self.std.log()

    @property
    def _std(self):
        return self._log_std.exp()

    @property
    def _dist(self):
        return _Normal(self.mean, self._std)

    # -------- required API ------------------------------------------------
    def rsample(self, sample_shape=torch.Size()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, value):
        return self._dist.log_prob(value)

    # convenient shorthand identical to your `sample(n)`
    def sample(self, n):
        return self.rsample((n,))


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _standard_normal_icdf(u):
    # inverse CDF (a.k.a. quantile) of N(0,1)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# main class
# ──────────────────────────────────────────────────────────────────────────────
class TruncatedNormal(Distribution):
    r"""
    N(μ, σ²) *restricted to* [low, high]  with reparameterised sampling.

    Args
    ----
    mean, std     : initial parameters (float, tensor or nn.Parameter)
    low, high     : scalars or tensors broadcastable to `mean`
    """

    def __init__(self, mean=0.0, std=1.0, *, low=0.0, high=math.inf):
        super().__init__()

        # register learnable μ and log σ
        self.mean = to_param(mean)  # nn.Parameter or buffer
        self._log_std = to_param(torch.as_tensor(std).log())

        # bounds are **not** typically trained, so we keep them buffers
        self.register_buffer("low", torch.as_tensor(low))
        self.register_buffer("high", torch.as_tensor(high))

    # ---------- derived helpers --------------------------------------------
    @property
    def std(self):
        return self._log_std.exp()  # ensure σ > 0

    @property
    def _a(self):  # standardised bounds
        return (self.low - self.mean) / self.std

    @property
    def _b(self):
        return (self.high - self.mean) / self.std

    @property
    def _Z(self):  # normalising constant
        return _standard_normal_cdf(self._b) - _standard_normal_cdf(self._a)

    # ---------- API ---------------------------------------------------------
    def rsample(self, sample_shape=torch.Size()):
        """
        Differentiable sample using inverse-CDF reparameterisation:
            1. u ~ Uniform[0,1]
            2. p = Φ(a) + u * Z
            3. z = Φ⁻¹(p)
            4. x = μ + σ z
        """
        u = torch.rand(sample_shape + self.mean.shape, device=self.mean.device)
        p = _standard_normal_cdf(self._a) + u * self._Z
        z = _standard_normal_icdf(p)
        return self.mean + self.std * z

    def log_prob(self, value):
        base_logp = -0.5 * ((value - self.mean) / self.std) ** 2 - torch.log(
            self.std * math.sqrt(2 * math.pi)
        )
        logp = base_logp - self._Z.log()
        # clamp to -inf outside support so training doesn't explode
        mask = (value < self.low) | (value > self.high)
        return torch.where(mask, torch.full_like(value, -math.inf), logp)

    # alias for symmetry with your Normal
    def sample(self, n):
        return self.rsample((n,))
