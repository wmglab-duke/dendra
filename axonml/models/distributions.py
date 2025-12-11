"""Learnable probability distributions used in AxonML models."""

import math

import torch
import torch.nn as nn
from torch.distributions import Normal as _Normal

from .parametric import to_param


class Distribution(nn.Module):
    """Abstract base class for re-parameterised probability distributions.

    Subclasses must implement :meth:`rsample` and :meth:`log_prob` following the
    PyTorch distribution API to support differentiable sampling.
    """

    def rsample(self, sample_shape=torch.Size()):
        """Draw differentiable samples.

        Parameters
        ----------
        sample_shape : torch.Size, optional
            Leading sample shape prepended to the distribution event shape.

        Returns
        -------
        torch.Tensor
            Reparameterised sample tensor.
        """
        raise NotImplementedError

    def log_prob(self, value):
        """Evaluate the log-likelihood of a given value.

        Parameters
        ----------
        value : torch.Tensor
            Value at which to evaluate the log-density.

        Returns
        -------
        torch.Tensor
            Log-probability broadcast to ``value.shape``.
        """
        raise NotImplementedError


class Normal(Distribution):
    r"""Diagonal normal distribution with learnable mean and scale.

    Parameters
    ----------
    mean : float or torch.Tensor or torch.nn.Parameter, optional
        Location parameter :math:`\mu`. Broadcast across event dimensions.
    std : float or torch.Tensor or torch.nn.Parameter, optional
        Positive scale parameter :math:`\sigma`.
    """

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
        """Draw a differentiable sample.

        Parameters
        ----------
        sample_shape : torch.Size, optional
            Leading shape of independent samples.

        Returns
        -------
        torch.Tensor
            Sample tensor with shape ``sample_shape + mean.shape``.
        """
        return self._dist.rsample(sample_shape)

    def log_prob(self, value):
        """Compute the log-density of ``value``.

        Parameters
        ----------
        value : torch.Tensor
            Sample locations broadcastable to ``mean.shape``.

        Returns
        -------
        torch.Tensor
            Log-likelihood of ``value``.
        """
        return self._dist.log_prob(value)

    def sample(self, n):
        """Draw ``n`` independent non-differentiable samples."""
        return self.rsample((n,))


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _standard_normal_cdf(x):
    """Evaluate the CDF of the standard normal distribution."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _standard_normal_icdf(u):
    """Evaluate the inverse CDF (quantile) of the standard normal."""
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# main class
# ──────────────────────────────────────────────────────────────────────────────
class TruncatedNormal(Distribution):
    r"""Normal distribution truncated to a bounded interval.

    Parameters
    ----------
    mean : float or torch.Tensor or torch.nn.Parameter, optional
        Location parameter :math:`\mu`.
    std : float or torch.Tensor or torch.nn.Parameter, optional
        Positive scale parameter :math:`\sigma`.
    low : float or torch.Tensor, optional
        Inclusive lower truncation bound.
    high : float or torch.Tensor, optional
        Inclusive upper truncation bound.
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
        """Draw differentiable samples using inverse-CDF reparameterisation.

        Parameters
        ----------
        sample_shape : torch.Size, optional
            Leading shape of independent samples.

        Returns
        -------
        torch.Tensor
            Sample tensor with shape ``sample_shape + mean.shape``.
        """
        u = torch.rand(sample_shape + self.mean.shape, device=self.mean.device)
        p = _standard_normal_cdf(self._a) + u * self._Z
        z = _standard_normal_icdf(p)
        return self.mean + self.std * z

    def log_prob(self, value):
        """Compute the log-density of ``value`` under the truncated normal.

        Parameters
        ----------
        value : torch.Tensor
            Points at which to evaluate the log-density.

        Returns
        -------
        torch.Tensor
            Log-probabilities with values outside ``[low, high]`` set to ``-inf``.
        """
        base_logp = -0.5 * ((value - self.mean) / self.std) ** 2 - torch.log(
            self.std * math.sqrt(2 * math.pi)
        )
        logp = base_logp - self._Z.log()
        # clamp to -inf outside support so training doesn't explode
        mask = (value < self.low) | (value > self.high)
        return torch.where(mask, torch.full_like(value, -math.inf), logp)

    # alias for symmetry with Normal
    def sample(self, n):
        """Draw ``n`` independent non-differentiable samples."""
        return self.rsample((n,))
