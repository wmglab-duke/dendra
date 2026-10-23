"""Learnable probability distributions used in Dendra models."""

import math

import torch
import torch.nn as nn
from torch.distributions import Normal as _Normal

from .parametric import to_param


def _normal_draw(shape, dtype, device, generator=None):
    """Use the same standard-normal kernel as Torch Normal.rsample."""
    return torch.empty(shape, dtype=dtype, device=device).normal_(generator=generator)


def _normal_from_draw(mean, std, draw):
    """Apply the affine normal reparameterization using explicit base noise."""
    return mean + draw * std


class Distribution(nn.Module):
    """Abstract base class for re-parameterised probability distributions.

    Implement :meth:`sample_draw_schema`, :meth:`sample_draw`, and
    :meth:`sample_from_draw` to share explicit sampling between native and
    functional connections. The inherited sampling adapters retain independent
    draws for checkpoint continuation and differentiable rematerialization.

    Existing subclasses can implement :meth:`rsample` and :meth:`log_prob`
    following the PyTorch distribution API for native sampling.
    """

    def sample_draw_schema(self, sample_shape=torch.Size()):
        """Return the base draw's ``(shape, floating dtype, device)`` metadata.

        The draw may have additional dimensions consumed by the transform; its
        shape need not equal the resulting sample shape. Metadata must describe
        the draw without consuming randomness or changing module state.
        """
        raise NotImplementedError

    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        """Draw independent base noise using only the supplied generator.

        ``generator=None`` uses the ordinary Torch RNG for the draw's device.
        Return one Tensor matching :meth:`sample_draw_schema`.
        """
        raise NotImplementedError

    def sample_from_draw(self, draw):
        """Purely transform explicit base noise into differentiable samples."""
        raise NotImplementedError

    def sample_draw_placeholder(self, sample_shape=torch.Size()):
        """Return a deterministic valid base draw without consuming randomness.

        The default is zeros. Override this hook when zeros lie outside the
        transform's domain, such as an inverse CDF requiring interior uniforms.
        This value permits safe evaluation of inactive legacy-checkpoint paths;
        it does not replace a sampled draw or reconstruct missing provenance.
        """
        shape, dtype, device = connection_distribution_draw_schema(self, sample_shape)
        return torch.zeros(shape, dtype=dtype, device=device)

    def sample_with_draw(self, sample_shape=torch.Size(), *, generator=None):
        """Return a differentiable sample and detached independent base draw."""
        sample_shape = torch.Size(sample_shape)
        # Validate metadata before consuming the caller's random stream.
        schema = connection_distribution_draw_schema(self, sample_shape)
        with torch.no_grad():
            draw = self.sample_draw(sample_shape, generator=generator)
        _validate_connection_distribution_draw(draw, schema)
        draw = draw.detach()
        value = self.sample_from_draw(draw)
        if not torch.is_tensor(value):
            raise TypeError("Connection distribution transforms must return a Tensor.")
        return value, draw

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
        return self.sample_with_draw(sample_shape)[0]

    def sample(self, n):
        """Draw ``n`` independent differentiable samples."""
        return self.rsample((n,))

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
    def sample_draw_schema(self, sample_shape=torch.Size()):
        """Describe the independent standard-normal draw without sampling."""
        return (
            torch.Size(sample_shape)
            + torch.broadcast_shapes(self.mean.shape, self.std.shape),
            self.mean.dtype,
            self.mean.device,
        )

    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        """Draw standard-normal noise for an explicit reparameterization."""
        distribution = self._dist
        shape = distribution._extended_shape(torch.Size(sample_shape))
        return _normal_draw(
            shape, distribution.loc.dtype, distribution.loc.device, generator
        )

    def sample_from_draw(self, draw):
        """Transform an explicit standard-normal draw without consuming RNG."""
        mean, std = torch.broadcast_tensors(self.mean, self._std)
        return _normal_from_draw(mean, std, draw)

    def sample_with_draw(self, sample_shape=torch.Size(), *, generator=None):
        """Sample and retain base noise while evaluating source getters once."""
        distribution = self._dist
        draw = _normal_draw(
            distribution._extended_shape(torch.Size(sample_shape)),
            distribution.loc.dtype,
            distribution.loc.device,
            generator,
        )
        return _normal_from_draw(distribution.loc, distribution.scale, draw), draw

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
        if type(self)._dist is not Normal._dist or "_dist" in vars(self):
            # Authored distribution recipes retain their own sampling method.
            # The built-in recipe shares the explicit-draw implementation.
            return self._dist.rsample(sample_shape)
        return self.sample_with_draw(sample_shape)[0]

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
        """Draw ``n`` independent differentiable samples."""
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


def _standardized_normal_bound(bound, mean, std):
    """Standardize a bound without an infinite scale derivative."""
    # Infinite truncation limits have constant standardized values. Avoid
    # differentiating (infinity - mean) / std before selecting those constants:
    # its backward otherwise multiplies a zero CDF derivative by infinity.
    safe_bound = torch.where(torch.isinf(bound), mean, bound)
    return torch.where(torch.isinf(bound), bound, (safe_bound - mean) / std)


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
        return _standardized_normal_bound(self.low, self.mean, self.std)

    @property
    def _b(self):
        return _standardized_normal_bound(self.high, self.mean, self.std)

    @property
    def _Z(self):  # normalising constant
        return _standard_normal_cdf(self._b) - _standard_normal_cdf(self._a)

    # ---------- API ---------------------------------------------------------
    def sample_draw_schema(self, sample_shape=torch.Size()):
        """Describe the independent uniform draw without sampling."""
        return (
            torch.Size(sample_shape) + self.mean.shape,
            torch.get_default_dtype(),
            self.mean.device,
        )

    def sample_draw(self, sample_shape=torch.Size(), *, generator=None):
        """Draw uniforms using the native default floating-point dtype."""
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    def sample_draw_placeholder(self, sample_shape=torch.Size()):
        """Use interior uniforms for safe inactive inverse-CDF evaluation."""
        shape, dtype, device = self.sample_draw_schema(sample_shape)
        return torch.full(shape, 0.5, dtype=dtype, device=device)

    def sample_from_draw(self, draw):
        """Apply the native inverse-CDF transform without consuming RNG."""
        p = _standard_normal_cdf(self._a) + draw * self._Z
        z = _standard_normal_icdf(p)
        return self.mean + self.std * z

    def sample_with_draw(self, sample_shape=torch.Size(), *, generator=None):
        """Sample and retain the uniforms used by the inverse-CDF transform."""
        draw = self.sample_draw(sample_shape, generator=generator)
        return self.sample_from_draw(draw), draw

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
        return self.sample_with_draw(sample_shape)[0]

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
        """Draw ``n`` independent differentiable samples."""
        return self.rsample((n,))


_CONNECTION_DISTRIBUTION_METHODS = {
    distribution: {
        name: getattr(distribution, name)
        for name in (
            "sample",
            "rsample",
            "sample_draw",
            "sample_draw_schema",
            "sample_from_draw",
            "sample_with_draw",
            "sample_draw_placeholder",
        )
    }
    for distribution in (Normal, TruncatedNormal)
}


def connection_distribution_is_supported(source):
    """Recognize explicit draw recipes while preserving authored fallbacks."""
    from torch.nn.utils import parametrize

    source_type = type(source)
    if source_type.__module__ == "torch.nn.utils.parametrize" and (
        parametrize.is_parametrized(source)
    ):
        source_type = parametrize.type_before_parametrizations(source)
    methods = _CONNECTION_DISTRIBUTION_METHODS.get(source_type)
    if methods is not None:
        if any(
            name in vars(source) or getattr(type(source), name) is not method
            for name, method in methods.items()
        ):
            return False
    else:
        if not isinstance(source, Distribution) or any(
            name in vars(source)
            or getattr(type(source), name) is not getattr(Distribution, name)
            for name in ("sample", "rsample", "sample_with_draw")
        ):
            return False
        if any(
            name in vars(source)
            or getattr(type(source), name) is getattr(Distribution, name)
            for name in ("sample_draw_schema", "sample_draw", "sample_from_draw")
        ):
            return False
    hook_names = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    if any(getattr(source, name) for name in hook_names):
        return False
    module_hooks = torch.nn.modules.module
    return not any(getattr(module_hooks, "_global" + name) for name in hook_names)


def connection_distribution_draw_schema(source, sample_shape=torch.Size()):
    """Validate fixed explicit-draw metadata without drawing random values."""
    schema = source.sample_draw_schema(torch.Size(sample_shape))
    if not isinstance(schema, (tuple, list)) or len(schema) != 3:
        raise ValueError(
            "Connection distribution draw schema must be (shape, dtype, device)."
        )
    shape, dtype, device = schema
    try:
        shape = torch.Size(shape)
        device = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Invalid connection distribution draw schema.") from error
    if any(size < 0 for size in shape) or not (
        isinstance(dtype, torch.dtype) and dtype.is_floating_point
    ):
        raise ValueError(
            "Connection distribution draws need a nonnegative shape and floating dtype."
        )
    return shape, dtype, device


def _validate_connection_distribution_draw(draw, schema):
    shape, dtype, device = schema
    if not torch.is_tensor(draw) or (
        draw.shape != shape or draw.dtype != dtype or draw.device != device
    ):
        raise ValueError(
            "Connection distribution draw does not match its shape, dtype, and device schema."
        )
    return draw


def connection_distribution_validate_draw(source, draw, sample_shape=torch.Size()):
    """Require one floating Tensor matching the source's declared draw schema."""
    return _validate_connection_distribution_draw(
        draw, connection_distribution_draw_schema(source, sample_shape)
    )


def connection_distribution_placeholder(source, sample_shape=torch.Size()):
    """Obtain a schema-compatible deterministic draw for inactive paths."""
    draw = source.sample_draw_placeholder(torch.Size(sample_shape))
    return connection_distribution_validate_draw(source, draw, sample_shape)
