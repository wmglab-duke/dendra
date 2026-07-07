"""Random parameter declarations for Dendra ``Parameterized`` modules."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

__all__ = [
    "RandomParameterSpec",
    "DistributionSpec",
    "available_random_distributions",
    "get_random_distribution",
    "register_random_distribution",
    "make_random_parameter_spec",
    "sample_random_parameter",
]

_ALLOWED_CONSTRAINTS = {"real", "positive", "negative"}


@dataclass(frozen=True)
class RandomParameterSpec:
    """Specification for a sampled parameter buffer."""

    name: str
    scope: str
    distribution: str
    params: Mapping[str, object]
    constraints: Mapping[str, str]
    resample_on_initialize: bool = True
    seed: int | None = None
    reparameterized: bool = True
    rng_name: str | None = None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}_{p}" for p in self.params)

    @property
    def effective_rng_name(self) -> str:
        return self.rng_name or f"{self.name}_rng"

    def full_parameter_name(self, distribution_parameter: str) -> str:
        return f"{self.name}_{distribution_parameter}"


def _readonly(mapping):
    return MappingProxyType(dict(mapping))


DistributionSampler = Callable[
    [Mapping[str, torch.Tensor], object, tuple[int, ...]], torch.Tensor
]


@dataclass(frozen=True)
class DistributionSpec:
    """Registered random-parameter distribution.

    Parameters
    ----------
    name:
        Canonical distribution name.
    parameter_defaults:
        Mapping of distribution-parameter names to defaults.
    parameter_constraints:
        Mapping of distribution-parameter names to Dendra parameter constraints.
        Supported constraints are ``"real"``, ``"positive"``, and ``"negative"``.
    sampler:
        Callable with signature ``sampler(parameters, rng, shape, *, device, dtype)``.
        ``parameters`` contains tensor-valued distribution parameters after all Dendra
        overrides/parametrizations have been applied. ``rng`` is a Dendra ``RNGModule``.
    """

    name: str
    parameter_defaults: Mapping[str, object]
    parameter_constraints: Mapping[str, str]
    sampler: Callable[..., torch.Tensor]


def _canonical_name(name: str) -> str:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Random distribution name cannot be empty.")
    return key


def _normal_sampler(parameters, rng, shape, *, device, dtype):
    eps = rng.randn(shape, device=device, dtype=dtype)
    return parameters["mu"] + parameters["sigma"] * eps


def _lognormal_sampler(parameters, rng, shape, *, device, dtype):
    eps = rng.randn(shape, device=device, dtype=dtype)
    return torch.exp(parameters["mu"] + parameters["sigma"] * eps)


def _uniform_sampler(parameters, rng, shape, *, device, dtype):
    u = rng.rand(shape, device=device, dtype=dtype)
    return parameters["low"] + (parameters["high"] - parameters["low"]) * u


def _standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _truncated_normal_sampler(parameters, rng, shape, *, device, dtype):
    """Sample from a true inverse-CDF truncated normal distribution.

    The distribution is ``Normal(mu, sigma)`` conditioned on ``low <= x <= high``.
    ``mu``, ``sigma``, ``low``, and ``high`` may be scalar or tensor-valued Dendra
    parameters, so this implementation avoids ``torch.nn.init.trunc_normal_``, whose
    mean/std/bounds API is scalar-oriented.
    """

    mu = parameters["mu"]
    sigma = parameters["sigma"]
    low = parameters["low"]
    high = parameters["high"]

    alpha = (low - mu) / sigma
    beta = (high - mu) / sigma
    cdf_low = _standard_normal_cdf(alpha)
    cdf_high = _standard_normal_cdf(beta)

    finfo = torch.finfo(dtype if dtype is not None else mu.dtype)
    eps = finfo.eps
    # Avoid an exactly zero CDF interval in extreme tails or malformed bounds. This
    # keeps sampling finite and graph-friendly; callers should still provide low < high.
    cdf_high = torch.maximum(cdf_high, cdf_low + eps)

    u = rng.rand(shape, device=device, dtype=dtype)
    p = cdf_low + (cdf_high - cdf_low) * u
    p = torch.clamp(p, eps, 1.0 - eps)
    sample = mu + sigma * math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
    return torch.minimum(torch.maximum(sample, low), high)


_DISTRIBUTIONS: dict[str, DistributionSpec] = {}


def register_random_distribution(
    name: str,
    parameter_defaults: Mapping[str, object],
    parameter_constraints: Mapping[str, str] | None = None,
    sampler: Callable[..., torch.Tensor] | None = None,
    *,
    overwrite: bool = False,
) -> DistributionSpec:
    """Register a random-parameter distribution.

    Parameters
    ----------
    name:
        Distribution name used by ``GLOBALRAND`` / ``RANGERAND`` / ``BATCHRAND``.
    parameter_defaults:
        Default scalar/tensor values for each distribution parameter.
    parameter_constraints:
        Optional constraints for distribution parameters. Missing entries default to
        ``"real"``. Supported values are ``"real"``, ``"positive"``, and
        ``"negative"``. These constraints determine whether generated distribution
        parameters are ordinary, positive, or negative Dendra parameters.
    sampler:
        Callable with signature ``sampler(parameters, rng, shape, *, device, dtype)``.
        It must return a tensor broadcastable to ``shape``. A sampler is required for
        user-defined distributions.
    overwrite:
        If ``False`` (default), registering an existing name raises ``ValueError``.

    Returns
    -------
    DistributionSpec
        The registered immutable distribution specification.
    """

    key = _canonical_name(name)
    if key in _DISTRIBUTIONS and not overwrite:
        raise ValueError(
            f"Random-parameter distribution {name!r} is already registered. "
            "Pass overwrite=True to replace it."
        )
    if sampler is None:
        raise TypeError("register_random_distribution requires a sampler callable.")
    if not callable(sampler):
        raise TypeError("sampler must be callable.")

    defaults = dict(parameter_defaults or {})
    constraints_in = dict(parameter_constraints or {})
    unknown_constraints = set(constraints_in) - set(defaults)
    if unknown_constraints:
        raise ValueError(
            "Random distribution constraints were provided for unknown parameters: "
            f"{sorted(unknown_constraints)}."
        )

    constraints = {}
    for param_name in defaults:
        constraint = str(constraints_in.get(param_name, "real")).strip().lower()
        if constraint not in _ALLOWED_CONSTRAINTS:
            raise ValueError(
                f"Invalid constraint {constraint!r} for distribution parameter "
                f"{param_name!r}. Valid constraints are: {sorted(_ALLOWED_CONSTRAINTS)}."
            )
        constraints[param_name] = constraint

    spec = DistributionSpec(
        key,
        _readonly(defaults),
        _readonly(constraints),
        sampler,
    )
    _DISTRIBUTIONS[key] = spec
    return spec


# Built-in distributions.
register_random_distribution(
    "normal",
    {"mu": 0.0, "sigma": 1.0},
    {"mu": "real", "sigma": "positive"},
    _normal_sampler,
)
register_random_distribution(
    "lognormal",
    {"mu": 0.0, "sigma": 1.0},
    {"mu": "real", "sigma": "positive"},
    _lognormal_sampler,
)
register_random_distribution(
    "uniform",
    {"low": 0.0, "high": 1.0},
    {"low": "real", "high": "real"},
    _uniform_sampler,
)
register_random_distribution(
    "truncated_normal",
    {"mu": 0.0, "sigma": 1.0, "low": -2.0, "high": 2.0},
    {"mu": "real", "sigma": "positive", "low": "real", "high": "real"},
    _truncated_normal_sampler,
)


def available_random_distributions() -> tuple[str, ...]:
    """Return registered random-parameter distribution names."""

    return tuple(sorted(_DISTRIBUTIONS))


def get_random_distribution(name: str) -> DistributionSpec:
    """Resolve a random-parameter distribution by name."""

    key = _canonical_name(name)
    try:
        return _DISTRIBUTIONS[key]
    except KeyError as exc:
        valid = ", ".join(available_random_distributions())
        raise ValueError(
            f"Unknown random-parameter distribution {name!r}. Valid: {valid}."
        ) from exc


def make_random_parameter_spec(
    name: str,
    *,
    scope: str,
    distribution: str = "normal",
    resample_on_initialize: bool = True,
    seed: int | None = None,
    reparameterized: bool = True,
    rng_name: str | None = None,
    **distribution_parameters,
) -> RandomParameterSpec:
    """Construct and validate a random-parameter declaration."""

    name = str(name)
    if not name:
        raise ValueError("Random parameter name cannot be empty.")
    scope = str(scope).strip().lower()
    if scope not in {"global", "range", "batch"}:
        raise ValueError(
            "Random parameter scope must be one of 'global', 'range', or 'batch'."
        )
    dist = get_random_distribution(distribution)
    params = dict(dist.parameter_defaults)
    unknown = set(distribution_parameters) - set(params)
    if unknown:
        valid = ", ".join(params)
        raise ValueError(
            f"Unknown parameter(s) for {dist.name!r} random distribution: "
            f"{sorted(unknown)}. Valid parameters are: {valid}."
        )
    params.update(distribution_parameters)
    return RandomParameterSpec(
        name=name,
        scope=scope,
        distribution=dist.name,
        params=_readonly(params),
        constraints=_readonly(dist.parameter_constraints),
        resample_on_initialize=bool(resample_on_initialize),
        seed=None if seed is None else int(seed),
        reparameterized=bool(reparameterized),
        rng_name=rng_name,
    )


def sample_random_parameter(
    spec: RandomParameterSpec,
    parameters: Mapping[str, torch.Tensor],
    rng,
    shape: tuple[int, ...],
    *,
    device,
    dtype,
) -> torch.Tensor:
    """Sample a random parameter buffer from resolved distribution parameters."""

    dist = get_random_distribution(spec.distribution)
    sample = dist.sampler(parameters, rng, shape, device=device, dtype=dtype)
    return sample if spec.reparameterized else sample.detach()
