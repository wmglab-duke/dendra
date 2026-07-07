"""Random parameter and runtime-noise declarations for Dendra models."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

__all__ = [
    "RandomParameterSpec",
    "RuntimeNoiseSpec",
    "DistributionSpec",
    "available_random_distributions",
    "get_random_distribution",
    "register_random_distribution",
    "make_random_parameter_spec",
    "make_runtime_noise_spec",
    "sample_random_parameter",
    "sample_runtime_noise",
]

_ALLOWED_CONSTRAINTS = {"real", "positive", "negative"}
_ALLOWED_SCOPES = {"global", "range", "batch"}
_ALLOWED_NOISE_PHASES = {"pre_state"}
_ALLOWED_NOISE_SCALES = {"standard", "dW", "white"}


def _readonly(mapping):
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class RandomParameterSpec:
    """Specification for a sampled, usually initialization-time, parameter buffer."""

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


@dataclass(frozen=True)
class RuntimeNoiseSpec:
    """Specification for a detached runtime-noise buffer.

    Runtime noise is intended for stochastic drives in mechanisms/states during
    simulation.  Samples are written into stable buffers in-place under
    ``torch.no_grad()`` by Dendra's fast path, so gradients do not flow through
    the distribution parameters.  For differentiable stochastic state dynamics,
    prefer first-class State SDE support via ``State.DIFFUSION`` and an SDE
    integration method.
    """

    name: str
    scope: str
    distribution: str
    params: Mapping[str, object]
    constraints: Mapping[str, str]
    seed: int | None = None
    rng_name: str | None = None
    cadence: str = "step"
    phase: str = "pre_state"
    scale: str = "standard"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}_{p}" for p in self.params)

    @property
    def effective_rng_name(self) -> str:
        return self.rng_name or f"{self.name}_rng"

    def full_parameter_name(self, distribution_parameter: str) -> str:
        return f"{self.name}_{distribution_parameter}"


@dataclass(frozen=True)
class DistributionSpec:
    """Registered random distribution.

    Parameters
    ----------
    name:
        Canonical distribution name.
    parameter_defaults:
        Default scalar/tensor values for each distribution parameter.
    parameter_constraints:
        Mapping from parameter names to Dendra parameter constraints. Supported
        constraints are ``"real"``, ``"positive"``, and ``"negative"``.
    sampler:
        Callable with signature ``sampler(parameters, rng, shape, *, device,
        dtype)``.  The sampler must return a tensor broadcastable to ``shape``.
    """

    name: str
    parameter_defaults: Mapping[str, object]
    parameter_constraints: Mapping[str, str]
    sampler: Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# Distribution registry


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
    """Sample a true inverse-CDF truncated normal distribution."""

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
    """Register a distribution for RAND/NOISE declarations.

    ``sampler`` receives tensor-valued distribution parameters after all Dendra
    overrides and parametrizations have been applied, plus a Dendra ``RNGModule``.
    """

    key = _canonical_name(name)
    if key in _DISTRIBUTIONS and not overwrite:
        raise ValueError(
            f"Random distribution {name!r} is already registered. Pass "
            "overwrite=True to replace it."
        )
    if sampler is None or not callable(sampler):
        raise TypeError("register_random_distribution requires a callable sampler.")

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

    spec = DistributionSpec(key, _readonly(defaults), _readonly(constraints), sampler)
    _DISTRIBUTIONS[key] = spec
    return spec


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
    """Return registered random distribution names."""

    return tuple(sorted(_DISTRIBUTIONS))


def get_random_distribution(name: str) -> DistributionSpec:
    """Resolve a registered random distribution by name."""

    key = _canonical_name(name)
    try:
        return _DISTRIBUTIONS[key]
    except KeyError as exc:
        valid = ", ".join(available_random_distributions())
        raise ValueError(
            f"Unknown random distribution {name!r}. Valid: {valid}."
        ) from exc


# ---------------------------------------------------------------------------
# Declarations


def _validate_scope(scope: str) -> str:
    scope = str(scope).strip().lower()
    if scope not in _ALLOWED_SCOPES:
        raise ValueError("Random scope must be one of 'global', 'range', or 'batch'.")
    return scope


def _merged_distribution_params(distribution: str, distribution_parameters):
    dist = get_random_distribution(distribution)
    params = dict(dist.parameter_defaults)
    unknown = set(distribution_parameters) - set(params)
    if unknown:
        valid = ", ".join(params)
        raise ValueError(
            f"Unknown parameter(s) for {dist.name!r} distribution: "
            f"{sorted(unknown)}. Valid parameters are: {valid}."
        )
    params.update(distribution_parameters)
    return dist, params


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
    """Construct and validate a quenched random-parameter declaration."""

    name = str(name)
    if not name:
        raise ValueError("Random parameter name cannot be empty.")
    scope = _validate_scope(scope)
    dist, params = _merged_distribution_params(distribution, distribution_parameters)
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


def make_runtime_noise_spec(
    name: str,
    *,
    scope: str,
    distribution: str = "normal",
    seed: int | None = None,
    rng_name: str | None = None,
    cadence: str = "step",
    phase: str = "pre_state",
    scale: str = "standard",
    **distribution_parameters,
) -> RuntimeNoiseSpec:
    """Construct and validate a detached runtime-noise declaration."""

    name = str(name)
    if not name:
        raise ValueError("Runtime noise name cannot be empty.")
    scope = _validate_scope(scope)
    dist, params = _merged_distribution_params(distribution, distribution_parameters)

    cadence_out = str(cadence).strip().lower()
    if cadence_out != "step":
        raise ValueError("Runtime noise v1 supports cadence='step' only.")

    phase_norm = str(phase).strip().lower()
    if phase_norm not in _ALLOWED_NOISE_PHASES:
        raise ValueError("Runtime noise v1 supports phase='pre_state' only.")

    scale_norm = str(scale).strip().lower()
    if scale_norm not in _ALLOWED_NOISE_SCALES:
        raise ValueError(
            f"Runtime noise scale must be one of {sorted(_ALLOWED_NOISE_SCALES)}."
        )

    return RuntimeNoiseSpec(
        name=name,
        scope=scope,
        distribution=dist.name,
        params=_readonly(params),
        constraints=_readonly(dist.parameter_constraints),
        seed=None if seed is None else int(seed),
        rng_name=rng_name,
        cadence=cadence_out,
        phase=phase_norm,
        scale=scale_norm,
    )


# ---------------------------------------------------------------------------
# Sampling


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


def _scale_runtime_noise(sample: torch.Tensor, dt, scale: str) -> torch.Tensor:
    if scale == "standard":
        return sample
    if dt is None:
        raise ValueError(f"Runtime noise scale {scale!r} requires dt.")
    dt_t = torch.as_tensor(dt, device=sample.device, dtype=sample.dtype)
    root_dt = torch.sqrt(dt_t)
    if scale == "dW":
        return sample * root_dt
    if scale == "white":
        return sample / root_dt
    raise ValueError(f"Unknown runtime noise scale: {scale!r}")


def sample_runtime_noise(
    spec: RuntimeNoiseSpec,
    parameters: Mapping[str, torch.Tensor],
    rng,
    shape: tuple[int, ...],
    *,
    device,
    dtype,
    dt=None,
) -> torch.Tensor:
    """Sample a detached runtime-noise buffer."""

    dist = get_random_distribution(spec.distribution)
    sample = dist.sampler(parameters, rng, shape, device=device, dtype=dtype)
    sample = _scale_runtime_noise(sample, dt, spec.scale)
    return sample.detach()
