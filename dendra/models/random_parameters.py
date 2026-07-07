"""Random parameter declarations for Dendra ``Parameterized`` modules."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

__all__ = [
    "RandomParameterSpec",
    "DistributionSpec",
    "available_random_distributions",
    "get_random_distribution",
    "make_random_parameter_spec",
    "sample_random_parameter",
]


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


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    parameter_defaults: Mapping[str, object]
    parameter_constraints: Mapping[str, str]


_DISTRIBUTIONS: dict[str, DistributionSpec] = {
    "normal": DistributionSpec(
        "normal",
        _readonly({"mu": 0.0, "sigma": 1.0}),
        _readonly({"mu": "real", "sigma": "positive"}),
    ),
    "lognormal": DistributionSpec(
        "lognormal",
        _readonly({"mu": 0.0, "sigma": 1.0}),
        _readonly({"mu": "real", "sigma": "positive"}),
    ),
    "uniform": DistributionSpec(
        "uniform",
        _readonly({"low": 0.0, "high": 1.0}),
        _readonly({"low": "real", "high": "real"}),
    ),
}


def available_random_distributions() -> tuple[str, ...]:
    """Return registered random-parameter distribution names."""

    return tuple(sorted(_DISTRIBUTIONS))


def get_random_distribution(name: str) -> DistributionSpec:
    """Resolve a random-parameter distribution by name."""

    key = str(name).strip().lower()
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

    if spec.distribution == "normal":
        eps = rng.randn(shape, device=device, dtype=dtype)
        sample = parameters["mu"] + parameters["sigma"] * eps
    elif spec.distribution == "lognormal":
        eps = rng.randn(shape, device=device, dtype=dtype)
        sample = torch.exp(parameters["mu"] + parameters["sigma"] * eps)
    elif spec.distribution == "uniform":
        u = rng.rand(shape, device=device, dtype=dtype)
        sample = parameters["low"] + (parameters["high"] - parameters["low"]) * u
    else:  # pragma: no cover
        raise ValueError(
            f"Unsupported random-parameter distribution: {spec.distribution!r}"
        )
    return sample if spec.reparameterized else sample.detach()
