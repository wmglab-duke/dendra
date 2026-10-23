"""Random parameter and runtime-noise declarations for Dendra models."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, NamedTuple

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
    "sample_random_parameter_draw",
    "evaluate_random_parameter",
    "sample_random_parameter_with_draw",
    "RandomParameterDrawRequest",
    "sample_random_parameter_draws",
    "prepare_rng_for_random_parameter",
    "RuntimeNoiseDrawRequest",
    "sample_runtime_noise_draws",
    "sample_runtime_noise_draw",
    "evaluate_runtime_noise",
    "sample_runtime_noise_with_draw",
    "sample_runtime_noise",
    "iter_initialization_random_declarations",
]

_ALLOWED_CONSTRAINTS = {"real", "positive", "negative"}
_ALLOWED_SCOPES = {"global", "range", "batch"}
_ALLOWED_NOISE_PHASES = {"pre_state"}
_ALLOWED_NOISE_SCALES = {"standard", "dW", "white"}
_SQRT_TWO = math.sqrt(2.0)


def _readonly(mapping):
    return MappingProxyType(dict(mapping))


def clone_random_parameter_tree(value, *, memo=None, detach=False):
    """Own declaration/provenance tensors without copying an autograd graph."""
    memo = {} if memo is None else memo
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if torch.is_tensor(value):
        cloned = value.detach().clone()
        if isinstance(value, torch.nn.Parameter) and not detach:
            cloned = torch.nn.Parameter(cloned, requires_grad=value.requires_grad)
        elif not detach:
            cloned.requires_grad_(value.requires_grad)
    elif isinstance(value, Mapping):
        copied = {
            name: clone_random_parameter_tree(child, memo=memo, detach=detach)
            for name, child in value.items()
        }
        cloned = _readonly(copied) if isinstance(value, MappingProxyType) else copied
    elif isinstance(value, tuple):
        children = tuple(
            clone_random_parameter_tree(child, memo=memo, detach=detach)
            for child in value
        )
        cloned = type(value)(*children) if hasattr(value, "_fields") else children
    elif isinstance(value, list):
        cloned = [
            clone_random_parameter_tree(child, memo=memo, detach=detach)
            for child in value
        ]
    else:
        cloned = copy.deepcopy(value, memo)
    memo[identity] = cloned
    return cloned


class _ImmutableSpec:
    def __deepcopy__(self, memo):
        cloned = object.__new__(type(self))
        memo[id(self)] = cloned
        for name in self.__dataclass_fields__:
            object.__setattr__(
                cloned,
                name,
                clone_random_parameter_tree(getattr(self, name), memo=memo),
            )
        return cloned


@dataclass(frozen=True)
class RandomParameterSpec(_ImmutableSpec):
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
    reseed_on_initialize: bool = False

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}_{p}" for p in self.params)

    @property
    def effective_rng_name(self) -> str:
        return self.rng_name or f"{self.name}_rng"

    def full_parameter_name(self, distribution_parameter: str) -> str:
        return f"{self.name}_{distribution_parameter}"


@dataclass(frozen=True)
class RuntimeNoiseSpec(_ImmutableSpec):
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
class DistributionSpec(_ImmutableSpec):
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
    draw_sampler:
        Optional ``draw_sampler(rng, shape, *, device, dtype)`` producing a
        Tensor tree of parameter-independent base draws.
    transform:
        Optional pure ``transform(parameters, draw)`` converting those base
        draws to samples. Both split hooks are required together. Native
        sampling uses the split hooks when provided; legacy-only samplers
        remain available for imperative execution.
    """

    name: str
    parameter_defaults: Mapping[str, object]
    parameter_constraints: Mapping[str, str]
    sampler: Callable[..., torch.Tensor]
    draw_sampler: Callable | None = None
    transform: Callable | None = None


# ---------------------------------------------------------------------------
# Distribution registry


def _canonical_name(name: str) -> str:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Random distribution name cannot be empty.")
    return key


def _normal_draw(rng, shape, *, device, dtype):
    return rng.randn(shape, device=device, dtype=dtype)


def _uniform_draw(rng, shape, *, device, dtype):
    return rng.rand(shape, device=device, dtype=dtype)


def _normal_transform(parameters, draw):
    return parameters["mu"] + parameters["sigma"] * draw


def _lognormal_transform(parameters, draw):
    return torch.exp(_normal_transform(parameters, draw))


def _uniform_transform(parameters, draw):
    return parameters["low"] + (parameters["high"] - parameters["low"]) * draw


def _standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / _SQRT_TWO))


def _truncated_normal_transform(parameters, draw):
    """Sample a true inverse-CDF truncated normal distribution."""

    mu = parameters["mu"]
    sigma = parameters["sigma"]
    low = parameters["low"]
    high = parameters["high"]

    alpha = (low - mu) / sigma
    beta = (high - mu) / sigma
    cdf_low = _standard_normal_cdf(alpha)
    cdf_high = _standard_normal_cdf(beta)

    finfo = torch.finfo(draw.dtype)
    eps = finfo.eps
    cdf_high = torch.maximum(cdf_high, cdf_low + eps)

    p = cdf_low + (cdf_high - cdf_low) * draw
    p = torch.clamp(p, eps, 1.0 - eps)
    sample = mu + sigma * _SQRT_TWO * torch.erfinv(2.0 * p - 1.0)
    return torch.minimum(torch.maximum(sample, low), high)


def _split_sampler(draw_sampler, transform):
    def sampler(parameters, rng, shape, *, device, dtype):
        draw = draw_sampler(rng, shape, device=device, dtype=dtype)
        return transform(parameters, draw)

    return sampler


_DISTRIBUTIONS: dict[str, DistributionSpec] = {}


def register_random_distribution(
    name: str,
    parameter_defaults: Mapping[str, object],
    parameter_constraints: Mapping[str, str] | None = None,
    sampler: Callable[..., torch.Tensor] | None = None,
    *,
    overwrite: bool = False,
    draw_sampler: Callable | None = None,
    transform: Callable | None = None,
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
    if (draw_sampler is None) != (transform is None):
        raise ValueError("draw_sampler and transform must be supplied together.")
    if draw_sampler is not None:
        if not callable(draw_sampler) or not callable(transform):
            raise TypeError("draw_sampler and transform must be callable.")
        sampler = _split_sampler(draw_sampler, transform)
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

    spec = DistributionSpec(
        key,
        _readonly(defaults),
        _readonly(constraints),
        sampler,
        draw_sampler,
        transform,
    )
    _DISTRIBUTIONS[key] = spec
    return spec


register_random_distribution(
    "normal",
    {"mu": 0.0, "sigma": 1.0},
    {"mu": "real", "sigma": "positive"},
    draw_sampler=_normal_draw,
    transform=_normal_transform,
)
register_random_distribution(
    "lognormal",
    {"mu": 0.0, "sigma": 1.0},
    {"mu": "real", "sigma": "positive"},
    draw_sampler=_normal_draw,
    transform=_lognormal_transform,
)
register_random_distribution(
    "uniform",
    {"low": 0.0, "high": 1.0},
    {"low": "real", "high": "real"},
    draw_sampler=_uniform_draw,
    transform=_uniform_transform,
)
register_random_distribution(
    "truncated_normal",
    {"mu": 0.0, "sigma": 1.0, "low": -2.0, "high": 2.0},
    {"mu": "real", "sigma": "positive", "low": "real", "high": "real"},
    draw_sampler=_uniform_draw,
    transform=_truncated_normal_transform,
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
    reseed_on_initialize: bool = False,
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
        reseed_on_initialize=bool(reseed_on_initialize),
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
    if scale_norm == "dw":
        scale_norm = "dW"
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


def iter_initialization_random_declarations(owner):
    """Yield canonical owner initialization draws in their native order.

    Each item is ``(owner, kind, name, spec)``, where ``kind`` is ``"random"``
    or ``"noise"``. Mechanism initialization samples its nested State noise
    through the mechanism's runtime sampler before separately populating each
    State's parameters. That early noise draw consumes the shared stream even
    though the State's later initialization noise replaces its value.
    """
    states = tuple(getattr(owner, "DE", {}).values())
    for name, spec in owner.random_parameters.items():
        yield owner, "random", name, spec
    for source in (owner, *states):
        for name, spec in source.runtime_noises.items():
            yield source, "noise", name, spec
    for state in states:
        yield from iter_initialization_random_declarations(state)


def _distribution(spec):
    return (
        spec
        if isinstance(spec, DistributionSpec)
        else get_random_distribution(spec.distribution)
    )


def sample_random_parameter_draw(spec, rng, shape, *, device, dtype):
    """Draw explicit parameter-independent noise using the Native RNG calls."""
    dist = _distribution(spec)
    if dist.draw_sampler is None or dist.transform is None:
        raise ValueError(
            f"Random distribution {dist.name!r} has no pure draw_sampler/transform contract"
        )
    return dist.draw_sampler(rng, shape, device=device, dtype=dtype)


def evaluate_random_parameter(spec, parameters, draw):
    """Purely evaluate one distribution from its parameters and explicit draw."""
    dist = _distribution(spec)
    if dist.draw_sampler is None or dist.transform is None:
        raise ValueError(
            f"Random distribution {dist.name!r} has no pure draw_sampler/transform contract"
        )
    sample = dist.transform(parameters, draw)
    return sample if getattr(spec, "reparameterized", True) else sample.detach()


def sample_random_parameter_with_draw(spec, parameters, rng, shape, *, device, dtype):
    """Sample through the shared pure core, retaining optional base provenance."""
    dist = _distribution(spec)
    if dist.draw_sampler is not None and dist.transform is not None:
        draw = sample_random_parameter_draw(
            dist, rng, shape, device=device, dtype=dtype
        )
        return evaluate_random_parameter(spec, parameters, draw), draw
    sample = dist.sampler(parameters, rng, shape, device=device, dtype=dtype)
    return (sample if spec.reparameterized else sample.detach()), None


class RandomParameterDrawRequest(NamedTuple):
    """One host draw in Native order, keyed by its explicit shared RNG stream."""

    rng_key: str
    spec: RandomParameterSpec | RuntimeNoiseSpec | DistributionSpec
    shape: tuple[int, ...]
    device: torch.device | str
    dtype: torch.dtype
    reseed_on_initialize: bool = False


class RuntimeNoiseDrawRequest(NamedTuple):
    """One detached runtime base draw from an explicit shared RNG stream."""

    rng_key: str
    spec: RuntimeNoiseSpec | DistributionSpec
    shape: tuple[int, ...]
    device: torch.device | str
    dtype: torch.dtype


def prepare_rng_for_random_parameter(
    spec, rng, generation, *, reseed_on_initialize=None
):
    """Apply one declaration's ordered initialization reseed request."""
    requested = (
        getattr(spec, "reseed_on_initialize", False)
        if reseed_on_initialize is None
        else reseed_on_initialize
    )
    if not requested or generation is None:
        return False
    reset = getattr(rng, "reseed_for_generation", None)
    if reset is None:
        raise TypeError("reseed_on_initialize requires an RNGModule stream")
    return reset(generation)


def sample_random_parameter_draws(
    requests, rng_states, *, initialization=True, samplers=None
):
    """Sample an ordered draw packet from independent explicit RNG checkpoints.

    Requests with the same key consume one shared stream in sequence. The
    returned checkpoints include every supplied stream and retain base seeds
    and deferred per-device state, without mutating a source model or packet.
    This host adapter is separate from pure distribution evaluation. An
    optional caller-owned ``samplers`` mapping lets adjacent owners consume
    isolated streams in a shared order. Those modules advance in place;
    ``rng_states`` and source models remain unchanged.
    """
    from .rng import RNGModule

    modules = (
        {key: RNGModule.from_checkpoint(state) for key, state in rng_states.items()}
        if samplers is None
        else samplers
    )
    generation = object() if initialization else None
    draws = []
    for request in requests:
        if not isinstance(
            request, (RandomParameterDrawRequest, RuntimeNoiseDrawRequest)
        ):
            raise TypeError(
                "random draw requests must be RandomParameterDrawRequest or "
                "RuntimeNoiseDrawRequest instances"
            )
        if request.rng_key not in modules:
            raise KeyError(
                f"Missing random-parameter RNG checkpoint {request.rng_key!r}"
            )
        prepare_rng_for_random_parameter(
            request.spec,
            modules[request.rng_key],
            generation,
            reseed_on_initialize=(
                True if getattr(request, "reseed_on_initialize", False) else None
            ),
        )
        draw = sample_random_parameter_draw(
            request.spec,
            modules[request.rng_key],
            request.shape,
            device=request.device,
            dtype=request.dtype,
        )
        draws.append(clone_random_parameter_tree(draw, detach=True))
    return tuple(draws), {
        key: module.get_extra_state() for key, module in modules.items()
    }


def sample_runtime_noise_draws(requests, rng_states, *, samplers=None):
    """Return detached base draws and advanced checkpoints.

    This host adapter shares the ordered stream machinery used by RAND
    initialization, including its optional caller-owned samplers. Runtime
    draws never apply initialization reseeding.
    """
    return sample_random_parameter_draws(
        requests, rng_states, initialization=False, samplers=samplers
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

    return sample_random_parameter_with_draw(
        spec, parameters, rng, shape, device=device, dtype=dtype
    )[0]


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


def sample_runtime_noise_draw(spec, rng, shape, *, device, dtype):
    """Sample a parameter-independent runtime base draw through explicit RNG."""
    return sample_random_parameter_draw(spec, rng, shape, device=device, dtype=dtype)


def evaluate_runtime_noise(spec, parameters, draw, *, dt=None):
    """Evaluate detached runtime noise from explicit parameters and a base draw.

    Distribution evaluation and timestep scaling use the same algebra as native
    sampling. The result is detached from parameters, base draws, and ``dt``.
    """
    dist = _distribution(spec)
    if dist.draw_sampler is None or dist.transform is None:
        raise ValueError(
            f"Random distribution {dist.name!r} has no pure draw_sampler/transform contract"
        )
    sample = dist.transform(parameters, draw)
    return _scale_runtime_noise(sample, dt, spec.scale).detach()


def sample_runtime_noise_with_draw(
    spec, parameters, rng, shape, *, device, dtype, dt=None
):
    """Sample detached noise through the pure core or a legacy native sampler."""
    dist = _distribution(spec)
    if dist.draw_sampler is not None and dist.transform is not None:
        draw = sample_runtime_noise_draw(dist, rng, shape, device=device, dtype=dtype)
        return evaluate_runtime_noise(spec, parameters, draw, dt=dt), draw
    sample = dist.sampler(parameters, rng, shape, device=device, dtype=dtype)
    return _scale_runtime_noise(sample, dt, spec.scale).detach(), None


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

    return sample_runtime_noise_with_draw(
        spec, parameters, rng, shape, device=device, dtype=dtype, dt=dt
    )[0]
