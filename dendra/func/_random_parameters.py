"""Explicit initialization draws and retained RAND materializations."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from torch.utils import _pytree

from dendra.helpers import current_check_purity
from dendra.models.networks._population_cache import (
    _restore_parameter,
    restore_population_cache_values,
)
from dendra.models.parametric import Parameterized
from dendra.models.random_parameters import (
    get_random_distribution,
    sample_random_parameter_draw,
)
from dendra.models.rng import RNGModule

from ._rng_diagnostics import _rng_diagnostic_scope
from ._rng_frames import _rng_extra, _rng_values, _seed_tensor
from ._types import FunctionalizationError, RandomParameterDraw


def _clone(tree):
    return _pytree.tree_map(lambda value: value.detach().clone(), tree)


def _owners(population):
    from ._population import _parameterized_owner_paths

    return tuple(_parameterized_owner_paths(population))


def random_declarations(population):
    return tuple(
        (path, name, spec)
        for path, owner in _owners(population)
        for name, spec in owner.random_parameters.items()
    )


def _current_draw_leaves(
    distribution, rng, sample, captured_spec, captured_leaves, *, check_purity=False
):
    """Probe a target's draw schema without consuming its retained stream."""
    with _rng_diagnostic_scope(sample.device, enabled=check_purity) as diagnostic:
        draw = sample_random_parameter_draw(
            distribution,
            RNGModule.from_checkpoint(rng.get_extra_state()),
            tuple(sample.shape),
            device=sample.device,
            dtype=sample.dtype,
        )
    if diagnostic.changes:
        raise FunctionalizationError(
            f"RAND draw sampler consumed global RNG state {sorted(diagnostic.changes)}; use its explicit RNGModule"
        )
    leaves, spec = _pytree.tree_flatten(draw)
    if (
        spec != captured_spec
        or not leaves
        or not all(torch.is_tensor(value) for value in leaves)
        or any(
            tuple(value.shape) != tuple(reference.shape)
            for value, reference in zip(captured_leaves, leaves, strict=True)
        )
    ):
        raise FunctionalizationError(
            "Captured RAND draw layout differs from the current sampler contract; resample the native model before lowering"
        )
    return tuple(
        value.detach().to(reference).clone()
        for value, reference in zip(captured_leaves, leaves, strict=True)
    )


def cache_metadata_tensors(value, device, *, name="", metadata=False):
    """Normalize Native RNG/provenance metadata into initialization operands."""
    if torch.is_tensor(value):
        return value.detach().to(device).clone() if metadata else value.detach().clone()
    if name == "base_seed" and isinstance(value, int):
        return {
            "seed": _seed_tensor(value, device),
            "negative": torch.tensor(value < 0, device=device, dtype=torch.bool),
        }
    if isinstance(value, dict):
        return {
            key: cache_metadata_tensors(child, device, name=key, metadata=True)
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        children = tuple(
            cache_metadata_tensors(child, device, metadata=True) for child in value
        )
        return (
            type(value)(*children)
            if hasattr(value, "_fields")
            else type(value)(children)
        )
    if isinstance(value, (bool, int, float)):
        return torch.tensor(value, device=device)
    return value


def _declaration_signature(population):
    return tuple(
        (
            path,
            name,
            spec.scope,
            spec.distribution,
            tuple(spec.params),
            tuple(spec.constraints.items()),
            spec.resample_on_initialize,
            spec.seed,
            spec.reparameterized,
            spec.rng_name,
            spec.reseed_on_initialize,
        )
        for path, name, spec in random_declarations(population)
    )


@dataclass(frozen=True)
class _RandomEntry:
    key: str
    path: str
    name: str
    spec: object
    distribution: object
    stream: str
    draw_spec: object
    draw_names: tuple
    parameter_names: tuple
    shape: tuple
    dtype: torch.dtype
    device: torch.device


class RandomParameterPlan:
    """Frozen stream/layout metadata; every numerical operand is explicit."""

    def __init__(self, population):
        self.population = population
        self.check_purity = current_check_purity()
        from ._population import _initialization_transform_hooks

        # Population sampling precedes authored pre-initialize parameter
        # writes. Its selected distribution operands therefore own the
        # incoming graph even after those writes replace the explicit raw
        # parameter frame used by later preparation.
        self._retain_population_initialization_frame = bool(
            population.random_parameters
        ) and any(
            reference.startswith("parameters.")
            for hook in _initialization_transform_hooks(population, "pre")
            for reference in population._initialization_transforms[
                hook.action_name
            ].writes
        )
        self._declarations = _declaration_signature(population)
        self.entries = []
        from ._runtime_noise import _stream_topology, stream_inventory

        self.streams, stream_ids = stream_inventory(population)
        self._stream_topology = _stream_topology(population)
        self.defaults = {}
        self._bindings = []
        self._rng_bindings = {}
        self._signatures = []
        from ._population import _initialization_callable_dependency_signature

        owners = dict(_owners(population))
        self._rng_bindings = {
            stream: getattr(owners[path], name)
            for stream, (path, name) in self.streams.items()
        }
        stream_by_identity = {
            id(rng): stream for stream, rng in self._rng_bindings.items()
        }
        self._rng_cache_bindings = tuple(
            (
                f"{path + '.' if path else ''}_extra_state",
                stream_by_identity[id(module)],
                module,
            )
            for path, module in population.named_modules(remove_duplicate=False)
            if id(module) in stream_by_identity
        )
        self.noise = None
        for path, name, spec in random_declarations(population):
            owner = owners[path]
            for adapter in (
                "_sample_random_parameters",
                "_prepare_random_parameter_state",
                "_apply_random_parameter_state",
                "_random_parameter_state_for_checkpoint",
            ):
                if adapter in owner.__dict__ or inspect.getattr_static(
                    type(owner), adapter
                ) is not inspect.getattr_static(Parameterized, adapter):
                    raise FunctionalizationError(
                        f"RAND requires canonical {adapter}; custom sampling adapters are currently unsupported"
                    )
            distribution = get_random_distribution(spec.distribution)
            if distribution.draw_sampler is None or distribution.transform is None:
                raise FunctionalizationError(
                    f"RAND distribution {spec.distribution!r} requires a pure draw_sampler and transform contract"
                )
            signatures = (
                tuple(
                    _initialization_callable_dependency_signature(
                        function,
                        label=f"RAND {spec.distribution}.{label}",
                        owner=owner,
                        strict=True,
                    )
                    for label, function in (
                        ("draw_sampler", distribution.draw_sampler),
                        ("transform", distribution.transform),
                    )
                )
                if self.check_purity
                else ()
            )
            self._signatures.append(signatures)
            draw = getattr(owner, "_random_parameter_draws", {}).get(name)
            resolved = getattr(owner, "_random_parameter_parameters", {}).get(name)
            if draw is None or resolved is None:
                raise FunctionalizationError(
                    f"RAND {path + '.' if path else ''}{name} has no captured draw provenance; initialize or resample the native model before lowering"
                )
            leaves, draw_spec = _pytree.tree_flatten(draw)
            if not leaves or not all(torch.is_tensor(value) for value in leaves):
                raise FunctionalizationError(
                    "RAND pure draws must be nonempty Tensor trees"
                )
            key = f"{path + '.' if path else ''}{name}"
            rng = getattr(owner, spec.effective_rng_name)
            if type(rng) is not RNGModule or any(
                name in rng.__dict__
                for name in (
                    "rand",
                    "randn",
                    "_generator_for",
                    "_rng",
                    "init",
                    "get_extra_state",
                    "set_extra_state",
                    "restore_rng_checkpoint",
                    "reseed_for_generation",
                )
            ):
                raise FunctionalizationError(
                    "RAND requires a canonical RNGModule stream"
                )
            stream = stream_ids.setdefault(id(rng), f"stream_{len(stream_ids)}")
            self.streams.setdefault(stream, (path, spec.effective_rng_name))
            self._rng_bindings.setdefault(stream, rng)
            sample = owner._buffers[name]
            leaves = _current_draw_leaves(
                distribution,
                rng,
                sample,
                draw_spec,
                leaves,
                check_purity=self.check_purity,
            )
            draw_names = tuple(f"random.draws.{key}.{i}" for i in range(len(leaves)))
            parameter_names = tuple(f"random.parameters.{key}.{p}" for p in spec.params)
            entry = _RandomEntry(
                key,
                path,
                name,
                spec,
                distribution,
                stream,
                draw_spec,
                draw_names,
                parameter_names,
                tuple(sample.shape),
                sample.dtype,
                sample.device,
            )
            self.entries.append(entry)
            self.defaults.update(zip(draw_names, _clone(leaves), strict=True))
            self.defaults.update(
                (
                    public,
                    resolved[p]
                    .detach()
                    .to(getattr(owner, spec.full_parameter_name(p)))
                    .clone(),
                )
                for p, public in zip(spec.params, parameter_names, strict=True)
            )
            self.defaults[f"random.values.{key}"] = sample.detach().clone()
            self.defaults[f"random.fresh.{key}"] = torch.zeros(
                (), dtype=torch.bool, device=sample.device
            )
            if self._holds_initialization_frame(entry):
                self.defaults[f"random.initialization_frame.{key}"] = torch.zeros(
                    (), dtype=torch.bool, device=sample.device
                )
            self._bindings.append(
                (path, name, spec, distribution.draw_sampler, distribution.transform)
            )
        self.entries = tuple(self.entries)
        self.schema = {
            name: (tuple(value.shape), value.dtype, value.device)
            for name, value in self.defaults.items()
        }

    def _holds_initialization_frame(self, entry):
        return self._retain_population_initialization_frame and not entry.path

    def _constant_value(self, name, value):
        shape, dtype, device = self.schema[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise FunctionalizationError(
                f"Captured RAND value {name!r} has incompatible Tensor shape"
            )
        return value.detach().to(device=device, dtype=dtype).clone()

    def validate_source(self):
        if self.noise is not None:
            self.noise.validate_source()
        owners = dict(_owners(self.population))
        if _declaration_signature(self.population) != self._declarations:
            raise FunctionalizationError(
                "RAND declarations changed; lower the Population again"
            )
        self.validate_stream_topology(self.population)
        from ._population import _initialization_callable_dependency_signature

        for stream, (path, name) in self.streams.items():
            rng = getattr(owners[path], name)
            if (
                rng is not self._rng_bindings[stream]
                or type(rng) is not RNGModule
                or any(
                    method in rng.__dict__
                    for method in (
                        "rand",
                        "randn",
                        "_generator_for",
                        "_rng",
                        "init",
                        "get_extra_state",
                        "set_extra_state",
                        "restore_rng_checkpoint",
                        "reseed_for_generation",
                    )
                )
            ):
                raise FunctionalizationError(
                    "Population RNG stream adapters changed; lower again"
                )
        for index, (path, name, spec, sampler, transform) in enumerate(self._bindings):
            owner = owners[path]
            for adapter in (
                "_sample_random_parameters",
                "_prepare_random_parameter_state",
                "_apply_random_parameter_state",
                "_random_parameter_state_for_checkpoint",
            ):
                if adapter in owner.__dict__ or inspect.getattr_static(
                    type(owner), adapter
                ) is not inspect.getattr_static(Parameterized, adapter):
                    raise FunctionalizationError(
                        f"RAND sampling adapter {adapter} changed; lower again"
                    )
            current = get_random_distribution(spec.distribution)
            if (
                current.draw_sampler is not sampler
                or current.transform is not transform
            ):
                raise FunctionalizationError(
                    "RAND distribution contract changed; lower the Population again"
                )
            signatures = (
                tuple(
                    _initialization_callable_dependency_signature(
                        function,
                        label=f"RAND {spec.distribution}.{label}",
                        owner=owners[path],
                        strict=True,
                    )
                    for label, function in (
                        ("draw_sampler", current.draw_sampler),
                        ("transform", current.transform),
                    )
                )
                if self.check_purity
                else ()
            )
            if signatures != self._signatures[index]:
                raise FunctionalizationError(
                    "RAND distribution dependencies changed; lower the Population again"
                )
            if not owners[path]._random_parameter_initialized.get(name, False):
                raise FunctionalizationError(
                    "RAND buffer is not initialized; initialize and lower again"
                )

    def validate_stream_topology(self, population):
        from ._runtime_noise import _stream_topology

        if _stream_topology(population) != self._stream_topology:
            raise FunctionalizationError(
                "Population RNG stream sharing does not match this functional plan; lower again"
            )

    def validate_compatible_population(self, population):
        self.validate_stream_topology(population)
        if self.noise is not None:
            self.noise.validate_compatible_population(population)

    def extract(self, population=None):
        population = self.population if population is None else population
        owners = dict(_owners(population))
        constants = {}
        for entry in self.entries:
            owner = owners[entry.path]
            draw = owner._random_parameter_draws[entry.name]
            leaves, spec = _pytree.tree_flatten(draw)
            if spec != entry.draw_spec:
                raise FunctionalizationError("RAND draw layout changed; lower again")
            constants.update(
                (name, self._constant_value(name, value))
                for name, value in zip(entry.draw_names, leaves, strict=True)
            )
            constants.update(
                (
                    public,
                    self._constant_value(
                        public, owner._random_parameter_parameters[entry.name][p]
                    ),
                )
                for p, public in zip(
                    entry.spec.params, entry.parameter_names, strict=True
                )
            )
            constants[f"random.values.{entry.key}"] = (
                owner._buffers[entry.name].detach().clone()
            )
            constants[f"random.fresh.{entry.key}"] = torch.zeros(
                (), dtype=torch.bool, device=entry.device
            )
            if self._holds_initialization_frame(entry):
                constants[f"random.initialization_frame.{entry.key}"] = torch.zeros(
                    (), dtype=torch.bool, device=entry.device
                )
        rng = {
            stream: _rng_values(getattr(owners[path], name), population.device())
            for stream, (path, name) in self.streams.items()
        }
        return constants, {"rng": rng, "values": _clone(constants)}

    def inputs(self, constants):
        return {
            entry.key: (
                _pytree.tree_unflatten(
                    [constants[name] for name in entry.draw_names], entry.draw_spec
                ),
                {
                    p: constants[public]
                    for p, public in zip(
                        entry.spec.params, entry.parameter_names, strict=True
                    )
                },
                constants[f"random.values.{entry.key}"],
                constants[f"random.fresh.{entry.key}"],
                constants.get(f"random.initialization_frame.{entry.key}"),
            )
            for entry in self.entries
        }

    def _initialization_entries(self):
        """Select native ordered resampling and initial NOISE requests."""
        selected = tuple(
            entry for entry in self.entries if entry.spec.resample_on_initialize
        )
        if self.noise is None:
            scheduled = selected
        else:
            from dendra.models.random_parameters import (
                iter_initialization_random_declarations,
            )

            owners = dict(_owners(self.population))
            by_identity = {id(owner): path for path, owner in owners.items()}
            random_entries = {(entry.path, entry.name): entry for entry in selected}
            noise_entries = {
                (entry.path, entry.name): entry for entry in self.noise.entries
            }
            scheduled = []
            visited = set()
            for path, owner in owners.items():
                if id(owner) in visited:
                    continue
                for child, kind, name, _spec in iter_initialization_random_declarations(
                    owner
                ):
                    child_path = by_identity[id(child)]
                    visited.add(id(child))
                    entry = (random_entries if kind == "random" else noise_entries).get(
                        (child_path, name)
                    )
                    if entry is not None:
                        scheduled.append(entry)
            scheduled = tuple(scheduled)
        return scheduled

    def sample(self, state, *, samplers=None):
        from dendra.models.random_parameters import (
            RandomParameterDrawRequest,
            sample_random_parameter_draws,
        )

        scheduled = self._initialization_entries()
        requests = tuple(
            RandomParameterDrawRequest(
                entry.stream,
                entry.spec,
                entry.shape,
                entry.device,
                entry.dtype,
                getattr(entry.spec, "reseed_on_initialize", False),
            )
            for entry in scheduled
        )
        from ._population import (
            _global_rng_changes,
            _global_rng_snapshot,
            _restore_global_rng,
        )

        snapshot = _global_rng_snapshot(self.population.device())
        changes = set()
        try:
            sampled, checkpoints = sample_random_parameter_draws(
                requests,
                (
                    {
                        stream: _rng_extra(values)
                        for stream, values in state["random"]["rng"].items()
                    }
                    if samplers is None
                    else {}
                ),
                samplers=samplers,
            )
        finally:
            changes = _global_rng_changes(snapshot, self.population.device())
            _restore_global_rng(snapshot, self.population.device())
        if changes:
            raise FunctionalizationError(
                f"RAND draw sampler consumed global RNG state {sorted(changes)}; use its explicit RNGModule"
            )
        draws = {}
        for entry, draw in zip(scheduled, sampled, strict=True):
            leaves, spec = _pytree.tree_flatten(draw)
            if spec != entry.draw_spec:
                raise FunctionalizationError(
                    "RAND sampler changed its Tensor-tree layout"
                )
            draws.update(zip(entry.draw_names, _clone(leaves), strict=True))
        retained = state["random"]["values"]
        for entry in self.entries:
            if not entry.spec.resample_on_initialize:
                draws.update(
                    (name, retained[name].clone()) for name in entry.draw_names
                )
        return RandomParameterDraw(
            draws,
            {
                stream: _rng_values(extra, self.population.device())
                for stream, extra in checkpoints.items()
            },
        )

    def initialize_constants(self, constants, draw):
        if not isinstance(draw, RandomParameterDraw):
            raise ValueError(
                "RAND initialization requires a draw packet from sample_initialization_draw"
            )
        expected = {name for entry in self.entries for name in entry.draw_names}
        full_expected = expected | (
            {name for entry in self.noise.entries for name in entry.draw_names}
            if self.noise is not None
            else set()
        )
        if set(draw.draws) != full_expected:
            raise ValueError(
                "RAND initialization draw names do not match the lowered plan"
            )
        result = dict(constants)
        for name, value in draw.draws.items():
            if name not in expected:
                continue
            shape, dtype, device = self.schema[name]
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != shape
                or value.dtype != dtype
                or value.device != device
            ):
                raise ValueError(f"RAND draw {name!r} has incompatible Tensor schema")
            result[name] = value.clone()
        for entry in self.entries:
            result[f"random.fresh.{entry.key}"] = torch.full(
                (),
                entry.spec.resample_on_initialize,
                device=entry.device,
                dtype=torch.bool,
            )
            if self._holds_initialization_frame(entry):
                # Initialization starts a new incoming parameter frame. Sticky
                # RAND retains its numerical sample, then reconnects it to the
                # current pre-transform operands through the ordinary frozen
                # provenance route; a resampled RAND instead consumes them
                # directly. Neither route should retain a discarded old graph.
                result.update(
                    (name, result[name].detach()) for name in entry.parameter_names
                )
                result[f"random.initialization_frame.{entry.key}"] = torch.zeros(
                    (), device=entry.device, dtype=torch.bool
                )
        return result

    def retain_constants(self, constants, state):
        if state is None:
            return constants
        result = dict(constants)
        retained = state["random"]["values"]
        for entry in self.entries:
            if not entry.spec.resample_on_initialize:
                names = (
                    *entry.draw_names,
                    *entry.parameter_names,
                    f"random.values.{entry.key}",
                    f"random.fresh.{entry.key}",
                    *(
                        (f"random.initialization_frame.{entry.key}",)
                        if self._holds_initialization_frame(entry)
                        else ()
                    ),
                )
                result.update((name, retained[name]) for name in names)
        return result

    def finish_constants(self, constants, materialized):
        result = dict(constants)
        for entry in self.entries:
            sample, parameters = materialized[entry.key]
            result[f"random.values.{entry.key}"] = sample.detach().clone()
            holds_initialization_frame = self._holds_initialization_frame(entry)
            for p, public in zip(entry.spec.params, entry.parameter_names, strict=True):
                result[public] = (
                    parameters[p].clone()
                    if holds_initialization_frame
                    else parameters[p].detach().clone()
                )
            result[f"random.fresh.{entry.key}"] = torch.zeros(
                (), device=entry.device, dtype=torch.bool
            )
            if holds_initialization_frame:
                result[f"random.initialization_frame.{entry.key}"] = torch.ones(
                    (), device=entry.device, dtype=torch.bool
                )
        return result

    def state(self, constants, rng):
        return {
            "rng": _clone(rng),
            "values": {
                name: value.clone()
                for name, value in constants.items()
                if name.startswith("random.")
            },
        }

    def restore_cache(self, cache, constants, incoming_rng):
        from dendra.models.rng import should_restore_rng

        self.validate_cache(cache)
        result = dict(constants)
        for entry in self.entries:
            prefix = f"{entry.path}." if entry.path else ""
            metadata = cache[f"{prefix}_random_parameter_state"]
            leaves, spec = _pytree.tree_flatten(metadata["draws"][entry.name])
            if spec != entry.draw_spec:
                raise FunctionalizationError("Cached RAND draw layout is incompatible")
            result.update(
                zip(
                    entry.draw_names,
                    (
                        self._constant_value(name, value)
                        for name, value in zip(entry.draw_names, leaves, strict=True)
                    ),
                    strict=True,
                )
            )
            result.update(
                (
                    public,
                    self._constant_value(public, metadata["parameters"][entry.name][p]),
                )
                for p, public in zip(
                    entry.spec.params, entry.parameter_names, strict=True
                )
            )
            result[f"random.values.{entry.key}"] = (
                cache[f"{prefix}{entry.name}"].detach().clone()
            )
            result[f"random.fresh.{entry.key}"] = torch.zeros(
                (), device=entry.device, dtype=torch.bool
            )
            if self._holds_initialization_frame(entry):
                result[f"random.initialization_frame.{entry.key}"] = torch.zeros(
                    (), device=entry.device, dtype=torch.bool
                )
        rng = dict(incoming_rng)
        for path, stream, source in self._rng_cache_bindings:
            if not should_restore_rng(
                policy="cache",
                ignore_rng_on_load=getattr(source, "_ignore_rng_on_load", False),
            ):
                continue
            extra = cache[path]
            values = {**extra["base_seed"], **extra["rng_state"]}
            rng[stream] = {
                key: value.detach().to(rng[stream][key]).clone()
                for key, value in values.items()
            }
        return result, self.state(result, rng)

    def validate_native_cache(self, cache):
        """Validate Native RAND provenance before tensor normalization."""
        owners = dict(_owners(self.population))
        seen = set()
        for entry in self.entries:
            if entry.path in seen:
                continue
            seen.add(entry.path)
            prefix = f"{entry.path}." if entry.path else ""
            metadata = cache[f"{prefix}_random_parameter_state"]
            # Explicit normalized packets already contain tensor flags; their
            # schemas and values are validated by validate_cache instead.
            if not torch.is_tensor(metadata["version"]):
                owners[entry.path]._prepare_random_parameter_state(metadata)

    def validate_cache(self, cache):
        if torch._C._are_functorch_transforms_active():
            return
        for entry in self.entries:
            prefix = f"{entry.path}." if entry.path else ""
            metadata = cache[f"{prefix}_random_parameter_state"]
            if (
                metadata["version"].dtype != torch.int64
                or tuple(metadata["version"].shape) != ()
                or metadata["initialized"][entry.name].dtype != torch.bool
                or tuple(metadata["initialized"][entry.name].shape) != ()
            ):
                raise FunctionalizationError(
                    "RAND cache provenance requires scalar integer version and Boolean initialized flags"
                )
            valid = (metadata["version"] == 1) & metadata["initialized"][entry.name]
            message = "RAND cache requires version 1 and initialized sampled provenance; pre-sampling caches are currently unsupported"
            if torch.compiler.is_compiling():
                torch._assert_async(valid.all(), message)
            elif not bool(valid.all()):
                raise FunctionalizationError(message)

    def prepare_commit(self, population, state):
        """Build retained samples against the destination's live Parameters."""
        from dendra.models.random_parameters import evaluate_random_parameter

        owners = dict(_owners(population))
        constants = state["values"]
        rng = {}
        for stream, values in state["rng"].items():
            extra = _rng_extra(values)
            # Validate all byte streams before publishing any target binding.
            rng[stream] = RNGModule.from_checkpoint(extra).get_extra_state()
        samples, draws, parameters = {}, {}, {}
        derived = {}
        for entry in self.entries:
            owner = owners[entry.path]
            if entry.path not in derived:
                derived[entry.path] = owner._derive_parameter_buffers()
            current = {
                p: derived[entry.path][entry.spec.full_parameter_name(p)]
                for p in entry.spec.params
            }
            saved = {
                p: constants[public].detach().clone()
                for p, public in zip(
                    entry.spec.params, entry.parameter_names, strict=True
                )
            }
            anchored = restore_population_cache_values(
                current, saved, parameter_names=current
            )
            draw = _pytree.tree_unflatten(
                [constants[name].detach().clone() for name in entry.draw_names],
                entry.draw_spec,
            )
            evaluated = evaluate_random_parameter(entry.spec, anchored, draw).to(
                constants[f"random.values.{entry.key}"]
            )
            sample = _restore_parameter(
                evaluated, constants[f"random.values.{entry.key}"]
            )
            samples[entry.key] = (
                sample if entry.spec.reparameterized else sample.detach()
            )
            draws[entry.key], parameters[entry.key] = draw, saved
        return {
            "rng": rng,
            "samples": samples,
            "draws": draws,
            "parameters": parameters,
        }

    def publish_commit(self, population, prepared):
        """Install validated retained graph/numerical/RNG bindings."""
        owners = dict(_owners(population))
        for stream, (path, name) in self.streams.items():
            rng = getattr(owners[path], name)
            checkpoint = prepared["rng"][stream]
            # Runtime continuation restores an exact checkpoint, independently
            # of the state-dict/cache loading policy in set_extra_state.
            rng.restore_rng_checkpoint(checkpoint, policy="checkpoint")
        for entry in self.entries:
            owner = owners[entry.path]
            owner._buffers[entry.name] = prepared["samples"][entry.key]
            owner._random_parameter_draws[entry.name] = _clone(
                prepared["draws"][entry.key]
            )
            owner._random_parameter_parameters[entry.name] = _clone(
                prepared["parameters"][entry.key]
            )
            owner._random_parameter_initialized[entry.name] = True

    def commit(self, population, state):
        self.publish_commit(population, self.prepare_commit(population, state))


def materialize_random_parameters(
    owner, path, values, inputs, *, check_purity=False, materialized=None
):
    """Pure distribution evaluation; owner writes remain inside functional_call."""
    from dendra.models.random_parameters import evaluate_random_parameter

    outputs = {}
    for name, spec in owner.random_parameters.items():
        key = f"{path + '.' if path else ''}{name}"
        if materialized is not None and key in materialized:
            value, parameters = materialized[key]
            owner._buffers[name] = value
            outputs[key] = (value, parameters)
            continue
        draw, saved, sample, fresh, initialization_frame = inputs[key]
        current = {p: values[spec.full_parameter_name(p)] for p in spec.params}
        frozen = restore_population_cache_values(
            current, saved, parameter_names=current
        )
        selected = {p: torch.where(fresh, current[p], frozen[p]) for p in current}
        if initialization_frame is not None:
            selected = {
                p: torch.where(initialization_frame, saved[p], selected[p])
                for p in current
            }
        from ._population import _initialization_transform_scalar_input

        # An empty hidden scalar lane cannot broadcast with the sampler's
        # visible singleton draw on PyTorch's scalar batching path. Pad only
        # evaluation operands; retained distribution provenance keeps its
        # declared shape.
        evaluation_parameters = {
            name: _initialization_transform_scalar_input(value)
            for name, value in selected.items()
        }
        evaluation_draw = _pytree.tree_map(_initialization_transform_scalar_input, draw)
        padded_arguments = any(
            value.ndim != padded.ndim
            for value, padded in zip(
                (*selected.values(), *_pytree.tree_leaves(draw)),
                (
                    *evaluation_parameters.values(),
                    *_pytree.tree_leaves(evaluation_draw),
                ),
                strict=True,
            )
        )
        arguments = (*selected.values(), *_pytree.tree_leaves(draw))
        versions = (
            tuple(value._version for value in arguments)
            if check_purity
            and not torch.compiler.is_compiling()
            and not torch._C._are_functorch_transforms_active()
            and not any(torch.is_inference(value) for value in arguments)
            else None
        )
        evaluated = evaluate_random_parameter(
            spec, evaluation_parameters, evaluation_draw
        )
        if padded_arguments:
            while evaluated.ndim > sample.ndim and evaluated.shape[0] == 1:
                evaluated = evaluated.squeeze(0)
        if versions is not None and any(
            value._version != version
            for value, version in zip(arguments, versions, strict=True)
        ):
            raise FunctionalizationError(
                "RAND transform mutated its explicit parameters or base draws"
            )
        # Native sampling normalizes the transform's result to the declared
        # buffer even when a custom distribution returns another dtype/device.
        evaluated = evaluated.to(sample)
        exact = _restore_parameter(evaluated, sample)
        value = torch.where(fresh, evaluated, exact)
        if not spec.reparameterized:
            value = value.detach()
        owner._buffers[name] = value
        outputs[key] = (value, selected)
    return outputs


@torch.compiler.assume_constant_result
def validate_random_source(plan):
    plan.validate_source()
    return True
