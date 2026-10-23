"""Explicit runtime NOISE/SDE draws with canonical shared RNG ownership."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from torch.utils import _pytree

from dendra.helpers import current_check_purity
from dendra.models.random_parameters import get_random_distribution
from dendra.models.rng import RNGModule

from ._types import FunctionalizationError, RuntimeNoiseDraw


def noise_declarations(population):
    from ._population import _parameterized_owner_paths

    return tuple(
        (path, name, spec)
        for path, owner in _parameterized_owner_paths(population)
        for name, spec in owner.runtime_noises.items()
    )


def stream_inventory(population):
    from ._population import _parameterized_owner_paths

    streams, identities = {}, {}
    for path, owner in _parameterized_owner_paths(population):
        for spec in (*owner.random_parameters.values(), *owner.runtime_noises.values()):
            rng = getattr(owner, spec.effective_rng_name)
            key = identities.setdefault(id(rng), f"stream_{len(identities)}")
            streams.setdefault(key, (path, spec.effective_rng_name))
        for name in getattr(owner, "_sde_rng_names", ()):
            rng = getattr(owner, name)
            key = identities.setdefault(id(rng), f"stream_{len(identities)}")
            streams.setdefault(key, (path, name))
    return streams, identities


def _stream_topology(population):
    """Describe declaration aliases independently of live RNG object IDs."""
    from ._population import _parameterized_owner_paths

    _streams, identities = stream_inventory(population)
    declared = tuple(
        (path, kind, name, identities[id(getattr(owner, spec.effective_rng_name))])
        for path, owner in _parameterized_owner_paths(population)
        for kind, declarations in (
            ("random", owner.random_parameters),
            ("noise", owner.runtime_noises),
        )
        for name, spec in declarations.items()
    )
    return declared + tuple(
        (path, "sde", name, identities[id(getattr(owner, name))])
        for path, owner in _parameterized_owner_paths(population)
        for name in getattr(owner, "_sde_rng_names", ())
    )


@dataclass(frozen=True)
class _NoiseEntry:
    key: str
    path: str
    name: str
    spec: object
    stream: str
    draw_spec: object
    draw_names: tuple
    leaves: tuple
    parameter_shapes: tuple
    visible_rank: int
    shape: tuple
    dtype: torch.dtype
    device: torch.device


def _buffer_value(value, shape):
    # Visible singleton padding carries hidden transform lanes while Native
    # copy_ accepts ordinary broadcastable distribution results.
    while value.ndim > len(shape) and value.shape[0] == 1:
        value = value.squeeze(0)
    return torch.broadcast_to(value, shape)


def _validate_noise_adapters(population, owner):
    from dendra.models.integrators.core import Integrator
    from dendra.models.mechanisms import Mechanism
    from dendra.models.mechanisms._handler import MechanismHandler
    from dendra.models.parametric import Parameterized

    canonical = Mechanism if isinstance(owner, Mechanism) else Parameterized
    for target, standard, method in (
        (owner, canonical, "sample_runtime_noises_"),
        (population.mech, MechanismHandler, "sample_runtime_noises_"),
        (population.integrator, Integrator, "_sample_runtime_noises"),
    ):
        if method in target.__dict__ or inspect.getattr_static(
            type(target), method
        ) is not inspect.getattr_static(standard, method):
            raise FunctionalizationError(
                f"NOISE requires canonical {method}; custom sampling adapters are currently unsupported"
            )


def _validate_rng_adapter(rng):
    if type(rng) is not RNGModule or any(
        method in rng.__dict__
        or inspect.getattr_static(type(rng), method)
        is not inspect.getattr_static(RNGModule, method)
        for method in (
            "rand",
            "randn",
            "_generator_for",
            "_rng",
            "init",
            "get_extra_state",
            "set_extra_state",
            "restore_rng_checkpoint",
            "reseed",
            "set_rng_state",
            "reseed_for_generation",
        )
    ):
        raise FunctionalizationError(
            "Runtime NOISE/SDE requires canonical RNGModule streams"
        )


class RuntimeNoisePlan:
    """Static draw schemas; numerical values and stream boundaries are operands."""

    def __init__(self, population):
        from dendra.models.random_parameters import sample_runtime_noise_draw

        from ._population import (
            _global_rng_changes,
            _global_rng_snapshot,
            _initialization_callable_dependency_signature,
            _module_at,
            _restore_global_rng,
        )

        self.population = population
        self.check_purity = current_check_purity()
        self.declarations = noise_declarations(population)
        self.streams, identities = stream_inventory(population)
        self._stream_topology = _stream_topology(population)
        entries, contracts = [], []
        self._rng_bindings = []
        for path, name, spec in self.declarations:
            owner = _module_at(population, path) if path else population
            _validate_noise_adapters(population, owner)
            rng = getattr(owner, spec.effective_rng_name)
            _validate_rng_adapter(rng)
            self._rng_bindings.append(rng)
            distribution = get_random_distribution(spec.distribution)
            if distribution.draw_sampler is None or distribution.transform is None:
                raise FunctionalizationError(
                    f"NOISE distribution {spec.distribution!r} requires a pure draw_sampler and transform contract"
                )
            signatures = (
                tuple(
                    _initialization_callable_dependency_signature(
                        fn,
                        label=f"NOISE {spec.distribution}.{label}",
                        owner=owner,
                        strict=True,
                    )
                    for label, fn in (
                        ("draw_sampler", distribution.draw_sampler),
                        ("transform", distribution.transform),
                    )
                )
                if self.check_purity
                else ()
            )
            value = owner._buffers[name]
            snapshot = (
                _global_rng_snapshot(population.device()) if self.check_purity else None
            )
            try:
                draw = sample_runtime_noise_draw(
                    spec,
                    RNGModule.from_checkpoint(rng.get_extra_state()),
                    tuple(value.shape),
                    device=value.device,
                    dtype=value.dtype,
                )
            finally:
                changes = (
                    _global_rng_changes(snapshot, population.device())
                    if self.check_purity
                    else ()
                )
                if self.check_purity:
                    _restore_global_rng(snapshot, population.device())
            if changes:
                raise FunctionalizationError(
                    "NOISE draw sampler consumed global RNG; use its explicit RNGModule"
                )
            leaves, tree_spec = _pytree.tree_flatten(draw)
            if not leaves or not all(torch.is_tensor(leaf) for leaf in leaves):
                raise FunctionalizationError(
                    "NOISE draws must be nonempty Tensor trees"
                )
            from dendra.models.random_parameters import evaluate_runtime_noise

            from ._population import _audit_tensor_binding

            audit_parameters = {
                parameter: getattr(owner, spec.full_parameter_name(parameter))
                .detach()
                .clone()
                for parameter in spec.params
            }
            audit_draw = _pytree.tree_map(lambda leaf: leaf.detach().clone(), draw)
            audit_dt = value.new_tensor(1.0)
            audited = (
                *audit_parameters.values(),
                *_pytree.tree_leaves(audit_draw),
                audit_dt,
            )
            versions = (
                tuple(_audit_tensor_binding(leaf) for leaf in audited)
                if self.check_purity
                else None
            )
            snapshot = (
                _global_rng_snapshot(population.device()) if self.check_purity else None
            )
            try:
                probe = evaluate_runtime_noise(
                    spec, audit_parameters, audit_draw, dt=audit_dt
                )
            finally:
                changes = (
                    _global_rng_changes(snapshot, population.device())
                    if self.check_purity
                    else ()
                )
                if self.check_purity:
                    _restore_global_rng(snapshot, population.device())
            if changes or (
                versions is not None
                and versions != tuple(_audit_tensor_binding(leaf) for leaf in audited)
            ):
                raise FunctionalizationError(
                    "NOISE transform must preserve its explicit parameters, draws, dt, and global RNG state"
                )
            if not torch.is_tensor(probe):
                raise FunctionalizationError("NOISE transform must return a Tensor")
            try:
                _buffer_value(probe, tuple(value.shape))
            except RuntimeError as error:
                raise FunctionalizationError(
                    "NOISE transform result cannot broadcast to its declared buffer"
                ) from error
            key = f"{path + '.' if path else ''}{name}"
            entries.append(
                _NoiseEntry(
                    key,
                    path,
                    name,
                    spec,
                    identities[id(rng)],
                    tree_spec,
                    tuple(f"noise.draws.{key}.{index}" for index in range(len(leaves))),
                    tuple(leaf.detach().clone() for leaf in leaves),
                    tuple(
                        tuple(getattr(owner, spec.full_parameter_name(parameter)).shape)
                        for parameter in spec.params
                    ),
                    len(owner.shape_f),
                    tuple(value.shape),
                    value.dtype,
                    value.device,
                )
            )
            contracts.append(
                (distribution.draw_sampler, distribution.transform, signatures)
            )
        self.entries, self.contracts = tuple(entries), tuple(contracts)
        self._rng_bindings = tuple(self._rng_bindings)
        from ._state_sde import StateSDEPlan

        self.noise_runtime = tuple(entry for entry in self.entries if entry.path)
        self.sde = StateSDEPlan(population, identities)
        self.runtime = (*self.noise_runtime, *self.sde.entries)

    def install(self, population):
        self.sde.install(population)

    def sde_values(self, packet):
        return self.sde.values(packet)

    def initialization_failures(self):
        """NOISE-backed workspaces use explicit initialization/configuration frames."""
        return ()

    def validate_source(self):
        from ._population import (
            _initialization_callable_dependency_signature,
            _module_at,
        )

        self.sde.validate_population(self.population, source=True)
        if noise_declarations(self.population) != self.declarations:
            raise FunctionalizationError(
                "NOISE declarations changed; lower the Population again"
            )
        if _stream_topology(self.population) != self._stream_topology:
            raise FunctionalizationError(
                "Population RNG stream sharing changed; lower the Population again"
            )
        for entry, (sampler, transform, signatures), captured_rng in zip(
            self.entries, self.contracts, self._rng_bindings, strict=True
        ):
            distribution = get_random_distribution(entry.spec.distribution)
            owner = (
                _module_at(self.population, entry.path)
                if entry.path
                else self.population
            )
            _validate_noise_adapters(self.population, owner)
            rng = getattr(owner, entry.spec.effective_rng_name)
            _validate_rng_adapter(rng)
            if rng is not captured_rng:
                raise FunctionalizationError(
                    "NOISE RNG stream binding changed; lower the Population again"
                )
            current = (
                tuple(
                    _initialization_callable_dependency_signature(
                        fn,
                        label=f"NOISE {entry.spec.distribution}.{label}",
                        owner=owner,
                        strict=True,
                    )
                    for label, fn in (
                        ("draw_sampler", distribution.draw_sampler),
                        ("transform", distribution.transform),
                    )
                )
                if self.check_purity
                else ()
            )
            if (
                distribution.draw_sampler is not sampler
                or distribution.transform is not transform
                or current != signatures
            ):
                raise FunctionalizationError(
                    "NOISE distribution contract changed; lower again"
                )

    def validate_compatible_population(self, population):
        """Require target checkpoints to preserve declaration stream aliases."""
        self.sde.validate_population(population)
        if noise_declarations(population) != self.declarations:
            raise FunctionalizationError(
                "Population NOISE declarations do not match this functional plan"
            )
        if _stream_topology(population) != self._stream_topology:
            raise FunctionalizationError(
                "Population RNG stream sharing does not match this functional plan"
            )
        from ._population import _module_at

        for entry in self.entries:
            owner = _module_at(population, entry.path) if entry.path else population
            _validate_noise_adapters(population, owner)
            _validate_rng_adapter(getattr(owner, entry.spec.effective_rng_name))

    def validate(self, packet, *, steps=None, initialization=False):
        if not isinstance(packet, RuntimeNoiseDraw) and not initialization:
            raise ValueError(
                "Runtime NOISE/SDE requires an explicit RuntimeNoiseDraw packet"
            )
        entries = self.entries if initialization else self.runtime
        expected = {name for entry in entries for name in entry.draw_names}
        selected = {
            name: value
            for name, value in packet.draws.items()
            if name.startswith("noise.draws.")
            or not initialization
            and name.startswith("sde.draws.")
        }
        if set(selected) != expected or (
            not initialization and set(packet.draws) != expected
        ):
            raise ValueError(
                "Runtime NOISE/SDE draw names do not match the lowered plan"
            )
        for entry in entries:
            for name, reference in zip(entry.draw_names, entry.leaves, strict=True):
                value = selected[name]
                shape = (
                    tuple(reference.shape)
                    if steps is None
                    else (steps, *reference.shape)
                )
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != shape
                    or value.dtype != reference.dtype
                    or value.device != reference.device
                ):
                    raise ValueError(
                        f"Runtime NOISE/SDE draw {name!r} has incompatible Tensor schema"
                    )

    def values(self, mapping, packet, dt, *, initialization=False, owner_paths=None):
        from dendra.models.random_parameters import evaluate_runtime_noise

        from ._population import _initialization_transform_scalar_input

        values = {}
        for entry in self.entries if initialization else self.noise_runtime:
            if owner_paths is not None and entry.path not in owner_paths:
                continue
            prefix = f"population.{entry.path + '.' if entry.path else ''}"
            parameters = {
                name: mapping[f"{prefix}{entry.spec.full_parameter_name(name)}"]
                .reshape(shape)
                .detach()
                .clone()
                for name, shape in zip(
                    entry.spec.params, entry.parameter_shapes, strict=True
                )
            }
            draw = _pytree.tree_unflatten(
                [
                    packet.draws[name]
                    .reshape(
                        (1,) * max(0, entry.visible_rank - packet.draws[name].ndim)
                        + tuple(packet.draws[name].shape)
                    )
                    .detach()
                    .clone()
                    for name in entry.draw_names
                ],
                entry.draw_spec,
            )
            versions = (
                tuple(
                    value._version
                    for value in (*parameters.values(), *_pytree.tree_leaves(draw))
                )
                if self.check_purity
                and not torch.compiler.is_compiling()
                and not torch._C._are_functorch_transforms_active()
                and not any(
                    torch.is_inference(value)
                    for value in (*parameters.values(), *_pytree.tree_leaves(draw))
                )
                else None
            )
            # A visible singleton avoids PyTorch's scalar broadcasting rule
            # selecting lane zero from an empty hidden vmap batch. Only the
            # evaluation views change; declared operands and output schemas
            # keep their original shapes.
            value = evaluate_runtime_noise(
                entry.spec,
                {
                    name: _initialization_transform_scalar_input(value)
                    for name, value in parameters.items()
                },
                _pytree.tree_map(_initialization_transform_scalar_input, draw),
                dt=_initialization_transform_scalar_input(dt.detach().clone()),
            )
            if versions is not None and versions != tuple(
                value._version
                for value in (*parameters.values(), *_pytree.tree_leaves(draw))
            ):
                raise FunctionalizationError(
                    "NOISE transform modified its explicit inputs"
                )
            values[entry.key] = _buffer_value(
                value.to(device=entry.device, dtype=entry.dtype), entry.shape
            )
        return values

    def sample(self, state, *, steps=None, samplers=None):
        from dendra.models.random_parameters import (
            RuntimeNoiseDrawRequest,
            sample_runtime_noise_draws,
        )

        from ._population import (
            _global_rng_changes,
            _global_rng_snapshot,
            _restore_global_rng,
        )
        from ._random_parameters import _rng_extra, _rng_values

        if steps is not None:
            return self._sample_tape_boundaries(state, steps, samplers=samplers)
        checkpoints = (
            {name: _rng_extra(value) for name, value in state["random"]["rng"].items()}
            if samplers is None
            else {}
        )
        draws = {name: [] for entry in self.runtime for name in entry.draw_names}
        requests = tuple(
            RuntimeNoiseDrawRequest(
                entry.stream, entry.spec, entry.shape, entry.device, entry.dtype
            )
            for entry in self.runtime
        )
        snapshot = _global_rng_snapshot(self.population.device())
        try:
            sampled, checkpoints = sample_runtime_noise_draws(
                requests, checkpoints, samplers=samplers
            )
            for entry, draw in zip(self.runtime, sampled, strict=True):
                leaves, spec = _pytree.tree_flatten(draw)
                if hasattr(entry, "draw_spec") and spec != entry.draw_spec:
                    raise FunctionalizationError("NOISE draw tree changed; lower again")
                for name, leaf in zip(entry.draw_names, leaves, strict=True):
                    draws[name].append(leaf.detach().clone())
        finally:
            changes = _global_rng_changes(snapshot, self.population.device())
            _restore_global_rng(snapshot, self.population.device())
        if changes:
            raise FunctionalizationError(
                "NOISE draw sampler consumed global RNG; use explicit RNGModule"
            )
        result = {}
        for entry in self.runtime:
            for name, reference in zip(entry.draw_names, entry.leaves, strict=True):
                result[name] = draws[name][0]
        rng = {
            name: _rng_values(extra, self.population.device())
            for name, extra in checkpoints.items()
        }
        return RuntimeNoiseDraw(result, rng)

    def _sample_tape_boundaries(self, state, steps, *, samplers=None):
        draws, rngs = [], []
        carry = state
        for _index in range(steps):
            packet = self.sample(carry, samplers=samplers)
            draws.append(packet.draws)
            rngs.append(packet.rng)
            carry = {**carry, "random": {**carry["random"], "rng": packet.rng}}
        if steps:
            return RuntimeNoiseDraw(
                _pytree.tree_map(lambda *values: torch.stack(values), *draws),
                _pytree.tree_map(lambda *values: torch.stack(values), *rngs),
            )
        from ._random_parameters import _rng_values

        initial_rng = (
            state["random"]["rng"]
            if samplers is None
            else {
                name: _rng_values(sampler, self.population.device())
                for name, sampler in samplers.items()
            }
        )
        return RuntimeNoiseDraw(
            {
                name: leaf.new_empty((0, *leaf.shape))
                for entry in self.runtime
                for name, leaf in zip(entry.draw_names, entry.leaves, strict=True)
            },
            _pytree.tree_map(
                lambda value: value.new_empty((0, *value.shape)), initial_rng
            ),
        )
