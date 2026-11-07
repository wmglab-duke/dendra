"""Native RAND/NOISE initialization bindings using the functional draw schema."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch.utils import _pytree

from dendra.models.random_parameters import (
    evaluate_random_parameter,
    evaluate_runtime_noise,
    prepare_rng_for_random_parameter,
    sample_random_parameter_draw,
    sample_runtime_noise_draw,
)
from dendra.models.rng import RNGModule

from ._population import _parameterized_owner_paths
from ._random_parameters import RandomParameterPlan, random_declarations
from ._rng_diagnostics import _rng_diagnostic_scope
from ._rng_frames import _rng_extra, _rng_values
from ._runtime_noise import RuntimeNoisePlan
from ._types import FunctionalizationError, RandomParameterDraw


def _clone(value, *, detach=False):
    return _pytree.tree_map_only(
        torch.Tensor,
        lambda leaf: (leaf.detach() if detach else leaf).clone(),
        value,
    )


class NativeRandomInitialization:
    """Capture or replay the existing native owner sampling boundaries."""

    def __deepcopy__(self, memo):
        # Pure workspace builders copy their numerical owner while this scoped
        # host observer still owns the enclosing initialization boundary.
        return self

    def __init__(self, population, packet=None, *, capture=False):
        if capture and packet is not None:
            raise TypeError("Initialization capture does not accept a random packet")
        self.population = population
        self.capture = capture
        self._supplied = packet
        self._skipped = False
        self.owners = dict(_parameterized_owner_paths(population))
        self.noise_plan = RuntimeNoisePlan(population, _metadata_only=True)
        self.check_purity = self.noise_plan.check_purity
        from ._population import _authored_workspace_classes

        self._diagnostic_classes = (
            _authored_workspace_classes(population) if self.check_purity else ()
        )
        self.random_plan = (
            RandomParameterPlan(population)
            if random_declarations(population)
            or self.noise_plan.entries
            or self.noise_plan.sde.entries
            else None
        )
        self.streams = self.noise_plan.streams
        self.sources = {
            stream: getattr(self.owners[path], name)
            for stream, (path, name) in self.streams.items()
        }
        self.reference = {
            stream: _rng_values(source, population.device())
            for stream, source in self.sources.items()
        }
        self._random = {}
        self._noise = {}
        self._draws = {}
        self.updates = ()
        if self.random_plan is not None:
            self.random_plan.noise = self.noise_plan
            constants, _state = self.random_plan.extract()
            for entry in self.random_plan.entries:
                self._random[id(self.owners[entry.path]), entry.name] = entry
                self._draws.update(
                    (name, constants[name].detach().clone())
                    for name in entry.draw_names
                )
        for entry in self.noise_plan.entries:
            self._noise[id(self.owners[entry.path]), entry.name] = entry
            self._draws.update(
                (name, leaf.detach().clone())
                for name, leaf in zip(entry.draw_names, entry.leaves, strict=True)
            )
        if packet is not None:
            self._validate_packet(packet)

    def _validate_packet(self, packet):
        if self.random_plan is None:
            raise ValueError("This Population has no initialization random draws")
        if not isinstance(packet, RandomParameterDraw):
            raise TypeError("Initialization random draws must be a RandomParameterDraw")
        if not isinstance(packet.draws, Mapping):
            raise ValueError("Initialization random draws must be a mapping")
        # The ordinary pure initializer owns the draw-name and leaf schemas.
        self.random_plan.initialize_constants(self.random_plan.defaults, packet)
        self.noise_plan.validate(packet, initialization=True)
        if not isinstance(packet.rng, Mapping) or set(packet.rng) != set(
            self.reference
        ):
            raise ValueError(
                "Initialization random RNG streams do not match the Population"
            )
        updates = []
        for stream, reference in self.reference.items():
            frame = packet.rng[stream]
            if not isinstance(frame, Mapping) or set(frame) != set(reference):
                raise ValueError(
                    "Initialization random RNG fields do not match the Population"
                )
            for name, tensor in frame.items():
                expected = reference[name]
                if (
                    not torch.is_tensor(tensor)
                    or tensor.shape != expected.shape
                    or tensor.dtype != expected.dtype
                    or tensor.device != expected.device
                ):
                    raise ValueError("Initialization random RNG Tensor schema changed")
            payload = _rng_extra(frame)
            RNGModule.from_checkpoint(payload)
            updates.append((self.sources[stream], payload))
        self.updates = tuple(updates)

    def validate_source(self):
        self.noise_plan.validate_source()
        if self.random_plan is not None:
            self.random_plan.validate_source()
        current = dict(_parameterized_owner_paths(self.population))
        if any(current.get(path) is not owner for path, owner in self.owners.items()):
            raise FunctionalizationError("Native initialization random owners changed")
        if any(
            getattr(current[path], name) is not self.sources[stream]
            for stream, (path, name) in self.streams.items()
        ):
            raise FunctionalizationError("Native initialization random streams changed")

    def slots(self):
        seen = set()
        for owner in self.owners.values():
            if id(owner) not in seen:
                yield owner, "_native_initialization_draws", self
                seen.add(id(owner))

    def _entry(self, entries, owner, name):
        try:
            return entries[id(owner), name]
        except KeyError as error:
            raise FunctionalizationError(
                "Native initialization random declarations changed"
            ) from error

    def _read(self, entry, *, detach=False):
        if self._supplied is None:
            raise ValueError("Initialization requires explicit RAND/NOISE draws")
        return _pytree.tree_unflatten(
            [
                _clone(self._supplied.draws[name], detach=detach)
                for name in entry.draw_names
            ],
            entry.draw_spec,
        )

    def _record(self, entry, draw):
        leaves, specification = _pytree.tree_flatten(draw)
        if specification != entry.draw_spec or len(leaves) != len(entry.draw_names):
            raise FunctionalizationError(
                "Initialization draw sampler changed its Tensor tree"
            )
        for name, value in zip(entry.draw_names, leaves, strict=True):
            reference = self._draws[name]
            if (
                not torch.is_tensor(value)
                or value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
            ):
                raise FunctionalizationError(
                    "Initialization draw sampler changed its Tensor schema"
                )
            self._draws[name] = value.detach().clone()

    @contextmanager
    def _callback_inputs(self, parameters, draw=None, dt=None):
        if not self.check_purity:
            yield parameters, draw
            return
        from ._population import _audit_tensor_binding, _class_state_diagnostic_scope

        # Diagnose authored writes on owned operands, while preserving their
        # gradient paths and the native source's unchanged inputs.
        parameters, draw = _clone(parameters), _clone(draw)
        tensors = tuple(
            value
            for value in _pytree.tree_leaves((parameters, draw, dt))
            if torch.is_tensor(value)
        )
        before = tuple(_audit_tensor_binding(value) for value in tensors)
        with (
            _rng_diagnostic_scope(self.population.device(), enabled=True) as diagnostic,
            _class_state_diagnostic_scope(self._diagnostic_classes) as changed_classes,
        ):
            yield parameters, draw
            changed_inputs = before != tuple(
                _audit_tensor_binding(value) for value in tensors
            )
        if changed_inputs:
            raise FunctionalizationError(
                "Initialization random callback must preserve its explicit inputs"
            )
        if changed_classes:
            raise FunctionalizationError(
                "Initialization random callback mutated shared class state"
            )
        if diagnostic.changes:
            raise FunctionalizationError(
                "Initialization random callback consumed global RNG; use its explicit RNGModule"
            )

    def random(self, owner, name, spec, parameters, rng, buffer, generation):
        entry = self._entry(self._random, owner, name)
        if self.capture:
            rng.init(buffer.device)
            prepare_rng_for_random_parameter(spec, rng, generation)
            with self._callback_inputs({}):
                draw = sample_random_parameter_draw(
                    spec,
                    rng,
                    tuple(buffer.shape),
                    device=buffer.device,
                    dtype=buffer.dtype,
                )
        else:
            draw = self._read(entry)
        with self._callback_inputs(parameters, draw) as (parameters, checked_draw):
            sample = evaluate_random_parameter(spec, parameters, checked_draw)
        if self.capture:
            self._record(entry, draw)
        return sample, draw

    def noise(self, owner, name, spec, parameters, rng, buffer, dt):
        entry = self._entry(self._noise, owner, name)
        if self.capture:
            rng.init(buffer.device)
            with self._callback_inputs({}):
                draw = sample_runtime_noise_draw(
                    spec,
                    rng,
                    tuple(buffer.shape),
                    device=buffer.device,
                    dtype=buffer.dtype,
                )
        else:
            draw = self._read(entry, detach=True)
        with self._callback_inputs(parameters, draw, dt) as (parameters, checked_draw):
            sample = evaluate_runtime_noise(spec, parameters, checked_draw, dt=dt)
        if self.capture:
            self._record(entry, draw)
        return sample

    def skip(self):
        """A cache restore uses its own random-load policy and consumes no draws."""
        self._skipped = True

    def packet(self):
        if self._skipped or self.random_plan is None:
            return None
        if not self.capture:
            return self._supplied
        draws = {
            name: self._draws[name]
            for entry in self.random_plan._initialization_entries()
            for name in entry.draw_names
        }
        for entry in self.random_plan.entries:
            if not entry.spec.resample_on_initialize:
                draws.update((name, self._draws[name]) for name in entry.draw_names)
        return RandomParameterDraw(
            _clone(draws, detach=True),
            {
                stream: _rng_values(source, self.population.device())
                for stream, source in self.sources.items()
            },
        )

    def publish(self):
        if not self.capture and not self._skipped:
            for source, payload in self.updates:
                source.restore_rng_checkpoint(payload, policy="checkpoint")
