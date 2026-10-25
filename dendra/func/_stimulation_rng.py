"""Shared host draw ordering for Poisson stimulation and model RNG streams."""

from __future__ import annotations

from collections import Counter
from types import MethodType

import torch
from torch.utils import _pytree

from dendra.models.random_parameters import sample_random_parameter_draw
from dendra.models.rng import RNGModule

from ._rng_diagnostics import _rng_diagnostic_scope
from ._rng_frames import _rng_values, _sampler_from_frame
from ._types import FunctionalizationError, RuntimeNoiseDraw, StimulationDraw


def _generator_bindings(bindings):
    result = {}
    for stream, source in bindings.items():
        result[id(source._cpu_gen)] = (stream, "cpu")
        for device, generator in source._device_gens.items():
            result[id(generator)] = (stream, str(device))
    return result


def _validate_registered_rng_owners(root, declared, requested):
    """Reject requested generators with undeclared registered RNG owners."""
    for owner in root.modules():
        if (
            isinstance(owner, RNGModule)
            and id(owner) not in declared
            and any(
                id(generator) in requested
                for generator in (owner._cpu_gen, *owner._device_gens.values())
            )
        ):
            raise FunctionalizationError(
                "Poisson generator aliases an untracked registered RNGModule; "
                "use a declared RAND/NOISE/SDE stream or a separate generator"
            )


def _validate_tracked_generators(population, bindings, generators):
    declared = {id(owner) for owner in bindings.values()}
    requested = {id(generator) for generator in generators}
    _validate_registered_rng_owners(population, declared, requested)


def _assert_shared_frame(actual, expected):
    """Reject conflicting explicit suffixes without reading traced scalars."""
    valid = (actual.to(expected) == expected).all()
    message = "Shared stimulation and model RNG packets have different end frames"
    from ._poisson_stimulation import _cold_validation

    if torch.compiler.is_compiling():
        torch._assert_async(valid, message)
    elif torch._C._are_functorch_transforms_active():
        functorch = torch._C._functorch
        while functorch.is_functorch_wrapped_tensor(valid):
            valid = functorch.get_unwrapped(valid)
        torch._assert_async(valid.all(), message)
    elif not _cold_validation():
        torch._assert_async(valid, message)
    elif not bool(valid):
        raise ValueError(message)


def _kernel_requests(functional, tensors):
    """Observe the native Poisson/SDE boundary order on disposable operands.

    State ``advance`` retains its native assigned-value-before-draw ordering.
    Only its Gaussian sampler is replaced by a schema-matched explicit value.
    The numerical waveform and State implementations are unchanged.
    """
    poisson = functional._poisson_plan
    noise = functional._noise_plan
    mechanism = () if poisson is None else poisson._entries_for_phase("mechanism")
    sde = () if noise is None else noise.sde.entries
    if not sde:
        return tuple(("poisson", entry) for entry in mechanism)
    if not mechanism:
        return tuple(("sde", entry) for entry in sde)
    from torch.func import functional_call

    from dendra.models.mechanisms._state import State

    from ._population import (
        _authored_workspace_classes,
        _class_state_snapshot,
        _deepcopy_execution_module,
        _global_rng_snapshot,
        _rebind_population_time_references,
        _restore_class_state,
        _restore_global_rng,
    )

    snapshot = (
        _global_rng_snapshot(functional.device) if functional._check_purity else None
    )
    classes = (
        _class_state_snapshot(_authored_workspace_classes(functional._transition))
        if functional._check_purity
        else None
    )
    observed = []
    try:
        with torch.inference_mode(False), torch.no_grad():
            parameters, constants, state = _pytree.tree_map_only(
                torch.Tensor,
                lambda value: value.detach().clone(),
                (tensors.parameters, tensors.constants, tensors.state),
            )
            sharing = getattr(functional, "_stimulation_rng_plan", None)
            if sharing is not None:
                state = sharing.expand_state(state)
            prepared = functional._prepare_values(parameters, constants)
            mapping, dt, celsius, voltage, time, remainder = functional._mapping(
                parameters, prepared, state, isolate_read_only=True
            )
            transition = _deepcopy_execution_module(functional._transition)
            _rebind_population_time_references(transition.population)
            # Registration occurrences already have distinct private owners.
            by_key = {entry.key: entry for entry in mechanism}
            for entry in by_key.values():
                owner = transition.get_submodule(entry.mapping.rsplit(".", 1)[0])
                original = owner.fn

                def call(self, *args, _entry=entry, _original=original, **kwargs):
                    del self
                    observed.append(("poisson", _entry))
                    return _original(*args, **kwargs)

                owner.fn = MethodType(call, owner)
            states = {}
            for entry in sde:
                states.setdefault(entry.path, {})[entry.name] = entry
            for path, entries in states.items():
                owner = transition.population.get_submodule(path)

                def normal(self, name, like, *, _entries=entries):
                    del self
                    observed.append(("sde", _entries[name]))
                    return torch.ones_like(like).detach()

                owner._derive_advance_values = MethodType(
                    State._derive_advance_values, owner
                )
                owner._sde_randn_like = MethodType(normal, owner)
            functional_call(
                transition,
                mapping,
                (
                    voltage,
                    time,
                    remainder,
                    dt,
                    celsius,
                    None,
                    None,
                    1,
                    None,
                    prepared.get("noise_workspaces"),
                ),
                tie_weights=False,
            )
    finally:
        try:
            if classes is not None:
                _restore_class_state(classes)
        finally:
            if snapshot is not None:
                _restore_global_rng(snapshot, functional.device)
    expected = Counter(("poisson", entry.key) for entry in mechanism)
    expected.update(("sde", entry.key) for entry in sde)
    if Counter((kind, entry.key) for kind, entry in observed) != expected:
        raise FunctionalizationError(
            "Shared stochastic execution needs fixed Poisson and State SDE invocation counts"
        )
    return tuple(observed)


class StimulationRNGPlan:
    """Expose borrowed model generators through one authoritative RNG carry."""

    def __init__(self, functional):
        self.functional = functional
        self.poisson = functional._poisson_plan
        self.random = functional._random_plan
        self.noise = functional._noise_plan
        bindings = {} if self.random is None else self.random._rng_bindings
        generators = _generator_bindings(bindings)
        self.borrowed = (
            {}
            if self.poisson is None
            else {
                stream: generators[id(generator)]
                for stream, generator in self.poisson.streams.items()
                if id(generator) in generators
            }
        )
        self.enabled = bool(self.borrowed)
        self._borrowed_generators = {
            stream: self.poisson.streams[stream] for stream in self.borrowed
        }
        if self.poisson is not None:
            self.poisson.externally_owned_streams = frozenset(self.borrowed)
            if not getattr(functional, "_network_stimulation_rng_scope", False):
                _validate_tracked_generators(
                    functional._population, bindings, self.poisson.streams.values()
                )
        self._configured = False
        self._requests = ()

    def configure(self, tensors):
        """Freeze the accepted-step request order once after lowering."""
        if self._configured:
            return
        bank = (
            ()
            if self.poisson is None
            else tuple(("poisson", entry) for entry in self.poisson._bank_entries)
        )
        noise = (
            ()
            if self.noise is None
            else tuple(("noise", entry) for entry in self.noise.noise_runtime)
        )
        self._requests = (*bank, *noise, *_kernel_requests(self.functional, tensors))
        self._configured = True

    def runtime_requests(self, phase=None):
        """Yield globally composable bank phases and one Population kernel."""
        if not self._configured:
            raise FunctionalizationError(
                "Shared stimulation request plan is not configured"
            )
        if phase is None:
            return self._requests
        return tuple(
            (kind, entry)
            for kind, entry in self._requests
            if (entry.kind if kind == "poisson" else "mechanism") == phase
        )

    def expand_state(self, state):
        if not self.enabled:
            return state
        return {
            **state,
            "stimulation": {
                **state["stimulation"],
                "rng": {
                    **state["stimulation"].get("rng", {}),
                    **{
                        stream: state["random"]["rng"][owner][device].to("cpu")
                        for stream, (owner, device) in self.borrowed.items()
                    },
                },
            },
        }

    def strip_state(self, state):
        if not self.enabled:
            return state
        stimulation = {
            key: value for key, value in state["stimulation"].items() if key != "rng"
        }
        rng = {
            stream: frame
            for stream, frame in state["stimulation"].get("rng", {}).items()
            if stream not in self.borrowed
        }
        if rng:
            stimulation["rng"] = rng
        return {**state, "stimulation": stimulation}

    def validate_source(self):
        """Borrowed handles retain the same registered physical generator."""
        self.validate_compatible_population(self.functional._population)
        if not self.enabled:
            return
        for stream, (owner, device) in self.borrowed.items():
            source = self.random._rng_bindings[owner]
            actual = (
                source._cpu_gen
                if device == "cpu"
                else source._device_gens.get(torch.device(device))
            )
            if actual is not self._borrowed_generators[stream]:
                raise FunctionalizationError(
                    "Shared stimulation RNG generator handles changed; lower again"
                )

    def validate_compatible_population(self, population):
        """Preserve model/stimulation aliases on extraction and native commit."""
        if self.poisson is None:
            return
        from ._random_parameters import _owners

        owners = dict(_owners(population))
        bindings = (
            {}
            if self.random is None
            else {
                stream: getattr(owners[path], attribute)
                for stream, (path, attribute) in self.random.streams.items()
            }
        )
        generators = _generator_bindings(bindings)
        sources = self.poisson._resolved_sources(population)
        borrowed = {
            entry.stream: generators[id(sources[entry.key].generator)]
            for entry in self.poisson.entries
            if id(sources[entry.key].generator) in generators
        }
        if borrowed != self.borrowed:
            raise FunctionalizationError(
                "Population stimulation and model RNG sharing differs from this plan; "
                "lower again"
            )
        if not self.functional._network_stimulation_rng_scope:
            _validate_tracked_generators(
                population, bindings, (source.generator for source in sources.values())
            )

    def validate_packets(self, noise, stimulation, *, initialization=False):
        """Paired explicit packets describe one common physical stream suffix."""
        if not self.enabled:
            return
        needs_noise = (
            self.random is not None
            if initialization
            else bool(self.noise is not None and self.noise.runtime)
        )
        needs_stimulation = bool(
            self.poisson is not None
            and (self.poisson.has_initialization if initialization else True)
        )
        if (
            needs_noise
            and needs_stimulation
            and ((noise is None) != (stimulation is None))
        ):
            raise ValueError(
                "Shared stimulation RNG requires both model and Poisson packets, "
                "or automatic sampling for both"
            )
        if noise is not None and stimulation is not None:
            for stream, (owner, device) in self.borrowed.items():
                _assert_shared_frame(stimulation.rng[stream], noise.rng[owner][device])

    def publish(self, state, noise=None, stimulation=None, *, tape=False):
        """Merge a paired suffix while retaining model seed/load metadata."""
        if not self.enabled:
            return state
        if tape:
            noise = (
                None
                if noise is None
                else _pytree.tree_map(lambda value: value[-1], noise)
            )
            stimulation = (
                None
                if stimulation is None
                else _pytree.tree_map(lambda value: value[-1], stimulation)
            )
        rng = dict(state["random"]["rng"] if noise is None else noise.rng)
        if stimulation is not None:
            for stream, (owner, device) in self.borrowed.items():
                frame = stimulation.rng[stream]
                if noise is not None:
                    _assert_shared_frame(frame, noise.rng[owner][device])
                rng[owner] = {
                    **rng[owner],
                    device: frame.to(rng[owner][device]).clone(),
                }
        return self.strip_state({**state, "random": {**state["random"], "rng": rng}})

    def begin_packets(self, state, *, samplers=None, poisson_samplers=None):
        """Create owned operands over caller-supplied or isolated host samplers."""
        state = self.expand_state(state)
        random_samplers = (
            {}
            if self.random is None
            else {
                stream: _sampler_from_frame(frame)
                for stream, frame in state["random"]["rng"].items()
            }
            if samplers is None
            else samplers
        )
        generators = (
            {}
            if self.poisson is None
            else self.poisson._samplers(state, poisson_samplers)
        )
        generators = dict(generators)
        for stream, (owner, device) in self.borrowed.items():
            generators[stream] = random_samplers[owner]._generator_for(device)
        return {
            "random_samplers": random_samplers,
            "poisson_samplers": generators,
            "noise_draws": {},
            "stimulation_draws": {}
            if self.poisson is None
            else self.poisson._placeholder_draws(),
            "positions": Counter(),
        }

    def sample_request(self, kind, entry, context):
        """Consume one existing sampler at its native request boundary."""
        if kind == "poisson":
            draw = self.poisson._sample_entry(
                entry, context["poisson_samplers"][entry.stream]
            )
            position = context["positions"][entry.key]
            self.poisson._assign_draw(
                context["stimulation_draws"], entry, position, draw
            )
            context["positions"][entry.key] += 1
            return
        with _rng_diagnostic_scope(
            self.functional.device, enabled=self.functional._check_purity
        ) as diagnostic:
            draw = sample_random_parameter_draw(
                entry.spec,
                context["random_samplers"][entry.stream],
                entry.shape,
                device=entry.device,
                dtype=entry.dtype,
            )
        if diagnostic.changes:
            raise FunctionalizationError(
                "Stochastic draw sampler consumed global RNG; use its explicit RNGModule"
            )
        leaves, specification = _pytree.tree_flatten(draw)
        if hasattr(entry, "draw_spec") and specification != entry.draw_spec:
            raise FunctionalizationError("NOISE draw tree changed; lower again")
        if len(leaves) != len(entry.draw_names):
            raise FunctionalizationError("Stochastic draw tree changed; lower again")
        context["noise_draws"].update(
            (name, leaf.detach().clone())
            for name, leaf in zip(entry.draw_names, leaves, strict=True)
        )

    def finish_packets(self, context):
        """Record the common current end frame after the complete step."""
        noise = (
            None
            if self.noise is None or not self.noise.runtime
            else RuntimeNoiseDraw(
                context["noise_draws"],
                {
                    stream: _rng_values(sampler, self.functional.device)
                    for stream, sampler in context["random_samplers"].items()
                },
            )
        )
        stimulation = (
            None
            if self.poisson is None
            else StimulationDraw(
                context["stimulation_draws"],
                {
                    stream: generator.get_state().clone()
                    for stream, generator in context["poisson_samplers"].items()
                },
            )
        )
        return noise, stimulation

    def sample_runtime(
        self, state, *, steps=None, samplers=None, poisson_samplers=None
    ):
        if steps is not None:
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise ValueError("steps must be a non-negative integer")
            context = self.begin_packets(
                state, samplers=samplers, poisson_samplers=poisson_samplers
            )
            packets = []
            for _ in range(steps):
                context["noise_draws"] = {}
                context["stimulation_draws"] = (
                    {} if self.poisson is None else self.poisson._placeholder_draws()
                )
                context["positions"] = Counter()
                for kind, entry in self.runtime_requests():
                    self.sample_request(kind, entry, context)
                packets.append(self.finish_packets(context))
            if packets:
                return _pytree.tree_map(
                    lambda *values: None if values[0] is None else torch.stack(values),
                    *packets,
                )
            noise, stimulation = self.finish_packets(context)
            if self.noise is not None and self.noise.runtime:
                noise = RuntimeNoiseDraw(
                    self.noise._empty_tape_draws(),
                    _pytree.tree_map(
                        lambda value: value.new_empty((0, *value.shape)), noise.rng
                    ),
                )
            if stimulation is not None:
                stimulation = StimulationDraw(
                    {
                        entry.key: torch.empty(
                            (0, *self.poisson._draw_shape(entry)),
                            dtype=self.poisson.draw_dtype,
                            device=self.poisson.streams[entry.stream].device,
                        )
                        for entry in self.poisson.entries
                    },
                    _pytree.tree_map(
                        lambda value: value.new_empty((0, *value.shape)),
                        stimulation.rng,
                    ),
                )
            return noise, stimulation
        context = self.begin_packets(
            state, samplers=samplers, poisson_samplers=poisson_samplers
        )
        for kind, entry in self.runtime_requests():
            self.sample_request(kind, entry, context)
        return self.finish_packets(context)

    def sample_initialization(self, state, *, samplers=None, poisson_samplers=None):
        """Sample RAND/initial NOISE before the mechanism's initial currents."""
        from ._types import RandomParameterDraw

        context = self.begin_packets(
            state, samplers=samplers, poisson_samplers=poisson_samplers
        )
        random = (
            None
            if self.random is None
            else self.random.sample(
                self.expand_state(state), samplers=context["random_samplers"]
            )
        )
        if self.poisson is not None:
            for entry in self.poisson._initialization_entries:
                self.sample_request("poisson", entry, context)
        _noise, stimulation = self.finish_packets(context)
        if self.poisson is None or not self.poisson.has_initialization:
            stimulation = None
        if random is not None:
            random = RandomParameterDraw(
                random.draws,
                {
                    stream: _rng_values(sampler, self.functional.device)
                    for stream, sampler in context["random_samplers"].items()
                },
            )
        return random, stimulation


__all__ = []
