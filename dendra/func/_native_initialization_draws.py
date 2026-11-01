"""Scoped native adapters for the canonical ordered initialization draw packet."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch

from ._continuation_checkpoints import _cold_boundary
from ._initialization_draws import InitializationDraws, _normalize_initialization_draws
from ._native_runtime_draws import _bound_replay_slots
from ._types import FunctionalizationError


def _populations(model):
    from dendra.models import Network

    from ._population import (
        _population_subclass_failures,
        _subclass_contract_failures,
        _subclass_slot_contract_failures,
    )

    if isinstance(model, Network):
        from ._network import _NETWORK_SUBCLASS_METHODS, _NETWORK_SUBCLASS_SLOTS

        failures = _subclass_contract_failures(
            model, Network, _NETWORK_SUBCLASS_METHODS
        )
        failures.extend(
            _subclass_slot_contract_failures(model, Network, _NETWORK_SUBCLASS_SLOTS)
        )
        if failures:
            raise FunctionalizationError("; ".join(failures))
        if not model.built or model.dt is None:
            raise FunctionalizationError(
                "Initialize the Network before capturing or replaying initialization draws"
            )
        populations = dict(model.populations)
    else:
        populations = {"population": model}
    failures = [
        failure
        for pop in populations.values()
        for failure in _population_subclass_failures(pop)
    ]
    if failures:
        raise FunctionalizationError("; ".join(failures))
    if not populations or any(
        not pop.initialized or pop.integrator is None for pop in populations.values()
    ):
        raise FunctionalizationError(
            "Initialize every Population before capturing or replaying initialization draws"
        )
    devices = {pop.device() for pop in populations.values()}
    if len(devices) != 1 or (
        isinstance(model, Network)
        and model.netstim is not None
        and model.netstim.device() not in devices
    ):
        raise FunctionalizationError(
            "Initialization draws currently require a single simulation device"
        )
    return populations


def _cached(population, *, populate=True):
    return "_steady_state" in population._caches and not (
        populate and population._has_random_parameters_resampled_on_initialize()
    )


class _PopulationBoundary:
    def __deepcopy__(self, memo):
        return self

    def __init__(
        self, population, random, poisson, *, capture, cached, populate, session
    ):
        self.population, self.random, self.poisson = population, random, poisson
        self.capture, self.cached, self.populate = capture, cached, populate
        self.completed = False
        self.result = (None, None)
        self.session = session

    def begin(self):
        self.session.active = self

    def finish(self, *, cached):
        if self.completed:
            raise ValueError("Population initialization boundary was completed twice")
        if cached or not self.populate:
            self.random.skip()
        if cached and self.poisson is not None:
            self.poisson.skip()
        if self.capture:
            self.result = (
                self.random.packet(),
                None if self.poisson is None else self.poisson.packet(),
            )
        else:
            if self.poisson is not None:
                self.poisson.finish()
            self.random.publish()
            if self.poisson is not None:
                from ._stimulation_rng import _generator_bindings

                self.poisson.publish(
                    model_generators=_generator_bindings(self.random.sources)
                )
        self.completed = True


class _NativeInitialization:
    def __deepcopy__(self, memo):
        return self

    def __init__(self, model, packet, *, capture, options):
        from dendra.models import Network

        from ._native_initialization_poisson import NativePoissonInitialization
        from ._native_initialization_random import NativeRandomInitialization
        from ._population import (
            _functional_initialization_failures,
            _owned_cache_initialization_failures,
        )

        self.model, self.capture = model, capture
        self.started = False
        self.network = isinstance(model, Network)
        self.populations = _populations(model)
        if options.get("force_rebuild", False):
            raise FunctionalizationError(
                "Explicit initialization draws do not support force_rebuild=True"
            )
        if self.network and (
            options.get("dt", model.dt) != model.dt
            or model._device_signature() != model._device_sig
        ):
            raise FunctionalizationError(
                "Explicit initialization draws cannot rebuild connections; retain the initialized Network timestep and devices"
            )
        network_cache = self.network and bool(model._state_cache)
        if not capture:
            packet = _normalize_initialization_draws(packet)
            if self.network:
                families = []
                for name, family in (
                    ("random", packet.random),
                    ("stimulation", packet.stimulation),
                ):
                    if family is None:
                        family = dict.fromkeys(self.populations)
                    if not isinstance(family, Mapping) or set(family) != set(
                        self.populations
                    ):
                        raise ValueError(
                            f"initialization {name} draws must match the Network populations"
                        )
                    families.append(family)
                random_packets, poisson_packets = families
            else:
                if packet.netstim is not None or packet.connections is not None:
                    raise ValueError(
                        "Population initialization does not accept NetStim or connection draws"
                    )
                random_packets = {"population": packet.random}
                poisson_packets = {"population": packet.stimulation}
        else:
            random_packets = poisson_packets = dict.fromkeys(self.populations)
        self.boundaries, self.slots = {}, []
        self.active = None
        poisson_nodes = set()
        for name, population in self.populations.items():
            populate = options.get("populate_parameter_buffers", True)
            cached = network_cache or _cached(population, populate=populate)
            failures = (
                _owned_cache_initialization_failures(population)
                if cached
                else _functional_initialization_failures(population)
            )
            if failures:
                raise FunctionalizationError("; ".join(failures))
            random = NativeRandomInitialization(
                population, random_packets[name], capture=capture
            )
            poisson = NativePoissonInitialization(
                population, poisson_packets[name], capture=capture, cached=cached
            )
            if not capture:
                if cached or not populate:
                    if random_packets[name] is not None:
                        raise ValueError(
                            "This initialization does not consume RAND/NOISE draws"
                        )
                elif random.random_plan is not None and random_packets[name] is None:
                    raise ValueError(
                        "Initialization requires explicit RAND/NOISE draws"
                    )
                if (
                    random_packets[name] is not None
                    and poisson_packets[name] is not None
                ):
                    from ._stimulation_rng import (
                        _assert_shared_frame,
                        _generator_bindings,
                    )

                    bindings = _generator_bindings(random.sources)
                    for stream, generator in poisson.plan.streams.items():
                        if id(generator) in bindings:
                            owner, device = bindings[id(generator)]
                            _assert_shared_frame(
                                poisson_packets[name].rng[stream],
                                random_packets[name].rng[owner][device],
                            )
            boundary = _PopulationBoundary(
                population,
                random,
                poisson,
                capture=capture,
                cached=cached,
                populate=populate,
                session=self,
            )
            self.boundaries[name] = boundary
            self.slots.extend(random.slots())
            for owner, slot, _controller in poisson.slots():
                if id(owner) not in poisson_nodes:
                    self.slots.append((owner, slot, self))
                    self.slots.append(
                        (owner, "_native_initialization_origin", id(owner))
                    )
                    poisson_nodes.add(id(owner))
            self.slots.append((population, "_native_initialization_boundary", boundary))
        from ._poisson_stimulation import _validate_generator_domains
        from ._stimulation_rng import _validate_tracked_generators

        generators = tuple(
            generator
            for boundary in self.boundaries.values()
            if boundary.poisson.plan is not None
            for generator in boundary.poisson.plan.streams.values()
        )
        sources = {
            (name, stream): source
            for name, boundary in self.boundaries.items()
            for stream, source in boundary.random.sources.items()
        }
        _validate_generator_domains(model, generators)
        _validate_tracked_generators(model, sources, generators)
        self.components = None
        if self.network:
            from ._native_initialization_network import NativeNetworkInitialization

            self.components = NativeNetworkInitialization(
                model,
                packet,
                capture=capture,
                reinit_weights=options.get("reinit_weights", True),
                reinit_delays=options.get("reinit_delays", True),
            )
            self.slots.extend(
                (owner, "_native_initialization_draws", controller)
                for owner, controller in self.components.slots()
            )

    def poisson_schedule(self, waveform, generator):
        if self.active is None:
            raise ValueError(
                "Poisson evaluation occurred outside a Population initialization boundary"
            )
        return self.active.poisson.poisson_schedule(waveform, generator)

    def run(self, options):
        for boundary in self.boundaries.values():
            boundary.random.validate_source()
            if boundary.poisson.plan is not None:
                boundary.poisson.plan.validate_source()
        with _bound_replay_slots(self.slots):
            self.started = True
            self.model.initialize(**options)
            for boundary in self.boundaries.values():
                if not boundary.completed:
                    if not boundary.cached:
                        raise ValueError(
                            "Initialization did not complete every Population boundary"
                        )
                    boundary.finish(cached=True)
            if self.components is not None:
                self.components.finish()
        if not self.capture:
            return self.model
        random = {
            name: boundary.result[0] for name, boundary in self.boundaries.items()
        }
        stimulation = {
            name: boundary.result[1] for name, boundary in self.boundaries.items()
        }
        if not self.network:
            return InitializationDraws(random["population"], stimulation["population"])
        return InitializationDraws(
            random,
            stimulation,
            self.components.netstim_packet(),
            self.components.connections_packet(),
        )


def initialize_native(model, *, draws, options):
    """Replay draws through the ordinary mutating native initializer once."""
    _cold_boundary("native initialization draw replay")
    from dendra.models import Network
    from dendra.models.core import _validate_time_scalar

    if isinstance(model, Network):
        options = dict(options)
        options["dt"] = _validate_time_scalar(options["dt"], name="dt", positive=True)
        options["t"] = _validate_time_scalar(
            options.get("t", 0.0), name="t", positive=None
        )
    session = _NativeInitialization(model, draws, capture=False, options=options)
    try:
        return session.run(options)
    except Exception:
        if session.started:
            for population in session.populations.values():
                population.initialized = False
                population.integrator.initialized = False
            if session.network:
                model._mode_requires_initialize = True
        raise


@contextmanager
def _capture_copy(model):
    from dendra.helpers import ctx, current_check_purity

    from ._population import (
        _authored_workspace_classes,
        _class_state_snapshot,
        _deepcopy_execution_module,
        _global_rng_changes,
        _global_rng_snapshot,
        _rebind_population_time_references,
        _restore_class_state,
        _restore_global_rng,
    )

    populations = _populations(model)
    device = next(iter(populations.values())).device()
    snapshot = _global_rng_snapshot(device)
    check_purity = current_check_purity()
    classes = (
        _class_state_snapshot(
            tuple(
                dict.fromkeys(
                    cls
                    for pop in populations.values()
                    for cls in _authored_workspace_classes(pop)
                )
            )
        )
        if check_purity
        else None
    )
    class_changes, rng_changes = set(), set()
    try:
        with (
            torch.inference_mode(False),
            torch.no_grad(),
            ctx(JIT=0, JIT_NETWORK_SOLVES=0),
        ):
            execution = _deepcopy_execution_module(model)
            from dendra.models import Network

            if isinstance(execution, Network):
                from dendra.models.stim.waveform._poisson import _poisson_waveforms

                original = _poisson_waveforms(model)
                cloned = _poisson_waveforms(execution)
                defaults = {}
                for path, node in original.items():
                    if node.generator is torch.default_generator:
                        defaults["cpu"] = cloned[path].generator
                    elif node.generator.device.type == "cuda":
                        draw_device = node.generator.device
                        if (
                            node.generator
                            is torch.cuda.default_generators[draw_device.index]
                        ):
                            defaults[str(draw_device)] = cloned[path].generator
                execution._native_initialization_default_generators = defaults

            copies = (
                execution.populations.values()
                if isinstance(execution, Network)
                else (execution,)
            )
            for population in copies:
                _rebind_population_time_references(population)
                population.jit = population.jit_network_solves = False
                population.integrator.clear_jit_cache()
            from dendra.models.stim.waveform._poisson import (
                _poisson_waveforms,
                restore_pending_poisson_rng,
            )

            for waveform in _poisson_waveforms(execution).values():
                restore_pending_poisson_rng(waveform)
            yield execution
    finally:
        try:
            if classes is not None:
                class_changes.update(_restore_class_state(classes))
        finally:
            try:
                if check_purity:
                    rng_changes.update(_global_rng_changes(snapshot, device))
            finally:
                _restore_global_rng(snapshot, device)
    if class_changes or rng_changes:
        raise FunctionalizationError(
            "Initialization capture purity diagnostics observed class-state mutation "
            f"or implicit RNG consumption: classes={sorted(class_changes)}, RNG={sorted(rng_changes)}"
        )


def sample_native_initialization_draws(model):
    """Capture a native default initialization using owned model and RNG state."""
    _cold_boundary("sample_initialization_draws")
    from dendra.models import Network

    with _capture_copy(model) as execution:
        options = {"dt": execution.dt} if isinstance(execution, Network) else {}
        return _NativeInitialization(
            execution, None, capture=True, options=options
        ).run(options)
