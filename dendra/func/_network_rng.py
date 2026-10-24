"""Canonical ownership of RNG streams shared by Network Populations."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils import _pytree

from dendra.models.rng import RNGModule, should_restore_rng

from ._random_parameters import _rng_extra, _rng_values
from ._types import FunctionalizationError, RuntimeNoiseDraw


class NetworkRNGPlan:
    """Project one explicit carry per authored stream into Population adapters.

    Independent Population streams already have a single owner and keep their
    existing nested carry. A cross-Population alias moves all Population RNG
    streams into one Network pool; the local adapters receive temporary views
    of that pool at their native execution boundary.
    """

    def __init__(self, functional, network):
        self.functional = functional
        self.network = network
        self.populations = functional.populations
        self.local_streams = {}
        self.streams = {}
        self._sources = {}
        identities = {}
        consumers = {}
        for name, plan in self.populations.items():
            local = {}
            random = plan._random_plan
            if random is not None:
                owners = self._owners(network.populations[name])
                for stream, (path, attribute) in random.streams.items():
                    source = getattr(owners[path], attribute)
                    key = identities.setdefault(id(source), f"stream_{len(identities)}")
                    self.streams.setdefault(key, (name, path, attribute))
                    self._sources.setdefault(key, source)
                    local[stream] = key
                    consumers.setdefault(key, set()).add(name)
            self.local_streams[name] = local
        self._topology = self._current_topology(network)
        self._cache_paths = {}
        for name, population in network.populations.items():
            self._cache_paths[name] = tuple(
                (
                    f"{path + '.' if path else ''}_extra_state",
                    identities[id(module)],
                    module,
                )
                for path, module in population.named_modules(remove_duplicate=False)
                if isinstance(module, RNGModule) and id(module) in identities
            )
            # An owner without declarations can still write a shared stream
            # through its registered alias when restoring a state-dict cache.
            for _path, canonical, _source in self._cache_paths[name]:
                consumers.setdefault(canonical, set()).add(name)
        self.enabled = any(len(names) > 1 for names in consumers.values())

    @staticmethod
    def _owners(population):
        from ._population import _parameterized_owner_paths

        return dict(_parameterized_owner_paths(population))

    def _current_topology(self, network):
        identities = {}
        result = []
        for name, plan in self.populations.items():
            owners = self._owners(network.populations[name])
            random = plan._random_plan
            if random is None:
                continue
            for stream, (path, attribute) in random.streams.items():
                source = getattr(owners[path], attribute)
                key = identities.setdefault(id(source), f"stream_{len(identities)}")
                result.append((name, stream, path, attribute, key))
        # load_state_dict visits every registered alias, including aliases
        # which do not themselves appear in a random declaration.
        result.extend(
            (name, "registered", path, identities[id(module)])
            for name, population in network.populations.items()
            for path, module in population.named_modules(remove_duplicate=False)
            if isinstance(module, RNGModule) and id(module) in identities
        )
        return tuple(result)

    def validate_source(self):
        """Reject changed ownership before consulting the captured stream plan."""
        self.validate_compatible_network(self.network)
        for key, (name, path, attribute) in self.streams.items():
            source = getattr(
                self._owners(self.network.populations[name])[path], attribute
            )
            if source is not self._sources[key]:
                raise FunctionalizationError(
                    "Network Population RNG stream owners changed; lower again"
                )

    def validate_compatible_network(self, network):
        """Check aliases independently of object identities after deepcopy/load."""
        if self._current_topology(network) != self._topology:
            raise FunctionalizationError(
                "Network Population RNG stream sharing differs from this functional plan; lower again"
            )

    def extract(self, network=None):
        """Snapshot each physical RNG once without changing any generator."""
        if not self.enabled:
            return {}
        network = self.network if network is None else network
        self.validate_compatible_network(network)
        return {
            key: _rng_values(
                getattr(self._owners(network.populations[name])[path], attribute),
                self.populations[name].device,
            )
            for key, (name, path, attribute) in self.streams.items()
        }

    def expand_population(self, name, state, pool):
        """Supply a Population's local stream names from the shared carry."""
        if not self.enabled or not self.local_streams[name]:
            return state
        return {
            **state,
            "random": {
                **state["random"],
                "rng": {
                    local: pool[canonical]
                    for local, canonical in self.local_streams[name].items()
                },
            },
        }

    def strip_population(self, state):
        """Remove adapter-only RNG aliases from a Network's public carry."""
        if not self.enabled or "random" not in state:
            return state
        return {
            **state,
            "random": {
                name: value for name, value in state["random"].items() if name != "rng"
            },
        }

    def update_pool(self, name, local_rng, pool):
        """Publish one Population's post-draw suffix in native execution order."""
        if not self.enabled:
            return pool
        return {
            **pool,
            **{
                canonical: local_rng[local]
                for local, canonical in self.local_streams[name].items()
            },
        }

    def cache_pool(self, name, cache, pool):
        """Replay native RNG state-dict writes, including repeated module aliases.

        Native loading traverses every registered alias in module order. A
        later saved occurrence of one stream therefore owns the resulting
        suffix even when a declaration used an earlier alias as its local key.
        """
        if not self.enabled:
            return pool
        result = dict(pool)
        for path, canonical, source in self._cache_paths[name]:
            if not should_restore_rng(
                policy="cache",
                ignore_rng_on_load=getattr(source, "_ignore_rng_on_load", False),
            ):
                continue
            extra = cache[path]
            if isinstance(extra["base_seed"], Mapping):
                values = {**extra["base_seed"], **extra["rng_state"]}
                result[canonical] = {
                    key: value.detach().to(result[canonical][key]).clone()
                    for key, value in values.items()
                }
            else:
                result[canonical] = _rng_values(extra, self.populations[name].device)
        return result

    @staticmethod
    def _samplers(pool):
        return {
            key: RNGModule.from_checkpoint(_rng_extra(values))
            for key, values in pool.items()
        }

    def _local_samplers(self, name, samplers):
        return {
            local: samplers[canonical]
            for local, canonical in self.local_streams[name].items()
        }

    def sample_population_initialization_draw(self, name, parameters, state, pool):
        """Sample at one host initialization boundary from the incoming pool."""
        plan = self.populations[name]
        if not self.enabled:
            return plan.sample_initialization_draw(parameters, state)
        if plan._random_plan is None or plan._owned_cache_bound:
            return None
        local = self.expand_population(name, state, pool)
        samplers = {
            stream: RNGModule.from_checkpoint(_rng_extra(pool[canonical]))
            for stream, canonical in self.local_streams[name].items()
        }
        return plan._random_plan.sample(local, samplers=samplers)

    def sample_init_packets(self, parameters, state):
        """Draw initialization packets in authored Population order.

        Population-owned caches overwrite their shared RNG suffix at the same
        boundary as native cache restoration, before later Populations draw.
        Every sampler is isolated from source modules and incoming carry.
        """
        if self.functional._stimulation_rng_plan.enabled:
            return self.functional._stimulation_rng_plan.sample_initialization_packets(
                state
            )[0]
        if not self.enabled:
            return {
                name: None
                if self.functional._cache_enabled
                else plan.sample_initialization_draw(
                    parameters["populations"][name],
                    self.functional._stimulation_plan.expand(
                        name,
                        state["populations"][name],
                        state.get("stimulation_rng", {}),
                    ),
                )
                for name, plan in self.populations.items()
            }
        if self.functional._cache_enabled:
            return dict.fromkeys(self.populations)
        pool = state["rng"]
        samplers = self._samplers(pool)
        result = {}
        for name, plan in self.populations.items():
            if plan._owned_cache_bound:
                if plan._owned_cache_error is not None:
                    raise FunctionalizationError(
                        f"Population steady-state cache cannot be restored: {plan._owned_cache_error}"
                    )
                pool = self.cache_pool(name, plan._owned_cache_plan.default, pool)
                for key in {
                    canonical for _path, canonical, _source in self._cache_paths[name]
                }:
                    samplers[key] = RNGModule.from_checkpoint(_rng_extra(pool[key]))
                result[name] = None
            elif plan._random_plan is None:
                result[name] = None
            else:
                local = self.expand_population(name, state["populations"][name], pool)
                packet = plan._random_plan.sample(
                    local, samplers=self._local_samplers(name, samplers)
                )
                pool = self.update_pool(name, packet.rng, pool)
                result[name] = packet
        return result

    def sample_noise_tapes(self, state, steps, *, provided=None):
        """Sample time-first packets with native timestep/Population ordering."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        provided = {} if provided is None else provided
        if not self.enabled:
            return {
                name: provided[name]
                if name in provided
                else plan.sample_noise_draw_tape(
                    self.functional._stimulation_plan.expand(
                        name,
                        state["populations"][name],
                        state.get("stimulation_rng", {}),
                    ),
                    steps,
                )
                for name, plan in self.populations.items()
            }
        samplers = self._samplers(state["rng"])
        boundaries = {
            name: []
            for name, plan in self.populations.items()
            if plan._noise_plan is not None and plan._noise_plan.runtime
        }
        for _index in range(steps):
            for name in boundaries:
                plan = self.populations[name]
                if name in provided:
                    packet = _pytree.tree_map(
                        lambda value: value[_index], provided[name]
                    )
                    for local, canonical in self.local_streams[name].items():
                        samplers[canonical] = RNGModule.from_checkpoint(
                            _rng_extra(packet.rng[local])
                        )
                else:
                    packet = plan._noise_plan.sample(
                        state["populations"][name],
                        samplers=self._local_samplers(name, samplers),
                    )
                boundaries[name].append(packet)
        result = dict.fromkeys(self.populations)
        for name, packets in boundaries.items():
            if packets:
                result[name] = _pytree.tree_map(
                    lambda *values: torch.stack(values), *packets
                )
                continue
            noise = self.populations[name]._noise_plan
            local = self.expand_population(
                name, state["populations"][name], state["rng"]
            )
            result[name] = RuntimeNoiseDraw(
                {
                    key: leaf.new_empty((0, *leaf.shape))
                    for entry in noise.runtime
                    for key, leaf in zip(entry.draw_names, entry.leaves, strict=True)
                },
                _pytree.tree_map(
                    lambda value: value.new_empty((0, *value.shape)),
                    local["random"]["rng"],
                ),
            )
        return result


class NetworkStimulationRNGPlan:
    """Share physical model generators with Network Poisson consumers.

    The existing Population and stimulation plans retain their local packet
    schemas. Only the Network carry owns generator bytes; adapter-only aliases
    are reconstructed at each Population boundary.
    """

    def __init__(self, functional):
        self.functional = functional
        self.random = functional._rng_plan
        self.stimulation = functional._stimulation_plan
        self.borrowed = self._current_borrowed(functional._source)
        self.enabled = bool(self.borrowed)
        if not self.enabled:
            return
        self.random.enabled = True
        for name, plan in functional.populations.items():
            sharing = plan._stimulation_rng_plan
            sharing.configure(plan.extract())
            if plan._poisson_plan is not None:
                plan._poisson_plan.externally_owned_streams = frozenset(
                    local
                    for local, canonical in self.stimulation.local[name].items()
                    if canonical in self.borrowed
                )

    def _current_borrowed(self, network):
        generators = {}
        declared = set()
        for canonical, (name, path, attribute) in self.random.streams.items():
            source = getattr(
                self.random._owners(network.populations[name])[path], attribute
            )
            declared.add(id(source))
            generators[id(source._cpu_gen)] = (canonical, "cpu")
            for device, generator in source._device_gens.items():
                generators[id(generator)] = (canonical, str(device))
        result = {}
        requested = set()
        for name, plan in self.stimulation.plans.items():
            sources = plan._resolved_sources(network.populations[name])
            for entry in plan.entries:
                generator = sources[entry.key].generator
                requested.add(id(generator))
                if id(generator) in generators:
                    canonical = self.stimulation.local[name][entry.stream]
                    result[canonical] = generators[id(generator)]
        for owner in network.modules():
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
        return result

    def validate_compatible_network(self, network):
        if self._current_borrowed(network) != self.borrowed:
            raise FunctionalizationError(
                "Network stimulation and model RNG sharing differs from this plan; lower again"
            )

    def validate_source(self):
        self.validate_compatible_network(self.functional._source)

    def expand_pool(self, random, stimulation):
        if not self.enabled:
            return stimulation
        return {
            **stimulation,
            **{
                stream: random[owner][device].to("cpu")
                for stream, (owner, device) in self.borrowed.items()
            },
        }

    def strip_pool(self, stimulation):
        if not self.enabled:
            return stimulation
        return {
            stream: value
            for stream, value in stimulation.items()
            if stream not in self.borrowed
        }

    def expand_population(self, name, state, random, stimulation):
        result = self.stimulation.expand(
            name,
            self.random.expand_population(name, state, random),
            self.expand_pool(random, stimulation),
        )
        return self.functional.populations[name]._stimulation_rng_plan.strip_state(
            result
        )

    def full_population(self, name, state):
        return self.functional.populations[name]._stimulation_rng_plan.expand_state(
            state
        )

    def refresh_borrowed(self, name, state, random):
        """Publish model-owned cache restoration without replacing other RNGs."""
        state = self.random.expand_population(name, state, random)
        result = self.full_population(name, state)
        if name in self.stimulation.local and self.enabled:
            result = {
                **result,
                "stimulation": {
                    **result["stimulation"],
                    "rng": {
                        **result["stimulation"]["rng"],
                        **{
                            local: random[owner][device].to("cpu")
                            for local, canonical in self.stimulation.local[name].items()
                            if canonical in self.borrowed
                            for owner, device in (self.borrowed[canonical],)
                        },
                    },
                },
            }
        return self.functional.populations[name]._stimulation_rng_plan.strip_state(
            result
        )

    def sample_population_initialization(self, name, state, random, stimulation):
        """Sample one Population's two initialization families in one stream."""
        samplers = self.random._samplers(random)
        pool = self.expand_pool(random, stimulation)
        poisson_samplers = {}
        for stream, source in self.stimulation.streams.items():
            if stream in self.borrowed:
                owner, device = self.borrowed[stream]
                poisson_samplers[stream] = samplers[owner]._generator_for(device)
            else:
                generator = torch.Generator(device=source.device)
                generator.set_state(pool[stream].detach().cpu())
                poisson_samplers[stream] = generator
        local = self.expand_population(name, state, random, stimulation)
        return self.functional.populations[
            name
        ]._stimulation_rng_plan.sample_initialization(
            local,
            samplers=self.random._local_samplers(name, samplers),
            poisson_samplers={
                local: poisson_samplers[canonical]
                for local, canonical in self.stimulation.local.get(name, {}).items()
            },
        )

    def sample_initialization_packets(self, state):
        """Return replayable paired inputs in native Population order."""
        packets = dict.fromkeys(self.functional.populations)
        stimulation_packets = dict(packets)
        if self.functional._cache_enabled:
            return packets, stimulation_packets
        random = state["rng"]
        stimulation = self.expand_pool(random, state.get("stimulation_rng", {}))
        for name, plan in self.functional.populations.items():
            if plan._owned_cache_bound:
                if plan._owned_cache_error is not None:
                    raise FunctionalizationError(
                        f"Population steady-state cache cannot be restored: {plan._owned_cache_error}"
                    )
                random = self.random.cache_pool(
                    name, plan._owned_cache_plan.default, random
                )
                stimulation = self.expand_pool(random, stimulation)
                continue
            local = self.expand_population(
                name, state["populations"][name], random, stimulation
            )
            packet, stim = self.sample_population_initialization(
                name, local, random, stimulation
            )
            packets[name], stimulation_packets[name] = packet, stim
            if packet is not None:
                random = self.random.update_pool(name, packet.rng, random)
            if stim is not None:
                for stream, canonical in self.stimulation.local[name].items():
                    frame = stim.rng[stream]
                    stimulation[canonical] = frame
                    if canonical in self.borrowed:
                        owner, device = self.borrowed[canonical]
                        random = {
                            **random,
                            owner: {
                                **random[owner],
                                device: frame.to(random[owner][device]).clone(),
                            },
                        }
            stimulation = self.expand_pool(random, stimulation)
        return packets, stimulation_packets

    def publish(self, name, state, random, stimulation):
        """Publish one complete local suffix without duplicate byte ownership."""
        if self.random.enabled and self.random.local_streams[name]:
            random = self.random.update_pool(name, state["random"]["rng"], random)
        local = self.full_population(name, state)
        if self.enabled and name in self.stimulation.local:
            for stream, canonical in self.stimulation.local[name].items():
                if canonical in self.borrowed:
                    owner, device = self.borrowed[canonical]
                    random = {
                        **random,
                        owner: {
                            **random[owner],
                            device: local["stimulation"]["rng"][stream]
                            .to(random[owner][device])
                            .clone(),
                        },
                    }
        stimulation = self.stimulation.update(name, local, stimulation)
        return random, self.expand_pool(random, stimulation)

    def validate_packets(self, noise, stimulation):
        """Require every alias of a shared physical stream to agree."""
        if not self.enabled:
            return
        from ._stimulation_rng import _assert_shared_frame

        frames = {}
        for name, packet in noise.items():
            if packet is None:
                continue
            for local, canonical in self.random.local_streams[name].items():
                for device in {
                    device
                    for owner, device in self.borrowed.values()
                    if owner == canonical
                }:
                    frame = packet.rng[local][device]
                    previous = frames.setdefault((canonical, device), frame)
                    _assert_shared_frame(frame, previous)
        for name, packet in stimulation.items():
            if packet is None:
                continue
            for local, canonical in self.stimulation.local[name].items():
                if canonical in self.borrowed:
                    owner = self.borrowed[canonical]
                    frame = packet.rng[local]
                    previous = frames.setdefault(owner, frame)
                    _assert_shared_frame(frame, previous)

    def _provided(self, noise, stimulation, steps):
        noise = {} if noise is None else dict(noise)
        stimulation = {} if stimulation is None else dict(stimulation)
        populations = self.functional.populations
        if (set(noise) | set(stimulation)) - set(populations):
            raise ValueError("Stochastic tapes name an unknown Population")
        noise = {name: value for name, value in noise.items() if value is not None}
        stimulation = {
            name: value for name, value in stimulation.items() if value is not None
        }
        for name, packet in noise.items():
            populations[name]._validate_noise_draw(packet, steps=steps)
        for name, packet in stimulation.items():
            populations[name]._validate_stimulation_draw(packet, steps=steps)
        from ._stimulation_rng import _assert_shared_frame

        for canonical in self.stimulation.streams:
            consumers = {
                name
                for name, local in self.stimulation.local.items()
                if canonical in local.values()
            }
            selected = consumers.intersection(stimulation)
            if selected and selected != consumers:
                raise ValueError(
                    "Shared Poisson generators require explicit tapes for every "
                    "consuming Population, or automatic sampling for all of them"
                )
            frames = [
                stimulation[name].rng[local]
                for name in selected
                for local, stream in self.stimulation.local[name].items()
                if stream == canonical
            ]
            for frame in frames[1:]:
                _assert_shared_frame(frame, frames[0])
        supplied = {("noise", name) for name in noise} | {
            ("stimulation", name) for name in stimulation
        }
        for canonical, device in set(self.borrowed.values()):
            consumers = {
                ("noise", name)
                for name, plan in populations.items()
                if plan._noise_plan is not None
                and plan._noise_plan.runtime
                and canonical in self.random.local_streams[name].values()
            } | {
                ("stimulation", name)
                for name, local in self.stimulation.local.items()
                if any(
                    self.borrowed.get(stream) == (canonical, device)
                    for stream in local.values()
                )
            }
            if supplied.intersection(consumers) not in (set(), consumers):
                raise ValueError(
                    "Shared stimulation and model RNG streams require explicit "
                    "tapes for every consuming Population and draw family, or "
                    "automatic sampling for all of them"
                )
        self.validate_packets(noise, stimulation)
        return noise, stimulation

    def _contexts(self, state, samplers, poisson_samplers):
        result = {}
        for name, plan in self.functional.populations.items():
            local = self.expand_population(
                name,
                state["populations"][name],
                state["rng"],
                state.get("stimulation_rng", {}),
            )
            result[name] = plan._stimulation_rng_plan.begin_packets(
                local,
                samplers=self.random._local_samplers(name, samplers),
                poisson_samplers={
                    local: poisson_samplers[canonical]
                    for local, canonical in self.stimulation.local.get(name, {}).items()
                },
            )
        return result

    def sample_tapes(self, state, steps, *, noise=None, stimulation=None):
        """Sample one global ordered program, returning both packet families."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        supplied_noise, supplied_stimulation = self._provided(noise, stimulation, steps)
        samplers = self.random._samplers(state["rng"])
        pool = self.expand_pool(state["rng"], state.get("stimulation_rng", {}))
        poisson_samplers = {}
        for stream, source in self.stimulation.streams.items():
            if stream in self.borrowed:
                owner, device = self.borrowed[stream]
                poisson_samplers[stream] = samplers[owner]._generator_for(device)
            else:
                generator = torch.Generator(device=source.device)
                generator.set_state(pool[stream].detach().cpu())
                poisson_samplers[stream] = generator
        boundaries = {name: [] for name in self.functional.populations}
        for index in range(steps):
            contexts = self._contexts(state, samplers, poisson_samplers)
            local_noise = {}
            for phase in ("extra", "intra", "mechanism"):
                names = (
                    self.stimulation._ordered_names(phase)
                    if phase != "mechanism"
                    else self.functional.populations
                )
                for name in names:
                    adapter = self.functional.populations[name]._stimulation_rng_plan
                    for kind, entry in adapter.runtime_requests(phase):
                        if (
                            name in supplied_stimulation
                            if kind == "poisson"
                            else name in supplied_noise
                        ):
                            continue
                        adapter.sample_request(kind, entry, contexts[name])
                    if phase == "mechanism":
                        if name in supplied_noise:
                            packet = _pytree.tree_map(
                                lambda value: value[index], supplied_noise[name]
                            )
                            # Unrelated streams retain their Population-boundary
                            # restoration semantics. Borrowed devices are owned
                            # by the common global suffix below.
                            for local, canonical in self.random.local_streams[
                                name
                            ].items():
                                sampler = samplers[canonical]
                                current = _rng_values(
                                    sampler, self.functional.populations[name].device
                                )
                                borrowed = {
                                    device
                                    for owner, device in self.borrowed.values()
                                    if owner == canonical
                                }
                                frame = {
                                    key: current[key] if key in borrowed else value
                                    for key, value in packet.rng[local].items()
                                }
                                sampler.restore_rng_checkpoint(
                                    _rng_extra(frame), policy="checkpoint"
                                )
                            local_noise[name] = packet
                        else:
                            local_noise[name] = adapter.finish_packets(contexts[name])[
                                0
                            ]
            # Explicit coupled packets own their common post-step frame.
            for name, tape in supplied_noise.items():
                frame = _pytree.tree_map(lambda value: value[index], tape.rng)
                for local, canonical in self.random.local_streams[name].items():
                    sampler = samplers[canonical]
                    borrowed = {
                        device
                        for owner, device in self.borrowed.values()
                        if owner == canonical
                    }
                    if borrowed:
                        current = _rng_values(
                            sampler, self.functional.populations[name].device
                        )
                        suffix = {
                            key: value if key in borrowed else current[key]
                            for key, value in frame[local].items()
                        }
                        sampler.restore_rng_checkpoint(
                            _rng_extra(suffix), policy="checkpoint"
                        )
            for name, tape in supplied_stimulation.items():
                for local, canonical in self.stimulation.local[name].items():
                    poisson_samplers[canonical].set_state(
                        tape.rng[local][index].detach().cpu()
                    )
            # Refresh borrowed handles after an RNGModule checkpoint adapter.
            for stream, (owner, device) in self.borrowed.items():
                poisson_samplers[stream] = samplers[owner]._generator_for(device)
            for name, plan in self.functional.populations.items():
                context = contexts[name]
                context["random_samplers"] = self.random._local_samplers(name, samplers)
                context["poisson_samplers"] = {
                    local: poisson_samplers[canonical]
                    for local, canonical in self.stimulation.local.get(name, {}).items()
                }
                pair = plan._stimulation_rng_plan.finish_packets(context)
                local_packet = local_noise[name]
                if local_packet is not None and name not in supplied_noise:
                    # Independent streams expose their historical per-Population
                    # boundary; borrowed bytes describe the complete timestep.
                    local_packet = RuntimeNoiseDraw(
                        local_packet.draws,
                        {
                            local: {
                                key: pair[0].rng[local][key]
                                if (canonical, key) in self.borrowed.values()
                                else value
                                for key, value in frame.items()
                            }
                            for local, frame in local_packet.rng.items()
                            for canonical in (self.random.local_streams[name][local],)
                        },
                    )
                pair = (
                    local_packet,
                    _pytree.tree_map(
                        lambda value: value[index], supplied_stimulation[name]
                    )
                    if name in supplied_stimulation
                    else pair[1],
                )
                boundaries[name].append(pair)
        noise_result, stimulation_result = {}, {}
        for name, pairs in boundaries.items():
            if pairs:
                packet = _pytree.tree_map(
                    lambda *values: None if values[0] is None else torch.stack(values),
                    *pairs,
                )
            else:
                local = self.expand_population(
                    name,
                    state["populations"][name],
                    state["rng"],
                    state.get("stimulation_rng", {}),
                )
                packet = self.functional.populations[
                    name
                ]._stimulation_rng_plan.sample_runtime(
                    local,
                    steps=0,
                    samplers=self.random._local_samplers(name, samplers),
                    poisson_samplers={
                        local: poisson_samplers[canonical]
                        for local, canonical in self.stimulation.local.get(
                            name, {}
                        ).items()
                    },
                )
            noise_result[name], stimulation_result[name] = packet
        self.validate_packets(noise_result, stimulation_result)
        return noise_result, stimulation_result
