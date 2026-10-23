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
