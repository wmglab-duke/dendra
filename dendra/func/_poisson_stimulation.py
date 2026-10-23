"""Explicit Poisson invocation draws and generator ownership for stimulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.utils import _pytree

from dendra.models.stim.waveform._poisson import (
    poisson_schedule_from_draws,
    sample_poisson_draws,
)
from dendra.models.stim.waveform.core import (
    Product,
    Reciprocal,
    Sum,
    _poisson,
    _randomized_poisson,
    _repeat,
)

from ._types import FunctionalizationError, StimulationDraw


def normalize_schedule(module, raw, *, dtype=None, device=None):
    """Own a fixed-size schedule without changing the native source buffer."""
    raw = raw.detach()
    width = max(module.n, 1)
    value = torch.full(
        (width,),
        torch.inf,
        dtype=raw.dtype if dtype is None else dtype,
        device=raw.device if device is None else device,
    )
    flat = raw.reshape(-1)
    count = min(width, flat.numel())
    value[:count] = flat[:count].to(value)
    return value.reshape(value.shape + (1,) * module._spike_time_trailing_dims)


def randomized_nodes(waveform):
    """Visit invocation paths, retaining independent registration occurrences."""
    seen = set()

    def visit(module, path, poisson_parent=False):
        canonical = (
            _randomized_poisson,
            _poisson,
            _repeat,
            Sum,
            Product,
            Reciprocal,
            torch.nn.ModuleList,
            torch.nn.ModuleDict,
        )
        if type(module) not in canonical and any(
            type(child) is _randomized_poisson
            for child in module.modules()
            if child is not module
        ):
            raise FunctionalizationError(
                "Custom waveform containers with randomized Poisson children are currently unsupported; "
                "compose Poisson waveforms with Waveform arithmetic or repeat()."
            )
        if type(module) is _randomized_poisson:
            if module.n is None:
                raise FunctionalizationError(
                    "Functional randomized Poisson stimulation requires finite n."
                )
            if poisson_parent or id(module) in seen:
                raise FunctionalizationError(
                    "Nested or repeatedly referenced randomized Poisson waveforms "
                    "within one expression are currently unsupported."
                )
            seen.add(id(module))
            yield path, module
        for name, child in module._modules.items():
            if child is not None:
                yield from visit(
                    child,
                    f"{path}.{name}" if path else name,
                    poisson_parent or isinstance(module, _poisson),
                )

    return tuple(visit(waveform, ""))


def configure_execution(module):
    """Remove RNG dispatch only from an isolated execution module."""
    for child in module.modules():
        if type(child) is _randomized_poisson:
            child._functional_poisson_recipe = True
            child.__class__ = _poisson


@dataclass(frozen=True)
class _Entry:
    key: str
    kind: str
    parameter: str
    mapping: str | None
    source: object
    stream: str
    dtype: torch.dtype
    device: torch.device
    root_index: int | None
    node_path: str


def _alias_signature(sources):
    modules, generators = {}, {}
    return tuple(
        (
            modules.setdefault(id(source), len(modules)),
            generators.setdefault(id(source.generator), len(generators)),
            str(source.generator.device),
        )
        for source in sources
    )


def _cold_validation():
    if torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():
        return False
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return get_proxy_mode() is None


def _validate_generator_domains(owner, generators):
    """Reject aliases across adapters with independently owned RNG carries."""
    from dendra.models.rng import RNGMixin

    identities = {id(generator) for generator in generators}
    if not identities:
        return
    for module in owner.modules():
        if isinstance(module, RNGMixin) and any(
            id(generator) in identities
            for generator in (module._cpu_gen, *module._device_gens.values())
        ):
            raise FunctionalizationError(
                "Sharing a Poisson generator with a registered RNGModule or "
                "model-owned private RNG is currently unsupported. Supply a "
                "separate torch.Generator for the Poisson waveform."
            )


class PoissonStimulationPlan:
    """Fixed invocation recipes; draws and RNG positions remain tensor inputs."""

    def __init__(self, functional):
        self.functional = functional
        self.population = functional._population
        entries, streams, identities = [], {}, {}
        # Native step evaluates extracellular contacts before intracellular
        # injections, then enters mechanism evaluation.
        for kind, plan in (("extra", functional._extra), ("intra", functional._intra)):
            bank = plan._waveforms
            for index, waveform in enumerate(bank._sources):
                for path, source in randomized_nodes(waveform):
                    relative = f"{path + '.' if path else ''}_spike_times"
                    parameter = bank._public_name(index, relative)
                    stream = identities.setdefault(
                        id(source.generator), f"stream_{len(identities)}"
                    )
                    streams.setdefault(stream, source.generator)
                    entries.append(
                        _Entry(
                            parameter,
                            kind,
                            parameter,
                            None,
                            source,
                            stream,
                            functional.dtype,
                            functional.device,
                            index,
                            path,
                        )
                    )
        from ._mechanism_injections import mechanism_injection_owners

        for owner_path, owner in mechanism_injection_owners(self.population):
            for index, waveform in enumerate(owner.injected_waveforms):
                for path, source in randomized_nodes(waveform):
                    relative = f"{owner_path}.injected_waveforms.{index}.{path + '.' if path else ''}_spike_times"
                    stream = identities.setdefault(
                        id(source.generator), f"stream_{len(identities)}"
                    )
                    streams.setdefault(stream, source.generator)
                    entries.append(
                        _Entry(
                            relative,
                            "mechanism",
                            relative,
                            f"population.{relative}",
                            source,
                            stream,
                            functional.dtype,
                            functional.device,
                            None,
                            relative.rsplit(".", 1)[0],
                        )
                    )
        self.entries, self.streams = tuple(entries), streams
        self.enabled = bool(entries)
        self.has_initialization = any(entry.kind == "mechanism" for entry in entries)
        self._runtime_keys = frozenset(entry.key for entry in self.entries)
        self._initialization_keys = frozenset(
            entry.key for entry in self.entries if entry.kind == "mechanism"
        )
        self._runtime_entries = self.entries
        self._initialization_entries = tuple(
            entry for entry in self.entries if entry.kind == "mechanism"
        )
        self.draw_dtype = torch.get_default_dtype()
        self._aliases = _alias_signature(entry.source for entry in self.entries)
        self._rng_shapes = {
            name: tuple(generator.get_state().shape)
            for name, generator in streams.items()
        }
        _validate_generator_domains(self.population, streams.values())

    def set_invocation_profile(
        self, *, runtime, initialization, runtime_order=None, initialization_order=None
    ):
        """Freeze the admitted zero-or-one call profile of mechanism hooks."""
        mechanism = {entry.key for entry in self.entries if entry.kind == "mechanism"}
        if set(runtime) != mechanism or set(initialization) != mechanism:
            raise FunctionalizationError(
                "Poisson mechanism invocation profiles changed registration keys."
            )
        for profile in (runtime, initialization):
            if any(
                type(value) is not int or value not in (0, 1)
                for value in profile.values()
            ):
                raise FunctionalizationError(
                    "A registered Poisson waveform may be evaluated at most once per mechanism phase."
                )
        self._runtime_keys = frozenset(
            {entry.key for entry in self.entries if entry.kind != "mechanism"}
            | {key for key, count in runtime.items() if count}
        )
        self._initialization_keys = frozenset(
            key for key, count in initialization.items() if count
        )
        self.has_initialization = bool(self._initialization_keys)
        by_key = {entry.key: entry for entry in self.entries}
        orders = []
        for profile, order in (
            (runtime, runtime_order),
            (initialization, initialization_order),
        ):
            expected = {key for key, count in profile.items() if count}
            if order is None:
                order = tuple(
                    entry.key for entry in self.entries if entry.key in expected
                )
            if len(order) != len(expected) or set(order) != expected:
                raise FunctionalizationError(
                    "Poisson mechanism invocation order does not match its active registrations."
                )
            orders.append(tuple(by_key[key] for key in order))
        self._runtime_entries = (
            tuple(entry for entry in self.entries if entry.kind != "mechanism")
            + orders[0]
        )
        self._initialization_entries = orders[1]

    def _entries_for_phase(self, kind, *, initialization=False):
        entries = (
            self._initialization_entries if initialization else self._runtime_entries
        )
        return tuple(entry for entry in entries if entry.kind == kind)

    def _resolve_source(self, entry, population=None):
        population = self.population if population is None else population
        if entry.kind == "mechanism":
            return population.get_submodule(entry.node_path)
        if entry.kind == "intra":
            from ._stimuli import _solver_injection_specs

            source = _solver_injection_specs(population)[entry.root_index][0]
        else:
            source = self.functional._extra._waveforms._sources[entry.root_index]
        return source.get_submodule(entry.node_path) if entry.node_path else source

    def _resolved_sources(self, population=None):
        original = population is None or population is self.population
        try:
            sources = {
                entry.key: self._resolve_source(entry, population)
                for entry in self.entries
            }
        except (AttributeError, IndexError, KeyError) as error:
            raise FunctionalizationError(
                "Poisson waveform bindings changed; lower again."
            ) from error
        if any(type(source) is not _randomized_poisson for source in sources.values()):
            raise FunctionalizationError(
                "Poisson waveform recipes changed; lower again."
            )
        if _alias_signature(sources.values()) != self._aliases:
            raise FunctionalizationError(
                "Poisson waveform or generator sharing changed; lower again."
            )
        if original and any(
            sources[entry.key] is not entry.source
            or sources[entry.key].generator is not self.streams[entry.stream]
            for entry in self.entries
        ):
            raise FunctionalizationError(
                "Poisson generator bindings changed; lower again."
            )
        return sources

    def validate_source(self):
        self._resolved_sources()
        _validate_generator_domains(self.population, self.streams.values())

    def validate_compatible_population(self, population):
        sources = self._resolved_sources(population)
        _validate_generator_domains(
            population, (source.generator for source in sources.values())
        )

    def extract(self, population=None):
        from dendra.models.stim.waveform._poisson import poisson_generator_state

        sources = self._resolved_sources(population)
        rng = {}
        schedules = {}
        for entry in self.entries:
            source = sources[entry.key]
            rng.setdefault(entry.stream, poisson_generator_state(source))
            schedules[entry.key] = normalize_schedule(
                source, source._spike_times, dtype=entry.dtype, device=entry.device
            ).reshape(-1)
        return {"rng": rng, "schedules": schedules}

    def _samplers(self, state, samplers):
        self.validate_source()
        if samplers is not None:
            if set(samplers) != set(self.streams):
                raise ValueError("Poisson sampler names do not match this plan.")
            return samplers
        result = {}
        for name, source in self.streams.items():
            generator = torch.Generator(device=source.device)
            generator.set_state(state["stimulation"]["rng"][name].detach().cpu())
            result[name] = generator
        return result

    def sample_initialization(self, state, *, samplers=None):
        """Sample mechanism-owned initialization calls without drawing bank stimuli."""
        samplers = self._samplers(state, samplers)
        draws = self._placeholder_draws()
        for entry in self._initialization_entries:
            source = entry.source
            draw, _ = sample_poisson_draws(
                interval=source.interval,
                n=source.n,
                start=source.start,
                noise=source.noise,
                off=source.off,
                generator=samplers[entry.stream],
                dtype=self.draw_dtype,
            )
            draws[entry.key] = draw.detach().clone()
        return StimulationDraw(
            draws,
            {
                name: generator.get_state().clone()
                for name, generator in samplers.items()
            },
        )

    def initialization_mapping(self, packet):
        return self._schedule_mapping(self.schedules(packet), self._initialization_keys)

    def _placeholder_draws(self):
        return {
            entry.key: torch.full(
                (entry.source.n,),
                0.5,
                dtype=self.draw_dtype,
                device=self.streams[entry.stream].device,
            )
            for entry in self.entries
        }

    def sample(self, state, *, steps=None, samplers=None):
        if steps is not None:
            packets, carry = [], state
            for _ in range(steps):
                packet = self.sample(carry, samplers=samplers)
                packets.append(packet)
                carry = {
                    **carry,
                    "stimulation": self.advance(carry["stimulation"], packet),
                }
            if packets:
                return _pytree.tree_map(lambda *values: torch.stack(values), *packets)
            return StimulationDraw(
                {
                    entry.key: torch.empty(
                        (0, entry.source.n),
                        dtype=self.draw_dtype,
                        device=self.streams[entry.stream].device,
                    )
                    for entry in self.entries
                },
                {
                    name: value.new_empty((0, *value.shape))
                    for name, value in state["stimulation"]["rng"].items()
                },
            )
        samplers = self._samplers(state, samplers)
        draws = self._placeholder_draws()
        for entry in self._runtime_entries:
            source = entry.source
            draw, _count = sample_poisson_draws(
                interval=source.interval,
                n=source.n,
                start=source.start,
                noise=source.noise,
                off=source.off,
                generator=samplers[entry.stream],
                dtype=self.draw_dtype,
            )
            draws[entry.key] = draw.detach().clone()
        return StimulationDraw(
            draws,
            {key: generator.get_state().clone() for key, generator in samplers.items()},
        )

    def _validate_rng(self, values, *, steps=None):
        if not isinstance(values, Mapping) or set(values) != set(self.streams):
            raise ValueError("Poisson stimulation RNG names do not match this plan.")
        for key, source in self.streams.items():
            value = values[key]
            shape = self._rng_shapes[key]
            if steps is not None:
                shape = (steps, *shape)
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != shape
                or value.dtype is not torch.uint8
                or value.device.type != "cpu"
            ):
                raise ValueError("Poisson stimulation RNG changed Tensor schema")
            if _cold_validation():
                validator = torch.Generator(device=source.device)
                for frame in (value,) if steps is None else value.unbind(0):
                    try:
                        validator.set_state(frame.detach().clone())
                    except RuntimeError as error:
                        raise ValueError(
                            "Poisson stimulation RNG contains an invalid frame."
                        ) from error

    def validate(self, packet, *, steps=None):
        if not isinstance(packet, StimulationDraw):
            raise ValueError(
                "Randomized Poisson stimulation requires an explicit StimulationDraw packet"
            )
        if not isinstance(packet.draws, Mapping) or set(packet.draws) != {
            entry.key for entry in self.entries
        }:
            raise ValueError(
                "Poisson stimulation draw or RNG names do not match this plan"
            )
        for entry in self.entries:
            value = packet.draws[entry.key]
            shape = (entry.source.n,) if steps is None else (steps, entry.source.n)
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != shape
                or value.dtype != self.draw_dtype
                or value.device != self.streams[entry.stream].device
            ):
                raise ValueError("Poisson stimulation draw changed Tensor schema")
            if _cold_validation() and bool(
                (
                    (value.detach() < 0)
                    | (value.detach() >= 1)
                    | ~torch.isfinite(value.detach())
                ).any()
            ):
                raise ValueError(
                    "Poisson stimulation uniforms must be finite in [0, 1)."
                )
        self._validate_rng(packet.rng, steps=steps)

    def validate_state(self, state):
        """Preflight complete carried state before a native commit transaction."""
        if not isinstance(state, Mapping) or set(state) != {"rng", "schedules"}:
            raise ValueError("Poisson carry requires RNG frames and latest schedules.")
        self._validate_rng(state["rng"])
        schedules = state["schedules"]
        if not isinstance(schedules, Mapping) or set(schedules) != {
            entry.key for entry in self.entries
        }:
            raise ValueError("Poisson carried schedule names do not match this plan.")
        latest = {}
        for entry in self.entries:
            value = schedules[entry.key]
            if (
                not torch.is_tensor(value)
                or value.shape != (max(entry.source.n, 1),)
                or value.dtype != entry.dtype
                or value.device != entry.device
            ):
                raise ValueError("Poisson carried schedule changed Tensor schema.")
            if _cold_validation():
                if bool((torch.isnan(value) | torch.isneginf(value)).any()) or bool(
                    (value[1:] < value[:-1]).any()
                ):
                    raise ValueError("Poisson carry contains an invalid schedule.")
                previous = latest.setdefault(id(entry.source), value)
                if not torch.equal(value, previous):
                    raise ValueError(
                        "Shared Poisson waveform aliases have different latest schedules."
                    )

    def schedules(self, packet):
        result = {}
        for entry in self.entries:
            source = entry.source
            result[entry.key] = poisson_schedule_from_draws(
                packet.draws[entry.key].detach(),
                interval=source.interval,
                start=source.start,
                noise=source.noise,
                off=source.off,
                dtype=entry.dtype,
                device=entry.device,
            )
        return result

    def parameters(self, parameters, packet):
        result = dict(parameters)
        schedules = self.schedules(packet)
        for entry in self.entries:
            if entry.kind != "mechanism":
                value = schedules[entry.key]
                result[entry.parameter] = value.reshape(
                    value.shape + (1,) * entry.source._spike_time_trailing_dims
                )
        return result

    def mapping(self, packet):
        return self._schedule_mapping(self.schedules(packet), self._runtime_keys)

    def _schedule_mapping(self, schedules, active):
        return {
            entry.mapping: schedules[entry.key].reshape(
                schedules[entry.key].shape
                + (1,) * entry.source._spike_time_trailing_dims
            )
            for entry in self.entries
            if entry.mapping is not None and entry.key in active
        }

    def state_mapping(self, state):
        """Bind snapshot or carried mechanism schedules without drawing randomness."""
        return {
            entry.mapping: state["schedules"][entry.key].reshape(
                state["schedules"][entry.key].shape
                + (1,) * entry.source._spike_time_trailing_dims
            )
            for entry in self.entries
            if entry.mapping is not None
        }

    def advance(self, state, packet, *, tape=False, initialization=False):
        if tape:
            packet = _pytree.tree_map(lambda value: value[-1], packet)
        schedules = self.schedules(packet)
        latest = {}
        active = self._initialization_keys if initialization else self._runtime_keys
        for entry in self.entries:
            if entry.key in active:
                latest[id(entry.source)] = schedules[entry.key]
        return {
            "rng": {name: value.detach().clone() for name, value in packet.rng.items()},
            "schedules": {
                entry.key: latest.get(id(entry.source), state["schedules"][entry.key])
                for entry in self.entries
            },
        }

    def commit(self, population, state):
        from dendra.models.stim.waveform._poisson import defer_poisson_generator_restore

        self.validate_state(state)
        resolved = self._resolved_sources(population)
        groups = {}
        prepared = []
        for entry in self.entries:
            if entry.kind == "extra":
                continue
            source = resolved[entry.key]
            value = state["schedules"][entry.key]
            valid = value[torch.isfinite(value)]
            value = valid if valid.numel() else value[:1]
            prepared.append(
                (
                    source,
                    value.to(source._spike_times)
                    .reshape(value.shape + (1,) * source._spike_time_trailing_dims)
                    .clone(),
                )
            )
            groups.setdefault(entry.stream, []).append(source)
        self._validate_rng(state["rng"])
        for source, value in prepared:
            source._spike_times = value
        for stream, sources in groups.items():
            defer_poisson_generator_restore(sources, state["rng"][stream])


class NetworkStimulationPlan:
    """One authoritative byte-state per generator shared by Network stimuli."""

    def __init__(self, functional):
        self.functional = functional
        self.plans = {
            name: plan._poisson_plan
            for name, plan in functional.populations.items()
            if plan._poisson_plan is not None
        }
        self.streams, self.local = {}, {}
        identities = {}
        for name, plan in self.plans.items():
            local = {}
            for key, generator in plan.streams.items():
                canonical = identities.setdefault(
                    id(generator), f"stream_{len(identities)}"
                )
                self.streams.setdefault(canonical, generator)
                local[key] = canonical
            self.local[name] = local
        self.enabled = bool(self.streams)
        from dendra.models.stim.waveform._poisson import _is_default_poisson_generator

        self.default_streams = {
            name: str(generator.device)
            for name, generator in self.streams.items()
            if _is_default_poisson_generator(generator)
        }
        self._aliases = _alias_signature(
            entry.source for plan in self.plans.values() for entry in plan.entries
        )
        _validate_generator_domains(functional._source, self.streams.values())

    def validate_source(self):
        _validate_generator_domains(self.functional._source, self.streams.values())
        for plan in self.plans.values():
            plan.validate_source()
        current = _alias_signature(
            source
            for plan in self.plans.values()
            for source in plan._resolved_sources().values()
        )
        if current != self._aliases:
            raise FunctionalizationError(
                "Network Poisson generator sharing changed; lower again."
            )

    def validate_compatible_network(self, network):
        resolved = {
            name: plan._resolved_sources(network.populations[name])
            for name, plan in self.plans.items()
        }
        sources = (source for values in resolved.values() for source in values.values())
        sources = tuple(sources)
        _validate_generator_domains(network, (source.generator for source in sources))
        if _alias_signature(sources) != self._aliases:
            raise FunctionalizationError(
                "Network Poisson waveform or generator sharing does not match this plan."
            )
        from dendra.models.stim.waveform._poisson import _is_default_poisson_generator

        connection = getattr(self.functional, "_connection_sampling_plan", None)
        devices = () if connection is None else connection.devices
        for name, plan in self.plans.items():
            for entry in plan.entries:
                canonical = self.local[name][entry.stream]
                if self.default_streams.get(
                    canonical
                ) in devices and not _is_default_poisson_generator(
                    resolved[name][entry.key].generator
                ):
                    raise FunctionalizationError(
                        "Poisson generators shared with connection sampling must retain default-stream ownership."
                    )

    def canonicalize(self, states, *, initialization=False, cached=()):
        """Publish the last native invocation or cache load to every source alias."""
        latest = {}
        for name, plan in self.plans.items():
            for entry in plan.entries:
                latest.setdefault(
                    id(entry.source),
                    states[name]["stimulation"]["schedules"][entry.key],
                )
        for kind in ("extra", "intra", "mechanism"):
            for name in self._ordered_names(kind):
                plan = self.plans[name]
                entries = (
                    tuple(entry for entry in plan.entries if entry.kind == kind)
                    if initialization and name in cached and kind == "mechanism"
                    else plan._entries_for_phase(kind, initialization=initialization)
                )
                for entry in entries:
                    latest[id(entry.source)] = states[name]["stimulation"]["schedules"][
                        entry.key
                    ]
        return {
            name: (
                {
                    **state,
                    "stimulation": {
                        **state["stimulation"],
                        "schedules": {
                            entry.key: latest[id(entry.source)]
                            for entry in self.plans[name].entries
                        },
                    },
                }
                if name in self.plans
                else state
            )
            for name, state in states.items()
        }

    def validate_state(self, state):
        """Validate shared source state before publishing a Network checkpoint."""
        schedules = {}
        for name, plan in self.plans.items():
            local = self.expand(
                name, state["populations"][name], state["stimulation_rng"]
            )["stimulation"]
            plan.validate_state(local)
            if _cold_validation():
                for entry in plan.entries:
                    value = local["schedules"][entry.key]
                    previous = schedules.setdefault(id(entry.source), value)
                    if not torch.equal(value, previous):
                        raise ValueError(
                            "Shared Poisson waveform aliases have different latest schedules across Populations."
                        )

    def _ordered_names(self, kind):
        order = (
            getattr(self.functional, "_extra_order", self.plans)
            if kind == "extra"
            else self.plans
        )
        return tuple(name for name in order if name in self.plans)

    def extract(self, network=None):
        if network is None:
            self.validate_source()
        else:
            self.validate_compatible_network(network)
        result = {}
        for name, plan in self.plans.items():
            state = plan.extract(None if network is None else network.populations[name])
            for local, canonical in self.local[name].items():
                result.setdefault(canonical, state["rng"][local])
        return result

    def expand(self, name, state, pool):
        if name not in self.plans:
            return state
        return {
            **state,
            "stimulation": {
                **state["stimulation"],
                "rng": {
                    local: pool[canonical]
                    for local, canonical in self.local[name].items()
                },
            },
        }

    def strip(self, state):
        if "stimulation" not in state:
            return state
        return {
            **state,
            "stimulation": {
                key: value
                for key, value in state["stimulation"].items()
                if key != "rng"
            },
        }

    def update(self, name, state, pool):
        if name not in self.plans:
            return pool
        return {
            **pool,
            **{
                canonical: state["stimulation"]["rng"][local]
                for local, canonical in self.local[name].items()
            },
        }

    def sample_tapes(self, state, steps, provided=None):
        self.validate_source()
        if provided is not None and not isinstance(provided, Mapping):
            raise ValueError("Poisson stimulation tapes must be a Population mapping.")
        provided = (
            {}
            if provided is None
            else {
                name: packet for name, packet in provided.items() if packet is not None
            }
        )
        if set(provided) - set(self.plans):
            raise ValueError("Poisson stimulation tapes name an unbound Population.")
        for name, packet in provided.items():
            self.plans[name].validate(packet, steps=steps)
        consumers = {key: set() for key in self.streams}
        for name, local in self.local.items():
            for canonical in local.values():
                consumers[canonical].add(name)
        explicit_streams = set()
        for canonical, names in consumers.items():
            selected = names.intersection(provided)
            if selected and selected != names:
                raise ValueError(
                    "Shared Poisson generators require explicit tapes for every consuming Population, "
                    "or automatic sampling for all of them."
                )
            if selected:
                explicit_streams.add(canonical)
                frames = [
                    provided[name].rng[local]
                    for name in names
                    for local, key in self.local[name].items()
                    if key == canonical
                ]
                if any(not torch.equal(frame, frames[0]) for frame in frames[1:]):
                    raise ValueError(
                        "Explicit tapes disagree on a shared Poisson RNG end frame."
                    )
        if not self.enabled:
            return {name: None for name in self.functional.populations}
        samplers = {}
        for key, source in self.streams.items():
            if key in explicit_streams:
                continue
            generator = torch.Generator(device=source.device)
            generator.set_state(state["stimulation_rng"][key].detach().cpu())
            samplers[key] = generator
        packets = {name: [] for name in self.plans if name not in provided}
        for index in range(steps):
            draws = {
                name: {
                    entry.key: torch.full(
                        (entry.source.n,),
                        0.5,
                        dtype=plan.draw_dtype,
                        device=self.streams[self.local[name][entry.stream]].device,
                    )
                    for entry in plan.entries
                }
                for name, plan in self.plans.items()
                if name in packets
            }
            for kind in ("extra", "intra", "mechanism"):
                for name in self._ordered_names(kind):
                    plan = self.plans[name]
                    if name in provided:
                        continue
                    for entry in plan._entries_for_phase(kind):
                        canonical = self.local[name][entry.stream]
                        source = entry.source
                        value, _count = sample_poisson_draws(
                            interval=source.interval,
                            n=source.n,
                            start=source.start,
                            noise=source.noise,
                            off=source.off,
                            generator=samplers[canonical],
                            dtype=plan.draw_dtype,
                        )
                        draws[name][entry.key] = value.detach().clone()
            for name in packets:
                packets[name].append(
                    StimulationDraw(
                        draws[name],
                        {
                            local: samplers[canonical].get_state().clone()
                            for local, canonical in self.local[name].items()
                        },
                    )
                )
        result = {name: None for name in self.functional.populations}
        result.update(provided)
        for name in packets:
            plan = self.plans[name]
            result[name] = (
                _pytree.tree_map(lambda *values: torch.stack(values), *packets[name])
                if steps
                else plan.sample(
                    self.expand(
                        name, state["populations"][name], state["stimulation_rng"]
                    ),
                    steps=0,
                )
            )
        return result
