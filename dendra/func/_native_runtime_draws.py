"""Native single-step bindings for NOISE/SDE, NetStim and bank Poisson draws."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext

import torch
from torch.utils import _pytree

from dendra.models.rng import RNGModule
from dendra.models.stim.waveform._poisson import (
    _NativePoissonInvocationReplay,
    _poisson_waveforms,
    native_poisson_schedule_from_draws,
)

from ._draws import _normalize_runtime_draws
from ._network_draws import _keys, _tensor
from ._network_rng import _network_stream_inventory
from ._poisson_stimulation import PoissonStimulationPlan
from ._population import _parameterized_owner_paths
from ._rng_frames import _rng_extra, _rng_values
from ._runtime_noise import RuntimeNoisePlan
from ._stimulation_rng import _generator_bindings, _validate_tracked_generators
from ._types import RuntimeNoiseDraw, StimulationDraw


def _reject_poisson(population, extra):
    if _poisson_waveforms(population):
        raise ValueError("Native draws do not support randomized Poisson stimulation")

    def visit(value):
        if isinstance(value, torch.nn.Module):
            if _poisson_waveforms(value):
                raise ValueError(
                    "Native draws do not support randomized Poisson stimulation"
                )
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(extra)


def _bank_poisson_plan(population, extra):
    handler = getattr(population, "mech", None)
    if handler is not None and _poisson_waveforms(handler):
        raise ValueError(
            "Native draws do not support mechanism-owned Poisson stimulation"
        )
    roots = []

    def visit(value):
        if isinstance(value, torch.nn.Module):
            roots.append(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(extra)
    registered = _poisson_waveforms(population)
    if not registered and not any(_poisson_waveforms(root) for root in roots):
        return None
    plan = PoissonStimulationPlan.native_banks(population, extra)
    bank_nodes = {id(entry.source) for entry in plan.entries}
    if any(id(node) not in bank_nodes for node in registered.values()):
        raise ValueError(
            "Native draws support randomized Poisson only in Population waveform banks"
        )
    return plan if plan.enabled else None


@contextmanager
def _bound_replay_slots(slots):
    previous = []
    missing = object()
    try:
        for owner, name, values in slots:
            previous.append((owner, name, vars(owner).get(name, missing)))
            setattr(owner, name, values)
        yield
    finally:
        for owner, name, value in reversed(previous):
            if value is missing:
                delattr(owner, name)
            else:
                setattr(owner, name, value)


def _prepare_rng(source, frame):
    """Own and validate bytes on an isolated sampler before live publication."""
    payload = _rng_extra(frame)
    RNGModule.from_checkpoint(payload)
    return source, payload


def _publish(updates):
    for source, payload in updates:
        source.restore_rng_checkpoint(payload, policy="checkpoint")


class _PopulationReplay:
    def __init__(self, population, draws, extra, *, bank_poisson=False):
        if not bank_poisson:
            _reject_poisson(population, extra)
        self.poisson = _bank_poisson_plan(population, extra) if bank_poisson else None
        self.noise = RuntimeNoisePlan(population)
        self.owners = dict(_parameterized_owner_paths(population))
        self.sources = {
            stream: getattr(self.owners[path], attribute)
            for stream, (path, attribute) in self.noise.streams.items()
        }
        generators = _generator_bindings(self.sources)
        self.borrowed = (
            {}
            if self.poisson is None
            else {
                stream: generators[id(generator)]
                for stream, generator in self.poisson.streams.items()
                if id(generator) in generators
            }
        )
        if self.poisson is not None:
            _validate_tracked_generators(
                population, self.sources, self.poisson.streams.values()
            )
        borrowed = {owner for owner, _device in self.borrowed.values()}
        self.reference = (
            {
                stream: _rng_values(source, population.device())
                for stream, source in self.sources.items()
                if self.noise.runtime or stream in borrowed
            }
            if self.noise.runtime or borrowed
            else {}
        )
        _keys(draws, ("noise", "stimulation"), "draws.draws")
        if self.poisson is None and draws["stimulation"] is not None:
            raise ValueError(
                "Native draws do not support randomized Poisson stimulation"
            )
        selected = draws["noise"]
        if self.noise.runtime:
            if not isinstance(selected, Mapping):
                raise ValueError("Native NOISE/SDE draws must be a mapping")
            self.noise.validate(RuntimeNoiseDraw(selected, {}))
        elif selected is not None:
            raise ValueError(
                "This Population has no recurrent NOISE or State SDE draws"
            )
        self.slots = []
        by_owner = {}
        for entry in self.noise.noise_runtime:
            owner = self.owners[entry.path]
            local = by_owner.setdefault(owner, {})
            local[entry.name] = _pytree.tree_unflatten(
                [selected[name].detach().clone() for name in entry.draw_names],
                entry.draw_spec,
            )
        self.slots.extend(
            (owner, "_native_runtime_noise_draws", values)
            for owner, values in by_owner.items()
        )
        sde = {}
        for entry in self.noise.sde.entries:
            sde.setdefault(self.owners[entry.path], {})[entry.name] = (
                selected[entry.key].detach().clone()
            )
        self.slots.extend(
            (owner, "_native_sde_draws", values) for owner, values in sde.items()
        )
        self.updates = []
        self.poisson_updates = []
        self.poisson_slots = []
        self.bank_replay = None
        if self.poisson is not None:
            packet = StimulationDraw(draws["stimulation"], {})
            self.poisson.validate(packet, validate_rng=False)
            positions = {}
            invocations = []
            for entry in self.poisson.entries:
                if (
                    entry.key in self.poisson._event_slots
                    and self.poisson._runtime_counts[entry.key]
                ):
                    values = packet.draws[entry.key]
                    rows = (
                        values.unsqueeze(0)
                        if self.poisson._capacities[entry.key] == 1
                        else values
                    )
                    self.poisson._check_event_slots(
                        entry, rows[: self.poisson._runtime_counts[entry.key]]
                    )
            for entry in self.poisson._runtime_entries:
                position = positions.get(entry.key, 0)
                values = packet.draws[entry.key]
                value = (
                    values
                    if self.poisson._capacities[entry.key] == 1
                    else values[position]
                )
                invocations.append(
                    (
                        entry.source,
                        native_poisson_schedule_from_draws(
                            entry.source, value, dtype=entry.dtype, device=entry.device
                        ),
                    )
                )
                positions[entry.key] = position + 1
            self.bank_replay = _NativePoissonInvocationReplay(invocations)
            seen = set()
            for entry in self.poisson.entries:
                if id(entry.source) not in seen:
                    self.poisson_slots.append(
                        (entry.source, "_native_poisson_replay", self.bank_replay)
                    )
                    seen.add(id(entry.source))

    def bind_rng(self, model):
        _keys(model, self.reference, "draws.rng.model")
        for stream, reference in self.reference.items():
            actual = model[stream]
            _keys(actual, reference, f"draws.rng.model.{stream}")
            frame = {}
            for leaf, value in reference.items():
                _keys(actual[leaf], ("step",), "draws.rng.model boundaries")
                for name, tensor in actual[leaf].items():
                    _tensor(
                        tensor, value, None, f"draws.rng.model.{stream}.{leaf}.{name}"
                    )
                frame[leaf] = actual[leaf]["step"]
            self.updates.append(_prepare_rng(self.sources[stream], frame))

    def bind_stimulation_rng(self, values):
        streams = {} if self.poisson is None else self.poisson.streams
        _keys(values, set(streams) - set(self.borrowed), "draws.rng.stimulation")
        if self.poisson is None:
            return
        complete = dict(values)
        for stream, (owner, device) in self.borrowed.items():
            payload = next(
                payload
                for source, payload in self.updates
                if source is self.sources[owner]
            )
            complete[stream] = payload["rng_state"][device]
        self.poisson._validate_rng(complete)
        self.poisson_updates = [
            (generator, values[stream].detach().cpu().clone())
            for stream, generator in streams.items()
            if stream not in self.borrowed
        ]

    def scope(self):
        return _bound_replay_slots(self.slots)

    @contextmanager
    def _bank_scope(self):
        with _bound_replay_slots(self.poisson_slots):
            yield
            self.bank_replay.finish()

    def bank_scope(self):
        return nullcontext() if self.bank_replay is None else self._bank_scope()

    def publish(self):
        _publish(self.updates)
        for generator, state in self.poisson_updates:
            generator.set_state(state)
        if self.poisson is not None:
            for entry in self.poisson.entries:
                pending = vars(entry.source).get("_functional_poisson_rng")
                if pending is not None and pending.generator is entry.source.generator:
                    pending.state = None


def bind_population_runtime_draws(population, packet, *, extra=None):
    packet = _normalize_runtime_draws(packet)
    _keys(packet.rng, ("model", "stimulation"), "draws.rng")
    replay = _PopulationReplay(population, packet.draws, extra, bank_poisson=True)
    replay.bind_rng(packet.rng["model"])
    replay.bind_stimulation_rng(packet.rng["stimulation"])
    return replay


class _NetworkReplay:
    def __init__(self, network, packet, extra):
        from dendra.models.networks.net import _network_step_schedule

        _keys(packet.draws, ("populations", "netstim"), "draws.draws")
        _keys(
            packet.draws["populations"], network.populations, "draws.draws.populations"
        )
        _keys(packet.rng, ("model", "stimulation", "netstim"), "draws.rng")
        _keys(packet.rng["stimulation"], (), "draws.rng.stimulation")
        populations = dict(_network_step_schedule(network).populations)
        self.populations = {
            name: _PopulationReplay(
                population,
                packet.draws["populations"][name],
                None if extra is None else extra.get(name),
            )
            for name, population in populations.items()
        }
        local, streams, sources, _identities, _consumers = _network_stream_inventory(
            populations,
            {name: replay.noise.streams for name, replay in self.populations.items()},
        )
        retained = {
            local[name][stream]
            for name, replay in self.populations.items()
            for stream in replay.reference
        }
        _keys(packet.rng["model"], retained, "draws.rng.model")
        for stream, source in sources.items():
            if stream not in retained:
                continue
            source_population = populations[streams[stream][0]]
            actual = packet.rng["model"][stream]
            reference = _rng_values(source, source_population.device())
            _keys(actual, reference, f"draws.rng.model.{stream}")
            consumers = tuple(
                name for name, bindings in local.items() if stream in bindings.values()
            )
            for leaf, value in reference.items():
                _keys(actual[leaf], consumers, "draws.rng.model boundaries")
                for name, tensor in actual[leaf].items():
                    _tensor(
                        tensor, value, None, f"draws.rng.model.{stream}.{leaf}.{name}"
                    )
            for name in consumers:
                frame = {leaf: actual[leaf][name] for leaf in reference}
                self.populations[name].updates.append(
                    _prepare_rng(sources[stream], frame)
                )
        self.netstim = network.netstim
        self.netstim_draw = None
        self.netstim_rng = None
        if self.netstim is None:
            if packet.draws["netstim"] is not None or packet.rng["netstim"] is not None:
                raise ValueError("A Network without NetStim requires no NetStim draws")
        else:
            value = packet.draws["netstim"]
            reference = self.netstim.next_stoch_time.new_empty(self.netstim.shape)
            _tensor(value, reference, None, "draws.draws.netstim")
            rng = packet.rng["netstim"]
            _tensor(rng, self.netstim._rng.get_state(), None, "draws.rng.netstim")
            owned = rng.detach().cpu().clone()
            torch.Generator(device=self.netstim._rng.device).set_state(owned)
            self.netstim_draw, self.netstim_rng = value.detach().clone(), owned

    def publish_netstim(self):
        self.netstim._rng.set_state(self.netstim_rng)


def bind_network_runtime_draws(network, packet, *, extra=None):
    return _NetworkReplay(network, _normalize_runtime_draws(packet), extra)
