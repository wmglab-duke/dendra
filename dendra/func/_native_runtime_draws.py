"""Native frame/tape bindings for NOISE/SDE, NetStim and bank Poisson draws."""

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

from ._draws import _normalize_runtime_draws, _runtime_draws_binding
from ._inputs import _index_runtime_inputs
from ._network_draws import _keys, _network_model_boundaries, _tensor
from ._network_rng import (
    _network_borrowed_stimulation_streams,
    _network_stream_inventory,
)
from ._poisson_stimulation import (
    PoissonStimulationPlan,
    _network_stimulation_stream_inventory,
    _validate_generator_domains,
)
from ._population import _parameterized_owner_paths
from ._rng_frames import _rng_extra, _rng_values
from ._runtime_noise import RuntimeNoisePlan
from ._stimulation_rng import _generator_bindings, _validate_tracked_generators
from ._types import FunctionalizationError, RuntimeNoiseDraw, StimulationDraw


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


def _bank_poisson_invocations(plan, draws):
    """Validate a Population's recipes and own each ordered native schedule."""
    packet = StimulationDraw(draws, {})
    plan.validate(packet, validate_rng=False)
    for entry in plan.entries:
        if entry.key in plan._event_slots and plan._runtime_counts[entry.key]:
            values = packet.draws[entry.key]
            rows = values.unsqueeze(0) if plan._capacities[entry.key] == 1 else values
            plan._check_event_slots(entry, rows[: plan._runtime_counts[entry.key]])
    positions, invocations = {}, []
    for entry in plan._runtime_entries:
        position = positions.get(entry.key, 0)
        values = packet.draws[entry.key]
        value = values if plan._capacities[entry.key] == 1 else values[position]
        invocations.append(
            (
                entry,
                native_poisson_schedule_from_draws(
                    entry.source, value, dtype=entry.dtype, device=entry.device
                ),
            )
        )
        positions[entry.key] = position + 1
    return tuple(invocations)


def _poisson_replay_slots(plans, replay):
    seen = set()
    for plan in plans:
        for entry in plan.entries:
            if id(entry.source) not in seen:
                yield entry.source, "_native_poisson_replay", replay
                seen.add(id(entry.source))


def _clear_pending_poisson_rng(plans):
    for plan in plans:
        for entry in plan.entries:
            pending = vars(entry.source).get("_functional_poisson_rng")
            if pending is not None and pending.generator is entry.source.generator:
                pending.state = None


def _validate_model_stream(actual, reference, boundaries, stream, *, steps=None):
    _keys(actual, reference, f"draws.rng.model.{stream}")
    for leaf, value in reference.items():
        _keys(actual[leaf], boundaries[leaf], "draws.rng.model boundaries")
        for name, tensor in actual[leaf].items():
            _tensor(tensor, value, steps, f"draws.rng.model.{stream}.{leaf}.{name}")


def _validate_model_rng(model, references, boundaries, *, steps=None):
    _keys(model, references, "draws.rng.model")
    for stream, reference in references.items():
        _validate_model_stream(
            model[stream], reference, boundaries[stream], stream, steps=steps
        )


class _PopulationReplayPlan:
    """Cold native source inventory shared by all frames of one invocation."""

    def __init__(self, population, extra, *, validate_rng_ownership=True):
        self.poisson = _bank_poisson_plan(population, extra)
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
        if self.poisson is not None and validate_rng_ownership:
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
        self.boundaries = {
            stream: {leaf: ("step",) for leaf in frame}
            for stream, frame in self.reference.items()
        }

    def validate_source(self):
        from ._population import _module_at

        self.noise.validate_source()
        for stream, (path, attribute) in self.noise.streams.items():
            owner = (
                _module_at(self.noise.population, path)
                if path
                else self.noise.population
            )
            if (
                owner is not self.owners[path]
                or getattr(owner, attribute) is not self.sources[stream]
            ):
                raise FunctionalizationError(
                    "Native runtime draw source bindings changed after validation"
                )
        if self.poisson is not None:
            self.poisson.validate_source()
            generators = _generator_bindings(self.sources)
            borrowed = {
                stream: generators[id(generator)]
                for stream, generator in self.poisson.streams.items()
                if id(generator) in generators
            }
            if borrowed != self.borrowed:
                raise FunctionalizationError(
                    "Native runtime draw generator sharing changed after validation"
                )

    def validate_draws(self, draws, *, steps=None):
        _keys(draws, ("noise", "stimulation"), "draws.draws")
        if self.poisson is None and draws["stimulation"] is not None:
            raise ValueError(
                "Native draws do not support randomized Poisson stimulation"
            )
        selected = draws["noise"]
        if self.noise.runtime:
            if not isinstance(selected, Mapping):
                raise ValueError("Native NOISE/SDE draws must be a mapping")
            self.noise.validate(RuntimeNoiseDraw(selected, {}), steps=steps)
        elif selected is not None:
            raise ValueError(
                "This Population has no recurrent NOISE or State SDE draws"
            )
        if self.poisson is not None and steps is not None:
            self.poisson.validate(
                StimulationDraw(draws["stimulation"], {}),
                steps=steps,
                validate_rng=False,
            )

    def validate_tape(self, packet, steps):
        self.validate_draws(packet.draws, steps=steps)
        _keys(packet.rng, ("model", "stimulation"), "draws.rng")
        _validate_model_rng(
            packet.rng["model"], self.reference, self.boundaries, steps=steps
        )
        streams = {} if self.poisson is None else self.poisson.streams
        _keys(
            packet.rng["stimulation"],
            set(streams) - set(self.borrowed),
            "draws.rng.stimulation",
        )
        if self.poisson is not None:
            complete = dict(packet.rng["stimulation"])
            for stream, (owner, device) in self.borrowed.items():
                complete[stream] = (
                    packet.rng["model"][owner][device]["step"].detach().to("cpu")
                )
            self.poisson._validate_rng(complete, steps=steps)


class _PopulationReplay:
    def __init__(
        self, population, draws, extra, *, validate_rng_ownership=True, _plan=None
    ):
        self.plan = (
            _PopulationReplayPlan(
                population, extra, validate_rng_ownership=validate_rng_ownership
            )
            if _plan is None
            else _plan
        )
        self.poisson = self.plan.poisson
        self.noise = self.plan.noise
        self.owners = self.plan.owners
        self.sources = self.plan.sources
        self.borrowed = self.plan.borrowed
        self.reference = self.plan.reference
        self.plan.validate_draws(draws)
        selected = draws["noise"]
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
        self.poisson_updates = None
        self.poisson_slots = []
        self.poisson_invocations = ()
        self.bank_replay = None
        if self.poisson is not None:
            self.poisson_invocations = _bank_poisson_invocations(
                self.poisson, draws["stimulation"]
            )
            self.bank_replay = _NativePoissonInvocationReplay(
                (entry.source, schedule) for entry, schedule in self.poisson_invocations
            )
            self.poisson_slots = tuple(
                _poisson_replay_slots((self.poisson,), self.bank_replay)
            )

    def bind_rng(self, model):
        _keys(model, self.reference, "draws.rng.model")
        for stream, reference in self.reference.items():
            actual = model[stream]
            _validate_model_stream(
                actual, reference, self.plan.boundaries[stream], stream
            )
            frame = {leaf: actual[leaf]["step"] for leaf in reference}
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
        if self.poisson_updates is not None:
            for generator, state in self.poisson_updates:
                generator.set_state(state)
            _clear_pending_poisson_rng((self.poisson,))


def bind_population_runtime_draws(population, packet, *, extra=None):
    packet = _normalize_runtime_draws(packet)
    _keys(packet.rng, ("model", "stimulation"), "draws.rng")
    replay = _PopulationReplay(population, packet.draws, extra)
    replay.bind_rng(packet.rng["model"])
    replay.bind_stimulation_rng(packet.rng["stimulation"])
    return replay


class _NetworkReplayPlan:
    """Canonical stream routes and source plans for one native Network call."""

    def __init__(self, network, extra, *, _population_models=None, _populations=None):
        from dendra.models.networks.net import _network_step_schedule

        self.network = network
        populations = self.population_models = (
            dict(_network_step_schedule(network).populations)
            if _population_models is None
            else _population_models
        )
        self.populations = (
            {
                name: _PopulationReplayPlan(
                    population,
                    None if extra is None else extra.get(name),
                    validate_rng_ownership=False,
                )
                for name, population in populations.items()
            }
            if _populations is None
            else _populations
        )
        self.local, self.streams, self.sources, _identities, _consumers = (
            _network_stream_inventory(
                populations,
                {
                    name: replay.noise.streams
                    for name, replay in self.populations.items()
                },
            )
        )
        self.poisson_plans = {
            name: replay.poisson
            for name, replay in self.populations.items()
            if replay.poisson is not None
        }
        self.poisson_streams, self.poisson_local = (
            _network_stimulation_stream_inventory(self.poisson_plans)
        )
        _validate_generator_domains(network, self.poisson_streams.values())
        self.borrowed = _network_borrowed_stimulation_streams(
            network, self.sources, self.poisson_plans, self.poisson_local
        )
        self.borrowed_devices = frozenset(self.borrowed.values())
        self.retained = {
            self.local[name][stream]
            for name, replay in self.populations.items()
            for stream in replay.reference
        } | {owner for owner, _device in self.borrowed_devices}
        self.references = {
            stream: _rng_values(source, populations[self.streams[stream][0]].device())
            for stream, source in self.sources.items()
            if stream in self.retained
        }
        self.boundaries = _network_model_boundaries(
            self.local, self.references, self.borrowed_devices
        )
        self.netstim = network.netstim
        self.netstim_rng = None if self.netstim is None else self.netstim._rng

    def validate_source(self):
        from dendra.models.networks.net import _network_step_schedule

        if tuple(_network_step_schedule(self.network).populations) != tuple(
            self.population_models.items()
        ):
            raise FunctionalizationError(
                "Native runtime draw Population bindings changed after validation"
            )
        if self.network.netstim is not self.netstim or (
            self.netstim is not None and self.netstim._rng is not self.netstim_rng
        ):
            raise FunctionalizationError(
                "Native runtime draw NetStim bindings changed after validation"
            )
        for plan in self.populations.values():
            plan.validate_source()
        _validate_generator_domains(self.network, self.poisson_streams.values())
        if (
            _network_borrowed_stimulation_streams(
                self.network, self.sources, self.poisson_plans, self.poisson_local
            )
            != self.borrowed
        ):
            raise FunctionalizationError(
                "Native runtime draw generator sharing changed after validation"
            )

    def validate_tape(self, packet, steps):
        _keys(packet.draws, ("populations", "netstim"), "draws.draws")
        _keys(packet.draws["populations"], self.populations, "draws.draws.populations")
        _keys(packet.rng, ("model", "stimulation", "netstim"), "draws.rng")
        for name, plan in self.populations.items():
            plan.validate_draws(packet.draws["populations"][name], steps=steps)
        _validate_model_rng(
            packet.rng["model"], self.references, self.boundaries, steps=steps
        )
        independent = set(self.poisson_streams) - set(self.borrowed)
        _keys(packet.rng["stimulation"], independent, "draws.rng.stimulation")
        for name, plan in self.poisson_plans.items():
            values = {}
            for stream, canonical in self.poisson_local[name].items():
                if canonical in self.borrowed:
                    owner, device = self.borrowed[canonical]
                    values[stream] = (
                        packet.rng["model"][owner][device]["step"].detach().to("cpu")
                    )
                else:
                    values[stream] = packet.rng["stimulation"][canonical]
            plan._validate_rng(values, steps=steps)
        if self.netstim is None:
            if packet.draws["netstim"] is not None or packet.rng["netstim"] is not None:
                raise ValueError("A Network without NetStim requires no NetStim draws")
        else:
            _tensor(
                packet.draws["netstim"],
                self.netstim.next_stoch_time.new_empty(self.netstim.shape),
                steps,
                "draws.draws.netstim",
            )
            _tensor(
                packet.rng["netstim"],
                self.netstim._rng.get_state(),
                steps,
                "draws.rng.netstim",
            )


class _NetworkReplay:
    def __init__(self, network, packet, extra, *, _plan=None):
        from dendra.models.networks.net import _network_step_schedule

        _keys(packet.draws, ("populations", "netstim"), "draws.draws")
        _keys(
            packet.draws["populations"], network.populations, "draws.draws.populations"
        )
        _keys(packet.rng, ("model", "stimulation", "netstim"), "draws.rng")
        population_models = (
            dict(_network_step_schedule(network).populations)
            if _plan is None
            else _plan.population_models
        )
        self.populations = {
            name: _PopulationReplay(
                population,
                packet.draws["populations"][name],
                None if extra is None else extra.get(name),
                validate_rng_ownership=False,
                _plan=None if _plan is None else _plan.populations[name],
            )
            for name, population in population_models.items()
        }
        self.plan = (
            _NetworkReplayPlan(
                network,
                extra,
                _population_models=population_models,
                _populations={
                    name: replay.plan for name, replay in self.populations.items()
                },
            )
            if _plan is None
            else _plan
        )
        self.poisson_plans = self.plan.poisson_plans
        local, sources = self.plan.local, self.plan.sources
        references, retained = self.plan.references, self.plan.retained
        borrowed, borrowed_devices = self.plan.borrowed, self.plan.borrowed_devices
        poisson_streams, poisson_local = (
            self.plan.poisson_streams,
            self.plan.poisson_local,
        )
        boundaries = self.plan.boundaries
        _keys(packet.rng["model"], retained, "draws.rng.model")
        model_payloads = {}
        for stream, source in sources.items():
            if stream not in retained:
                continue
            actual = packet.rng["model"][stream]
            reference = references[stream]
            _validate_model_stream(actual, reference, boundaries[stream], stream)
            consumers = tuple(
                name for name, bindings in local.items() if stream in bindings.values()
            )
            for name in consumers:
                frame = {
                    leaf: actual[leaf][
                        "step" if (stream, leaf) in borrowed_devices else name
                    ]
                    for leaf in reference
                }
                update = _prepare_rng(source, frame)
                self.populations[name].updates.append(update)
                model_payloads[stream] = update[1]
        independent = set(poisson_streams) - set(borrowed)
        _keys(packet.rng["stimulation"], independent, "draws.rng.stimulation")
        # Validate each canonical frame through the existing Population recipe,
        # reconstructing only adapter-local aliases for borrowed generators.
        for name, plan in self.poisson_plans.items():
            values = {}
            for stream, canonical in poisson_local[name].items():
                if canonical in borrowed:
                    owner, device = borrowed[canonical]
                    values[stream] = model_payloads[owner]["rng_state"][device]
                else:
                    values[stream] = packet.rng["stimulation"][canonical]
            plan._validate_rng(values)
        self.poisson_updates = [
            (generator, packet.rng["stimulation"][stream].detach().cpu().clone())
            for stream, generator in poisson_streams.items()
            if stream in independent
        ]
        invocations = []
        for phase, names in (
            ("extra", () if extra is None else extra),
            ("intra", network.populations),
        ):
            for name in names:
                for entry, schedule in self.populations[name].poisson_invocations:
                    if entry.kind == phase:
                        invocations.append((entry.source, schedule))
        self.bank_replay = (
            _NativePoissonInvocationReplay(invocations) if self.poisson_plans else None
        )
        self.poisson_slots = (
            ()
            if self.bank_replay is None
            else tuple(
                _poisson_replay_slots(self.poisson_plans.values(), self.bank_replay)
            )
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

    @contextmanager
    def bank_scope(self, *, finish=True):
        with _bound_replay_slots(self.poisson_slots):
            yield
            if finish and self.bank_replay is not None:
                self.bank_replay.finish()

    def publish_stimulation(self):
        for generator, state in self.poisson_updates:
            generator.set_state(state)
        _clear_pending_poisson_rng(self.poisson_plans.values())

    def publish_netstim(self):
        self.netstim._rng.set_state(self.netstim_rng)


def bind_network_runtime_draws(network, packet, *, extra=None):
    return _NetworkReplay(network, _normalize_runtime_draws(packet), extra)


class _PreserveReplayStateDuringRecompute:
    """Restore native state and caller-owned Poisson lifecycle after backward."""

    def __init__(self, model, sources):
        from dendra.models._checkpoint import _checkpoint_recompute_contexts

        self.model_context = _checkpoint_recompute_contexts(model)[1]
        self.sources = sources
        self.previous = []

    def __enter__(self):
        schedules, bindings, holders, generators = [], [], {}, {}
        for source in self.sources:
            schedules.append((source, source._spike_times))
            namespace = vars(source)
            bindings.append(
                (
                    namespace,
                    "_functional_poisson_rng" in namespace,
                    namespace.get("_functional_poisson_rng"),
                )
            )
            holder = namespace.get("_functional_poisson_rng")
            while holder is not None and id(holder) not in holders:
                holders[id(holder)] = (holder, holder._state, holder._parent)
                holder = holder._parent
            generator = source.generator
            generators.setdefault(
                id(generator), (generator, generator.get_state().clone())
            )
        self.model_context.__enter__()
        self.previous.append((schedules, bindings, tuple(holders.values()), generators))
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        schedules, bindings, holders, generators = self.previous.pop()
        try:
            return self.model_context.__exit__(exc_type, exc_value, traceback)
        finally:
            for source, schedule in schedules:
                source._spike_times = schedule
            for namespace, present, holder in bindings:
                if present:
                    namespace["_functional_poisson_rng"] = holder
                else:
                    namespace.pop("_functional_poisson_rng", None)
            for holder, state, parent in holders:
                holder._state, holder._parent = state, parent
            for generator, state in generators.values():
                generator.set_state(state)


class _NativeRuntimeDrawTape:
    """Validated caller-owned tape and one cold source inventory per run.

    Preflight constructs and discards one frame at a time, so validation cannot
    leave a partial live run or retain horizon-sized copies of draws/schedules.
    Execution owns only its currently selected frame. Checkpoint recomputation
    reads the same tape and rejects mutation before restoring native state.
    """

    def __init__(self, model, packet, steps, extra, *, network, checkpointed=False):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        self.model, self.extra = model, extra
        self.packet = _normalize_runtime_draws(packet)
        self.steps, self.network = steps, network
        self.plan = (
            _NetworkReplayPlan(model, extra)
            if network
            else _PopulationReplayPlan(model, extra)
        )
        self.plan.validate_tape(self.packet, steps)
        self.binding = _runtime_draws_binding(self.packet)
        if (
            checkpointed
            and steps
            and model.training
            and not torch.is_inference_mode_enabled()
            and self.binding[-1]
        ):
            raise ValueError(
                "Checkpointed native draw tapes require tensors with mutation "
                "versions; clone inference tensors outside torch.inference_mode()."
            )
        for index in range(steps):
            self._frame(index)
        plans = (
            self.plan.poisson_plans.values()
            if network
            else (() if self.plan.poisson is None else (self.plan.poisson,))
        )
        self.sources = tuple(
            {
                id(entry.source): entry.source
                for plan in plans
                for entry in plan.entries
            }.values()
        )

    def validate_fresh(self):
        if _runtime_draws_binding(self.packet) != self.binding:
            raise ValueError("Native runtime draw tape changed after validation")
        self.plan.validate_source()

    def _frame(self, index):
        packet = _index_runtime_inputs(
            None, None, None, None, index, draws=self.packet
        ).draws
        if self.network:
            return _NetworkReplay(self.model, packet, self.extra, _plan=self.plan)
        replay = _PopulationReplay(
            self.model, packet.draws, self.extra, _plan=self.plan
        )
        replay.bind_rng(packet.rng["model"])
        replay.bind_stimulation_rng(packet.rng["stimulation"])
        return replay

    def frame(self, index):
        self.validate_fresh()
        if type(index) is not int or not 0 <= index < self.steps:
            raise IndexError("Native runtime draw frame index is outside the tape")
        return self._frame(index)

    def checkpoint_contexts(self):
        return nullcontext(), _PreserveReplayStateDuringRecompute(
            self.model, self.sources
        )


def bind_population_runtime_draw_tape(
    population, packet, *, steps, extra=None, checkpointed=False
):
    return _NativeRuntimeDrawTape(
        population, packet, steps, extra, network=False, checkpointed=checkpointed
    )


def bind_network_runtime_draw_tape(
    network, packet, *, steps, extra=None, checkpointed=False
):
    return _NativeRuntimeDrawTape(
        network, packet, steps, extra, network=True, checkpointed=checkpointed
    )
