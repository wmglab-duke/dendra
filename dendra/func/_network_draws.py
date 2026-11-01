"""Canonical Network runtime values and physical RNG publication timelines."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils import _pytree

from ._draws import RuntimeDraws, _normalize_runtime_draws
from ._rng_diagnostics import _rng_diagnostic_scope
from ._rng_frames import _rng_values
from ._types import FunctionalizationError


def _owned_runtime_sampling(functional):
    """Use the canonical program only when old public samplers are unchanged."""
    from ._network import (
        _OWNED_NETWORK_NETSTIM_TAPE,
        _OWNED_NETWORK_NOISE_TAPES,
        _OWNED_NETWORK_STIMULATION_TAPES,
    )
    from ._population import (
        _OWNED_POPULATION_NOISE_TAPE,
        _OWNED_POPULATION_STIMULATION_TAPE,
    )

    methods = (
        (functional.sample_draw_tape, functional, _OWNED_NETWORK_NETSTIM_TAPE),
        (
            functional.sample_population_noise_tapes,
            functional,
            _OWNED_NETWORK_NOISE_TAPES,
        ),
        (
            functional.sample_population_stimulation_tapes,
            functional,
            _OWNED_NETWORK_STIMULATION_TAPES,
        ),
        *(
            (method, plan, original)
            for plan in functional.populations.values()
            for method, original in (
                (plan.sample_noise_draw_tape, _OWNED_POPULATION_NOISE_TAPE),
                (plan.sample_stimulation_draw_tape, _OWNED_POPULATION_STIMULATION_TAPE),
            )
        ),
    )
    if any(
        getattr(method, "__self__", None) is not owner
        or getattr(method, "__func__", None) is not original
        for method, owner, original in methods
    ):
        return False
    if functional.netstim is not None:
        from . import _network_rollout
        from ._netstim import FunctionalNetStim

        method = functional.netstim.sample_step_draw
        if (
            getattr(method, "__self__", None) is not functional.netstim
            or getattr(method, "__func__", None)
            is not FunctionalNetStim.sample_step_draw
            or _network_rollout.sample_draw_tape
            is not _network_rollout._OWNED_NETSTIM_DRAW_TAPE
        ):
            return False
    return True


def _keys(value, expected, label):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{label} must match the canonical runtime draw schema")


def _tensor(value, reference, steps, label):
    shape = tuple(reference.shape)
    if steps is not None:
        shape = (steps, *shape)
    if (
        not torch.is_tensor(value)
        or tuple(value.shape) != shape
        or value.dtype != reference.dtype
        or value.device != reference.device
    ):
        raise ValueError(f"{label} has an incompatible Tensor schema")


def _with_network_runtime_draws(inputs, draws):
    """Normalize canonical ingress without changing the public input tuple."""
    if draws is None:
        return inputs, None
    draws = _normalize_runtime_draws(draws)
    populations, netstim = _network_inputs(inputs)
    if netstim is not None:
        raise ValueError("draws cannot be combined with legacy NetStim packets")
    if populations is not None:
        if not isinstance(populations, Mapping):
            raise ValueError(
                "populations must map known Network names to RolloutInput tapes"
            )
        for selected in populations.values():
            if (
                getattr(selected, "noise", None) is not None
                or getattr(selected, "stimulation", None) is not None
                or getattr(selected, "draws", None) is not None
            ):
                raise ValueError(
                    "draws cannot be combined with legacy noise or stimulation packets"
                )
    return inputs, draws


def _network_inputs(inputs):
    """Read deterministic name bindings or the unchanged legacy envelope."""
    from ._network_rollout import NetworkRolloutInput

    if inputs is None:
        return {}, None
    if isinstance(inputs, Mapping):
        return inputs, None
    if not isinstance(inputs, NetworkRolloutInput):
        raise TypeError("inputs must be a NetworkRolloutInput or None")
    return ({} if inputs.populations is None else inputs.populations), inputs.netstim


def _legacy_population_noise(inputs):
    """Project the local authority only for legacy RNG validation/sampling."""
    draws = getattr(inputs, "draws", None)
    return getattr(inputs, "noise", None) if draws is None else draws


def _network_model_boundaries(local_streams, references, borrowed_devices):
    """Retain ordinary consumer boundaries and one suffix per borrowed device."""
    result = {}
    for canonical, reference in references.items():
        consumers = tuple(
            population
            for population, streams in local_streams.items()
            if canonical in streams.values()
        )
        result[canonical] = {
            leaf: ("step",) if (canonical, leaf) in borrowed_devices else consumers
            for leaf in reference
        }
    return result


class NetworkRuntimeDrawPlan:
    """Route every alias to one authoritative physical-stream boundary.

    Ordinary model streams expose their historical Population boundaries.
    Devices borrowed by Poisson consumers expose a common complete-timestep
    suffix. Draw families contain values only; no family owns RNG bytes.
    """

    def __init__(self, functional):
        self.functional = functional
        self.random = functional._rng_plan
        self.stimulation = functional._stimulation_plan
        self.sharing = functional._stimulation_rng_plan
        self.borrowed_devices = frozenset(self.sharing.borrowed.values())
        retained = {
            self.random.local_streams[name][local]
            for name, plan in functional.populations.items()
            for local in plan._runtime_draw_plan.model_reference
        } | {owner for owner, _device in self.borrowed_devices}
        self.model_reference = {}
        for canonical, (name, _path, _attribute) in self.random.streams.items():
            if canonical not in retained:
                continue
            local = next(
                local
                for local, owner in self.random.local_streams[name].items()
                if owner == canonical
            )
            reference = functional.populations[name]._random_plan.rng_reference[local]
            self.model_reference[canonical] = reference
        self.boundaries = _network_model_boundaries(
            self.random.local_streams, self.model_reference, self.borrowed_devices
        )
        self.stimulation_reference = {
            stream: source.get_state().detach().clone()
            for stream, source in self.stimulation.streams.items()
            if stream not in self.sharing.borrowed
        }
        self.netstim_reference = (
            None
            if functional.netstim is None
            else {
                "draw": functional.netstim._source.next_stoch_time.new_empty(
                    functional.netstim.shape
                ),
                "rng": functional.netstim._source._rng.get_state().detach().clone(),
            }
        )

    def _model_pool(self, state):
        if self.random.enabled:
            return state["rng"]
        pool = {}
        for name, streams in self.random.local_streams.items():
            for local, canonical in streams.items():
                pool.setdefault(
                    canonical, state["populations"][name]["random"]["rng"][local]
                )
        return pool

    def _stimulation_pool(self, state, model):
        return self.sharing.expand_pool(model, state.get("stimulation_rng", {}))

    def _empty(self, steps=0):
        populations = {}
        for name, plan in self.functional.populations.items():
            noise, poisson = plan._noise_plan, plan._poisson_plan
            populations[name] = {
                "noise": None
                if noise is None or not noise.runtime
                else noise._empty_tape_draws(),
                "stimulation": None
                if poisson is None
                else {
                    entry.key: torch.empty(
                        (steps, *poisson._draw_shape(entry)),
                        device=poisson.streams[entry.stream].device,
                        dtype=poisson.draw_dtype,
                    )
                    for entry in poisson.entries
                },
            }
        return RuntimeDraws(
            {
                "populations": populations,
                "netstim": None
                if self.netstim_reference is None
                else self.netstim_reference["draw"].new_empty(
                    (steps, *self.netstim_reference["draw"].shape)
                ),
            },
            {
                "model": {
                    canonical: {
                        leaf: {
                            boundary: reference.new_empty((steps, *reference.shape))
                            for boundary in self.boundaries[canonical][leaf]
                        }
                        for leaf, reference in frame.items()
                    }
                    for canonical, frame in self.model_reference.items()
                },
                "stimulation": {
                    stream: reference.new_empty((steps, *reference.shape))
                    for stream, reference in self.stimulation_reference.items()
                },
                "netstim": None
                if self.netstim_reference is None
                else self.netstim_reference["rng"].new_empty(
                    (steps, *self.netstim_reference["rng"].shape)
                ),
            },
        )

    def sample(self, state, *, steps=None):
        """Execute the existing native request program with one frame authority."""
        if steps is not None and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
        ):
            raise ValueError("steps must be a non-negative integer")
        if steps == 0:
            return self._empty()
        model = self._model_pool(state)
        samplers = self.random._samplers(
            {owner: model[owner] for owner in self.model_reference}
        )
        pool = self._stimulation_pool(state, model)
        poisson_samplers = self.sharing._poisson_samplers(samplers, pool)
        netstim_state = state["netstim"]
        packets = []
        for _index in range(1 if steps is None else steps):
            contexts = {}
            for name, plan in self.functional.populations.items():
                local = self.sharing.expand_population(
                    name, state["populations"][name], model, pool
                )
                contexts[name] = plan._stimulation_rng_plan.begin_packets(
                    local,
                    samplers={
                        local: samplers[self.random.local_streams[name][local]]
                        for local in plan._runtime_draw_plan.model_reference
                    },
                    poisson_samplers={
                        local: poisson_samplers[canonical]
                        for local, canonical in self.stimulation.local.get(
                            name, {}
                        ).items()
                    },
                )
            model_frames = {
                canonical: {leaf: {} for leaf in reference}
                for canonical, reference in self.model_reference.items()
            }
            netstim = (
                None
                if self.functional.netstim is None
                else self.functional.netstim.sample_step_draw(netstim_state)
            )
            if netstim is not None:
                netstim_state = {**netstim_state, "rng_state": netstim.rng_state}
            for phase in ("extra", "intra", "mechanism"):
                names = (
                    self.stimulation._ordered_names(phase)
                    if phase != "mechanism"
                    else self.functional.populations
                )
                for name in names:
                    plan = self.functional.populations[name]
                    adapter, context = plan._stimulation_rng_plan, contexts[name]
                    ordinary = (
                        phase == "mechanism"
                        and not self.sharing.enabled
                        and plan._noise_plan is not None
                        and bool(plan._noise_plan.runtime)
                    )
                    if ordinary:
                        context["diagnostic_enabled"] = False
                    with _rng_diagnostic_scope(
                        plan.device, enabled=ordinary
                    ) as diagnostic:
                        for kind, entry in adapter.runtime_requests(phase):
                            adapter.sample_request(kind, entry, context)
                    if diagnostic.changes:
                        raise FunctionalizationError(
                            "NOISE draw sampler consumed global RNG; use explicit RNGModule"
                        )
                    if phase == "mechanism":
                        for local, canonical in self.random.local_streams[name].items():
                            if canonical not in self.model_reference:
                                continue
                            frame = _rng_values(samplers[canonical], plan.device)
                            for leaf, value in frame.items():
                                if (canonical, leaf) not in self.borrowed_devices:
                                    model_frames[canonical][leaf][name] = value
            for canonical, leaf in self.borrowed_devices:
                reference = self.model_reference[canonical][leaf]
                model_frames[canonical][leaf]["step"] = _rng_values(
                    samplers[canonical], reference.device
                )[leaf]
            packet = RuntimeDraws(
                {
                    "populations": {
                        name: {
                            "noise": None
                            if plan._noise_plan is None or not plan._noise_plan.runtime
                            else contexts[name]["noise_draws"],
                            "stimulation": None
                            if plan._poisson_plan is None
                            else contexts[name]["stimulation_draws"],
                        }
                        for name, plan in self.functional.populations.items()
                    },
                    "netstim": None if netstim is None else netstim.exp_rand,
                },
                {
                    "model": model_frames,
                    "stimulation": {
                        stream: poisson_samplers[stream].get_state().clone()
                        for stream in self.stimulation_reference
                    },
                    "netstim": None if netstim is None else netstim.rng_state,
                },
            )
            packets.append(packet)
        if steps is None:
            return packets[0]
        return _pytree.tree_map(
            lambda *values: None if values[0] is None else torch.stack(values),
            *packets,
        )

    def local(self, name, packet):
        """Expose one Population's static routes at numerical consumption."""
        plan = self.functional.populations[name]
        model = {
            local: {
                leaf: {
                    "step": packet.rng["model"][canonical][leaf][
                        "step" if (canonical, leaf) in self.borrowed_devices else name
                    ]
                }
                for leaf in reference
            }
            for local, reference in plan._runtime_draw_plan.model_reference.items()
            for canonical in (self.random.local_streams[name][local],)
        }
        stimulation = {}
        for local, canonical in self.stimulation.local.get(name, {}).items():
            if local in plan._stimulation_rng_plan.borrowed:
                continue
            if canonical in self.sharing.borrowed:
                owner, device = self.sharing.borrowed[canonical]
                value = packet.rng["model"][owner][device]["step"].to("cpu")
            else:
                value = packet.rng["stimulation"][canonical]
            stimulation[local] = value
        return RuntimeDraws(
            packet.draws["populations"][name],
            {"model": model, "stimulation": stimulation},
        )

    def validate(self, packet, *, steps=None):
        if not isinstance(packet, RuntimeDraws):
            raise TypeError("draws must be a RuntimeDraws packet")
        if steps is not None and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
        ):
            raise ValueError("steps must be a non-negative integer")
        _keys(packet.draws, ("populations", "netstim"), "draws.draws")
        _keys(
            packet.draws["populations"],
            self.functional.populations,
            "draws.draws.populations",
        )
        _keys(packet.rng, ("model", "stimulation", "netstim"), "draws.rng")
        _keys(packet.rng["model"], self.model_reference, "draws.rng.model")
        for canonical, frame in self.model_reference.items():
            actual = packet.rng["model"][canonical]
            _keys(actual, frame, f"draws.rng.model.{canonical}")
            for leaf, reference in frame.items():
                _keys(
                    actual[leaf],
                    self.boundaries[canonical][leaf],
                    "draws.rng.model boundaries",
                )
                for boundary, value in actual[leaf].items():
                    _tensor(
                        value,
                        reference,
                        steps,
                        f"draws.rng.model.{canonical}.{leaf}.{boundary}",
                    )
        _keys(
            packet.rng["stimulation"],
            self.stimulation_reference,
            "draws.rng.stimulation",
        )
        for stream, reference in self.stimulation_reference.items():
            _tensor(
                packet.rng["stimulation"][stream],
                reference,
                steps,
                f"draws.rng.stimulation.{stream}",
            )
        for name, plan in self.functional.populations.items():
            plan._runtime_draw_plan.validate(self.local(name, packet), steps=steps)
        if self.netstim_reference is None:
            if packet.draws["netstim"] is not None or packet.rng["netstim"] is not None:
                raise ValueError("A Network without NetStim requires no NetStim draws")
        else:
            _tensor(
                packet.draws["netstim"],
                self.netstim_reference["draw"],
                steps,
                "draws.draws.netstim",
            )
            _tensor(
                packet.rng["netstim"],
                self.netstim_reference["rng"],
                steps,
                "draws.rng.netstim",
            )

    def publish_population(self, name, state, random, stimulation, packet):
        """Publish routed local frames without reconciling copied family suffixes."""
        plan = self.functional.populations[name]._runtime_draw_plan
        model_updates = {}
        if self.random.enabled:
            for local, canonical in self.random.local_streams[name].items():
                if canonical not in self.model_reference:
                    continue
                model_updates[canonical] = (
                    state["random"]["rng"][local]
                    if local in plan.model_reference
                    else {
                        leaf: packet.rng["model"][canonical][leaf][
                            "step"
                            if (canonical, leaf) in self.borrowed_devices
                            else name
                        ].clone()
                        for leaf in self.model_reference[canonical]
                    }
                )
            random = {
                **random,
                **model_updates,
            }
        updates = {}
        for local, canonical in self.stimulation.local.get(name, {}).items():
            if canonical in self.sharing.borrowed:
                owner, device = self.sharing.borrowed[canonical]
                if owner in model_updates:
                    frame = random[owner][device]
                else:
                    frame = packet.rng["model"][owner][device]["step"].clone()
                    random = {**random, owner: {**random[owner], device: frame}}
                updates[canonical] = frame.to("cpu")
            else:
                updates[canonical] = state["stimulation"]["rng"][local]
        return random, {**stimulation, **updates}
