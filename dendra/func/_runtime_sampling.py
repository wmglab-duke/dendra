"""Lowering-independent host execution of ordered runtime draw requests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import BuiltinFunctionType, GetSetDescriptorType

import torch
from torch.utils import _pytree

from dendra.models.random_parameters import sample_random_parameter_draw

from ._draws import RuntimeDraws
from ._rng_diagnostics import _rng_diagnostic_scope
from ._rng_frames import _rng_values, _sampler_from_frame
from ._state_projection import (
    _plain_projection_context,
    _plain_state_tree,
    _tensor_descriptor,
)
from ._types import FunctionalizationError

_STACK = torch.stack
_REQUIRES_GRAD = _tensor_descriptor("requires_grad")


def _stack_runtime_packets(packets):
    """Allocate detached host tapes independently of the caller's inference mode.

    Sampling requests retain their original context and ordering. Only the final
    owned stack uses ordinary tensors, matching explicit replay and allowing its
    short compiled tails to reuse the same input guards. Observable tensors and
    execution modes keep the original allocation path.
    """

    def stack():
        return _pytree.tree_map(
            lambda *values: None if values[0] is None else torch.stack(values),
            *packets,
        )

    if (
        torch.is_inference_mode_enabled()
        and not torch.is_grad_enabled()
        and type(_STACK) is BuiltinFunctionType
        and torch.stack is _STACK
        and _plain_projection_context()
        and type(_REQUIRES_GRAD) is GetSetDescriptorType
        and _tensor_descriptor("requires_grad") is _REQUIRES_GRAD
        and all(
            type(packet) is RuntimeDraws
            and _plain_state_tree(packet.draws)
            and _plain_state_tree(packet.rng)
            for packet in packets
        )
        and all(
            value is None or not value.requires_grad
            for value in _pytree.tree_leaves(packets)
        )
    ):
        with torch.inference_mode(False), torch.no_grad():
            return stack()
    return stack()


def sample_runtime_request(poisson, kind, entry, context, *, device, check_purity):
    """Consume one existing distribution or Poisson sampler at its boundary."""
    if kind == "poisson":
        draw = poisson._sample_entry(entry, context["poisson_samplers"][entry.stream])
        position = context["positions"][entry.key]
        poisson._assign_draw(
            context["stimulation_draws"],
            entry,
            position,
            draw,
            runtime=context.get("runtime_layout", False),
        )
        context["positions"][entry.key] += 1
        return
    with _rng_diagnostic_scope(
        device, enabled=context.get("diagnostic_enabled", check_purity)
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


@dataclass(frozen=True)
class PopulationSamplingSpec:
    """Static local routes into a runtime sampling program's owned streams."""

    noise: object
    poisson: object
    requests: tuple
    device: torch.device
    check_purity: bool
    model_routes: dict
    stimulation_routes: dict
    ordinary_noise: bool
    runtime_layout: bool = False

    @property
    def has_noise(self):
        return self.noise is not None and bool(self.noise.runtime)

    def requests_for(self, phase=None):
        if phase is None:
            return self.requests
        return tuple(
            (kind, entry)
            for kind, entry in self.requests
            if (entry.kind if kind == "poisson" else "mechanism") == phase
        )

    def context(self, model, stimulation):
        return {
            "random_samplers": {
                local: model[canonical]
                for local, canonical in self.model_routes.items()
            },
            "poisson_samplers": {
                local: stimulation[canonical]
                for local, canonical in self.stimulation_routes.items()
            },
            "noise_draws": {},
            "stimulation_draws": {}
            if self.poisson is None
            else self.poisson._placeholder_draws(runtime=self.runtime_layout),
            "positions": Counter(),
            "runtime_layout": self.runtime_layout,
        }

    def sample(self, context, phase=None):
        # Ordinary NOISE keeps its historical unconditional global-RNG audit;
        # mixed borrowed streams follow their captured optional purity policy.
        ordinary = self.ordinary_noise and phase in (None, "mechanism")
        if ordinary:
            context["diagnostic_enabled"] = False
        with _rng_diagnostic_scope(self.device, enabled=ordinary) as diagnostic:
            for kind, entry in self.requests_for(phase):
                sample_runtime_request(
                    self.poisson,
                    kind,
                    entry,
                    context,
                    device=self.device,
                    check_purity=self.check_purity,
                )
        if diagnostic.changes:
            raise FunctionalizationError(
                "NOISE draw sampler consumed global RNG; use explicit RNGModule"
            )


@dataclass(frozen=True)
class RuntimeSamplingProgram:
    """One request executor for native and functional Population/Network draws."""

    populations: dict
    model_reference: dict
    stimulation_sources: dict
    stimulation_reference: dict
    borrowed: dict
    boundaries: dict
    extra_names: tuple
    intra_names: tuple
    network: bool
    netstim_reference: object = None

    def empty(self):
        populations = {}
        for name, spec in self.populations.items():
            poisson = spec.poisson
            populations[name] = {
                "noise": spec.noise._empty_tape_draws() if spec.has_noise else None,
                "stimulation": None
                if poisson is None
                else {
                    entry.key: torch.empty(
                        (0, *poisson._draw_shape(entry, runtime=spec.runtime_layout)),
                        dtype=poisson.draw_dtype,
                        device=poisson.streams[entry.stream].device,
                    )
                    for entry in poisson.entries
                },
            }
        model = {
            canonical: {
                leaf: {
                    boundary: reference.new_empty((0, *reference.shape))
                    for boundary in self.boundaries[canonical][leaf]
                }
                for leaf, reference in frame.items()
            }
            for canonical, frame in self.model_reference.items()
        }
        stimulation = {
            stream: reference.new_empty((0, *reference.shape))
            for stream, reference in self.stimulation_reference.items()
        }
        return self._packet(
            populations,
            model,
            stimulation,
            None
            if self.netstim_reference is None
            else (
                self.netstim_reference["draw"].new_empty(
                    (0, *self.netstim_reference["draw"].shape)
                ),
                self.netstim_reference["rng"].new_empty(
                    (0, *self.netstim_reference["rng"].shape)
                ),
            ),
        )

    def _packet(self, populations, model, stimulation, netstim):
        if not self.network:
            return RuntimeDraws(
                next(iter(populations.values())),
                {"model": model, "stimulation": stimulation},
            )
        return RuntimeDraws(
            {
                "populations": populations,
                "netstim": None if netstim is None else netstim[0],
            },
            {
                "model": model,
                "stimulation": stimulation,
                "netstim": None if netstim is None else netstim[1],
            },
        )

    def sample(
        self,
        model,
        stimulation,
        *,
        steps=None,
        samplers=None,
        poisson_samplers=None,
        netstim_state=None,
        sample_netstim=None,
    ):
        if steps is not None and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
        ):
            raise ValueError("steps must be a non-negative integer")
        if steps == 0:
            return self.empty()
        samplers = (
            {
                stream: _sampler_from_frame(model[stream])
                for stream in self.model_reference
            }
            if samplers is None
            else {stream: samplers[stream] for stream in self.model_reference}
        )
        if poisson_samplers is None:
            poisson_samplers = {}
            for stream, source in self.stimulation_sources.items():
                if stream in self.borrowed:
                    owner, device = self.borrowed[stream]
                    generator = samplers[owner]._generator_for(device)
                else:
                    generator = torch.Generator(device=source.device)
                    generator.set_state(stimulation[stream].detach().cpu().clone())
                poisson_samplers[stream] = generator
        borrowed_devices = frozenset(self.borrowed.values())
        packets = []
        for _ in range(1 if steps is None else steps):
            contexts = {
                name: spec.context(samplers, poisson_samplers)
                for name, spec in self.populations.items()
            }
            model_frames = {
                canonical: {leaf: {} for leaf in reference}
                for canonical, reference in self.model_reference.items()
            }
            netstim = None
            if sample_netstim is not None:
                draw = sample_netstim(netstim_state)
                netstim = draw.exp_rand, draw.rng_state
                netstim_state = {**netstim_state, "rng_state": draw.rng_state}
            if self.network:
                phases = (
                    ("extra", self.extra_names),
                    ("intra", self.intra_names),
                    ("mechanism", self.populations),
                )
                for phase, names in phases:
                    for name in names:
                        spec, context = self.populations[name], contexts[name]
                        spec.sample(context, phase)
                        if phase == "mechanism":
                            for local, canonical in spec.model_routes.items():
                                frame = _rng_values(samplers[canonical], spec.device)
                                for leaf, value in frame.items():
                                    if (canonical, leaf) not in borrowed_devices:
                                        model_frames[canonical][leaf][name] = value
            else:
                name, spec = next(iter(self.populations.items()))
                spec.sample(contexts[name])
                for canonical in self.model_reference:
                    frame = _rng_values(samplers[canonical], spec.device)
                    model_frames[canonical] = {
                        leaf: {"step": value} for leaf, value in frame.items()
                    }
            if self.network:
                for canonical, leaf in borrowed_devices:
                    reference = self.model_reference[canonical][leaf]
                    model_frames[canonical][leaf]["step"] = _rng_values(
                        samplers[canonical], reference.device
                    )[leaf]
            packets.append(
                self._packet(
                    {
                        name: {
                            "noise": contexts[name]["noise_draws"]
                            if spec.has_noise
                            else None,
                            "stimulation": None
                            if spec.poisson is None
                            else contexts[name]["stimulation_draws"],
                        }
                        for name, spec in self.populations.items()
                    },
                    model_frames,
                    {
                        stream: poisson_samplers[stream].get_state().clone()
                        for stream in self.stimulation_reference
                    },
                    netstim,
                )
            )
        if steps is None:
            return packets[0]
        return _stack_runtime_packets(packets)


def sample_native_runtime_draws(model, *, dt, steps=None, extra=None):
    """Sample initialized native owners through the shared host request executor."""
    if steps is not None and (
        isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
    ):
        raise ValueError("steps must be a non-negative integer")
    from dendra.models.networks.net import Network
    from dendra.models.stim.waveform._poisson import poisson_stream_state

    from ._native_runtime_draws import _NetworkReplayPlan, _PopulationReplayPlan
    from ._stimulation_rng import native_runtime_requests

    network = isinstance(model, Network)
    with torch.inference_mode(False), torch.no_grad():
        inventory = (
            _NetworkReplayPlan(model, extra, dt=dt)
            if network
            else _PopulationReplayPlan(model, extra, dt=dt)
        )
        inventory.validate_source()
        local_plans = inventory.populations if network else {"population": inventory}
        model_reference = inventory.references if network else inventory.reference
        local_streams = (
            inventory.local
            if network
            else {"population": {stream: stream for stream in model_reference}}
        )
        poisson_plans = (
            inventory.poisson_plans
            if network
            else {}
            if inventory.poisson is None
            else {"population": inventory.poisson}
        )
        stimulation_sources = (
            inventory.poisson_streams
            if network
            else ({} if inventory.poisson is None else inventory.poisson.streams)
        )
        stimulation_local = (
            inventory.poisson_local
            if network
            else {"population": {stream: stream for stream in stimulation_sources}}
        )
        borrowed = inventory.borrowed
        groups = {stream: [] for stream in stimulation_sources}
        for name, plan in poisson_plans.items():
            for entry in plan.entries:
                groups[stimulation_local[name][entry.stream]].append(entry.source)
        stimulation = {
            stream: poisson_stream_state(nodes) for stream, nodes in groups.items()
        }
        # Deferred waveform frames describe the logical shared position. Read
        # them into the owned model snapshot without publishing live RNG state.
        model_reference = {
            stream: dict(frame) for stream, frame in model_reference.items()
        }
        for stream, (owner, device) in borrowed.items():
            model_reference[owner][device] = (
                stimulation[stream].to(model_reference[owner][device]).clone()
            )
        independent = {
            stream: value
            for stream, value in stimulation.items()
            if stream not in borrowed
        }
        specs = {}
        for name, local in local_plans.items():
            population = inventory.population_models[name] if network else model
            specs[name] = PopulationSamplingSpec(
                local.noise,
                local.poisson,
                native_runtime_requests(population, local.noise, local.poisson, dt=dt),
                population.device(),
                local.noise.check_purity,
                {
                    stream: canonical
                    for stream, canonical in local_streams[name].items()
                    if canonical in model_reference
                },
                stimulation_local.get(name, {}),
                bool(local.noise.runtime) and not borrowed,
                True,
            )
        stimulus = model.netstim if network else None
        netstim_reference = None
        netstim_state = None
        sample_netstim = None
        if stimulus is not None:
            from dendra.models.networks._netstim_renewal import (
                sample_netstim_exponential,
            )

            from ._netstim import NetStimDraw

            netstim_reference = {
                "draw": stimulus.next_stoch_time.new_empty(stimulus.shape),
                "rng": stimulus._rng.get_state().detach().clone(),
            }
            netstim_state = {"rng_state": netstim_reference["rng"]}

            def sample_netstim(state):
                generator = torch.Generator(device=stimulus._rng.device)
                generator.set_state(state["rng_state"].detach().cpu().clone())
                draw = sample_netstim_exponential(
                    generator,
                    stimulus.shape,
                    device=stimulus.next_stoch_time.device,
                    dtype=stimulus.next_stoch_time.dtype,
                )
                return NetStimDraw(draw, generator.get_state())

        program = RuntimeSamplingProgram(
            specs,
            model_reference,
            stimulation_sources,
            independent,
            borrowed,
            inventory.boundaries,
            tuple(() if extra is None else extra) if network else (),
            tuple(model.populations) if network else (),
            network,
            netstim_reference,
        )
        return program.sample(
            model_reference,
            stimulation,
            steps=steps,
            netstim_state=netstim_state,
            sample_netstim=sample_netstim,
        )
