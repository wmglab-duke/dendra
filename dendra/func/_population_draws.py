"""Canonical runtime draw values and RNG ownership for one Population."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils import _pytree as pytree

from ._draws import RuntimeDraws
from ._types import RuntimeNoiseDraw, StimulationDraw


def _model_frames(rng):
    return {
        stream: {name: {"step": value} for name, value in frame.items()}
        for stream, frame in rng.items()
    }


class PopulationRuntimeDrawPlan:
    """One runtime authority with legacy family packets as projections."""

    def __init__(self, functional):
        self.functional = functional
        self.noise = functional._noise_plan
        self.poisson = functional._poisson_plan
        self.sharing = functional._stimulation_rng_plan
        model_reference = (
            {}
            if functional._random_plan is None
            else functional._random_plan.rng_reference
        )
        borrowed_owners = {owner for owner, _device in self.sharing.borrowed.values()}
        self.model_reference = (
            model_reference
            if self.has_noise
            else {
                stream: frame
                for stream, frame in model_reference.items()
                if stream in borrowed_owners
            }
        )
        self.stimulation_reference = (
            {}
            if self.poisson is None
            else functional._stimulation_state_template.get("rng", {})
        )

    @property
    def has_noise(self):
        return self.noise is not None and bool(self.noise.runtime)

    def _packet(self, noise, stimulation, model_rng, stimulation_rng):
        return RuntimeDraws(
            {"noise": noise, "stimulation": stimulation},
            {"model": _model_frames(model_rng), "stimulation": stimulation_rng},
        )

    def finish_context(self, context):
        """Capture a joint sampler once without recording borrowed bytes twice."""
        from ._rng_frames import _rng_values

        return self._packet(
            context["noise_draws"] if self.has_noise else None,
            None if self.poisson is None else context["stimulation_draws"],
            {
                stream: _rng_values(
                    context["random_samplers"][stream], self.functional.device
                )
                for stream in self.model_reference
            },
            {
                stream: generator.get_state().clone()
                for stream, generator in context["poisson_samplers"].items()
                if stream not in self.sharing.borrowed
            },
        )

    def sample(self, state, *, steps=None):
        if steps is not None and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
        ):
            raise ValueError("steps must be a non-negative integer")
        if self.has_noise or self.poisson is not None:
            return self.sharing.sample_runtime_draws(state, steps=steps)
        return self._packet(None, None, {}, {})

    def legacy(self, packet):
        """Project the existing public packet layouts without copying values."""
        model_rng = {
            stream: {name: value["step"] for name, value in frame.items()}
            for stream, frame in packet.rng["model"].items()
        }
        stimulation_rng = {}
        if self.poisson is not None:
            for stream in self.poisson.streams:
                if stream in self.sharing.borrowed:
                    owner, device = self.sharing.borrowed[stream]
                    stimulation_rng[stream] = (
                        model_rng[owner][device].detach().to("cpu").clone()
                    )
                else:
                    stimulation_rng[stream] = packet.rng["stimulation"][stream]
        return (
            None
            if packet.draws["noise"] is None
            else RuntimeNoiseDraw(packet.draws["noise"], model_rng),
            None
            if packet.draws["stimulation"] is None
            else StimulationDraw(packet.draws["stimulation"], stimulation_rng),
        )

    def arithmetic(self, packet):
        """Numerical evaluators need draw values, never duplicated RNG frames."""
        return (
            None
            if packet.draws["noise"] is None
            else RuntimeNoiseDraw(packet.draws["noise"], {}),
            None
            if packet.draws["stimulation"] is None
            else StimulationDraw(packet.draws["stimulation"], {}),
        )

    @staticmethod
    def _validate_frame(value, reference, steps, message):
        shape = tuple(reference.shape)
        if steps is not None:
            shape = (steps, *shape)
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise ValueError(message)

    def validate(self, packet, *, steps=None):
        if not isinstance(packet, RuntimeDraws):
            raise TypeError("draws must be a RuntimeDraws packet")
        if not isinstance(packet.draws, Mapping) or set(packet.draws) != {
            "noise",
            "stimulation",
        }:
            raise ValueError("Runtime draw families do not match this Population")
        if not isinstance(packet.rng, Mapping) or set(packet.rng) != {
            "model",
            "stimulation",
        }:
            raise ValueError("Runtime RNG namespaces do not match this Population")
        if not isinstance(packet.rng["model"], Mapping) or set(
            packet.rng["model"]
        ) != set(self.model_reference):
            raise ValueError("Runtime RNG stream names do not match this Population")
        for stream, reference in self.model_reference.items():
            frame = packet.rng["model"][stream]
            if not isinstance(frame, Mapping) or set(frame) != set(reference):
                raise ValueError("Runtime RNG model frame has incompatible schema")
            for name, expected in reference.items():
                selected = frame[name]
                if not isinstance(selected, Mapping) or set(selected) != {"step"}:
                    raise ValueError(
                        "Runtime RNG model frame has incompatible boundary"
                    )
                self._validate_frame(
                    selected["step"],
                    expected,
                    steps,
                    "Runtime RNG model frame has incompatible Tensor schema",
                )
        expected_stimulation = set(self.stimulation_reference) - set(
            self.sharing.borrowed
        )
        if (
            not isinstance(packet.rng["stimulation"], Mapping)
            or set(packet.rng["stimulation"]) != expected_stimulation
        ):
            raise ValueError(
                "Runtime RNG stimulation stream names do not match this Population"
            )
        for stream in expected_stimulation:
            self._validate_frame(
                packet.rng["stimulation"][stream],
                self.stimulation_reference[stream],
                steps,
                "Runtime RNG stimulation frame has incompatible Tensor schema",
            )
        noise, stimulation = self.arithmetic(packet)
        if self.has_noise:
            self.noise.validate(noise, steps=steps)
        elif noise is not None:
            raise ValueError(
                "This Population has no recurrent NOISE or State SDE draws"
            )
        if self.poisson is None:
            if stimulation is not None:
                raise ValueError(
                    "This Population has no randomized Poisson stimulation"
                )
        else:
            self.poisson.validate(stimulation, steps=steps, validate_rng=False)
            from ._poisson_stimulation import _cold_validation

            if _cold_validation():
                self.poisson._validate_rng(self.legacy(packet)[1].rng, steps=steps)

    def publish(self, state, packet, *, tape=False):
        """Publish each canonical frame once; Poisson advances only schedules."""
        selected = (
            pytree.tree_map_only(torch.Tensor, lambda value: value[-1], packet)
            if tape
            else packet
        )
        result = dict(state)
        if self.model_reference:
            result["random"] = {
                **state["random"],
                "rng": {
                    stream: {
                        name: selected.rng["model"][stream][name]["step"].clone()
                        for name in frame
                    }
                    if stream in self.model_reference
                    else frame
                    for stream, frame in state["random"]["rng"].items()
                },
            }
        if self.poisson is not None:
            stimulation = StimulationDraw(selected.draws["stimulation"], {})
            advanced = self.poisson.advance(
                state["stimulation"], stimulation, publish_rng=False
            )
            rng = {
                stream: selected.rng["stimulation"][stream].detach().clone()
                for stream in self.stimulation_reference
                if stream not in self.sharing.borrowed
            }
            result["stimulation"] = {
                key: rng if key == "rng" else advanced[key]
                for key in state["stimulation"]
            }
        return result


__all__ = []
