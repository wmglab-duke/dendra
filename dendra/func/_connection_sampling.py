"""Ordered connection initialization draws with independent explicit RNG carry."""

from collections.abc import Mapping
from typing import NamedTuple

import torch

from dendra.models.networks._connection_sampling import clone_connection_sampling_rng


class ConnectionSamplingDraw(NamedTuple):
    """Parameter-independent base draws and resulting generator checkpoints.

    ``values`` maps ``synapses`` and ``continuous_synapses`` to authored
    connection names, then ``weights``/``delays`` to stochastic block indices.
    ``rng`` maps canonical draw-device names to CPU generator-state byte vectors.
    Draws and generator states are detached initialization inputs.
    """

    values: Mapping
    rng: Mapping


class ConnectionSamplingPlan:
    """Share native draw cadence while numerical initialization stays pure."""

    def __init__(self, plans):
        self.plans = plans
        self.devices = tuple(
            dict.fromkeys(
                str(device)
                for plan in plans.values()
                for device in plan.sampling_devices
            )
        )

    def extract(self, network):
        if not self.devices:
            return None
        explicit = vars(network).get("_functional_connection_sampling_rng")
        if explicit is None:
            values = {
                name: torch.random.get_rng_state()
                if name == "cpu"
                else torch.cuda.get_rng_state(torch.device(name))
                for name in self.devices
            }
        else:
            values = explicit
        return {"rng": self.validate_rng(values)}

    def validate_rng(self, values):
        if not isinstance(values, Mapping) or set(values) != set(self.devices):
            raise ValueError(
                "Connection sampling RNG devices do not match this Network."
            )
        return clone_connection_sampling_rng(values)

    def requires_draw(self, *, reinit_weights=True, reinit_delays=True):
        return any(
            plan.requires_initialization_draw(
                reinit_weights=reinit_weights, reinit_delays=reinit_delays
            )
            for plan in self.plans.values()
        )

    def sample(
        self, parameters, constants, state, *, reinit_weights=True, reinit_delays=True
    ):
        if type(reinit_weights) is not bool or type(reinit_delays) is not bool:
            raise TypeError("Connection reinitialization flags must be booleans.")
        if not self.devices:
            return None
        checkpoints = self.validate_rng(state["connection_sampling"]["rng"])
        generators = {
            name: torch.Generator(device=torch.device(name)).set_state(value)
            for name, value in checkpoints.items()
        }
        values = {"synapses": {}, "continuous_synapses": {}}
        # Dict insertion order matches Network.init_synapses: all event
        # connections, then continuous connections; each samples weights first.
        for (collection, name), plan in self.plans.items():
            values[collection][name] = plan.sample_initialization_draw(
                parameters[collection][name],
                constants[collection][name],
                reinit_weights=reinit_weights,
                reinit_delays=reinit_delays,
                generators=generators,
            )
        return ConnectionSamplingDraw(
            values,
            {
                name: generator.get_state().clone()
                for name, generator in generators.items()
            },
        )

    def validate_draw(
        self, draw, *, required=False, reinit_weights=True, reinit_delays=True
    ):
        if draw is None:
            if required:
                raise ValueError(
                    "Connection initialization requires an explicit connection_draw; "
                    "use sample_connection_initialization_draw() outside transformed execution."
                )
            return
        if not self.devices:
            raise ValueError(
                "This Network has no supported stochastic connection sources."
            )
        if not isinstance(draw, ConnectionSamplingDraw):
            raise TypeError("connection_draw must be a ConnectionSamplingDraw.")
        if not isinstance(draw.values, Mapping) or set(draw.values) != {
            "synapses",
            "continuous_synapses",
        }:
            raise ValueError("Connection draw collections do not match this Network.")
        for collection in ("synapses", "continuous_synapses"):
            expected = {name for kind, name in self.plans if kind == collection}
            if (
                not isinstance(draw.values[collection], Mapping)
                or set(draw.values[collection]) != expected
            ):
                raise ValueError("Connection draw names do not match this Network.")
            for name in expected:
                values = draw.values[collection][name]
                if not isinstance(values, Mapping) or set(values) != {
                    "weights",
                    "delays",
                }:
                    raise ValueError(
                        "Connection draws must contain weights and delays."
                    )
                plan = self.plans[collection, name]
                for role, refresh in (
                    ("weights", reinit_weights),
                    ("delays", reinit_delays),
                ):
                    blocks = plan.draw_schemas[role] if refresh else {}
                    if not isinstance(values[role], Mapping) or set(
                        values[role]
                    ) != set(blocks):
                        raise ValueError(
                            "Connection draw blocks must match the initialization resampling flags."
                        )
        if not isinstance(draw.rng, Mapping) or set(draw.rng) != set(self.devices):
            raise ValueError("Connection draw RNG devices do not match this Network.")
