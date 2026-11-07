"""Native connection and NetStim boundaries for initialization draw packets."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import torch

from dendra.models.distributions import connection_distribution_is_supported
from dendra.models.networks._connection_sampling import clone_connection_sampling_rng
from dendra.models.networks._netstim_renewal import sample_netstim_exponential
from dendra.models.stim.waveform._poisson import (
    _is_default_poisson_generator,
    _poisson_waveforms,
    defer_poisson_generator_restore,
    poisson_stream_state,
)

from ._connection_parameters import (
    ConnectionParameterPlan,
    _binding,
    _clone,
    _helpers,
    _registered,
)
from ._connection_sampling import ConnectionSamplingDraw, ConnectionSamplingPlan
from ._netstim import FunctionalNetStim, NetStimDraw
from ._rng_diagnostics import _rng_diagnostic_scope
from ._types import FunctionalizationError


class NativeNetworkInitialization:
    """Route one existing native initializer through isolated explicit draws."""

    def __init__(
        self,
        network,
        packet,
        *,
        capture=False,
        reinit_weights=True,
        reinit_delays=True,
    ):
        self.network = network
        self.capture = capture
        self.flags = {"weights": reinit_weights, "delays": reinit_delays}
        if any(type(value) is not bool for value in self.flags.values()):
            raise TypeError("Connection reinitialization flags must be booleans.")
        self.plans = {}
        self.routes = {}
        self.owners = {}
        for collection in ("synapses", "continuous_synapses"):
            for name, connection in getattr(network, collection).items():
                self.plans[collection, name] = (
                    ConnectionParameterPlan.sampling_metadata(connection)
                )
                for role, native in (("weights", "weight"), ("delays", "delay_ms")):
                    owner = getattr(connection, native)
                    if any(
                        not source.deterministic()
                        and not connection_distribution_is_supported(source.param)
                        for source in owner.weights
                    ):
                        raise FunctionalizationError(
                            "Native initialization draws require pure connection distribution hooks"
                        )
                    self.owners[id(owner)] = owner
                    self.routes.setdefault(id(owner), []).append(
                        (collection, name, role)
                    )
        self.connection_plan = ConnectionSamplingPlan(self.plans)
        nodes = tuple(_poisson_waveforms(network).values())
        self.default_generators = vars(network).get(
            "_native_initialization_default_generators"
        )
        if self.default_generators is None:
            self.default_generators = {
                str(node.generator.device): node.generator
                for node in nodes
                if _is_default_poisson_generator(node.generator)
            }
        self.default_nodes = {
            device: tuple(node for node in nodes if node.generator is generator)
            for device, generator in self.default_generators.items()
        }
        self.connections_started = False
        self.connection_draw = None if packet is None else packet.connections
        self.values = {
            collection: {
                name: {"weights": {}, "delays": {}}
                for kind, name in self.plans
                if kind == collection
            }
            for collection in ("synapses", "continuous_synapses")
        }
        self.positions = Counter()
        self.active = {}
        self.completed = Counter()
        self.generators = {}
        self.endpoint = None
        if capture:
            if self.connection_plan.devices:
                frames = self.connection_plan.extract(network)["rng"]
                self.generators = {
                    name: torch.Generator(device=torch.device(name)).set_state(frame)
                    for name, frame in frames.items()
                }
        else:
            self.connection_plan.validate_draw(
                self.connection_draw,
                required=self.connection_plan.requires_draw(**self._flags()),
                **self._flags(),
            )
            if self.connection_draw is not None:
                self.endpoint = self.connection_plan.validate_rng(
                    self.connection_draw.rng
                )
                for (collection, name), plan in self.plans.items():
                    for role, refresh in self.flags.items():
                        if refresh:
                            plan._validate_draws(
                                role,
                                self.connection_draw.values[collection][name][role],
                            )
            elif self.connection_plan.devices:
                self.endpoint = self.connection_plan.extract(network)["rng"]
        self.stimulus = network.netstim
        self.stimulus_plan = (
            None if self.stimulus is None else FunctionalNetStim(self.stimulus)
        )
        self.stimulus_draw = None if packet is None else packet.netstim
        self.stimulus_completed = False
        if not capture:
            if self.stimulus_plan is None:
                if self.stimulus_draw is not None:
                    raise ValueError("This Network has no NetStim initialization draws")
            else:
                self.stimulus_plan._validate_draw(self.stimulus_draw)
                generator = torch.Generator(device=self.stimulus._rng.device)
                generator.set_state(self.stimulus_draw.rng_state.detach().cpu().clone())

    def __deepcopy__(self, memo):
        return self

    def _flags(self):
        return {
            "reinit_weights": self.flags["weights"],
            "reinit_delays": self.flags["delays"],
        }

    def slots(self):
        """Yield private owner slots installed by the outer restoration scope."""
        yield self.network, self
        for owner in self.owners.values():
            yield owner, self
        if self.stimulus is not None:
            yield self.stimulus, self

    def begin_connection(self, owner, *, reinit):
        if not self.connections_started:
            # Population initialization consumes these physical default streams
            # before connection draws. Read the resulting logical endpoint at
            # the first actual connection boundary, rather than at setup.
            for device, nodes in self.default_nodes.items():
                if not nodes or device not in self.connection_plan.devices:
                    continue
                state = poisson_stream_state(nodes)
                if self.capture:
                    self.generators[device].set_state(state)
                elif self.connection_draw is None:
                    self.endpoint[device] = state
            self.connections_started = True
        identity = id(owner)
        position = self.positions[identity]
        routes = self.routes[identity]
        if position >= len(routes):
            raise ValueError("Connection initialization invocation count changed")
        route = routes[position]
        if bool(reinit) != self.flags[route[2]]:
            raise ValueError("Connection initialization resampling flags changed")
        if (
            not reinit
            and not owner.w.numel()
            and any(not source.deterministic() for source in owner.weights)
        ):
            raise FunctionalizationError(
                "Retaining stochastic connection values requires initialized nonempty storage"
            )
        self.active[identity] = route
        self.positions[identity] += 1

    def connection_sample(self, owner, index, source, count):
        collection, name, role = self.active[id(owner)]
        with self._connection_diagnostics(collection, name, source, count):
            if self.capture:
                device = source.sampling_draw_schema(count)[2]
                value, draw = source.param.sample_with_draw(
                    (count,), generator=self.generators[str(device)]
                )
                value = source._convert(value)
            else:
                draw = self.connection_draw.values[collection][name][role][str(index)]
                value = source.derive_values(count, draw)
        self.values[collection][name][role][str(index)] = draw.detach().clone()
        return value, draw

    @contextmanager
    def _connection_diagnostics(self, collection, name, source, count):
        plan = self.plans[collection, name]
        if not plan._check_purity:
            yield
            return
        helpers = _helpers()
        classes = tuple(dict.fromkeys(cls for cls, *_ in plan._class_guards))
        before = _binding(source, identities=False)
        inputs = tuple(
            (
                value,
                None if torch.is_inference(value) else value._version,
                _clone(value),
            )
            for _name, value, _kind in _registered(source)
        )
        device = source.sampling_draw_schema(count)[2]
        frames = {
            name: generator.get_state().clone()
            for name, generator in self.generators.items()
        }
        try:
            with (
                _rng_diagnostic_scope(device, enabled=True) as diagnostic,
                helpers._class_state_diagnostic_scope(classes) as mutated_classes,
            ):
                yield
                if before != _binding(source, identities=False):
                    raise FunctionalizationError(
                        "Connection sampler purity diagnostics observed mutable state."
                    )
                if any(
                    (version is not None and value._version != version)
                    or not torch.equal(value, saved)
                    for value, version, saved in inputs
                ):
                    raise FunctionalizationError(
                        "Connection sampler purity diagnostics observed an in-place tensor write."
                    )
            if mutated_classes:
                raise FunctionalizationError(
                    "Connection sampler purity diagnostics observed shared class mutation."
                )
            if diagnostic.changes:
                raise FunctionalizationError(
                    "Connection sampler purity diagnostics observed implicit random draws instead of the supplied generator."
                )
        except Exception:
            for name, frame in frames.items():
                self.generators[name].set_state(frame)
            raise

    def complete_connection(self, owner):
        route = self.active.pop(id(owner))
        collection, name, role = route
        if self.flags[role]:
            self.plans[collection, name]._validate_draws(
                role, self.values[collection][name][role]
            )
        self.completed[route] += 1

    def netstim_draw(self, stimulus, noise):
        if stimulus is not self.stimulus or self.stimulus_completed:
            raise ValueError("NetStim initialization invocation changed")
        if not self.capture:
            return self.stimulus_draw.exp_rand.detach()
        generator = torch.Generator(device=stimulus._rng.device)
        generator.set_state(stimulus._rng.get_state().clone())
        draw = (
            sample_netstim_exponential(
                generator,
                stimulus.shape,
                device=stimulus.device(),
                dtype=stimulus.dtype(),
            )
            if bool(torch.any(noise > 0))
            else torch.zeros_like(noise)
        )
        self.stimulus_draw = NetStimDraw(draw.detach(), generator.get_state().clone())
        return draw

    def complete_netstim(self, stimulus):
        if stimulus is not self.stimulus or self.stimulus_draw is None:
            raise ValueError("NetStim initialization invocation changed")
        stimulus._rng.set_state(self.stimulus_draw.rng_state.detach().cpu().clone())
        self.stimulus_completed = True

    def connections_packet(self):
        if not self.connection_plan.devices or (
            not self.capture and self.connection_draw is None
        ):
            return None
        endpoint = (
            {
                name: generator.get_state().clone()
                for name, generator in self.generators.items()
            }
            if self.capture
            else self.endpoint
        )
        return ConnectionSamplingDraw(self.values, endpoint)

    def netstim_packet(self):
        return self.stimulus_draw

    def finish(self):
        expected = Counter(route for routes in self.routes.values() for route in routes)
        if self.completed != expected or self.active:
            raise ValueError("Connection initialization invocation count changed")
        if self.stimulus_completed != (self.stimulus is not None):
            raise ValueError("NetStim initialization invocation count changed")
        if not self.connection_plan.devices:
            self.network.__dict__.pop("_functional_connection_sampling_rng", None)
        else:
            endpoint = self.connections_packet().rng if self.capture else self.endpoint
            self.network._functional_connection_sampling_rng = (
                clone_connection_sampling_rng(endpoint)
            )
            for device, nodes in self.default_nodes.items():
                if not nodes or device not in endpoint:
                    continue
                if self.capture:
                    self.default_generators[device].set_state(endpoint[device])
                    for node in nodes:
                        holder = vars(node).get("_functional_poisson_rng")
                        if holder is not None and holder.generator is node.generator:
                            holder.state = None
                else:
                    # An explicit native replay carries its default-stream
                    # endpoint with the waveform aliases. Publication must not
                    # reset the caller's ambient Torch generator.
                    defer_poisson_generator_restore(nodes, endpoint[device])
