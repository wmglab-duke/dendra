"""Pure execution composed from an initialized standard Network's components."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from contextlib import nullcontext
from typing import NamedTuple

import torch

from dendra.models.integrators import _bwd_euler_sc_skip
from dendra.models.networks.net import (
    Network,
    _network_step_schedule,
)
from dendra.models.networks.net import step as _native_step

from . import _network_parameters
from ._continuous import FunctionalContinuousCon
from ._continuous_transform import _named_tensors
from ._netcon import FunctionalNetCon, _effective_tensor, _snapshot
from ._netstim import FunctionalNetStim
from ._network_bindings import AliasSnapshots, SourceBinding
from ._network_initialization import NetworkInitializationInput
from ._population import (
    _deepcopy_execution_module,
    _RawClassHookIdentity,
    make_functional,
)
from ._schedules import TensorScheduleTensors, _TensorSchedulePlan
from ._types import FunctionalizationError, RolloutInput, StepInput


class NetworkTensors(NamedTuple):
    """Aggregate parameters, constants, and complete-boundary carry.

    Extraction produces independent snapshots retaining genuine shared
    parameter identities. Fresh initialization returns effective parameters,
    preserving unchanged explicit leaves and their gradient history. Component
    names match the ordinary Network's collections. Clock and RNG byte tensors
    retain their native devices, including CPU clock metadata in an otherwise
    accelerator-resident Network.
    """

    parameters: dict
    constants: dict
    state: dict

    @property
    def named_parameters(self):
        """Flatten explicit leaves by canonical collection/component-qualified names."""
        return _network_parameters.named_parameters(self.parameters)

    def find_parameters(self, query):
        """Return all parameter leaves whose canonical names contain ``query``."""
        return _network_parameters.find_parameters(self.parameters, query)

    def parameter_name(self, query):
        """Resolve an exact canonical name or unique partial-name query."""
        return _network_parameters.parameter_name(self.parameters, query)

    def get_parameter(self, query):
        """Return the leaf selected by an exact or unique partial name."""
        return _network_parameters.get_parameter(self.parameters, query)

    def independent_parameters(self, trainable=()):
        """Clone nested parameters with shared identities and trainable selection."""
        return _network_parameters.independent_parameters(self.parameters, trainable)


class _TensorSchema(NamedTuple):
    shape: tuple
    dtype: torch.dtype
    device: torch.device


def _schema(value):
    if torch.is_tensor(value):
        return _TensorSchema(tuple(value.shape), value.dtype, value.device)
    if isinstance(value, Mapping):
        return {name: _schema(child) for name, child in value.items()}
    if isinstance(value, tuple):
        return (
            type(value)(*map(_schema, value))
            if hasattr(value, "_fields")
            else tuple(map(_schema, value))
        )
    if value is not None:
        raise TypeError(
            "Network tensor packets contain only mappings, tuples, Tensors, and None"
        )
    return None


def _validate_tree(value, schema, label):
    if isinstance(schema, _TensorSchema):
        if not torch.is_tensor(value):
            raise TypeError(f"{label} must be a Tensor")
        if (tuple(value.shape), value.dtype, value.device) != schema:
            raise ValueError(f"{label} must match {schema}")
    elif isinstance(schema, Mapping):
        if not isinstance(value, Mapping) or set(value) != set(schema):
            raise ValueError(f"{label} must match this Network's component schema")
        for name, child in schema.items():
            _validate_tree(value[name], child, f"{label}.{name}")
    elif isinstance(schema, tuple):
        if type(value) is not type(schema) or len(value) != len(schema):
            raise ValueError(f"{label} must match this Network's tuple schema")
        for index, child in enumerate(schema):
            _validate_tree(value[index], child, f"{label}[{index}]")
    elif value is not None:
        raise ValueError(f"{label} must be None")


def _tensor_paths(value, path=()):
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, Mapping):
        for name, child in value.items():
            yield from _tensor_paths(child, (*path, name))
    elif isinstance(value, tuple):
        for index, child in enumerate(value):
            yield from _tensor_paths(child, (*path, index))


def _at(value, path):
    for key in path:
        value = value[key]
    return value


_NETWORK_METHODS = {
    name: inspect.getattr_static(Network, name)
    for name in (
        "step",
        "_refresh_step_schedule",
        "_runtime_clock_value",
        "_sync_runtime_clock",
        "_advance_runtime_clock",
        "_sync_population_clocks",
        "state_dict_for_checkpoint",
        "restore_dict_from_checkpoint",
    )
}
_NETWORK_INITIALIZATION_METHODS = {
    name: inspect.getattr_static(Network, name)
    for name in (
        "initialize",
        "build",
        "init_synapses",
        "_reset_runtime_clock",
        "_clear_duration_remainder",
        "_set_duration_remainder",
    )
}
_COLLECTIONS = ("populations", "synapses", "continuous_synapses", "continuous_targets")
_CACHES = (
    "_population_step_items",
    "_synapse_step_items",
    "_continuous_synapse_step_items",
    "_continuous_target_step_items",
)


def _eligible(network):
    if type(network) is not Network:
        raise FunctionalizationError("Expected an initialized canonical Network.")
    for name, expected in _NETWORK_METHODS.items():
        if (
            name in network.__dict__
            or inspect.getattr_static(type(network), name) is not expected
        ):
            raise FunctionalizationError(f"Network needs its canonical {name} adapter.")
    if not network.built or network.dt is None:
        raise FunctionalizationError("Initialize the Network before lowering it.")
    if not math.isfinite(float(network.dt)) or float(network.dt) <= 0:
        raise FunctionalizationError("Network timestep must be finite and positive.")
    if network._mode_requires_initialize:
        raise FunctionalizationError("Initialize the Network after changing its mode.")
    if any(
        getattr(network, name) is not _native_step
        for name in ("_step", "_step_train", "_step_eval")
    ):
        raise FunctionalizationError("Network requires its canonical timestep phases.")
    if not network.populations:
        raise FunctionalizationError(
            "A functional Network needs at least one Population."
        )
    for name in ("populations", "synapses", "continuous_synapses"):
        owners = tuple(getattr(network, name).values())
        if len({id(owner) for owner in owners}) != len(owners):
            raise FunctionalizationError(
                "Functional Network requires distinct mutable component owners."
            )
    population_modules = set()
    for population in network.populations.values():
        owned = {id(module) for module in population.modules()}
        if population_modules & owned:
            raise FunctionalizationError(
                "Functional Network Populations must own distinct modules; "
                "share raw parameter Tensors instead."
            )
        population_modules.update(owned)
    devices = {population.device() for population in network.populations.values()}
    if len(devices) != 1 or (
        network.netstim is not None and network.netstim.device() not in devices
    ):
        raise FunctionalizationError(
            "Functional Network currently requires a single simulation device."
        )
    for population in network.populations.values():
        if population.integrator is None or not population.integrator.initialized:
            raise FunctionalizationError(
                "Initialize every Network Population before lowering."
            )


class FunctionalNetwork:
    """A frozen canonical graph composing the existing pure component plans.

    The phase order is shared with native Network execution: NetStim, connected
    continuous-target resets, continuous projections, event projections, then
    population solves. Sources bind automatically to the current explicit
    population frame, including updates from preceding receives.

    ``step`` consumes opaque reusable preparation; ``prepare_and_step`` derives
    it within one compile-safe invocation. Both return ``(state, aux)`` without
    changing the native model or input trees. Sample NetStim draw packets with
    ``sample_step_draw`` outside compiled/transformed code, then supply one per
    timestep. Complete-boundary carry contains clocks, queues and RNG state;
    receiver scratch is local to a single invocation.

    Parameters/constants follow the component adapters' conventions. Extracted
    Population parameters are raw parameters, while projection and NetStim
    parameters are effective values. Snapshot extraction is detached and does
    not perform fresh initialization. ``commit_state_`` publishes runtime carry
    only; align native parameters/schedules separately before continuation.
    Topology, source bindings, mode, timestep, and native phase ordering are
    frozen. Canonical single-device Networks use the dense or source-history
    event runtimes and integer continuous delays supported by the component
    plans. ``dn.compile`` uses this plan for snapshot or fresh Network
    initialization; the lower-level methods also support explicit preparation
    and functional transforms.
    """

    def __init__(self, network):
        _eligible(network)
        self._source = network
        self.dt, self.training = float(network.dt), bool(network.training)
        self.device = next(iter(network.populations.values())).device()
        self._collections = tuple(
            tuple(getattr(network, name).items()) for name in _COLLECTIONS
        )
        self._schedule = _network_step_schedule(network)
        for cache, entries in zip(_CACHES, self._schedule, strict=True):
            if tuple(getattr(network, cache)) != entries:
                raise FunctionalizationError(
                    "Network cached schedule is stale; initialize again."
                )
        self._netstim_source = network.netstim
        self.populations, initialization = {}, {}
        for name, population in self._schedule.populations:
            plan, packet = make_functional(population, dt=self.dt)
            self.populations[name] = plan
            initialization[name] = packet.initialization
        self.synapses = {
            name: FunctionalNetCon(connection)
            for name, connection in self._schedule.synapses
        }
        self.continuous_synapses = {
            name: FunctionalContinuousCon(connection)
            for name, connection in self._schedule.continuous_synapses
        }
        self.netstim = (
            None if network.netstim is None else FunctionalNetStim(network.netstim)
        )
        self._bindings, self._targets, self._continuous_inputs = {}, {}, {}
        for collection in ("synapses", "continuous_synapses"):
            for name, connection in getattr(network, collection).items():
                post = self._population_name(connection.post)
                target = self._target_name(post, connection.syn)
                receivers = (
                    self.populations[post]._receivers
                    if collection == "synapses"
                    else self.populations[post]._continuous_receivers
                )
                if target not in receivers:
                    raise FunctionalizationError(
                        f"Connection {name!r} has no supported pure receiver."
                    )
                self._targets[collection, name] = post, target
                if collection == "continuous_synapses":
                    inputs = self.populations[post]._continuous_receiver_inputs[target]
                    input_name = connection.input
                    if input_name is None:
                        if len(inputs) != 1:
                            raise FunctionalizationError(
                                "A continuous target with multiple inputs needs an explicit input name."
                            )
                        input_name = inputs[0]
                    if input_name not in inputs:
                        raise FunctionalizationError("Unknown continuous target input.")
                    self._continuous_inputs[name] = input_name
                if connection.pre is network.netstim:
                    if collection != "synapses":
                        raise FunctionalizationError(
                            "Continuous connections require Population sources."
                        )
                    self._bindings[collection, name] = None
                else:
                    pre = self._population_name(connection.pre)
                    self._bindings[collection, name] = (
                        pre,
                        SourceBinding(
                            self.populations[pre],
                            network.populations[pre],
                            connection.pre_var,
                        ),
                    )
        self._reset_targets = tuple(
            self._locate_target(target)
            for _, target in self._schedule.continuous_targets
        )
        tensors = self._extract(network)
        self._parameter_schema, self._constant_schema, self._state_schema = map(
            _schema, tensors
        )
        aliases = {}
        for path, value in _tensor_paths(tensors.parameters):
            aliases.setdefault(id(value), []).append(path)
        self._parameter_aliases = tuple(
            tuple(paths) for paths in aliases.values() if len(paths) > 1
        )
        constant_aliases = {}
        for path, value in _tensor_paths(tensors.constants):
            constant_aliases.setdefault(id(value), []).append(path)
        self._parameter_constant_aliases = tuple(
            (path, tuple(constant_aliases[id(value)]))
            for path, value in _tensor_paths(tensors.parameters)
            if id(value) in constant_aliases
        )
        self._initialization_inputs = NetworkInitializationInput(
            populations=initialization,
            t=torch.zeros_like(tensors.state["clock"]["t"]),
        )
        from ._network_initialization import initialization_alias_failures

        self._initialization_alias_failures = initialization_alias_failures(self)

    @property
    def initialization_inputs(self):
        """Independent initialization-only defaults with editable Tensor leaves."""
        return self._initialization_inputs

    def _validate_initialization_source(self):
        self._validate_source()
        network = self._source
        if network._state_cache or network._syn_cache:
            raise FunctionalizationError(
                "Network state-cache restoration is currently unsupported by fresh initialization"
            )
        if self._initialization_alias_failures:
            raise FunctionalizationError("\n".join(self._initialization_alias_failures))
        for name, expected in _NETWORK_INITIALIZATION_METHODS.items():
            namespace = type(network).__dict__
            matches = (
                namespace == _RawClassHookIdentity(name, expected)
                if torch.compiler.is_compiling()
                else namespace.get(name) is expected
            )
            if name in network.__dict__ or not matches:
                raise FunctionalizationError(
                    f"Fresh Network initialization requires its canonical {name} adapter"
                )

    def initialize(self, parameters, constants, state, inputs, *, netstim_draw=None):
        """Initialize current explicit parameters and carry in native order.

        Population inputs and the starting clock are explicit. Existing carry
        supplies NetStim RNG/seeder/calendar positions; delivery and detector
        carry is reset. Supply an explicit NetStim initialization draw when
        present. Effective parameter replacements preserve Network aliases and
        are returned with complete state in :class:`NetworkTensors`.
        """
        from ._network_initialization import initialize

        return initialize(
            self, parameters, constants, state, inputs, netstim_draw=netstim_draw
        )

    def sample_initialization_draw(self, parameters, state):
        """Sample one initialization packet from independent explicit RNG carry."""
        from ._network_bindings import _canonicalize_mapping_tree

        (parameters, state), _ = _canonicalize_mapping_tree((parameters, state))
        self._validate_initialization_source()
        self._validate_parameters(parameters)
        self._validate_state(state)
        return (
            None
            if self.netstim is None
            else self.netstim.sample_initialization_draw(
                parameters["netstim"]["parameters"], state["netstim"]
            )
        )

    def _initialize_host(self, parameters, constants, state, inputs):
        from ._network_initialization import initialize_host

        return initialize_host(self, parameters, constants, state, inputs)

    def _population_name(self, population):
        for name, candidate in self._source.populations.items():
            if population is candidate:
                return name
        raise FunctionalizationError(
            "Every connection endpoint must belong to this Network."
        )

    def _target_name(self, population, target):
        for name, candidate in self._source.populations[
            population
        ].mech.mechanisms.items():
            if target is candidate:
                return name
        raise FunctionalizationError(
            "Connection target must be a registered Population mechanism."
        )

    def _locate_target(self, target):
        for name, population in self._source.populations.items():
            for key, candidate in population.mech.mechanisms.items():
                if target is candidate:
                    if key not in self.populations[name]._continuous_receivers:
                        raise FunctionalizationError(
                            "Connected continuous target needs canonical pure adapters."
                        )
                    return name, key
        raise FunctionalizationError("Continuous target must belong to this Network.")

    def _validate_source(self):
        network = self._source
        if (
            type(network) is not Network
            or not network.built
            or network._mode_requires_initialize
            or network.dt != self.dt
            or bool(network.training) != self.training
            or network.netstim is not self._netstim_source
        ):
            raise FunctionalizationError(
                "The Network configuration changed; initialize and lower again."
            )
        for name, entries in zip(_COLLECTIONS, self._collections, strict=True):
            collection = getattr(network, name)
            if tuple(collection) != tuple(key for key, _ in entries):
                raise FunctionalizationError("Network wiring changed; lower again.")
            for key, expected in entries:
                if collection[key] is not expected:
                    raise FunctionalizationError(
                        "Network component bindings changed; lower again."
                    )
        for cache, entries in zip(_CACHES, self._schedule, strict=True):
            current = getattr(network, cache)
            if len(current) != len(entries):
                raise FunctionalizationError(
                    "Network phase ordering changed; lower again."
                )
            for (key, component), (expected_key, expected) in zip(
                current, entries, strict=True
            ):
                if key != expected_key or component is not expected:
                    raise FunctionalizationError(
                        "Network phase ordering changed; lower again."
                    )
        for name, expected in _NETWORK_METHODS.items():
            namespace = type(network).__dict__
            matches = (
                namespace == _RawClassHookIdentity(name, expected)
                if torch.compiler.is_compiling()
                else namespace.get(name) is expected
            )
            if name in network.__dict__ or not matches:
                raise FunctionalizationError("Network adapters changed; lower again.")
        if any(
            getattr(network, name) is not _native_step
            for name in ("_step", "_step_train", "_step_eval")
        ):
            raise FunctionalizationError(
                "Network timestep phases changed; lower again."
            )
        for plan in (
            *self.populations.values(),
            *self.synapses.values(),
            *self.continuous_synapses.values(),
        ):
            plan._validate_source()
        if self.netstim is not None:
            self.netstim._validate_source()

    def _validate_parameters(self, parameters):
        _validate_tree(parameters, self._parameter_schema, "parameters")
        for name, plan in self.continuous_synapses.items():
            plan._validate_transform_parameters(
                parameters["continuous_synapses"][name]["transform_parameters"]
            )
        for group in self._parameter_aliases:
            first = _at(parameters, group[0])
            if any(_at(parameters, path) is not first for path in group[1:]):
                raise ValueError(
                    "Explicit Network parameters must preserve their shared tensor aliases"
                )

    def _validate_state(self, state):
        _validate_tree(state, self._state_schema, "state")

    def _extract(self, network):
        snapshots = AliasSnapshots()
        parameters = {
            "populations": {},
            "synapses": {},
            "continuous_synapses": {},
            "netstim": None,
        }
        constants = {"populations": {}, "netstim": None}
        state = {
            "populations": {},
            "synapses": {},
            "continuous_synapses": {},
            "netstim": None,
        }
        for name, plan in self.populations.items():
            population = network.populations[name]
            packet = plan.extract(population)
            parameters["populations"][name] = snapshots.tree(
                dict(population.named_parameters()), packet.parameters
            )
            native_constants = {
                entry.key: population.get_submodule(entry.module_path)._buffers[
                    entry.buffer_name
                ]
                for entry in plan._parametrization_constants
            }
            constants["populations"][name] = snapshots.tree(
                native_constants, packet.constants
            )
            state["populations"][name] = packet.state
        for name, plan in self.synapses.items():
            connection = network.synapses[name]
            packet = plan.extract(connection)
            schedule = (
                None
                if packet.schedule is None
                else TensorScheduleTensors(**_TensorSchedulePlan._values(connection))
            )
            parameters["synapses"][name] = snapshots.tree(
                {
                    "weights": _effective_tensor(connection.weight, "weight"),
                    "delays": _effective_tensor(connection.delay_ms, "delay")
                    if packet.delays is not None
                    else None,
                    "schedule": schedule,
                },
                {
                    "weights": packet.weights,
                    "delays": packet.delays,
                    "schedule": packet.schedule,
                },
            )
            state["synapses"][name] = packet.state
        for name, plan in self.continuous_synapses.items():
            connection = network.continuous_synapses[name]
            packet = plan.extract(connection)
            native_transform = (
                _named_tensors(connection.transform)
                if isinstance(connection.transform, torch.nn.Module)
                else None
            )
            parameters["continuous_synapses"][name] = snapshots.tree(
                {
                    "weights": _effective_tensor(connection.weight, "weight"),
                    "transform_parameters": native_transform,
                },
                {
                    "weights": packet.weights,
                    "transform_parameters": packet.transform_parameters,
                },
            )
            state["continuous_synapses"][name] = packet.state
        if self.netstim is not None:
            source = network.netstim
            packet = self.netstim.extract(source)
            native_values = {"start": source.start, "noise": source.noise}
            if torch.is_tensor(source.tau):
                native_values["tau"] = source.tau
            schedule = (
                None
                if packet.schedule is None
                else TensorScheduleTensors(**_TensorSchedulePlan._values(source))
            )
            parameters["netstim"] = snapshots.tree(
                {"parameters": native_values, "schedule": schedule},
                {"parameters": packet.parameters, "schedule": packet.schedule},
            )
            constants["netstim"] = snapshots.tree(
                {name: getattr(source, name) for name in packet.constants},
                packet.constants,
            )
            state["netstim"] = packet.state
        state["clock"] = {
            "t": _snapshot(network.t),
            "origin": _snapshot(network._clock_origin),
            "step": _snapshot(network._clock_step),
            "duration_remainder": _snapshot(network._duration_remainder),
        }
        return NetworkTensors(parameters, constants, state)

    def extract(self, network=None):
        """Take independent detached snapshots from a compatible initialized graph."""
        if torch.compiler.is_compiling():
            raise FunctionalizationError(
                "Extract Network tensors outside torch.compile."
            )
        self._validate_source()
        network = self._source if network is None else network
        self._validate_compatible(network)
        result = self._extract(network)
        self._validate_parameters(result.parameters)
        _validate_tree(result.constants, self._constant_schema, "constants")
        self._validate_state(result.state)
        return result

    def _validate_compatible(self, network):
        _eligible(network)
        if float(network.dt) != self.dt or bool(network.training) != self.training:
            raise FunctionalizationError(
                "Network timestep or mode does not match this plan."
            )
        expected_schedule = self._schedule
        schedule = _network_step_schedule(network)
        for expected, current in zip(expected_schedule, schedule, strict=True):
            if tuple(name for name, _ in expected) != tuple(
                name for name, _ in current
            ):
                raise FunctionalizationError(
                    "Network phase topology does not match this plan."
                )
        for (_, target), (population, name) in zip(
            schedule.continuous_targets, self._reset_targets, strict=True
        ):
            if target is not network.populations[population].mech.mechanisms[name]:
                raise FunctionalizationError(
                    "Network continuous reset target bindings do not match."
                )
        for name, connection in network.synapses.items():
            post, target = self._targets["synapses", name]
            if (
                connection.post is not network.populations[post]
                or connection.syn
                is not network.populations[post].mech.mechanisms[target]
            ):
                raise FunctionalizationError(
                    "Network event target bindings do not match."
                )
            binding = self._bindings["synapses", name]
            expected_pre = (
                network.netstim if binding is None else network.populations[binding[0]]
            )
            if connection.pre is not expected_pre:
                raise FunctionalizationError(
                    "Network event source bindings do not match."
                )
        for name, connection in network.continuous_synapses.items():
            post, target = self._targets["continuous_synapses", name]
            pre, _ = self._bindings["continuous_synapses", name]
            if (
                connection.pre is not network.populations[pre]
                or connection.post is not network.populations[post]
                or connection.syn
                is not network.populations[post].mech.mechanisms[target]
            ):
                raise FunctionalizationError(
                    "Network continuous bindings do not match."
                )
        if (network.netstim is None) != (self.netstim is None):
            raise FunctionalizationError("Network NetStim presence does not match.")

    def prepare(self, parameters, constants):
        """Prepare reusable Population coefficients outside compiled code."""
        self._validate_source()
        self._validate_parameters(parameters)
        _validate_tree(constants, self._constant_schema, "constants")
        return {
            "populations": {
                name: plan.prepare(
                    parameters["populations"][name], constants["populations"][name]
                )
                for name, plan in self.populations.items()
            },
            "netstim": constants["netstim"],
        }

    def sample_step_draw(self, state):
        """Sample a replayable packet from an independent explicit RNG copy."""
        self._validate_source()
        self._validate_state(state)
        return (
            None
            if self.netstim is None
            else self.netstim.sample_step_draw(state["netstim"])
        )

    def _prepare_values(self, parameters, constants):
        return {
            "populations": {
                name: plan._prepare_values(
                    parameters["populations"][name], constants["populations"][name]
                )
                for name, plan in self.populations.items()
            },
            "netstim": constants["netstim"],
        }

    def _validate_prepared(self, prepared, parameters):
        if (
            not isinstance(prepared, Mapping)
            or set(prepared) != {"populations", "netstim"}
            or not isinstance(prepared["populations"], Mapping)
            or set(prepared["populations"]) != set(self.populations)
        ):
            raise ValueError("prepared must match this Network's component plans")
        _validate_tree(
            prepared["netstim"], self._constant_schema["netstim"], "prepared.netstim"
        )
        return {
            "populations": {
                name: plan._validate_prepared(
                    prepared["populations"][name], parameters["populations"][name]
                )
                for name, plan in self.populations.items()
            },
            "netstim": prepared["netstim"],
        }

    def make_callbacks(self, callbacks):
        """Bind qualified Network recordings or pure custom callbacks."""
        from ._network_callbacks import make_network_callbacks

        self._validate_source()
        return make_network_callbacks(self, callbacks)

    def sample_draw_tape(self, state, steps):
        """Sample replayable NetStim draw/RNG packets outside compiled code."""
        from ._network_rollout import sample_draw_tape

        return sample_draw_tape(self, state, steps)

    def rollout(
        self,
        parameters,
        prepared,
        state,
        inputs=None,
        *,
        steps=None,
        callbacks=None,
        callback_state=None,
    ):
        """Execute a pure horizon with per-Population inputs and explicit draws."""
        from ._network_rollout import rollout

        return rollout(
            self,
            parameters,
            prepared,
            state,
            inputs,
            steps=steps,
            callbacks=callbacks,
            callback_state=callback_state,
        )

    def prepare_and_rollout(
        self,
        parameters,
        constants,
        state,
        inputs=None,
        *,
        steps=None,
        callbacks=None,
        callback_state=None,
    ):
        """Derive coefficients and execute an explicit, differentiable horizon."""
        from ._network_rollout import rollout

        return rollout(
            self,
            parameters,
            None,
            state,
            inputs,
            steps=steps,
            callbacks=callbacks,
            callback_state=callback_state,
            constants=constants,
        )

    def compile_rollout_chunk(self, steps, *, execution="scan", **compile_options):
        """Build a fixed-horizon chunk using the shared scan or unrolled body."""
        from ._network_rollout import CompiledNetworkChunk

        return CompiledNetworkChunk(
            self, steps=steps, execution=execution, **compile_options
        )

    def run(
        self,
        parameters,
        constants,
        state,
        inputs=None,
        *,
        steps=None,
        chunklength=64,
        checkpoint_every=None,
        callbacks=None,
        callback_state=None,
        compiled_chunk=None,
    ):
        """Schedule exact chunks, optional checkpoints and bounded RNG sampling.

        Returned state and ``aux['callbacks'].state`` can be passed directly to
        another call. Checkpoint replay retains gradients through all chunks.
        """
        from ._network_rollout import run

        return run(
            self,
            parameters,
            constants,
            state,
            inputs,
            steps=steps,
            chunklength=chunklength,
            checkpoint_every=checkpoint_every,
            callbacks=callbacks,
            callback_state=callback_state,
            compiled_chunk=compiled_chunk,
        )

    def _source_value(self, collection, name, parameters, prepared, states, stimulus):
        binding = self._bindings[collection, name]
        if binding is None:
            plan = self.synapses[name]
            return (stimulus.spike_gate if plan._diff_spiking else stimulus.spikes).to(
                device=plan.source_device, dtype=plan.source_dtype
            )
        population, source = binding
        value = source.resolve(
            parameters["populations"][population],
            prepared["populations"][population],
            states[population],
        )
        plan = (
            self.synapses[name]
            if collection == "synapses"
            else self.continuous_synapses[name]
        )
        return value.to(device=plan.source_device, dtype=plan.source_dtype)

    def _step_values(
        self,
        parameters,
        prepared,
        state,
        *,
        netstim_draw=None,
        inputs=None,
        extra=None,
        intra=None,
    ):
        for label, values in (("inputs", inputs), ("extra", extra), ("intra", intra)):
            if values is not None and (
                not isinstance(values, Mapping)
                or not set(values) <= set(self.populations)
            ):
                raise ValueError(
                    f"{label} must map known Population names to explicit inputs"
                )
        inputs, extra, intra = inputs or {}, extra or {}, intra or {}
        clock = state["clock"]
        time = (
            clock["origin"] + clock["step"].to(clock["origin"].dtype) * self.dt
        ).detach()
        states = {
            name: self._snap_population_clock(value, time)
            for name, value in state["populations"].items()
        }
        next_stim, stimulus = None, None
        if self.netstim is None:
            if netstim_draw is not None:
                raise ValueError("A Network without NetStim requires netstim_draw=None")
        else:
            stim = parameters["netstim"]
            next_stim, stimulus = self.netstim.step(
                stim["parameters"],
                prepared["netstim"],
                state["netstim"],
                time,
                netstim_draw,
                bptt=self.training,
                schedule=stim["schedule"],
            )
        scratch = {}
        for population, target in self._reset_targets:
            plan = self.populations[population]
            plan._validate_receiver_hooks(target, continuous=True)
            states[population], local = plan._reset_continuous_inputs_values(
                states[population], (target,)
            )
            scratch.setdefault(population, {}).update(local)
        continuous = {}
        for name, plan in self.continuous_synapses.items():
            value, queue = (
                parameters["continuous_synapses"][name],
                state["continuous_synapses"][name],
            )
            post, target = self._targets["continuous_synapses", name]
            receiver = self.populations[post]
            if plan.has_connections and plan.receive_before_source:
                _, payload, mask = plan._read_due_values(queue)
                states[post], scratch[post] = receiver._continuous_receive_values(
                    parameters["populations"][post],
                    prepared["populations"][post],
                    states[post],
                    scratch[post],
                    target,
                    payload,
                    self._continuous_inputs[name],
                    plan.reduce,
                    mask,
                )
            source = self._source_value(
                "continuous_synapses", name, parameters, prepared, states, stimulus
            )
            queue, payload, mask = plan._advance_values(
                value["weights"],
                queue,
                source,
                transform_parameters=value["transform_parameters"],
            )
            if plan.has_connections and not plan.receive_before_source:
                states[post], scratch[post] = receiver._continuous_receive_values(
                    parameters["populations"][post],
                    prepared["populations"][post],
                    states[post],
                    scratch[post],
                    target,
                    payload,
                    self._continuous_inputs[name],
                    plan.reduce,
                    mask,
                )
            continuous[name] = queue
        events = {}
        for name, plan in self.synapses.items():
            value = parameters["synapses"][name]
            queue, payload = plan.read_due(
                state["synapses"][name],
                weights=value["weights"],
                delays=value["delays"],
            )
            post, target = self._targets["synapses", name]
            receiver = self.populations[post]
            receiver._validate_receiver_hooks(target)
            states[post] = receiver._receive_values(
                parameters["populations"][post],
                prepared["populations"][post],
                states[post],
                target,
                payload,
            )
            source = self._source_value(
                "synapses", name, parameters, prepared, states, stimulus
            )
            events[name] = plan.advance(
                value["weights"],
                queue,
                source,
                delays=value["delays"],
                schedule=value["schedule"],
            )
        aux = {}
        for name, plan in self.populations.items():
            selected = inputs.get(name, StepInput())
            if not isinstance(selected, StepInput):
                raise TypeError("Network inputs must contain StepInput values")
            ve, injection = selected.ve, intra.get(name, selected.intra)
            if any(
                value is not None and not torch.is_tensor(value)
                for value in (ve, injection)
            ):
                raise TypeError(
                    "Explicit extracellular and intracellular step inputs must be Tensors"
                )
            rollout = RolloutInput(
                ve=None if ve is None else ve.unsqueeze(0),
                intra=None if injection is None else injection.unsqueeze(0),
            )
            states[name], aux[name] = plan._rollout_values(
                parameters["populations"][name],
                prepared["populations"][name],
                states[name],
                rollout,
                steps=1,
                extra=extra.get(name),
            )
        step = clock["step"] + 1
        next_time = (
            clock["origin"] + step.to(clock["origin"].dtype) * self.dt
        ).detach()
        states = {
            name: self._snap_population_clock(value, next_time)
            for name, value in states.items()
        }
        return {
            "populations": states,
            "synapses": events,
            "continuous_synapses": continuous,
            "netstim": next_stim,
            "clock": {
                "t": next_time,
                "origin": clock["origin"],
                "step": step,
                "duration_remainder": clock["duration_remainder"],
            },
        }, {"populations": aux, "netstim": stimulus}

    @staticmethod
    def _snap_population_clock(state, time):
        result, clock = dict(state), dict(state["clock"])
        value = clock["t"]
        clock["t"] = time.to(device=value.device, dtype=value.dtype).reshape(
            value.shape
        )
        result["clock"] = clock
        return result

    def step(
        self,
        parameters,
        prepared,
        state,
        *,
        netstim_draw=None,
        inputs=None,
        extra=None,
        intra=None,
    ):
        """Advance one complete native-ordered timestep with reusable preparation."""
        self._validate_source()
        self._validate_parameters(parameters)
        self._validate_state(state)
        values = self._validate_prepared(prepared, parameters)
        with nullcontext() if self.training else torch.no_grad():
            return self._step_values(
                parameters,
                values,
                state,
                netstim_draw=netstim_draw,
                inputs=inputs,
                extra=extra,
                intra=intra,
            )

    def prepare_and_step(
        self,
        parameters,
        constants,
        state,
        *,
        netstim_draw=None,
        inputs=None,
        extra=None,
        intra=None,
    ):
        """Derive coefficients and execute one pure, fullgraph-compatible timestep."""
        self._validate_source()
        self._validate_parameters(parameters)
        _validate_tree(constants, self._constant_schema, "constants")
        self._validate_state(state)
        with nullcontext() if self.training else torch.no_grad():
            values = self._prepare_values(parameters, constants)
            return self._step_values(
                parameters,
                values,
                state,
                netstim_draw=netstim_draw,
                inputs=inputs,
                extra=extra,
                intra=intra,
            )

    def commit_state_(self, network, state):
        """Stage and atomically publish complete runtime carry to a compatible graph."""
        if torch.compiler.is_compiling():
            raise FunctionalizationError("Commit Network state outside torch.compile.")
        self._validate_source()
        self._validate_compatible(network)
        self._validate_state(state)
        clock = state["clock"]
        if (
            not bool(torch.isfinite(clock["origin"]))
            or bool(clock["step"] < 0)
            or not bool(torch.isfinite(clock["duration_remainder"]))
            or bool(clock["duration_remainder"] < 0)
            or not torch.equal(
                clock["t"],
                clock["origin"] + clock["step"].to(clock["origin"].dtype) * self.dt,
            )
        ):
            raise ValueError(
                "Network checkpoint clock is not a finite coherent step boundary"
            )
        for name, population_state in state["populations"].items():
            time = population_state["clock"]["t"]
            expected = (
                clock["t"].to(device=time.device, dtype=time.dtype).reshape(time.shape)
            )
            if not torch.equal(time, expected):
                raise ValueError(
                    f"Population {name!r} clock does not match the Network step boundary"
                )
        staged = _deepcopy_execution_module(network)
        for name, plan in self.populations.items():
            plan.commit_state_(staged.populations[name], state["populations"][name])
        for name, plan in self.synapses.items():
            plan.commit_state_(staged.synapses[name], state["synapses"][name])
        for name, plan in self.continuous_synapses.items():
            plan.commit_state_(
                staged.continuous_synapses[name], state["continuous_synapses"][name]
            )
        if self.netstim is not None:
            self.netstim.commit_state_(staged.netstim, state["netstim"])
        staged._buffers["t"] = clock["t"].clone()
        staged._buffers["_clock_origin"] = clock["origin"].clone()
        staged._buffers["_clock_step"] = clock["step"].clone()
        staged._buffers["_duration_remainder"] = clock["duration_remainder"].clone()
        checkpoint = staged.state_dict_for_checkpoint()
        network.restore_dict_from_checkpoint(checkpoint)
        # SCNV's native checkpoint omits its exposed voltage. The Population
        # adapter has already staged/validated that separate functional carry.
        for name in self.populations:
            if isinstance(network.populations[name].integrator, _bwd_euler_sc_skip):
                network.populations[name]._buffers["v"] = staged.populations[
                    name
                ]._buffers["v"]
        return network


def make_functional_network(network):
    """Lower an initialized standard Network and extract independent explicit tensors."""
    plan = FunctionalNetwork(network)
    return plan, plan.extract()


__all__ = ["FunctionalNetwork", "NetworkTensors", "make_functional_network"]
