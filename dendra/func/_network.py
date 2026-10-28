"""Pure execution composed from an initialized standard Network's components."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from contextlib import nullcontext
from typing import NamedTuple

import torch
from torch.utils import _pytree as pytree

from dendra.models.integrators import _bwd_euler_sc_skip
from dendra.models.networks.net import (
    Network,
    _network_step_schedule,
)
from dendra.models.networks.net import step as _native_step

from . import _network_parameters
from ._connection_parameters import ConnectionParameterPlan
from ._connection_sampling import ConnectionSamplingPlan
from ._continuous import FunctionalContinuousCon
from ._continuous_transform import _named_tensors
from ._initialization_sources import (
    initialization_source_ownership_failures,
    initialization_source_roots,
    initialization_source_runtime_buffer_ids,
    network_initialization_source_alias_signature,
)
from ._mechanism_delays import network_mechanism_delay_alias_signature
from ._mechanism_injections import network_mechanism_injection_alias_signature
from ._netcon import FunctionalNetCon, _snapshot
from ._netstim import FunctionalNetStim, NetStimDraw
from ._netstim_parameters import NetStimParameterPlan
from ._network_bindings import AliasSnapshots, SourceBinding
from ._network_initialization import (
    NetworkInitializationCache,
    NetworkInitializationInput,
)
from ._population import (
    _deepcopy_execution_module,
    _subclass_contract_failures,
    _subclass_contract_matches,
    _subclass_slot_contract,
    _subclass_slot_contract_failures,
    _subclass_slot_contract_matches,
    make_functional,
)
from ._schedules import TensorScheduleTensors, _TensorSchedulePlan
from ._tables import network_table_alias_signature
from ._tree_paths import _get_path as _at
from ._types import FunctionalizationError, RolloutInput, StepInput


class NetworkTensors(NamedTuple):
    """Aggregate parameters, constants, and complete-boundary carry.

    Extraction produces independent snapshots retaining genuine shared
    parameter identities. Fresh initialization returns the resulting raw parameters,
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

    @property
    def trainable_parameters(self):
        """Return named leaves currently requiring gradients, including shared aliases.

        Deduplicate values by identity before passing them to an optimizer.
        """
        return {
            name: value
            for name, value in self.named_parameters.items()
            if value.requires_grad
        }

    def find_parameters(self, query):
        """Return parameter leaves matching a substring, glob, or regex query."""
        return _network_parameters.find_parameters(self.parameters, query)

    def parameter_name(self, query):
        """Resolve one exact, partial, glob, or regex parameter-name match."""
        return _network_parameters.parameter_name(self.parameters, query)

    def get_parameter(self, query):
        """Return the leaf selected by one unambiguous parameter query."""
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
        "_refresh_compile_config_from_ctx",
        "build",
        "init_synapses",
        "initialize_pops_from_state_cache",
        "initialize_synapses_from_state_cache",
        "_reset_runtime_clock",
        "_clear_duration_remainder",
        "_set_duration_remainder",
    )
}
_NETWORK_SUBCLASS_METHODS = {
    name: inspect.getattr_static(Network, name)
    for name in dict.fromkeys(
        (
            *_NETWORK_METHODS,
            *_NETWORK_INITIALIZATION_METHODS,
            "train",
            "eval",
            "devices",
            "dtypes",
            "device",
            "dtype",
            "_device_signature",
            "_require_mode_runtime_ready",
            "_validate_population_runtime_contracts",
            "_bind_runtime_clocks",
            "_reanchor_runtime_clock_from_time",
            "build_synapses",
            "build_continuous_synapses",
            "populations_state_dict_for_checkpoint",
            "netcons_state_dict_for_checkpoint",
            "netstim_state_dict_for_checkpoint",
            "checkpoint_structure",
            "_restore_dict_from_checkpoint_unchecked",
            "_rebuild_loaded_population_workspaces",
            "__getattribute__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
            "__deepcopy__",
            "__getstate__",
            "__setstate__",
        )
    )
    if hasattr(Network, name)
}
_COLLECTIONS = ("populations", "synapses", "continuous_synapses", "continuous_targets")
_CACHES = (
    "_population_step_items",
    "_synapse_step_items",
    "_continuous_synapse_step_items",
    "_continuous_target_step_items",
)
_NETWORK_SUBCLASS_SLOTS = _subclass_slot_contract(
    Network,
    (
        *_COLLECTIONS,
        *_CACHES,
        "netstim",
        "dt",
        "t",
        "training",
        "built",
        "_mode_requires_initialize",
        "_step",
        "_step_train",
        "_step_eval",
        "_runtime_clock",
        "_clock_origin",
        "_clock_step",
        "_duration_remainder",
        "_state_cache",
        "_syn_cache",
    ),
)


def _eligible(network):
    if not isinstance(network, Network):
        raise FunctionalizationError("Expected an initialized Network.")
    if type(network) is not Network:
        failures = _subclass_contract_failures(
            network, Network, _NETWORK_SUBCLASS_METHODS
        )
        failures.extend(
            _subclass_slot_contract_failures(network, Network, _NETWORK_SUBCLASS_SLOTS)
        )
        if failures:
            raise FunctionalizationError("\n".join(failures))
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
    from ._custom_material_processes import custom_material_constant_slots
    from ._torch_parametrizations import buffer_slots as torch_buffer_slots

    custom_config = [
        (name, slot, value)
        for name, population in network.populations.items()
        for process in population.mech.material_processes.values()
        for slot, value in custom_material_constant_slots(process)
    ]
    custom_config.extend(
        (name, f"torch_parametrizations.{slot}", value)
        for name, population in network.populations.items()
        for slot, value in torch_buffer_slots(population)
    )
    network_parameters = (
        {id(value) for value in network.parameters()} if custom_config else set()
    )
    runtime_buffers = (
        initialization_source_runtime_buffer_ids(network) if custom_config else set()
    )
    for name, population in network.populations.items():
        slots = [(slot, value) for owner, slot, value in custom_config if owner == name]
        if not slots:
            continue
        local_parameters = {id(value) for value in population.parameters()}
        for slot, value in slots:
            kind = (
                "torch parametrization"
                if slot.startswith("torch_parametrizations.")
                else "custom MaterialProcess"
            )
            if id(value) in runtime_buffers:
                raise FunctionalizationError(
                    f"Population {name!r} {kind} config "
                    f"buffer {slot!r} aliases runtime state, geometry, or a "
                    "reconstructed parameter; use an independent config "
                    "buffer or a shared registered raw Parameter."
                )
            if id(value) in network_parameters - local_parameters:
                raise FunctionalizationError(
                    f"Population {name!r} {kind} config "
                    f"buffer {slot!r} aliases a raw parameter registered "
                    "only outside that Population. Cross-Population "
                    "buffer/parameter aliases are currently unsupported; "
                    "register the same Parameter on the process as a "
                    "Parameter attribute to preserve its shared binding."
                )
    for name in ("populations", "synapses", "continuous_synapses"):
        owners = tuple(getattr(network, name).values())
        if len({id(owner) for owner in owners}) != len(owners):
            raise FunctionalizationError(
                "Functional Network requires distinct mutable component owners."
            )
    population_modules = set()
    mutable_population_modules = set()
    for population in network.populations.values():
        owned = {id(module) for module in population.modules()}
        # Authored initialization sources and injection Waveforms are audited
        # by each Population plan and get isolated execution copies. Their
        # parameters may be shared while runtime owners remain disjoint.
        from ._mechanism_injections import mechanism_injection_owners

        waveform_roots = {
            id(waveform)
            for _path, owner in mechanism_injection_owners(population)
            for waveform in owner.injected_waveforms
        }
        recipe_roots = waveform_roots | {
            id(source) for _path, source in initialization_source_roots(population)
        }
        from ._torch_parametrizations import registries as torch_registries

        recipe_roots.update(
            id(registry) for _path, _owner, registry in torch_registries(population)
        )
        mutable_owned = set()

        def visit(module):
            if id(module) in recipe_roots or id(module) in mutable_owned:
                return
            mutable_owned.add(id(module))
            for child in module.children():
                visit(child)

        visit(population)
        shared = population_modules & owned
        from dendra.models.rng import RNGModule

        rng_modules = {
            id(module) for module in population.modules() if type(module) is RNGModule
        }
        shared_mutable = shared & (mutable_population_modules | mutable_owned)
        if shared_mutable - rng_modules:
            raise FunctionalizationError(
                "Functional Network Populations must own distinct modules; "
                "share raw parameter Tensors instead."
            )
        population_modules.update(owned)
        mutable_population_modules.update(mutable_owned)
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

    Parameters/constants follow the component adapters' conventions. Population
    and connection parameters retain their raw source coordinates and shared
    identities, including NetStim renewal sources. Snapshot extraction is detached and does
    not perform fresh initialization. ``commit_state_`` publishes runtime carry
    only; align native parameters/schedules separately before continuation.
    Topology, source bindings, mode, timestep, and native phase ordering are
    frozen. Canonical single-device Networks use dense, source-history training,
    packed-history, or sparse-calendar inference event runtimes and integer
    continuous delays supported by the component plans. ``dn.compile`` uses this
    plan for snapshot or fresh Network initialization; the lower-level methods support
    explicit preparation and functional transforms.

    ``extra`` maps Population names to extracellular stimulation bindings,
    each using one ``(field, Waveform)`` pair or a sequence of contacts. Fields
    and waveform leaves appear in the selected Population's explicit
    ``stimulation.extra.*`` parameters. Shared authored Tensor objects retain
    their aliases across the Network. Bound stimulation uses the Network clock
    before each timestep; explicit ``ve`` inputs for the same Population cannot
    be combined with it.

    ``pending_capacity`` bounds each sparse pending calendar independently of
    delay depth. Supply one positive integer or a connection-name mapping.
    Active-calendar defaults cover at least 16 entries, twice the edge count,
    and the imported pending count. Intrinsic-only packed plans with no pending
    traffic omit unused calendar storage. Host execution checks overflow before
    returning a result.
    """

    def __init__(self, network, *, extra=None, pending_capacity=None):
        _eligible(network)
        self._source_type = type(network)
        if extra is not None and not isinstance(extra, Mapping):
            raise TypeError("extra must map Population names to stimulation bindings")
        extra = {} if extra is None else dict(extra)
        unknown = extra.keys() - network.populations.keys()
        if unknown:
            raise ValueError(
                f"extra names unknown Populations: {sorted(unknown, key=str)!r}"
            )
        if isinstance(pending_capacity, Mapping):
            capacities = dict(pending_capacity)
            unknown = capacities.keys() - network.synapses.keys()
            if unknown:
                raise ValueError(
                    f"pending_capacity names unknown event connections: {sorted(unknown, key=str)!r}"
                )
        else:
            capacities = {name: pending_capacity for name in network.synapses}
        selected_capacities = (
            capacities.values()
            if isinstance(pending_capacity, Mapping)
            else (pending_capacity,)
        )
        for capacity in selected_capacities:
            if capacity is not None and (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity < 1
            ):
                raise ValueError("pending_capacity must contain positive integers")
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
            plan, packet = make_functional(
                population,
                dt=self.dt,
                extra=extra.get(name),
                _bind_owned_cache=not bool(network._state_cache),
                _retain_initialization_materializations=True,
                _network_stimulation_rng_scope=True,
            )
            if network._state_cache:
                plan._enable_cache_materializations()
            plan._enable_retained_materializations()
            self.populations[name] = plan
            initialization[name] = packet.initialization
        from ._network_rng import NetworkRNGPlan, NetworkStimulationRNGPlan

        self._rng_plan = NetworkRNGPlan(self, network)
        from ._poisson_stimulation import NetworkStimulationPlan

        self._extra_order = tuple(extra)
        self._stimulation_plan = NetworkStimulationPlan(self)
        self._stimulation_rng_plan = NetworkStimulationRNGPlan(self)
        self._source_table_aliases = network_table_alias_signature(network)
        self._source_mechanism_delay_aliases = network_mechanism_delay_alias_signature(
            network
        )
        self._source_mechanism_injection_aliases = (
            network_mechanism_injection_alias_signature(network)
        )
        self._source_initialization_source_aliases = (
            network_initialization_source_alias_signature(network)
        )
        self.synapses = {
            name: FunctionalNetCon(connection, pending_capacity=capacities.get(name))
            for name, connection in self._schedule.synapses
        }
        self.continuous_synapses = {
            name: FunctionalContinuousCon(connection)
            for name, connection in self._schedule.continuous_synapses
        }
        self._connection_parameter_plans = {
            (collection, name): ConnectionParameterPlan(connection)
            for collection in ("synapses", "continuous_synapses")
            for name, connection in getattr(network, collection).items()
        }
        self._connection_sampling_plan = ConnectionSamplingPlan(
            self._connection_parameter_plans
        )
        for plan in self._connection_parameter_plans.values():
            plan._validate_runtime_ownership(
                plan._source, owners=network.populations.values()
            )
        self.netstim = (
            None if network.netstim is None else FunctionalNetStim(network.netstim)
        )
        self._netstim_parameter_plan = (
            None
            if network.netstim is None
            else NetStimParameterPlan(
                network.netstim, owners=network.populations.values()
            )
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
        self._source_parameter_constant_aliases = (
            self._native_parameter_constant_aliases(network)
        )
        self._population_cache_plans = {}
        self._cache_enabled = bool(network._state_cache)
        self._cache_error = None
        initialization_cache = None
        if self._cache_enabled:
            try:
                from ._population_cache import PopulationCachePlan

                for name, plan in self.populations.items():
                    self._population_cache_plans[name] = PopulationCachePlan(
                        plan, network._state_cache[name]
                    )
                saved = network._syn_cache or {}
                if not isinstance(saved, Mapping):
                    raise TypeError("Network connection state cache must be a mapping")
                events = (
                    saved.get("event", {})
                    if "event" in saved or "continuous" in saved
                    else saved
                )
                analog = saved.get("continuous", {})
                initialization_cache = NetworkInitializationCache(
                    populations={
                        name: plan.default
                        for name, plan in self._population_cache_plans.items()
                    },
                    synapses={
                        name: plan.normalize_cache(events[name])
                        for name, plan in self.synapses.items()
                        if name in events
                    },
                    continuous_synapses={
                        name: plan.normalize_cache(analog[name])
                        for name, plan in self.continuous_synapses.items()
                        if name in analog
                    },
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                self._cache_error = str(error)
        self._cache_schema = _schema(initialization_cache)
        from ._network_initialization import cache_alias_bindings

        self._cache_alias_bindings = cache_alias_bindings(self)
        from ._network_initialization import (
            population_cache_alias_bindings,
            population_materialization_alias_bindings,
        )

        self._owned_cache_alias_plans, self._owned_cache_alias_error = (
            population_cache_alias_bindings(self)
        )
        self._materialization_alias_plans, self._materialization_alias_error = (
            population_materialization_alias_bindings(self)
        )
        self._initialization_inputs = NetworkInitializationInput(
            populations=initialization,
            t=torch.zeros_like(tensors.state["clock"]["t"]),
            cache=initialization_cache,
        )
        from ._network_initialization import (
            initialization_alias_failures,
            nonpersistent_material_cache_failures,
        )

        self._nonpersistent_material_cache_failures = (
            nonpersistent_material_cache_failures(self)
        )

        runtime_buffer_ids = initialization_source_runtime_buffer_ids(network)
        self._initialization_alias_failures = (
            *initialization_alias_failures(self),
            *(
                f"Population {name!r}: {failure}"
                for name, plan in self.populations.items()
                if not plan._owned_cache_bound
                for failure in initialization_source_ownership_failures(
                    plan._population, runtime_buffer_ids=runtime_buffer_ids
                )
            ),
        )

        from ._network_draws import NetworkRuntimeDrawPlan

        self._runtime_draw_plan = NetworkRuntimeDrawPlan(self)

    @property
    def initialization_inputs(self):
        """Independent initialization-only defaults with editable Tensor leaves."""
        return self._initialization_inputs

    def _validate_initialization_source(self):
        self._validate_source()
        network = self._source
        if self._cache_error is not None:
            raise FunctionalizationError(
                f"Network state cache cannot be restored: {self._cache_error}"
            )
        if self._owned_cache_alias_error is not None:
            raise FunctionalizationError(self._owned_cache_alias_error)
        if self._materialization_alias_error is not None:
            raise FunctionalizationError(self._materialization_alias_error)
        if self._nonpersistent_material_cache_failures:
            raise FunctionalizationError(
                "\n".join(self._nonpersistent_material_cache_failures)
            )
        if not self._cache_enabled and self._initialization_alias_failures:
            raise FunctionalizationError("\n".join(self._initialization_alias_failures))
        for name, expected in _NETWORK_INITIALIZATION_METHODS.items():
            if not _subclass_contract_matches(network, Network, {name: expected}):
                raise FunctionalizationError(
                    f"Fresh Network initialization requires its canonical {name} adapter"
                )

    def _validate_initialization_cache(self, cache):
        if self._cache_enabled and cache is None:
            raise ValueError(
                "This Network initialization requires its captured cache packet"
            )
        if not self._cache_enabled and cache is not None:
            raise ValueError("This Network plan has no bound initialization cache")
        _validate_tree(cache, self._cache_schema, "initialization.cache")

    def initialize(self, parameters, constants, state, inputs, *, netstim_draw=None):
        """Initialize current explicit parameters and carry in native order.

        Population inputs and the starting clock are explicit. Existing carry
        supplies NetStim RNG/seeder/calendar positions; delivery and detector
        carry is reset. Supply an explicit NetStim initialization draw when
        present. A bound Network-wide cache restores detached Population
        materializations and connection history. Without that cache, each
        Population restores its own bound cache or runs fresh initialization
        in authored order. Shared raw writes reach later owners, while each
        Population retains the effective buffers and solver workspaces prepared
        at its own boundary. Resulting parameters and constants accompany state
        in :class:`NetworkTensors`; keep all three for explicit continuation.
        """
        from ._network_initialization import initialize

        return initialize(
            self, parameters, constants, state, inputs, netstim_draw=netstim_draw
        )

    def sample_initialization_draw(self, parameters, state, *, constants=None):
        """Sample one initialization packet from independent explicit RNG carry.

        Supply ``constants`` when a renewal transform reads registered config
        buffers, including values restored by Population initialization caches.
        """
        from ._network_bindings import _canonicalize_mapping_tree

        (parameters, state, constants), _ = _canonicalize_mapping_tree(
            (parameters, state, constants)
        )
        self._validate_initialization_source()
        self._validate_parameters(parameters)
        self._validate_state(state)
        if self.netstim is None:
            return None
        if constants is None:
            if self._constant_schema["netstim"]["sources"]:
                raise ValueError(
                    "Supply constants when sampling NetStim sources with configuration buffers."
                )
            source_constants = {}
        else:
            _validate_tree(constants, self._constant_schema, "constants")
            source_constants = self._bind_parameter_constants(parameters, constants)[
                "netstim"
            ]["sources"]
        effective = self._netstim_parameter_plan.materialize(
            parameters["netstim"]["parameters"], source_constants
        )
        return self.netstim.sample_initialization_draw(effective, state["netstim"])

    def sample_population_initialization_draws(self, parameters, state):
        """Draw Population initialization packets without changing the carry.

        Assign each returned packet to its Population's
        ``InitializationInput.random_draw`` before calling :meth:`initialize`.
        Network-wide cache restoration consumes no Population draws. NetStim
        has its separate :meth:`sample_initialization_draw` packet.
        """
        from ._network_bindings import _canonicalize_mapping_tree

        (parameters, state), _ = _canonicalize_mapping_tree((parameters, state))
        self._validate_initialization_source()
        self._validate_parameters(parameters)
        self._validate_state(state)
        return self._rng_plan.sample_init_packets(parameters, state)

    def sample_population_stimulation_initialization_draws(self, parameters, state):
        """Draw replayable Population waveform-initialization packets.

        Assign each packet to ``InitializationInput.stimulation``. When model
        randomness shares a generator with a waveform, obtain the matching
        ``random_draw`` packets with :meth:`sample_population_initialization_draws`
        from the same incoming state.
        """
        from ._network_bindings import _canonicalize_mapping_tree

        (parameters, state), _ = _canonicalize_mapping_tree((parameters, state))
        self._validate_initialization_source()
        self._validate_parameters(parameters)
        self._validate_state(state)
        if self._stimulation_rng_plan.enabled:
            return self._stimulation_rng_plan.sample_initialization_packets(state)[1]
        return {
            name: None
            if self._cache_enabled or plan._owned_cache_bound
            else plan.sample_stimulation_initialization_draw(
                self._expand_population_state(name, state["populations"][name], state)
            )
            for name, plan in self.populations.items()
        }

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
        if type(network) is not self._source_type:
            raise FunctionalizationError(
                "The Network configuration changed; initialize and lower again."
            )
        hooks = (
            _NETWORK_METHODS if type(network) is Network else _NETWORK_SUBCLASS_METHODS
        )
        if not _subclass_contract_matches(network, Network, hooks) or not (
            _subclass_slot_contract_matches(network, Network, _NETWORK_SUBCLASS_SLOTS)
        ):
            raise FunctionalizationError("Network adapters changed; lower again.")
        self._stimulation_plan.validate_source()
        if (
            not network.built
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
        if any(
            getattr(network, name) is not _native_step
            for name in ("_step", "_step_train", "_step_eval")
        ):
            raise FunctionalizationError(
                "Network timestep phases changed; lower again."
            )
        self._rng_plan.validate_source()
        self._stimulation_rng_plan.validate_source()
        for plan in (
            *self.populations.values(),
            *self.synapses.values(),
            *self.continuous_synapses.values(),
        ):
            plan._validate_source()
        for plan in self._connection_parameter_plans.values():
            plan.validate_source()
        if network_table_alias_signature(network) != self._source_table_aliases:
            raise FunctionalizationError(
                "The Network TABLE tensor aliases changed; lower again."
            )
        if (
            network_mechanism_delay_alias_signature(network)
            != self._source_mechanism_delay_aliases
        ):
            raise FunctionalizationError(
                "The Network mechanism delay tensor aliases changed; lower again."
            )
        if (
            network_mechanism_injection_alias_signature(network)
            != self._source_mechanism_injection_aliases
        ):
            raise FunctionalizationError(
                "The Network mechanism injection tensor aliases changed; lower again."
            )
        if (
            network_initialization_source_alias_signature(network)
            != self._source_initialization_source_aliases
        ):
            raise FunctionalizationError(
                "The Network initialization source aliases changed; lower again."
            )
        if self.netstim is not None:
            self.netstim._validate_source()
            self._netstim_parameter_plan.validate_source()

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
        self.check_overflow(state)

    def _clone_state_tree(self, state):
        """Own state leaves in the frozen schema's mapping and tuple order."""

        def clone(value, schema):
            if isinstance(schema, _TensorSchema):
                return value.clone(memory_format=torch.contiguous_format)
            if isinstance(schema, Mapping):
                return {
                    name: clone(value[name], child) for name, child in schema.items()
                }
            if isinstance(schema, tuple):
                values = tuple(clone(item, child) for item, child in zip(value, schema))
                return type(schema)(*values) if hasattr(schema, "_fields") else values
            return None

        return clone(state, self._state_schema)

    def _extract(self, network):
        snapshots = AliasSnapshots()
        parameters = {
            "populations": {},
            "synapses": {},
            "continuous_synapses": {},
            "netstim": None,
        }
        constants = {
            "populations": {},
            "synapses": {},
            "continuous_synapses": {},
            "netstim": None,
        }
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
                plan._parameter_sources(population), packet.parameters
            )
            native_constants = plan._parametrization_constant_sources(population)
            constants["populations"][name] = snapshots.tree(
                native_constants, packet.constants
            )
            state["populations"][name] = self._stimulation_plan.strip(
                self._rng_plan.strip_population(packet.state)
            )
        for name, plan in self.synapses.items():
            connection = network.synapses[name]
            sources = self._connection_parameter_plans["synapses", name]
            source_parameters, source_constants = sources.extract(snapshots, connection)
            packet = plan.extract(connection)
            schedule = (
                None
                if packet.schedule is None
                else TensorScheduleTensors(**_TensorSchedulePlan._values(connection))
            )
            parameters["synapses"][name] = {
                **source_parameters,
                "schedule": snapshots.tree(schedule, packet.schedule),
            }
            constants["synapses"][name] = source_constants
            state["synapses"][name] = packet.state
        for name, plan in self.continuous_synapses.items():
            connection = network.continuous_synapses[name]
            sources = self._connection_parameter_plans["continuous_synapses", name]
            source_parameters, source_constants = sources.extract(snapshots, connection)
            packet = plan.extract(connection)
            native_transform = (
                _named_tensors(connection.transform)
                if isinstance(connection.transform, torch.nn.Module)
                else None
            )
            parameters["continuous_synapses"][name] = {
                **source_parameters,
                "transform_parameters": snapshots.tree(
                    native_transform, packet.transform_parameters
                ),
            }
            constants["continuous_synapses"][name] = source_constants
            state["continuous_synapses"][name] = packet.state
        if self.netstim is not None:
            source = network.netstim
            packet = self.netstim.extract(source)
            raw_parameters, source_constants = self._netstim_parameter_plan.extract(
                snapshots, source
            )
            schedule = (
                None
                if packet.schedule is None
                else TensorScheduleTensors(**_TensorSchedulePlan._values(source))
            )
            parameters["netstim"] = {
                "parameters": raw_parameters,
                "schedule": snapshots.tree(schedule, packet.schedule),
            }
            constants["netstim"] = snapshots.tree(
                {name: getattr(source, name) for name in packet.constants},
                packet.constants,
            )
            constants["netstim"]["sources"] = source_constants
            state["netstim"] = packet.state
        state["clock"] = {
            "t": _snapshot(network.t),
            "origin": _snapshot(network._clock_origin),
            "step": _snapshot(network._clock_step),
            "duration_remainder": _snapshot(network._duration_remainder),
        }
        sampling = self._connection_sampling_plan.extract(network)
        if sampling is not None:
            state["connection_sampling"] = sampling
        if self._rng_plan.enabled:
            state["rng"] = self._rng_plan.extract(network)
        if self._stimulation_plan.enabled:
            state["stimulation_rng"] = self._stimulation_rng_plan.strip_pool(
                self._stimulation_plan.extract(network)
            )
            state = self._connection_rng_from_stimulation(state)
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
        self._stimulation_plan.validate_compatible_network(network)
        self._rng_plan.validate_compatible_network(network)
        self._stimulation_rng_plan.validate_compatible_network(network)
        if (
            network_initialization_source_alias_signature(network)
            != self._source_initialization_source_aliases
        ):
            raise FunctionalizationError(
                "Network initialization source sharing does not match this plan."
            )
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
        if self.netstim is not None:
            self._netstim_parameter_plan.validate_source(network.netstim)
            self._netstim_parameter_plan._validate_runtime_ownership(
                network.netstim, owners=network.populations.values()
            )
        for (collection, name), plan in self._connection_parameter_plans.items():
            connection = getattr(network, collection)[name]
            plan.validate_source(connection)
            plan._validate_runtime_ownership(
                connection, owners=network.populations.values()
            )
        if (
            self._native_parameter_constant_aliases(network)
            != self._source_parameter_constant_aliases
        ):
            raise FunctionalizationError(
                "Network parameter/config source sharing does not match this plan"
            )

    def _native_parameter_constant_aliases(self, network):
        """Inspect exact source bindings, including native cache detach views."""
        parameters, constants = {}, {}
        for name, plan in self.populations.items():
            population = network.populations[name]
            parameters["populations", name] = plan._parameter_sources(population)
            constants["populations", name] = plan._parametrization_constant_sources(
                population
            )
        for (collection, name), plan in self._connection_parameter_plans.items():
            connection = getattr(network, collection)[name]
            sources = plan.parameter_sources(connection)
            if collection == "synapses":
                sources = {
                    **sources,
                    "schedule": TensorScheduleTensors(
                        **_TensorSchedulePlan._values(connection)
                    ),
                }
            else:
                sources = {
                    **sources,
                    "transform_parameters": _named_tensors(connection.transform)
                    if isinstance(connection.transform, torch.nn.Module)
                    else None,
                }
            parameters[collection, name] = sources
            constants[collection, name] = plan.constant_sources(connection)
        if self.netstim is not None:
            source = network.netstim
            plan = self._netstim_parameter_plan
            parameters["netstim"] = {
                "parameters": plan.parameter_sources(source),
                "schedule": TensorScheduleTensors(
                    **_TensorSchedulePlan._values(source)
                ),
            }
            constants["netstim"] = {
                "sources": plan.constant_sources(source),
                **{
                    name: getattr(source, name)
                    for name in ("max_spikes", "heap_times", "heap_row_offsets")
                },
            }

        def frame(value):
            # Native cache restoration can detach a registered buffer while
            # preserving its exact storage coordinates. That is still the
            # same source/config binding, unlike an independent clone or a
            # differently indexed view. Empty tensors have no storage identity.
            if value.layout != torch.strided or not value.numel():
                return ("identity", id(value))
            return (
                "storage",
                value.device,
                value.dtype,
                value.untyped_storage().data_ptr(),
                value.storage_offset(),
                tuple(value.shape),
                tuple(value.stride()),
                value.is_conj(),
                value.is_neg(),
            )

        readers = {}
        for path, value in _tensor_paths(constants):
            readers.setdefault(frame(value), []).append(path)
        return tuple(
            (path, tuple(readers[frame(value)]))
            for path, value in _tensor_paths(parameters)
            if frame(value) in readers
        )

    def prepare(self, parameters, constants):
        """Prepare reusable Population, connection, and NetStim values."""
        self._validate_source()
        self._validate_parameters(parameters)
        _validate_tree(constants, self._constant_schema, "constants")
        constants = self._bind_parameter_constants(parameters, constants)
        return {
            "populations": {
                name: plan.prepare(
                    parameters["populations"][name], constants["populations"][name]
                )
                for name, plan in self.populations.items()
            },
            "netstim": None
            if self.netstim is None
            else {
                "parameters": self._netstim_parameter_plan.prepare(
                    parameters["netstim"]["parameters"], constants["netstim"]["sources"]
                ),
                "constants": self._netstim_constants(constants),
            },
            "connections": {
                collection: {
                    name: plan.prepare(
                        parameters[collection][name], constants[collection][name]
                    )
                    for (kind, name), plan in self._connection_parameter_plans.items()
                    if kind == collection
                }
                for collection in ("synapses", "continuous_synapses")
            },
        }

    def _connection_values(self, collection, name, values):
        projection = getattr(self, collection)[name]
        if collection == "continuous_synapses" or not projection._diff_delays:
            steps = (values["delays"] / projection._dt).round().long()
            if collection == "synapses":
                steps = steps.clamp_min(1)
            valid = (steps == projection._delay_steps).all()
            message = "Connection delays change frozen integer routing; lower the Network again"
            if torch.compiler.is_compiling():
                torch._assert_async(valid, message)
            elif torch._C._are_functorch_transforms_active():
                # The predicate carries no gradient. Inspect every physical
                # lane without sending a BatchedTensor through _assert_async,
                # whose operator has no vmap batching rule.
                functorch = torch._C._functorch
                while functorch.is_functorch_wrapped_tensor(valid):
                    valid = functorch.get_unwrapped(valid)
                torch._assert_async(valid.all(), message)
            elif not bool(valid):
                raise FunctionalizationError(message)
            return {**values, "delays": None}
        return values

    def _bind_parameter_constants(self, parameters, constants):
        """Resolve shared config readers against their canonical raw input.

        A native Tensor can be a selectable input of one component and a
        registered config buffer of another. Parameters own that shared input;
        cloning or replacing their packet must update every config reader too.
        Keep the caller's constants untouched, including retained cache values.
        """
        from ._network_initialization import _replace_at

        for path, readers in self._parameter_constant_aliases:
            value = _at(parameters, path)
            for reader in readers:
                if _at(constants, reader) is not value:
                    constants = _replace_at(constants, reader, value)
        return constants

    @staticmethod
    def _prepared_constant(prepared, path):
        collection, *rest = path
        if collection == "populations":
            name, key = rest
            return prepared["populations"][name].constant_sources[key]
        if collection in {"synapses", "continuous_synapses"}:
            name, key = rest
            return prepared["connections"][collection][name].constants[key]
        if collection == "netstim":
            if rest[0] == "sources":
                return prepared["netstim"]["parameters"].constants[rest[1]]
            return prepared["netstim"]["constants"][rest[0]]
        raise FunctionalizationError("Prepared shared constant binding is invalid")

    def _materialize_connections(self, parameters, constants):
        constants = self._bind_parameter_constants(parameters, constants)
        return {
            collection: {
                name: self._connection_values(
                    collection,
                    name,
                    plan.materialize(
                        parameters[collection][name], constants[collection][name]
                    ),
                )
                for (kind, name), plan in self._connection_parameter_plans.items()
                if kind == collection
            }
            for collection in ("synapses", "continuous_synapses")
        }

    @staticmethod
    def _netstim_constants(constants):
        if constants["netstim"] is None:
            return None
        return {
            name: value
            for name, value in constants["netstim"].items()
            if name != "sources"
        }

    def _materialize_netstim_parameters(self, parameters, constants):
        if self.netstim is None:
            return None
        constants = self._bind_parameter_constants(parameters, constants)
        return self._netstim_parameter_plan.materialize(
            parameters["netstim"]["parameters"], constants["netstim"]["sources"]
        )

    def sample_step_draw(self, state):
        """Sample a replayable packet from an independent explicit RNG copy."""
        self._validate_source()
        self._validate_state(state)
        return (
            None
            if self.netstim is None
            else self.netstim.sample_step_draw(state["netstim"])
        )

    def sample_connection_initialization_draw(
        self, parameters, constants, state, *, reinit_weights=True, reinit_delays=True
    ):
        """Draw connection initialization inputs from independent explicit RNG.

        The returned packet follows native connection/block order. Sampling
        consumes no global Torch RNG and changes none of the supplied tensors.
        Population/cache writes that affect distribution parameters are applied
        before sampling by the runner's fresh-initialization host adapter.
        """
        from ._network_bindings import _canonicalize_mapping_tree

        if torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():
            raise FunctionalizationError(
                "Sample connection initialization draws outside torch.compile and functional transforms."
            )
        (parameters, constants, state), _ = _canonicalize_mapping_tree(
            (parameters, constants, state)
        )
        self._validate_initialization_source()
        self._validate_parameters(parameters)
        _validate_tree(constants, self._constant_schema, "constants")
        self._validate_state(state)
        constants = self._bind_parameter_constants(parameters, constants)
        return self._connection_sampling_plan.sample(
            parameters,
            constants,
            state,
            reinit_weights=reinit_weights,
            reinit_delays=reinit_delays,
        )

    def _prepare_values(self, parameters, constants):
        constants = self._bind_parameter_constants(parameters, constants)
        return {
            "populations": {
                name: plan._prepare_values(
                    parameters["populations"][name], constants["populations"][name]
                )
                for name, plan in self.populations.items()
            },
            "netstim": None
            if self.netstim is None
            else {
                "parameters": self._materialize_netstim_parameters(
                    parameters, constants
                ),
                "constants": self._netstim_constants(constants),
            },
            "connections": self._materialize_connections(parameters, constants),
        }

    def _validate_prepared(self, prepared, parameters):
        if (
            not isinstance(prepared, Mapping)
            or set(prepared) != {"populations", "netstim", "connections"}
            or not isinstance(prepared["populations"], Mapping)
            or set(prepared["populations"]) != set(self.populations)
            or not isinstance(prepared["connections"], Mapping)
            or set(prepared["connections"]) != {"synapses", "continuous_synapses"}
            or any(
                not isinstance(prepared["connections"][collection], Mapping)
                or set(prepared["connections"][collection])
                != set(getattr(self, collection))
                for collection in ("synapses", "continuous_synapses")
            )
        ):
            raise ValueError("prepared must match this Network's component plans")
        for path, readers in self._parameter_constant_aliases:
            value = _at(parameters, path)
            try:
                changed = any(
                    self._prepared_constant(prepared, reader) is not value
                    for reader in readers
                )
            except (AttributeError, KeyError, TypeError) as error:
                raise FunctionalizationError(
                    "Prepared shared constant binding is invalid"
                ) from error
            if changed:
                raise FunctionalizationError(
                    "Shared parameters changed after prepare(); call prepare() again"
                )
        netstim = prepared["netstim"]
        if self.netstim is None:
            if netstim is not None:
                raise ValueError(
                    "A Network without NetStim requires prepared.netstim=None"
                )
        else:
            if not isinstance(netstim, Mapping) or set(netstim) != {
                "parameters",
                "constants",
            }:
                raise ValueError(
                    "prepared.netstim must match this Network's source plan"
                )
            _validate_tree(
                netstim["constants"],
                {
                    name: value
                    for name, value in self._constant_schema["netstim"].items()
                    if name != "sources"
                },
                "prepared.netstim.constants",
            )
            netstim = {
                "parameters": self._netstim_parameter_plan.validate_prepared(
                    netstim["parameters"], parameters["netstim"]["parameters"]
                ),
                "constants": netstim["constants"],
            }
        populations = {
            name: plan._validate_prepared(
                prepared["populations"][name], parameters["populations"][name]
            )
            for name, plan in self.populations.items()
        }
        # Shared config inputs can require gradients even when the Population's
        # own raw parameter packet is unchanged. Rebuild those coefficients
        # for this invocation after validating the opaque preparation, so
        # independent backwards never reuse a previous materialization graph.
        shared_populations = {
            reader[1]
            for _path, readers in self._parameter_constant_aliases
            for reader in readers
            if reader[0] == "populations"
        }
        for name in shared_populations:
            populations[name] = self.populations[name]._prepare_values(
                parameters["populations"][name],
                prepared["populations"][name].constant_sources,
            )
        return {
            "populations": populations,
            "netstim": netstim,
            "connections": {
                collection: {
                    name: self._connection_values(
                        collection,
                        name,
                        plan.validate_prepared(
                            prepared["connections"][collection][name],
                            parameters[collection][name],
                        ),
                    )
                    for (kind, name), plan in self._connection_parameter_plans.items()
                    if kind == collection
                }
                for collection in ("synapses", "continuous_synapses")
            },
        }

    def make_callbacks(self, callbacks):
        """Bind qualified Network recordings or pure custom callbacks."""
        from ._network_callbacks import make_network_callbacks

        self._validate_source()
        return make_network_callbacks(self, callbacks)

    def sample_runtime_draws(self, state, *, steps=None):
        """Sample one canonical runtime packet or time-leading draw/RNG tape."""
        from ._network_bindings import _canonicalize_mapping_tree

        if torch.compiler.is_compiling():
            raise FunctionalizationError("Sample runtime draws outside torch.compile")
        (state,), _ = _canonicalize_mapping_tree((state,))
        self._validate_source()
        self._validate_state(state)
        return self._runtime_draw_plan.sample(state, steps=steps)

    def sample_draw_tape(self, state, steps):
        """Sample replayable NetStim draw/RNG packets outside compiled code."""
        from ._network_rollout import sample_draw_tape

        return sample_draw_tape(self, state, steps)

    def sample_population_noise_tapes(self, state, steps):
        """Sample per-Population NOISE draw/RNG tapes from independent carry.

        Assign each returned packet to its Population's ``RolloutInput.noise``.
        NetStim has its separate :meth:`sample_draw_tape` packet. Sampling
        follows the authored Population order and never changes the source
        Network or supplied state.
        """
        from ._network_bindings import _canonicalize_mapping_tree

        (state,), _ = _canonicalize_mapping_tree((state,))
        self._validate_source()
        self._validate_state(state)
        if self._stimulation_rng_plan.enabled:
            return self._stimulation_rng_plan.sample_tapes(state, steps)[0]
        return self._rng_plan.sample_noise_tapes(state, steps)

    def sample_population_stimulation_tapes(self, state, steps, provided=None):
        """Sample invocation-ordered Poisson tapes from explicit shared carry."""
        if not self._stimulation_plan.enabled:
            return dict.fromkeys(self.populations)
        self._validate_source()
        self._validate_state(state)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if self._stimulation_rng_plan.enabled:
            return self._stimulation_rng_plan.sample_tapes(
                state, steps, stimulation=provided
            )[1]
        return self._stimulation_plan.sample_tapes(state, steps, provided)

    def _expand_population_state(self, name, population_state, state):
        return self._stimulation_rng_plan.expand_population(
            name,
            population_state,
            state.get("rng", {}),
            state.get("stimulation_rng", {}),
        )

    def _connection_rng_from_stimulation(self, state, pool=None):
        if not self._stimulation_plan.enabled or "connection_sampling" not in state:
            return state
        pool = state["stimulation_rng"] if pool is None else pool
        rng = dict(state["connection_sampling"]["rng"])
        for canonical, device in self._stimulation_plan.default_streams.items():
            if device in rng:
                rng[device] = pool[canonical]
        return {**state, "connection_sampling": {"rng": rng}}

    def _stimulation_rng_after_connections(self, pool, rng):
        if not self._stimulation_plan.enabled:
            return pool
        return {
            **pool,
            **{
                canonical: rng[device]
                for canonical, device in self._stimulation_plan.default_streams.items()
                if device in rng
            },
        }

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
        draws=None,
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
            **({"draws": draws} if draws is not None else {}),
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
        draws=None,
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
            **({"draws": draws} if draws is not None else {}),
            constants=constants,
        )

    def compile_rollout_chunk(self, steps, *, execution="scan", **compile_options):
        """Build a scan, unrolled, or inference-only shared while-loop chunk."""
        from ._network_rollout import CompiledNetworkChunk

        return CompiledNetworkChunk(
            self, steps=steps, execution=execution, **compile_options
        )

    def bind(self, parameters, constants, *, compiled_chunk=None):
        """Bind raw inputs for eager steps or canonical host scheduling.

        Use ``bind(parameters, constants=constants)`` for the shared Population
        and Network binding contract. Positional constants are also supported.
        Tensor leaves stay live and differentiable. Network bindings retain raw
        constants, deriving fresh coefficients for every invocation instead of
        retaining graph-bearing materializations. Use ``chunk.bind`` for a
        compiled transition; both bindings work with ``dn.func.run``.
        """
        from ._network_binding import BoundNetwork

        return BoundNetwork(self, parameters, constants, compiled_chunk=compiled_chunk)

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
        draws=None,
        compiled_chunk=None,
    ):
        """Compatibility form of explicit bound Network host scheduling.

        Use ``run(functional.bind(parameters, constants), state, ...)`` for the
        common scheduling contract. This adapter retains ``chunklength`` and
        its existing interaction with ``checkpoint_every``.
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
            **({"draws": draws} if draws is not None else {}),
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
        draws=None,
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
        if draws is not None:
            from ._draws import _normalize_runtime_draws

            draws = _normalize_runtime_draws(draws)
            if netstim_draw is not None or any(
                isinstance(packet, StepInput)
                and (packet.noise is not None or packet.stimulation is not None)
                for packet in inputs.values()
            ):
                raise ValueError(
                    "draws cannot be combined with legacy runtime draw packets"
                )
            self._runtime_draw_plan.validate(draws)
            netstim_draw = (
                None
                if self.netstim is None
                else NetStimDraw(draws.draws["netstim"], draws.rng["netstim"].clone())
            )
        else:
            self._stimulation_rng_plan.validate_packets(
                {
                    name: packet.noise
                    for name, packet in inputs.items()
                    if isinstance(packet, StepInput)
                },
                {
                    name: packet.stimulation
                    for name, packet in inputs.items()
                    if isinstance(packet, StepInput)
                },
            )
        clock = state["clock"]
        rng_pool = state.get("rng", {})
        stimulation_pool = self._stimulation_rng_plan.expand_pool(
            rng_pool, state.get("stimulation_rng", {})
        )
        time = (
            clock["origin"] + clock["step"].to(clock["origin"].dtype) * self.dt
        ).detach()
        states = {
            name: self._snap_population_clock(
                self._stimulation_rng_plan.expand_population(
                    name, value, rng_pool, stimulation_pool
                ),
                time,
            )
            for name, value in state["populations"].items()
        }
        next_stim, stimulus = None, None
        if self.netstim is None:
            if netstim_draw is not None:
                raise ValueError("A Network without NetStim requires netstim_draw=None")
        else:
            stim = parameters["netstim"]
            next_stim, stimulus = self.netstim.step(
                prepared["netstim"]["parameters"],
                prepared["netstim"]["constants"],
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
                prepared["connections"]["continuous_synapses"][name]["weights"],
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
            effective = prepared["connections"]["synapses"][name]
            queue, payload = plan.read_due(
                state["synapses"][name],
                weights=effective["weights"],
                delays=effective["delays"],
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
                effective["weights"],
                queue,
                source,
                delays=effective["delays"],
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
                noise=pytree.tree_map_only(
                    torch.Tensor, lambda value: value.unsqueeze(0), selected.noise
                ),
                stimulation=pytree.tree_map_only(
                    torch.Tensor, lambda value: value.unsqueeze(0), selected.stimulation
                ),
            )
            runtime_draws = (
                None
                if draws is None
                else pytree.tree_map_only(
                    torch.Tensor,
                    lambda value: value.unsqueeze(0),
                    self._runtime_draw_plan.local(name, draws),
                )
            )
            states[name], aux[name] = plan._rollout_values(
                parameters["populations"][name],
                prepared["populations"][name],
                self._stimulation_rng_plan.expand_population(
                    name, states[name], rng_pool, stimulation_pool
                ),
                rollout,
                steps=1,
                extra=extra.get(name),
                **({"draws": runtime_draws} if runtime_draws is not None else {}),
            )
            if draws is None:
                rng_pool, stimulation_pool = self._stimulation_rng_plan.publish(
                    name, states[name], rng_pool, stimulation_pool
                )
            else:
                rng_pool, stimulation_pool = self._runtime_draw_plan.publish_population(
                    name, states[name], rng_pool, stimulation_pool, draws
                )
        step = clock["step"] + 1
        next_time = (
            clock["origin"] + step.to(clock["origin"].dtype) * self.dt
        ).detach()
        states = self._stimulation_plan.canonicalize(states)
        states = {
            name: self._stimulation_plan.strip(
                self._rng_plan.strip_population(
                    self._snap_population_clock(value, next_time)
                )
            )
            for name, value in states.items()
        }
        next_state = {
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
            **(
                {"connection_sampling": state["connection_sampling"]}
                if self._connection_sampling_plan.devices
                else {}
            ),
            **({"rng": rng_pool} if self._rng_plan.enabled else {}),
            **(
                {
                    "stimulation_rng": self._stimulation_rng_plan.strip_pool(
                        stimulation_pool
                    )
                }
                if self._stimulation_plan.enabled
                else {}
            ),
        }
        return self._connection_rng_from_stimulation(next_state), {
            "populations": aux,
            "netstim": stimulus,
        }

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
        draws=None,
        inputs=None,
        extra=None,
        intra=None,
    ):
        """Advance one complete native-ordered timestep with reusable preparation."""
        if draws is not None:
            from ._draws import _normalize_runtime_draws
            from ._network_bindings import _canonicalize_mapping_tree

            (parameters, state, inputs, draws), _ = _canonicalize_mapping_tree(
                (parameters, state, inputs, _normalize_runtime_draws(draws))
            )
        self._validate_source()
        self._validate_parameters(parameters)
        self._validate_state(state)
        values = self._validate_prepared(prepared, parameters)
        with nullcontext() if self.training else torch.no_grad():
            result = self._step_values(
                parameters,
                values,
                state,
                netstim_draw=netstim_draw,
                **({"draws": draws} if draws is not None else {}),
                inputs=inputs,
                extra=extra,
                intra=intra,
            )
        self.check_overflow(result[0])
        return result

    def prepare_and_step(
        self,
        parameters,
        constants,
        state,
        *,
        netstim_draw=None,
        draws=None,
        inputs=None,
        extra=None,
        intra=None,
    ):
        """Derive coefficients and execute one pure, fullgraph-compatible timestep."""
        if draws is not None:
            from ._draws import _normalize_runtime_draws
            from ._network_bindings import _canonicalize_mapping_tree

            (parameters, state, inputs, draws), _ = _canonicalize_mapping_tree(
                (parameters, state, inputs, _normalize_runtime_draws(draws))
            )
        self._validate_source()
        self._validate_parameters(parameters)
        _validate_tree(constants, self._constant_schema, "constants")
        self._validate_state(state)
        with nullcontext() if self.training else torch.no_grad():
            values = self._prepare_values(parameters, constants)
            result = self._step_values(
                parameters,
                values,
                state,
                netstim_draw=netstim_draw,
                **({"draws": draws} if draws is not None else {}),
                inputs=inputs,
                extra=extra,
                intra=intra,
            )
        self.check_overflow(result[0])
        return result

    def check_overflow(self, state):
        """Reject incomplete pending calendars at an untransformed host boundary.

        Compiled and transformed pure calls propagate sticky Tensor flags.
        Validate their returned carry here before consuming or committing it.
        Host rollout and compiled runners perform this check automatically.
        """
        for name, plan in self.synapses.items():
            try:
                plan.check_overflow(state["synapses"][name])
            except RuntimeError as error:
                raise RuntimeError(f"Event connection {name!r}: {error}") from error

    def commit_state_(self, network, state):
        """Stage and atomically publish complete runtime carry to a compatible graph."""
        if torch.compiler.is_compiling():
            raise FunctionalizationError("Commit Network state outside torch.compile.")
        self._validate_source()
        self._validate_compatible(network)
        self._validate_state(state)
        self._stimulation_plan.validate_state(
            {
                **state,
                "stimulation_rng": self._stimulation_rng_plan.expand_pool(
                    state.get("rng", {}), state.get("stimulation_rng", {})
                ),
            }
        )
        self.check_overflow(state)
        sampling_rng = (
            self._connection_sampling_plan.validate_rng(
                state["connection_sampling"]["rng"]
            )
            if self._connection_sampling_plan.devices
            else None
        )
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
        # Build graph-carrying RAND materializations against the destination's
        # raw Parameters before publishing any state. A staged checkpoint's
        # sample graphs instead belong to its private parameter copies.
        random_commits = {
            name: plan._random_plan.prepare_commit(
                network.populations[name],
                self._expand_population_state(name, state["populations"][name], state)[
                    "random"
                ],
            )
            for name, plan in self.populations.items()
            if plan._random_plan is not None
        }
        staged = _deepcopy_execution_module(network)
        for name, plan in self.populations.items():
            plan.commit_state_(
                staged.populations[name],
                self._expand_population_state(name, state["populations"][name], state),
            )
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
        if sampling_rng is not None:
            staged._functional_connection_sampling_rng = sampling_rng
        checkpoint = staged.state_dict_for_checkpoint()
        network.restore_dict_from_checkpoint(checkpoint)
        for name, plan in self.synapses.items():
            plan._commit_pending_(network.synapses[name], state["synapses"][name])
        for name, prepared in random_commits.items():
            self.populations[name]._random_plan.publish_commit(
                network.populations[name], prepared
            )
        # SCNV's native checkpoint omits its exposed voltage. The Population
        # adapter has already staged/validated that separate functional carry.
        for name in self.populations:
            if isinstance(network.populations[name].integrator, _bwd_euler_sc_skip):
                network.populations[name]._buffers["v"] = staged.populations[
                    name
                ]._buffers["v"]
        return network


def make_functional_network(network, *, extra=None, pending_capacity=None):
    """Lower a Network with independent explicit tensors and bounded pending carry.

    ``extra`` maps Population names to one ``(field, Waveform)`` pair or a
    sequence of contacts, using the same stimulation bindings as
    :func:`make_functional`. Omitted Populations have no bound extracellular
    stimulation. Fields and waveform leaves are explicit parameters, with
    shared Tensor aliases preserved across components.

    ``pending_capacity`` is a positive entry count per calendar, or a mapping
    from event-connection names to counts. Omitted capacities use at least
    twice the connection count and accommodate existing checkpoint traffic;
    intrinsic-only packed connections with no pending traffic need no calendar
    allocation.
    Overflow raises at host execution boundaries; retry from the input state
    with a larger-capacity plan.
    """
    plan = FunctionalNetwork(network, extra=extra, pending_capacity=pending_capacity)
    return plan, plan.extract()


__all__ = ["FunctionalNetwork", "NetworkTensors", "make_functional_network"]

_OWNED_NETWORK_NOISE_TAPES = FunctionalNetwork.sample_population_noise_tapes
_OWNED_NETWORK_STIMULATION_TAPES = FunctionalNetwork.sample_population_stimulation_tapes
_OWNED_NETWORK_NETSTIM_TAPE = FunctionalNetwork.sample_draw_tape
