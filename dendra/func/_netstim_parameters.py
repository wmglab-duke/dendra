"""Explicit raw NetStim sources over the shared native materializer."""

from __future__ import annotations

import inspect

import torch
from torch.func import functional_call

from dendra.helpers import current_check_purity
from dendra.models.networks._netstim_parameters import (
    evaluate_netstim_interval,
    materialize_netstim_parameters,
)
from dendra.models.networks.netstim import NetStim

from ._connection_parameters import (
    _binding,
    _callable_binding,
    _clone,
    _helpers,
    _registered,
)
from ._network_bindings import AliasSnapshots
from ._prepared_values import (
    _capture_prepared_values,
    _validate_prepared_freshness,
    _validate_prepared_owner,
)
from ._torch_parametrizations import (
    canonical_type,
    failures,
    install_output_schemas,
    source_binding_signature,
)
from ._types import FunctionalizationError

_FIELDS = ("start", "noise", "tau")


def _primary_source_path(path):
    if path in _FIELDS:
        return True
    parts = path.split(".")
    if len(parts) != 3 or parts[0] != "parametrizations" or parts[1] not in _FIELDS:
        return False
    name = parts[2]
    return name == "original" or (
        name.startswith("original") and name[len("original") :].isdigit()
    )


def _source_view(source):
    """Borrow only the registered renewal recipe, without clocks or generators."""
    view = type(source).__new__(type(source))
    torch.nn.Module.__init__(view)
    view.training = source.training
    view.shape = tuple(source.shape)
    view._parameters = {
        name: value for name, value in source._parameters.items() if name in _FIELDS
    }
    view._buffers = {
        name: value for name, value in source._buffers.items() if name in _FIELDS
    }
    view._modules = {"interval": source.interval}
    if "parametrizations" in source._modules:
        view._modules["parametrizations"] = source._modules["parametrizations"]
    for name in _FIELDS:
        if name in vars(source):
            value = vars(source)[name]
            if torch.is_tensor(value) and not isinstance(value, torch.nn.Parameter):
                view.register_buffer(name, value, persistent=False)
            else:
                setattr(view, name, value)
    return view


class _MaterializeNetStim(torch.nn.Module):
    def __init__(self, source, shape, dtype, device):
        super().__init__()
        self.source = source
        self.shape, self.dtype, self.device = shape, dtype, device

    def forward(self):
        return materialize_netstim_parameters(
            evaluate_netstim_interval(self.source.interval),
            self.source.start,
            self.source.noise,
            self.source.tau,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )


class NetStimParameterPlan:
    """Bind raw renewal inputs and trainable transforms with fixed schemas."""

    def __init__(self, source, *, owners=()):
        if canonical_type(source) is not NetStim:
            raise FunctionalizationError(
                "Expected a canonical NetStim parameter source."
            )
        for name in ("renewal_parameters", "_broadcast_to_state"):
            if name in vars(source) or inspect.getattr_static(
                type(source), name
            ) is not inspect.getattr_static(NetStim, name):
                raise FunctionalizationError(
                    "NetStim parameters need their canonical materialization adapter."
                )
        self._source = source
        self._check_purity = current_check_purity()
        self._token = object()
        self._owners = tuple(owners)
        registry = source._modules.get("parametrizations")
        if registry is not None and not set(registry).issubset(_FIELDS):
            raise FunctionalizationError(
                "NetStim parametrizations support start, noise, tau, and interval raw sources; runtime fields and integer constants are currently unsupported."
            )
        errors = failures(source, diagnostic=self._check_purity)
        if errors:
            raise FunctionalizationError("; ".join(errors))
        view = _source_view(source)
        self._parameter_paths, self._constant_paths = {}, {}
        self._parameter_aliases, self._constant_aliases = {}, {}
        parameter_ids, constant_ids = {}, {}
        registered = tuple(_registered(view))
        primary = {
            name
            for name, _value, _kind in registered
            if _primary_source_path(name)
            or name == "interval.rho"
            or name.startswith("interval.parametrizations.rho.original")
        }
        trainable_ids = {
            id(value)
            for name, value, kind in registered
            if name in primary
            or kind == "_parameters"
            or isinstance(value, torch.nn.Parameter)
        }
        for name, value, _kind in registered:
            if id(value) in trainable_ids:
                key = parameter_ids.setdefault(id(value), name)
                self._parameter_paths[name] = name
                self._parameter_aliases[name] = key
            else:
                key = constant_ids.setdefault(id(value), name)
                self._constant_paths[name] = name
                self._constant_aliases[name] = key
        self._synthetic = {}
        if "tau" in vars(source) and not torch.is_tensor(source.tau):
            self._synthetic["tau"] = torch.as_tensor(
                source.tau,
                dtype=source.next_stoch_time.dtype,
                device=source.next_stoch_time.device,
            )
            # The execution slot is explicit even though native tau is a float.
            delattr(view, "tau")
            view.register_buffer("tau", self._synthetic["tau"], persistent=False)
            self._parameter_paths["tau"] = None
            self._parameter_aliases["tau"] = "tau"
        self._packet_aliases = tuple(
            (kind, path, key)
            for kind, aliases in (
                ("parameter", self._parameter_aliases),
                ("constant", self._constant_aliases),
            )
            for path, key in aliases.items()
            if path != key
        )
        self._class_guards = []
        seen = set()
        p = _helpers()
        for child in view.modules():
            for cls in type(child).__mro__:
                if (
                    cls in seen
                    or cls in (torch.nn.Module, object)
                    or cls.__module__.startswith("torch.")
                ):
                    continue
                seen.add(cls)
                for name, value in vars(cls).items():
                    if name not in p._CLASS_STATE_METADATA_NAMES:
                        self._class_guards.append(
                            (cls, name, value, _callable_binding(value))
                        )
        self._source_bindings = self._binding(source)
        self._compatible_bindings = self._binding(source, identities=False)
        self._parameter_schema = {
            name: (tuple(value.shape), value.dtype, value.device)
            for name, value in self.parameter_sources().items()
        }
        self._constant_schema = {
            name: (tuple(value.shape), value.dtype, value.device)
            for name, value in self.constant_sources().items()
        }
        self._schema = (
            tuple(source.shape),
            source.next_stoch_time.dtype,
            source.next_stoch_time.device,
        )
        self._validate_runtime_ownership(source, owners=self._owners)
        execution = p._deepcopy_execution_module(view)
        if self._check_purity:
            device = source.next_stoch_time.device
            rng = p._global_rng_snapshot(device)
            try:
                install_output_schemas(execution)
                if p._global_rng_changes(rng, device):
                    raise FunctionalizationError(
                        "NetStim purity diagnostics observed implicit random draws."
                    )
            finally:
                p._restore_global_rng(rng, device)
        else:
            install_output_schemas(execution)
        self._execution = _MaterializeNetStim(execution, *self._schema)
        if self._check_purity:
            self._audit_sources()
        self.materialize(self.parameter_sources(), self.constant_sources())

    @staticmethod
    def _resolve(source, path):
        prefix, _, name = path.rpartition(".")
        owner = source.get_submodule(prefix)
        return owner._parameters.get(
            name, owner._buffers.get(name, vars(owner).get(name))
        )

    def parameter_sources(self, source=None):
        source = self._source if source is None else source
        return {
            key: (
                self._resolve(source, path)
                if path is not None
                else torch.as_tensor(
                    source.tau,
                    dtype=self._synthetic[key].dtype,
                    device=self._synthetic[key].device,
                )
            )
            for key, path in self._parameter_paths.items()
        }

    def constant_sources(self, source=None):
        source = self._source if source is None else source
        return {
            key: self._resolve(source, path)
            for key, path in self._constant_paths.items()
        }

    def cache_constant_sources(self, source=None):
        return self.constant_sources(source)

    def extract(self, snapshots=None, source=None):
        source = self._source if source is None else source
        self.validate_source(source)
        snapshots = AliasSnapshots() if snapshots is None else snapshots
        return (
            {
                key: snapshots.snapshot(value, value)
                for key, value in self.parameter_sources(source).items()
            },
            {
                key: snapshots.snapshot(value, value)
                for key, value in self.constant_sources(source).items()
            },
        )

    def _binding(self, source, *, identities=True):
        registry = source._modules.get("parametrizations")
        return (
            _binding(source.interval, identities=identities),
            None if registry is None else _binding(registry, identities=identities),
            source_binding_signature(source) if identities else None,
            tuple(
                (
                    name,
                    id(value) if identities else None,
                    tuple(value.shape),
                    value.dtype,
                    value.device,
                )
                for name, value in (
                    *self.parameter_sources(source).items(),
                    *self.constant_sources(source).items(),
                )
                if self._parameter_paths.get(name, "present") is not None
            ),
            tuple(
                (name, type(vars(source)[name]))
                for name in _FIELDS
                if name in vars(source) and not torch.is_tensor(vars(source)[name])
            ),
            tuple(
                (name, _callable_binding(getattr(source, name)))
                for name in (
                    "renewal_parameters",
                    "_broadcast_to_state",
                    "forward",
                    "initialize",
                )
            ),
        )

    def validate_source(self, source=None):
        source = self._source if source is None else source
        p = _helpers()
        for cls, name, expected, signature in self._class_guards:
            same = (
                cls.__dict__ == p._RawClassHookIdentity(name, expected)
                if torch.compiler.is_compiling()
                else vars(cls).get(name) is expected
            )
            if not same or _callable_binding(expected) != signature:
                raise FunctionalizationError(
                    "NetStim parameter source class changed; lower again."
                )
        expected = (
            self._source_bindings
            if source is self._source
            else self._compatible_bindings
        )
        if self._binding(source, identities=source is self._source) != expected:
            raise FunctionalizationError(
                "NetStim parameter source structure changed; lower again."
            )
        if source is not self._source:
            self._validate_runtime_ownership(source, owners=self._owners)
            for aliases, sources in (
                (self._parameter_aliases, self.parameter_sources(source)),
                (self._constant_aliases, self.constant_sources(source)),
            ):
                for path, key in aliases.items():
                    if (
                        self._parameter_paths.get(key, "present") is not None
                        and self._resolve(source, path) is not sources[key]
                    ):
                        raise FunctionalizationError(
                            "Compatible NetStim parameter aliases changed; lower again."
                        )

    def _validate_runtime_ownership(self, source, *, owners=()):
        from dendra.models.core import Population

        from ._initialization_sources import initialization_source_runtime_buffer_ids

        runtime = [
            value
            for name, value in source._buffers.items()
            if name not in _FIELDS and torch.is_tensor(value)
        ]
        # The differentiable spike gate is intentionally an ordinary Tensor
        # attribute, but it still belongs to renewed runtime state.
        if torch.is_tensor(getattr(source, "spike_gate", None)):
            runtime.append(source.spike_gate)
        population_runtime = []
        for owner in owners:
            if isinstance(owner, Population):
                ids = initialization_source_runtime_buffer_ids(owner)
                population_runtime.extend(
                    value for value in owner.buffers() if id(value) in ids
                )

        def aliases(value, candidates):
            return any(
                value is other
                or value.numel()
                and other.numel()
                and value.device == other.device
                and value.untyped_storage().data_ptr()
                == other.untyped_storage().data_ptr()
                for other in candidates
            )

        for path, value in {
            **self.parameter_sources(source),
            **self.constant_sources(source),
        }.items():
            # Native start/noise/tau inputs can intentionally read Population
            # storage, including cache-restored state or materialized fields.
            # Their explicit primary originals retain that existing binding;
            # hidden transform dependencies and interval recipes cannot borrow
            # mutable Population workspaces as fixed configuration.
            primary = _primary_source_path(path)
            if aliases(value, runtime) or (
                not primary and aliases(value, population_runtime)
            ):
                raise FunctionalizationError(
                    f"NetStim source {path!r} aliases runtime state; use independent config buffers or shared raw Parameters."
                )
        p = _helpers()
        if p._module_autograd_hook_signature(source):
            raise FunctionalizationError(
                "NetStim forward hooks are currently unsupported by functional renewal."
            )
        for child in _source_view(source).modules():
            if p._module_autograd_hook_signature(child):
                raise FunctionalizationError(
                    "NetStim parameter source hooks are currently unsupported."
                )

    def materialize(self, parameters, constants):
        self.validate_source()
        for values, schema, label in (
            (parameters, self._parameter_schema, "parameter"),
            (constants, self._constant_schema, "constant"),
        ):
            if set(values) != set(schema):
                raise FunctionalizationError(f"NetStim raw {label} keys changed.")
            for key, expected in schema.items():
                value = values[key]
                if (
                    not torch.is_tensor(value)
                    or (tuple(value.shape), value.dtype, value.device) != expected
                ):
                    raise FunctionalizationError(
                        f"NetStim raw {label} {key!r} changed tensor schema."
                    )
        for kind, path, canonical in self._packet_aliases:
            values = parameters if kind == "parameter" else constants
            if values[path] is not values[canonical]:
                raise FunctionalizationError(
                    "Explicit NetStim sources must preserve their shared tensor aliases."
                )
        p = _helpers()
        mapping = {
            name: value.clone()
            for name, value in p._module_tensor_slots(self._execution)
        }
        canonical = {}
        for aliases, values in (
            (self._parameter_aliases, parameters),
            (self._constant_aliases, constants),
        ):
            for path, key in aliases.items():
                if key not in canonical:
                    canonical[key] = values[key].clone()
                mapping[f"source.{path}"] = canonical[key]
        result = functional_call(self._execution, mapping, (), tie_weights=False)
        if set(result) != {"interval", *_FIELDS} or any(
            not torch.is_tensor(value)
            or (tuple(value.shape), value.dtype, value.device) != self._schema
            for value in result.values()
        ):
            raise FunctionalizationError(
                "NetStim materialization changed output shape, dtype, or device."
            )
        return result

    def prepare(self, parameters, constants):
        values = self.materialize(parameters, constants)
        return _capture_prepared_values(
            self._token,
            parameters,
            constants,
            values,
        )

    def validate_prepared(self, prepared, parameters):
        _validate_prepared_owner(
            prepared,
            self._token,
            message="Prepared NetStim parameters belong to another plan.",
        )
        _validate_prepared_freshness(
            prepared,
            parameters,
            message="NetStim parameters, constants, or prepared values changed; prepare again.",
        )
        return self.materialize(parameters, prepared.constants)

    def _audit_sources(self):
        p = _helpers()
        execution = self._execution
        for child in execution.modules():
            if (
                child is execution
                or isinstance(child, NetStim)
                or type(child).__module__.startswith(("torch.", "dendra."))
            ):
                continue
            p._initialization_callable_contract_signature(
                child.forward,
                label="netstim.parameters",
                owner=child,
                strict=True,
                diagnostic=True,
            )
        device = self._source.next_stoch_time.device
        rng = p._global_rng_snapshot(device)
        before = _binding(execution, identities=False)
        tensors = {
            name: (value._version, _clone(value))
            for name, value in p._module_tensor_slots(execution)
        }
        try:
            with torch.no_grad():
                first, second = execution(), execution()
            if (
                p._global_rng_changes(rng, device)
                or before != _binding(execution, identities=False)
                or any(not torch.equal(first[name], second[name]) for name in first)
            ):
                raise FunctionalizationError(
                    "NetStim purity diagnostics observed implicit RNG or mutable state."
                )
            for name, value in p._module_tensor_slots(execution):
                version, saved = tensors[name]
                if value._version != version or not torch.equal(value, saved):
                    raise FunctionalizationError(
                        "NetStim purity diagnostics observed an in-place tensor write."
                    )
        finally:
            p._restore_global_rng(rng, device)
