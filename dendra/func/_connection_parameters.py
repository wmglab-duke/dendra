"""Explicit raw sources for the native connection parameter materializer."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MethodType

import torch
from torch.func import functional_call

from dendra.helpers import current_check_purity
from dendra.models.distributions import connection_distribution_placeholder
from dendra.models.networks._population_cache import restore_population_cache_values
from dendra.models.networks.net import (
    ParameterOrDistributionWrapper,
    WeightExpander,
    _connection_source_can_compute_uncached,
    connection_distribution_is_supported,
)
from dendra.models.parametric import Bounded, NegativeParam, is_parametric

from ._network_bindings import AliasSnapshots
from ._types import FunctionalizationError

_ROLES = (("weights", "weight"), ("delays", "delay_ms"))


def _helpers():
    from . import _population

    return _population


def _clone(value):
    with torch.inference_mode(False):
        return value.detach().clone(memory_format=torch.preserve_format)


def _registered(module):
    """Read canonical registered slots without invoking numerical properties."""
    from ._continuous_transform import _safe_modules

    for path, child in _safe_modules(module):
        for kind in ("_parameters", "_buffers"):
            for name, value in getattr(child, kind).items():
                if value is not None:
                    yield f"{path + '.' if path else ''}{name}", value, kind


def _callable_binding(value):
    target = (
        value.__func__
        if isinstance(value, (MethodType, staticmethod, classmethod))
        else value
    )
    return id(target), id(getattr(target, "__code__", None))


def _binding(module, *, identities=True):
    from ._stimuli import _is_internal_module_state, _trusted_state_signature
    from ._torch_parametrizations import (
        canonical_type,
        registries,
        source_binding_signature,
    )

    entries = []
    for path, child in module.named_modules():
        methods = tuple(
            (
                name,
                _callable_binding(getattr(child, name)),
            )
            for name in (
                "forward",
                "derive_values",
                "deterministic",
                "sample",
                "rsample",
                "sample_draw",
                "sample_from_draw",
                "sample_with_draw",
                "sample_draw_schema",
                "sample_draw_placeholder",
                "sample_draws",
                "_compute",
                "repeat",
            )
            if hasattr(child, name)
        )
        entries.append(
            (
                path,
                id(child) if identities else None,
                type(child) if identities else canonical_type(child),
                methods,
                tuple(
                    (
                        name,
                        tuple(
                            (key, id(hook))
                            for key, hook in getattr(child, name).items()
                        ),
                    )
                    for name in (
                        "_forward_hooks",
                        "_forward_pre_hooks",
                        "_backward_hooks",
                        "_backward_pre_hooks",
                    )
                ),
                tuple(
                    (name, _trusted_state_signature(value))
                    for name, value in vars(child).items()
                    if not _is_internal_module_state(child, name)
                    and name not in ("_cache", "_sampling_draws")
                    and not isinstance(value, torch.nn.Module)
                ),
                tuple(
                    (
                        kind,
                        name,
                        id(value)
                        if identities and not (path == "" and name == "w")
                        else None,
                        None
                        if value is None
                        else (tuple(value.shape), value.dtype, value.device),
                    )
                    for kind in ("_parameters", "_buffers")
                    for name, value in getattr(child, kind).items()
                ),
            )
        )
    torch_bindings = (
        source_binding_signature(module)
        if identities
        else tuple(
            (
                path,
                canonical_type(owner),
                name,
                len(transforms),
                transforms.is_tensor,
                transforms.ntensors,
                transforms.unsafe,
            )
            for path, owner, registry in registries(module)
            for name, transforms in registry.items()
        )
    )
    return tuple(entries), torch_bindings


class _DeriveValues(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, draws=None):
        return self.source.derive_values(draws=draws)


class _SampleDraws(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, generators=None):
        return self.source.sample_draws(generator=generators)


@dataclass(frozen=True)
class _PreparedConnectionParameters:
    token: object
    parameter_binding: tuple
    constant_binding: tuple
    constants: dict
    value_binding: tuple
    values: dict


class ConnectionParameterPlan:
    """Lower initialized native weight/delay sources to fixed explicit packets.

    Deterministic and explicit-draw distribution sources share
    ``WeightExpander.derive_values`` with native initialization. Random base
    draws live in the explicit constants packet, separately from raw sources.
    """

    def __init__(self, connection):
        self._source = connection
        self._check_purity = current_check_purity()
        self._token = object()
        self._raw = {}
        self._draw_schemas = {}
        self._snapshot_draw_schemas = {}
        self._alternate_draws = {}
        self._schemas = {}
        self._execution = {}
        self._parameter_paths = {}
        self._constant_paths = {}
        self._role_parameter_aliases = {}
        self._role_constant_aliases = {}
        self._metadata = {}
        self._class_guards = []
        parameter_identity, constant_identity = {}, {}
        modules = {}
        p = _helpers()
        for role, native in _ROLES:
            module = getattr(connection, native)
            if type(module) is not WeightExpander or any(
                name in vars(module)
                or inspect.getattr_static(type(module), name)
                is not inspect.getattr_static(WeightExpander, name)
                for name in ("forward", "derive_values", "init", "deterministic")
            ):
                raise FunctionalizationError(
                    "Connection parameters need the canonical initialized WeightExpander adapter."
                )
            if any(
                type(source) is not ParameterOrDistributionWrapper
                for source in module.weights
            ):
                raise FunctionalizationError(
                    "Connection parameter source wrappers changed; lower again."
                )
            modules[native] = module
            from ._torch_parametrizations import failures as torch_failures

            failures = torch_failures(module, diagnostic=self._check_purity)
            if failures:
                raise FunctionalizationError("; ".join(failures))
            value = module._buffers.get("w")
            if not torch.is_tensor(value) or value.ndim != 1:
                raise FunctionalizationError(
                    "Initialize one-dimensional connection parameter storage before lowering."
                )
            self._schemas[role] = (tuple(value.shape), value.dtype, value.device)
            self._raw[role] = all(
                source.deterministic()
                or connection_distribution_is_supported(source.param)
                for source in module.weights
            )
            self._draw_schemas[role] = (
                {
                    str(index): self._sampling_schema(source, count, role=role)
                    for index, (source, count) in enumerate(
                        zip(module.weights, module.n, strict=True)
                    )
                    if not source.deterministic()
                }
                if self._raw[role]
                else {}
            )
            provenance = module.sampling_draws if self._raw[role] else None
            self._snapshot_draw_schemas[role] = {
                index: (
                    tuple(shape),
                    dtype if provenance is None else provenance[index].dtype,
                    device,
                )
                for index, (shape, dtype, device) in self._draw_schemas[role].items()
            }
            self._alternate_draws[role] = any(
                self._snapshot_draw_schemas[role][index]
                != (tuple(shape), dtype, device)
                for index, (shape, dtype, device) in self._draw_schemas[role].items()
            )
            if not self._raw[role]:
                key = f"{native}.w"
                self._parameter_paths[key] = (native, "w")
                continue
            parameter_aliases, constant_aliases = {}, {}
            registered = tuple(_registered(module))
            raw_parameter_ids = {
                id(value)
                for _name, value, kind in registered
                if kind == "_parameters" or isinstance(value, torch.nn.Parameter)
            }
            for relative, tensor, kind in registered:
                if relative == "w":
                    continue
                path = f"{native}.{relative}"
                if id(tensor) in raw_parameter_ids:
                    key = parameter_identity.setdefault(id(tensor), path)
                    self._parameter_paths.setdefault(key, (native, relative))
                    parameter_aliases[relative] = key
                else:
                    key = constant_identity.setdefault(id(tensor), path)
                    self._constant_paths.setdefault(key, (native, relative))
                    constant_aliases[relative] = key
            self._role_parameter_aliases[role] = parameter_aliases
            self._role_constant_aliases[role] = constant_aliases
            execution = p._deepcopy_execution_module(module)
            from ._torch_parametrizations import install_output_schemas

            install_output_schemas(execution)
            self._execution[role] = _DeriveValues(execution)
        # A Parameter may also have a buffer registration in the other role.
        # Promote that route to the same public raw leaf rather than copying it.
        all_parameters = {
            id(value): key for key, value in self.parameter_sources().items()
        }
        for role, native in _ROLES:
            for relative, key in tuple(
                self._role_constant_aliases.get(role, {}).items()
            ):
                value = self._resolve(connection, self._constant_paths[key])
                if id(value) in all_parameters:
                    self._role_parameter_aliases[role][relative] = all_parameters[
                        id(value)
                    ]
                    del self._role_constant_aliases[role][relative]
        used_constants = {
            key
            for aliases in self._role_constant_aliases.values()
            for key in aliases.values()
        }
        self._constant_paths = {
            key: path
            for key, path in self._constant_paths.items()
            if key in used_constants
        }
        self._source_bindings = tuple(
            (native, _binding(module)) for native, module in modules.items()
        )
        self._compatible_bindings = tuple(
            (native, _binding(module, identities=False))
            for native, module in modules.items()
        )
        seen_classes = set()
        for module in modules.values():
            for child in module.modules():
                for cls in type(child).__mro__:
                    if (
                        cls in seen_classes
                        or cls in (torch.nn.Module, object)
                        or cls.__module__.startswith("torch.")
                    ):
                        continue
                    seen_classes.add(cls)
                    for name, value in vars(cls).items():
                        if name not in p._CLASS_STATE_METADATA_NAMES:
                            self._class_guards.append(
                                (cls, name, value, _callable_binding(value))
                            )
        self._parameter_schema = {
            key: (tuple(value.shape), value.dtype, value.device)
            for key, value in self.parameter_sources().items()
        }
        self._constant_schema = {
            key: (tuple(value.shape), value.dtype, value.device)
            for key, value in self.constant_sources().items()
        }
        self._validate_runtime_ownership(connection)
        if self._check_purity:
            self._audit_sources()
        parameters, constants = self.parameter_sources(), self.constant_sources()
        for role, native in _ROLES:
            self._metadata.update(self._sampling_metadata(connection, role, native))
            value = getattr(connection, native).w
            self._metadata[f"stored.{role}"] = _clone(value)
            self._metadata[f"retain_{role}"] = torch.zeros(
                (), dtype=torch.bool, device=value.device
            )
            self._metadata[f"use_offset.{role}"] = torch.ones(
                (), dtype=torch.bool, device=value.device
            )
            for key in set(self._role_parameter_aliases.get(role, {}).values()):
                self._metadata[f"baseline.{role}.parameters.{key}"] = _clone(
                    parameters[key]
                )
            for key in set(self._role_constant_aliases.get(role, {}).values()):
                self._metadata[f"baseline.{role}.constants.{key}"] = _clone(
                    constants[key]
                )
            derived = (
                self._derive(role, parameters, {**constants, **self._metadata})
                if self._raw[role]
                else value
            )
            self._metadata[f"baseline_effective.{role}"] = _clone(derived)
        self._metadata_schema = {
            key: (tuple(value.shape), value.dtype, value.device)
            for key, value in self._metadata.items()
        }

    @staticmethod
    def _resolve(connection, path):
        native, relative = path
        owner = getattr(connection, native)
        prefix, _, name = relative.rpartition(".")
        owner = owner.get_submodule(prefix)
        return owner._parameters.get(name, owner._buffers.get(name))

    def _sampling_schema(self, source, count, *, role):
        if not self._check_purity:
            return source.sampling_draw_schema(count)
        # Metadata hooks run before numerical materialization. Probe a private
        # module so diagnostics can reject mutation without touching the source.
        p = _helpers()
        execution = p._deepcopy_execution_module(source)
        device = self._schemas[role][2]
        rng = p._global_rng_snapshot(device)
        before = _binding(execution, identities=False)
        tensors = {
            name: (value._version, _clone(value))
            for name, value in p._module_tensor_slots(execution)
        }
        try:
            with torch.no_grad():
                first = execution.sampling_draw_schema(count)
                second = execution.sampling_draw_schema(count)
                placeholder = connection_distribution_placeholder(
                    execution.param, (count,)
                )
                repeated = connection_distribution_placeholder(
                    execution.param, (count,)
                )
            if p._global_rng_changes(rng, device):
                raise FunctionalizationError(
                    "Connection sampling metadata purity diagnostics observed implicit random draws."
                )
            if (
                first != second
                or not torch.equal(placeholder, repeated)
                or (before != _binding(execution, identities=False))
            ):
                raise FunctionalizationError(
                    "Connection sampling metadata purity diagnostics observed mutable or nondeterministic state."
                )
            for name, value in p._module_tensor_slots(execution):
                version, saved = tensors[name]
                if value._version != version or not torch.equal(value, saved):
                    raise FunctionalizationError(
                        "Connection sampling metadata purity diagnostics observed an in-place tensor write."
                    )
            return first
        finally:
            p._restore_global_rng(rng, device)

    def parameter_sources(self, connection=None):
        connection = self._source if connection is None else connection
        return {
            key: self._resolve(connection, path)
            for key, path in self._parameter_paths.items()
        }

    def constant_sources(self, connection=None):
        connection = self._source if connection is None else connection
        return {
            key: self._resolve(connection, path)
            for key, path in self._constant_paths.items()
        }

    def cache_constant_sources(self, connection=None):
        connection = self._source if connection is None else connection
        return {
            **self.constant_sources(connection),
            **{
                f"stored.{role}": getattr(connection, native).w
                for role, native in _ROLES
            },
        }

    def metadata_constants(self, connection=None):
        if connection is None:
            return {key: _clone(value) for key, value in self._metadata.items()}
        parameters, constants = (
            self.parameter_sources(connection),
            self.constant_sources(connection),
        )
        metadata = {}
        for role, native in _ROLES:
            metadata.update(self._sampling_metadata(connection, role, native))
            value = getattr(connection, native).w
            metadata[f"stored.{role}"] = _clone(value)
            metadata[f"retain_{role}"] = torch.zeros(
                (), dtype=torch.bool, device=value.device
            )
            metadata[f"use_offset.{role}"] = torch.ones(
                (), dtype=torch.bool, device=value.device
            )
            for key in set(self._role_parameter_aliases.get(role, {}).values()):
                metadata[f"baseline.{role}.parameters.{key}"] = _clone(parameters[key])
            for key in set(self._role_constant_aliases.get(role, {}).values()):
                metadata[f"baseline.{role}.constants.{key}"] = _clone(constants[key])
            derived = (
                self._derive(role, parameters, {**constants, **metadata})
                if self._raw[role]
                else value
            )
            metadata[f"baseline_effective.{role}"] = _clone(derived)
        return metadata

    def _sampling_metadata(self, connection, role, native):
        schemas = self._snapshot_draw_schemas[role]
        if not schemas:
            return {}
        module = getattr(connection, native)
        draws = module.sampling_draws
        valid = draws is not None
        if valid and set(draws) != set(schemas):
            raise FunctionalizationError(
                "Connection sampling provenance changed block structure; lower again."
            )
        result = {
            f"draw_valid.{role}": torch.tensor(
                valid, dtype=torch.bool, device=self._schemas[role][2]
            )
        }
        use_fresh = False
        if valid:
            snapshot_match = all(
                (tuple(draws[index].shape), draws[index].dtype, draws[index].device)
                == (tuple(shape), dtype, device)
                for index, (shape, dtype, device) in schemas.items()
            )
            fresh_match = all(
                (tuple(draws[index].shape), draws[index].dtype, draws[index].device)
                == (tuple(shape), dtype, device)
                for index, (shape, dtype, device) in self._draw_schemas[role].items()
            )
            if not snapshot_match:
                if not self._alternate_draws[role] or not fresh_match:
                    raise FunctionalizationError(
                        "Connection sampling provenance changed draw schema; lower again."
                    )
                use_fresh = True
        if self._alternate_draws[role]:
            result[f"use_fresh_draw.{role}"] = torch.tensor(
                use_fresh, dtype=torch.bool, device=self._schemas[role][2]
            )
        for index, (shape, dtype, device) in schemas.items():
            value = (
                draws[index]
                if valid and not use_fresh
                else self._placeholder_draw(role, index, dtype=dtype, device=device)
            )
            if (tuple(value.shape), value.dtype, value.device) != (
                tuple(shape),
                dtype,
                device,
            ):
                raise FunctionalizationError(
                    "Connection sampling provenance changed draw schema; lower again."
                )
            result[f"draw.{role}.{index}"] = _clone(value)
            if self._alternate_draws[role]:
                shape, dtype, device = self._draw_schemas[role][index]
                value = (
                    draws[index]
                    if valid and use_fresh
                    else self._placeholder_draw(role, index, dtype=dtype, device=device)
                )
                result[f"fresh_draw.{role}.{index}"] = _clone(value)
        return result

    def _placeholder_draw(self, role, index, *, dtype=None, device=None):
        source = getattr(self._source, dict(_ROLES)[role]).weights[int(index)].param
        count = getattr(self._source, dict(_ROLES)[role]).n[int(index)]
        # Recipes provide a valid deterministic draw for inactive paths; zero
        # is outside the domain of some inverse-CDF transforms. Preserve the
        # retained lane's dtype when it differs from fresh sampling metadata.
        value = connection_distribution_placeholder(source, (count,))
        return _clone(value.to(dtype=dtype, device=device))

    @property
    def sampling_devices(self):
        """Devices whose global random streams native initialization consumes."""
        return tuple(
            dict.fromkeys(
                device
                for schemas in self._draw_schemas.values()
                for _shape, _dtype, device in schemas.values()
            )
        )

    @property
    def has_sampling(self):
        return any(self._draw_schemas.values())

    @property
    def draw_schemas(self):
        return {role: dict(schemas) for role, schemas in self._draw_schemas.items()}

    def extract(self, snapshots=None, connection=None):
        connection = self._source if connection is None else connection
        self.validate_source(connection)
        snapshots = AliasSnapshots() if snapshots is None else snapshots
        parameters = {
            key: snapshots.snapshot(value, value)
            for key, value in self.parameter_sources(connection).items()
        }
        constants = {
            key: snapshots.snapshot(value, value)
            for key, value in self.constant_sources(connection).items()
        }
        constants.update(self.metadata_constants(connection))
        return parameters, constants

    def source_binding_signature(self, connection=None):
        connection = self._source if connection is None else connection
        p = _helpers()
        for cls, name, expected, callable_binding in self._class_guards:
            same = (
                cls.__dict__ == p._RawClassHookIdentity(name, expected)
                if torch.compiler.is_compiling()
                else vars(cls).get(name) is expected
            )
            if not same or _callable_binding(expected) != callable_binding:
                raise FunctionalizationError(
                    "Connection parameter source class changed; lower again."
                )
        return tuple(
            (native, _binding(getattr(connection, native))) for _role, native in _ROLES
        )

    def validate_source(self, connection=None):
        connection = self._source if connection is None else connection
        if connection is self._source:
            actual, expected = self.source_binding_signature(), self._source_bindings
        else:
            actual = tuple(
                (native, _binding(getattr(connection, native), identities=False))
                for _role, native in _ROLES
            )
            expected = self._compatible_bindings
        if actual != expected:
            raise FunctionalizationError(
                "Connection parameter source structure changed; lower again."
            )
        if connection is not self._source:
            self._validate_runtime_ownership(connection)
            for role, native in _ROLES:
                for aliases, sources in (
                    (
                        self._role_parameter_aliases.get(role, {}),
                        self.parameter_sources(connection),
                    ),
                    (
                        self._role_constant_aliases.get(role, {}),
                        self.constant_sources(connection),
                    ),
                ):
                    for relative, key in aliases.items():
                        if (
                            self._resolve(connection, (native, relative))
                            is not sources[key]
                        ):
                            raise FunctionalizationError(
                                "Compatible connection parameter aliases changed; lower again."
                            )

    def _validate_runtime_ownership(self, connection, *, owners=()):
        from dendra.models.core import Population

        from ._initialization_sources import initialization_source_runtime_buffer_ids

        runtime_ids, runtime_values = set(), []
        owners = (
            *owners,
            *(getattr(connection, name, None) for name in ("pre", "post")),
        )
        for owner in owners:
            if isinstance(owner, Population):
                runtime_ids.update(initialization_source_runtime_buffer_ids(owner))
                runtime_values.extend(
                    value for value in owner.buffers() if id(value) in runtime_ids
                )

        def storage_span(value):
            if value.layout is not torch.strided or not value.numel():
                return None
            size = value.element_size()
            start = value.storage_offset() * size
            stop = start + size * (
                1
                + sum(
                    (length - 1) * stride
                    for length, stride in zip(value.shape, value.stride(), strict=True)
                )
            )
            return value.device, value.untyped_storage().data_ptr(), start, stop

        runtime_spans = tuple(
            span
            for value in runtime_values
            if (span := storage_span(value)) is not None
        )

        def aliases_runtime(value):
            if id(value) in runtime_ids:
                return True
            span = storage_span(value)
            return span is not None and any(
                span[:2] == other[:2] and span[2] < other[3] and other[2] < span[3]
                for other in runtime_spans
            )

        sources = {
            **self.parameter_sources(connection),
            **self.constant_sources(connection),
        }
        conflicts = tuple(
            name for name, value in sources.items() if aliases_runtime(value)
        )
        if conflicts:
            raise FunctionalizationError(
                f"Connection parameter sources {conflicts} alias runtime state, geometry, or reconstructed parameters; use independent config buffers or shared raw Parameters."
            )
        p = _helpers()
        for _role, native in _ROLES:
            for _path, module in getattr(connection, native).named_modules():
                if p._module_autograd_hook_signature(module):
                    raise FunctionalizationError(
                        "Connection parameter source hooks are currently unsupported."
                    )

    def _validate_packet(self, parameters, constants):
        for values, schema, label in (
            (parameters, self._parameter_schema, "parameter"),
            (constants, self._constant_schema, "constant"),
            (constants, self._metadata_schema, "metadata"),
        ):
            for name, expected in schema.items():
                value = values.get(name)
                if (
                    not torch.is_tensor(value)
                    or (tuple(value.shape), value.dtype, value.device) != expected
                ):
                    raise FunctionalizationError(
                        f"Connection {label} {name!r} changed tensor schema."
                    )

    def _mapping(self, role, parameters, constants, execution):
        p = _helpers()
        mapping = {
            name: value.clone() for name, value in p._module_tensor_slots(execution)
        }
        canonical = {}
        for relative, key in self._role_parameter_aliases[role].items():
            if key not in canonical:
                canonical[key] = parameters[key].clone()
            mapping[f"source.{relative}"] = canonical[key]
        for relative, key in self._role_constant_aliases[role].items():
            if key not in canonical:
                canonical[key] = constants[key].clone()
            mapping[f"source.{relative}"] = canonical[key]
        return mapping

    def _derive(self, role, parameters, constants):
        if not self._raw[role]:
            native = dict(_ROLES)[role]
            return parameters[f"{native}.w"]
        execution = self._execution[role]
        draws = {
            index: constants[f"draw.{role}.{index}"]
            for index in self._draw_schemas[role]
        }
        value = functional_call(
            execution,
            self._mapping(role, parameters, constants, execution),
            (draws,),
            tie_weights=False,
        )
        if self._alternate_draws[role]:
            fresh_draws = {
                index: constants[f"fresh_draw.{role}.{index}"]
                for index in self._draw_schemas[role]
            }
            fresh = functional_call(
                execution,
                self._mapping(role, parameters, constants, execution),
                (fresh_draws,),
                tie_weights=False,
            )
            value = torch.where(constants[f"use_fresh_draw.{role}"], fresh, value)
        if (
            not torch.is_tensor(value)
            or (tuple(value.shape), value.dtype, value.device) != self._schemas[role]
        ):
            raise FunctionalizationError(
                f"Connection {role} materialization changed output shape, dtype, or device."
            )
        return value

    def materialize(self, parameters, constants):
        self.validate_source()
        self._validate_packet(parameters, constants)
        values = {}
        for role, _native in _ROLES:
            if not self._raw[role]:
                values[role] = self._derive(role, parameters, constants)
                continue
            baseline = {
                key: constants[f"baseline.{role}.parameters.{key}"]
                for key in set(self._role_parameter_aliases[role].values())
            }
            restored = restore_population_cache_values(
                parameters, baseline, parameter_names=baseline
            )
            retain = constants[f"retain_{role}"]
            selected_parameters = {
                **parameters,
                **{
                    key: torch.where(retain, restored[key], parameters[key])
                    for key in baseline
                },
            }
            constant_baseline = {
                key: constants[f"baseline.{role}.constants.{key}"]
                for key in set(self._role_constant_aliases[role].values())
            }
            # A local config buffer can be a selectable raw source elsewhere
            # in a Network. Retention evaluates its stored source value while
            # preserving that shared input's reverse- and forward-mode identity.
            restored_constants = restore_population_cache_values(
                constants, constant_baseline, parameter_names=constant_baseline
            )
            selected_constants = {
                **constants,
                **{
                    key: torch.where(
                        retain,
                        restored_constants[key],
                        constants[key],
                    )
                    for key in constant_baseline
                },
            }
            # Select raw/config inputs before invoking authored transforms. An
            # inactive invalid source derivative must not poison retained
            # native values through a later where-backward.
            derived = self._derive(role, selected_parameters, selected_constants)
            use_anchor = constants[f"use_offset.{role}"]
            # Subtract the effective source baseline before adding native
            # storage. Storing (native - derived) as an offset loses small
            # native values when the source baseline is much larger.
            baseline_effective = torch.where(
                use_anchor,
                constants[f"baseline_effective.{role}"],
                torch.zeros_like(constants[f"baseline_effective.{role}"]),
            )
            stored = torch.where(
                use_anchor,
                constants[f"stored.{role}"],
                torch.zeros_like(constants[f"stored.{role}"]),
            )
            anchored = torch.where(
                derived == baseline_effective,
                stored,
                stored + (derived - baseline_effective),
            )
            # Preserve the exact snapshot value even for equal nonfinite
            # sources while retaining the current raw source's Jacobian.
            current = restore_population_cache_values(
                {"effective": derived},
                {"effective": torch.where(use_anchor, anchored, derived)},
                parameter_names=("effective",),
            )["effective"]
            retained = restore_population_cache_values(
                {"effective": derived},
                {"effective": constants[f"stored.{role}"]},
                parameter_names=("effective",),
            )["effective"]
            values[role] = torch.where(retain, retained, current)
            if self._draw_schemas[role]:
                # Old native checkpoints contain initialized values without
                # their random base draws. Preserve those values without
                # inventing a pathwise derivative to a newly guessed draw.
                values[role] = torch.where(
                    constants[f"draw_valid.{role}"],
                    values[role],
                    constants[f"stored.{role}"],
                )
        return values

    def initialize_constants(
        self,
        parameters,
        constants,
        *,
        reinit_weights=True,
        reinit_delays=True,
        draws=None,
    ):
        self.validate_source()
        self._validate_packet(parameters, constants)
        flags = {"weights": reinit_weights, "delays": reinit_delays}
        if any(type(value) is not bool for value in flags.values()):
            raise TypeError("Connection reinitialization flags must be booleans.")
        if draws is not None and (
            not isinstance(draws, Mapping) or not set(draws).issubset(flags)
        ):
            raise FunctionalizationError(
                "Connection initialization draws need weight/delay role mappings."
            )
        for role, refresh in flags.items():
            if refresh and not self._raw[role]:
                raise FunctionalizationError(
                    f"Fresh custom-distribution connection {role} sampling is currently unsupported; use reinit_{role}=False with initialized fixed storage."
                )
        result = dict(constants)
        for role, refresh in flags.items():
            if not refresh:
                result[f"retain_{role}"] = torch.ones_like(constants[f"retain_{role}"])
                continue
            if self._draw_schemas[role]:
                role_draws = None if draws is None else draws.get(role)
                self._validate_draws(role, role_draws)
                prefix = "fresh_draw" if self._alternate_draws[role] else "draw"
                for index, draw in role_draws.items():
                    result[f"{prefix}.{role}.{index}"] = _clone(draw)
                if self._alternate_draws[role]:
                    result[f"use_fresh_draw.{role}"] = torch.ones_like(
                        constants[f"use_fresh_draw.{role}"]
                    )
                result[f"draw_valid.{role}"] = torch.ones_like(
                    constants[f"draw_valid.{role}"]
                )
            value = self._derive(role, parameters, result)
            for key in set(self._role_parameter_aliases[role].values()):
                result[f"baseline.{role}.parameters.{key}"] = _clone(parameters[key])
            for key in set(self._role_constant_aliases[role].values()):
                result[f"baseline.{role}.constants.{key}"] = _clone(constants[key])
            result[f"stored.{role}"] = _clone(value)
            result[f"baseline_effective.{role}"] = _clone(value)
            result[f"retain_{role}"] = torch.zeros_like(constants[f"retain_{role}"])
            result[f"use_offset.{role}"] = torch.zeros_like(
                constants[f"use_offset.{role}"]
            )
        return result

    def requires_initialization_draw(self, *, reinit_weights=True, reinit_delays=True):
        flags = {"weights": reinit_weights, "delays": reinit_delays}
        if any(type(value) is not bool for value in flags.values()):
            raise TypeError("Connection reinitialization flags must be booleans.")
        return any(
            flags[role] and schemas for role, schemas in self._draw_schemas.items()
        )

    def _validate_draws(self, role, draws):
        schemas = self._draw_schemas[role]
        if not isinstance(draws, Mapping) or set(draws) != set(schemas):
            raise FunctionalizationError(
                f"Connection {role} initialization needs explicit base draws for blocks {tuple(schemas)}."
            )
        for index, (shape, dtype, device) in schemas.items():
            value = draws[index]
            if not torch.is_tensor(value) or (
                tuple(value.shape),
                value.dtype,
                value.device,
            ) != (tuple(shape), dtype, device):
                raise FunctionalizationError(
                    f"Connection {role} base draw {index!r} changed tensor schema."
                )

    def sample_initialization_draw(
        self,
        parameters,
        constants,
        *,
        reinit_weights=True,
        reinit_delays=True,
        generators=None,
    ):
        """Cold adapter consuming the same streams as native block sampling.

        ``generators`` maps device strings to private generators. Omitting it
        consumes Torch's global streams, as native initialization does.
        Numerical initialization uses ``initialize_constants`` and explicit
        draws, without consuming any generator.
        """
        self.validate_source()
        self._validate_packet(parameters, constants)
        self.requires_initialization_draw(
            reinit_weights=reinit_weights, reinit_delays=reinit_delays
        )
        flags = {"weights": reinit_weights, "delays": reinit_delays}
        result = {"weights": {}, "delays": {}}
        generator_frames = ()
        if self._check_purity and generators is not None:
            provided = (
                generators.values()
                if isinstance(generators, Mapping)
                else (generators,)
            )
            generator_frames = tuple(
                {
                    id(generator): (generator, generator.get_state().clone())
                    for generator in provided
                    if isinstance(generator, torch.Generator)
                }.values()
            )
        for role, refresh in flags.items():
            if not refresh or not self._draw_schemas[role]:
                continue
            execution = _SampleDraws(self._execution[role].source)
            mapping = self._mapping(role, parameters, constants, execution)
            p = _helpers()
            device = self._schemas[role][2]
            rng = (
                p._global_rng_snapshot(device)
                if self._check_purity and generators is not None
                else None
            )
            before = (
                _binding(execution, identities=False) if self._check_purity else None
            )
            tensors = (
                {
                    name: (value._version, _clone(value))
                    for name, value in mapping.items()
                }
                if self._check_purity
                else {}
            )
            try:
                draws = functional_call(
                    execution,
                    mapping,
                    (generators,),
                    tie_weights=False,
                )
                if self._check_purity:
                    if rng is not None and p._global_rng_changes(rng, device):
                        raise FunctionalizationError(
                            "Connection sampler purity diagnostics observed implicit random draws instead of the supplied generator."
                        )
                    if before != _binding(execution, identities=False):
                        raise FunctionalizationError(
                            "Connection sampler purity diagnostics observed mutable state."
                        )
                    for name, value in mapping.items():
                        version, saved = tensors[name]
                        if value._version != version or not torch.equal(value, saved):
                            raise FunctionalizationError(
                                "Connection sampler purity diagnostics observed an in-place tensor write."
                            )
                self._validate_draws(role, draws)
            except Exception:
                # A diagnostic failure must not consume a caller's explicit
                # stream, including draws from an earlier successful role.
                for generator, frame in generator_frames:
                    generator.set_state(frame)
                raise
            finally:
                if rng is not None:
                    p._restore_global_rng(rng, device)
            result[role] = {index: _clone(value) for index, value in draws.items()}
        return result

    def prepare(self, parameters, constants):
        p = _helpers()
        # Prepared values support inspection and freshness checking. Each
        # consumption builds its own graph from the explicit raw leaves, so
        # one prepared packet can serve independent backward passes/chunks.
        values = {
            name: _clone(value)
            for name, value in self.materialize(parameters, constants).items()
        }
        return _PreparedConnectionParameters(
            self._token,
            p._tensor_binding({key: parameters[key] for key in self._parameter_paths}),
            p._tensor_binding(constants),
            constants,
            p._tensor_tree_binding(values),
            values,
        )

    def validate_prepared(self, prepared, parameters):
        p = _helpers()
        self.validate_source()
        if (
            not isinstance(prepared, _PreparedConnectionParameters)
            or prepared.token is not self._token
        ):
            raise FunctionalizationError(
                "Prepared connection parameters belong to another plan."
            )
        if (
            prepared.parameter_binding
            != p._tensor_binding(
                {key: parameters[key] for key in self._parameter_paths}
            )
            or prepared.constant_binding != p._tensor_binding(prepared.constants)
            or prepared.value_binding != p._tensor_tree_binding(prepared.values)
        ):
            raise FunctionalizationError(
                "Connection parameters, constants, or prepared values changed; prepare again."
            )
        return self.materialize(parameters, prepared.constants)

    def _audit_sources(self):
        p = _helpers()
        for role, native in _ROLES:
            if not self._raw[role]:
                continue
            for source in getattr(self._source, native).weights:
                module = source.param
                if not source.deterministic():
                    from dendra.models.distributions import Normal, TruncatedNormal

                    from ._torch_parametrizations import canonical_type

                    if canonical_type(module) not in (Normal, TruncatedNormal):
                        p._initialization_callable_contract_signature(
                            module.sample_from_draw,
                            label=f"connection.{native}.sample_from_draw",
                            owner=module,
                            strict=True,
                            diagnostic=True,
                        )
                    continue
                if not isinstance(module, torch.nn.Module):
                    continue
                direct_compute = _connection_source_can_compute_uncached(module)
                method = (
                    module._compute
                    if direct_compute
                    else (module.repeat if is_parametric(module) else module.forward)
                )
                canonical_compute = direct_compute and inspect.getattr_static(
                    type(module), "_compute"
                ) in (
                    Bounded._compute,
                    NegativeParam._compute,
                )
                if not canonical_compute:
                    p._initialization_callable_contract_signature(
                        method,
                        label=f"connection.{native}",
                        owner=module,
                        strict=True,
                        diagnostic=True,
                    )
            execution = self._execution[role]
            device = self._schemas[role][2]
            rng = p._global_rng_snapshot(device)
            before = _binding(execution, identities=False)
            tensors = {
                name: (value._version, _clone(value))
                for name, value in p._module_tensor_slots(execution)
            }
            try:
                with torch.no_grad():
                    draws = {
                        index: self._placeholder_draw(
                            role, index, dtype=dtype, device=device
                        )
                        for index, (shape, dtype, device) in self._draw_schemas[
                            role
                        ].items()
                    }
                    input_draws = {
                        index: (value._version, _clone(value))
                        for index, value in draws.items()
                    }
                    first, second = execution(draws), execution(draws)
                if p._global_rng_changes(rng, device):
                    raise FunctionalizationError(
                        "Connection parameter purity diagnostics observed implicit random draws."
                    )
                if before != _binding(execution, identities=False) or not torch.equal(
                    first, second
                ):
                    raise FunctionalizationError(
                        "Connection parameter purity diagnostics observed mutable or nondeterministic state."
                    )
                for index, value in draws.items():
                    version, saved = input_draws[index]
                    if value._version != version or not torch.equal(value, saved):
                        raise FunctionalizationError(
                            "Connection parameter purity diagnostics observed an in-place draw write."
                        )
                for name, value in p._module_tensor_slots(execution):
                    version, saved = tensors[name]
                    if value._version != version or not torch.equal(value, saved):
                        raise FunctionalizationError(
                            "Connection parameter purity diagnostics observed an in-place tensor write."
                        )
            finally:
                p._restore_global_rng(rng, device)
