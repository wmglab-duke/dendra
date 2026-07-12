"""Generic population-wide material fields.

This module contains the light-weight, ion-agnostic Material container used by
mechanisms and material processes.  A Material owns one or more tensor fields
that are present over the full population shape.  Mechanisms bind local views of
those fields through Mechanism.USEMATERIAL(...), while process-level dynamics
such as diffusion operate on the full fields directly.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from dendra.helpers import DEBUG

from ..parametric import resolve, to_param


@dataclass(frozen=True)
class MaterialFieldSpec:
    """Declaration for one population-wide material field."""

    name: str
    initial: object = 0.0
    min_value: float | None = None
    conserved: bool = True
    domain: str = "i"
    units: str | None = None


MATERIAL_SPECS: dict[str, dict[str, MaterialFieldSpec]] = {}
MATERIAL_ALIASES: dict[str, str] = {}


def _canonical_material_name(name: str) -> str:
    name = str(name)
    return MATERIAL_ALIASES.get(name, name)


def _make_into_shape(shape, value, *, like: torch.Tensor | None = None):
    """Resolve and broadcast one material initial-value source."""
    value = resolve(value)
    if not torch.is_tensor(value):
        if like is None:
            value = torch.as_tensor(value)
        else:
            value = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    elif like is not None:
        value = value.to(device=like.device, dtype=like.dtype)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(()).expand(shape).clone()
    return value.expand(shape).clone()


def _normalize_field_specs(
    fields=None,
    *,
    initial_values: Mapping[str, object] | None = None,
    min_values: Mapping[str, float | None] | None = None,
    conserved: Mapping[str, bool] | bool = True,
    domain: Mapping[str, str] | str = "i",
    units: Mapping[str, str | None] | str | None = None,
) -> dict[str, MaterialFieldSpec]:
    """Normalize user field declarations into MaterialFieldSpec objects."""
    initial_values = dict(initial_values or {})
    min_values = dict(min_values or {})

    if fields is None:
        # Infer every declared field without losing minimum-only declarations.
        # Keep the caller's initial-value order, then append new minimum keys in
        # their own insertion order for deterministic construction/state dicts.
        names = list(initial_values)
        names.extend(name for name in min_values if name not in initial_values)
        field_initials = {name: initial_values.get(name, 0.0) for name in names}
    elif isinstance(fields, Mapping):
        field_initials = dict(fields)
        for name, value in initial_values.items():
            field_initials[name] = value
    elif isinstance(fields, str):
        field_initials = {fields: initial_values.get(fields, 0.0)}
    else:
        field_initials = {
            str(name): initial_values.get(str(name), 0.0) for name in fields
        }

    def pick(obj, key, default):
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return obj

    specs = {}
    for name, init in field_initials.items():
        name = str(name)
        specs[name] = MaterialFieldSpec(
            name=name,
            initial=init,
            min_value=min_values.get(name, None),
            conserved=bool(pick(conserved, name, True)),
            domain=str(pick(domain, name, "i")),
            units=pick(units, name, None),
        )
    return specs


def register_material(
    name: str,
    fields=None,
    *,
    initial_values: Mapping[str, object] | None = None,
    min_values: Mapping[str, float | None] | None = None,
    conserved: Mapping[str, bool] | bool = True,
    domain: Mapping[str, str] | str = "i",
    units: Mapping[str, str | None] | str | None = None,
    aliases: Sequence[str] | None = None,
):
    """Register a named material type and its default field declarations.

    Registration is optional: callers may instantiate Material directly with
    explicit fields.  Registering is useful when a Population convenience API
    wants to construct the material by name.
    """
    canonical = str(name)
    specs = _normalize_field_specs(
        fields,
        initial_values=initial_values,
        min_values=min_values,
        conserved=conserved,
        domain=domain,
        units=units,
    )
    MATERIAL_SPECS[canonical] = specs
    if isinstance(aliases, str):
        aliases = (aliases,)
    for alias in aliases or ():
        MATERIAL_ALIASES[str(alias)] = canonical
    return specs


def material_specs():
    """Return the registered material field specs."""
    return MATERIAL_SPECS


def valid_materials():
    """Return names of registered generic materials."""
    return list(MATERIAL_SPECS.keys())


class material_defaults(ContextDecorator):
    """Temporarily update registered material initial values.

    This mirrors the existing ion concentration/equilibrium context managers but
    is intentionally generic.  Only fields that already exist in the registered
    material spec may be updated.
    """

    _last = {}

    def __init__(self, material: str, use_last=False, **field_updates):
        material = _canonical_material_name(material)
        if material not in MATERIAL_SPECS:
            raise ValueError(
                f"Unknown material {material!r}. Registered materials: {valid_materials()}."
            )
        unknown = [k for k in field_updates if k not in MATERIAL_SPECS[material]]
        if unknown:
            raise ValueError(f"Unknown field(s) for material {material!r}: {unknown}")
        self.material = material
        self.updates = field_updates
        if not use_last:
            material_defaults._last = {
                (material, k): v for k, v in field_updates.items()
            }
            if DEBUG:
                print(material_defaults._last)
        if use_last:
            for (mat, key), value in material_defaults._last.items():
                if mat == material:
                    self.updates[key] = value
            if DEBUG:
                print(self.updates)
            material_defaults._last = {}
        self._original_stack = []

    def __enter__(self):
        specs = MATERIAL_SPECS[self.material]
        original = {}
        for field, value in self.updates.items():
            old = specs[field]
            original[field] = old
            specs[field] = MaterialFieldSpec(
                name=old.name,
                initial=value,
                min_value=old.min_value,
                conserved=old.conserved,
                domain=old.domain,
                units=old.units,
            )
        self._original_stack.append(original)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        MATERIAL_SPECS[self.material].update(self._original_stack.pop())
        return False


class Material(torch.nn.Module):
    """Population-wide material field container.

    A Material is present over the full compartmental shape of a population.
    Mechanisms read/write local views using Mechanism.USEMATERIAL(...), while
    MaterialProcess subclasses can operate on full fields directly.

    Each field has two distinct roles: :attr:`field` is mutable runtime state,
    while :meth:`initial_source` returns its registered initial-value parameter
    or parametric module. :meth:`initialize` resolves that source afresh. This
    preserves gradients to trainable initial values in training mode; evaluation
    initialization keeps the historical detached-runtime-state behavior.
    """

    __constants__ = ("name",)

    def __init__(
        self,
        name: str,
        shape,
        fields=None,
        *,
        initial_values: Mapping[str, object] | None = None,
        min_values: Mapping[str, float | None] | None = None,
        specs: Mapping[str, MaterialFieldSpec] | None = None,
        _register_initial_sources: bool = True,
    ):
        super().__init__()
        self.name = _canonical_material_name(name)

        if specs is None:
            if (
                fields is None
                and initial_values is None
                and self.name in MATERIAL_SPECS
            ):
                specs = MATERIAL_SPECS[self.name]
            else:
                specs = _normalize_field_specs(
                    fields,
                    initial_values=initial_values,
                    min_values=min_values,
                )
        else:
            specs = dict(specs)

        if not specs:
            raise ValueError(
                f"Material {self.name!r} has no fields. Pass fields=..., "
                "initial_values=..., or register the material before construction."
            )

        self._field_specs = dict(specs)
        self._material_fields = tuple(specs.keys())
        self._material_min_fields = tuple(
            field for field, spec in specs.items() if spec.min_value is not None
        )
        self._material_has_initial_sources = bool(_register_initial_sources)
        if self._material_has_initial_sources:
            self._initial_sources = torch.nn.Module()

        for field, spec in specs.items():
            init = to_param(spec.initial)
            if self._material_has_initial_sources:
                setattr(self._initial_sources, field, init)
                init_source = getattr(self._initial_sources, field)
            else:
                # Ion owns its historical e_init/i_init/o_init parameters and
                # deliberately opts out of generic Material initial sources.
                init_source = init
            init_t = _make_into_shape(shape, init_source)
            self.register_buffer(field, init_t.clone())
            if not _register_initial_sources:
                self.register_buffer(
                    f"_initial_{field}",
                    init_t.detach().clone(),
                    persistent=False,
                )
            if spec.min_value is not None:
                self.register_buffer(
                    f"_min_{field}",
                    torch.as_tensor(
                        spec.min_value, dtype=init_t.dtype, device=init_t.device
                    ),
                    persistent=False,
                )

    @property
    def fields(self) -> tuple[str, ...]:
        return self._material_fields

    def has_field(self, field: str) -> bool:
        return str(field) in self._buffers and str(field) in self._material_fields

    def field_spec(self, field: str) -> MaterialFieldSpec:
        return self._field_specs[str(field)]

    def initial_source(self, field: str):
        """Return the registered parameter/module used to reset ``field``.

        The returned object is the source itself, not its resolved tensor value.
        It therefore remains visible to :meth:`named_parameters`, follows
        device/dtype moves, and is serialized by :meth:`state_dict`.
        """
        field = str(field)
        if not self._material_has_initial_sources or field not in self._material_fields:
            raise KeyError(
                f"Material {self.name!r} has no generic initial source for "
                f"field {field!r}."
            )
        return getattr(self._initial_sources, field)

    def initialize(self, *args, **kwargs) -> None:
        """Reset material fields from their registered initial-value sources."""
        for field in self._material_fields:
            current = self._buffers[field]
            if self._material_has_initial_sources:
                source = self.initial_source(field)
            else:
                source = self._buffers[f"_initial_{field}"]
            self._buffers[field] = _make_into_shape(
                current.shape,
                source,
                like=current,
            )
        self.advance(*args, **kwargs)
        if not self.training:
            self.detach()

    def detach(self):
        for field in self._material_fields:
            self._buffers[field] = self._buffers[field].detach()
        return self

    def advance(self, *args, **kwargs) -> None:
        """Apply generic field guards/derived updates.

        Generic materials only enforce minimum values.  Subclasses such as Ion
        override this to update derived fields like reversal potentials.
        """
        for field in self._material_min_fields:
            value = self._buffers[field]
            min_value = self._buffers[f"_min_{field}"].to(
                device=value.device, dtype=value.dtype
            )
            self._buffers[field] = torch.where(value <= min_value, min_value, value)
