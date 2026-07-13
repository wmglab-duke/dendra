"""MaterialProcess and initial material diffusion implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from dendra.models._class_declarations import (
    consume_class_values,
    declare_class_value,
)

from ._materials import _canonical_material_name
from ._mechanism import Mechanism
from ._spatial import SpatialOperator1D, SpatialOperatorTree


@dataclass(frozen=True)
class DiffusionSpec:
    material: str
    field: str
    D: Any
    domain: str | None = None


@dataclass(frozen=True)
class ClearanceSpec:
    material: str
    field: str
    rate: Any
    target: Any = 0.0
    domain: str | None = None


@dataclass(frozen=True)
class ClampSpec:
    material: str
    field: str
    value: Any = 0.0
    where: Any = "all"
    mode: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class ExchangeSpec:
    material_a: str
    field_a: str
    material_b: str
    field_b: str
    rate: Any | None = None
    conductance: Any | None = None
    volume_a: Any | None = None
    volume_b: Any | None = None
    domain_a: str | None = None
    domain_b: str | None = None


def _canonical_domain(domain: str | None) -> str | None:
    if domain is None:
        return None
    d = str(domain).lower()
    aliases = {
        "i": "intracellular",
        "inside": "intracellular",
        "intra": "intracellular",
        "cytosol": "intracellular",
        "cytosolic": "intracellular",
        "intracellular": "intracellular",
        "o": "extracellular",
        "outside": "extracellular",
        "extra": "extracellular",
        "extracellular": "extracellular",
        "membrane": "membrane",
        "surface": "membrane",
    }
    return aliases.get(d, d)


def _copy_method_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(kwargs or {})


def _safe_key(index: int, material: str, field: str) -> str:
    raw = f"{index}_{material}_{field}"
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)


def _material_field_ref(ref, *, field=None) -> tuple[str, str]:
    """Normalize a material/field reference.

    Accepted forms are ``("ca", "cai")``, ``"ca.cai"``, or a material
    name such as ``"ca"`` with an optional ``field=...`` override.  If the
    field is omitted, it defaults to ``f"{material}i"``.
    """
    if isinstance(ref, Mapping):
        material = ref.get("material", ref.get("name", None))
        field_ = ref.get("field", field)
        if material is None:
            raise ValueError(
                "Material reference mapping must contain 'material' or 'name'."
            )
        material = str(material)
        if field_ is None:
            field_ = f"{material}i"
        return material, str(field_)

    if isinstance(ref, (tuple, list)) and len(ref) == 2:
        return str(ref[0]), str(ref[1])

    if isinstance(ref, str):
        if "." in ref and field is None:
            material, field_ = ref.split(".", 1)
            return str(material), str(field_)
        material = str(ref)
        field_ = field if field is not None else f"{material}i"
        return material, str(field_)

    raise ValueError(
        "Material field reference must be ('material', 'field'), 'material.field', "
        "or a material name with field=... ."
    )


class _MaterialFieldTransaction:
    """Stage full-field replacements and publish them only after all specs pass.

    Material process updates are intentionally out-of-place, so retaining each
    staged tensor is sufficient to preserve sequential multi-spec semantics
    without mutating the bound ``Material`` objects before the whole process has
    completed successfully.
    """

    def __init__(self):
        self._pending: dict[tuple[int, str], tuple[Any, str, torch.Tensor]] = {}

    @staticmethod
    def _key(material, field: str) -> tuple[int, str]:
        return id(material), str(field)

    def read(self, material, field: str) -> torch.Tensor:
        pending = self._pending.get(self._key(material, field))
        if pending is not None:
            return pending[2]
        return material._buffers[str(field)]

    def write(self, material, field: str, value: torch.Tensor) -> None:
        field = str(field)
        self._pending[self._key(material, field)] = (material, field, value)

    def commit(self) -> None:
        for material, field, value in self._pending.values():
            material._buffers[field] = value


class MaterialProcess(Mechanism):
    """Base class for full-field processes over shared Material objects.

    A MaterialProcess is inserted like a mechanism but scheduled in the material
    phase.  It binds full population-wide Material fields instead of local views.
    """

    _material_process_phase = "post_local"
    _material_process_method = "none"
    _material_process_method_kwargs: dict[str, Any] = {}

    _phase_declarations = []
    _method_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        phase = "post_local"
        method = "none"
        method_kwargs: dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            if "_material_process_phase" in base.__dict__:
                phase = base._material_process_phase
            if "_material_process_method" in base.__dict__:
                method = base._material_process_method
            if "_material_process_method_kwargs" in base.__dict__:
                method_kwargs = dict(base._material_process_method_kwargs)

        phase_declarations = consume_class_values(
            cls, "material_process.phase", MaterialProcess._phase_declarations
        )
        if phase_declarations:
            phase = phase_declarations[-1]
        method_declarations = consume_class_values(
            cls, "material_process.method", MaterialProcess._method_declarations
        )
        if method_declarations:
            method, method_kwargs = method_declarations[-1]

        cls._material_process_phase = str(phase)
        cls._material_process_method = str(method).lower()
        cls._material_process_method_kwargs = dict(method_kwargs)

    @staticmethod
    def METHOD(method="none", **kwargs):
        """Declare the process-level numerical method."""
        declare_class_value(
            "material_process.method",
            (str(method).lower(), dict(kwargs)),
            MaterialProcess._method_declarations,
        )

    @staticmethod
    def PHASE(phase="post_local"):
        """Declare the material scheduler phase for this process."""
        declare_class_value(
            "material_process.phase", str(phase), MaterialProcess._phase_declarations
        )

    def bind_materials(self, material_resolver, *, population=None):
        """Bind process to the Material registry and population geometry."""
        self._material_resolver = material_resolver
        self._population_shape = None if population is None else tuple(population.shape)
        self.configure_process(population)
        return self

    def configure_process(self, population=None):
        """Hook called once after full Material/Population binding."""
        return None

    def _get_material(self, name: str):
        try:
            resolver = self._material_resolver
        except AttributeError as exc:
            raise RuntimeError(
                f"MaterialProcess {self.name!r} has not been bound to materials yet."
            ) from exc
        requested = str(name)
        canonical = _canonical_material_name(requested)
        try:
            return resolver(canonical)
        except KeyError:
            if canonical != requested:
                return resolver(requested)
            raise

    def material_field(self, material: str, field: str) -> torch.Tensor:
        return self._get_material(material)._buffers[str(field)]

    def set_material_field(
        self, material: str, field: str, value: torch.Tensor
    ) -> None:
        self._get_material(material)._buffers[str(field)] = value

    def _resolve_quantity(self, value, like: torch.Tensor, *, what: str = "quantity"):
        """Resolve a scalar/tensor/process parameter into a tensor like ``like``.

        MaterialProcess subclasses use this for declarations such as
        ``rate="kclear"`` or ``target="ko_bath"``.  A string first resolves
        as an attribute, then as a registered buffer, so RANGE/GLOBAL parameters
        and process-local buffers are both supported.
        """
        if isinstance(value, str):
            if hasattr(self, value):
                return getattr(self, value).to(device=like.device, dtype=like.dtype)
            try:
                return self._buffers[value].to(device=like.device, dtype=like.dtype)
            except KeyError as exc:
                raise AttributeError(
                    f"{type(self).__name__} {self.name!r} expected a parameter/buffer "
                    f"named {value!r} for {what}. Declare it with RANGE/GLOBAL "
                    "or pass a scalar/tensor value."
                ) from exc
        if torch.is_tensor(value):
            return value.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(value, device=like.device, dtype=like.dtype)

    def advance_materials(self, dt):
        raise NotImplementedError(
            f"{type(self).__name__}.advance_materials(dt) must be implemented."
        )


class ClearanceProcess(MaterialProcess):
    """Pointwise first-order clearance/relaxation of full Material fields.

    A ClearanceProcess applies

        c' = -rate * (c - target)

    to one or more population-wide material fields.  It is intended for generic
    decay, degradation, uptake, bath relaxation, and recovery-to-baseline
    dynamics that do not require spatial coupling.  The default method is the
    exact exponential update, which is stable for non-negative rates and fixed
    targets over the timestep.
    """

    _clearance_specs: tuple[ClearanceSpec, ...] = tuple()
    _clearance_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        specs: list[ClearanceSpec] = []
        for base in reversed(cls.__mro__):
            if "_clearance_specs" in base.__dict__:
                specs.extend(list(base._clearance_specs))
        specs.extend(
            consume_class_values(
                cls,
                "clearance_process.specs",
                ClearanceProcess._clearance_declarations,
            )
        )
        cls._clearance_specs = tuple(specs)

    @staticmethod
    def CLEAR(material, *, field=None, rate=None, target=0.0, domain=None):
        """Declare a first-order clearance/relaxation for a material field.

        Parameters
        ----------
        material
            Material name, e.g. ``"ip3"`` or ``"k"``.
        field
            Field name, e.g. ``"ip3i"`` or ``"ko"``.  If omitted, defaults to
            ``f"{material}i"``.
        rate
            First-order rate in ``1/ms``.  May be a scalar/tensor or the name of
            a process parameter, e.g. ``rate="kclear"``.  If omitted, defaults to
            the parameter name ``"rate"``.
        target
            Relaxation target.  May be a scalar/tensor or the name of a process
            parameter, e.g. ``target="ko_bath"``.
        domain
            Optional domain metadata override.  Clearance is pointwise, so the
            domain is currently validation metadata rather than geometry input.
        """
        material = str(material)
        if field is None:
            field = f"{material}i"
        if rate is None:
            rate = "rate"
        spec = ClearanceSpec(
            str(material),
            str(field),
            rate,
            target,
            None if domain is None else str(domain),
        )
        declare_class_value(
            "clearance_process.specs", spec, ClearanceProcess._clearance_declarations
        )

    @staticmethod
    def DECAY(material, *, field=None, rate=None, target=0.0, domain=None):
        """Alias for :meth:`CLEAR`."""
        ClearanceProcess.CLEAR(
            material,
            field=field,
            rate=rate,
            target=target,
            domain=domain,
        )

    @staticmethod
    def RELAX(material, *, field=None, rate=None, target=0.0, domain=None):
        """Alias for :meth:`CLEAR`."""
        ClearanceProcess.CLEAR(
            material,
            field=field,
            rate=rate,
            target=target,
            domain=domain,
        )

    def configure_process(self, population=None):
        if self.key is not None:
            raise NotImplementedError(
                "ClearanceProcess MVP must be inserted globally. Region-restricted "
                "clearance will require explicit masks or restricted process operators."
            )

        kwargs = _copy_method_kwargs(type(self)._material_process_method_kwargs)
        method = str(type(self)._material_process_method or "exact").lower()
        if method in {"none", ""}:
            method = "exact"
        aliases = {
            "exp": "exact",
            "exponential": "exact",
            "cnexp": "exact",
            "rush_larsen": "exact",
            "be": "implicit",
            "backward_euler": "implicit",
            "bwd_euler": "implicit",
            "forward_euler": "explicit",
            "euler": "explicit",
        }
        self._clearance_method = aliases.get(method, method)
        if self._clearance_method not in {"exact", "implicit", "explicit"}:
            raise NotImplementedError(
                "ClearanceProcess supports METHOD('exact'), METHOD('implicit'), "
                "and METHOD('explicit')."
            )

        self._clearance_domain = kwargs.get("domain", None)
        self._clearance_allow_negative_rate = bool(
            kwargs.get("allow_negative_rate", False)
        )

        if not type(self)._clearance_specs:
            raise ValueError(
                f"{type(self).__name__} declares no cleared fields. Add "
                "ClearanceProcess.CLEAR(...)."
            )
        self._validate_clearance_specs()
        return None

    def _effective_clearance_domain(self, material, spec: ClearanceSpec) -> str | None:
        domain = spec.domain
        if domain is None:
            domain = self._clearance_domain
        if domain is None:
            try:
                domain = material.field_spec(spec.field).domain
            except Exception:
                domain = None
        return _canonical_domain(domain)

    def _validate_clearance_specs(self):
        for spec in type(self)._clearance_specs:
            material = self._get_material(spec.material)
            if not material.has_field(spec.field):
                raise ValueError(
                    f"Material {spec.material!r} has no field {spec.field!r}. "
                    f"Available fields: {material.fields}."
                )
            # Domain is intentionally not restricted: pointwise clearance can be
            # applied to intracellular, extracellular, membrane, or custom fields.
            self._effective_clearance_domain(material, spec)

    def advance_materials(self, dt):
        dt_t = None
        transaction = _MaterialFieldTransaction()
        for spec in type(self)._clearance_specs:
            material = self._get_material(spec.material)
            c = transaction.read(material, spec.field)
            rate = self._resolve_quantity(spec.rate, c, what="clearance rate")
            target = self._resolve_quantity(spec.target, c, what="clearance target")

            if dt_t is None:
                dt_t = torch.as_tensor(dt, device=c.device, dtype=c.dtype)
            else:
                dt_t = dt_t.to(device=c.device, dtype=c.dtype)

            if not self._clearance_allow_negative_rate:
                rate = torch.clamp_min(rate, 0.0)

            if self._clearance_method == "explicit":
                c_new = c + dt_t * (-rate * (c - target))
            elif self._clearance_method == "implicit":
                c_new = (c + dt_t * rate * target) / (1.0 + dt_t * rate)
            else:  # exact exponential relaxation
                c_new = target + (c - target) * torch.exp(-rate * dt_t)

            transaction.write(material, spec.field, c_new)
        transaction.commit()


class ClampProcess(MaterialProcess):
    """Pointwise hard or bound clamp of full Material fields.

    A ClampProcess imposes an externally prescribed value or bound on one or
    more population-wide material fields.  It is intended for bath conditions,
    boundary conditions after transport, fixed experimental fields, and safety
    bounds that should be applied as a well-defined material-process phase.

    The default phase is ``post_transport`` so a clamp inserted with diffusion
    behaves like a boundary/bath condition: local reactions run first, transport
    runs second, and the clamp is imposed last.  Override with
    ``ClampProcess.PHASE(...)`` if a different ordering is desired.
    """

    _material_process_phase = "post_transport"
    _clamp_specs: tuple[ClampSpec, ...] = tuple()
    _clamp_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        specs: list[ClampSpec] = []
        for base in reversed(cls.__mro__):
            if "_clamp_specs" in base.__dict__:
                specs.extend(list(base._clamp_specs))
        specs.extend(
            consume_class_values(
                cls, "clamp_process.specs", ClampProcess._clamp_declarations
            )
        )
        cls._clamp_specs = tuple(specs)

    @staticmethod
    def PHASE(phase="post_transport"):
        """ClampProcess defaults to the post-transport material phase."""
        MaterialProcess.PHASE(phase)

    @staticmethod
    def CLAMP(material, *, field=None, value=0.0, where="all", mode=None, domain=None):
        """Declare a clamp for a material field.

        Parameters
        ----------
        material
            Material name, e.g. ``"ip3"``, ``"k"``, or ``"ca"``.
        field
            Field name, e.g. ``"ip3i"``, ``"ko"``, or ``"cai"``.  If omitted,
            defaults to ``f"{material}i"``.
        value
            Clamp value or bound.  May be a scalar/tensor or the name of a
            process parameter/buffer.  For ``mode="range"``, pass a two-element
            tuple/list ``(lower, upper)`` whose entries may also be parameter
            names.
        where
            Region where the clamp applies.  Supported MVP values include
            ``"all"``, ``"terminal"``/``"boundary"``, ``"root"``/``"proximal"``,
            a boolean mask tensor, or an index/slice into the material field.
        mode
            ``"set"``/``"hard"`` sets the field to ``value``.  ``"min"`` imposes
            a lower bound.  ``"max"`` imposes an upper bound.  ``"range"`` imposes
            lower and upper bounds.
        domain
            Optional domain metadata override.  Clamp is pointwise, so this is
            currently validation/semantic metadata rather than geometry input.
        """
        material = str(material)
        if field is None:
            field = f"{material}i"
        spec = ClampSpec(
            str(material),
            str(field),
            value,
            where,
            None if mode is None else str(mode),
            None if domain is None else str(domain),
        )
        declare_class_value(
            "clamp_process.specs", spec, ClampProcess._clamp_declarations
        )

    @staticmethod
    def SET(material, *, field=None, value=0.0, where="all", domain=None):
        """Alias for a hard set clamp."""
        ClampProcess.CLAMP(
            material,
            field=field,
            value=value,
            where=where,
            mode="set",
            domain=domain,
        )

    @staticmethod
    def MIN(material, *, field=None, value=0.0, where="all", domain=None):
        """Declare a lower-bound clamp: ``field = max(field, value)``."""
        ClampProcess.CLAMP(
            material,
            field=field,
            value=value,
            where=where,
            mode="min",
            domain=domain,
        )

    @staticmethod
    def LOWER(material, *, field=None, value=0.0, where="all", domain=None):
        """Alias for :meth:`MIN`."""
        ClampProcess.MIN(material, field=field, value=value, where=where, domain=domain)

    @staticmethod
    def MAX(material, *, field=None, value=0.0, where="all", domain=None):
        """Declare an upper-bound clamp: ``field = min(field, value)``."""
        ClampProcess.CLAMP(
            material,
            field=field,
            value=value,
            where=where,
            mode="max",
            domain=domain,
        )

    @staticmethod
    def UPPER(material, *, field=None, value=0.0, where="all", domain=None):
        """Alias for :meth:`MAX`."""
        ClampProcess.MAX(material, field=field, value=value, where=where, domain=domain)

    @staticmethod
    def BOUNDS(
        material, *, field=None, lower=0.0, upper=None, where="all", domain=None
    ):
        """Declare a lower/upper bound clamp.

        ``lower`` and ``upper`` may be scalars/tensors or parameter names.  If
        ``upper`` is omitted, only the lower bound is applied through ``MIN``.

        This helper is intentionally named ``BOUNDS`` rather than ``RANGE`` so it
        does not shadow :meth:`Mechanism.RANGE`, which is still used to declare
        process parameters such as bath values.
        """
        if upper is None:
            ClampProcess.MIN(
                material, field=field, value=lower, where=where, domain=domain
            )
            return
        ClampProcess.CLAMP(
            material,
            field=field,
            value=(lower, upper),
            where=where,
            mode="range",
            domain=domain,
        )

    def configure_process(self, population=None):
        if self.key is not None:
            raise NotImplementedError(
                "ClampProcess MVP must be inserted globally. Region-restricted "
                "clamps should be expressed with the `where=` mask/index argument."
            )

        kwargs = _copy_method_kwargs(type(self)._material_process_method_kwargs)
        method = str(type(self)._material_process_method or "set").lower()
        if method in {"none", ""}:
            method = "set"
        aliases = {
            "hard": "set",
            "value": "set",
            "assign": "set",
            "assignment": "set",
            "lower": "min",
            "lower_bound": "min",
            "minimum": "min",
            "upper": "max",
            "upper_bound": "max",
            "maximum": "max",
            "bounds": "range",
            "clip": "range",
        }
        self._clamp_method = aliases.get(method, method)
        if self._clamp_method not in {"set", "min", "max", "range"}:
            raise NotImplementedError(
                "ClampProcess supports METHOD('set'), METHOD('min'), "
                "METHOD('max'), and METHOD('range')."
            )
        self._clamp_domain = kwargs.get("domain", None)

        if not type(self)._clamp_specs:
            raise ValueError(
                f"{type(self).__name__} declares no clamped fields. Add "
                "ClampProcess.CLAMP(...)."
            )

        self._clamp_operator_keys = tuple(
            _safe_key(i, spec.material, spec.field)
            for i, spec in enumerate(type(self)._clamp_specs)
        )
        self._clamp_mask_names: dict[str, str | None] = {}
        self._validate_and_configure_clamp_specs(population)
        return None

    def _effective_clamp_domain(self, material, spec: ClampSpec) -> str | None:
        domain = spec.domain
        if domain is None:
            domain = self._clamp_domain
        if domain is None:
            try:
                domain = material.field_spec(spec.field).domain
            except Exception:
                domain = None
        return _canonical_domain(domain)

    def _effective_clamp_mode(self, spec: ClampSpec) -> str:
        mode = spec.mode if spec.mode is not None else self._clamp_method
        mode = str(mode or "set").lower()
        aliases = {
            "hard": "set",
            "value": "set",
            "assign": "set",
            "assignment": "set",
            "lower": "min",
            "lower_bound": "min",
            "minimum": "min",
            "upper": "max",
            "upper_bound": "max",
            "maximum": "max",
            "bounds": "range",
            "clip": "range",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"set", "min", "max", "range"}:
            raise NotImplementedError(
                f"Unsupported clamp mode {mode!r}; expected 'set', 'min', 'max', or 'range'."
            )
        return mode

    def _validate_and_configure_clamp_specs(self, population=None):
        for key, spec in zip(self._clamp_operator_keys, type(self)._clamp_specs):
            material = self._get_material(spec.material)
            if not material.has_field(spec.field):
                raise ValueError(
                    f"Material {spec.material!r} has no field {spec.field!r}. "
                    f"Available fields: {material.fields}."
                )
            self._effective_clamp_domain(material, spec)
            self._effective_clamp_mode(spec)

            c = material._buffers[spec.field]
            mask = self._build_where_mask(spec.where, c, population=population)
            if mask is None:
                self._clamp_mask_names[key] = None
            else:
                mask_name = f"_clamp_mask_{key}"
                mask = mask.to(device=c.device, dtype=torch.bool)
                if mask_name in self._buffers:
                    self._buffers[mask_name] = mask
                else:
                    self.register_buffer(mask_name, mask)
                self._clamp_mask_names[key] = mask_name

    def _build_where_mask(self, where, like: torch.Tensor, *, population=None):
        if where is None:
            return None
        if isinstance(where, str):
            w = where.lower().replace("-", "_")
            if w in {"", "all", "everywhere", "global", "none"}:
                return None
            if w in {
                "terminal",
                "terminals",
                "leaf",
                "leaves",
                "boundary",
                "boundaries",
                "end",
                "ends",
            }:
                return self._terminal_mask(like, population)
            if w in {"root", "roots", "proximal", "soma"}:
                return self._root_mask(like, population)
            if hasattr(self, where):
                return torch.as_tensor(getattr(self, where), device=like.device).to(
                    dtype=torch.bool
                )
            if where in self._buffers:
                return self._buffers[where].to(device=like.device, dtype=torch.bool)
            raise ValueError(
                f"Unsupported clamp region {where!r}. Use 'all', 'terminal', 'root', "
                "a boolean mask tensor, or a process buffer/parameter name."
            )
        if torch.is_tensor(where):
            return where.to(device=like.device, dtype=torch.bool)
        if isinstance(where, bool):
            return None if where else torch.zeros_like(like, dtype=torch.bool)

        mask = torch.zeros_like(like, dtype=torch.bool)
        try:
            if isinstance(where, slice):
                mask[(..., where)] = True
            elif isinstance(where, tuple):
                mask[(...,) + where] = True
            else:
                mask[(..., where)] = True
        except Exception as exc:
            raise ValueError(
                f"Could not interpret clamp `where={where!r}` as a mask or index."
            ) from exc
        return mask

    def _terminal_mask(self, like: torch.Tensor, population=None):
        mask = torch.zeros_like(like, dtype=torch.bool)
        graph = getattr(population, "graph", None) if population is not None else None
        if graph is not None:
            graphs = graph if isinstance(graph, (list, tuple)) else [graph]
            # MaterialProcess MVP is global over one population.  If several
            # graphs are provided, use the first topology; multi-tree material
            # processes should get their own packed operator later.
            g = graphs[0]
            terminals = [int(n) for n in g.nodes if len(list(g.successors(n))) == 0]
            if terminals:
                mask[
                    ...,
                    torch.as_tensor(terminals, device=like.device, dtype=torch.long),
                ] = True
            return mask
        if like.shape[-1] == 0:
            return mask
        mask[..., 0] = True
        if like.shape[-1] > 1:
            mask[..., -1] = True
        return mask

    def _root_mask(self, like: torch.Tensor, population=None):
        mask = torch.zeros_like(like, dtype=torch.bool)
        graph = getattr(population, "graph", None) if population is not None else None
        if graph is not None:
            graphs = graph if isinstance(graph, (list, tuple)) else [graph]
            g = graphs[0]
            roots = [int(n) for n in g.nodes if len(list(g.predecessors(n))) == 0]
            if roots:
                mask[
                    ..., torch.as_tensor(roots, device=like.device, dtype=torch.long)
                ] = True
            return mask
        if like.shape[-1] > 0:
            mask[..., 0] = True
        return mask

    def _broadcast_mask(self, mask, c: torch.Tensor):
        if mask is None:
            return None
        mask = mask.to(device=c.device, dtype=torch.bool)
        while mask.ndim < c.ndim:
            mask = mask.unsqueeze(0)
        return mask.expand_as(c)

    def _apply_clamp_mode(self, c: torch.Tensor, spec: ClampSpec, mode: str):
        if mode == "range":
            if not isinstance(spec.value, (tuple, list)) or len(spec.value) != 2:
                raise ValueError(
                    "ClampProcess mode='range' requires value=(lower, upper)."
                )
            lower = self._resolve_quantity(spec.value[0], c, what="clamp lower bound")
            upper = self._resolve_quantity(spec.value[1], c, what="clamp upper bound")
            return torch.maximum(torch.minimum(c, upper), lower)

        value = self._resolve_quantity(spec.value, c, what="clamp value")
        if mode == "min":
            return torch.maximum(c, value)
        if mode == "max":
            return torch.minimum(c, value)
        return value + torch.zeros_like(c)

    def advance_materials(self, dt):
        del dt  # clamps are algebraic process updates
        transaction = _MaterialFieldTransaction()
        for key, spec in zip(self._clamp_operator_keys, type(self)._clamp_specs):
            material = self._get_material(spec.material)
            c = transaction.read(material, spec.field)
            mode = self._effective_clamp_mode(spec)
            c_clamped = self._apply_clamp_mode(c, spec, mode)
            mask_name = self._clamp_mask_names.get(key, None)
            if mask_name is None:
                c_new = c_clamped
            else:
                mask = self._broadcast_mask(self._buffers[mask_name], c)
                c_new = torch.where(mask, c_clamped, c)
            transaction.write(material, spec.field, c_new)
        transaction.commit()


class ExchangeProcess(MaterialProcess):
    """Conservative pointwise exchange between two full Material fields.

    An ExchangeProcess applies a local two-pool exchange at every compartment::

        V_a * a' = -g * (a - b)
        V_b * b' =  g * (a - b)

    where ``a`` and ``b`` are concentration-like material fields, ``V_a`` and
    ``V_b`` are their local mass/volume weights, and ``g`` is an exchange
    conductance.  If ``rate`` is supplied instead of ``conductance``, Dendra uses
    ``g = rate * V_a``; thus ``rate`` has units ``1/ms`` and describes the
    first-order relaxation rate of pool ``a`` toward pool ``b`` when volumes are
    fixed.

    The default method is the exact closed-form update for fixed ``rate``/``g``
    and fixed volumes over the timestep.  It preserves ``V_a*a + V_b*b`` up to
    floating-point roundoff on compartments with positive volumes.
    """

    _exchange_specs: tuple[ExchangeSpec, ...] = tuple()
    _exchange_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        specs: list[ExchangeSpec] = []
        for base in reversed(cls.__mro__):
            if "_exchange_specs" in base.__dict__:
                specs.extend(list(base._exchange_specs))
        specs.extend(
            consume_class_values(
                cls,
                "exchange_process.specs",
                ExchangeProcess._exchange_declarations,
            )
        )
        cls._exchange_specs = tuple(specs)

    @staticmethod
    def EXCHANGE(
        a,
        b,
        *,
        field_a=None,
        field_b=None,
        rate=None,
        conductance=None,
        volume_a=None,
        volume_b=None,
        domain_a=None,
        domain_b=None,
    ):
        """Declare conservative exchange between two material fields.

        Parameters
        ----------
        a, b
            Material field references.  Accepted forms are ``("ca", "cai")``,
            ``"ca.cai"``, or a material name plus ``field_a=...`` /
            ``field_b=...``.
        rate
            First-order rate in ``1/ms``.  Dendra converts this to an exchange
            conductance via ``g = rate * volume_a``.  If omitted and
            ``conductance`` is also omitted, defaults to the process parameter
            name ``"rate"``.
        conductance
            Direct exchange conductance in the chosen geometry/mass-weight
            units per ms (for example µm³/ms for volumetric fields or µm²/ms
            for membrane-domain fields). This is useful when the coupling
            itself is a geometric/permeability term. Specify either ``rate``
            or ``conductance``, not both.
        volume_a, volume_b
            Optional mass/volume weights.  May be scalars, tensors, parameter
            names, or domain names such as ``"intracellular"``.  If omitted,
            Dendra tries to infer volumes from the material field domain and
            population geometry; if no compatible geometry is available, unit
            volumes are used unless ``METHOD(..., require_volumes=True)`` is set.
        domain_a, domain_b
            Optional domain metadata overrides.
        """
        material_a, field_a_ = _material_field_ref(a, field=field_a)
        material_b, field_b_ = _material_field_ref(b, field=field_b)
        if rate is not None and conductance is not None:
            raise ValueError(
                "ExchangeProcess.EXCHANGE accepts either rate or conductance, not both."
            )
        if rate is None and conductance is None:
            rate = "rate"
        spec = ExchangeSpec(
            material_a,
            field_a_,
            material_b,
            field_b_,
            rate,
            conductance,
            volume_a,
            volume_b,
            None if domain_a is None else str(domain_a),
            None if domain_b is None else str(domain_b),
        )
        declare_class_value(
            "exchange_process.specs", spec, ExchangeProcess._exchange_declarations
        )

    @staticmethod
    def COUPLE(*args, **kwargs):
        """Alias for :meth:`EXCHANGE`."""
        ExchangeProcess.EXCHANGE(*args, **kwargs)

    @staticmethod
    def TRANSFER(*args, **kwargs):
        """Alias for :meth:`EXCHANGE`."""
        ExchangeProcess.EXCHANGE(*args, **kwargs)

    def configure_process(self, population=None):
        if self.key is not None:
            raise NotImplementedError(
                "ExchangeProcess MVP must be inserted globally. Region-restricted "
                "exchange should be expressed with masks or region-aware process operators."
            )

        kwargs = _copy_method_kwargs(type(self)._material_process_method_kwargs)
        method = str(type(self)._material_process_method or "exact").lower()
        if method in {"none", ""}:
            method = "exact"
        aliases = {
            "exp": "exact",
            "exponential": "exact",
            "closed_form": "exact",
            "analytic": "exact",
            "be": "implicit",
            "backward_euler": "implicit",
            "bwd_euler": "implicit",
            "forward_euler": "explicit",
            "euler": "explicit",
        }
        self._exchange_method = aliases.get(method, method)
        if self._exchange_method not in {"exact", "implicit", "explicit"}:
            raise NotImplementedError(
                "ExchangeProcess supports METHOD('exact'), METHOD('implicit'), "
                "and METHOD('explicit')."
            )
        self._exchange_domain = kwargs.get("domain", None)
        self._exchange_eps = float(kwargs.get("eps", 1e-30))
        self._exchange_require_volumes = bool(kwargs.get("require_volumes", False))
        self._exchange_allow_negative_rate = bool(
            kwargs.get("allow_negative_rate", False)
        )

        self._configure_exchange_geometry(population)

        if not type(self)._exchange_specs:
            raise ValueError(
                f"{type(self).__name__} declares no exchanged fields. Add "
                "ExchangeProcess.EXCHANGE(...)."
            )
        self._validate_exchange_specs()
        return None

    def _set_exchange_geometry_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def _configure_exchange_geometry(self, population=None):
        if population is None:
            return None

        for name in ("volume", "volume_um3", "volume_i", "volume_o"):
            if hasattr(population, name):
                value = getattr(population, name)
                if torch.is_tensor(value):
                    self._set_exchange_geometry_buffer(
                        f"_mp_{name}", value.detach().clone()
                    )

        if hasattr(population, "dx"):
            value = getattr(population, "dx")
            if torch.is_tensor(value):
                self._set_exchange_geometry_buffer("_mp_dx", value.detach().clone())

        try:
            area = population.area
            if torch.is_tensor(area):
                # Population.area is in cm^2; convert to µm^2 for surface-like
                # material masses.  This is available for future membrane-domain
                # processes, but ExchangeProcess does not require it.
                self._set_exchange_geometry_buffer(
                    "_mp_area_um2", (area.detach().clone() * 1e8)
                )
        except Exception:
            pass
        return None

    def _validate_exchange_specs(self):
        for spec in type(self)._exchange_specs:
            ma = self._get_material(spec.material_a)
            mb = self._get_material(spec.material_b)
            if not ma.has_field(spec.field_a):
                raise ValueError(
                    f"Material {spec.material_a!r} has no field {spec.field_a!r}. "
                    f"Available fields: {ma.fields}."
                )
            if not mb.has_field(spec.field_b):
                raise ValueError(
                    f"Material {spec.material_b!r} has no field {spec.field_b!r}. "
                    f"Available fields: {mb.fields}."
                )
            if spec.material_a == spec.material_b and spec.field_a == spec.field_b:
                raise ValueError("ExchangeProcess cannot exchange a field with itself.")
            self._effective_exchange_domain(ma, spec.field_a, spec.domain_a)
            self._effective_exchange_domain(mb, spec.field_b, spec.domain_b)

    def _effective_exchange_domain(self, material, field: str, domain: str | None):
        if domain is None:
            domain = self._exchange_domain
        if domain is None:
            try:
                domain = material.field_spec(field).domain
            except Exception:
                domain = None
        return _canonical_domain(domain)

    def _geometry_volume_for_domain(
        self, domain: str | None, like: torch.Tensor, *, what: str
    ):
        domain = _canonical_domain(domain) or "unit"
        device, dtype = like.device, like.dtype

        def _buf(name):
            return self._buffers[name].to(device=device, dtype=dtype)

        if domain == "intracellular":
            for name in ("_mp_volume_i", "_mp_volume", "_mp_volume_um3"):
                if name in self._buffers:
                    return _buf(name)
            if "_mp_dx" in self._buffers:
                dx = _buf("_mp_dx")
                diam = self.diam.to(device=device, dtype=dtype)
                return torch.pi * (0.5 * diam) ** 2 * dx

        if domain == "extracellular":
            if "_mp_volume_o" in self._buffers:
                return _buf("_mp_volume_o")

        if domain == "membrane":
            if "_mp_area_um2" in self._buffers:
                return _buf("_mp_area_um2")

        if self._exchange_require_volumes:
            raise NotImplementedError(
                f"Could not infer a volume/mass tensor for {what} with domain {domain!r}. "
                "Pass volume_a=/volume_b= explicitly, or set "
                "METHOD(..., require_volumes=False) to use unit volumes."
            )
        return torch.ones_like(like)

    def _resolve_exchange_volume(
        self, value, like: torch.Tensor, *, domain: str | None, what: str
    ):
        if value is None:
            return self._geometry_volume_for_domain(domain, like, what=what)

        if isinstance(value, str):
            lower = value.lower()
            domain_words = {
                "domain",
                "field",
                "auto",
                "i",
                "inside",
                "intra",
                "cytosol",
                "cytosolic",
                "intracellular",
                "o",
                "outside",
                "extra",
                "extracellular",
                "membrane",
                "surface",
                "volume",
                "volume_i",
                "volume_um3",
                "intracellular_volume",
                "volume_o",
                "extracellular_volume",
                "area",
                "surface_area",
                "membrane_area",
            }
            if lower in {"domain", "field", "auto"}:
                return self._geometry_volume_for_domain(domain, like, what=what)
            if lower in {"volume", "volume_um3"}:
                for name in ("_mp_volume", "_mp_volume_um3", "_mp_volume_i"):
                    if name in self._buffers:
                        return self._buffers[name].to(
                            device=like.device, dtype=like.dtype
                        )
                return self._geometry_volume_for_domain(
                    "intracellular", like, what=what
                )
            if lower in {"volume_i", "intracellular_volume"}:
                return self._geometry_volume_for_domain(
                    "intracellular", like, what=what
                )
            if lower in {"volume_o", "extracellular_volume"}:
                return self._geometry_volume_for_domain(
                    "extracellular", like, what=what
                )
            if lower in {"area", "surface_area", "membrane_area"}:
                return self._geometry_volume_for_domain("membrane", like, what=what)
            if lower in domain_words:
                return self._geometry_volume_for_domain(value, like, what=what)
            # Otherwise interpret the string as a RANGE/GLOBAL parameter/buffer name.
            return self._resolve_quantity(value, like, what=what)

        if torch.is_tensor(value):
            return value.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(value, device=like.device, dtype=like.dtype)

    def _exchange_conductance(
        self, spec: ExchangeSpec, a: torch.Tensor, Va: torch.Tensor
    ):
        if spec.conductance is not None:
            g = self._resolve_quantity(spec.conductance, a, what="exchange conductance")
        else:
            rate = self._resolve_quantity(spec.rate, a, what="exchange rate")
            if not self._exchange_allow_negative_rate:
                rate = torch.clamp_min(rate, 0.0)
            g = rate * Va
        if not self._exchange_allow_negative_rate:
            g = torch.clamp_min(g, 0.0)
        return g

    def advance_materials(self, dt):
        dt_t = None
        transaction = _MaterialFieldTransaction()
        for spec in type(self)._exchange_specs:
            ma = self._get_material(spec.material_a)
            mb = self._get_material(spec.material_b)
            a = transaction.read(ma, spec.field_a)
            b = transaction.read(mb, spec.field_b)

            if dt_t is None:
                dt_t = torch.as_tensor(dt, device=a.device, dtype=a.dtype)
            else:
                dt_t = dt_t.to(device=a.device, dtype=a.dtype)

            domain_a = self._effective_exchange_domain(ma, spec.field_a, spec.domain_a)
            domain_b = self._effective_exchange_domain(mb, spec.field_b, spec.domain_b)

            Va = self._resolve_exchange_volume(
                spec.volume_a, a, domain=domain_a, what="exchange volume_a"
            )
            Vb = self._resolve_exchange_volume(
                spec.volume_b, b, domain=domain_b, what="exchange volume_b"
            ).to(device=a.device, dtype=a.dtype)
            b = b.to(device=a.device, dtype=a.dtype)

            eps_t = torch.as_tensor(self._exchange_eps, device=a.device, dtype=a.dtype)
            Va_safe = torch.clamp_min(Va, eps_t)
            Vb_safe = torch.clamp_min(Vb, eps_t)
            active = (Va > eps_t) & (Vb > eps_t)

            g = self._exchange_conductance(spec, a, Va_safe)
            diff = a - b

            if self._exchange_method == "explicit":
                flux = g * diff
                a_new = a - dt_t * flux / Va_safe
                b_new = b + dt_t * flux / Vb_safe
            else:
                lam = g * (1.0 / Va_safe + 1.0 / Vb_safe)
                if self._exchange_method == "implicit":
                    diff_new = diff / (1.0 + dt_t * lam)
                else:  # exact
                    diff_new = diff * torch.exp(-dt_t * lam)
                total = Va_safe * a + Vb_safe * b
                denom = Va_safe + Vb_safe
                a_new = (total + Vb_safe * diff_new) / denom
                b_new = (total - Va_safe * diff_new) / denom

            a_new = torch.where(active, a_new, a)
            b_new = torch.where(active, b_new, b)

            transaction.write(ma, spec.field_a, a_new)
            transaction.write(mb, spec.field_b, b_new)
        transaction.commit()


class DiffusionProcess(MaterialProcess):
    """Finite-volume diffusion process for full population-wide material fields.

    MVP scope:
      - one-dimensional unbranched Axon geometry along the final tensor axis
      - branched Tree geometry using the DHS/Hines spatial operator when Tree
        material-geometry buffers are available
      - sealed/no-flux boundaries
      - intracellular/cytosolic domain
      - explicit and implicit methods, with implicit as the default

    Geometry/topology and timestep-scaled coefficients are configured from
    ``set_dt(dt)`` so the per-timestep material phase only calls a preconfigured
    spatial operator.
    """

    _material_process_phase = "transport"
    _diffusion_specs: tuple[DiffusionSpec, ...] = tuple()
    _diffusion_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        specs: list[DiffusionSpec] = []
        for base in reversed(cls.__mro__):
            if "_diffusion_specs" in base.__dict__:
                specs.extend(list(base._diffusion_specs))
        specs.extend(
            consume_class_values(
                cls,
                "diffusion_process.specs",
                DiffusionProcess._diffusion_declarations,
            )
        )
        cls._diffusion_specs = tuple(specs)

    @staticmethod
    def DIFFUSE(material, *, field=None, D=None, domain=None):
        """Declare a material field to diffuse.

        Parameters
        ----------
        material
            Material name, e.g. ``"ca"`` or ``"ip3"``.
        field
            Field name, e.g. ``"cai"`` or ``"ip3i"``.  If omitted, defaults to
            ``f"{material}i"``.
        D
            Diffusivity in ``um^2 / ms`` by default.  May be a scalar or the name
            of a process parameter, e.g. ``D="Dca"``.
        domain
            Optional domain override.  If omitted, the material field spec's
            domain metadata is used.
        """
        material = str(material)
        if field is None:
            field = f"{material}i"
        if D is None:
            D = "D"
        spec = DiffusionSpec(
            str(material), str(field), D, None if domain is None else str(domain)
        )
        declare_class_value(
            "diffusion_process.specs", spec, DiffusionProcess._diffusion_declarations
        )

    def configure_process(self, population=None):
        if self.key is not None:
            raise NotImplementedError(
                "DiffusionProcess MVP must be inserted globally. Region-restricted "
                "diffusion will require restricted spatial operators."
            )
        if population is None:
            raise RuntimeError(
                "DiffusionProcess requires population geometry during binding."
            )
        kwargs = _copy_method_kwargs(type(self)._material_process_method_kwargs)
        self._diffusion_method = str(
            type(self)._material_process_method or "implicit"
        ).lower()
        if self._diffusion_method in {"none", ""}:
            self._diffusion_method = "implicit"
        aliases = {
            "be": "implicit",
            "backward_euler": "implicit",
            "bwd_euler": "implicit",
            "forward_euler": "explicit",
            "euler": "explicit",
        }
        self._diffusion_method = aliases.get(
            self._diffusion_method, self._diffusion_method
        )
        if self._diffusion_method not in {"implicit", "explicit"}:
            raise NotImplementedError(
                "DiffusionProcess MVP supports METHOD('implicit') and METHOD('explicit')."
            )
        self._diffusion_solver = str(kwargs.get("solver", "auto")).lower()
        self._diffusion_boundary = str(kwargs.get("boundary", "sealed")).lower()
        self._diffusion_volume_fraction = kwargs.get("volume_fraction", 1.0)
        self._diffusion_area_fraction = kwargs.get("area_fraction", 1.0)
        self._diffusion_domain = kwargs.get("domain", None)
        self._diffusion_threads = int(kwargs.get("threads", 16))

        if not type(self)._diffusion_specs:
            raise ValueError(
                f"{type(self).__name__} declares no diffused fields. Add "
                "DiffusionProcess.DIFFUSE(...)."
            )

        self._diffusion_operator_keys = tuple(
            _safe_key(i, spec.material, spec.field)
            for i, spec in enumerate(type(self)._diffusion_specs)
        )

        self._diffusion_geometry_kind = self._select_geometry_kind(population)
        if self._diffusion_geometry_kind == "tree":
            self._configure_tree_geometry(population)
        else:
            self._configure_1d_geometry(population)

        self._spatial_operators = self._new_spatial_operators()

        self._validate_diffusion_specs()
        self._spatial_configured = False
        return None

    def _new_spatial_operators(self) -> torch.nn.ModuleDict:
        if self._diffusion_geometry_kind == "tree":
            operator_type = SpatialOperatorTree
            operator_kwargs = {
                "solver": self._diffusion_solver,
                "boundary": self._diffusion_boundary,
                "threads": self._diffusion_threads,
            }
        else:
            operator_type = SpatialOperator1D
            operator_kwargs = {
                "solver": self._diffusion_solver,
                "boundary": self._diffusion_boundary,
            }
        operators = torch.nn.ModuleDict(
            {
                key: operator_type(**operator_kwargs)
                for key in self._diffusion_operator_keys
            }
        )
        operators.train(self.training)
        return operators

    def _select_geometry_kind(self, population) -> str:
        """Select the spatial backend for this population.

        Unbranched Axon populations expose ``graph=None`` and use the analytic 1D
        finite-volume geometry.  Tree populations expose a graph; for those we
        require the material-geometry buffers added by the Tree geometry patch so
        that pt3d-aware volumes and edge diffusion geometry are used rather than
        stylized ``diam*L`` approximations.
        """
        graph = getattr(population, "graph", None)
        if graph is None:
            return "1d"

        has_volume = any(
            hasattr(population, name) for name in ("volume_i", "volume", "volume_um3")
        )
        if not has_volume:
            raise RuntimeError(
                "DiffusionProcess detected a Tree/graph morphology, but the model "
                "does not expose material volume buffers. Apply the Tree material-"
                "geometry patch so Tree.gather_morphology(...) registers volume_i/volume."
            )

        # SpatialOperatorTree currently precomputes topology/couplings from graph
        # edge metadata during set_dt(...).  The preferred edge attribute is
        # diff_geom_um; R_ohm is accepted as a legacy fallback because the tree
        # spatial operator can recover geometry from R_ohm and endpoint Ra.
        has_edge_geom_graph = False
        try:
            graphs = graph if isinstance(graph, (list, tuple)) else [graph]
            if graphs:
                has_edge_geom_graph = all(
                    "diff_geom_um" in data or "R_ohm" in data
                    for g in graphs
                    for _, _, data in g.edges(data=True)
                )
        except Exception:
            has_edge_geom_graph = False

        if not has_edge_geom_graph:
            raise RuntimeError(
                "DiffusionProcess detected a Tree/graph morphology, but graph edges "
                "do not expose diff_geom_um or R_ohm. Patch the NEURON graph import "
                "path to add edge['diff_geom_um'] before using Tree material diffusion."
            )
        return "tree"

    def _set_geometry_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def _configure_1d_geometry(self, population) -> None:
        # dx is geometry, not a dynamic material field.  Keep a local registered
        # copy so device/dtype movement follows the process module.  The heavy
        # finite-volume coefficients are built later in set_dt(...), mirroring
        # the voltage-integrator initialize(...) split.
        dx = population.dx.detach().clone()
        self._set_geometry_buffer("_mp_dx", dx)
        return None

    def _configure_tree_geometry(self, population) -> None:
        # Store the Python graph object without registering the Population itself
        # as a submodule, which would create a module cycle.  The graph is static
        # topology/metadata used during set_dt(...) precomputation only.
        object.__setattr__(self, "_mp_graph", getattr(population, "graph"))

        for name in ("volume", "volume_um3", "volume_i", "volume_o", "diff_geom_um"):
            if hasattr(population, name):
                value = getattr(population, name)
                if torch.is_tensor(value):
                    self._set_geometry_buffer(f"_mp_{name}", value.detach().clone())
        return None

    @property
    def graph(self):
        """Graph proxy used by SpatialOperatorTree during set_dt(...) precompute."""
        return getattr(self, "_mp_graph", None)

    def material_volume(self, domain="intracellular"):
        """Return the process-local volume/mass buffer for a material domain."""
        domain = _canonical_domain(domain) or "intracellular"
        if domain == "intracellular":
            if "_mp_volume_i" in self._buffers:
                return self._buffers["_mp_volume_i"]
            if "_mp_volume" in self._buffers:
                return self._buffers["_mp_volume"]
            if "_mp_volume_um3" in self._buffers:
                return self._buffers["_mp_volume_um3"]
        if domain == "extracellular":
            if "_mp_volume_o" in self._buffers:
                return self._buffers["_mp_volume_o"]
            raise NotImplementedError(
                "Extracellular material diffusion on Tree morphologies requires volume_o."
            )
        if domain == "membrane":
            if hasattr(self, "area"):
                return self.area
            raise NotImplementedError(
                "Membrane/surface material diffusion requires an area/mass buffer."
            )
        raise NotImplementedError(f"Unsupported material diffusion domain: {domain!r}")

    def _validate_diffusion_specs(self):
        for spec in type(self)._diffusion_specs:
            material = self._get_material(spec.material)
            if not material.has_field(spec.field):
                raise ValueError(
                    f"Material {spec.material!r} has no field {spec.field!r}. "
                    f"Available fields: {material.fields}."
                )
            domain = self._effective_domain(material, spec)
            if domain not in {"intracellular", None}:
                raise NotImplementedError(
                    "DiffusionProcess MVP only supports intracellular/cytosolic "
                    f"fields; {spec.material}.{spec.field} has domain {domain!r}."
                )

    def _effective_domain(self, material, spec: DiffusionSpec) -> str | None:
        domain = spec.domain
        if domain is None:
            domain = self._diffusion_domain
        if domain is None:
            try:
                domain = material.field_spec(spec.field).domain
            except Exception:
                domain = "i"
        return _canonical_domain(domain)

    def _resolve_quantity(self, value, like: torch.Tensor):
        if isinstance(value, str):
            if hasattr(self, value):
                return getattr(self, value)
            try:
                return self._buffers[value]
            except KeyError as exc:
                raise AttributeError(
                    f"DiffusionProcess {self.name!r} expected a parameter/buffer "
                    f"named {value!r}. Declare it with RANGE/GLOBAL or pass a scalar D."
                ) from exc
        if torch.is_tensor(value):
            return value.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(value, device=like.device, dtype=like.dtype)

    def set_dt(self, dt):
        """Update dt and precompute diffusion operators for the current run."""
        # Validate and stage the mechanism dt without mutating the registered
        # buffer. Spatial configuration may still fail on a later specification.
        dt_new = self.dt.clone()
        dt_new.fill_(dt)
        dt_new = dt_new.detach()
        if hasattr(self, "_material_resolver"):
            self.configure_spatial_operators(dt)
        self.dt = dt_new
        return None

    def configure_spatial_operators(self, dt):
        if self._diffusion_geometry_kind == "tree":
            return self._configure_tree_spatial_operators(dt)
        return self._configure_1d_spatial_operators(dt)

    def _configure_1d_spatial_operators(self, dt):
        dx = self._buffers["_mp_dx"].to(device=self.diam.device, dtype=self.diam.dtype)
        diam_base = self.diam
        resolved = []
        for key, spec in zip(
            self._diffusion_operator_keys, type(self)._diffusion_specs
        ):
            material = self._get_material(spec.material)
            c = material._buffers[spec.field]
            D = self._resolve_quantity(spec.D, c)
            resolved.append((key, c, D))

        staged_operators = self._new_spatial_operators()
        for key, c, D in resolved:
            diam = diam_base.to(device=c.device, dtype=c.dtype)
            dx_c = dx.to(device=c.device, dtype=c.dtype)
            staged_operators[key].configure_diffusion(
                c,
                dt,
                D,
                diam,
                dx_c,
                volume_fraction=self._diffusion_volume_fraction,
                area_fraction=self._diffusion_area_fraction,
                solver=self._diffusion_solver,
            )
        self._spatial_operators = staged_operators
        self._spatial_configured = True
        return None

    def _configure_tree_spatial_operators(self, dt):
        resolved = []
        for key, spec in zip(
            self._diffusion_operator_keys, type(self)._diffusion_specs
        ):
            material = self._get_material(spec.material)
            c = material._buffers[spec.field]
            D = self._resolve_quantity(spec.D, c)
            domain = self._effective_domain(material, spec) or "intracellular"
            resolved.append((key, c, D, domain))

        staged_operators = self._new_spatial_operators()
        for key, c, D, domain in resolved:
            staged_operators[key].configure_diffusion(
                c,
                dt,
                D,
                self,
                domain=domain,
                solver=self._diffusion_solver,
            )
        self._spatial_operators = staged_operators
        self._spatial_configured = True
        return None

    def advance_materials(self, dt):
        if not getattr(self, "_spatial_configured", False):
            # This should normally be done by MechanismHandler.set_dt(...), which
            # is called from Integrator._initialize(...).  The fallback keeps
            # manual/eager calls usable while still moving the work out of the
            # steady-state hot path after the first call.
            self.configure_spatial_operators(dt)
        transaction = _MaterialFieldTransaction()
        for key, spec in zip(
            self._diffusion_operator_keys, type(self)._diffusion_specs
        ):
            material = self._get_material(spec.material)
            c = transaction.read(material, spec.field)
            op = self._spatial_operators[key]
            if self._diffusion_method == "explicit":
                c_new = op.diffuse_explicit_configured(c)
            else:
                c_new = op.diffuse_implicit_configured(c)
            transaction.write(material, spec.field, c_new)
        transaction.commit()
