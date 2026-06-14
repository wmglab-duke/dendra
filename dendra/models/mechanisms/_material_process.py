"""MaterialProcess and initial material diffusion implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ._mechanism import Mechanism
from ._spatial import SpatialOperator1D, SpatialOperatorTree


@dataclass(frozen=True)
class DiffusionSpec:
    material: str
    field: str
    D: Any
    domain: str | None = None


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

        if MaterialProcess._phase_declarations:
            phase = MaterialProcess._phase_declarations[-1]
            MaterialProcess._phase_declarations = []
        if MaterialProcess._method_declarations:
            method, method_kwargs = MaterialProcess._method_declarations[-1]
            MaterialProcess._method_declarations = []

        cls._material_process_phase = str(phase)
        cls._material_process_method = str(method).lower()
        cls._material_process_method_kwargs = dict(method_kwargs)

    @staticmethod
    def METHOD(method="none", **kwargs):
        """Declare the process-level numerical method."""
        MaterialProcess._method_declarations.append((str(method).lower(), dict(kwargs)))

    @staticmethod
    def PHASE(phase="post_local"):
        """Declare the material scheduler phase for this process."""
        MaterialProcess._phase_declarations.append(str(phase))

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
        return resolver(str(name))

    def material_field(self, material: str, field: str) -> torch.Tensor:
        return self._get_material(material)._buffers[str(field)]

    def set_material_field(
        self, material: str, field: str, value: torch.Tensor
    ) -> None:
        self._get_material(material)._buffers[str(field)] = value

    def advance_materials(self, dt):
        raise NotImplementedError(
            f"{type(self).__name__}.advance_materials(dt) must be implemented."
        )


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

    _diffusion_specs: tuple[DiffusionSpec, ...] = tuple()
    _diffusion_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        specs: list[DiffusionSpec] = []
        for base in reversed(cls.__mro__):
            if "_diffusion_specs" in base.__dict__:
                specs.extend(list(base._diffusion_specs))
        if DiffusionProcess._diffusion_declarations:
            specs.extend(DiffusionProcess._diffusion_declarations)
            DiffusionProcess._diffusion_declarations = []
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
        DiffusionProcess._diffusion_declarations.append(spec)

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
            operator_type = SpatialOperatorTree
            operator_kwargs = {
                "solver": self._diffusion_solver,
                "boundary": self._diffusion_boundary,
                "threads": self._diffusion_threads,
            }
        else:
            self._configure_1d_geometry(population)
            operator_type = SpatialOperator1D
            operator_kwargs = {
                "solver": self._diffusion_solver,
                "boundary": self._diffusion_boundary,
            }

        self._spatial_operators = torch.nn.ModuleDict(
            {
                key: operator_type(**operator_kwargs)
                for key in self._diffusion_operator_keys
            }
        )

        self._validate_diffusion_specs()
        self._spatial_configured = False
        return None

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
        super().set_dt(dt)
        if hasattr(self, "_material_resolver"):
            self.configure_spatial_operators(dt)
        return None

    def configure_spatial_operators(self, dt):
        if self._diffusion_geometry_kind == "tree":
            return self._configure_tree_spatial_operators(dt)
        return self._configure_1d_spatial_operators(dt)

    def _configure_1d_spatial_operators(self, dt):
        dx = self._buffers["_mp_dx"].to(device=self.diam.device, dtype=self.diam.dtype)
        diam_base = self.diam
        for key, spec in zip(
            self._diffusion_operator_keys, type(self)._diffusion_specs
        ):
            material = self._get_material(spec.material)
            c = material._buffers[spec.field]
            D = self._resolve_quantity(spec.D, c)
            diam = diam_base.to(device=c.device, dtype=c.dtype)
            dx_c = dx.to(device=c.device, dtype=c.dtype)
            self._spatial_operators[key].configure_diffusion(
                c,
                dt,
                D,
                diam,
                dx_c,
                volume_fraction=self._diffusion_volume_fraction,
                area_fraction=self._diffusion_area_fraction,
                solver=self._diffusion_solver,
            )
        self._spatial_configured = True
        return None

    def _configure_tree_spatial_operators(self, dt):
        for key, spec in zip(
            self._diffusion_operator_keys, type(self)._diffusion_specs
        ):
            material = self._get_material(spec.material)
            c = material._buffers[spec.field]
            D = self._resolve_quantity(spec.D, c)
            domain = self._effective_domain(material, spec) or "intracellular"
            self._spatial_operators[key].configure_diffusion(
                c,
                dt,
                D,
                self,
                domain=domain,
                solver=self._diffusion_solver,
            )
        self._spatial_configured = True
        return None

    def advance_materials(self, dt):
        if not getattr(self, "_spatial_configured", False):
            # This should normally be done by MechanismHandler.set_dt(...), which
            # is called from Integrator._initialize(...).  The fallback keeps
            # manual/eager calls usable while still moving the work out of the
            # steady-state hot path after the first call.
            self.configure_spatial_operators(dt)
        for key, spec in zip(
            self._diffusion_operator_keys, type(self)._diffusion_specs
        ):
            material = self._get_material(spec.material)
            c = material._buffers[spec.field]
            op = self._spatial_operators[key]
            if self._diffusion_method == "explicit":
                c_new = op.diffuse_explicit_configured(c)
            else:
                c_new = op.diffuse_implicit_configured(c)
            material._buffers[spec.field] = c_new
