"""Utilities for composing multiple population models into a single system."""

import copyreg
import math
from collections.abc import Mapping
from dataclasses import replace

import torch

from ..helpers import logger
from .core import (
    Cable,
    Population,
    _canonical_material_process_fields,
    _core_key_from_flat,
    _global_configuration_values_equal,
    _mechanism_global_parameter_names,
    _unpack_mechanism_insertion_record,
)
from .integrators import bwd_euler_sc_multi, dhs_bt_multi, dhs_multi
from .mechanisms._ions import cinits, reversals, valid_ions
from .mechanisms._material_process import MaterialProcess, _canonical_domain
from .mechanisms._materials import (
    Material,
    MaterialFieldSpec,
    _canonical_material_name,
    _make_into_shape,
    material_specs,
)
from .parametric import to_param
from .tree import Tree

_MISSING = object()


def _global_defaults_by_name(mechanism):
    """Collect every owner-specific default for each GLOBAL parameter name."""
    defaults = {}
    for owner in (mechanism, *tuple(getattr(mechanism, "_state", ()))):
        for declaration in ("_global", "_global_p", "_global_n"):
            for name, value in getattr(owner, declaration, {}).items():
                defaults.setdefault(name, []).append(value)
    return defaults


def _component_explicit_globals(population, mechanism):
    """Return one component's explicit class-wide GLOBAL overrides."""
    global_names = _mechanism_global_parameter_names(mechanism)
    explicit = {}
    everywhere = population._mech_everywhere.get(mechanism)
    if everywhere is not None:
        explicit.update(
            {
                name: value
                for name, value in everywhere[2].items()
                if name in global_names
            }
        )
    for name, value in population._mech_data_base_kwargs.get(mechanism, {}).items():
        if name in explicit and not _global_configuration_values_equal(
            explicit[name], value
        ):
            raise ValueError(
                f"Mechanism {mechanism.__name__!r} has different class-wide "
                f"GLOBAL values for {name!r} in one component population."
            )
        explicit[name] = value
    return explicit


def _merged_global_overrides(populations):
    """Validate and merge component GLOBAL semantics by exact mechanism class."""
    configurations = {}
    for component_name, population in populations.items():
        mechanisms = set(population._mech_everywhere) | set(population._mech_data)
        for mechanism in mechanisms:
            # Full-field material operators remain separate per component.
            # Their GLOBAL parameters therefore do not acquire the class-wide
            # sharing semantics of concatenated voltage-local mechanisms.
            if issubclass(mechanism, MaterialProcess):
                continue
            configurations.setdefault(mechanism, []).append(
                (
                    component_name,
                    _component_explicit_globals(population, mechanism),
                )
            )

    merged = {}
    for mechanism, component_configs in configurations.items():
        defaults = _global_defaults_by_name(mechanism)
        parameter_names = set().union(*(set(config) for _, config in component_configs))
        mechanism_overrides = {}
        for parameter_name in parameter_names:
            explicit = [
                (component_name, config[parameter_name])
                for component_name, config in component_configs
                if parameter_name in config
            ]
            selected_component, selected_value = explicit[0]
            for component_name, value in explicit[1:]:
                if not _global_configuration_values_equal(selected_value, value):
                    raise ValueError(
                        f"Mechanism {mechanism.__name__!r} has a different "
                        f"class-wide GLOBAL value for {parameter_name!r} in "
                        f"components {selected_component!r} and {component_name!r}."
                    )

            for component_name, config in component_configs:
                if parameter_name in config:
                    continue
                owner_defaults = defaults.get(parameter_name, ())
                if any(
                    not _global_configuration_values_equal(selected_value, default)
                    for default in owner_defaults
                ):
                    raise ValueError(
                        f"Mechanism {mechanism.__name__!r} has explicit "
                        f"class-wide GLOBAL value {parameter_name!r} in component "
                        f"{selected_component!r}, but component {component_name!r} "
                        "uses a different declared default."
                    )
            mechanism_overrides[parameter_name] = selected_value
        merged[mechanism] = mechanism_overrides
    return merged


class _PackedMaterialInitial(torch.nn.Module):
    """Resolve independent material initial sources without severing autograd.

    Concatenating values into a new Parameter would silently replace the child
    parameters. This small pure module instead owns the original sources and
    constructs the packed field afresh on every initialization.
    """

    _functional_material_initial = True

    def __init__(self, sources, shapes):
        super().__init__()
        self.shapes = tuple(tuple(shape) for shape in shapes)
        for index, source in enumerate(sources):
            setattr(self, f"source_{index}", to_param(source))

    def forward(self):
        parts = [
            _make_into_shape(shape, getattr(self, f"source_{index}")).reshape(-1)
            for index, shape in enumerate(self.shapes)
        ]
        return torch.cat(parts).reshape(1, -1)


class _PackedMaterialGeometry(torch.nn.Module):
    """Source-only pure adapter for packed chemical storage/transport fields.

    In particular this module does not retain component Population objects or
    voltage solvers. Functional preparation can therefore prune their runtimes
    while still deriving chemical coefficients from the original parameters.
    """

    def __init__(self, population):
        super().__init__()
        source_names = {}
        source_paths = {}

        def register_source(value, component_name, attribute):
            if id(value) in source_names:
                return source_names[id(value)]
            name = f"source_{len(source_names)}"
            if isinstance(value, torch.nn.Parameter):
                self.register_parameter(name, value)
            else:
                self.register_buffer(name, value)
            source_names[id(value)] = name
            source_paths[name] = (component_name, attribute)
            return name

        components = []
        named = []
        offset = 0
        for component_name, component in population.populations.items():
            core = tuple(component.core_shape())
            native = hasattr(component, "diff_parent_index")
            sources = {
                role: register_source(getattr(component, role), component_name, role)
                for role in ("volume", "volume_i", "volume_o", "diff_geom_um")
                if hasattr(component, role)
            }
            components.append(
                (
                    component_name,
                    core,
                    offset,
                    native,
                    sources,
                    population._material_component_edge_counts[component_name],
                )
            )
            for local_name, config in component._material_geometries.items():
                packed_name = population._component_material_geometry_names[
                    component_name
                ][local_name]
                packed = population._material_geometries[packed_name]
                for role, source in config.items():
                    if role in {"name", "domain"}:
                        continue
                    named.append(
                        (
                            component_name,
                            role,
                            register_source(
                                getattr(component, source), component_name, source
                            ),
                            packed[role],
                        )
                    )
            offset += math.prod(core)
        self.components = tuple(components)
        self.named_geometry = tuple(named)
        self.source_paths = source_paths

    def refresh_sources(self, population):
        """Rebind live source buffers after owner dtype moves or replacement."""
        for name, (component_name, attribute) in self.source_paths.items():
            value = getattr(population.populations[component_name], attribute)
            if name in self._parameters:
                self._parameters[name] = value
            else:
                self._buffers[name] = value

    def forward(self, packed_diam, packed_dx):
        batch_shape = tuple(packed_diam.shape[:-2])
        values = {
            name: [] for name in ("volume", "volume_i", "volume_o", "diff_geom_um")
        }
        for _name, core, offset, native, sources, _edges in self.components:
            shape = (*batch_shape, *core)
            end = offset + math.prod(core)
            diam = packed_diam[..., offset:end].reshape(shape)
            dx = packed_dx[..., offset:end].reshape(shape)
            area = torch.pi * (diam * 0.5) ** 2
            volume = (
                torch.broadcast_to(getattr(self, sources["volume"]), shape)
                if "volume" in sources
                else area * dx
            )
            volume_i = (
                torch.broadcast_to(getattr(self, sources["volume_i"]), shape)
                if "volume_i" in sources
                else volume
            )
            volume_o = (
                torch.broadcast_to(getattr(self, sources["volume_o"]), shape)
                if "volume_o" in sources
                else torch.zeros_like(volume)
            )
            if native:
                factor = torch.broadcast_to(
                    getattr(self, sources["diff_geom_um"]), shape
                )
            else:
                edge = (
                    0.5
                    * (area[..., :-1] + area[..., 1:])
                    / (0.5 * (dx[..., :-1] + dx[..., 1:])).clamp_min(
                        torch.finfo(dx.dtype).tiny
                    )
                )
                factor = torch.cat((torch.zeros_like(dx[..., :1]), edge), dim=-1)
            for role, value in (
                ("volume", volume),
                ("volume_i", volume_i),
                ("volume_o", volume_o),
                ("diff_geom_um", factor),
            ):
                values[role].append(value.reshape(*batch_shape, -1))
        result = {
            name: torch.cat(parts, dim=-1).unsqueeze(-2)
            for name, parts in values.items()
        }
        for component_name, role, source, destination in self.named_geometry:
            value = getattr(self, source)
            parts = []
            for other_name, core, _offset, _native, _sources, edges in self.components:
                shape = (
                    (*batch_shape, *core) if role == "volume" else (*batch_shape, edges)
                )
                if other_name == component_name:
                    if role != "volume":
                        value = torch.broadcast_to(
                            value, (*batch_shape, core[0], edges // core[0])
                        ).reshape(*batch_shape, -1)
                    part = torch.broadcast_to(value, shape)
                else:
                    fill = 1.0 if role in {"volume", "edge_distance"} else 0.0
                    part = torch.full(
                        shape, fill, device=packed_diam.device, dtype=packed_diam.dtype
                    )
                parts.append(part.reshape(*batch_shape, -1))
            combined = torch.cat(parts, dim=-1)
            # Packed fields have one logical neuron row. Keep that axis for
            # compact edge quantities too, so an explicit batch prefix cannot
            # be mistaken for the neuron axis by the spatial operator.
            result[destination] = combined.unsqueeze(-2)
        return result


class _PackedMaterialProcessMeta(type(MaterialProcess)):
    """Pickle-aware class identity for independent component operators."""


def _make_packed_material_process_class(
    original, component_name, class_name, geometry_names, quantity_names
):
    scoped = _PackedMaterialProcessMeta(
        class_name,
        (original,),
        {
            "__module__": __name__,
            "_material_process_component": component_name,
            "_material_process_original_class": original,
            "_material_process_geometry_names": geometry_names,
            "_material_process_quantity_names": quantity_names,
        },
    )
    # Family metaclasses collect inherited declarations. Keep the already
    # resolved author declarations exactly once in the adapter.
    for attribute in (
        "_clearance_specs",
        "_clamp_specs",
        "_exchange_specs",
        "_diffusion_specs",
        "_diffusion_relaxation_specs",
    ):
        if hasattr(original, attribute):
            setattr(scoped, attribute, getattr(original, attribute))
    return scoped


def _reduce_packed_material_process_class(scoped):
    return _make_packed_material_process_class, (
        scoped._material_process_original_class,
        scoped._material_process_component,
        scoped.__name__,
        scoped._material_process_geometry_names,
        scoped._material_process_quantity_names,
    )


copyreg.pickle(_PackedMaterialProcessMeta, _reduce_packed_material_process_class)


def _component_material_specs(population):
    """Resolve configured material metadata without initializing a population."""
    declarations = {}
    names = set(population._material_configs)
    for mechanism in set(population._mech_everywhere) | set(population._mech_data):
        names.update(getattr(mechanism, "_material", ()))
        for attribute in (
            "_clearance_specs",
            "_clamp_specs",
            "_exchange_specs",
            "_diffusion_specs",
        ):
            for spec in getattr(mechanism, attribute, ()):
                for role in ("material", "material_a", "material_b"):
                    if hasattr(spec, role):
                        names.add(getattr(spec, role))
    names = {_canonical_material_name(name) for name in names} - set(valid_ions())
    for name in sorted(names):
        material = Material(
            name,
            population.core_shape(),
            **population._material_constructor_kwargs(name),
        )
        # A pending rebuild discards the old registry on the child's next
        # initialize(). In particular, material() may have replaced an initial
        # source since that registry was built; concatenation must then resolve
        # the new declaration rather than resurrect the stale source.
        built = (
            getattr(population, "mech", None)
            if population.is_built and not population._flag_rebuild
            else None
        )
        existing = (
            built.materials[name]
            if built is not None and name in built.materials
            else None
        )
        declarations[name] = {
            field: replace(
                material.field_spec(field),
                initial=(
                    existing.initial_source(field)
                    if existing is not None and existing.has_field(field)
                    else material.initial_source(field)
                ),
            )
            for field in material.fields
        }
    return declarations


def _component_ion_styles(population):
    """Infer effective ion flags from declarations without building children."""
    used = set(population._ion_style)
    reads = {}
    writes = set()
    ions = set(valid_ions())
    for mechanism in set(population._mech_everywhere) | set(population._mech_data):
        used.update(set(getattr(mechanism, "_ion", ())) & ions)
        for declaration in ("_read_ion", "_read_material"):
            for name, fields in getattr(mechanism, declaration, {}).items():
                name = _canonical_material_name(name)
                if name in ions:
                    used.add(name)
                    reads.setdefault(name, set()).update(fields)
        for declaration in (
            "_write_ion",
            "_write_ion_c",
            "_write_material",
            "_source_material",
        ):
            for name in getattr(mechanism, declaration, {}):
                name = _canonical_material_name(name)
                if name in ions:
                    used.add(name)
                    if declaration != "_write_ion":
                        writes.add(name)
        if issubclass(mechanism, MaterialProcess):
            for name, fields in _canonical_material_process_fields(mechanism).items():
                if name in ions:
                    used.add(name)
                    reads.setdefault(name, set()).update(fields)
                    writes.add(name)
    styles = {}
    for name in used:
        if name in population._ion_style:
            styles[name] = tuple(int(flag != 0) for flag in population._ion_style[name])
        else:
            fields = reads.get(name, set())
            concentration_read = bool({f"{name}i", f"{name}o"} & fields)
            reversal_read = f"e{name}" in fields
            styles[name] = (
                int(reversal_read and (name in writes or concentration_read)),
                int(reversal_read and name in writes),
            )
    return styles


def _component_ion_initials(population, attribute):
    """Keep trained Ion initial leaves when the child's registry is current."""
    sources = dict(getattr(population, attribute))
    built = (
        getattr(population, "mech", None)
        if population.is_built and not population._flag_rebuild
        else None
    )
    if built is not None:
        for name, ion in built.ions.items():
            if attribute == "_equilibria":
                sources[f"e{name}"] = ion.e_init
            else:
                sources[f"{name}i0"] = ion.i_init
                sources[f"{name}o0"] = ion.o_init
    return sources


def _merged_ion_styles(populations):
    configurations = {}
    for component_name, population in populations.items():
        for ion, style in _component_ion_styles(population).items():
            if ion in configurations:
                previous_component, previous_style = configurations[ion]
                if style != previous_style:
                    raise ValueError(
                        f"Ion {ion!r} has incompatible effective ion_style flags "
                        f"in components {previous_component!r} ({previous_style}) "
                        f"and {component_name!r} ({style}). Packed ion fields "
                        "currently require compatible initialization and reversal updates."
                    )
            else:
                configurations[ion] = (component_name, style)
    return {ion: style for ion, (_component, style) in configurations.items()}


def _uses_block_voltage_state(population):
    """Return whether a population requires a block-valued voltage solve.

    Multi-solver selection must follow the electrical state carried by a model,
    rather than only its Python inheritance.  In particular, ``ExtCellTree``
    is a ``Tree`` subclass but its integrator evolves the additional ``vc``
    circuit state and therefore cannot be passed to scalar DHS.
    """
    if getattr(population, "n_layers", 0):
        return True

    candidates = (
        getattr(population, "integrator", None),
        getattr(population, "_integrator_class", None),
    )
    return any(
        "vc" in tuple(getattr(candidate, "v_vars", ()))
        for candidate in candidates
        if candidate is not None
    )


def _multi_solver_capability(population):
    """Describe the currently supported multi-solver capability of a model."""
    if _uses_block_voltage_state(population):
        return "block"
    if isinstance(population, Tree):
        return "scalar_tree"
    if isinstance(population, Cable):
        return "scalar_path"
    if isinstance(population, MultiPopulation):
        return "unsupported"
    if isinstance(population, Population):
        return "scalar_independent"
    return "unsupported"


def _validate_block_populations(populations):
    """Admit two-layer extracellular packs without mixing voltage contracts."""
    block = [
        name
        for name, population in populations.items()
        if _multi_solver_capability(population) == "block"
    ]
    if block and len(block) != len(populations):
        names = ", ".join(repr(name) for name in block)
        raise TypeError(
            "Block-state population(s) "
            f"{names} require a block multi-solver and cannot be mixed with "
            "scalar components; their extracellular circuit states differ."
        )
    if block:
        from .extcell import ExtCellAxon, ExtCellTree

        for name, population in populations.items():
            if not isinstance(population, (ExtCellAxon, ExtCellTree)):
                raise TypeError(
                    "Custom block-state populations require a block multi-solver "
                    "with a supported topology. Packed components must be ExtCellAxon or "
                    f"ExtCellTree; got {type(population).__name__} for {name!r}."
                )
            if int(getattr(population, "n_layers", 0)) != 2:
                raise ValueError(
                    "All packed extracellular components require exactly two layers."
                )
    return bool(block)


def _assess_type_and_make_integrator(populations, threads=16, write_back=True):
    """Select an integrator compatible with the provided populations.

    Parameters
    ----------
    populations : dict[str, Population]
        Mapping of population names to instances.
    threads : int, optional
        Number of solver threads for dendritic tree models.
    write_back : bool, optional
        Whether integrators should write-back states in-place.

    Returns
    -------
    callable
        Integrator factory compatible with the population types.

    Raises
    ------
    ValueError
        If no populations are provided.
    TypeError
        If populations are heterogeneous in a way that is not supported.
    """
    if not populations:
        raise ValueError("At least one population is required.")
    if _validate_block_populations(populations):
        return dhs_bt_multi(threads=threads, write_back=write_back)
    capabilities = {_multi_solver_capability(pop) for pop in populations.values()}
    if capabilities == {"scalar_independent"}:
        return bwd_euler_sc_multi(write_back=write_back)
    scalar_capabilities = {"scalar_independent", "scalar_tree", "scalar_path"}
    if capabilities and capabilities <= scalar_capabilities:
        return dhs_multi(threads=threads, write_back=write_back)

    unsupported = [
        name
        for name, population in populations.items()
        if _multi_solver_capability(population) == "unsupported"
    ]
    names = ", ".join(repr(name) for name in unsupported)
    raise TypeError(
        "Population(s) "
        f"{names} do not expose a supported scalar multi-solver topology. "
        "Supported components are ordinary Population, Tree, and scalar Cable "
        "models (including Axon subclasses)."
    )


def _check_celsius(celsius, populations):
    """Validate that all populations share the same temperature."""
    if not isinstance(celsius, (int, float)):
        raise TypeError("celsius must be a number.")
    if any(abs(pop.celsius - celsius) > 1e-6 for pop in populations.values()):
        logger.warning(
            "Inconsistent celsius values found in populations. When concatenating populations, the single value provided to concat is used, however at least one population has a different celsius value."
        )


def _validate_component_device_and_dtype(populations):
    """Fail with component-specific diagnostics before tensor concatenation."""
    devices = {population.device() for population in populations.values()}
    if len(devices) > 1:
        details = ", ".join(
            f"{name}={population.device()}" for name, population in populations.items()
        )
        raise ValueError(f"All populations must be on the same device; got {details}.")

    dtypes = {population.dtype() for population in populations.values()}
    if len(dtypes) > 1:
        details = ", ".join(
            f"{name}={population.dtype()}" for name, population in populations.items()
        )
        raise ValueError(f"All populations must be of the same dtype; got {details}.")


def _validate_component_input_bindings(populations):
    """Reject unsupported binding consolidation before touching components."""
    for name, population in populations.items():
        if population._input_bindings:
            raise ValueError(
                f"Component {name!r} has State input bindings; packed input-binding "
                "consolidation is not supported yet."
            )


def concat_models(
    populations: dict[str, Population],
    threads=16,
    write_back=True,
    celsius=37.0,
    v_init=None,
):
    """Pack independent populations into one :class:`MultiPopulation`.

    Ordinary :class:`~dendra.models.core.Population` and
    :class:`~dendra.models.core.SingleCompartment` models, branched
    :class:`~dendra.models.tree.Tree` models, and scalar
    :class:`~dendra.models.core.Cable` models (including Axon subclasses) can
    be combined in one call.
    Mixed scalar components use the universal packed DHS solver; a collection
    containing only independent point compartments keeps the optimized
    single-compartment multi-integrator.

    Alternatively, two-layer :class:`~dendra.models.extcell.ExtCellAxon` and
    :class:`~dendra.models.extcell.ExtCellTree` components can be combined in a
    block-DHS pack. Scalar and finite-extracellular components cannot be mixed.

    Concatenation does not create axial edges between component populations.
    It packs their independent electrical systems into one launch and exposes
    each component name as a labelled Slice on the composite model.

    Parameters
    ----------
    populations : dict[str, Population]
        Mapping of unique component names to unbatched populations. All
        components must share a device and dtype.
    threads : int, optional
        DHS lane count for packed Tree/Cable groups. Must divide 32.
    write_back : bool, optional
        If ``True``, write solved ``v`` (and block ``vc``) views to components after
        each step. If ``False``, only the composite voltage is updated.
    celsius : float, optional
        Common simulation temperature.
    v_init : scalar, tensor-like, dict[str, tensor-like], optional
        Optional initial voltage override for the combined model. If ``None``,
        each component population contributes its own expanded ``v_init``.
        If a mapping is provided, keys are population names and omitted
        populations keep their own ``v_init``.

    Returns
    -------
    MultiPopulation
        Combined population. Call :meth:`MultiPopulation.batch` on this result
        when explicit leading batch dimensions are required.

    Notes
    -----
    A packed extracellular system retains both membrane voltage ``v`` and
    absolute circuit potentials ``vc``. Concatenation still creates no edges
    between components. Nested ``MultiPopulation`` objects are unsupported.
    """
    _validate_component_input_bindings(populations)
    _check_celsius(celsius, populations)
    integrator = _assess_type_and_make_integrator(
        populations, threads=threads, write_back=write_back
    )
    _validate_component_device_and_dtype(populations)
    # concatenate x, y, z
    x = torch.cat([pop.x.flatten() for pop in populations.values()])
    y = torch.cat([pop.y.flatten() for pop in populations.values()])
    z = torch.cat([pop.z.flatten() for pop in populations.values()])
    mp = MultiPopulation(
        integrator=integrator,
        celsius=celsius,
        v_init=v_init,
        **populations,
    )
    mp.x.copy_(x)
    mp.y.copy_(y)
    mp.z.copy_(z)
    return mp


def offsets(populations):
    """Compute flattened offsets for each population in the concatenation."""
    result = []
    total = 0
    for population in populations.values():
        result.append(total)
        total += math.prod(population.shape)
    return result


def indices(populations):
    """Return per-population index arrays aligned with the concatenation."""
    off = offsets(populations)
    sizes = [math.prod(p.shape) for p in populations.values()]
    indices = [
        torch.arange(s, device=p.device(), dtype=torch.long).reshape(p.shape) + off[idx]
        for idx, (s, p) in enumerate(zip(sizes, populations.values()))
    ]
    return indices


def key_to_flat_index(indices, key):
    """Convert a structured key into flattened indices."""
    return indices[key].flatten()


def _concat_component_field(populations, name):
    """Project one component field onto the composite's flattened core.

    Leading batch dimensions are retained.  The final two component dimensions
    are flattened into the single structural axis used by MultiPopulation.
    """
    parts = []
    batch_shape = None
    for population_name, population in populations.items():
        value = getattr(population, name)
        population_batch_shape = tuple(population.shape[:-2])
        if batch_shape is None:
            batch_shape = population_batch_shape
        elif population_batch_shape != batch_shape:
            raise ValueError(
                "All component populations must have the same leading batch "
                f"shape; got {batch_shape} and {population_batch_shape}."
            )
        try:
            value = torch.broadcast_to(value, tuple(population.shape))
        except RuntimeError as error:
            raise ValueError(
                f"Component field {name!r} on population {population_name!r} "
                f"must broadcast to shape {tuple(population.shape)}; got "
                f"{tuple(value.shape)}."
            ) from error
        parts.append(value.reshape(*population_batch_shape, -1))

    flattened = torch.cat(parts, dim=-1)
    return flattened.unsqueeze(-2).contiguous()


def _concat_component_shell_field(populations, name):
    """Flatten component cores while retaining their two extracellular layers."""
    parts = []
    batch_shape = None
    for component_name, population in populations.items():
        component_batch = tuple(population.shape[:-2])
        if batch_shape is None:
            batch_shape = component_batch
        elif batch_shape != component_batch:
            raise ValueError("Packed components must share their leading batch shape.")
        value = getattr(population, name)
        target_shape = tuple(population.shape) + (2,)
        try:
            value = torch.broadcast_to(value, target_shape)
        except RuntimeError as error:
            raise ValueError(
                f"Component field {name!r} on {component_name!r} must "
                f"broadcast to {target_shape}; got {tuple(value.shape)}."
            ) from error
        parts.append(value.reshape(*component_batch, -1, 2))
    return torch.cat(parts, dim=-2).unsqueeze(-3).contiguous()


def flatten_key(n, shape, key):
    """Compute flattened indices for ``key`` within a tensor of ``shape``."""
    indices = torch.arange(n).reshape(shape)
    return indices[key].flatten()


def _component_label_core_indices(pop, component_slice):
    """Project a component label to one ordered structural-core sequence.

    Component labels are installed on the composite model's flattened core
    axis and therefore must describe the same ordered compartments in every
    batch replica. Duplicate occurrences and their interleaving are retained.
    """
    full_shape = tuple(pop.shape)
    core_shape = tuple(pop.core_shape())
    core_numel = math.prod(core_shape)
    full_grid = torch.arange(
        math.prod(full_shape), device=pop.device(), dtype=torch.long
    ).reshape(full_shape)
    selected = full_grid[component_slice.index].reshape(-1)
    if not selected.numel():
        return selected

    batch_indices = torch.div(selected, core_numel, rounding_mode="floor")
    core_indices = torch.remainder(selected, core_numel)
    replicas = torch.unique(batch_indices, sorted=True)
    n_replicas = math.prod(full_shape[: -len(core_shape)]) or 1
    expected_replicas = torch.arange(n_replicas, device=pop.device(), dtype=torch.long)
    if not torch.equal(replicas, expected_replicas):
        raise ValueError(
            "A component Slice label must select the same ordered structural "
            "compartments in every batch replica before it can be projected "
            "onto a MultiPopulation."
        )
    reference = core_indices[batch_indices == replicas[0]]
    for replica in replicas[1:]:
        candidate = core_indices[batch_indices == replica]
        if not torch.equal(candidate, reference):
            raise ValueError(
                "A component Slice label must select the same ordered structural "
                "compartments in every batch replica before it can be projected "
                "onto a MultiPopulation."
            )
    return reference


def _expand_v_init_like(value, target_shape, *, device, dtype, name="v_init"):
    """Expand one v_init value using Population.expanded_v_init semantics.

    This helper mirrors the model-level rules without mutating the source
    population. It is used for per-population overrides in MultiPopulation.
    """
    target_shape = tuple(int(x) for x in target_shape)
    if len(target_shape) < 2:
        raise ValueError(
            f"target_shape must have at least population and compartment axes; got {target_shape}."
        )

    n_pop = int(target_shape[-2])
    n_comp = int(target_shape[-1])
    v0 = torch.as_tensor(value, device=device, dtype=dtype)

    # Scalar or scalar-like tensor/list: broadcast everywhere.
    if v0.ndim == 0 or v0.numel() == 1:
        return v0.reshape(()).expand(target_shape)

    # Length-nc vector: one initial value per compartment, broadcast across
    # cells/fibers and batch dimensions.
    if v0.ndim == 1:
        if v0.numel() != n_comp:
            raise ValueError(
                f"{name} has length {v0.numel()}, but the target population has nc={n_comp}. "
                "Use a scalar, a vector of length nc, or an explicit core/full shape."
            )
        view_shape = (1,) * (len(target_shape) - 1) + (n_comp,)
        return v0.reshape(view_shape).expand(target_shape)

    # Explicit initial voltage for the unbatched core shape.
    core_shape = (n_pop, n_comp)
    if tuple(v0.shape) == core_shape:
        view_shape = (1,) * (len(target_shape) - 2) + core_shape
        return v0.reshape(view_shape).expand(target_shape)

    # Explicit initial voltage for the full target shape.
    if tuple(v0.shape) == target_shape:
        return v0

    # Common explicit-broadcast form: [1, nc].
    if tuple(v0.shape) == (1, n_comp):
        view_shape = (1,) * (len(target_shape) - 2) + (1, n_comp)
        return v0.reshape(view_shape).expand(target_shape)

    raise ValueError(
        f"Unsupported {name} shape. Expected a scalar, a 1D vector of length nc "
        f"({n_comp}), shape (np, nc) = {core_shape}, or full shape {target_shape}; "
        f"got shape {tuple(v0.shape)}."
    )


def _expanded_population_v_init(pop, value=_MISSING, *, name="v_init"):
    """Return one component population's v_init expanded to ``pop.shape``."""
    if value is _MISSING and hasattr(pop, "expanded_v_init"):
        return pop.expanded_v_init(tuple(pop.shape))

    value = pop.v_init if value is _MISSING else value
    return _expand_v_init_like(
        value,
        tuple(pop.shape),
        device=pop.device(),
        dtype=pop.dtype(),
        name=name,
    )


def _concat_population_v_init(populations, v_init=None):
    """Flatten and concatenate component population initial voltages."""
    if isinstance(v_init, dict):
        unknown = set(v_init) - set(populations)
        if unknown:
            unknown_s = ", ".join(sorted(unknown))
            raise KeyError(
                f"Unknown population name(s) in v_init override: {unknown_s}"
            )

    parts = []
    for name, pop in populations.items():
        value = _MISSING
        if isinstance(v_init, dict) and name in v_init:
            value = v_init[name]
        parts.append(
            _expanded_population_v_init(pop, value, name=f"v_init[{name!r}]")
            .reshape(-1)
            .contiguous()
        )
    return torch.cat(parts, dim=0)


class MultiPopulation(Population):
    """One packed model composed of independent scalar or extracellular models.

    Prefer :func:`concat_models` so Dendra can select the appropriate packed
    integrator. Component topology, geometry, physical scale factors,
    mechanisms, injections, initial voltages, and structural labels remain
    component-specific; temperature is shared by the composite.

    Parameters
    ----------
    integrator : callable, optional
        Integrator factory. Selected automatically when ``None``.
    celsius : float, optional
        Shared simulation temperature.
    v_init : scalar, tensor-like, dict[str, tensor-like], optional
        Initial voltage for the composite population. If ``None``, the
        composite ``v_init`` is built by expanding and concatenating each
        component population's own ``v_init``. If a mapping is provided, keys
        are component population names and omitted populations keep their own
        ``v_init``.
    **populations
        Mapping of population names to unbatched instances on one device and
        dtype. Supported public components are ordinary Population,
        SingleCompartment, Tree, Cable, Unmyelinated, and Myelinated models, or
        a collection containing only ExtCellAxon and ExtCellTree models.

    Notes
    -----
    ``write_back=True`` is an integrator-factory option supplied by
    :func:`concat_models`, not a constructor argument here. Scalar/block mixtures
    and nested MultiPopulation components are rejected fail-closed.
    """

    def __init__(self, integrator=None, celsius=37.0, v_init=None, **populations):
        if not populations:
            raise ValueError("At least one population is required.")
        _validate_component_input_bindings(populations)
        if any(b.is_batched() for b in populations.values()):
            raise ValueError("All populations must be unbatched.")

        is_block = _validate_block_populations(populations)
        if any(
            isinstance(population, MultiPopulation)
            for population in populations.values()
        ):
            raise TypeError("Nested MultiPopulation components are unsupported.")
        if is_block:
            self.n_layers = 2

        # Check all populations are on the same device/dtype before composing
        # tensor-valued v_init values from them.
        _validate_component_device_and_dtype(populations)
        devices = {pop.device() for pop in populations.values()}
        dtypes = {pop.dtype() for pop in populations.values()}

        if integrator is None:
            integrator = _assess_type_and_make_integrator(populations)
        if (
            is_block
            and getattr(integrator, "_FUNCTIONAL_OPERATOR_KIND", None)
            != "block_multi_tree"
        ):
            raise TypeError(
                "Extracellular components require a block multi-solver; "
                "use dhs_bt_multi so independent component circuits remain disconnected."
            )
        if not is_block and "vc" in tuple(getattr(integrator, "v_vars", ())):
            raise TypeError("Scalar components cannot use a block multi-solver.")

        C = sum(math.prod(pop.shape) for pop in populations.values())
        composite_v_init = (
            _concat_population_v_init(populations, v_init)
            if v_init is None or isinstance(v_init, dict)
            else v_init
        )
        composite_cm = _concat_component_field(populations, "cm")
        composite_rhoa = _concat_component_field(populations, "rhoa")
        component_fields = {
            name: _concat_component_field(populations, name)
            for name in ("diam", "dx", "x", "y", "z")
        }
        component_area = _concat_component_field(populations, "area")
        component_scales = {
            name: _concat_component_field(populations, name)
            for name in ("cm_scale", "rhoa_scale", "area_scale")
        }

        init_device = next(iter(devices))
        init_dtype = next(iter(dtypes))
        super().__init__(
            1,
            C,
            integrator=integrator,
            celsius=celsius,
            v_init=composite_v_init,
            device=init_device,
            dtype=init_dtype,
            cm=composite_cm,
            rhoa=composite_rhoa,
        )

        # Population geometry buffers are not declared RANGE parameters, so
        # construct them explicitly from the component models.  Keeping the
        # exact component area separately also preserves graph-supplied areas
        # that cannot necessarily be reconstructed from ``diam * dx``.
        for name, value in component_fields.items():
            getattr(self, name).copy_(value)
        for name, value in component_scales.items():
            setattr(self, name, value)
        self.register_buffer("_component_area", component_area)

        self.populations = torch.nn.ModuleDict(populations)
        self._ion_style.update(_merged_ion_styles(self.populations))
        self._packed_material_process_classes = {}
        self._component_material_geometry_names = {}
        self._component_material_quantity_names = {}
        self._material_field_support = {}
        self._configure_component_materials()
        self._refresh_material_geometry()
        self._material_geometry = _PackedMaterialGeometry(self)
        if is_block:
            for name in ("xraxial", "xc", "xg"):
                self.register_buffer(
                    name, _concat_component_shell_field(self.populations, name)
                )

        for destination, attribute, defaults in (
            (self._equilibria, "_equilibria", reversals()),
            (self._concentrations, "_concentrations", cinits()),
        ):
            sources = [
                _component_ion_initials(pop, attribute)
                for pop in self.populations.values()
            ]
            names = set().union(*(set(source) for source in sources))
            for name in sorted(names):
                destination[name] = _PackedMaterialInitial(
                    [source.get(name, defaults[name]) for source in sources],
                    [pop.core_shape() for pop in self.populations.values()],
                )

        # Re-validate/apply after self.populations exists so the overridden
        # device()/dtype() methods resolve to the component population device.
        self.set_v_init(composite_v_init)
        self.v = self.expanded_v_init((1, C)).clone().contiguous()
        if is_block:
            vc = torch.zeros(*self.shape, 3, device=self.device(), dtype=self.dtype())
            vc[..., 0] = self.v
            self.register_buffer("vc", vc)
        self._sync_core_buffers_to_population_device()

        if all(hasattr(pop, "names") for pop in self.populations.values()):
            self.names = []
            for name, pop in self.populations.items():
                self.names.extend([f"{name}.{n}" for n in pop.names])

        self.reinsert_all()
        self.reinject_all()
        self.register_labels()

    @property
    def area(self):
        """Component membrane areas in the composite core ordering."""
        component_area = getattr(self, "_component_area", None)
        if component_area is not None:
            return component_area
        return super().area

    @property
    def material_edge_index(self):
        """Independent chemical edges in stable packed child-storage order."""
        return torch.stack((self.diff_edge_parent, self.diff_edge_child), dim=0)

    @property
    def material_edges(self):
        return self.diff_edge_parent, self.diff_edge_child

    def material_volume(self, domain="intracellular"):
        """Return exact component chemical storage volumes, not a packed tube."""
        domain = str(domain or "intracellular").lower()
        if domain in {"i", "inside", "cytosol", "cytoplasm", "intracellular"}:
            return self.volume_i
        if domain in {"o", "outside", "extracellular"}:
            return self.volume_o
        if domain in {"total", "volume", "all"}:
            return self.volume
        if domain in {"membrane", "surface", "area"}:
            return self.area
        raise ValueError(f"Unsupported material domain {domain!r}.")

    def material(
        self,
        name,
        fields=None,
        *,
        initial_values=None,
        min_values=None,
        specs=None,
        domain=None,
        units=None,
        conserved=None,
        **field_initials,
    ):
        """Declare packed-parent fields globally without changing other fields.

        Fields inherited from children retain their independent availability.
        Explicitly reconfiguring a field on the packed parent makes that field
        available throughout the parent. Partial updates preserve the other
        fields and their metadata; they do not discard concatenated specs.
        """
        name = _canonical_material_name(name)
        if name in valid_ions():
            return super().material(name, fields=fields)
        initial_values = dict(initial_values or {})
        initial_values.update(field_initials)
        minima = dict(min_values or {})
        declared = {}
        if fields is not None:
            if isinstance(fields, Mapping):
                declared.update(fields)
            else:
                names = (fields,) if isinstance(fields, str) else fields
                declared.update((str(field), 0.0) for field in names)
        declared.update(initial_values)
        existing = self._material_configs.get(name, {})
        existing_kwargs = self._material_constructor_kwargs(name) if existing else {}
        if (
            any(
                existing_kwargs.get(key)
                for key in ("specs", "fields", "initial_values")
            )
            or name in material_specs()
        ):
            previous = Material(name, self.core_shape(), **existing_kwargs)
            merged = {field: previous.field_spec(field) for field in previous.fields}
        else:
            merged = dict(material_specs().get(name, {}))
        touched = set(declared) | set(minima) | set(specs or {})
        for metadata in (domain, units, conserved):
            if isinstance(metadata, Mapping):
                touched.update(metadata)
            elif metadata is not None:
                touched.update(merged)
        if not touched:
            return super().material(
                name,
                fields=fields,
                initial_values=initial_values,
                min_values=min_values,
                specs=specs,
                domain=domain,
                units=units,
                conserved=conserved,
            )

        def selected(metadata, field, default):
            if isinstance(metadata, Mapping):
                return metadata.get(field, default)
            return default if metadata is None else metadata

        for field in sorted(touched):
            previous = (
                specs[field]
                if specs is not None and field in specs
                else merged.get(field, MaterialFieldSpec(name=field))
            )
            merged[field] = replace(
                previous,
                initial=declared.get(field, previous.initial),
                min_value=minima.get(field, previous.min_value),
                domain=str(selected(domain, field, previous.domain)),
                units=selected(units, field, previous.units),
                conserved=bool(selected(conserved, field, previous.conserved)),
            )
        result = super().material(name, specs=merged)
        supports = getattr(self, "_material_field_support", None)
        if supports is not None:
            for field in touched:
                supports[(name, field)] = torch.ones(
                    self.core_shape(), device=self.device(), dtype=torch.bool
                )
        return result

    def _set_material_geometry_buffer(self, name, value):
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def _configure_component_materials(self):
        declarations = {
            name: _component_material_specs(pop)
            for name, pop in self.populations.items()
        }
        material_names = sorted(
            set().union(*(set(configs) for configs in declarations.values()))
        )
        shapes = [pop.core_shape() for pop in self.populations.values()]
        for material_name in material_names:
            field_names = sorted(
                set().union(
                    *(
                        set(configs.get(material_name, {}))
                        for configs in declarations.values()
                    )
                )
            )
            specs = {}
            for field in field_names:
                present = [
                    configs.get(material_name, {}).get(field)
                    for configs in declarations.values()
                ]
                authored = [spec for spec in present if spec is not None]
                first = authored[0]
                for spec in authored[1:]:
                    if (_canonical_domain(spec.domain), spec.units, spec.conserved) != (
                        _canonical_domain(first.domain),
                        first.units,
                        first.conserved,
                    ):
                        raise ValueError(
                            f"Material field {material_name}.{field} has incompatible "
                            "domain, units, or conservation metadata across components."
                        )
                sources = [
                    spec.initial
                    if spec is not None
                    else torch.zeros((), device=self.device(), dtype=self.dtype())
                    for spec in present
                ]
                minimum = None
                if any(spec.min_value is not None for spec in authored):
                    minimum = torch.cat(
                        [
                            torch.as_tensor(
                                spec.min_value
                                if spec is not None and spec.min_value is not None
                                else -torch.inf,
                                device=self.device(),
                                dtype=self.dtype(),
                            )
                            .expand(shape)
                            .reshape(-1)
                            for spec, shape in zip(present, shapes)
                        ]
                    ).reshape(1, -1)
                specs[field] = replace(
                    first,
                    initial=_PackedMaterialInitial(sources, shapes),
                    min_value=minimum,
                )
                self._material_field_support[(material_name, field)] = torch.cat(
                    [
                        torch.full(
                            (math.prod(shape),),
                            spec is not None,
                            device=self.device(),
                            dtype=torch.bool,
                        )
                        for spec, shape in zip(present, shapes)
                    ]
                ).reshape(1, -1)
            super().material(material_name, specs=specs)

    def _refresh_material_geometry(self, *, diam=None, dx=None):
        """Assemble a disconnected chemical forest and differentiable geometry."""
        helper = getattr(self, "_material_geometry", None)
        if helper is not None:
            helper.refresh_sources(self)
        configure_topology = not getattr(self, "_material_forest_configured", False)
        parents, factors, volumes, volumes_i, volumes_o, analytic = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        offset = 0
        component_edge_counts = {}
        for component_name, pop in self.populations.items():
            shape = tuple(pop.shape)
            core = tuple(pop.core_shape())
            count, compartments = core
            batch_shape = shape[:-2]
            native = hasattr(pop, "diff_parent_index")
            if configure_topology:
                local_parent = (
                    pop.diff_parent_index.to(device=self.device(), dtype=torch.long)
                    if native
                    else torch.arange(compartments, device=self.device()) - 1
                )
                row_offsets = (
                    torch.arange(count, device=self.device()).unsqueeze(-1)
                    * compartments
                    + offset
                )
                parents.append(
                    torch.where(
                        local_parent >= 0, local_parent + row_offsets, -1
                    ).reshape(-1)
                )
            component_diam = (
                torch.broadcast_to(pop.diam, shape)
                if diam is None
                else diam[..., offset : offset + math.prod(core)].reshape(shape)
            )
            component_dx = (
                torch.broadcast_to(pop.dx, shape)
                if dx is None
                else dx[..., offset : offset + math.prod(core)].reshape(shape)
            )
            area = torch.pi * (component_diam * 0.5) ** 2
            cylinder_volume = area * component_dx
            volume = torch.broadcast_to(getattr(pop, "volume", cylinder_volume), shape)
            volume_i = torch.broadcast_to(getattr(pop, "volume_i", volume), shape)
            volume_o = torch.broadcast_to(
                getattr(pop, "volume_o", torch.zeros_like(volume)), shape
            )
            if native:
                factor = torch.broadcast_to(pop.diff_geom_um, shape)
            else:
                edge = (
                    0.5
                    * (area[..., :-1] + area[..., 1:])
                    / (
                        0.5 * (component_dx[..., :-1] + component_dx[..., 1:])
                    ).clamp_min(torch.finfo(component_dx.dtype).tiny)
                )
                factor = torch.cat(
                    (torch.zeros_like(component_dx[..., :1]), edge), dim=-1
                )
            for destination, value in (
                (volumes, volume),
                (volumes_i, volume_i),
                (volumes_o, volume_o),
                (factors, factor),
            ):
                destination.append(value.reshape(*batch_shape, -1))
            if configure_topology:
                analytic.append(
                    torch.full(
                        (math.prod(core),),
                        not native,
                        device=self.device(),
                        dtype=torch.bool,
                    )
                )
                component_edge_counts[component_name] = count * int(
                    (local_parent >= 0).sum()
                )
            offset += math.prod(core)
        if configure_topology:
            parent = torch.cat(parents)
            child = torch.nonzero(parent >= 0, as_tuple=False).flatten()
            for name, value in (
                ("diff_parent_index", parent),
                ("diff_edge_child", child),
                ("diff_edge_parent", parent.index_select(0, child)),
                ("_material_analytic_geometry", torch.cat(analytic).reshape(1, -1)),
            ):
                self._set_material_geometry_buffer(name, value)
            self._material_component_edge_counts = component_edge_counts
            self._material_forest_configured = True
        for name, value in (
            ("volume", torch.cat(volumes, dim=-1).unsqueeze(-2)),
            ("volume_i", torch.cat(volumes_i, dim=-1).unsqueeze(-2)),
            ("volume_o", torch.cat(volumes_o, dim=-1).unsqueeze(-2)),
            ("diff_geom_um", torch.cat(factors, dim=-1).unsqueeze(-2)),
        ):
            self._set_material_geometry_buffer(name, value)
        self._refresh_named_material_geometry()

    def _refresh_named_material_geometry(self):
        for component_index, (component_name, pop) in enumerate(
            self.populations.items()
        ):
            names = self._component_material_geometry_names.setdefault(
                component_name, {}
            )
            quantities = self._component_material_quantity_names.setdefault(
                component_name, {}
            )
            for local_name, config in pop._material_geometries.items():
                name = names.setdefault(
                    local_name, f"packed_{component_index}_{local_name}"
                )
                packed = {"name": name, "domain": config["domain"]}
                for role, source in config.items():
                    if role in {"name", "domain"}:
                        continue
                    value = getattr(pop, source)
                    parts = []
                    for other_name, other in self.populations.items():
                        batch_shape = tuple(other.shape[:-2])
                        if role == "volume":
                            target_shape = tuple(other.shape)
                            fill = 1.0
                        else:
                            edges = self._material_component_edge_counts[other_name]
                            target_shape = (*batch_shape, edges)
                            fill = 1.0 if role == "edge_distance" else 0.0
                        if other_name == component_name:
                            if role != "volume":
                                per_row = (
                                    self._material_component_edge_counts[other_name]
                                    // other.np
                                )
                                value = torch.broadcast_to(
                                    value,
                                    (*batch_shape, other.np, per_row),
                                ).reshape(*batch_shape, -1)
                            part = torch.broadcast_to(value, target_shape)
                        else:
                            part = torch.full(
                                target_shape,
                                fill,
                                device=self.device(),
                                dtype=self.dtype(),
                            )
                        parts.append(part.reshape(*batch_shape, -1))
                    tensor = torch.cat(parts, dim=-1).unsqueeze(-2)
                    destination = f"_packed_material_geometry_{component_index}_{local_name}_{role}"
                    self._set_material_geometry_buffer(destination, tensor)
                    packed[role] = destination
                self._material_geometries[name] = packed
            # Population-owned edge diffusivities and other authored process
            # references keep their original leaf identity, independently of
            # same-named quantities on another component.
            process_classes = set(pop._mech_everywhere) | set(pop._mech_data)
            for process_class in process_classes:
                if not issubclass(process_class, MaterialProcess):
                    continue
                sources = {
                    spec.D
                    for spec in getattr(process_class, "_diffusion_specs", ())
                    if isinstance(spec.D, str) and spec.D_location == "edge"
                }
                for spec in getattr(process_class, "_diffusion_relaxation_specs", ()):
                    sources.update(
                        value
                        for value in (spec.rate, spec.target, spec.where)
                        if isinstance(value, str)
                    )
                for source in sorted(sources):
                    if source not in pop._buffers and source not in pop._parameters:
                        continue
                    destination = quantities.setdefault(
                        source, f"_packed_material_quantity_{component_index}_{source}"
                    )
                    value = getattr(pop, source)
                    if source in pop._parameters:
                        if destination not in self._parameters:
                            self.register_parameter(destination, value)
                        else:
                            self._parameters[destination] = value
                    else:
                        self._set_material_geometry_buffer(destination, value)

    def _component_material_process_class(self, component_name, original):
        key = (component_name, original)
        if key not in self._packed_material_process_classes:
            component_index = tuple(self.populations).index(component_name)
            scoped = _make_packed_material_process_class(
                original,
                component_name,
                f"Packed{component_index}_{original.__name__}",
                self._component_material_geometry_names[component_name],
                self._component_material_quantity_names[component_name],
            )
            self._packed_material_process_classes[key] = scoped
        return self._packed_material_process_classes[key]

    def __iter__(self):
        return iter(self.populations.values())

    def __len__(self):
        return len(self.populations)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # A same-sized packed model need not have the same chemical forest.
        # Loading its parent indices would silently reconnect independent
        # components while leaving the authored electrical topology unchanged.
        for name in ("diff_parent_index", "diff_edge_parent", "diff_edge_child"):
            incoming = state_dict.get(f"{prefix}{name}")
            expected = getattr(self, name, None)
            if (
                incoming is not None
                and expected is not None
                and (
                    incoming.dtype != expected.dtype
                    or incoming.shape != expected.shape
                    or not torch.equal(incoming.detach().cpu(), expected.detach().cpu())
                )
            ):
                error_msgs.append(
                    f"Packed material topology buffer {prefix}{name!s} disagrees "
                    "with the component chemical forest."
                )
                return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def device(self):
        populations = getattr(self, "populations", None)
        if populations is not None and len(populations) > 0:
            return next(iter(populations.values())).device()
        return Population.device(self)

    def dtype(self):
        populations = getattr(self, "populations", None)
        if populations is not None and len(populations) > 0:
            return next(iter(populations.values())).dtype()
        return Population.dtype(self)

    def _validate_static_runtime_contracts(self, mode):
        """Validate immutable component contracts without using child solvers."""
        for population in self.populations.values():
            population._validate_static_runtime_contracts(mode)

    def _runtime_workspace_rebuild_pending(self):
        """Require owner initialization to refresh concatenated component fields."""
        # A forced packed-integrator rebuild consumes the MultiPopulation's
        # concatenated copies; it does not call _refresh_component_fields().
        # Therefore training-mode force=True alone cannot make a changed child
        # dependency current. Fail closed until MultiPopulation.initialize().
        return False

    def _integrator_workspace_contract_signature(self):
        """Track packed copies and every component solver dependency."""
        local = []
        for name, value in self._workspace_contract_inputs().items():
            if not torch.is_tensor(value):
                return None
            try:
                version = value._version
            except RuntimeError:
                version = None
            local.append(
                (
                    name,
                    id(value),
                    version,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                    value.dtype,
                    value.device,
                    value.layout,
                    value.data_ptr(),
                )
            )
        components = tuple(
            (name, population._integrator_workspace_contract_signature())
            for name, population in self.populations.items()
        )
        return tuple(local), components

    def _workspace_contract_inputs(self):
        """Include block child physics: packed copies refresh at initialization."""
        names = (
            "cm",
            "rhoa",
            "cm_scale",
            "rhoa_scale",
            "area_scale",
            "diam",
            "dx",
            "_component_area",
        )
        inputs = {name: getattr(self, name, None) for name in names}
        if getattr(self, "n_layers", 0):
            for name in ("xraxial", "xc", "xg"):
                inputs[name] = getattr(self, name, None)
            for component_name, population in self.populations.items():
                # Legacy ExtCellAxon has no canonical Cable workspace signature.
                # Changes to any child field used by the packed block workspace
                # must therefore be observed here, not only shell changes. Area
                # is derived from diam/dx for those axons; tracking that property
                # itself would manufacture a new tensor identity on every check.
                for name in (
                    "cm",
                    "rhoa",
                    "cm_scale",
                    "rhoa_scale",
                    "area_scale",
                    "diam",
                    "dx",
                    "xraxial",
                    "xc",
                    "xg",
                ):
                    parametrizations = getattr(population, "parametrizations", None)
                    if parametrizations is not None and name in parametrizations:
                        # Parametrized public values may be fresh views on every
                        # access. Their stable original/parameter/buffer inputs
                        # carry the actual storage/version invalidation contract.
                        owner = parametrizations[name]
                        for path, value in (
                            *owner.named_parameters(remove_duplicate=False),
                            *owner.named_buffers(remove_duplicate=False),
                        ):
                            inputs[
                                f"populations.{component_name}.parametrizations.{name}.{path}"
                            ] = value
                    else:
                        inputs[f"populations.{component_name}.{name}"] = getattr(
                            population, name, None
                        )
        return inputs

    def _record_unversioned_integrator_workspace_values(self, signature):
        """Snapshot packed inference-tensor inputs that expose no versions."""
        local_signature, component_signatures = signature
        local_snapshots = None
        if any(entry[2] is None for entry in local_signature):
            local_snapshots = tuple(
                (name, value.detach().clone())
                for name, value in self._workspace_contract_inputs().items()
            )

        component_snapshots = {}
        signatures = dict(component_signatures)
        for name, population in self.populations.items():
            component_signature = signatures.get(name)
            value_names = getattr(
                population, "_UNVERSIONED_WORKSPACE_VALUE_TENSORS", ()
            )
            if (
                component_signature is not None
                and value_names
                and any(entry[2] is None for entry in component_signature)
            ):
                component_snapshots[name] = tuple(
                    (field, getattr(population, field).detach().clone())
                    for field in value_names
                )

        self._validated_unversioned_workspace_values = (
            local_snapshots,
            component_snapshots,
        )

    def _unversioned_integrator_workspace_values_match(self, signature=None):
        """Compare packed inference dependencies with their built snapshots."""
        if signature is None:
            signature = self._integrator_workspace_contract_signature()
        if signature is None:
            return True
        local_signature, component_signatures = signature
        local_needs_values = any(entry[2] is None for entry in local_signature)
        component_needs_values = {
            name
            for name, component_signature in component_signatures
            if component_signature is not None
            and any(entry[2] is None for entry in component_signature)
        }
        if not local_needs_values and not component_needs_values:
            return True

        snapshots = getattr(self, "_validated_unversioned_workspace_values", None)
        if snapshots is None:
            return False
        local_snapshots, component_snapshots = snapshots
        if local_needs_values and (
            local_snapshots is None
            or not all(
                torch.equal(self._workspace_contract_inputs()[name], expected)
                for name, expected in local_snapshots
            )
        ):
            return False
        for name in component_needs_values:
            expected_values = component_snapshots.get(name)
            if expected_values is None:
                return False
            population = self.populations[name]
            if not all(
                torch.equal(getattr(population, field), expected)
                for field, expected in expected_values
            ):
                return False
        return True

    def __getstate__(self):
        """Serialize packed workspace validity without process-local identities."""
        state = super().__getstate__()
        current = self._integrator_workspace_contract_signature()
        integrator = getattr(self, "integrator", None)
        state["_serialized_integrator_workspace_contract_valid"] = bool(
            current is not None
            and integrator is not None
            and integrator.initialized
            and current
            == getattr(self, "_validated_integrator_workspace_signature", None)
            and self._unversioned_integrator_workspace_values_match(current)
        )
        state["_validated_integrator_workspace_signature"] = None
        state["_validated_unversioned_workspace_values"] = None
        return state

    def __setstate__(self, state):
        """Rebase a coherent packed workspace after deserialization."""
        workspace_was_valid = bool(
            state.pop("_serialized_integrator_workspace_contract_valid", False)
        )
        super().__setstate__(state)
        self._validated_integrator_workspace_signature = None
        self._validated_unversioned_workspace_values = None
        if workspace_was_valid:
            self._record_integrator_workspace_contracts()

    def _validate_integrator_rebuild_contracts(self):
        """Validate component contracts before packed solver workspace rebuilds."""
        for population in self.populations.values():
            population._validate_integrator_rebuild_contracts()

    def _sync_core_buffers_to_population_device(self):
        """Keep composite buffers on the same device/dtype as components."""
        device = self.device()
        dtype = self.dtype()
        for name in (
            "v",
            "cm",
            "rhoa",
            "cm_scale",
            "rhoa_scale",
            "area_scale",
            "diam",
            "dx",
            "x",
            "y",
            "z",
            "_component_area",
            "vc",
            "xraxial",
            "xc",
            "xg",
        ):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        if hasattr(self, "t"):
            self.t = self.t.to(device=device, dtype=dtype)
        if self.imem and self.i_membrane is not None:
            self.i_membrane = self.i_membrane.to(device=device, dtype=dtype)

    def _refresh_component_fields(self):
        """Refresh composite physical fields from the component populations.

        Component parameter graphs remain the source of truth.  In particular,
        Myelinated computes its effective axial resistivity through a parameter
        graph, so retaining constructor-time copies would silently discard that
        transform during a composite initialization.
        """
        for name in (
            "cm",
            "rhoa",
            "cm_scale",
            "rhoa_scale",
            "area_scale",
            "diam",
            "dx",
            "x",
            "y",
            "z",
        ):
            setattr(self, name, _concat_component_field(self.populations, name))
        self._component_area = _concat_component_field(self.populations, "area")
        self._refresh_material_geometry()
        if getattr(self, "n_layers", 0):
            for name in ("xraxial", "xc", "xg"):
                setattr(
                    self, name, _concat_component_shell_field(self.populations, name)
                )
        self._sync_core_buffers_to_population_device()

    @classmethod
    def _rematerialize_parameter_owner(cls, owner, *, replay_random=False):
        # Continued training rebuilds packed physical fields from their child
        # sources. Refresh those root owners after the shared cache-clear pass,
        # without initializing or retaining any child mechanism/state runtime.
        if replay_random and isinstance(owner, MultiPopulation):
            for population in owner.populations.values():
                cls._rematerialize_parameter_owner(population, replay_random=True)
                population._refresh_parameter_views_for_initialization()
        return super()._rematerialize_parameter_owner(
            owner, replay_random=replay_random
        )

    def populate_parameter_buffers(self, random_generation=None):
        """Populate component parameters, then synchronize composite geometry."""
        for population in self.populations.values():
            population.populate_parameter_buffers(random_generation=random_generation)
        super().populate_parameter_buffers(random_generation=random_generation)
        self._refresh_component_fields()

    def _refresh_parameter_views_for_initialization(self):
        """Refresh packed copies even when direct overrides skip repopulation."""
        self._refresh_component_fields()

    def set_v_init(self, v_init):
        """Set composite initial voltage, with optional per-population mapping.

        ``v_init`` may be any form accepted by ``Population.expanded_v_init``
        for the flattened composite model, or a mapping from component
        population name to that population's own scalar/vector/full-shape
        initial voltage specification.
        """
        if isinstance(v_init, dict):
            v_init = _concat_population_v_init(self.populations, v_init)
        return super().set_v_init(v_init)

    def register_labels(self):
        """Propagate labels from component populations to the composite."""
        self.clear_labels()
        all_indices = indices(self.populations)
        for index, (name, pop) in zip(all_indices, self.populations.items()):
            label_name = name
            self[:, index.flatten()].label(label_name)
            self._sync_component_labels(label_name, pop)

    def _sync_component_labels(self, name, pop):
        """Synchronize one component's nested labels on its composite slice."""
        population_slice = getattr(self, name)
        namespace = object.__getattribute__(population_slice, "__dict__")
        registry = object.__getattribute__(population_slice, "_labels")
        previous = dict(namespace.get("_component_labels", {}))
        previous_sources = dict(namespace.get("_component_label_sources", {}))

        # Validate the entire update before changing either namespace. A label
        # explicitly attached to the composite component slice belongs to the
        # user; component-label synchronization must not silently replace it.
        for label in pop._labels:
            previous_label = previous.get(label)
            registered = registry.get(label)
            attribute = namespace.get(label, _MISSING)
            if registered is not None and registered is not previous_label:
                raise ValueError(
                    f"Component label {label!r} conflicts with a nested label "
                    f"already owned by composite component {name!r}."
                )
            if attribute is not _MISSING and attribute is not previous_label:
                raise ValueError(
                    f"Component label {label!r} conflicts with an attribute "
                    f"already present on composite component {name!r}."
                )

        component_labels = {}
        for label, component_slice in pop._labels.items():
            previous_label = previous.get(label)
            unchanged = (
                previous_label is not None
                and previous_sources.get(label) is component_slice
                and registry.get(label) is previous_label
                and namespace.get(label) is previous_label
            )
            if unchanged:
                component_labels[label] = previous_label
                continue

            # Project replicated batch coordinates to one exact ordered core
            # sequence, then address the flattened component axis behind every
            # composite batch axis.
            core_flat = _component_label_core_indices(pop, component_slice)
            component_labels[label] = population_slice[..., core_flat]

        current = set(pop._labels)
        for label, previous_label in previous.items():
            if label in current:
                continue
            if registry.get(label) is previous_label:
                registry.pop(label)
            if namespace.get(label) is previous_label:
                object.__delattr__(population_slice, label)

        for label, nested in component_labels.items():
            object.__setattr__(population_slice, label, nested)
            registry[label] = nested

        object.__setattr__(population_slice, "_component_labels", component_labels)
        object.__setattr__(
            population_slice, "_component_label_sources", dict(pop._labels)
        )
        return list(component_labels.values())

    def reinject_all(self):
        """Reinject intracellular currents for all component populations."""
        all_indices = indices(self.populations)
        for index, (name, pop) in zip(all_indices, self.populations.items()):
            for stim, _, idx in pop.injections:
                self[:, key_to_flat_index(index, idx)].inject(stim)

    def reinsert_all(self):
        """Recreate mechanisms for all component populations."""
        all_indices = indices(self.populations)
        merged_globals = _merged_global_overrides(self.populations)
        for index, (name, pop) in zip(all_indices, self.populations.items()):
            # first do _mech_everywhere
            for m_class, (_, ic, kwargs) in pop._mech_everywhere.items():
                process = issubclass(m_class, MaterialProcess)
                insertion_class = (
                    self._component_material_process_class(name, m_class)
                    if process
                    else m_class
                )
                global_overrides = (
                    _component_explicit_globals(pop, m_class)
                    if process
                    else merged_globals[m_class]
                )
                alias = name
                global_names = _mechanism_global_parameter_names(m_class)
                indexed_kwargs = {
                    parameter_name: value
                    for parameter_name, value in kwargs.items()
                    if parameter_name not in global_names
                }
                effective_ic = dict(ic or {})
                local_flat = torch.arange(math.prod(pop.core_shape()), dtype=torch.long)
                excluded = torch.as_tensor(
                    pop._mech_exclusions.get(m_class, []), dtype=torch.long
                ).reshape(-1)
                if excluded.numel():
                    local_flat = local_flat[~torch.isin(local_flat, excluded)]
                local_key = _core_key_from_flat(local_flat, tuple(pop.core_shape()))
                index_f = key_to_flat_index(index, local_key)
                self[:, index_f].insert(
                    insertion_class,
                    alias=alias,
                    ic=effective_ic,
                    **global_overrides,
                    **indexed_kwargs,
                )
            # now do _mech_data
            for m_class, list_of_aliases_kwargs_keys in pop._mech_data.items():
                process = issubclass(m_class, MaterialProcess)
                insertion_class = (
                    self._component_material_process_class(name, m_class)
                    if process
                    else m_class
                )
                global_overrides = (
                    _component_explicit_globals(pop, m_class)
                    if process
                    else merged_globals[m_class]
                )
                idx = 0
                global_names = _mechanism_global_parameter_names(m_class)
                effective_ic = dict(pop._mech_data_ic.get(m_class) or {})
                for record in list_of_aliases_kwargs_keys:
                    (
                        alias,
                        kwargs,
                        key,
                        preserve_duplicate_indices,
                        copies,
                    ) = _unpack_mechanism_insertion_record(record)
                    indexed_kwargs = {
                        parameter_name: value
                        for parameter_name, value in kwargs.items()
                        if parameter_name not in global_names
                    }
                    index_f = key_to_flat_index(index, key)
                    if alias is not None:
                        alias_n = f"{name}_{alias}"
                    else:
                        alias_n = f"{name}_{idx}"
                        idx += 1
                    self[:, index_f].insert(
                        insertion_class,
                        alias=alias_n,
                        ic=effective_ic,
                        preserve_duplicate_indices=preserve_duplicate_indices,
                        copies=copies,
                        **global_overrides,
                        **indexed_kwargs,
                    )

    def batch(self, batch_size: int):
        """Create a batched view of the multi-population."""
        nested_labels = []
        for name, pop in self.populations.items():
            nested_labels.extend(self._sync_component_labels(name, pop))

        super().batch(batch_size)
        for nested_label in nested_labels:
            nested_label._batch()
        # Do not add a leading dimension to v_init. A scalar, length-nc vector,
        # or [1, nc] tensor already expands correctly to [batch, 1, nc].
        for pop in self.populations.values():
            pop.batch(batch_size)
        return self

    def batch_(self, batch_size: int):
        """In-place variant of :meth:`batch`."""
        self.batch(batch_size)
