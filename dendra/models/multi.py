"""Utilities for composing multiple population models into a single system."""

import math

import torch

from ..helpers import logger
from .core import (
    Cable,
    Population,
    _core_key_from_flat,
    _global_configuration_values_equal,
    _mechanism_global_parameter_names,
    _mechanism_initial_defaults,
    _unpack_mechanism_insertion_record,
)
from .integrators import bwd_euler_sc_multi, dhs_multi
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


def _reject_block_populations(populations):
    """Fail closed while no block-state multi-solver is implemented."""
    block = [
        name
        for name, population in populations.items()
        if _multi_solver_capability(population) == "block"
    ]
    if block:
        names = ", ".join(repr(name) for name in block)
        raise TypeError(
            "Block-state population(s) "
            f"{names} require a block multi-solver; scalar dhs_multi and "
            "bwd_euler_sc_multi cannot evolve their extracellular circuit state."
        )


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
    _reject_block_populations(populations)
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


def concat_models(
    populations: dict[str, Population],
    threads=16,
    write_back=True,
    celsius=37.0,
    v_init=None,
):
    """Pack independent scalar populations into one :class:`MultiPopulation`.

    Ordinary :class:`~dendra.models.core.Population` and
    :class:`~dendra.models.core.SingleCompartment` models, branched
    :class:`~dendra.models.tree.Tree` models, and scalar
    :class:`~dendra.models.core.Cable` models (including Axon subclasses) can
    be combined in one call.
    Mixed scalar components use the universal packed DHS solver; a collection
    containing only independent point compartments keeps the optimized
    single-compartment multi-integrator.

    Concatenation does not create axial edges between component populations.
    It packs their independent electrical systems into one launch and exposes
    each component name as a labelled Slice on the composite model.

    Parameters
    ----------
    populations : dict[str, Population]
        Mapping of unique component names to unbatched scalar populations. All
        components must share a device and dtype.
    threads : int, optional
        DHS lane count for packed Tree/Cable groups. Must divide 32.
    write_back : bool, optional
        If ``True``, write solved voltage views back to every component after
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
    Finite-extracellular :class:`~dendra.models.extcell.ExtCellTree` and
    :class:`~dendra.models.extcell.ExtCellAxon` models carry block-valued
    voltage state and are intentionally rejected until a packed block-DHS
    solver is available. Nested ``MultiPopulation`` objects are likewise
    unsupported.
    """
    _check_celsius(celsius, populations)
    integrator = _assess_type_and_make_integrator(
        populations, threads=threads, write_back=write_back
    )
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
    sizes = [math.prod(p.shape) for p in populations.values()]
    off = [0] + list(torch.cumsum(torch.tensor(sizes), dim=0).numpy().astype(int))[:-1]
    return off


def indices(populations):
    """Return per-population index arrays aligned with the concatenation."""
    off = offsets(populations)
    sizes = [math.prod(p.shape) for p in populations.values()]
    indices = [
        torch.arange(s).reshape(p.shape) + off[idx]
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
    """One packed model composed of independent scalar sub-populations.

    Prefer :func:`concat_models` so Dendra can select the appropriate packed
    scalar integrator. Component topology, geometry, physical scale factors,
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
        SingleCompartment, Tree, Cable, Unmyelinated, and Myelinated models.

    Notes
    -----
    ``write_back=True`` is an integrator-factory option supplied by
    :func:`concat_models`, not a constructor argument here. ExtCell block
    states and nested MultiPopulation components are rejected fail-closed.
    """

    def __init__(self, integrator=None, celsius=37.0, v_init=None, **populations):
        if not populations:
            raise ValueError("At least one population is required.")
        if any(b.is_batched() for b in populations.values()):
            raise ValueError("All populations must be unbatched.")

        # There is not yet a packed block-tree solver.  Check this even when a
        # caller supplies an explicit integrator so a scalar multi-integrator
        # cannot silently discard the extracellular circuit state.
        _reject_block_populations(populations)

        # Check all populations are on the same device/dtype before composing
        # tensor-valued v_init values from them.
        devices = {pop.device() for pop in populations.values()}
        dtypes = {pop.dtype() for pop in populations.values()}
        if len(devices) > 1:
            raise ValueError("All populations must be on the same device.")
        if len(dtypes) > 1:
            raise ValueError("All populations must be of the same dtype.")

        if integrator is None:
            integrator = _assess_type_and_make_integrator(populations)

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

        for pop in self.populations.values():
            self._equilibria.update(pop._equilibria)
            self._concentrations.update(pop._concentrations)

        # Re-validate/apply after self.populations exists so the overridden
        # device()/dtype() methods resolve to the component population device.
        self.set_v_init(composite_v_init)
        self.v = self.expanded_v_init((1, C)).clone().contiguous()
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

    def __iter__(self):
        return iter(self.populations.values())

    def __len__(self):
        return len(self.populations)

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
        for name in (
            "cm",
            "rhoa",
            "cm_scale",
            "rhoa_scale",
            "area_scale",
            "diam",
            "dx",
            "_component_area",
        ):
            value = getattr(self, name, None)
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

    def _record_unversioned_integrator_workspace_values(self, signature):
        """Snapshot packed inference-tensor inputs that expose no versions."""
        local_signature, component_signatures = signature
        local_snapshots = None
        if any(entry[2] is None for entry in local_signature):
            local_snapshots = tuple(
                (name, getattr(self, name).detach().clone())
                for name in (
                    "cm",
                    "rhoa",
                    "cm_scale",
                    "rhoa_scale",
                    "area_scale",
                    "diam",
                    "dx",
                    "_component_area",
                )
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
                torch.equal(getattr(self, name), expected)
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
        self._sync_core_buffers_to_population_device()

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
                alias = name
                global_names = _mechanism_global_parameter_names(m_class)
                indexed_kwargs = {
                    parameter_name: value
                    for parameter_name, value in kwargs.items()
                    if parameter_name not in global_names
                }
                effective_ic = _mechanism_initial_defaults(m_class)
                effective_ic.update(ic or {})
                local_flat = torch.arange(math.prod(pop.core_shape()), dtype=torch.long)
                excluded = torch.as_tensor(
                    pop._mech_exclusions.get(m_class, []), dtype=torch.long
                ).reshape(-1)
                if excluded.numel():
                    local_flat = local_flat[~torch.isin(local_flat, excluded)]
                local_key = _core_key_from_flat(local_flat, tuple(pop.core_shape()))
                index_f = key_to_flat_index(index, local_key)
                self[:, index_f].insert(
                    m_class,
                    alias=alias,
                    ic=effective_ic,
                    **merged_globals[m_class],
                    **indexed_kwargs,
                )
            # now do _mech_data
            for m_class, list_of_aliases_kwargs_keys in pop._mech_data.items():
                idx = 0
                global_names = _mechanism_global_parameter_names(m_class)
                effective_ic = _mechanism_initial_defaults(m_class)
                effective_ic.update(pop._mech_data_ic.get(m_class) or {})
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
                        m_class,
                        alias=alias_n,
                        ic=effective_ic,
                        preserve_duplicate_indices=preserve_duplicate_indices,
                        copies=copies,
                        **merged_globals[m_class],
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
