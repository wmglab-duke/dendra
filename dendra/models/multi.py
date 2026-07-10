"""Utilities for composing multiple population models into a single system."""

import math

import torch

from ..helpers import logger
from .core import Population
from .integrators import bwd_euler_sc_multi, dhs_multi
from .tree import Tree

_MISSING = object()


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
    if all(isinstance(pop, Tree) for pop in populations.values()):
        return dhs_multi(threads=threads, write_back=write_back)
    if all(type(pop) is Population for pop in populations.values()):
        return bwd_euler_sc_multi(write_back=write_back)
    raise TypeError("Incompatible population types.")


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
    """Concatenate multiple populations into a :class:`MultiPopulation`.

    Parameters
    ----------
    populations : dict[str, Population]
        Mapping of names to populations to concatenate.
    threads : int, optional
        Number of solver threads for dendritic tree models.
    write_back : bool, optional
        Whether integrators should write-back states in-place.
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
        Combined population.
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


def flatten_key(n, shape, key):
    """Compute flattened indices for ``key`` within a tensor of ``shape``."""
    indices = torch.arange(n).reshape(shape)
    return indices[key].flatten()


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
    """Population composed of multiple independent sub-populations.

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
        Mapping of population names to instances.
    """

    def __init__(self, integrator=None, celsius=37.0, v_init=None, **populations):
        if not populations:
            raise ValueError("At least one population is required.")
        if any(b.is_batched() for b in populations.values()):
            raise ValueError("All populations must be unbatched.")

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
        )

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

    def _sync_core_buffers_to_population_device(self):
        """Keep composite buffers on the same device/dtype as components."""
        device = self.device()
        dtype = self.dtype()
        for name in ("v", "diam", "dx", "x", "y", "z"):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        if hasattr(self, "t"):
            self.t = self.t.to(device=device, dtype=dtype)
        if self.imem and self.i_membrane is not None:
            self.i_membrane = self.i_membrane.to(device=device, dtype=dtype)

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
            for label, slice in pop._labels.items():
                getattr(self, label_name)[
                    :, flatten_key(pop.numel(), pop.shape, slice.index)
                ].label(label)

    def reinject_all(self):
        """Reinject intracellular currents for all component populations."""
        all_indices = indices(self.populations)
        for index, (name, pop) in zip(all_indices, self.populations.items()):
            for stim, _, idx in pop.injections:
                self[:, key_to_flat_index(index, idx)].inject(stim)

    def reinsert_all(self):
        """Recreate mechanisms for all component populations."""
        all_indices = indices(self.populations)
        for index, (name, pop) in zip(all_indices, self.populations.items()):
            # first do _mech_everywhere
            for m_class, (_, _, kwargs) in pop._mech_everywhere.items():
                alias = name
                index_f = index.flatten()
                self[:, index_f].insert(m_class, alias=alias, **kwargs)
            # now do _mech_data
            for m_class, list_of_aliases_kwargs_keys in pop._mech_data.items():
                idx = 0
                for record in list_of_aliases_kwargs_keys:
                    if len(record) == 3:
                        alias, kwargs, key = record
                        preserve_duplicate_indices = False
                        copies = 1
                    elif len(record) == 4:
                        alias, kwargs, key, preserve_duplicate_indices = record
                        copies = 1
                    elif len(record) == 5:
                        alias, kwargs, key, preserve_duplicate_indices, copies = record
                    else:
                        raise ValueError(
                            "Mechanism insertion records must contain either "
                            "(alias, kwargs, key), "
                            "(alias, kwargs, key, preserve_duplicate_indices), "
                            "or (alias, kwargs, key, preserve_duplicate_indices, copies)."
                        )
                    index_f = key_to_flat_index(index, key)
                    if alias is not None:
                        alias_n = f"{name}_{alias}"
                    else:
                        alias_n = f"{name}_{idx}"
                        idx += 1
                    self[:, index_f].insert(
                        m_class,
                        alias=alias_n,
                        preserve_duplicate_indices=preserve_duplicate_indices,
                        copies=copies,
                        **kwargs,
                    )

    def batch(self, batch_size: int):
        """Create a batched view of the multi-population."""
        super().batch(batch_size)
        # Do not add a leading dimension to v_init. A scalar, length-nc vector,
        # or [1, nc] tensor already expands correctly to [batch, 1, nc].
        for pop in self.populations.values():
            pop.batch(batch_size)
        return self

    def batch_(self, batch_size: int):
        """In-place variant of :meth:`batch`."""
        self.batch(batch_size)
