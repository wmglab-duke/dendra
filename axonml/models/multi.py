"""Utilities for composing multiple population models into a single system."""

import math

import torch

from ..helpers import logger
from .core import Population
from .integrators import bwd_euler_sc_multi, dhs_multi
from .tree import Tree


def assess_type_and_make_integrator(populations, threads=16, write_back=True):
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
    TypeError
        If populations are heterogeneous in a way that is not supported.
    """
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


def concat(
    populations: dict[str, Population], threads=16, write_back=True, celsius=37.0
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

    Returns
    -------
    MultiPopulation
        Combined population.
    """
    _check_celsius(celsius, populations)
    integrator = assess_type_and_make_integrator(
        populations, threads=threads, write_back=write_back
    )
    # concatenate x, y, z
    x = torch.cat([pop.x.flatten() for pop in populations.values()], dim=1)
    y = torch.cat([pop.y.flatten() for pop in populations.values()], dim=1)
    z = torch.cat([pop.z.flatten() for pop in populations.values()], dim=1)
    mp = MultiPopulation(
        integrator=integrator,
        celsius=celsius,
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


class MultiPopulation(Population):
    """Population composed of multiple independent sub-populations.

    Parameters
    ----------
    integrator : callable, optional
        Integrator factory. Selected automatically when ``None``.
    celsius : float, optional
        Shared simulation temperature.
    **populations
        Mapping of population names to instances.
    """

    def __init__(self, integrator=None, celsius=37.0, **populations):
        if any(b.is_batched() for b in populations.values()):
            raise ValueError("All populations must be unbatched.")
        if integrator is None:
            integrator = assess_type_and_make_integrator(populations)
        C = sum(math.prod(pop.shape) for pop in populations.values())
        super().__init__(1, C, integrator=integrator, celsius=celsius)

        self.populations = populations

        for pop in self.populations.values():
            self._equilibria.update(pop._equilibria)
            self._concentrations.update(pop._concentrations)

        delattr(self, "v_init")
        v_init = torch.cat(
            [
                torch.full(pop.shape, pop.v_init).flatten()
                for pop in self.populations.values()
            ],
            dim=0,
        ).unsqueeze(0)
        self.register_buffer("v_init", v_init)

        self.reinsert_all()
        self.register_labels()

    def __iter__(self):
        return iter(self.populations.values())

    def __len__(self):
        return len(self.populations)

    def device(self):
        return next(iter(self.populations.values())).device()

    def dtype(self):
        return next(iter(self.populations.values())).dtype()

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

    def reinsert_all(self):
        """Recreate mechanisms for all component populations."""
        all_indices = indices(self.populations)
        for index, (name, pop) in zip(all_indices, self.populations.items()):
            # first do _mech_everywhere
            for m_class, (_, _, kwargs) in pop._mech_everywhere.items():
                alias = name
                index = index.flatten()
                self[:, index].insert(m_class, alias=alias, **kwargs)
            # now do _mech_data
            for m_class, list_of_aliases_kwargs_keys in pop._mech_data.items():
                for alias, kwargs, key in list_of_aliases_kwargs_keys:
                    index_f = key_to_flat_index(index, key)
                    alias = f"{name}_{alias}"
                    self[:, index_f].insert(m_class, alias=alias, **kwargs)

    def batch(self, batch_size: int):
        """Create a batched view of the multi-population."""
        super().batch(batch_size)
        self.v_init.unsqueeze(0)
        for pop in self.populations.values():
            pop.batch(batch_size)
        return self

    def batch_(self, batch_size: int):
        """In-place variant of :meth:`batch`."""
        self.batch(batch_size)
