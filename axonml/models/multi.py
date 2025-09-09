import math

import torch

from .core import Population
from .integrators import bwd_euler_sc_multi, dhs_multi
from .tree import Tree


def assess_type_and_make_integrator(populations, threads=16, write_back=True):
    if all(isinstance(pop, Tree) for pop in populations.values()):
        return dhs_multi(threads=threads, write_back=write_back)
    if all(type(pop) is Population for pop in populations.values()):
        return bwd_euler_sc_multi(write_back=write_back)
    raise TypeError("Incompatible population types.")


def concat(
    populations: dict[str, Population], threads=16, write_back=True, celsius=37.0
):
    integrator = assess_type_and_make_integrator(
        populations, threads=threads, write_back=write_back
    )
    return MultiPopulation(
        integrator=integrator,
        celsius=celsius,
        **populations,
    )


def offsets(populations):
    sizes = [math.prod(p.shape) for p in populations.values()]
    off = [0] + list(torch.cumsum(torch.tensor(sizes), dim=0).numpy().astype(int))[:-1]
    return off


def indices(populations):
    off = offsets(populations)
    sizes = [math.prod(p.shape) for p in populations.values()]
    indices = [
        torch.arange(s).reshape(p.shape) + off[idx]
        for idx, (s, p) in enumerate(zip(sizes, populations.values()))
    ]
    return indices


def key_to_flat_index(indices, key):
    return indices[key].flatten()


class MultiPopulation(Population):
    def __init__(self, integrator=None, celsius=37.0, **populations):
        if any(b.is_batched() for b in populations.values()):
            raise ValueError("All populations must be unbatched.")
        if integrator is None:
            integrator = assess_type_and_make_integrator(populations)
        C = sum(math.prod(pop.shape) for pop in populations.values())
        super().__init__(1, C, integrator=integrator, celsius=celsius)

        self.populations = populations
        for name, pop in self.populations.items():
            setattr(self, name, pop)

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
        all_indices = indices(self.populations)
        for index, (name, pop) in zip(all_indices, self.populations.items()):
            for label, slice in pop._labels.items():
                index_f = key_to_flat_index(index, slice.index)
                self[:, index_f].label(f"{name}_{label}")

    def reinsert_all(self):
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
        super().batch(batch_size)
        self.v_init.unsqueeze(0)
        for pop in self.populations.values():
            pop.batch(batch_size)
        return self

    def batch_(self, batch_size: int):
        self.batch(batch_size)
