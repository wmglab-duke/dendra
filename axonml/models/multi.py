import math

import torch

from .core import Population
from .integrators import dhs_multi
from .tree import Tree


def concat(threads=16, **kwargs):
    return MultiPopulation(integrator=dhs_multi(threads=threads), **kwargs)


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
    def __init__(self, integrator=None, **kwargs):
        if integrator is None:
            integrator = dhs_multi()
        C = sum(math.prod(pop.shape) for pop in kwargs.values())
        super().__init__(1, C, integrator=integrator)

        self.populations = kwargs
        for name, pop in self.populations.items():
            if not isinstance(pop, Tree):
                raise TypeError(f"Expected Tree instance for '{name}', got {type(pop)}")
            setattr(self, name, pop)

        for pop in self.populations.values():
            self._equilibria.update(pop._equilibria)
            self._concentrations.update(pop._concentrations)

        self.reinsert_all()

    def __iter__(self):
        return iter(self.populations.values())

    def __len__(self):
        return len(self.populations)

    def device(self):
        return next(iter(self.populations.values())).device()

    def dtype(self):
        return next(iter(self.populations.values())).dtype()

    def reinsert_all(self):
        all_indices = indices(self.populations)
        for index, (name, pop) in zip(all_indices, self.populations.items()):
            # first do _mech_everywhere
            for m_class, (_, _, kwargs) in pop._mech_everywhere.items():
                alias = name
                index = index.flatten()
                self[:, index].insert(m_class, alias=alias, **kwargs)
            # now do _mech_data
            for m_class, data in pop._mech_data.items():
                alias, kwargs, key = data
                index_f = key_to_flat_index(index, key)
                alias = f"{name}_{alias}"
                self[:, index_f].insert(m_class, alias=alias, **kwargs)
