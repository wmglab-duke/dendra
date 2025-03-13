from typing import Callable, List
import re

import torch
import numpy as np


class CompartmentID:
    def __init__(self, names: List[str], n_repeats: int):
        self._names = names
        self.n_repeats = n_repeats
        self.names = np.array(expand_string_list(names) * n_repeats + names[:1])

    def nc(self):
        return len(self.names)
    
    def __getitem__(self, i):
        return self.names[i]
    
    def __len__(self):
        return len(self.names)
    
    def __iter__(self):
        return iter(self.names)
    
    def unique(self):
        return np.unique(self.names).tolist()
    
    def loc(self, name):
        return np.where(self.names == name)[0].tolist()
    
    def locs(self, names):
        return np.where(np.isin(self.names, names))[0].tolist()
    
    def build(self, funcs, model):
        assert model.n_comp == self.nc()
        out = np.empty((model.n_ax, self.nc()))

        s = self.unique()

        for s_ in s:
            d_ = funcs[s_](model)
            out[:, self.names==s_] = d_[:, None]
                    
        return out


class CompartmentIDTorch:
    def __init__(self, names: List[str], n_repeats: int):
        self._names = names
        self.n_repeats = n_repeats
        self.names = torch.tensor(expand_string_list(names) * n_repeats + names[:1])

    def nc(self):
        return len(self.names)
    
    def __getitem__(self, i):
        return self.names[i]
    
    def __len__(self):
        return len(self.names)
    
    def unique(self):
        return self.names.unique().tolist()
    
    def loc(self, name):
        return torch.where(self.names == name)[0].tolist()
    
    def locs(self, names):
        return torch.where(self.names.unsqueeze(0) == torch.tensor(names).unsqueeze(1))[1].tolist()
    
    def build(self, funcs, model):
        assert model.n_comp == self.nc()
        out = torch.empty((model.n_ax, self.nc()))  # type: ignore

        s = self.unique()

        for s_ in s:
            d_ = funcs[s_](model)
            out[:, self.names==s_] = d_[:, None]

        return out


def expand_string_list(strings):
    """
    Expands any element of the form "text * number" into
    'number' copies of "text". Other elements remain unchanged.
    
    Parameters:
        strings (list of str): The input list of strings.
    
    Returns:
        list of str: The expanded list.
    """
    pattern = re.compile(r'^(.*?)\s*\*\s*(\d+)$')
    expanded = []
    
    for s in strings:
        match = pattern.match(s)
        if match:
            text, count_str = match.groups()
            count = int(count_str)
            expanded.extend([text] * count)
        else:
            expanded.append(s)
    
    return expanded
