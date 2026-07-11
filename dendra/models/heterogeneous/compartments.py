"""Compartment-label helpers for repeated axon model definitions."""

import re
from typing import List

import numpy as np

__all__ = ["CompartmentID"]


class CompartmentID:
    """Describe the repeating compartment labels in an axon model.

    ``CompartmentID`` is a model-definition helper for axons assembled from a
    repeated sequence of named regions. Entries such as ``"internode * 3"``
    are expanded before the sequence is repeated. The resulting
    :attr:`names` array can be registered with
    :meth:`dendra.models.core.Axon.register_cid` to enable name-based slicing
    and install one labelled :class:`~dendra.models.slice.Slice` per unique
    compartment name.

    Parameters
    ----------
    names : list of str
        One repeat of the compartment-name pattern. A value of the form
        ``"name * count"`` contributes ``count`` adjacent copies of ``name``.
    n_repeats : int
        Number of times to repeat the expanded pattern.
    wrap : bool, default=True
        Append the first entry in ``names`` after the final repeat. This is
        useful for models whose repeated units share a boundary node.

    Attributes
    ----------
    names : numpy.ndarray
        Expanded compartment labels. Its length should equal ``axon.n_comp``
        before the table is passed to :meth:`~dendra.models.core.Axon.register_cid`.

    Examples
    --------
    A two-unit model with a shared node at the distal boundary has seven
    compartments:

    >>> cid = CompartmentID(["node", "internode * 2"], n_repeats=2)
    >>> cid.names.tolist()
    ['node', 'internode', 'internode', 'node', 'internode', 'internode', 'node']
    >>> cid.loc("node")
    [0, 3, 6]
    """

    def __init__(self, names: List[str], n_repeats: int, wrap=True):
        self._names = names
        self.n_repeats = n_repeats
        expanded = expand_string_list(names)
        if wrap:
            self.names = np.array(expanded * n_repeats + expanded[:1])
        else:
            self.names = np.array(expanded * n_repeats)

    def nc(self):
        """Return the number of expanded compartment labels."""
        return len(self.names)

    def __getitem__(self, i):
        return self.names[i]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def unique(self):
        """Return the sorted unique compartment names."""
        return np.unique(self.names).tolist()

    def loc(self, name):
        """Return all compartment indices carrying ``name``."""
        return np.where(self.names == name)[0].tolist()

    def locs(self, names):
        """Return indices carrying any name in ``names``."""
        return np.where(np.isin(self.names, names))[0].tolist()

    def build(self, funcs, model):
        """Build a per-axon, per-compartment NumPy parameter array.

        Parameters
        ----------
        funcs : Mapping[str, Callable]
            Mapping from every unique compartment name to a callable. Each
            callable receives ``model`` and returns one value per axon.
        model : Axon
            Model exposing ``n_ax`` and ``n_comp``. Its compartment count must
            match this table.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(model.n_ax, model.n_comp)``. Values returned for
            each name are copied into every compartment carrying that label.
        """
        assert model.n_comp == self.nc()
        out = np.empty((model.n_ax, self.nc()))

        s = self.unique()

        for s_ in s:
            d_ = funcs[s_](model)
            out[:, self.names == s_] = d_[:, None]

        return out


def expand_string_list(strings):
    """Expand ``"text * count"`` entries in a list of strings.

    Parameters
    ----------
    strings : iterable of str
        Compartment-name patterns to expand.

    Returns
    -------
    list of str
        Expanded names in input order. Entries without a repeat suffix are
        returned unchanged.
    """
    pattern = re.compile(r"^(.*?)\s*\*\s*(\d+)$")
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
