from typing import List, Optional, Union, NamedTuple, Dict
import re

import torch
import numpy as np
import pandas as pd

from axonml.models.declarations import PARAMETER
from axonml.models.integrators import dhs

from .core import Population


def gather_morphology(graph):
    # iterate through nodes and gather morphology data
    L, diam, rhoa, cm, x, y, z = [], [], [], [], [], [], []

    for i in range(len(graph.nodes)):
        attrs = graph.nodes[i]
        L       .append(attrs.get('L'))
        diam    .append(attrs.get('diam'))
        rhoa    .append(attrs.get('Ra'))
        cm      .append(attrs.get('cm'))
        x       .append(attrs.get('x', 0.0))
        y       .append(attrs.get('y', 0.0))
        z       .append(attrs.get('z', 0.0))
    return {
        'dx':   torch.tensor(L,     dtype=torch.float32).unsqueeze(0),
        'diam': torch.tensor(diam,  dtype=torch.float32).unsqueeze(0),
        'rhoa': torch.tensor(rhoa,  dtype=torch.float32).unsqueeze(0),
        'cm':   torch.tensor(cm,    dtype=torch.float32).unsqueeze(0),
        'x':    torch.tensor(x,     dtype=torch.float32).unsqueeze(0),
        'y':    torch.tensor(y,     dtype=torch.float32).unsqueeze(0),
        'z':    torch.tensor(z,     dtype=torch.float32).unsqueeze(0),
    }


class Tree(Population):
    """
    Base class for tree-like structures in axonal models.

    This class serves as a foundation for creating tree structures that can
    represent branching axons or dendrites in neural models. It inherits from
    the Population class, allowing it to utilize population-level features.

    Parameters
    ----------
    name : str
        Name of the tree structure.
    nodes : int
        Number of nodes in the tree.
    """

    PARAMETER(celsius=37.0)
    
    def __init__(self, N, C, graph, integrator=None, **kwargs):
        if integrator is None:
            integrator = dhs()
        super().__init__(N, C, integrator=integrator, **kwargs)
        self._graph = graph
        names = []
        for i in range(len(graph.nodes)):
            attrs = graph.nodes[i]
            name = attrs.get('name')
            if 'branchpoint' in name:
                name = name.replace('_', '.')
            names.append(name)
        self.names = names

        self[:, self.find_not('branchpoint')].label('internal_nodes')

    @property
    def graph(self):
        """
        Returns the graph structure of the tree.

        Returns
        -------
        networkx.DiGraph
            The directed graph representing the tree structure.
        """
        return self._graph
        
    @classmethod
    def from_graph(cls, graph, N=1, integrator=None, **kwargs):
        """
        Create a Tree instance from a graph structure.

        Parameters
        ----------
        graph : networkx.DiGraph
            A graph representing the tree structure.
        integrator : Integrator, optional
            The integrator to use for the model. Defaults to None.

        Returns
        -------
        Tree
            An instance of the Tree class.
        """
        C = len(graph.nodes)
        data = gather_morphology(graph)
        tree = cls(N, C, graph, integrator, **kwargs)
        for key, value in data.items():
            tree.register_buffer(key, value.expand(N, -1))
        return tree

    @classmethod
    def from_swc(cls, file_path, d_lambda=0.1, freq=100.0, N=1, integrator=None, **kwargs):
        """
        Create a Tree instance from an SWC file.

        Parameters
        ----------
        file_path : str
            Path to the SWC file.
        N : int, optional
            Number of instances of the tree. Default is 1.
        integrator : Integrator, optional
            The integrator to use for the model. Defaults to None.

        Returns
        -------
        Tree
            An instance of the Tree class.
        """
        from axonml.models.io import read_swc
        graph, _ = read_swc(file_path, d_lambda=d_lambda, freq=freq)
        cell = cls.from_graph(graph, N, integrator, **kwargs)
        cell[:, cell.find('soma')].label('soma')
        cell[:, cell.find('axon')].label('axon')
        cell[:, cell.find('dend')].label('dend')
        cell[:, cell.find('apic')].label('apic')
        return cell

    @classmethod
    def from_neurolucida(cls, file_path, d_lambda=0.1, freq=100.0, N=1, integrator=None, **kwargs):
        """
        Create a Tree instance from a Neurolucida file.

        Parameters
        ----------
        file_path : str
            Path to the Neurolucida file.
        N : int, optional
            Number of instances of the tree. Default is 1.
        integrator : Integrator, optional
            The integrator to use for the model. Defaults to None.

        Returns
        -------
        Tree
            An instance of the Tree class.
        """
        from axonml.models.io import read_neurolucida
        graph, _ = read_neurolucida(file_path, d_lambda=d_lambda, freq=freq)
        cell = cls.from_graph(graph, N, integrator, **kwargs)
        cell[:, cell.find('soma')].label('soma')
        cell[:, cell.find('axon')].label('axon')
        cell[:, cell.find('dend')].label('dend')
        cell[:, cell.find('apic')].label('apic')
        return cell

    def recentre(self, x=0.0, y=0.0, z=0.0):
        """
        Recenters the tree structure so soma is at the origin.
        Parameters
        ----------
        x : float, optional
            X-coordinate of the new center. Default is 0.0.
        y : float, optional
            Y-coordinate of the new center. Default is 0.0.
        z : float, optional
            Z-coordinate of the new center. Default is 0.0.
        """ 
        current_centre_x = self.x[:, 0]
        current_centre_y = self.y[:, 0]
        current_centre_z = self.z[:, 0]

        offsets = [
            x - current_centre_x,
            y - current_centre_y,
            z - current_centre_z
        ]
        
        self.x += offsets[0]
        self.y += offsets[1]
        self.z += offsets[2]

    def move(self, dx=0.0, dy=0.0, dz=0.0):
        """
        Moves the tree structure by specified offsets.

        Parameters
        ----------
        dx : float, optional
            Offset in the x-direction. Default is 0.0.
        dy : float, optional
            Offset in the y-direction. Default is 0.0.
        dz : float, optional
            Offset in the z-direction. Default is 0.0.
        """
        self.x += dx
        self.y += dy
        self.z += dz

    def find(self, include=None, fuzzy=True, match_case=False, full_report=False):
        indices = find_indices_smart(
            self.names,
            include=include,
            fuzzy=fuzzy,
            match_case=match_case,
            device=self.device()
        )
        if full_report:
            return indices
        else:
            # Return only the indices of the matches
            if isinstance(indices.indices, slice):
                return indices.indices
            else:
                return indices.indices.tolist()
        return indices

    def find_not(self, exclude=None, fuzzy=True, match_case=False):
        indices = find_indices_smart(
            self.names,
            exclude=exclude,
            fuzzy=fuzzy,
            match_case=match_case,
            device=self.device()
        )
        return indices.indices

    def calculate_geometric_params(self):
        return


# Define the return type for clarity
class FindResult(NamedTuple):
    indices: Union[slice, torch.Tensor]
    local_indices: Dict[str, Union[slice, torch.Tensor]]
    local_sizes: Dict[str, int]
    total_size: int

# Helper function to convert numpy indices to a slice or tensor
def _indices_to_slice_or_tensor(
    numpy_indices: np.ndarray,
    device: Optional[torch.device] = None
) -> Union[slice, torch.Tensor]:
    """Converts a 1D numpy array of indices into a slice if possible, else a tensor."""
    num_indices = len(numpy_indices)

    if num_indices == 0:
        return slice(0, 0, None)

    if num_indices == 1:
        start = int(numpy_indices[0])
        return slice(start, start + 1, None)

    # Check if the step between all indices is constant
    diffs = np.diff(numpy_indices)
    step = int(diffs[0])

    if np.all(diffs == step):
        # The indices form an arithmetic progression. It can be a slice!
        start = int(numpy_indices[0])
        stop = int(numpy_indices[-1]) + step
        return slice(start, stop, step if step != 1 else None)
    else:
        # Indices are not contiguous, fall back to returning a tensor
        torch_indices = torch.from_numpy(numpy_indices)
        return torch_indices.to(device) if device else torch_indices

def find_indices_smart(
    data: List[str],
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    fuzzy: bool = True,
    match_case: bool = False,
    device: Optional[torch.device] = None
) -> FindResult:
    """
    Finds indices based on criteria and returns detailed results including local indices
    for each included pattern.

    Handles complex patterns like 'axon[0]' correctly. If fuzzy=True:
    - A simple pattern like 'axon' will match 'axon', 'axon[0]', but not 'taxons'.
    - A complex pattern like 'axon[0]' will match strings containing the literal 'axon[0]'.

    Args:
        data (List[str]): The list of strings to search through.
        include (Optional[Union[str, List[str]]]): Patterns to include.
        exclude (Optional[Union[str, List[str]]]): Patterns to exclude.
        fuzzy (bool): If True, performs smart whole-word/substring matching. If False, an exact match.
        match_case (bool): If True, the matching is case-sensitive.
        device (Optional[torch.device]): PyTorch device for resulting tensors.

    Returns:
        FindResult: A named tuple with detailed matching results.
    """
    empty_result = FindResult(slice(0, 0), {}, {}, 0)
    if not data:
        return empty_result

    s = pd.Series(data, dtype="string")
    final_mask = pd.Series(True, index=s.index)
    
    local_indices_map = {}
    local_sizes_map = {}
    pattern_masks: Dict[str, pd.Series] = {}

    def get_mask_for_pattern(pattern: str) -> pd.Series:
        """Helper to generate a boolean mask for a given pattern."""
        if fuzzy:
            # If pattern contains non-word chars (e.g., 'axon[0]'), treat as literal substring.
            if re.search(r'[^a-zA-Z0-9_]', pattern):
                regex_pattern = re.escape(pattern)
            # Otherwise, it's a simple name (e.g., 'axon'). Match as a "root" word.
            # Use a negative lookahead to allow suffixes like '[0]' but not more letters.
            else:
                regex_pattern = fr"\b{re.escape(pattern)}(?![a-zA-Z0-9_])"
            return s.str.contains(regex_pattern, case=match_case, regex=True, na=False)
        else: # Exact match
            series_to_compare = s.str.lower() if not match_case else s
            pattern_to_compare = pattern.lower() if not match_case else pattern
            return series_to_compare == pattern_to_compare

    if include:
        include_patterns = [include] if isinstance(include, str) else include
        for pattern in include_patterns:
            pattern_masks[pattern] = get_mask_for_pattern(pattern)
        if pattern_masks:
            combined_include_mask = pd.concat(pattern_masks.values(), axis=1).any(axis=1)
            final_mask &= combined_include_mask
    else:
        include_patterns = []

    if exclude:
        exclude_patterns = [exclude] if isinstance(exclude, str) else exclude
        combined_exclude_mask = pd.Series(False, index=s.index)
        for pattern in exclude_patterns:
            combined_exclude_mask |= get_mask_for_pattern(pattern)
        final_mask &= ~combined_exclude_mask

    numpy_indices = s.index[final_mask].to_numpy()
    total_size = len(numpy_indices)

    if total_size == 0:
        return empty_result
    
    total_indices_result = _indices_to_slice_or_tensor(numpy_indices, device)

    if include_patterns:
        global_to_local_map = {global_idx: local_idx for local_idx, global_idx in enumerate(numpy_indices)}
        
        for pattern in include_patterns:
            pattern_final_mask = pattern_masks[pattern] & final_mask
            pattern_global_indices = s.index[pattern_final_mask].to_numpy()
            
            if len(pattern_global_indices) > 0:
                local_indices_list = [global_to_local_map[g_idx] for g_idx in pattern_global_indices]
                local_numpy_indices = np.array(local_indices_list, dtype=np.int64)
                local_indices_map[pattern] = _indices_to_slice_or_tensor(local_numpy_indices, device)
                local_sizes_map[pattern] = len(local_indices_list)
            else:
                local_indices_map[pattern] = slice(0, 0)
                local_sizes_map[pattern] = 0


    return FindResult(
        indices=total_indices_result,
        local_indices=local_indices_map,
        local_sizes=local_sizes_map,
        total_size=total_size
    )
