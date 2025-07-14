from typing import List, Optional, Union, NamedTuple, Dict
import re
import math

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

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
    Base class for tree-like structures.

    This class serves as a foundation for creating tree structures that can
    represent branching axons or dendrites in neural models. It inherits from
    the Population class, allowing it to utilize population-level features.
    """
    
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

        self.register_buffer('directions', torch.tensor([[0.0, 0.0, 1.0]], dtype=self.dtype(), device=self.device()).expand(N, -1))
        self.register_buffer('azimuthal_rotations', torch.tensor(0.0, dtype=self.dtype(), device=self.device()).expand(N))

        self.register_buffer('base_direction', torch.tensor([[0.0, 0.0, 1.0]], dtype=self.dtype(), device=self.device()))
        self.register_buffer('base_azimuthal_rotation', torch.tensor(0.0, dtype=self.dtype(), device=self.device()))

        self[:, self.find_not('branchpoint')].label('internal_nodes')

    @property
    def graph(self):
        """
        Returns the graph structure of the tree.

        Returns
        -------
        networkx.DiGraphs
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
    def from_NEURON(cls, root_sec=None, N=1, integrator=None, **kwargs):
        """
        Create a Tree instance from a NEURON root section.

        Parameters
        ----------
        root_sec : h.Section
            The root section of the NEURON model.
        N : int, optional
            Number of instances of the tree. Default is 1.
        integrator : Integrator, optional
            The integrator to use for the model. Defaults to None.

        Returns
        -------
        Tree
            An instance of the Tree class.
        """
        from axonml.models.io import neuron_to_axonml_graph
        graph, _ = neuron_to_axonml_graph(root_sec)
        cell = cls.from_graph(graph, N, integrator, **kwargs)
        cell[:, cell.find('soma')].label('soma')
        cell[:, cell.find('axon')].label('axon')
        cell[:, cell.find('dend')].label('dend')
        cell[:, cell.find('apic')].label('apic')
        return cell

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

    from_asc = from_neurolucida

    def recentre(self, x=0.0, y=0.0, z=0.0, origin=None):
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
        x = torch.as_tensor(x, dtype=self.x.dtype, device=self.x.device).reshape(-1, 1)
        y = torch.as_tensor(y, dtype=self.y.dtype, device=self.y.device).reshape(-1, 1)
        z = torch.as_tensor(z, dtype=self.z.dtype, device=self.z.device).reshape(-1, 1)

        if origin is None:
            origin = self.find('soma', as_list=True)
            origin = origin[int(len(origin) / 2)]

        current_centre_x = self.x[:, origin]
        current_centre_y = self.y[:, origin]
        current_centre_z = self.z[:, origin]

        offsets = [
            x - current_centre_x,
            y - current_centre_y,
            z - current_centre_z
        ]
        
        self.x += offsets[0]
        self.y += offsets[1]
        self.z += offsets[2]

        return self

    def shift(self, dx=0.0, dy=0.0, dz=0.0):
        """
        Shifts the tree structure by specified offsets.

        Parameters
        ----------
        dx : float, optional
            Offset in the x-direction. Default is 0.0.
        dy : float, optional
            Offset in the y-direction. Default is 0.0.
        dz : float, optional
            Offset in the z-direction. Default is 0.0.
        """

        dx = torch.as_tensor(dx, dtype=self.x.dtype, device=self.x.device).reshape(-1, 1)
        dy = torch.as_tensor(dy, dtype=self.y.dtype, device=self.y.device).reshape(-1, 1)
        dz = torch.as_tensor(dz, dtype=self.z.dtype, device=self.z.device).reshape(-1, 1)

        self.x += dx
        self.y += dy
        self.z += dz

        return self

    def move_to(self, x=0.0, y=0.0, z=0.0, origin=None):
        """
        Moves the tree structure to a new position.

        Parameters
        ----------
        x : float, optional
            New x-coordinate. Default is 0.0.
        y : float, optional
            New y-coordinate. Default is 0.0.
        z : float, optional
            New z-coordinate. Default is 0.0.
        """
        return self.recentre(x, y, z, origin)

    def _get_points_as_tensor(self) -> torch.Tensor:
        """Helper to stack x, y, z into a (B, N, 3) tensor."""
        return torch.stack([self.x, self.y, self.z], dim=2)

    def _update_points_from_tensor(self, points: torch.Tensor):
        """Helper to un-stack a (B, N, 3) tensor back into x, y, z buffers."""
        self.x.copy_(points[:, :, 0])
        self.y.copy_(points[:, :, 1])
        self.z.copy_(points[:, :, 2])

    def _apply_rotation(self, rotation_matrices: torch.Tensor, origin_idx: int):
        """
        Applies a batch of rotation matrices to the cell compartments.

        Args:
            rotation_matrices (torch.Tensor): A (B, 3, 3) tensor of rotation matrices.
            origin_idx (int): The index of the compartment to use as the rotation origin.
        """
        points = self._get_points_as_tensor()
        
        # 1. Get the origin for each cell in the batch
        # Shape: (B, 3) -> unsqueeze to (B, 1, 3) for broadcasting
        origins = points[:, origin_idx, :].clone().unsqueeze(1)

        # 2. Translate points so the origin is at (0,0,0)
        points_centered = points - origins

        # 3. Apply the batch of rotations
        # (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
        # We need to transpose the rotation matrices for matmul with (B,N,3)
        rotated_points_centered = points_centered @ rotation_matrices.transpose(1, 2)
        
        # 4. Translate points back
        rotated_points = rotated_points_centered + origins
        
        # 5. Update the internal buffers
        self._update_points_from_tensor(rotated_points)

    def rotate_into_direction(self, target_directions: torch.Tensor, origin: int=None):
        """
        Rotates cells to align their current directions with target directions.
    
        Args:
            target_directions (torch.Tensor): A (B, 3) or (1, 3) tensor of target directions.
            origin_idx (int): The index of the compartment to use as the rotation origin.
        """
        target_directions = torch.as_tensor(
            target_directions, dtype=self.directions.dtype, device=self.directions.device
        ).reshape(-1, 3)

        if origin is None:
            origin = self.find('soma', as_list=True)
            origin = origin[int(len(origin) / 2)]

        device = self.directions.device
        
        if target_directions.shape[0] == 1:
            target_directions = target_directions.repeat(self.np, 1)
        target_directions = target_directions.to(device)
    
        a = F.normalize(self.directions, p=2, dim=1)
        b = F.normalize(target_directions, p=2, dim=1)
    
        # --- Use Rodrigue's formula to get the rotation matrix R ---
        # c is the cosine of the angle (dot product), shape (B,)
        c = torch.sum(a * b, dim=1)
        
        # Mask for when vectors are already aligned (identity rotation)
        is_identity = c > 1.0 - 1e-6
        # Mask for when vectors are anti-parallel (180-degree rotation)
        is_anti_parallel = c < -1.0 + 1e-6
        
        # v is the axis of rotation (cross product), shape (B, 3)
        v = torch.cross(a, b, dim=1)
        
        # Handle the anti-parallel case where the cross product is near zero
        if torch.any(is_anti_parallel):
            # Find an arbitrary perpendicular axis for the 180-degree rotation
            temp_vec = torch.tensor([1.0, 0.0, 0.0], device=device).expand(self.np, -1)
            parallel_to_temp = torch.all(torch.isclose(a, temp_vec) | torch.isclose(a, -temp_vec), dim=1)
            temp_vec[parallel_to_temp] = torch.tensor([0.0, 1.0, 0.0], device=device)
            
            v[is_anti_parallel] = F.normalize(
                torch.cross(a[is_anti_parallel], temp_vec[is_anti_parallel], dim=1), 
                dim=1
            )
    
        # s is the sine of the angle. Clamp to prevent sqrt of negative due to float errors.
        s = torch.sqrt(torch.clamp(1 - c*c, min=0.0))
    
        # Skew-symmetric cross-product matrix K
        K = torch.zeros(self.np, 3, 3, device=device)
        K[:, 0, 1] = -v[:, 2]
        K[:, 0, 2] = v[:, 1]
        K[:, 1, 0] = v[:, 2]
        K[:, 1, 2] = -v[:, 0]
        K[:, 2, 0] = -v[:, 1]
        K[:, 2, 1] = v[:, 0]
        
        # --- Now, reshape for the main formula ---
        # Using new names for clarity
        s_mat = s.view(self.np, 1, 1)
        c_mat = c.view(self.np, 1, 1)

        I = torch.eye(3, device=device).expand(self.np, -1, -1)
        R = I + s_mat * K + (1 - c_mat) * (K @ K)
        
        # --- Apply special cases using the (B,) shaped masks ---
        # This is now correct because `is_identity` has shape (B,)
        R[is_identity] = torch.eye(3, device=device)
        
        # This was already correct, but the logic is now more robust
        if torch.any(is_anti_parallel):
            v_ap = v[is_anti_parallel]
            # Formula for 180-degree rotation matrix around axis v
            R_ap = 2 * torch.einsum('bi,bj->bij', v_ap, v_ap) - torch.eye(3, device=device)
            R[is_anti_parallel] = R_ap
        
        self._apply_rotation(R, origin)
        
        # Update the cell's direction vector
        # We use b, the normalized target, for consistency
        self.directions.copy_(b)
        self.azimuthal_rotations.fill_(0.0)
        return self

    def rotate_azimuthal(self, azimuthal_angle: float | torch.Tensor, origin: int=None):
        """
        Rotates cells around their current direction vector by a given angle.

        Args:
            azimuthal_angle (float or torch.Tensor): Angle in degrees. Can be a single
                                                     float or a (B,) tensor for individual angles.
            origin_idx (int): The index of the compartment to use as the rotation origin.
        """
        if origin is None:
            origin = self.find('soma', as_list=True)
            origin = origin[int(len(origin) / 2)]

        device = self.directions.device

        # Axis of rotation is the cell's own direction
        v = F.normalize(self.directions, p=2, dim=1)

        # Convert angle to radians and ensure it's a (B,) tensor
        if isinstance(azimuthal_angle, (int, float)):
            theta = torch.full((self.np,), float(azimuthal_angle), device=device)
        else:
            theta = torch.as_tensor(azimuthal_angle).to(device).reshape(self.np)
        theta_rad = torch.deg2rad(theta)

        c = torch.cos(theta_rad)
        s = torch.sin(theta_rad)
        
        # Skew-symmetric cross-product matrix K
        K = torch.zeros(self.np, 3, 3, device=device)
        K[:, 0, 1] = -v[:, 2]
        K[:, 0, 2] = v[:, 1]
        K[:, 1, 0] = v[:, 2]
        K[:, 1, 2] = -v[:, 0]
        K[:, 2, 0] = -v[:, 1]
        K[:, 2, 1] = v[:, 0]

        s = s.view(self.np, 1, 1)
        c = c.view(self.np, 1, 1)

        I = torch.eye(3, device=device).expand(self.np, -1, -1)
        R = I + s * K + (1 - c) * (K @ K)
        
        self._apply_rotation(R, origin)
        # Note: self.directions does NOT change in an azimuthal rotation
        self.azimuthal_rotations.add_(theta)
        self.azimuthal_rotations.copy_((self.azimuthal_rotations + 180.0) % 360.0 - 180.0)
        return self

    def reset_rotations(self, origin=None):
        if origin is None:
            origin = self.find('soma', as_list=True)
            origin = origin[int(len(origin) / 2)]
        x_c = self.x[:, origin]
        y_c = self.y[:, origin]
        z_c = self.z[:, origin]

        # Reset directions to the base direction
        self.directions.copy_(self.base_direction.expand(self.np, -1))
        # Reset azimuthal rotations to the base azimuthal rotation
        self.azimuthal_rotations.copy_(self.base_azimuthal_rotation.expand(self.np))

        morph = gather_morphology(self.graph)
        self.x.copy_(morph['x'].expand(self.np, -1).to(dtype=self.x.dtype, device=self.x.device) + x_c.unsqueeze(1))
        self.y.copy_(morph['y'].expand(self.np, -1).to(dtype=self.y.dtype, device=self.y.device) + y_c.unsqueeze(1))
        self.z.copy_(morph['z'].expand(self.np, -1).to(dtype=self.z.dtype, device=self.z.device) + z_c.unsqueeze(1))

        return self

    def reset_directions(self, origin: int=None):
        self.rotate_into_direction(self.base_direction, origin)
        return self

    def reset_azimuthal_rotations(self, origin: int=None):
        angles_to_undo = -self.azimuthal_rotations.clone()
        self.rotate_azimuthal(angles_to_undo, origin)
        self.azimuthal_rotations.copy_(self.base_azimuthal_rotation.expand(self.np))
        return self

    def find_not(self, exclude=None, fuzzy=True, match_case=False):
        indices = find_indices_smart(
            self.names,
            exclude=exclude,
            fuzzy=fuzzy,
            match_case=match_case,
            device=self.device()
        )
        return indices.indices

    def find(
        self, 
        include=None, 
        exclude='branchpoint', 
        fuzzy=True, 
        match_case=False, 
        full_report=False,
        as_list=False
    ):
        indices = find_indices_smart(
            self.names,
            include=include,
            exclude=exclude,
            fuzzy=fuzzy,
            match_case=match_case,
            device=self.device()
        )
        if full_report:
            return indices
        else:
            # Return only the indices of the matches
            if isinstance(indices.indices, slice):
                if as_list:
                    return list(range(indices.indices.start, indices.indices.stop, indices.indices.step or 1))
                return indices.indices
            else:
                return indices.indices.tolist()
        return indices

    def terminal_indices(self):
        """
        Returns the indices of terminal nodes in the tree.

        A terminal node is defined as a node that has no children in the graph.
        """
        terminal_mask = torch.tensor([len(list(self.graph.successors(i))) == 0 for i in range(len(self.graph.nodes))], device=self.device())
        return torch.nonzero(terminal_mask, as_tuple=False).squeeze(1).tolist()

    def slice(self, include=None, exclude='branchpoint', fuzzy=True, match_case=False):
        """
        Finds all indices in the tree structure based on inclusion and exclusion criteria.

        Parameters
        ----------
        include : str or list of str, optional
            Patterns to include in the search.
        exclude : str or list of str, optional
            Patterns to exclude from the search.
        fuzzy : bool, optional
            If True, performs fuzzy matching. Default is True.
        match_case : bool, optional
            If True, matches case sensitively. Default is False.

        Returns
        -------
        List[int]
            A list of indices that match the criteria.
        """
        return self[:, self.find(include=include, exclude=exclude, fuzzy=fuzzy, match_case=match_case, full_report=False)]


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
                regex_pattern = fr"\b{re.escape(pattern)}(?![a-zA-Z0-9])"
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
