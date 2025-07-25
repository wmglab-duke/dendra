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
        L.append(attrs.get("L"))
        diam.append(attrs.get("diam"))
        rhoa.append(attrs.get("Ra"))
        cm.append(attrs.get("cm"))
        x.append(attrs.get("x", 0.0))
        y.append(attrs.get("y", 0.0))
        z.append(attrs.get("z", 0.0))
    return {
        "dx": torch.tensor(L, dtype=torch.float32).unsqueeze(0),
        "diam": torch.tensor(diam, dtype=torch.float32).unsqueeze(0),
        "rhoa": torch.tensor(rhoa, dtype=torch.float32).unsqueeze(0),
        "cm": torch.tensor(cm, dtype=torch.float32).unsqueeze(0),
        "x": torch.tensor(x, dtype=torch.float32).unsqueeze(0),
        "y": torch.tensor(y, dtype=torch.float32).unsqueeze(0),
        "z": torch.tensor(z, dtype=torch.float32).unsqueeze(0),
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
            name = attrs.get("name")
            if "branchpoint" in name:
                name = name.replace("_", ".")
            names.append(name)
        self.names = names

        self.register_buffer(
            "directions",
            torch.tensor(
                [[0.0, 0.0, 1.0]], dtype=self.dtype(), device=self.device()
            ).expand(N, -1),
        )
        self.register_buffer(
            "azimuthal_rotations",
            torch.tensor(0.0, dtype=self.dtype(), device=self.device()).expand(N),
        )

        self.register_buffer(
            "base_direction",
            torch.tensor([[0.0, 0.0, 1.0]], dtype=self.dtype(), device=self.device()),
        )
        self.register_buffer(
            "base_azimuthal_rotation",
            torch.tensor(0.0, dtype=self.dtype(), device=self.device()),
        )

        self[:, self.find_not("branchpoint")].label("internal_nodes")

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
        tree.set_value("cm", data["cm"])
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
        cell[:, cell.find("soma")].label("soma")
        cell[:, cell.find("axon")].label("axon")
        cell[:, cell.find("dend")].label("dend")
        cell[:, cell.find("apic")].label("apic")
        return cell

    @classmethod
    def from_swc(
        cls, file_path, d_lambda=0.1, freq=100.0, N=1, integrator=None, **kwargs
    ):
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
        cell[:, cell.find("soma")].label("soma")
        cell[:, cell.find("axon")].label("axon")
        cell[:, cell.find("dend")].label("dend")
        cell[:, cell.find("apic")].label("apic")
        return cell

    @classmethod
    def from_neurolucida(
        cls, file_path, d_lambda=0.1, freq=100.0, N=1, integrator=None, **kwargs
    ):
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
        cell[:, cell.find("soma")].label("soma")
        cell[:, cell.find("axon")].label("axon")
        cell[:, cell.find("dend")].label("dend")
        cell[:, cell.find("apic")].label("apic")
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
            origin = self.find("soma", as_list=True)
            origin = origin[int(len(origin) / 2)]

        current_centre_x = self.x[:, origin][:, None]
        current_centre_y = self.y[:, origin][:, None]
        current_centre_z = self.z[:, origin][:, None]

        offsets = [x - current_centre_x, y - current_centre_y, z - current_centre_z]

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

        dx = torch.as_tensor(dx, dtype=self.x.dtype, device=self.x.device).reshape(
            -1, 1
        )
        dy = torch.as_tensor(dy, dtype=self.y.dtype, device=self.y.device).reshape(
            -1, 1
        )
        dz = torch.as_tensor(dz, dtype=self.z.dtype, device=self.z.device).reshape(
            -1, 1
        )

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

    def rotate_into_direction(
        self, target_directions: torch.Tensor, origin: int = None
    ):
        """
        Rotates cells to align their current directions with target directions.

        Args:
            target_directions (torch.Tensor): A (B, 3) or (1, 3) tensor of target directions.
            origin_idx (int): The index of the compartment to use as the rotation origin.
        """
        target_directions = torch.as_tensor(
            target_directions,
            dtype=self.directions.dtype,
            device=self.directions.device,
        ).reshape(-1, 3)

        if origin is None:
            origin = self.find("soma", as_list=True)
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
            parallel_to_temp = torch.all(
                torch.isclose(a, temp_vec) | torch.isclose(a, -temp_vec), dim=1
            )
            temp_vec[parallel_to_temp] = torch.tensor([0.0, 1.0, 0.0], device=device)

            v[is_anti_parallel] = F.normalize(
                torch.cross(a[is_anti_parallel], temp_vec[is_anti_parallel], dim=1),
                dim=1,
            )

        # s is the sine of the angle. Clamp to prevent sqrt of negative due to float errors.
        s = torch.sqrt(torch.clamp(1 - c * c, min=0.0))

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
            R_ap = 2 * torch.einsum("bi,bj->bij", v_ap, v_ap) - torch.eye(
                3, device=device
            )
            R[is_anti_parallel] = R_ap

        self._apply_rotation(R, origin)

        # Update the cell's direction vector
        # We use b, the normalized target, for consistency
        self.directions.copy_(b)
        self.azimuthal_rotations.fill_(0.0)
        return self

    def rotate_azimuthal(
        self, azimuthal_angle: float | torch.Tensor, origin: int = None
    ):
        """
        Rotates cells around their current direction vector by a given angle.

        Args:
            azimuthal_angle (float or torch.Tensor): Angle in degrees. Can be a single
                                                     float or a (B,) tensor for individual angles.
            origin_idx (int): The index of the compartment to use as the rotation origin.
        """
        if origin is None:
            origin = self.find("soma", as_list=True)
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
        self.azimuthal_rotations.copy_(
            (self.azimuthal_rotations + 180.0) % 360.0 - 180.0
        )
        return self

    def reset_rotations(self, origin=None):
        if origin is None:
            origin = self.find("soma", as_list=True)
            origin = origin[int(len(origin) / 2)]
        x_c = self.x[:, origin]
        y_c = self.y[:, origin]
        z_c = self.z[:, origin]

        # Reset directions to the base direction
        self.directions.copy_(self.base_direction.expand(self.np, -1))
        # Reset azimuthal rotations to the base azimuthal rotation
        self.azimuthal_rotations.copy_(self.base_azimuthal_rotation.expand(self.np))

        morph = gather_morphology(self.graph)
        self.x.copy_(
            morph["x"].expand(self.np, -1).to(dtype=self.x.dtype, device=self.x.device)
            + x_c.unsqueeze(1)
        )
        self.y.copy_(
            morph["y"].expand(self.np, -1).to(dtype=self.y.dtype, device=self.y.device)
            + y_c.unsqueeze(1)
        )
        self.z.copy_(
            morph["z"].expand(self.np, -1).to(dtype=self.z.dtype, device=self.z.device)
            + z_c.unsqueeze(1)
        )

        return self

    def reset_directions(self, origin: int = None):
        self.rotate_into_direction(self.base_direction, origin)
        return self

    def reset_azimuthal_rotations(self, origin: int = None):
        angles_to_undo = -self.azimuthal_rotations.clone()
        self.rotate_azimuthal(angles_to_undo, origin)
        self.azimuthal_rotations.copy_(self.base_azimuthal_rotation.expand(self.np))
        return self
