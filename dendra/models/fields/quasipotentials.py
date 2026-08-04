from collections import deque
from typing import List

import networkx as nx
import torch


def calculate_quasipotentials_batched_coords(
    G: nx.DiGraph,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    z_batch: torch.Tensor,
    e_fields_batch: torch.Tensor,
) -> torch.Tensor:
    """Integrate E-fields for coordinates with arbitrary leading batch axes.

    The final coordinate axis is always the morphology/node axis. All axes
    before it are logical population or batch axes and are preserved exactly::

        coordinates:  (*leading, C)
        E-fields:      (*leading, C, 3)
        result:        (*leading, C)

    Internally the leading axes are flattened, the common morphology graph is
    traversed once, and the result is restored to the input coordinate shape.
    This makes rank-2, rank-3, and higher-rank model coordinates equivalent to
    evaluating each leading-axis member independently.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph for the constant neuron morphology with edges from
        parent to child; node IDs must match tensor indices.
    x_batch, y_batch, z_batch : torch.Tensor
        Coordinates of identical shape ``(*leading, C)`` in microns (µm),
        where ``C`` equals ``G.number_of_nodes()``.
    e_fields_batch : torch.Tensor
        E-fields of shape ``(*leading, C, 3)`` in Volts/meter (V/m).

    Returns
    -------
    torch.Tensor
        Quasipotentials in millivolts with the same shape as the coordinate
        tensors. Device, differentiability, and the common promoted floating
        dtype of the coordinates and E-field are preserved.
    """
    # --- Step 1: Input Validation and Data Preparation ---
    num_nodes = G.number_of_nodes()
    if num_nodes < 1:
        raise ValueError("Graph must contain at least one morphology node.")
    coord_shape = x_batch.shape

    if x_batch.ndim < 1:
        raise ValueError("Coordinate tensors must have shape (*leading, C).")
    if not (x_batch.shape == y_batch.shape == z_batch.shape):
        raise ValueError(
            "Shape mismatch: x, y, and z must have identical shapes; got "
            f"{tuple(x_batch.shape)}, {tuple(y_batch.shape)}, and "
            f"{tuple(z_batch.shape)}."
        )
    if coord_shape[-1] != num_nodes:
        raise ValueError(
            "The final coordinate axis must match the morphology graph: "
            f"got {coord_shape[-1]} compartments but the graph has {num_nodes} nodes."
        )
    expected_efield_shape = (*coord_shape, 3)
    if e_fields_batch.shape != expected_efield_shape:
        raise ValueError(
            "Shape mismatch: e_fields_batch must have shape "
            f"{expected_efield_shape}; got {tuple(e_fields_batch.shape)}."
        )
    tensors = (x_batch, y_batch, z_batch, e_fields_batch)
    if not all(t.device == x_batch.device for t in tensors):
        raise ValueError("Coordinates and E-fields must be on the same device.")
    if not all(torch.is_floating_point(t) for t in tensors):
        raise TypeError("Coordinates and E-fields must be floating-point tensors.")

    leading_shape = coord_shape[:-1]
    batch_size = x_batch.numel() // num_nodes
    work_dtype = x_batch.dtype
    for tensor in (y_batch, z_batch, e_fields_batch):
        work_dtype = torch.promote_types(work_dtype, tensor.dtype)

    # Stack coordinates into a single (B, N, 3) tensor for easier indexing.
    coords_batch = torch.stack([x_batch, y_batch, z_batch], dim=-1)
    coords_batch = coords_batch.reshape(batch_size, num_nodes, 3).to(work_dtype)
    e_fields_batch = e_fields_batch.reshape(batch_size, num_nodes, 3).to(work_dtype)

    # --- Step 2: Unit Conversion ---
    coords_batch_m = coords_batch * 1e-6  # Convert µm to m
    volts_to_millivolts = 1000.0  # Convert V to mV

    # --- Step 3: Find Root and Initialize Data Structures ---
    # The graph traversal logic remains on the CPU
    roots: List[int] = [node for node, in_degree in G.in_degree() if in_degree == 0]
    if not roots:
        raise ValueError("Graph has no root node (a node with in-degree 0).")

    # Initialize psi tensor on the correct device with a high-precision dtype
    psi_batch = torch.full(
        (batch_size, num_nodes),
        torch.nan,
        dtype=work_dtype,
        device=x_batch.device,
    )

    # The queue for the BFS is a standard Python object
    queue = deque()

    for root_id in roots:
        psi_batch[:, root_id] = 0.0
        queue.append(root_id)

    # --- Step 4: Batched BFS Traversal ---
    visited_count = 0
    while queue:
        parent_id = queue.popleft()
        visited_count += 1

        # Get parent data for the entire batch (these are tensor slices)
        psi_p = psi_batch[:, parent_id]  # Shape: (B,)
        pos_p = coords_batch_m[:, parent_id, :]  # Shape: (B, 3)
        E_p = e_fields_batch[:, parent_id, :]  # Shape: (B, 3)

        for child_id in G.successors(parent_id):
            # Get child data for the entire batch
            pos_c = coords_batch_m[:, child_id, :]  # Shape: (B, 3)
            E_c = e_fields_batch[:, child_id, :]  # Shape: (B, 3)

            # --- Step 5: Fully Batched Calculation using PyTorch ---
            # Displacement vector s_pc, shape (B, 3)
            s_pc_batch = pos_c - pos_p

            # Average E-field vector, shape (B, 3)
            E_avg_batch = 0.5 * (E_c + E_p)

            # Batched row-wise dot product.
            # (B, 3) * (B, 3) -> element-wise product, then sum along component dimension.
            dot_product_batch = torch.sum(
                E_avg_batch * s_pc_batch, dim=1
            )  # Shape: (B,)

            # Convert result from Volts to mV
            dot_product_mv = dot_product_batch * volts_to_millivolts

            # Calculate child's potential for the whole batch
            psi_child_batch = psi_p - dot_product_mv

            # Store results and enqueue the child
            psi_batch[:, child_id] = psi_child_batch
            queue.append(child_id)

    if visited_count != num_nodes:
        print(
            f"Warning: Traversal visited {visited_count} nodes, but graph has {num_nodes} nodes."
        )

    return psi_batch.reshape(*leading_shape, num_nodes)
