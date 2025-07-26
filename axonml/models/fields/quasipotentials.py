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
    """
    Calculates extracellular quasipotentials (ψ) for a BATCH of E-fields and
    a BATCH of corresponding coordinates using PyTorch.

    This version is optimized for scenarios where both the E-field and the
    neuron's spatial coordinates vary per batch instance, while the underlying
    graph topology remains constant. The calculations are performed on the
    device of the input tensors (e.g., 'cpu' or 'cuda').

    Args:
        G (nx.DiGraph): A single directed graph representing the constant neuron
                        morphology. Edges must point from parent to child. The
                        node IDs in the graph must correspond to the indices
                        in the last dimension of the input tensors.
        x_batch (torch.Tensor): Tensor of shape (B, N) for x-coordinates.
                                Assumed to be in **microns (µm)**.
        y_batch (torch.Tensor): Tensor of shape (B, N) for y-coordinates.
                                Assumed to be in **microns (µm)**.
        z_batch (torch.Tensor): Tensor of shape (B, N) for z-coordinates.
                                Assumed to be in **microns (µm)**.
        e_fields_batch (torch.Tensor): Tensor of shape (B, N, 3) for E-fields.
                                       Assumed to be in **Volts/meter (V/m)**.

    Returns:
        torch.Tensor: A 2D Tensor of shape (B, N) containing the calculated
                      quasipotential `ψ` for each batch instance and node, in
                      **millivolts (mV)**.
    """
    # --- Step 1: Input Validation and Data Preparation ---
    num_nodes = G.number_of_nodes()
    batch_size = e_fields_batch.shape[0]
    device = e_fields_batch.device  # Use the device of the input tensors

    if not (x_batch.shape == y_batch.shape == z_batch.shape == (batch_size, num_nodes)):
        raise ValueError(
            f"Shape mismatch: x, y, z batches must have shape ({batch_size}, {num_nodes})."
        )
    if e_fields_batch.shape != (batch_size, num_nodes, 3):
        raise ValueError(
            f"Shape mismatch: e_fields_batch must have shape ({batch_size}, {num_nodes}, 3)."
        )

    # Stack coordinates into a single (B, N, 3) tensor for easier indexing.
    coords_batch = torch.stack([x_batch, y_batch, z_batch], dim=2)

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
        dtype=coords_batch.dtype,  # Use the same dtype as coordinates
        device=device,
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

    return psi_batch
