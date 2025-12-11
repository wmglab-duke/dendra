# test_quasipotentials.py
from __future__ import annotations

from typing import Tuple

import networkx as nx
import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# -----------------------------------------------------------------------------#
# 1.  SUT import (adjust the import path if you keep the function elsewhere)
# -----------------------------------------------------------------------------#
from axonml.models.fields.quasipotentials import (
    calculate_quasipotentials_batched_coords,
)

# -----------------------------------------------------------------------------#
# 2.  Helpers & Hypothesis strategies
# -----------------------------------------------------------------------------#

FLOAT_DTYPES = (torch.float32, torch.float64)  # both dtypes supported by torch.linalg
COORD_RANGE_UM = 2_000  # ±2 mm in µm
EFIELD_RANGE = 1_000  # ±1 kV / m


def _torch_from_numpy(arr: np.ndarray, dtype: torch.dtype, device: torch.device):
    """Convenience wrapper converting a NumPy array into a torch tensor
    of the desired dtype / device.
    """
    return torch.as_tensor(arr, dtype=dtype, device=device)


@st.composite
def _graph_and_tensor_batches(draw) -> Tuple[nx.DiGraph, torch.Tensor, torch.Tensor]:
    """
    Hypothesis strategy:
      • directed, rooted tree with N≤10 nodes
      • batch size B ≤ 5
      • coordinates (B,N,3) in µm
      • E-fields  (B,N,3) in V/m
    The tensors are still NumPy arrays here - they get converted to torch
    in the test body so we can easily move them to CPU / CUDA and change dtype.
    """
    # --- graph ---
    n_nodes = draw(st.integers(min_value=2, max_value=10))
    root_id = 0
    undirected_tree = nx.random_labeled_tree(
        n_nodes, seed=draw(st.integers(0, 2**32 - 1))
    )
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    # orient edges outwards from the root to guarantee a single root
    for parent, child in nx.bfs_edges(undirected_tree, root_id):
        G.add_edge(parent, child)

    # --- tensors ---
    batch_size = draw(st.integers(min_value=1, max_value=5))
    coords = draw(
        hnp.arrays(
            np.float64,
            (batch_size, n_nodes, 3),
            elements=st.floats(-COORD_RANGE_UM, COORD_RANGE_UM, allow_nan=False),
        )
    )
    efields = draw(
        hnp.arrays(
            np.float64,
            (batch_size, n_nodes, 3),
            elements=st.floats(-EFIELD_RANGE, EFIELD_RANGE, allow_nan=False),
        )
    )
    return G, coords, efields


def _expected_child_potential(
    pos_p_m: torch.Tensor,
    pos_c_m: torch.Tensor,
    E_p: torch.Tensor,
    E_c: torch.Tensor,
    psi_p: torch.Tensor,
) -> torch.Tensor:
    """
    One-edge analytic expectation:

        ψ_c = ψ_p - 1000 * ⟨0.5(E_p + E_c),  s_pc⟩

    Shapes: all (B,3) except psi_p  (B,)
    """
    s_pc = pos_c_m - pos_p_m
    E_avg = 0.5 * (E_p + E_c.to(E_p.dtype))  # ensure same dtype
    dot = (E_avg * s_pc).sum(dim=1)  # (B,)
    return psi_p - 1_000.0 * dot  # convert V→mV


def _all_close(a: torch.Tensor, b: torch.Tensor, rtol=1e-4, atol=1e-4) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------#
# 3.  Property-based core correctness test
# -----------------------------------------------------------------------------#
@given(
    data=_graph_and_tensor_batches(),
    dtype=st.sampled_from(FLOAT_DTYPES),
    device=st.sampled_from(["cpu"] + (["cuda"] if torch.cuda.is_available() else [])),
)
@settings(max_examples=200, deadline=None)
def test_edgewise_formula_holds(data, dtype: torch.dtype, device: str):
    """
    For every edge (parent→child) in the graph and for every sample in the batch,

        ψ_child == ψ_parent - 1000 * ⟨0.5(E_p+E_c),  x_c-x_p⟩

    within numerical tolerance.
    """
    G, coords_np, efields_np = data

    # Convert to torch
    coords = _torch_from_numpy(coords_np, dtype=dtype, device=device)  # (B,N,3)
    efields = _torch_from_numpy(efields_np, dtype=dtype, device=device)  # (B,N,3)
    B, N, _ = coords.shape

    # Split coordinate tensor for API
    x, y, z = (coords[..., i].clone() for i in range(3))

    # SUT
    psi = calculate_quasipotentials_batched_coords(G, x, y, z, efields)  # (B,N)

    # Path-wise checks ---------------------------------------------------------
    coords_m = coords * 1e-6  # µm → m
    for parent, child in G.edges:
        expected = _expected_child_potential(
            coords_m[:, parent, :],
            coords_m[:, child, :],
            efields[:, parent, :],
            efields[:, child, :],
            psi[:, parent],
        )
        assert _all_close(psi[:, child], expected), (
            f"Edge ({parent}->{child}) violated quasipotential formula "
            f"max|Δ|={torch.max(torch.abs(psi[:, child] - expected))}"
        )

    # Root potentials should be ≈ 0 mV
    roots = [n for n, deg in G.in_degree() if deg == 0]
    assume(roots)  # they *should* exist by construction
    for r in roots:
        assert torch.allclose(
            psi[:, r], torch.zeros_like(psi[:, r]), rtol=0, atol=1e-7
        ), "Root potentials must be zero."


# -----------------------------------------------------------------------------#
# 4.  Shape & input-validation error smoke tests
# -----------------------------------------------------------------------------#
def _dummy_graph(n=3):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    return G


@pytest.mark.parametrize("bad_shape", [(2, 3), (3, 4)])
def test_coord_shape_mismatch_raises(bad_shape):
    G = _dummy_graph(3)
    B = 1
    good = torch.zeros(B, 3)
    bad = torch.zeros(*bad_shape)
    e = torch.zeros(B, 3, 3)
    with pytest.raises(ValueError):
        calculate_quasipotentials_batched_coords(G, good, bad, good, e)


def test_efield_shape_mismatch_raises():
    G = _dummy_graph(3)
    B = 2
    coords = torch.zeros(B, 3)
    e_bad = torch.zeros(B, 3, 2)  # wrong last dim
    with pytest.raises(ValueError):
        calculate_quasipotentials_batched_coords(G, coords, coords, coords, e_bad)


def test_no_root_raises():
    # Construct a simple 1-cycle graph (every node has in-degree ≥ 1)
    G = nx.DiGraph([(0, 1), (1, 0)])
    coords = torch.zeros(1, 2)
    efields = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError):
        calculate_quasipotentials_batched_coords(G, coords, coords, coords, efields)


# -----------------------------------------------------------------------------#
# 5.  Device/dtype smoke test (non-property) for largeish inputs
# -----------------------------------------------------------------------------#
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_batch_cuda_smoke():
    """
    End-to-end CUDA run large enough to catch obvious device / dtype bugs.
    """
    torch.manual_seed(0)
    B = 8
    r, h = 2, 6  # 2^7-1 = 127 nodes -- fits nicely in one warp
    G_undirected = nx.balanced_tree(r=r, h=h)

    # Orient every edge away from the root (node 0) so we have a single root
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes)
    G.add_edges_from(nx.bfs_edges(G_undirected, source=0))

    N = G.number_of_nodes()  # 127
    coords = torch.randn(B, N, 3, dtype=torch.float32, device="cuda") * COORD_RANGE_UM
    efields = torch.randn(B, N, 3, dtype=torch.float32, device="cuda") * EFIELD_RANGE
    x, y, z = (coords[..., i] for i in range(3))

    psi = calculate_quasipotentials_batched_coords(G, x, y, z, efields)

    assert psi.shape == (B, N) and psi.is_cuda
