from unittest.mock import MagicMock, call, patch

import networkx as nx
import pytest

# Import the code you want to test
import axonml.models.io as io

# Create a conditional skip marker for tests that need NEURON
# This checks your `is_neuron_installed` function at the start of the test session.
requires_neuron = pytest.mark.skipif(
    not io.is_neuron_installed(), reason="NEURON is not installed"
)


def test_is_neuron_installed_checker():
    """Tests the helper function itself."""
    # This will be True or False depending on the test environment
    assert isinstance(io.is_neuron_installed(), bool)


@patch("axonml.models.io.NEURON_INSTALLED", False)
@pytest.mark.parametrize(
    "func, args",
    [
        (io.lambda_f, (MagicMock(), 100)),
        (io.apply_d_lambda, ([], 0.1)),
        (io.read_swc, ("file.swc",)),
        (io.read_neurolucida, ("file.asc",)),
    ],
)
def test_functions_raise_importerror_when_neuron_is_missing(func, args):
    """
    Verify that all NEURON-dependent functions raise ImportError if NEURON is not available.
    """
    with pytest.raises(ImportError, match="NEURON is not installed"):
        func(*args)


def test_find_branch_points():
    """Test the branch point identification logic."""
    G = nx.DiGraph()
    # 0 -> 1 -> 2
    # |
    # -> 3
    G.add_edges_from([(0, 1), (1, 2), (0, 3)])
    assert io.find_branch_points(G) == [0]


def test_reorder_graph_by_patterns():
    """Test the graph reordering logic."""
    G = nx.DiGraph()
    G.add_node(0, name="soma_seg_0")
    G.add_node(1, name="axon_part_1")
    G.add_node(2, name="unmyelinated_axon")
    G.add_node(3, name="myelin_sheath")
    G.add_node(4, name="other_part")

    patterns = {
        "SOMA": r"soma",
        "UNMYELIN": r"unmyelin",
        "MYELIN": r"\bmyelin\b",
        "AXON": r"axon",
    }
    # Note the order: SOMA, AXON, UNMYELIN, MYELIN
    group_order = ["SOMA", "AXON", "UNMYELIN", "MYELIN"]

    G_reordered, mapping = io.reorder_graph_by_patterns(G, patterns, group_order)

    # Expected new IDs based on group_order:
    # 0 (soma) -> 0
    # 1 (axon) -> 1
    # 2 (unmyelin) -> 2
    # 3 (myelin) -> 3
    # 4 (other) -> 4
    assert list(G_reordered.nodes) == [0, 1, 2, 3, 4]

    original_names = {mapping[old_id]: G.nodes[old_id]["name"] for old_id in mapping}
    assert "soma" in original_names[0]
    assert "axon" in original_names[1] and "unmyelin" not in original_names[1]
    assert "unmyelin" in original_names[2]
    assert "myelin" in original_names[3]
    assert "other" in original_names[4]
