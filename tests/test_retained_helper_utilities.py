from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from dendra.models.utils import distance, undirected_weighted_lengths
from dendra.utils.tensor_ops import add_dims_as_necessary, cartesian_product


def _weighted_graph():
    graph = nx.DiGraph()
    graph.add_nodes_from(["root", "middle", "tip", "isolated"])
    graph.add_edge("root", "middle", L=2.0, delay=7.0)
    graph.add_edge("middle", "tip", L=3.5, delay=11.0)
    return graph


def test_undirected_weighted_lengths_preserves_targets_and_unreachable_nodes():
    graph = _weighted_graph()

    result = undirected_weighted_lengths(
        graph, "tip", ["root", "middle", "tip", "isolated"]
    )

    torch.testing.assert_close(result, torch.tensor([5.5, 3.5, 0.0, float("inf")]))


def test_undirected_weighted_lengths_accepts_scalar_and_tuple_node_targets():
    graph = _weighted_graph()
    torch.testing.assert_close(
        undirected_weighted_lengths(graph, "root", "tip"), torch.tensor([5.5])
    )

    tuple_graph = nx.DiGraph()
    tuple_graph.add_edge((0, "left"), (1, "right"), L=4.0)
    torch.testing.assert_close(
        undirected_weighted_lengths(tuple_graph, (0, "left"), (1, "right")),
        torch.tensor([4.0]),
    )


def test_undirected_weighted_lengths_supports_a_custom_weight_attribute():
    graph = _weighted_graph()
    torch.testing.assert_close(
        undirected_weighted_lengths(
            graph, "root", ["middle", "tip"], weight_attr="delay"
        ),
        torch.tensor([7.0, 18.0]),
    )


def test_distance_resolves_slices_against_noninteger_graph_nodes():
    cell = SimpleNamespace(graph=_weighted_graph())

    result = distance(cell, slice(1, 2), slice(0, 4, 2))

    # The origin is "middle" and the targets are "root" and "tip".
    torch.testing.assert_close(result, torch.tensor([2.0, 3.5]))


def test_distance_reports_missing_graph_and_empty_origin_slice():
    with pytest.raises(ValueError, match="Graph is not defined"):
        distance(SimpleNamespace(graph=None), 0, [1])

    with pytest.raises(ValueError, match="origin slice selects no graph nodes"):
        distance(SimpleNamespace(graph=_weighted_graph()), slice(0, 0), ["tip"])


def test_cartesian_product_order_scalars_and_stacked_output():
    first, second, third = cartesian_product(
        [1, 2], np.array([10, 20]), torch.tensor(30)
    )
    assert first.tolist() == [1, 1, 2, 2]
    assert second.tolist() == [10, 20, 10, 20]
    assert third.tolist() == [30, 30, 30, 30]

    stacked = cartesian_product([1, 2], [10, 20], return_stacked=True)
    torch.testing.assert_close(
        stacked, torch.tensor([[1, 10], [1, 20], [2, 10], [2, 20]])
    )


def test_cartesian_product_promotes_or_overrides_dtype_and_device():
    promoted = cartesian_product(
        torch.tensor([1.0], dtype=torch.float32),
        np.array([2.0], dtype=np.float64),
        return_stacked=True,
    )
    assert promoted.dtype == torch.float64
    assert promoted.device.type == "cpu"

    overridden = cartesian_product(
        [1, 2], [3.5], dtype=torch.float32, device=torch.device("cpu")
    )
    assert all(value.dtype == torch.float32 for value in overridden)
    assert all(value.device.type == "cpu" for value in overridden)


def test_cartesian_product_preserves_autograd_for_repeated_values():
    first = torch.tensor([1.0, 2.0], requires_grad=True)
    second = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)

    expanded_first, expanded_second = cartesian_product(first, second)
    (expanded_first.sum() + expanded_second.sum()).backward()

    torch.testing.assert_close(first.grad, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(second.grad, torch.tensor([2.0, 2.0, 2.0]))


def test_empty_cartesian_product_remains_connected_to_autograd_inputs():
    empty = torch.empty(0, requires_grad=True)
    values = torch.tensor([1.0, 2.0], requires_grad=True)

    expanded_empty, expanded_values = cartesian_product(empty, values)
    assert expanded_empty.shape == expanded_values.shape == (0,)
    assert expanded_empty.requires_grad
    assert expanded_values.requires_grad

    (expanded_empty.sum() + expanded_values.sum()).backward()
    assert empty.grad is not None and empty.grad.numel() == 0
    torch.testing.assert_close(values.grad, torch.zeros_like(values))


def test_cartesian_product_shape_validation_and_flatten_opt_in():
    with pytest.raises(ValueError, match="at least one input"):
        cartesian_product()
    with pytest.raises(ValueError, match="Expected 1D inputs"):
        cartesian_product(torch.ones(2, 2))

    (flattened,) = cartesian_product(torch.tensor([[1, 2], [3, 4]]), require_1d=False)
    assert flattened.tolist() == [1, 2, 3, 4]


def test_add_dims_as_necessary_returns_broadcast_ready_autograd_views():
    target = torch.zeros(2, 3, 4)
    vector = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    aligned = add_dims_as_necessary(vector, target)

    assert aligned.shape == (1, 3, 1)
    torch.testing.assert_close(
        aligned + target,
        vector.detach().reshape(1, 3, 1).expand_as(target),
    )
    (aligned + target).sum().backward()
    torch.testing.assert_close(vector.grad, torch.full_like(vector, 8.0))


def test_add_dims_as_necessary_handles_scalars_and_uses_first_matching_axis():
    scalar = torch.tensor(2.0)
    assert add_dims_as_necessary(scalar, torch.zeros(2, 3)) is scalar

    vector = torch.tensor([1.0, 2.0])
    assert add_dims_as_necessary(vector, torch.zeros(2, 3, 2)).shape == (2, 1, 1)


def test_add_dims_as_necessary_rejects_incompatible_inputs():
    with pytest.raises(ValueError, match="Cannot broadcast"):
        add_dims_as_necessary(torch.ones(5), torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="must be 0D or 1D"):
        add_dims_as_necessary(torch.ones(2, 2), torch.zeros(2, 2))
