"""Scan joint capture retains data-dependent integer index/frame examples."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode

from dendra.func import FunctionalizationError, _scan_compat

pytestmark = [
    pytest.mark.cpu,
]


def test_scan_joint_capture_preserves_integer_indices_and_frame_values(monkeypatch):
    _scan_compat.install_scan_compatibility()
    utils = importlib.import_module("torch._higher_order_ops.utils")
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    shared_materializer, shared_from_fun = utils.materialize_as_graph, utils._from_fun
    table = torch.arange(1.0, 13.0).reshape(4, 3).requires_grad_()
    # Noncontiguous node indices also exercise value preservation across a
    # clone whose layout follows the verified helper's Boolean clone policy.
    nodes = torch.tensor([0, -1, 2, -1])[::2]
    frame = torch.tensor(1)
    empty_strided = torch.empty_strided

    def invalid_discrete_examples(*args, **kwargs):
        value = empty_strided(*args, **kwargs)
        if not (value.dtype.is_floating_point or value.dtype.is_complex):
            value.fill_(999_999)
        return value

    # Make the old uninitialized-example bug deterministic. Float/complex
    # examples retain the original allocation behavior during graph capture.
    monkeypatch.setattr(torch, "empty_strided", invalid_discrete_examples)

    def objective(values, selected_nodes, step):
        row = values.index_select(0, step.reshape(1)).squeeze(0)
        return (row.index_select(0, selected_nodes).square().sum(),)

    joint = partitioner.create_hop_joint_graph(objective, (table, nodes, frame), False)
    for selected_nodes, step in ((nodes, frame), (torch.tensor([1, 2]), frame + 2)):
        expected = objective(table, selected_nodes, step)[0]
        expected_gradient = torch.autograd.grad(expected, table)[0]
        value, gradient, node_gradient, frame_gradient = joint.joint_gm.forward(
            table, selected_nodes, step, table.new_ones(())
        )
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
        assert not torch.count_nonzero(node_gradient)
        assert not torch.count_nonzero(frame_gradient)
    assert utils.materialize_as_graph is shared_materializer
    assert utils._from_fun is shared_from_fun
    assert partitioner.materialize_as_graph is shared_materializer


@pytest.mark.parametrize("wrapper", ["ordinary", "functional", "python_functional"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_scan_discrete_examples_unwrap_functional_tensors_without_aliases(
    wrapper, dtype
):
    utils = importlib.import_module("torch._higher_order_ops.utils")
    materialize = _scan_compat._make_value_preserving_materializer(utils)
    convert = materialize.__globals__["_from_fun"]
    value = torch.tensor([0, 1, 1, 0], dtype=dtype)[::2]
    wrapped = value
    if wrapper == "functional":
        wrapped = torch._to_functional_tensor(value)
    elif wrapper == "python_functional":
        with FunctionalTensorMode():
            wrapped = FunctionalTensor.to_functional(value)
    actual = convert(wrapped)
    assert type(actual) is torch.Tensor
    assert not torch._is_functional_tensor(actual)
    assert not torch._C._is_alias_of(actual, value)
    torch.testing.assert_close(actual, value, rtol=0, atol=0)
    assert actual.stride() == value.clone().stride()


@pytest.mark.parametrize("target", ["_from_fun", "materialize_as_graph", "alias"])
def test_scan_rechecks_private_materializer_dependencies(monkeypatch, target):
    _scan_compat.install_scan_compatibility()
    utils = importlib.import_module("torch._higher_order_ops.utils")
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    owner, name = (
        (partitioner, "materialize_as_graph") if target == "alias" else (utils, target)
    )
    monkeypatch.setattr(owner, name, lambda *args, **kwargs: None)
    with pytest.raises(FunctionalizationError, match="active helper.*changed"):
        _scan_compat.install_scan_compatibility()
