"""Scan joint capture retains data-dependent integer index/frame examples."""

from __future__ import annotations

import importlib
from contextlib import nullcontext

import pytest
import torch
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
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


@pytest.mark.parametrize("functionalize", (False, True))
def test_real_joint_examples_keep_native_plan_resources_metadata_only(functionalize):
    """A mixed bool carry must not execute a native kernel on copied handles."""

    solvers = pytest.importorskip("dendra_solvers")
    if not hasattr(solvers, "DHSPlan"):
        pytest.skip("native DHS plan support is unavailable")
    _scan_compat.install_scan_compatibility()
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    plan = solvers.DHSPlan(
        torch.tensor([-1, 0, 1]),
        torch.tensor([2, 1, 0]),
        torch.tensor([0, 1, 2, 3]),
    )
    diagonal = torch.tensor([[1.1, 1.2, 1.3]], dtype=torch.float64, requires_grad=True)
    coupling = torch.tensor([[0.0, 0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor([[0.4, -0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    flag = torch.tensor([[False, True, False]])
    inputs = (diagonal, coupling, rhs, flag, plan._token)
    snapshots = tuple((value.detach().clone(), value._version) for value in inputs)

    def objective(diagonal, coupling, rhs, flag, token):
        gated = torch.where(flag, rhs * 0.5, rhs)
        solved = torch.ops.dendra_solvers.dhs_solve_plan.default(
            diagonal, coupling, gated, token
        )
        return solved, ~flag

    before_mode = active_fake_mode()
    joint = partitioner.create_hop_joint_graph(objective, inputs, functionalize)
    assert active_fake_mode() is before_mode
    expected, expected_flag = objective(*inputs)
    cotangent = expected.detach() * 2
    expected_gradients = torch.autograd.grad(expected, inputs[:3], cotangent)
    value, updated_flag, *gradients = joint.joint_gm(
        *inputs, cotangent, torch.zeros_like(flag)
    )
    torch.testing.assert_close(value, expected)
    assert torch.equal(updated_flag, expected_flag)
    for actual, reference in zip(gradients[:3], expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)
    assert not torch.count_nonzero(gradients[3])
    assert not torch.count_nonzero(gradients[4])
    for value, (snapshot, version) in zip(inputs, snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
        assert value._version == version


@pytest.mark.parametrize("caller_fake", (False, True))
@pytest.mark.parametrize("fail", (False, True))
def test_private_materializer_restores_caller_context(caller_fake, fail):
    utils = importlib.import_module("torch._higher_order_ops.utils")
    materialize = _scan_compat._make_value_preserving_materializer(utils)
    value = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    snapshot, version = value.detach().clone(), value._version
    caller = FakeTensorMode(allow_non_fake_inputs=True) if caller_fake else None

    def objective(value):
        assert active_fake_mode() is not None
        if fail:
            raise RuntimeError("injected metadata materialization error")
        return value.sin()

    with caller if caller is not None else nullcontext():
        with torch.no_grad():
            before = active_fake_mode()
            if fail:
                with pytest.raises(RuntimeError, match="injected metadata"):
                    materialize(objective, (value,), force_enable_grad=True)
            else:
                graph = materialize(objective, (value,), force_enable_grad=True)
                assert isinstance(graph, torch.fx.GraphModule)
            assert active_fake_mode() is before
            assert not torch.is_grad_enabled()
    assert active_fake_mode() is None
    torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    assert value._version == version


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


@pytest.mark.parametrize(
    "target",
    [
        "_from_fun",
        "materialize_as_graph",
        "alias",
        "detect_fake_mode",
        "FakeTensorMode",
    ],
)
def test_scan_rechecks_private_materializer_dependencies(monkeypatch, target):
    _scan_compat.install_scan_compatibility()
    utils = importlib.import_module("torch._higher_order_ops.utils")
    partitioner = importlib.import_module("torch._higher_order_ops.partitioner")
    if target == "alias":
        owner, name = partitioner, "materialize_as_graph"
    elif target == "detect_fake_mode":
        owner, name = importlib.import_module("torch._guards"), target
    elif target == "FakeTensorMode":
        owner, name = importlib.import_module("torch._subclasses.fake_tensor"), target
    else:
        owner, name = utils, target
    monkeypatch.setattr(owner, name, lambda *args, **kwargs: None)
    with pytest.raises(FunctionalizationError, match="active helper.*changed"):
        _scan_compat.install_scan_compatibility()
