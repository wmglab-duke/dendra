"""Owned inference packs reduce HOP carry without changing public ownership."""

import weakref
from dataclasses import fields

import pytest
import torch
from test_while_block_ownership import (
    _assert_owned,
    _capture,
    _case,
    _clone,
    _close,
    _eager,
    _offset_view,
    _tensor_leaves,
    _ViewFunctional,
)

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._carry_packing import _clone_tree_like, _PackedCarryLayout

pytestmark = pytest.mark.cpu


def test_layout_packs_mixed_scalar_empty_and_noncontiguous_leaves_without_examples():
    floating = torch.arange(12, dtype=torch.float64).reshape(4, 3).T
    tree = (
        {"v": floating, "alias": floating.view_as(floating)},
        {
            "scalar": torch.tensor(0.2, dtype=torch.float64),
            "empty": torch.empty((2, 0, 3), dtype=torch.float64),
            "integer": torch.arange(6, dtype=torch.int64).reshape(3, 2).T,
            "boolean": torch.tensor([[True, False], [False, True]]).T,
            "no_carry": (),
        },
    )
    original = _clone(tree)
    layout = _PackedCarryLayout.from_tree(tree)
    assert len(layout.groups) == 3
    assert all(
        not torch.is_tensor(getattr(layout, field.name)) for field in fields(layout)
    )
    assert all(isinstance(leaf.shape, tuple) for leaf in layout.leaves)
    packs = layout.pack(tree)
    assert len(packs) == 3 and all(
        pack.ndim == 1 and pack.is_contiguous() for pack in packs
    )
    for pack in packs:
        assert all(
            not torch._C._is_alias_of(pack, value) for value in _tensor_leaves(tree)
        )
    unpacked = layout.unpack(packs)
    _close(unpacked, tree, exact=True)
    owned = _clone_tree_like(unpacked, tree)
    _close(owned, tree, exact=True)
    for value, template in zip(
        _tensor_leaves(owned), _tensor_leaves(tree), strict=True
    ):
        assert value.stride() == template.clone().stride()
    leaves = _tensor_leaves(owned)
    for index, value in enumerate(leaves):
        assert all(
            not torch._C._is_alias_of(value, other) for other in leaves[index + 1 :]
        )
        assert all(not torch._C._is_alias_of(value, pack) for pack in packs)
    _close(tree, original, exact=True)

    def disposable_layout():
        example = torch.arange(3.0)
        return _PackedCarryLayout.from_tree({"x": example}), weakref.ref(example)

    retained_layout, reference = disposable_layout()
    assert retained_layout.leaves[0].shape == (3,) and reference() is None


def test_empty_tree_and_empty_only_group_remain_owned():
    empty_tree = ({}, {"no_carry": ()})
    layout = _PackedCarryLayout.from_tree(empty_tree)
    assert layout.pack(empty_tree) == () and layout.unpack(()) == empty_tree
    empty = {"x": torch.empty((2, 0, 3)), "y": torch.empty((0,))}
    layout = _PackedCarryLayout.from_tree(empty)
    first, second = layout.pack(empty), layout.pack(empty)
    assert first[0].numel() == 0 and second[0].numel() == 0
    assert not torch._C._is_alias_of(first[0], second[0])
    _close(layout.unpack(first), empty, exact=True)


def test_nonzero_offset_views_preserve_clone_strides_values_versions_and_ownership():
    storage = torch.arange(40, dtype=torch.float64)
    tree = {
        "non_dense": storage[3:27:2].view(2, 6),
        "offset_dense": storage[4:10].view(2, 3),
        "scalar": storage[17].view(()),
        "empty": storage[5:5].view(0),
    }
    assert all(value.storage_offset() > 0 for value in tree.values())
    original = _clone(tree)
    versions = tuple(value._version for value in _tensor_leaves(tree))
    storage_original, storage_version = storage.clone(), storage._version
    layout = _PackedCarryLayout.from_tree(tree)
    first = _clone_tree_like(layout.unpack(layout.pack(tree)), tree)
    second = _clone_tree_like(layout.unpack(layout.pack(tree)), tree)
    for outputs in (first, second):
        _close(outputs, tree, exact=True)
        for name, value in outputs.items():
            assert value.stride() == tree[name].clone().stride()
            assert value.storage_offset() == 0
            assert not torch._C._is_alias_of(value, storage)
            if outputs is first:
                assert all(
                    not torch._C._is_alias_of(value, other) for other in second.values()
                )
    first["non_dense"].add_(100)
    _close(second, original, exact=True)
    _close(tree, original, exact=True)
    torch.testing.assert_close(storage, storage_original, rtol=0, atol=0)
    assert storage._version == storage_version
    assert versions == tuple(value._version for value in _tensor_leaves(tree))


def test_generic_pack_and_public_copy_preserve_autograd_identity():
    # The executor is inference-only; its generic storage utility nevertheless
    # must not detach values or add a dependency on format-only templates.
    source = torch.linspace(-0.5, 0.5, 6, dtype=torch.float64, requires_grad=True)
    value = source.reshape(3, 2).T
    tree = {"x": value, "alias": value.view_as(value)}
    templates = {name: torch.empty_like(leaf) for name, leaf in tree.items()}
    layout = _PackedCarryLayout.from_tree(tree)
    result = _clone_tree_like(layout.unpack(layout.pack(tree)), templates)
    loss = result["x"].square().sum() + result["alias"].square().sum()
    gradient = torch.autograd.grad(loss, source)[0]
    torch.testing.assert_close(gradient, 4 * source, rtol=0, atol=0)


def test_layout_rejects_non_tensor_leaves_and_changed_structure():
    with pytest.raises(TypeError, match="strided Tensor"):
        _PackedCarryLayout.from_tree({"not_tensor": 1})
    layout = _PackedCarryLayout.from_tree({"x": torch.ones(2)})
    with pytest.raises(ValueError, match="structure"):
        layout.pack({"different": torch.ones(2)})
    with pytest.raises(ValueError, match="groups"):
        layout.unpack(())


class _DuplicateStateFunctional(_ViewFunctional):
    """Identical sibling state outputs must occupy one pack, not two handles."""

    @staticmethod
    def _step_values(parameters, prepared, state, inputs):
        result, auxiliary = _ViewFunctional._step_values(
            parameters, prepared, state, inputs
        )
        result["integrator"]["view"] = result["integrator"]["v"].view_as(
            result["integrator"]["v"]
        )
        return result, auxiliary

    @staticmethod
    def _clone_state_tree(state):
        result = _ViewFunctional._clone_state_tree(state)
        result["integrator"]["view"] = state["integrator"]["view"].clone(
            memory_format=torch.contiguous_format
        )
        return result


def _duplicate_case():
    case = _case(5)
    case.functional = _DuplicateStateFunctional()
    case.state["integrator"]["v"] = _offset_view(case.state["integrator"]["v"])
    case.state["integrator"]["view"] = case.state["integrator"]["v"].view_as(
        case.state["integrator"]["v"]
    )
    case.state["fixed"] = _offset_view(case.state["fixed"])
    case.ve, case.intra = _offset_view(case.ve), _offset_view(case.intra)
    floating, _distinct, integer, boolean = case.carries["unchanged"]
    floating, integer, boolean = map(_offset_view, (floating, integer, boolean))
    case.carries["unchanged"] = (floating, floating.view_as(floating), integer, boolean)
    assert all(
        value.storage_offset() > 0
        for value in (
            case.state["integrator"]["v"],
            case.ve,
            case.intra,
            floating,
            integer,
            boolean,
        )
    )
    return case


def test_packed_hop_has_only_index_and_three_dtype_group_carries():
    case = _duplicate_case()
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    kernel = _capture(case, 4, {"backend": backend, "fullgraph": True})
    with torch.no_grad():
        actual = kernel(
            case.parameters,
            case.prepared,
            case.state,
            case.ve[:4],
            case.intra[:4],
            case.carries,
        )
    _close(actual, _eager(case, 4))
    hops = [
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.higher_order.while_loop
    ]
    assert len(hops) == 1 and len(hops[0].args[2]) == 4
    _assert_owned(actual, (case.state, case.carries, case.ve, case.intra))


@pytest.mark.slow
@pytest.mark.parametrize("threads", [1, 4], ids=["t1", "t4"])
def test_cpp_packed_while_handles_duplicate_state_and_carry_outputs(threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        _check_cpp_duplicate_outputs()
    finally:
        torch.set_num_threads(previous)


def _check_cpp_duplicate_outputs():
    case = _duplicate_case()
    inputs = (
        case.parameters,
        case.prepared,
        case.state,
        case.ve,
        case.intra,
        case.carries,
    )
    originals = _clone(inputs)
    versions = tuple(value._version for value in _tensor_leaves(inputs))
    options = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": {"cpp_wrapper": True, "compile_threads": 1, "fx_graph_cache": False},
    }
    with torch.inference_mode(), torch_compiler_warning_context():
        for steps in (4, 5):
            kernel = _capture(case, steps, options)
            assert callable(kernel.get_compiler_config)
            assert callable(kernel.__wrapped__)
            actual = kernel(
                case.parameters,
                case.prepared,
                case.state,
                case.ve[:steps],
                case.intra[:steps],
                case.carries,
            )
            _close(actual, _eager(case, steps))
            _assert_owned(actual, inputs)
            for value, template in zip(
                _tensor_leaves(actual[2]), _tensor_leaves(case.carries), strict=True
            ):
                assert value.stride() == template.clone().stride()
            saved = _clone(actual)
            repeated = kernel(
                {"gain": case.parameters["gain"] + 0.2},
                case.prepared,
                case.state,
                case.ve[:steps],
                -case.intra[:steps],
                case.carries,
            )
            _assert_owned(repeated, (inputs, actual))
            _close(actual, saved, exact=True)
    _close(inputs, originals, exact=True)
    assert versions == tuple(value._version for value in _tensor_leaves(inputs))
