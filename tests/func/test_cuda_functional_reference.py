"""CPU algebra/transform contracts for the CUDA facade, without Triton imports.

These tests inject dense nonmutating raw solves. They validate the shared
autograd/JVP/vmap machinery, not GPU kernels, launches, or CUDA performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from dendra.models.integrators import _cuda_functional as facade

pytestmark = pytest.mark.cpu
FAMILIES = ("thomas", "pcr", "block_thomas", "dhs", "dhs_bt", "multi")
PARENTS = (2, 0, -1, 2, 3)


def _solve(matrix, rhs):
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)


def _block_diagonal(main):
    size, block = main.shape[-3], main.shape[-1]
    identity = torch.eye(size, dtype=main.dtype, device=main.device)
    return torch.einsum("ij,...iab->...iajb", identity, main).reshape(
        *main.shape[:-3], size * block, size * block
    )


def _tridiagonal_matrix(lower, main, upper, *, block=False):
    if not block:
        return (
            torch.diag_embed(main)
            + torch.diag_embed(lower, offset=-1)
            + torch.diag_embed(upper, offset=1)
        )
    size = main.shape[-3]
    matrix = _block_diagonal(main)
    identity = torch.eye(size, dtype=main.dtype, device=main.device)
    component = torch.eye(3, dtype=main.dtype, device=main.device)
    for child in range(1, size):
        position = torch.outer(identity[child], identity[child - 1])
        matrix = matrix + torch.einsum(
            "ij,...a,ab->...iajb", position, lower[..., child - 1, :], component
        ).reshape(matrix.shape)
        matrix = matrix + torch.einsum(
            "ji,...a,ab->...iajb", position, upper[..., child - 1, :], component
        ).reshape(matrix.shape)
    return matrix


def _tree_matrix(main, edge, parents, *, block=False):
    size = len(parents)
    identity = torch.eye(size, dtype=main.dtype, device=main.device)
    matrix = _block_diagonal(main) if block else torch.diag_embed(main)
    for child, parent in enumerate(parents):
        # The root placeholder is diagonal-only. Although ordinarily zero,
        # its derivative is not the zero derivative of an ignored argument.
        incidence = (
            identity[child] if parent < 0 else identity[child] - identity[parent]
        )
        edge_matrix = torch.outer(incidence, incidence)
        if block:
            component = torch.eye(3, dtype=main.dtype, device=main.device)
            term = torch.einsum(
                "ij,...a,ab->...iajb", edge_matrix, edge[..., child, :], component
            ).reshape(matrix.shape)
        else:
            term = edge[..., child, None, None] * edge_matrix
        matrix = matrix + term
    return matrix


def _raw_tridiagonal(lower, main, upper, rhs, block, pcr):
    del pcr
    matrix = _tridiagonal_matrix(lower, main, upper, block=block)
    if block:
        return _solve(matrix, rhs.flatten(-2)).reshape(rhs.shape)
    return _solve(matrix, rhs)


def _raw_tree(main, edge, rhs, parent, order, layers, threads, block):
    del parent, order, layers, threads
    matrix = _tree_matrix(main, edge, PARENTS, block=block)
    if block:
        return _solve(matrix, rhs.flatten(-2)).reshape(rhs.shape)
    return _solve(matrix, rhs)


def _multi_reference(main, edge, rhs, plan, *, poison_padding=False):
    parent, _order, layers, p_off, _o_off, l_off, levels, row_base, row_count = plan
    result = []
    # Metadata are deliberately ordinary CPU tensors in this injected raw
    # reference. They are not part of the production CUDA tensor program.
    groups = zip(
        p_off.tolist(),
        l_off.tolist(),
        levels.tolist(),
        row_base.tolist(),
        row_count.tolist(),
        strict=True,
    )
    for parent_offset, layer_offset, depth, first_row, rows in groups:
        width = int(layers[layer_offset + depth])
        parents = parent[parent_offset : parent_offset + width].tolist()
        selection = slice(first_row, first_row + rows)
        solved = _solve(
            _tree_matrix(main[selection, :width], edge[selection, :width], parents),
            rhs[selection, :width],
        )
        padding = torch.full(
            (rows, rhs.shape[-1] - width),
            float("nan") if poison_padding else 0.0,
            dtype=rhs.dtype,
            device=rhs.device,
        )
        result.append(torch.cat((solved, padding), dim=-1))
    return torch.cat(result, dim=0)


@pytest.fixture(autouse=True)
def dense_raw_operators(monkeypatch):
    calls = {"tridiagonal": 0, "tree": 0, "multi": 0}

    def tridiagonal(*args):
        calls["tridiagonal"] += 1
        return _raw_tridiagonal(*args)

    def tree(*args):
        calls["tree"] += 1
        return _raw_tree(*args)

    def multi(main, edge, rhs, *metadata):
        calls["multi"] += 1
        threads, grid = metadata[11:13]
        per_warp = metadata[3:9]
        assert all(value.numel() == grid for value in per_warp)
        assert grid == metadata[7].numel()
        assert all(count <= 32 // threads for count in metadata[8].tolist())
        return _multi_reference(main, edge, rhs, metadata[:9], poison_padding=True)

    # Preserve the dispatcher operators: capture must retain their registered
    # AD/fake/vmap boundary rather than trace through this dense implementation.
    monkeypatch.setattr(facade, "_launch_tridiagonal", tridiagonal)
    monkeypatch.setattr(facade, "_launch_tree", tree)
    monkeypatch.setattr(facade, "_launch_multi", multi)
    return calls


@dataclass
class Case:
    solve: object
    reference: object
    values: tuple
    metadata: tuple = ()


def _values(shape, *, start=0.1):
    count = 1
    for size in shape:
        count *= size
    return torch.arange(count, dtype=torch.float64).reshape(shape) * 0.013 + start


def _case(family):
    block = family in ("block_thomas", "dhs_bt")
    batch, size = 2, len(PARENTS)
    diagonal = _values((batch, size), start=4.0)
    if block:
        coupling = torch.tensor(
            [[0.0, 0.07, -0.03], [-0.02, 0.0, 0.05], [0.01, 0.04, 0.0]],
            dtype=torch.float64,
        )
        diagonal = (
            diagonal[..., None, None] * torch.eye(3, dtype=torch.float64) + coupling
        )
    rhs = _values((batch, size, 3) if block else (batch, size), start=0.3)
    if family in ("thomas", "pcr", "block_thomas"):
        shape = (batch, size - 1, 3) if block else (batch, size - 1)
        lower, upper = _values(shape, start=-0.18), _values(shape, start=0.07)

        def solve(a, b, c, d):
            return facade._apply_tridiagonal(a, b, c, d, block, family == "pcr")

        def reference(a, b, c, d):
            return _raw_tridiagonal(a, b, c, d, block, family == "pcr")

        return Case(solve, reference, (lower, diagonal, upper, rhs))
    if family in ("dhs", "dhs_bt"):
        edge = _values(rhs.shape, start=0.11)
        edge[:, 2] = 0.0
        parent = torch.tensor(PARENTS, dtype=torch.long)
        order = torch.tensor([1, 4, 0, 3, 2], dtype=torch.long)
        layers = torch.tensor([0, 2, 4, 5], dtype=torch.long)

        def solve(main, edges, source):
            return facade._apply_tree(
                main, edges, source, parent, order, layers, 32, block
            )

        def reference(main, edges, source):
            return _raw_tree(main, edges, source, parent, order, layers, 32, block)

        return Case(solve, reference, (diagonal, edge, rhs), (parent, order, layers))
    plan = tuple(
        torch.tensor(values, dtype=torch.long)
        for values in (
            [1, -1, 1, -1, 0],
            [0, 2, 1, 1, 0],
            [0, 2, 3, 0, 1, 2],
            [0, 3],
            [0, 3],
            [0, 3],
            [2, 2],
            [0, 2],
            [2, 1],
        )
    )
    main, edge, rhs = (_values((3, 5), start=start) for start in (4.0, 0.1, 0.3))
    edge[:2, 1] = 0.0
    edge[2, 0] = 0.0

    def solve(main, edge, source):
        return facade._apply_multi(main, edge, source, *plan, 5, 2, 16, 2)

    def reference(main, edge, source):
        return _multi_reference(main, edge, source, plan)

    return Case(solve, reference, (main, edge, rhs), plan)


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_reference_primal_all_coefficient_gradients_and_input_ownership(family):
    case = _case(family)
    actual_inputs = tuple(value.clone().requires_grad_() for value in case.values)
    expected_inputs = tuple(value.clone().requires_grad_() for value in case.values)
    snapshots = tuple(
        (value.clone(), value._version) for value in (*actual_inputs, *case.metadata)
    )
    actual, expected = case.solve(*actual_inputs), case.reference(*expected_inputs)
    _assert_close(actual, expected)
    weights = _values(actual.shape, start=0.2)
    actual_grads = torch.autograd.grad((actual.square() * weights).sum(), actual_inputs)
    expected_grads = torch.autograd.grad(
        (expected.square() * weights).sum(), expected_inputs
    )
    for actual_gradient, expected_gradient in zip(
        actual_grads, expected_grads, strict=True
    ):
        _assert_close(actual_gradient, expected_gradient)
    for value, (snapshot, version) in zip(
        (*actual_inputs, *case.metadata), snapshots, strict=True
    ):
        assert value._version == version
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    if family in ("dhs", "dhs_bt"):
        # Nonzero root cotangent despite the conventional zero root coefficient.
        assert torch.count_nonzero(actual_grads[1][:, 2])
    if family == "multi":
        assert torch.count_nonzero(actual[:2, 3:]) == 0
        assert torch.count_nonzero(actual[2, 2:]) == 0
        for gradient in actual_grads:
            assert torch.count_nonzero(gradient[:2, 3:]) == 0
            assert torch.count_nonzero(gradient[2, 2:]) == 0


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_reference_joint_jvp_matches_dense_algebra(family):
    case = _case(family)
    tangents = tuple(_values(value.shape, start=0.02) for value in case.values)
    actual = torch.func.jvp(case.solve, case.values, tangents)
    expected = torch.func.jvp(case.reference, case.values, tangents)
    for value, reference in zip(actual, expected, strict=True):
        _assert_close(value, reference)


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_reference_rhs_vmap_and_nested_vmap(family):
    case = _case(family)
    coefficients, rhs = case.values[:-1], case.values[-1]
    sources = torch.stack((rhs, rhs * 0.8, rhs * 1.2))

    def solve(source):
        return case.solve(*coefficients, source)

    actual = torch.vmap(solve)(sources)
    expected = torch.stack(
        [case.reference(*coefficients, source) for source in sources]
    )
    _assert_close(actual, expected)
    nested = torch.stack((sources, sources * 0.9))
    _assert_close(
        torch.vmap(torch.vmap(solve))(nested), torch.stack((expected, expected * 0.9))
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_reference_empty_mapped_coefficients_are_differentiable(
    family, dense_raw_operators
):
    case = _case(family)
    mapped = case.values[0].new_empty((0, *case.values[0].shape)).requires_grad_()

    def solve(first):
        return case.solve(first, *case.values[1:])

    result = torch.vmap(solve)(mapped)
    assert result.shape == (0, *case.values[-1].shape)
    gradient = torch.autograd.grad(result.sum(), mapped)[0]
    assert gradient.shape == mapped.shape and gradient.numel() == 0
    assert sum(dense_raw_operators.values()) == 0


def test_tree_topology_mapped_vmap_fails_before_raw_dispatch(dense_raw_operators):
    case = _case("dhs")
    parent, order, layers = case.metadata

    def solve(mapped_parent):
        return facade._apply_tree(*case.values, mapped_parent, order, layers, 32, False)

    with pytest.raises(ValueError, match="topology.*shared"):
        torch.vmap(solve)(torch.stack((parent, parent)))
    assert dense_raw_operators["tree"] == 0


@pytest.mark.parametrize("family", ("dhs", "dhs_bt", "multi"))
def test_strided_topology_is_normalized_before_cuda_launch(monkeypatch, family):
    case = _case(family)
    strided = tuple(
        torch.stack((value, torch.zeros_like(value)), dim=-1).reshape(-1)[::2]
        for value in case.metadata
    )
    assert all(
        value.stride() == (2,) and not value.is_contiguous() for value in strided
    )
    snapshots = tuple((value.clone(), value._version) for value in strided)
    if family == "multi":
        original = facade._launch_multi

        def checked(main, edge, rhs, *metadata):
            assert all(value.is_contiguous() for value in metadata[:9])
            return original(main, edge, rhs, *metadata)

        monkeypatch.setattr(facade, "_launch_multi", checked)
        actual = facade._apply_multi(*case.values, *strided, 5, 2, 16, 2)
    else:
        original = facade._launch_tree

        def checked(main, edge, rhs, parent, order, layers, threads, block):
            assert all(value.is_contiguous() for value in (parent, order, layers))
            return original(main, edge, rhs, parent, order, layers, threads, block)

        monkeypatch.setattr(facade, "_launch_tree", checked)
        actual = facade._apply_tree(*case.values, *strided, 32, family == "dhs_bt")
    _assert_close(actual, case.reference(*case.values))
    for value, (snapshot, version) in zip(strided, snapshots, strict=True):
        assert value._version == version
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_no_grad_capture_retains_opaque_operator_and_registered_reverse_ad(family):
    case = _case(family)
    with torch.no_grad():
        graph = make_fx(case.solve)(*case.values)
    operator_nodes = [
        node
        for node in graph.graph.nodes
        if node.op == "call_function"
        and str(node.target).startswith("dendra_functional_cuda.")
    ]
    assert operator_nodes, "The solver dispatcher boundary must survive make_fx"
    # The injected dense implementation must not have been traced instead.
    assert not any(
        "linalg_solve" in str(node.target)
        for node in graph.graph.nodes
        if node.op == "call_function"
    )
    inputs = tuple(value.clone().requires_grad_() for value in case.values)
    expected_inputs = tuple(value.clone().requires_grad_() for value in case.values)
    actual, expected = graph(*inputs), case.reference(*expected_inputs)
    _assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), expected_inputs),
        strict=True,
    ):
        _assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_reference_fullgraph_aot_forward_and_backward(family):
    case = _case(family)
    compiled = torch.compile(case.solve, backend="aot_eager", fullgraph=True)
    inputs = tuple(value.clone().requires_grad_() for value in case.values)
    expected_inputs = tuple(value.clone().requires_grad_() for value in case.values)
    actual, expected = compiled(*inputs), case.reference(*expected_inputs)
    _assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), expected_inputs),
        strict=True,
    ):
        _assert_close(actual_gradient, expected_gradient)
