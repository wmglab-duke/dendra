"""Read-only interpolation and composable differentiation of prepared tables."""

import copy

import numpy as np
import pytest
import torch
from torch.func import functional_call, jacfwd, jacrev, jvp, vmap

from dendra.utils import PreparedInterp1d


def _table(uniform, learnable, *, batched=False, exact_clamp=True):
    x = torch.tensor(
        [0.0, 1.0, 2.0, 3.0] if uniform else [0.0, 0.4, 1.7, 3.0],
        dtype=torch.float64,
    )
    y = torch.tensor([0.3, 1.4, -0.2, 0.8], dtype=x.dtype)
    if batched:
        x = torch.stack((x, x * 2 - 0.5))
        y = torch.stack((y, y.flip(0) + 0.2))
    return PreparedInterp1d(
        x,
        y,
        learnable_y=learnable,
        uniform="always" if uniform else "never",
        exact_clamp=exact_clamp,
    )


def _numpy_rows(table, query, indices, outside, fill):
    x = table._x_search.detach().numpy()
    y = (table.y if table.learnable_y else table._y).detach().numpy()
    if not table.batched:
        x, y = x[None], y[None]
    query = query.detach().numpy()
    one_dimensional = query.ndim == 1
    query = query[None] if one_dimensional else query
    if not table.batched:
        rows = [0] * len(query)
    elif indices is None:
        rows = list(range(len(query)))
    else:
        rows = indices
        if one_dimensional:
            query = np.broadcast_to(query, (len(rows), query.shape[1]))
    result = []
    for row, points in zip(rows, query):
        left = y[row, 0] if outside == "clamp" else fill
        right = y[row, -1] if outside == "clamp" else fill
        result.append(np.interp(points, x[row], y[row], left=left, right=right))
    result = np.stack(result)
    if one_dimensional and len(result) == 1:
        result = result[0]
    return torch.from_numpy(result)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("outside", ["clamp", "zero", "fill"])
@pytest.mark.parametrize("regime", ["unbatched", "aligned", "selected", "shared"])
def test_prepared_read_only_matches_reference(uniform, learnable, outside, regime):
    table = _table(uniform, learnable, batched=regime != "unbatched")
    query = torch.tensor([-1.0, 0.0, 0.25, 1.25, 3.0, 6.0], dtype=torch.float64)
    indices = None
    if regime == "aligned":
        query = torch.stack((query, query + 0.125))
    elif regime == "selected":
        query = torch.stack((query, query + 0.125, query - 0.125))
        indices = [1, 0, 1]
    elif regime == "shared":
        indices = [1, 0, 1]
    before = copy.deepcopy(table.state_dict())
    identities = {name: id(value) for name, value in table.named_buffers()}
    fill = -0.75 if outside == "fill" else 0.0
    actual = table(query, indices=indices, outside=outside, fill_value=fill)
    expected = _numpy_rows(table, query, indices, outside, fill)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert table._ind_cache is None
    assert identities == {name: id(value) for name, value in table.named_buffers()}
    for name, value in table.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    # Varying shapes must not create a persistent index cache either.
    table(query[..., :1], indices=indices)
    assert table._ind_cache is None


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("exact_clamp", [False, True])
def test_prepared_query_and_ordinate_derivatives(uniform, exact_clamp):
    table = _table(uniform, True, exact_clamp=exact_clamp)
    query = torch.tensor([-0.3, 0.2, 0.3, 1.2, 2.5, 3.4], dtype=torch.float64)
    ordinates = table.y.detach().clone()

    def evaluate(y, q):
        return functional_call(table, {"y": y}, (q,))

    backward = jacrev(evaluate, argnums=(0, 1))(ordinates, query)
    forward = jacfwd(evaluate, argnums=(0, 1))(ordinates, query)
    for actual, expected in zip(forward, backward):
        torch.testing.assert_close(actual, expected)
    x = table._x_search
    ind = (torch.searchsorted(x, query) - 1).clamp(0, len(x) - 2)
    slope = (ordinates[ind + 1] - ordinates[ind]) / (x[ind + 1] - x[ind])
    slope = torch.where((query < x[0]) | (query > x[-1]), 0.0, slope)
    torch.testing.assert_close(backward[1], torch.diag(slope))
    # Every row is an affine combination of ordinates, including clamp endpoints.
    torch.testing.assert_close(backward[0].sum(-1), torch.ones_like(query))
    assert backward[0].abs().sum() > 0
    assert backward[1].abs().sum() > 0
    _, tangent = jvp(
        evaluate,
        (ordinates, query),
        (torch.ones_like(ordinates), torch.ones_like(query)),
    )
    torch.testing.assert_close(tangent, 1 + slope)


@pytest.mark.parametrize("uniform", [False, True])
def test_prepared_repeated_forwards_preserve_saved_gradient_indices(uniform):
    table = _table(uniform, True)
    first = torch.tensor([0.2, 1.3, 2.2], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([2.7, 0.15, 1.2], dtype=torch.float64, requires_grad=True)
    one = table(first)
    two = table(second)
    gradients = torch.autograd.grad(one.sum() + 2 * two.sum(), (table.y, first, second))

    def evaluate(y, first, second):
        return (
            functional_call(table, {"y": y}, (first,)).sum()
            + 2 * functional_call(table, {"y": y}, (second,)).sum()
        )

    expected = jacrev(evaluate, argnums=(0, 1, 2))(
        table.y.detach(), first.detach(), second.detach()
    )
    for actual, wanted in zip(gradients, expected):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_prepared_vmap_and_empty_vmap(uniform, learnable, batched):
    table = _table(uniform, learnable, batched=batched)
    query = torch.tensor([[0.2, 0.3, 1.2], [0.1, 1.3, 2.4]], dtype=torch.float64)
    if batched:
        query = torch.stack((query, query + 0.125), dim=1)
    expected = torch.stack([table(points) for points in query])
    torch.testing.assert_close(vmap(table)(query), expected)
    empty = vmap(table)(query[:0])
    assert empty.shape == expected[:0].shape
    assert table._ind_cache is None


@pytest.mark.parametrize("uniform", [False, True])
def test_prepared_mapped_indices_and_ordinates(uniform):
    table = _table(uniform, True, batched=True)
    queries = torch.tensor([[0.2, 1.2], [0.3, 2.3]], dtype=torch.float64)
    indices = torch.tensor([[1, 0], [0, 1]])
    ordinates = torch.stack((table.y.detach(), table.y.detach() + 0.125))

    def evaluate(y, query, rows):
        return functional_call(table, {"y": y}, (query,), {"indices": rows})

    expected = torch.stack(
        [
            evaluate(y, query, rows)
            for y, query, rows in zip(ordinates, queries, indices)
        ]
    )
    actual = vmap(evaluate)(ordinates, queries, indices)
    torch.testing.assert_close(actual, expected)
    assert vmap(evaluate)(ordinates[:0], queries[:0], indices[:0]).shape == (0, 2, 2)
    derivative = jacfwd(evaluate, argnums=0)(ordinates[0], queries[0], indices[0])
    torch.testing.assert_close(
        derivative.sum((-1, -2)), torch.ones((2, 2), dtype=queries.dtype)
    )


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_prepared_fullgraph_and_explicit_output(uniform, learnable, backend):
    table = _table(uniform, learnable, batched=True)
    query = torch.tensor([0.2, 1.2, 3.2], dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([1, 0, 1])
    evaluate = torch.compile(table, backend=backend, fullgraph=True)
    expected = table(query, indices=indices)
    actual = evaluate(query, indices=indices)
    torch.testing.assert_close(actual, expected)
    differentiable = (query, table.y) if learnable else (query,)
    gradients = torch.autograd.grad(actual.sum(), differentiable)
    wanted = torch.autograd.grad(expected.sum(), differentiable)
    for gradient, reference in zip(gradients, wanted):
        torch.testing.assert_close(gradient, reference)
    with pytest.raises((ValueError, RuntimeError)):
        evaluate(query, indices=torch.tensor([-1, 0, 1]))
    output = torch.empty_like(expected)
    assert table(query, indices=indices, out=output) is output
    torch.testing.assert_close(output, expected)
    with pytest.raises(ValueError, match="out has shape"):
        table(query, indices=indices, out=output[:1])


@pytest.mark.parametrize("uniform", [False, True])
def test_prepared_empty_queries_and_index_validation(uniform):
    table = _table(uniform, True, batched=True)
    assert table(torch.empty(2, 0, dtype=torch.float64)).shape == (2, 0)
    assert table(torch.empty(0, dtype=torch.float64), indices=1).shape == (0,)
    assert table(torch.empty(0, dtype=torch.float64), indices=[1, 0]).shape == (2, 0)
    for indices in ([-1], [2], torch.tensor([-1]), torch.tensor([2])):
        with pytest.raises(ValueError, match="indices values must be"):
            table(torch.zeros(1, dtype=torch.float64), indices=indices)
    with pytest.raises(ValueError, match="indices values must be"):
        vmap(lambda rows: table(torch.zeros(1, dtype=torch.float64), indices=rows))(
            torch.tensor([[0], [2]])
        )
    with pytest.raises(ValueError, match="non-empty"):
        table(torch.zeros(1, dtype=torch.float64), indices=[])
    with pytest.raises(ValueError, match="length 3"):
        table(torch.zeros(3, 1, dtype=torch.float64), indices=[0, 1])
    with pytest.raises(ValueError, match="1D sequence"):
        table(torch.zeros(1, dtype=torch.float64), indices=[[0]])
    unbatched = _table(uniform, False)
    with pytest.raises(ValueError, match="only contain 0"):
        unbatched(torch.zeros(1, dtype=torch.float64), indices=[1])
