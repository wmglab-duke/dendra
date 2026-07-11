"""Correctness contracts for :func:`dendra.opt.sanitize_grad`."""

from __future__ import annotations

import pytest
import torch

from dendra.opt import sanitize_grad


def _sgd(*parameter_groups):
    return torch.optim.SGD(
        [{"params": list(parameters)} for parameters in parameter_groups],
        lr=0.1,
    )


def test_sanitize_grad_zeros_every_nonfinite_value_without_replacing_grad():
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    untouched = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    optimizer = _sgd([parameter, untouched])
    parameter.grad = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 3.0],
        dtype=torch.float64,
    )
    original_gradient = parameter.grad

    result = sanitize_grad(optimizer)

    assert result is None
    assert parameter.grad is original_gradient
    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([0.0, 0.0, 0.0, 3.0], dtype=torch.float64),
    )
    assert untouched.grad is None


def test_sanitize_grad_clips_one_global_norm_across_parameter_groups():
    first = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = _sgd([first], [second])
    first.grad = torch.tensor(3.0, dtype=torch.float64)
    second.grad = torch.tensor(4.0, dtype=torch.float64)

    sanitize_grad(optimizer, clip_norm=1.0)

    torch.testing.assert_close(
        first.grad,
        torch.tensor(0.6, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    torch.testing.assert_close(
        second.grad,
        torch.tensor(0.8, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(torch.stack([first.grad, second.grad])),
        torch.tensor(1.0, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )


def test_sanitize_grad_applies_value_clipping_before_norm_clipping():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = _sgd([parameter])
    parameter.grad = torch.tensor([-6.0, 8.0], dtype=torch.float64)

    sanitize_grad(optimizer, clip_value=4.0, clip_norm=2.0)

    expected = torch.tensor([-(2**-0.5), 2**-0.5], dtype=torch.float64) * 2.0
    torch.testing.assert_close(parameter.grad, expected, rtol=1e-6, atol=1e-8)


def test_sanitize_grad_accepts_zero_limits_and_no_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = _sgd([parameter])

    assert sanitize_grad(optimizer, clip_norm=0, clip_value=0) is None
    assert parameter.grad is None

    parameter.grad = torch.tensor([2.0, -3.0])
    sanitize_grad(optimizer, clip_norm=0, clip_value=0)
    torch.testing.assert_close(parameter.grad, torch.zeros(2))


@pytest.mark.parametrize("name", ["clip_norm", "clip_value"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_sanitize_grad_rejects_invalid_numeric_limits(name, value):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = _sgd([parameter])

    with pytest.raises(ValueError, match=rf"{name} must be finite and non-negative"):
        sanitize_grad(optimizer, **{name: value})


@pytest.mark.parametrize("name", ["clip_norm", "clip_value"])
@pytest.mark.parametrize("value", [True, "1.0", object()])
def test_sanitize_grad_rejects_non_real_limits(name, value):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = _sgd([parameter])

    with pytest.raises(TypeError, match=rf"{name} must be a real number or None"):
        sanitize_grad(optimizer, **{name: value})


def test_sanitize_grad_rejects_sparse_gradients_before_mutating_any_gradient():
    dense = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    embedding = torch.nn.Embedding(4, 2, sparse=True)
    optimizer = _sgd([dense, embedding.weight])
    dense.grad = torch.tensor([float("nan"), 1.0])
    embedding(torch.tensor([1, 1])).sum().backward()
    sparse_values_before = embedding.weight.grad._values().clone()

    with pytest.raises(
        TypeError,
        match=r"supports only dense, strided gradients; found torch\.sparse_coo",
    ):
        sanitize_grad(optimizer, clip_norm=1.0)

    assert torch.isnan(dense.grad[0])
    torch.testing.assert_close(embedding.weight.grad._values(), sparse_values_before)


def test_sanitize_grad_preserves_higher_order_autograd_and_optimizer_use():
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = _sgd([parameter])
    (first_derivative,) = torch.autograd.grad(
        (parameter**3).sum(), parameter, create_graph=True
    )
    parameter.grad = first_derivative
    original_gradient = parameter.grad
    original_grad_fn = parameter.grad.grad_fn

    sanitize_grad(optimizer, clip_value=100.0)

    assert parameter.grad is original_gradient
    assert parameter.grad.grad_fn is original_grad_fn
    (second_derivative,) = torch.autograd.grad(parameter.grad.sum(), parameter)
    torch.testing.assert_close(
        second_derivative, torch.tensor([12.0], dtype=torch.float64)
    )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (parameter.square().sum()).backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
