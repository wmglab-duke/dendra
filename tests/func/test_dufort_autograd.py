"""Authored current derivatives survive tensor-program capture and scan AD."""

import copy
import gc
import weakref

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from dendra.func._dufort_autograd import _PROGRAMS, capture_autograd_program
from dendra.func._scan_inductor import _compile_scan


class _AuthoredCurrent(torch.autograd.Function):
    @staticmethod
    def forward(value):
        return 2 * value

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, tangent):
        return 3 * ctx.saved_tensors[0] * tangent


class _TangentLayoutCurrent(torch.autograd.Function):
    @staticmethod
    def forward(value):
        return 2 * value

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.shape = inputs[0].shape

    @staticmethod
    def backward(ctx, tangent):
        return (3 * tangent.contiguous().view(-1)).view(ctx.shape)


def _evaluate(value, weight):
    return _AuthoredCurrent.apply(value) * weight.square()


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_dufort_authored_vjp_recomputes_live_activations_through_scan(backend):
    from torch._higher_order_ops.scan import scan

    examples = (
        torch.ones(2, dtype=torch.float64, requires_grad=True),
        torch.tensor(1.1, dtype=torch.float64, requires_grad=True),
    )
    program = capture_autograd_program(_evaluate, examples)

    def step(value, weight):
        return torch.cond(
            torch.tensor(True),
            lambda value, weight: program(value, weight).clone(),
            lambda value, weight: program(value, weight).clone() + 1,
            (value, weight),
        )

    graph = make_fx(step, pre_dispatch=True)(*examples)

    def rollout(value, weight):
        def body(carry, sample):
            result = graph(carry, weight)
            return result.clone(), result.clone()

        return scan(body, value, torch.arange(3))[0]

    compiled = _compile_scan(rollout, {"backend": backend, "fullgraph": True})
    for initial, gain in ((0.7, 0.9), (1.3, 1.2)):
        inputs = (
            torch.full((2,), initial, dtype=torch.float64, requires_grad=True),
            torch.tensor(gain, dtype=torch.float64, requires_grad=True),
        )
        expected = inputs[0]
        for _ in range(3):
            expected = _evaluate(expected, inputs[1])
        expected_gradients = torch.autograd.grad(expected.sum(), inputs)
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), inputs), expected_gradients
        )


def test_dufort_authored_program_retains_unused_and_nondifferentiable_contracts():
    def evaluate(value, weight, unused, indices):
        return {
            "current": _evaluate(value, weight),
            "detached": value.detach(),
            "indices": indices.clone(),
        }

    inputs = (
        torch.ones(2, dtype=torch.float64, requires_grad=True),
        torch.tensor(0.7, dtype=torch.float64, requires_grad=True),
        torch.ones(2, dtype=torch.float64, requires_grad=True),
        torch.arange(2),
    )
    program = capture_autograd_program(evaluate, inputs)
    actual = program(*inputs)
    assert not actual["detached"].requires_grad
    assert not actual["indices"].requires_grad
    gradients = torch.autograd.grad(
        actual["current"].sum(), inputs[:3], allow_unused=True
    )
    assert gradients[2] is None
    torch.testing.assert_close(
        gradients[:2],
        (torch.full_like(inputs[0], 1.47), torch.tensor(5.6, dtype=torch.float64)),
    )


def test_dufort_authored_program_registry_does_not_own_plans_or_sample_tensors():
    inputs = (torch.ones(2, requires_grad=True), torch.tensor(1.1, requires_grad=True))
    references = tuple(weakref.ref(value) for value in inputs)
    program = capture_autograd_program(_evaluate, inputs)
    identifier = program.identifier
    del inputs
    gc.collect()
    assert all(reference() is None for reference in references)
    del program
    gc.collect()
    assert identifier not in _PROGRAMS


def test_dufort_authored_program_copy_survives_original_plan_lifetime():
    inputs = (torch.ones(2, requires_grad=True), torch.tensor(1.1, requires_grad=True))
    program = capture_autograd_program(_evaluate, inputs)
    copied = copy.deepcopy(program)
    del program
    gc.collect()
    actual = copied(*inputs)
    expected = _evaluate(*inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), inputs),
        torch.autograd.grad(expected.sum(), inputs),
    )


def test_dufort_authored_program_normalizes_strided_backward_tangents():
    value = torch.ones(2, 3, requires_grad=True)
    program = capture_autograd_program(_TangentLayoutCurrent.apply, (value,))
    tangent = torch.arange(6.0).reshape(3, 2).t()
    assert not tangent.is_contiguous()
    actual = program(value)
    (gradient,) = torch.autograd.grad(actual, value, tangent)
    torch.testing.assert_close(gradient, 3 * tangent)
