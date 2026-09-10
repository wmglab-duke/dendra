"""First-order guard behavior independent of the optional scan compatibility patch."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import FunctionalizationError
from dendra.func._scan_autograd import guard_scan_outputs


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _assert_finite(*values):
    for value in values:
        assert torch.isfinite(value).all()


def _make_chunk(steps, backend):
    def core(state, prepared):
        for index in range(steps):
            state = torch.tanh(0.81 * state + (index + 1) * 0.05 * prepared)
        return state

    return (
        core
        if backend == "eager"
        else torch.compile(core, backend=backend, fullgraph=True)
    )


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("nonlinear_loss", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_scan_guard_preserves_first_order_and_repeated_backward(
    backend, nonlinear_loss, checkpointed
):
    chunks = [_make_chunk(steps, backend) for steps in (3, 2)]
    reference_chunks = [_make_chunk(steps, "eager") for steps in (3, 2)]

    def run(parameter, guarded):
        # Preparation remains outside the guarded/compiled region, as it does
        # when the simulation prepares parameter-dependent geometry once.
        prepared = 0.3 + 0.15 * parameter.sin()
        state = parameter.new_tensor([-0.13, 0.07, 0.19])
        for chunk in chunks if guarded else reference_chunks:

            def execute(state, prepared, chunk=chunk):
                value = chunk(state, prepared)
                return guard_scan_outputs(value) if guarded else value

            state = (
                checkpoint(execute, state, prepared, use_reentrant=False)
                if checkpointed and guarded
                else execute(state, prepared)
            )
        return state.square().sum() if nonlinear_loss else state.sum()

    expected_parameter = torch.tensor(
        [0.19, -0.11, 0.23], dtype=torch.float64, requires_grad=True
    )
    actual_parameter = expected_parameter.detach().clone().requires_grad_()
    expected = run(expected_parameter, False)
    actual = run(actual_parameter, True)
    expected_gradient = torch.autograd.grad(expected, expected_parameter)[0]
    actual_gradient = torch.autograd.grad(actual, actual_parameter, retain_graph=True)[
        0
    ]
    repeated_gradient = torch.autograd.grad(actual, actual_parameter)[0]
    _assert_finite(
        expected, actual, expected_gradient, actual_gradient, repeated_gradient
    )
    assert expected_gradient.abs().max() > 0
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(
        repeated_gradient, expected_gradient, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("nonlinear_loss", [False, True])
def test_scan_guard_rejects_graph_recording_with_nonlinear_preparation(
    backend, nonlinear_loss
):
    parameter = torch.tensor(
        [0.19, -0.11, 0.23], dtype=torch.float64, requires_grad=True
    )
    prepared = 0.3 + 0.15 * parameter.sin()
    state = parameter.new_tensor([-0.13, 0.07, 0.19])
    actual = guard_scan_outputs(_make_chunk(3, backend)(state, prepared))
    loss = actual.square().sum() if nonlinear_loss else actual.sum()
    with pytest.raises(FunctionalizationError, match="use fmodel.rollout"):
        torch.autograd.grad(loss, parameter, create_graph=True)


def test_scan_guard_preserves_non_differentiable_leaves():
    parameter = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    constant, counter, flag = torch.tensor(3.0), torch.tensor(4), torch.tensor(True)
    tree = {
        "state": parameter.sin(),
        "constant": constant,
        "counter": counter,
        "flag": flag,
        "none": None,
    }
    result = guard_scan_outputs(tree)
    assert result["constant"] is constant
    assert result["counter"] is counter
    assert result["flag"] is flag
    assert result["none"] is None
    gradient = torch.autograd.grad(result["state"].sum(), parameter)[0]
    _assert_finite(result["state"], gradient)
    torch.testing.assert_close(gradient, parameter.cos())
    with torch.no_grad():
        assert guard_scan_outputs(tree) is tree


def test_scan_guard_rejects_outer_compile():
    def wrapped(value):
        return guard_scan_outputs(value.sin())

    compiled = torch.compile(wrapped, backend="aot_eager", fullgraph=True)
    value = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    # Dynamo wraps the intentional user exception during fullgraph tracing.
    with pytest.raises(Exception, match="cannot be wrapped|Observed exception"):
        compiled(value)
