"""Disposable scan joint capture must not enter the caller's save hooks."""

import importlib

import pytest
import torch

from dendra.func import _scan_compat

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("fail", [False, True], ids=["success", "error"])
def test_joint_capture_isolates_saved_tensor_hooks_and_restores_caller(fail):
    utils = importlib.import_module("torch._higher_order_ops.utils")
    aot = importlib.import_module("torch._functorch.aot_autograd")
    captured = []

    def create_joint(*args, **kwargs):
        def joint(primals, tangents):
            # A real save/unpack pair exercises the same autograd mechanism
            # used while make_fx traces the scan's disposable joint graph.
            output = primals[0].sin()
            if fail:
                raise RuntimeError("injected joint capture error")
            gradient = torch.autograd.grad(
                output, primals[0], grad_outputs=tangents[0]
            )[0]
            captured.append(gradient.detach().clone())
            return [output], [gradient]

        return joint

    primal = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    tangent = torch.tensor([1.3, -0.7], dtype=torch.float64)
    factory = _scan_compat._make_full_output_backward(
        utils, aot.AOTConfig, create_joint
    )
    backward = factory(lambda value: value.sin(), (primal,))
    outer_events = []
    marker = object()

    def pack(tensor):
        outer_events.append("pack")
        return marker, tensor.detach()

    def unpack(saved):
        outer_events.append("unpack")
        saved_marker, tensor = saved
        assert saved_marker is marker
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        if fail:
            with pytest.raises(RuntimeError, match="injected joint capture error"):
                backward(primal, tangent)
            assert not captured
        else:
            actual = backward(primal, tangent)
            expected = primal.detach().cos() * tangent
            torch.testing.assert_close(actual, [expected])
            torch.testing.assert_close(captured, [expected])
        assert outer_events == []

        # Capture isolation is scoped, including on exceptions. A subsequent
        # ordinary runtime save must reach exactly the caller's original hooks.
        runtime_input = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        runtime_output = runtime_input.exp()
        runtime_gradient = torch.autograd.grad(runtime_output, runtime_input)[0]
        assert outer_events == ["pack", "unpack"]
        torch.testing.assert_close(runtime_gradient, runtime_input.detach().exp())
