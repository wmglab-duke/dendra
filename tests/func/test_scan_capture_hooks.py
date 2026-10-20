"""Disposable scan joint capture must not enter the caller's save hooks."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from dendra.func import _scan as scan_module
from dendra.func import _scan_compat

pytestmark = pytest.mark.cpu


def test_training_body_capture_preserves_authored_stop_gradients(monkeypatch):
    """Forward-equivalent alias lowering is not an equivalent AD program."""

    class Functional:
        device = torch.device("cpu")
        intra = extra = SimpleNamespace(enabled=False)

        @staticmethod
        def _clone_state_tree(state):
            return torch.utils._pytree.tree_map(lambda value: value.clone(), state)

        @staticmethod
        def _transition_rollout_values(parameters, prepared, state, *args):
            voltage = state["integrator"]["v"]
            soft = voltage.sin()
            updated = voltage.tanh().detach() + parameters["scale"] * (
                soft - soft.detach()
            )
            return {
                "integrator": {"v": updated},
                "control": state["control"],
            }, {}

    captured = []
    make_fx = scan_module.make_fx

    def record_graph(function, **options):
        def capture(*examples):
            graph = make_fx(function, **options)(*examples)
            captured.append((graph, examples, options))
            return graph

        return capture

    monkeypatch.setattr(scan_module, "make_fx", record_graph)
    monkeypatch.setattr(
        scan_module, "_compile_scan", lambda function, *a, **kw: function
    )
    chunk = object.__new__(scan_module._CompiledScanPopulationChunk)
    chunk._functional = Functional()
    chunk._steps = 2
    chunk._options = {}
    chunk._stepwise = False
    chunk._plans = None
    state = {
        "integrator": {"v": torch.tensor([-0.3, 0.2], dtype=torch.float64)},
        "control": {"duration_remainder": torch.tensor(0.0, dtype=torch.float64)},
    }
    chunk._capture(
        {"scale": torch.tensor(0.75, dtype=torch.float64)},
        {},
        state,
        None,
        None,
        {},
        grad_enabled=True,
    )
    ((graph, examples, options),) = captured
    assert options["pre_dispatch"] is True
    assert any(
        node.target is torch.ops.aten.detach.default for node in graph.graph.nodes
    )
    parameters, prepared, state, carries, invariant_carries, sample = examples
    voltage = state["integrator"]["v"].detach().clone().requires_grad_()
    scale = parameters["scale"].detach().clone().requires_grad_()
    state["integrator"]["v"] = voltage
    parameters["scale"] = scale
    (updated, _), _ = graph(
        parameters, prepared, state, carries, invariant_carries, sample
    )
    actual = updated["integrator"]["v"]
    gradients = torch.autograd.grad(actual.sum(), (voltage, scale))
    torch.testing.assert_close(actual, voltage.detach().tanh())
    torch.testing.assert_close(gradients[0], scale.detach() * voltage.detach().cos())
    torch.testing.assert_close(gradients[1], torch.zeros_like(scale))


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
