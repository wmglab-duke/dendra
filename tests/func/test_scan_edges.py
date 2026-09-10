"""Scheduling, graph size and transform boundaries of opt-in scan execution."""

import warnings
from functools import partial

import pytest
import torch
from test_compiled_chunk import _assert_tree_close, _model
from test_runner_stimulation_clock import DT, STEPS, _case

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

pytestmark = pytest.mark.skipif(
    torch.__version__.split("+")[0] != "2.14.0",
    reason="scan compatibility is validated against PyTorch 2.14.0",
)


@pytest.fixture(autouse=True)
def _known_compiler_warnings():
    with torch_compiler_warning_context(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated.*",
            category=FutureWarning,
            module=r"torch\.jit\._script",
        )
        yield


@pytest.mark.parametrize("stimulus_kind", ["intra", "extra"])
def test_scan_preserves_atomic_and_runner_waveform_clocks(stimulus_kind):
    functional, tensors, parameters, prepared, amplitude = _case(stimulus_kind)
    atomic, _ = functional.rollout(parameters, prepared, tensors.state, steps=STEPS)
    expected = tensors.state
    for _ in range(STEPS):
        expected, _ = functional.step(parameters, prepared, expected)
    expected_gradient = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), amplitude, retain_graph=True
    )[0]
    assert torch.isfinite(expected_gradient).all()
    assert expected_gradient.abs().item() > 0
    chunk = functional.compile_rollout_chunk(
        STEPS, execution="scan", backend="aot_eager"
    )
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    with torch_compiler_warning_context():
        actual_atomic, _ = chunk(parameters, prepared, tensors.state)
        _assert_tree_close(actual_atomic, atomic)
        for runner in (dn.func.run, dn.func.longrun, dn.func.longrun_checkpointed):
            options = {} if runner is dn.func.run else {"chunklength": 5}
            actual, auxiliary = runner(
                functional,
                partial(chunk, parameters, prepared),
                tensors.state,
                tstop=STEPS * DT,
                callbacks=callbacks,
                **options,
            )
            gradient = torch.autograd.grad(
                actual["integrator"]["v"].square().mean(), amplitude, retain_graph=True
            )[0]
            assert torch.isfinite(gradient).all()
            assert torch.isfinite(actual["integrator"]["v"]).all()
            torch.testing.assert_close(
                actual["integrator"]["v"], expected["integrator"]["v"]
            )
            torch.testing.assert_close(gradient, expected_gradient)
            trace = auxiliary["callbacks"]["trace"]
            assert trace["v"].shape[0] == STEPS + 1
            torch.testing.assert_close(trace["t"][-1], expected["clock"]["t"])


def test_scan_graph_size_does_not_grow_with_horizon():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    captures = []

    def backend(graph, _examples):
        captures.append(graph)
        return graph.forward

    with torch.no_grad(), torch_compiler_warning_context():
        for steps in (2, 9):
            chunk = functional.compile_rollout_chunk(
                steps, execution="scan", backend=backend
            )
            actual, _ = chunk(tensors.parameters, prepared, tensors.state)
            expected, _ = functional.rollout(
                tensors.parameters, prepared, tensors.state, steps=steps
            )
            _assert_tree_close(actual, expected)
    assert len(captures) == 2
    for graph in captures:
        assert sum(str(node.target) == "scan" for node in graph.graph.nodes) == 1
    assert [
        len(list(module.graph.nodes))
        for module in captures[0].modules()
        if isinstance(module, torch.fx.GraphModule)
    ] == [
        len(list(module.graph.nodes))
        for module in captures[1].modules()
        if isinstance(module, torch.fx.GraphModule)
    ]


@pytest.mark.parametrize("transform", ["grad", "jacrev", "jacfwd", "vmap"])
def test_scan_chunk_transform_fallback_does_not_enter_scan(transform):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def unexpected_backend(_graph, _examples):
        raise AssertionError("active transforms must use the eager recurrence")

    chunk = functional.compile_rollout_chunk(
        2, execution="scan", backend=unexpected_backend
    )

    def loss(voltage, *, compiled):
        state = dict(tensors.state)
        state["integrator"] = dict(state["integrator"], v=voltage)
        if compiled:
            final, _ = chunk(tensors.parameters, prepared, state)
        else:
            final, _ = functional.rollout(tensors.parameters, prepared, state, steps=2)
        return final["integrator"]["v"].square().mean()

    voltage = tensors.state["integrator"]["v"].detach()
    if transform == "vmap":
        voltage = torch.stack((voltage, voltage + 1))
    operator = getattr(torch.func, transform)
    expected = operator(partial(loss, compiled=False))(voltage)
    actual = operator(partial(loss, compiled=True))(voltage)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not chunk._compiled._kernels
