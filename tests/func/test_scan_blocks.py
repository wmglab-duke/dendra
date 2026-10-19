"""Forward scan blocks preserve per-step observations and training reuse."""

import importlib
import warnings
from dataclasses import dataclass
from functools import partial

import pytest
import torch
from test_runner_compiled_callbacks import _TraceMSE
from test_runner_compiled_chunks import _run
from test_runner_stimulation_clock import STEPS as CLOCK_STEPS
from test_runner_stimulation_clock import _case as _clock_case
from test_runners import (
    DT,
    _assert_tree_close,
    _clone_state,
    _differentiable_case,
    _drives,
    _model,
)
from test_scan_chunk import _assert_gradients

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

pytestmark = pytest.mark.cpu


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


def _graph_backend(graph, _examples):
    return graph.forward


def _forward_mode(mode):
    return torch.inference_mode() if mode == "inference" else torch.no_grad()


@pytest.mark.parametrize("mode", ["no_grad", "inference"])
@pytest.mark.parametrize(
    "steps,presence",
    [
        (1, (False, False)),
        (2, (True, False)),
        (3, (False, True)),
        (4, (True, True)),
        (5, (True, False)),
        (7, (False, True)),
        (9, (True, True)),
        (9, (False, False)),
    ],
)
def test_forward_scan_short_horizons_and_input_presence(mode, steps, presence):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, steps)
    inputs = dn.func.RolloutInput(
        ve=ve if presence[0] else None,
        intra=intra if presence[1] else None,
    )
    original = _clone_state((tensors.state, ve, intra))
    chunk = functional.compile_rollout_chunk(
        steps, execution="scan", backend=_graph_backend
    )
    with _forward_mode(mode):
        expected = functional.rollout(
            tensors.parameters, prepared, tensors.state, inputs, steps=steps
        )
        actual = chunk(tensors.parameters, prepared, tensors.state, inputs)
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    _assert_tree_close((tensors.state, ve, intra), original, rtol=0, atol=0)
    assert not actual[0]["control"]["duration_remainder"].requires_grad


@dataclass(frozen=True)
class _ViewsAndCounters(dn.func.FunctionalCallback):
    """Deliberately return aliases, views, and non-floating reducer carry."""

    @staticmethod
    def _observe(voltage):
        return {"voltage": voltage, "alias": voltage, "view": voltage[..., ::2]}

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "frames": voltage.new_ones((), dtype=torch.int64),
            "seen": (voltage > -60).any(),
            "energy": voltage.square().mean(),
        }, self._observe(voltage)

    def update(self, carry, state, auxiliary):
        # Auxiliary voltage aliases the final state in the authored transition.
        voltage = auxiliary["v"]
        return {
            "frames": carry["frames"] + 1,
            "seen": carry["seen"] | (voltage > -60).any(),
            "energy": carry["energy"] + voltage.square().mean(),
        }, self._observe(state["integrator"]["v"])

    def finalize(self, carry, emissions):
        return {"carry": carry, "samples": emissions}


def _noncontiguous_voltage(state):
    state = _clone_state(state)
    voltage = state["integrator"]["v"]
    voltage = voltage.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not voltage.is_contiguous()
    state["integrator"]["v"] = voltage
    return state


@pytest.mark.parametrize("block", [1, 2, 4])
@pytest.mark.parametrize("runner", ["run", "longrun"])
def test_forward_blocks_preserve_views_mixed_callbacks_and_bounded_tails(
    monkeypatch, block, runner
):
    scan_module = importlib.import_module("dendra.func._scan")
    monkeypatch.setattr(scan_module, "_FORWARD_SCAN_BLOCK", block)
    steps = 11
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = _noncontiguous_voltage(tensors.state)
    ve, intra = _drives(model, steps)
    inputs = dn.func.RolloutInput(ve, intra)
    original = _clone_state((state, ve, intra))
    target = torch.linspace(
        -64.0, -52.0, (steps + 1) * model.v.numel(), dtype=model.dtype()
    ).reshape(steps + 1, *model.shape)
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(["v", "hh.m", "t"]),
            "mse": _TraceMSE(target),
            "views": _ViewsAndCounters(),
            "anomaly": dn.func.AnomalyDetector(),
        }
    )
    chunk = functional.compile_rollout_chunk(
        7, execution="scan", backend=_graph_backend
    )
    with torch.no_grad():
        expected = _run(
            "run",
            functional,
            partial(functional.step, tensors.parameters, prepared),
            state,
            inputs,
            steps=steps,
            callbacks=callbacks,
        )
        # Width seven with a five-step host chunk gives both a block tail and
        # a separate one-step final segment, each with per-step observations.
        actual = _run(
            runner,
            functional,
            partial(chunk, tensors.parameters, prepared),
            state,
            inputs,
            steps=steps,
            chunklength=5,
            callbacks=callbacks,
        )
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    _assert_tree_close((state, ve, intra), original, rtol=0, atol=0)
    results = actual[1]["callbacks"]
    trace, views = results["trace"], results["views"]
    assert trace["v"].shape[0] == steps + 1
    assert views["carry"]["frames"].item() == steps + 1
    assert views["carry"]["frames"].dtype == torch.int64
    assert views["carry"]["seen"].dtype == torch.bool
    assert not views["carry"]["frames"].requires_grad
    assert not views["carry"]["seen"].requires_grad
    torch.testing.assert_close(views["samples"]["voltage"], trace["v"])
    torch.testing.assert_close(views["samples"]["view"], trace["v"][..., ::2])
    saved_alias = views["samples"]["alias"].clone()
    saved_final = actual[0]["integrator"]["v"].clone()
    # Returning one authored tensor in several output leaves must not make a
    # caller's later edits overwrite another recording or final model state.
    views["samples"]["voltage"][0].add_(100)
    torch.testing.assert_close(views["samples"]["alias"], saved_alias, rtol=0, atol=0)
    torch.testing.assert_close(
        actual[0]["integrator"]["v"], saved_final, rtol=0, atol=0
    )


@pytest.mark.parametrize("mode", ["no_grad", "inference"])
@pytest.mark.parametrize("stimulus_kind", ["intra", "extra"])
def test_forward_scan_blocks_sample_bound_pulse_on_each_accepted_clock(
    mode, stimulus_kind
):
    functional, tensors, parameters, prepared, _amplitude = _clock_case(stimulus_kind)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    chunk = functional.compile_rollout_chunk(
        CLOCK_STEPS, execution="scan", backend=_graph_backend
    )
    with _forward_mode(mode):
        expected = dn.func.run(
            functional.bind(parameters, prepared),
            tensors.state,
            steps=CLOCK_STEPS,
            callbacks=callbacks,
        )
        actual = dn.func.run(
            chunk.bind(parameters, prepared),
            tensors.state,
            steps=CLOCK_STEPS,
            callbacks=callbacks,
        )
        atomic, _ = functional.rollout(
            parameters, prepared, tensors.state, steps=CLOCK_STEPS
        )
    _assert_tree_close(actual, expected, rtol=0, atol=0)
    assert actual[1]["callbacks"]["trace"]["v"].shape[0] == CLOCK_STEPS + 1
    # This pulse lies between the float32 vector clock and the repeatedly
    # added accepted clock. A fixture without that difference would miss a
    # block implementation that incorrectly precomputes all waveform times.
    assert not torch.equal(actual[0]["integrator"]["v"], atomic["integrator"]["v"])


def _gradient_evaluation(functional, tensors, ve, intra, chunk=None):
    parameters, constants, state, ve, intra, targets = _differentiable_case(
        tensors, ve, intra
    )
    original = _clone_state((parameters, constants, state, ve, intra))
    prepared = functional.prepare(parameters, constants)
    inputs = dn.func.RolloutInput(ve, intra)
    if chunk is None:
        final, auxiliary = functional.rollout(parameters, prepared, state, inputs)
    else:
        final, auxiliary = chunk(parameters, prepared, state, inputs)
    loss = (
        final["integrator"]["v"].square().mean()
        + 0.1 * final["mechanisms"]["hh"]["m"].square().mean()
        + 0.01 * auxiliary["v"].sin().mean()
    )
    gradients = dict(
        zip(targets, torch.autograd.grad(loss, tuple(targets.values())), strict=True)
    )
    _assert_tree_close(
        (parameters, constants, state, ve, intra), original, rtol=0, atol=0
    )
    return final, auxiliary, gradients


@pytest.mark.parametrize("first_mode", ["no_grad", "inference"])
def test_forward_first_scan_kernel_preserves_later_first_order_gradients(first_mode):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 9)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(9, execution="scan", backend="aot_eager")
    with _forward_mode(first_mode):
        first = chunk(
            tensors.parameters, prepared, tensors.state, dn.func.RolloutInput(ve, intra)
        )
        expected_forward = functional.rollout(
            tensors.parameters, prepared, tensors.state, dn.func.RolloutInput(ve, intra)
        )
    _assert_tree_close(first, expected_forward, rtol=2e-10, atol=2e-11)
    expected = _gradient_evaluation(functional, tensors, ve, intra)
    actual = _gradient_evaluation(functional, tensors, ve, intra, chunk)
    _assert_tree_close(actual[:2], expected[:2], rtol=2e-10, atol=2e-11)
    _assert_gradients(actual[2], expected[2])
    # Training must not replace the forward specialization with a graph that
    # retained an earlier preparation or autograd tape.
    with torch.no_grad():
        repeated = chunk(
            tensors.parameters, prepared, tensors.state, dn.func.RolloutInput(ve, intra)
        )
    _assert_tree_close(repeated, expected_forward, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("block", [1, 2, 4])
def test_forward_scan_graph_size_is_bounded_for_equal_remainders(monkeypatch, block):
    scan_module = importlib.import_module("dendra.func._scan")
    monkeypatch.setattr(scan_module, "_FORWARD_SCAN_BLOCK", block)
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    with torch.no_grad():
        for steps in (12, 20):
            chunk = functional.compile_rollout_chunk(
                steps, execution="scan", backend=backend
            )
            actual = chunk(tensors.parameters, prepared, tensors.state)
            expected = functional.rollout(
                tensors.parameters, prepared, tensors.state, steps=steps
            )
            _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    assert len(graphs) == 2
    sizes = []
    for graph in graphs:
        assert sum(str(node.target) == "scan" for node in graph.graph.nodes) == 1
        sizes.append(
            [
                len(list(module.graph.nodes))
                for module in graph.modules()
                if isinstance(module, torch.fx.GraphModule)
            ]
        )
    assert sizes[0] == sizes[1]


@pytest.mark.cpu
@pytest.mark.slow
def test_inductor_forward_blocks_record_every_hh_state_and_reuse_for_gradients():
    from test_scan_hh_validation import _assert_close, _case, _evaluate

    case = _case("sc", torch.float64, 1)
    oracle = _case("sc", torch.float64, 1, legacy=True)
    chunk = case.functional.compile_rollout_chunk(9, execution="scan")
    with torch.no_grad():
        expected = _evaluate(oracle, 9, gradients=False)
        actual = _evaluate(case, 9, kernel=chunk, gradients=False)
    _assert_tree_close(actual.trace, expected.trace, rtol=2e-10, atol=2e-9)
    _assert_tree_close(actual.diagnostics, expected.diagnostics, rtol=2e-10, atol=2e-9)
    # The same public chunk first used for blocked inference must still expose
    # all eight first-order sources when its ordinary training path is entered.
    expected_gradients = _evaluate(oracle, 9)
    actual_gradients = _evaluate(case, 9, kernel=chunk)
    _assert_close(actual_gradients, expected_gradients, torch.float64)
