"""Explicit CUDA graph replay through Dendra's public compiled runner."""

from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_scalar_tree import _morphology

DT = 0.125
STEPS = 4
CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


def _cpu_population():
    with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float32, IMEM=False):
        model = dn.SingleCompartment(N=1, C=1, v_init=-60.0)
        model.initialize(DT)
    return model


@pytest.mark.cpu
@pytest.mark.parametrize("factory", (dn.compile, dn.func.CompiledRunner))
def test_cuda_graph_runner_requires_cuda_population(factory):
    with pytest.raises(ValueError, match=r"(?i)CUDA"):
        factory(_cpu_population(), dt=DT, steps=STEPS, cuda_graphs=True)


@pytest.mark.cpu
def test_cuda_graph_runner_rejects_network():
    network = dn.Network({"cell": _cpu_population()})
    with pytest.raises(ValueError, match="Population"):
        dn.compile(network, dt=DT, steps=STEPS, cuda_graphs=True)


@pytest.mark.cpu
@pytest.mark.parametrize("value", (0, None, "true"))
def test_cuda_graph_runner_requires_boolean_option(value):
    with pytest.raises(TypeError, match="cuda_graphs"):
        dn.compile(_cpu_population(), dt=DT, steps=STEPS, cuda_graphs=value)


def _cuda_tree():
    with dn.ctx(JIT=0, DEVICE="cuda", DTYPE=torch.float32, REQUIRE_GRAD=1, IMEM=False):
        model = dn.Tree.from_morphology(
            _morphology(), N=1, v_init=-62.0, integrator=dn.dhs(threads=32, imem=False)
        )
        model.insert(pas, g=3e-4, e=-71.0)
        model.initialize()
        model.train()
    return model


@pytest.mark.cuda
@CUDA_ONLY
def test_cuda_graph_runner_rejects_unsupported_execution_options():
    model = _cuda_tree()
    for options in (
        {"execution": "while"},
        {"backend": "eager"},
        {"disable": True},
        {"checkpoint_every": 2, "inference": False},
    ):
        with pytest.raises(ValueError, match=r"(?i)CUDA"):
            dn.compile(model, dt=DT, steps=STEPS, cuda_graphs=True, **options)


def _track_replay(monkeypatch):
    replays = []
    original = torch.cuda.CUDAGraph.replay

    def replay(graph):
        result = original(graph)
        replays.append(id(graph))
        return result

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
    return replays


def _runner(*, training):
    return dn.compile(
        _cuda_tree(),
        dt=DT,
        steps=STEPS,
        chunk_steps=STEPS,
        cuda_graphs=True,
        inference=not training,
        fullgraph=True,
        options={"compile_threads": 1},
        callbacks={"trace": dn.Recorder("v")} if training else None,
    )


def _call(run, runner, variant, *, training):
    state = torch.utils._pytree.tree_map_only(
        torch.Tensor, torch.clone, runner.initial_state
    )
    initial_v = (state["integrator"]["v"] + 0.75 * variant).requires_grad_(training)
    state["integrator"]["v"] = initial_v
    state["control"]["duration_remainder"].fill_(variant * DT / 8)
    gain = initial_v.new_tensor(0.7 + 0.25 * variant, requires_grad=training)
    profile = torch.linspace(
        0.1,
        1.0,
        STEPS * initial_v.numel(),
        device=initial_v.device,
        dtype=initial_v.dtype,
    )
    pulse = profile.reshape(STEPS, *initial_v.shape) * (0.3 * dn.units.nA)
    result, auxiliary = run(
        state,
        inputs=dn.func.Inputs(intra=pulse * gain),
        duration=(STEPS + variant / 8) * DT,
    )
    remainder = result["control"]["duration_remainder"]
    assert remainder.device.type == "cpu"
    assert remainder.dtype == torch.float64
    assert remainder.item() == pytest.approx(variant * DT / 4)
    voltage = result["integrator"]["v"]
    assert voltage.device.type == "cuda"
    observations = (voltage, result["clock"]["t"], remainder)
    if not training:
        return observations, None

    trace = auxiliary["callbacks"]["trace"]["v"]
    assert trace.shape[0] == STEPS + 1
    weights = torch.linspace(
        0.2, 1.5, trace.shape[0], device=voltage.device, dtype=voltage.dtype
    )
    if variant:
        weights = weights.flip(0)
    weights = weights.reshape(-1, *([1] * voltage.ndim))
    loss = (trace.square() * weights).mean() + (0.125 + variant) * voltage.mean()
    return (*observations, trace), partial(torch.autograd.grad, loss, (initial_v, gain))


def _eager(runner):
    return partial(
        dn.func.run,
        runner.functional.bind(runner.parameters, constants=runner.constants),
        callbacks=runner.callbacks,
        host_span_steps=STEPS,
    )


def _close(actual, expected, *, exact=False):
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            value, reference, rtol=0 if exact else 2e-4, atol=0 if exact else 3e-6
        )


@pytest.mark.cuda
@pytest.mark.slow
@CUDA_ONLY
def test_cuda_graph_runner_replays_changed_inference_without_callbacks(monkeypatch):
    replays = _track_replay(monkeypatch)
    with torch_compiler_warning_context(), torch.no_grad():
        runner = _runner(training=False)
        eager = _eager(runner)
        first, _ = _call(runner, runner, 0, training=False)
        snapshot = tuple(value.clone() for value in first)

        before = len(replays)
        changed, _ = _call(runner, runner, 1, training=False)
        assert len(replays) > before
        captured_graph = replays[-1]
        changed_snapshot = tuple(value.clone() for value in changed)
        before = len(replays)
        again, _ = _call(runner, runner, 2, training=False)
        assert captured_graph in replays[before:]
        _close(first, snapshot, exact=True)
        _close(changed, changed_snapshot, exact=True)
        assert not torch.equal(first[0], changed[0])
        for variant, got in enumerate((first, changed, again)):
            expected, _ = _call(eager, runner, variant, training=False)
            _close(got, expected)


@pytest.mark.cuda
@pytest.mark.slow
@CUDA_ONLY
def test_cuda_graph_runner_training_overlapping_forwards_and_gradients(monkeypatch):
    replays = _track_replay(monkeypatch)
    with torch_compiler_warning_context(), torch.enable_grad():
        runner = _runner(training=True)
        eager = _eager(runner)
        first, first_backward = _call(runner, runner, 0, training=True)
        first_grads = first_backward()
        snapshot = tuple(value.detach().clone() for value in (*first, *first_grads))

        before = len(replays)
        left, left_backward = _call(runner, runner, 1, training=True)
        assert len(replays) > before
        captured_forward = replays[-1]
        pending = tuple(value.detach().clone() for value in left)
        right, right_backward = _call(runner, runner, 2, training=True)
        _close(left, pending, exact=True)
        before = len(replays)
        right_grads = right_backward()
        left_grads = left_backward()
        assert len(replays) > before
        captured_backward = replays[-1]
        held_gradients = tuple(value.clone() for value in left_grads)
        before = len(replays)
        again, again_backward = _call(runner, runner, 3, training=True)
        assert captured_forward in replays[before:]
        before = len(replays)
        again_grads = again_backward()
        assert captured_backward in replays[before:]
        _close((*first, *first_grads), snapshot, exact=True)
        _close(left, pending, exact=True)
        _close(left_grads, held_gradients, exact=True)

        before = len(replays)
        _, retained_backward = _call(runner, runner, 1, training=True)
        assert captured_forward in replays[before:]
        retained_gradients = retained_backward(retain_graph=True)
        before = len(replays)
        _, overlapping_backward = _call(runner, runner, 2, training=True)
        overlapping_backward()
        assert len(replays) == before
        _close(retained_gradients, left_grads)
        _close(retained_backward(), retained_gradients, exact=True)
        assert captured_backward in replays[before:]

        for variant, (value, gradients) in enumerate(
            (
                (first, first_grads),
                (left, left_grads),
                (right, right_grads),
                (again, again_grads),
            )
        ):
            expected, backward = _call(eager, runner, variant, training=True)
            _close(value, expected)
            _close(gradients, backward())
            for gradient in gradients:
                assert torch.isfinite(gradient).all()
                assert torch.count_nonzero(gradient)
