"""Compiled convenience runners retain pure threshold-callback semantics."""

from __future__ import annotations

import pytest
import torch
from test_compiled_chunk import (
    DT,
    GNABAR,
    _assert_tree_close,
    _clone_mapping,
    _clone_state,
    _model,
)
from test_recorder_while_loop import _case

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import Recorder as ImperativeRecorder
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE

pytestmark = [
    pytest.mark.cpu,
]

TRACE = ("v", "hh.m", "hh.h", "hh.n", "t")


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _thresholds(*, start=2 * DT, end=12 * DT):
    options = dict(
        threshold=-60.0,
        node_check=(0, 1, 3, -1),
        t_start_check=start,
        t_end_check=end,
    )
    return {
        "raster": dn.func.Raster(**options),
        "count": dn.func.APCount(**options),
        "active": dn.func.Active(**options),
        "inactive": dn.func.Active(**options, inv=True),
        "active_nodes": dn.func.ActiveAL(**options, at_least=2),
        "active_events": dn.func.ActiveALCount(**options, at_least=2),
        "partitioned": dn.func.ActiveAL(
            **options,
            at_least=2,
            partition=(2, 2),
        ),
        "partitioned_events": dn.func.ActiveALCount(
            **options,
            at_least=2,
            partition=(2, 2),
            inv=True,
        ),
        "anomaly": dn.func.AnomalyDetector(),
    }


def _inputs(model, steps):
    # Repeated reversals exercise crossing carry rather than just first-step
    # detection. The finite window is deliberately not aligned with chunks.
    signs = torch.where(torch.arange(steps) % 4 < 2, 1.0, -1.0)
    intra = signs.to(dtype=model.dtype()).reshape(steps, 1, 1)
    intra = intra.expand(steps, *model.shape).clone() * 1.0e-8
    return dn.func.RolloutInput(intra=intra)


def _slice_inputs(inputs, begin, end):
    return dn.func.RolloutInput(
        ve=None if inputs.ve is None else inputs.ve[begin:end],
        intra=None if inputs.intra is None else inputs.intra[begin:end],
    )


def _eager(runner, state, *, steps, specifications, parameters=None, inputs=None):
    parameters = runner.parameters if parameters is None else parameters
    prepared = runner.functional.prepare(parameters, runner.constants)
    callbacks = runner.functional.make_callbacks(specifications)
    return dn.func.run(
        runner.functional.bind(parameters, prepared),
        state,
        steps=steps,
        callbacks=callbacks,
        inputs=inputs,
    )


def _ordered(value):
    # Named callback collections are mappings; insertion order must not be
    # part of the comparison contract.
    if isinstance(value, dict):
        return {name: _ordered(value[name]) for name in sorted(value)}
    if isinstance(value, tuple):
        return tuple(_ordered(part) for part in value)
    if isinstance(value, list):
        return [_ordered(part) for part in value]
    return value


def _close_result(actual, expected):
    _assert_tree_close(actual[0], expected[0], rtol=2.0e-8, atol=2.0e-9)
    _assert_tree_close(
        _ordered(dict(actual[1]["callbacks"])),
        _ordered(dict(expected[1]["callbacks"])),
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def _assert_reductions(outputs):
    counts = outputs["count"]
    assert counts.dtype == torch.int64
    assert outputs["raster"].dtype == torch.bool
    torch.testing.assert_close(counts, outputs["raster"].sum(0))
    torch.testing.assert_close(outputs["active"], (counts != 0).any(-1))
    torch.testing.assert_close(outputs["inactive"], ~outputs["active"])
    torch.testing.assert_close(outputs["active_nodes"], (counts != 0).sum(-1) >= 2)
    torch.testing.assert_close(outputs["active_events"], counts.sum(-1) >= 2)
    partitions = counts.split((2, 2), dim=-1)
    torch.testing.assert_close(
        outputs["partitioned"],
        torch.stack([(part != 0).sum(-1) >= 2 for part in partitions], dim=-1),
    )
    torch.testing.assert_close(
        outputs["partitioned_events"],
        ~torch.stack([part.sum(-1) >= 2 for part in partitions], dim=-1),
    )
    for name in outputs:
        if name != "trace":
            assert not outputs[name].requires_grad


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_named_threshold_callbacks_match_eager_full_traces_and_restart(execution):
    model = _model()
    source = (model.v.detach().clone(), model.t.detach().clone())
    specifications = {"trace": dn.func.Recorder(TRACE), **_thresholds()}
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=9,
        execution=execution,
        callbacks=specifications,
        backend="eager",
    )
    state_before = _clone_state(runner.initial_state)
    parameters_before = _clone_mapping(runner.parameters)
    inputs = _inputs(model, 9)
    input_before = inputs.intra.clone()
    with torch.no_grad():
        expected = _eager(
            runner,
            state_before,
            steps=9,
            inputs=inputs,
            specifications={"trace": dn.func.Recorder(TRACE), **_thresholds()},
        )
    actual = runner(inputs=inputs)
    saved = (actual[0], dict(actual[1]["callbacks"]))
    saved = torch.utils._pytree.tree_map(lambda value: value.clone(), saved)
    repeated = runner(inputs=inputs)
    _close_result(actual, expected)
    _close_result(repeated, expected)
    _assert_tree_close((actual[0], dict(actual[1]["callbacks"])), saved)
    _assert_tree_close(runner.initial_state, state_before)
    _assert_tree_close(dict(runner.parameters), parameters_before)
    torch.testing.assert_close(inputs.intra, input_before, rtol=0, atol=0)
    torch.testing.assert_close(model.v, source[0], rtol=0, atol=0)
    torch.testing.assert_close(model.t, source[1], rtol=0, atol=0)
    assert actual[1]["callbacks"]["trace"]["v"].shape[0] == 10
    _assert_reductions(actual[1]["callbacks"])
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("kind", ("reducers", "raster"))
def test_callbacks_without_recorder_support_reducers_and_poststep_emissions(
    execution, kind
):
    model = _model()
    specifications = _thresholds(start=0, end=None)
    if kind == "raster":
        specifications = {"raster": specifications["raster"]}
    else:
        specifications.pop("raster")
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=5,
        execution=execution,
        callbacks=specifications,
        backend="eager",
    )
    inputs = _inputs(model, 5)
    with torch.no_grad():
        expected = _eager(
            runner,
            runner.initial_state,
            steps=5,
            inputs=inputs,
            specifications=specifications,
        )
    actual = runner(inputs=inputs)
    _close_result(actual, expected)
    assert "trace" not in actual[1]["callbacks"]
    if kind == "raster":
        assert actual[1]["callbacks"]["raster"].shape[0] == 5
        assert actual[1]["callbacks"].state["raster"][1].item() == 5
    else:
        assert actual[1]["callbacks"].state["count"][2].item() == 5
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("width", (5, 9))
def test_threshold_callback_resume_keeps_window_phase_and_bounded_tail(
    execution, width
):
    model = _model()
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=width,
        execution=execution,
        callbacks={"trace": dn.func.Recorder(TRACE), **_thresholds()},
        backend="eager",
    )
    inputs = _inputs(model, 2 * width)
    first = runner(inputs=_slice_inputs(inputs, 0, width))
    saved_carries = _clone_state(first[1]["callbacks"].state.carries)
    second = runner(
        first[0],
        inputs=_slice_inputs(inputs, width, 2 * width),
        callback_state=first[1]["callbacks"].state,
    )
    with torch.no_grad():
        expected = _eager(
            runner,
            runner.initial_state,
            steps=2 * width,
            inputs=inputs,
            specifications={"trace": dn.func.Recorder(TRACE), **_thresholds()},
        )
    _assert_tree_close(second[0], expected[0], rtol=2e-8, atol=2e-9)
    outputs = dict(second[1]["callbacks"])
    outputs["trace"] = {
        name: torch.cat((first[1]["callbacks"]["trace"][name], value), dim=0)
        for name, value in outputs["trace"].items()
    }
    outputs["raster"] = torch.cat(
        (
            first[1]["callbacks"]["raster"],
            second[1]["callbacks"]["raster"],
        ),
        dim=0,
    )
    _assert_tree_close(
        _ordered(outputs),
        _ordered(dict(expected[1]["callbacks"])),
        rtol=2e-8,
        atol=2e-9,
    )
    _assert_tree_close(first[1]["callbacks"].state.carries, saved_carries)
    _assert_tree_close(
        _ordered(second[1]["callbacks"].state.carries),
        _ordered(expected[1]["callbacks"].state.carries),
    )
    assert outputs["trace"]["v"].shape[0] == 2 * width + 1
    assert outputs["raster"].shape[0] == 2 * width
    assert second[1]["callbacks"].state["count"][2].item() == 2 * width
    _assert_reductions(outputs)


def _trace_loss(result):
    state, auxiliary = result
    return (
        state["integrator"]["v"].square().mean()
        + 0.01 * auxiliary["callbacks"]["trace"]["v"].square().mean()
        + 0.1 * auxiliary["callbacks"]["trace"]["hh.m"].square().mean()
    )


def test_scan_threshold_callbacks_preserve_stimulus_voltage_channel_gradients():
    model = _model()
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=9,
        inference=False,
        trainable=(GNABAR,),
        callbacks={"trace": dn.func.Recorder(TRACE), **_thresholds()},
        backend="aot_eager",
    )
    drives = _inputs(model, 9)

    def independent():
        parameters = _clone_mapping(runner.parameters)
        parameters[GNABAR].requires_grad_()
        state = _clone_state(runner.initial_state)
        state["integrator"]["v"].requires_grad_()
        intra = drives.intra.detach().clone().requires_grad_()
        targets = (parameters[GNABAR], state["integrator"]["v"], intra)
        return parameters, state, dn.func.RolloutInput(intra=intra), targets

    parameters, state, inputs, targets = independent()
    reference = _eager(
        runner,
        state,
        steps=9,
        inputs=inputs,
        parameters=parameters,
        specifications={"trace": dn.func.Recorder(TRACE)},
    )
    expected_gradients = torch.autograd.grad(_trace_loss(reference), targets)
    for _ in range(2):
        parameters, state, inputs, targets = independent()
        actual = runner(state, parameters=parameters, inputs=inputs)
        _assert_tree_close(actual[0], reference[0], rtol=2e-8, atol=2e-9)
        _assert_tree_close(
            actual[1]["callbacks"]["trace"],
            reference[1]["callbacks"]["trace"],
            rtol=2e-8,
            atol=2e-9,
        )
        _assert_reductions(actual[1]["callbacks"])
        gradients = torch.autograd.grad(_trace_loss(actual), targets)
        for gradient, expected in zip(gradients, expected_gradients, strict=True):
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0
            torch.testing.assert_close(gradient, expected, rtol=2e-7, atol=2e-9)


class _Energy(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        return carry + state["integrator"]["v"].square().mean(), None

    def finalize(self, carry, emissions):
        return carry


def test_custom_pure_callback_works_with_scan():
    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=5,
        callbacks={"energy": _Energy()},
        backend="eager",
    )
    actual = runner()
    assert set(actual[1]["callbacks"]) == {"energy"}
    with torch.no_grad():
        expected = _eager(
            runner,
            runner.initial_state,
            steps=5,
            specifications={"energy": _Energy()},
        )
    _close_result(actual, expected)


def test_strict_while_rejects_uncapturable_custom_callback_before_compiling_unrolled_horizon():
    class TensorBranch(_Energy):
        def update(self, carry, state, auxiliary):
            if state["integrator"]["v"].mean().item() > 0:
                return carry + 1, None
            return carry - 1, None

    captures = []

    def backend(graph, examples):
        captures.append(graph)
        return graph.forward

    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=800,
        execution="while",
        callbacks={"energy": TensorBranch()},
        backend=backend,
    )
    with pytest.raises(dn.func.FunctionalizationError, match="unavailable"):
        runner()
    assert not captures
    assert runner.execution_report() is None


def test_callback_mapping_validation_and_trace_name_is_ordinary():
    model = _model()
    with pytest.raises(TypeError):
        dn.func.compile(model, dt=DT, steps=4, callbacks=[], backend="eager")
    with pytest.raises(TypeError):
        dn.func.compile(
            model,
            dt=DT,
            steps=4,
            callbacks={"imperative": ImperativeRecorder(("v",))},
            backend="eager",
        )
    foreign, _ = dn.func.make_functional(model, dt=DT)
    collection = foreign.make_callbacks({"count": dn.func.APCount(node_check=(0,))})
    with pytest.raises((TypeError, dn.func.FunctionalizationError)):
        dn.func.compile(model, dt=DT, steps=4, callbacks=collection, backend="eager")
    explicit_trace = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        callbacks={"trace": dn.func.Recorder(("v",))},
        backend="eager",
    )()
    assert explicit_trace[1]["callbacks"]["trace"]["v"].shape[0] == 5
    unrecorded = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        callbacks={},
        backend="eager",
    )()
    assert "callbacks" not in unrecorded[1]
    trace_count = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        callbacks={"trace": dn.func.APCount(node_check=(0,))},
        backend="eager",
    )()
    assert trace_count[1]["callbacks"]["trace"].dtype == torch.int64
    unrecorded_while = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        execution="while",
        callbacks={},
        backend="eager",
    )()
    assert "callbacks" not in unrecorded_while[1]
    assert unrecorded_while[1]["continuation"].callback_state is None


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE, reason="native DHS requires dendra-solvers"
)
@pytest.mark.slow
@pytest.mark.parametrize(
    ("execution", "threads", "steps"),
    [
        pytest.param("scan", 1, 8, id="scan-t1-no-tail"),
        pytest.param("scan", 4, 9, id="scan-t4-tail"),
        pytest.param("while", 1, 9, id="while-t1-tail"),
        pytest.param("while", 4, 8, id="while-t4-no-tail"),
    ],
)
def test_inductor_native_dhs_mixed_threshold_callbacks(execution, threads, steps):
    # Pair modes, thread counts and tail branches orthogonally: each executor
    # runs on one/four threads and with/without a tail, without duplicating the
    # same native compile in a separate plain-Recorder convenience test.
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        case = _case(stimulus_kind="intra")
        source = _clone_state((case.model.v, case.model.t))
        runner = dn.func.compile(
            case.model,
            dt=DT,
            steps=steps,
            execution=execution,
            callbacks={"trace": dn.func.Recorder(TRACE), **_thresholds()},
        )
        originals = _clone_state(
            (runner.initial_state, dict(runner.parameters), dict(runner.constants))
        )
        with torch.no_grad():
            expected = _eager(
                runner,
                runner.initial_state,
                steps=steps,
                specifications={"trace": dn.func.Recorder(TRACE), **_thresholds()},
            )
        actual = runner()
        saved = _clone_state((actual[0], dict(actual[1]["callbacks"])))
        repeated = runner()
        _close_result(actual, expected)
        _close_result(repeated, expected)
        _assert_reductions(actual[1]["callbacks"])
        _assert_reductions(repeated[1]["callbacks"])
        _assert_tree_close(
            (actual[0], dict(actual[1]["callbacks"])), saved, rtol=0, atol=0
        )
        _assert_tree_close(
            (runner.initial_state, dict(runner.parameters), dict(runner.constants)),
            originals,
            rtol=0,
            atol=0,
        )
        _assert_tree_close((case.model.v, case.model.t), source, rtol=0, atol=0)
        assert actual[1]["callbacks"]["trace"]["v"].shape[0] == steps + 1
        assert runner.kernel._compile_options["options"]["cpp_wrapper"] is True
        assert runner.execution_report().strategy == (
            "compiled_scan" if execution == "scan" else "compiled_while_loop"
        )
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE, reason="native DHS requires dendra-solvers"
)
@pytest.mark.slow
@pytest.mark.parametrize("threads", (1, 4))
def test_inductor_native_dhs_threshold_scan_gradients(threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        case = _case(stimulus_kind="intra")
        runner = dn.func.compile(
            case.model,
            dt=DT,
            steps=5,
            inference=False,
            trainable=("gnabar", "amp"),
            callbacks={"trace": dn.func.Recorder(TRACE), **_thresholds()},
        )
        names = tuple(
            name for name, leaf in runner.parameters.items() if leaf.requires_grad
        )
        assert len(names) == 2

        def independent():
            parameters = _clone_mapping(runner.parameters)
            for name in names:
                parameters[name].requires_grad_()
            state = _clone_state(runner.initial_state)
            state["integrator"]["v"].requires_grad_()
            targets = tuple(parameters[name] for name in names) + (
                state["integrator"]["v"],
            )
            return parameters, state, targets

        parameters, state, targets = independent()
        expected = _eager(
            runner,
            state,
            steps=5,
            parameters=parameters,
            specifications={"trace": dn.func.Recorder(TRACE)},
        )
        gradients = torch.autograd.grad(_trace_loss(expected), targets)
        for _ in range(2):
            parameters, state, targets = independent()
            actual = runner(state, parameters=parameters)
            _assert_tree_close(actual[0], expected[0], rtol=2e-8, atol=2e-9)
            _assert_tree_close(
                actual[1]["callbacks"]["trace"],
                expected[1]["callbacks"]["trace"],
                rtol=2e-8,
                atol=2e-9,
            )
            for observed, reference in zip(
                torch.autograd.grad(_trace_loss(actual), targets),
                gradients,
                strict=True,
            ):
                assert torch.isfinite(observed).all()
                assert torch.count_nonzero(observed) > 0
                torch.testing.assert_close(observed, reference, rtol=2e-7, atol=2e-9)
    finally:
        torch.set_num_threads(previous)
