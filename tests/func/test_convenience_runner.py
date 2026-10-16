"""The opt-in convenience runner preserves explicit functional semantics."""

from __future__ import annotations

import inspect

import pytest
import torch
from test_compiled_chunk import (
    DT,
    GNABAR,
    _assert_tree_close,
    _clone_mapping,
    _clone_state,
    _drives,
    _model,
)

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population import FunctionalPopulation

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        torch.__version__.split("+")[0] not in ("2.14.0", "2.14.1"),
        reason="functional scan is audited for PyTorch 2.14.0 and 2.14.1",
    ),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _close_result(actual, expected):
    _assert_tree_close(actual[0], expected[0], rtol=2.0e-10, atol=2.0e-11)
    _assert_tree_close(
        dict(actual[1]["callbacks"]),
        dict(expected[1]["callbacks"]),
        rtol=2.0e-10,
        atol=2.0e-11,
    )


def _reference(runner, state, *, parameters=None, inputs=None, steps=8):
    parameters = runner.parameters if parameters is None else parameters
    prepared = runner.functional.prepare(parameters, runner.constants)
    bound = runner.functional.bind(parameters, prepared)
    callbacks = runner.functional.make_callbacks(
        {
            "trace": dn.func.Recorder(["v", "hh.m", "hh.h", "hh.n", "t"]),
        }
    )
    return dn.func.run(bound, state, steps=steps, inputs=inputs, callbacks=callbacks)


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_compiled_runner_records_full_traces_without_advancing_source(execution):
    model = _model()
    source_v = model.v.detach().clone()
    source_t = model.t.detach().clone()
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=8,
        execution=execution,
        callbacks={"trace": dn.func.Recorder(["v", "hh.m", "hh.h", "hh.n", "t"])},
        backend="eager",
    )
    assert isinstance(runner, dn.func.CompiledRunner)
    assert runner.execution_report() is None
    captured_state = _clone_state(runner.initial_state)
    captured_parameters = _clone_mapping(runner.parameters)
    with torch.no_grad():
        expected = _reference(runner, captured_state)
    actual = runner()
    repeated = runner()
    _close_result(actual, expected)
    _close_result(repeated, expected)
    _assert_tree_close(runner.initial_state, captured_state)
    _assert_tree_close(dict(runner.parameters), captured_parameters)
    torch.testing.assert_close(model.v, source_v, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.t, source_t, rtol=0.0, atol=0.0)
    assert actual[1]["callbacks"]["trace"]["v"].shape[0] == 9
    assert not actual[0]["integrator"]["v"].requires_grad
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


def test_runner_parameters_are_independent_live_leaves_in_read_only_mapping():
    model = _model()
    runner = dn.func.compile(model, dt=DT, steps=4, backend="eager")
    _, source = dn.func.make_functional(model, dt=DT)
    for name, value in runner.parameters.items():
        assert value.data_ptr() != source.parameters[name].data_ptr()
        assert not value.requires_grad
    with pytest.raises(TypeError):
        runner.parameters[GNABAR] = torch.zeros_like(runner.parameters[GNABAR])


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_runner_continuation_and_callback_state_do_not_duplicate_boundaries(execution):
    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=4,
        execution=execution,
        callbacks={"trace": dn.func.Recorder("v")},
        backend="eager",
    )
    first = runner()
    second = runner(first[0], callback_state=first[1]["callbacks"].state)
    callbacks = runner.functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
    with torch.no_grad():
        prepared = runner.functional.prepare(runner.parameters, runner.constants)
        expected = dn.func.run(
            runner.functional.bind(runner.parameters, prepared),
            runner.initial_state,
            steps=8,
            callbacks=callbacks,
        )
    joined = torch.cat(
        [
            first[1]["callbacks"]["trace"]["v"],
            second[1]["callbacks"]["trace"]["v"],
        ]
    )
    torch.testing.assert_close(joined, expected[1]["callbacks"]["trace"]["v"])
    assert joined.shape[0] == 9
    _assert_tree_close(second[0], expected[0], rtol=2.0e-10, atol=2.0e-11)


def test_record_selection_and_scan_without_recording():
    model = _model()
    selected = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        callbacks={"selected": dn.func.Recorder("v", node_indices=[0, 3])},
        backend="eager",
    )()
    full = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        callbacks={"trace": dn.func.Recorder("v")},
        backend="eager",
    )()
    unrecorded_runner = dn.func.compile(model, dt=DT, steps=4, backend="eager")
    unrecorded = unrecorded_runner()
    assert set(selected[1]["callbacks"]) == {"selected"}
    assert selected[1]["callbacks"]["selected"]["v"].shape[-1] == 2
    torch.testing.assert_close(
        selected[1]["callbacks"]["selected"]["v"],
        full[1]["callbacks"]["trace"]["v"][..., [0, 3]],
    )
    assert "callbacks" not in unrecorded[1]
    assert unrecorded_runner._callbacks is None
    _assert_tree_close(selected[0], unrecorded[0], rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize("factory", [dn.func.compile, dn.func.CompiledRunner])
def test_recording_is_configured_only_through_callbacks_in_public_signatures(factory):
    parameters = inspect.signature(factory).parameters
    assert "callbacks" in parameters
    assert "record" not in parameters
    assert "node_indices" not in parameters


@pytest.mark.parametrize("factory", [dn.func.compile, dn.func.CompiledRunner])
@pytest.mark.parametrize(
    "keywords",
    [
        {"record": ("v",)},
        {"record": None},
        {"node_indices": [0]},
        {"node_indices": None},
        {"record": None, "node_indices": None},
    ],
)
def test_removed_recording_keywords_fail_before_functionalization(
    monkeypatch, factory, keywords
):
    import dendra.func._convenience as convenience

    def must_not_functionalize(*args, **kwargs):
        pytest.fail("removed recording keywords must fail before functionalization")

    monkeypatch.setattr(convenience, "make_functional", must_not_functionalize)
    with pytest.raises(TypeError, match="callbacks.*Recorder|Recorder.*callbacks"):
        factory(object(), dt=DT, steps=4, backend="eager", **keywords)


@pytest.mark.parametrize("factory", [dn.func.compile, dn.func.CompiledRunner])
@pytest.mark.parametrize(
    "callback_options", [{}, {"callbacks": None}, {"callbacks": {}}]
)
def test_scan_without_callbacks_allocates_no_recording(factory, callback_options):
    runner = factory(_model(), dt=DT, steps=4, backend="eager", **callback_options)
    final, auxiliary = runner()
    assert runner._callbacks is None
    assert "callbacks" not in auxiliary
    assert torch.isfinite(final["integrator"]["v"]).all()
    assert runner.execution_report().strategy == "compiled_scan"


def test_compiled_runner_constructor_accepts_explicit_named_recorder():
    runner = dn.func.CompiledRunner(
        _model(),
        dt=DT,
        steps=4,
        execution="while",
        callbacks={"chosen": dn.func.Recorder("v", node_indices=[1])},
        backend="eager",
    )
    _, auxiliary = runner()
    assert set(auxiliary["callbacks"]) == {"chosen"}
    assert auxiliary["callbacks"]["chosen"]["v"].shape[-1] == 1
    assert runner.execution_report().strategy == "compiled_while_loop"


def test_cached_inference_preparation_refreshes_mutated_constants(monkeypatch):
    calls = []
    original = FunctionalPopulation.prepare

    def prepare(functional, parameters, constants):
        calls.append(functional)
        return original(functional, parameters, constants)

    monkeypatch.setattr(FunctionalPopulation, "prepare", prepare)
    runner = dn.func.compile(_model(), dt=DT, steps=4, backend="eager")
    first = runner()
    calls_after_first = len(calls)
    runner()
    assert len(calls) == calls_after_first
    with torch.no_grad():
        runner.constants["diam"].mul_(1.1)
    changed = runner()
    assert len(calls) == calls_after_first + 1
    with torch.no_grad():
        prepared = runner.functional.prepare(runner.parameters, runner.constants)
        expected = dn.func.run(
            runner.functional.bind(runner.parameters, prepared),
            runner.initial_state,
            steps=4,
        )
    _assert_tree_close(changed[0], expected[0], rtol=2.0e-10, atol=2.0e-11)
    assert not torch.equal(first[0]["integrator"]["v"], changed[0]["integrator"]["v"])


def _loss(result):
    state, auxiliary = result
    return (
        state["integrator"]["v"].square().mean()
        + 0.1 * state["mechanisms"]["hh"]["m"].square().mean()
        + 0.01 * auxiliary["callbacks"]["trace"]["v"].sin().mean()
    )


def test_training_scan_matches_eager_traces_and_all_explicit_gradient_sources():
    model = _model()
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=8,
        inference=False,
        trainable=[GNABAR],
        callbacks={"trace": dn.func.Recorder(["v", "hh.m", "hh.h", "hh.n", "t"])},
        backend="eager",
    )
    assert runner.parameters[GNABAR].requires_grad
    ve, intra = _drives(model, 8)

    def independent_case():
        parameters = _clone_mapping(runner.parameters)
        parameters[GNABAR].requires_grad_()
        state = _clone_state(runner.initial_state)
        state["integrator"]["v"].requires_grad_()
        inputs = dn.func.RolloutInput(
            ve=ve.clone().requires_grad_(),
            intra=intra.clone().requires_grad_(),
        )
        targets = (
            parameters[GNABAR],
            state["integrator"]["v"],
            inputs.ve,
            inputs.intra,
        )
        return parameters, state, inputs, targets

    expected_parameters, expected_state, expected_inputs, expected_targets = (
        independent_case()
    )
    expected = _reference(
        runner,
        expected_state,
        parameters=expected_parameters,
        inputs=expected_inputs,
    )
    expected_gradients = torch.autograd.grad(_loss(expected), expected_targets)
    parameters, state, inputs, targets = independent_case()
    actual = runner(state, parameters=parameters, inputs=inputs)
    gradients = torch.autograd.grad(_loss(actual), targets)
    _close_result(actual, expected)
    for gradient, reference in zip(gradients, expected_gradients, strict=True):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
        torch.testing.assert_close(gradient, reference, rtol=2.0e-8, atol=2.0e-10)


def test_training_runner_builds_fresh_autograd_graph_for_each_backward(monkeypatch):
    calls = []
    original = FunctionalPopulation.prepare

    def prepare(functional, parameters, constants):
        calls.append(torch.is_grad_enabled())
        return original(functional, parameters, constants)

    monkeypatch.setattr(FunctionalPopulation, "prepare", prepare)
    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=4,
        inference=False,
        trainable=[GNABAR],
        callbacks={"trace": dn.func.Recorder("v")},
        backend="eager",
    )
    gradients = []
    previous_calls = len(calls)
    for _ in range(2):
        result = runner()
        gradients.append(
            torch.autograd.grad(_loss(result), runner.parameters[GNABAR])[0]
        )
        assert len(calls) == previous_calls + 1
        assert calls[-1]
        previous_calls = len(calls)
    torch.testing.assert_close(gradients[0], gradients[1])


@pytest.mark.parametrize(
    "execution,inference",
    [
        ("scan", True),
        ("while", True),
        ("scan", False),
    ],
)
def test_bound_stimulation_is_included_and_amplitude_can_be_trainable(
    execution, inference
):
    model = _model()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model[:, 1].inject(dn.mono_rect(amp=0.2 * dn.units.nA, delay=0.02, pw=0.04))
        model.initialize()
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=8,
        execution=execution,
        inference=inference,
        trainable=() if inference else ("amp",),
        callbacks={"trace": dn.func.Recorder(["v", "hh.m", "hh.h", "hh.n", "t"])},
        backend="eager",
    )
    parameters = _clone_mapping(runner.parameters)
    amplitude_name = next(name for name in parameters if name.endswith(".amp"))
    if not inference:
        parameters[amplitude_name].requires_grad_()
        expected = _reference(runner, runner.initial_state, parameters=parameters)
    else:
        with torch.no_grad():
            expected = _reference(runner, runner.initial_state, parameters=parameters)
    actual = runner()
    _close_result(actual, expected)
    if not inference:
        reference_gradient = torch.autograd.grad(
            _loss(expected), parameters[amplitude_name]
        )[0]
        gradient = torch.autograd.grad(
            _loss(actual), runner.parameters[amplitude_name]
        )[0]
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        torch.testing.assert_close(
            gradient, reference_gradient, rtol=2.0e-8, atol=2.0e-10
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"execution": "while", "callbacks": None}, "callback"),
        ({"execution": "while", "callbacks": {}}, "callback"),
        ({"execution": "while", "steps": 3}, "4|four"),
        ({"execution": "while", "inference": False}, "inference"),
        ({"execution": "unrolled"}, "execution"),
    ],
)
def test_invalid_execution_contract_is_rejected_before_compilation(kwargs, match):
    options = {
        "dt": DT,
        "steps": 4,
        "backend": "eager",
        "callbacks": {"trace": dn.func.Recorder("v")},
    }
    options.update(kwargs)
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match=match):
        dn.func.compile(_model(), **options)


def test_strict_while_rejects_capture_failure_without_unrolled_fallback(monkeypatch):
    from dendra.func._recorder_while import _RecorderWhileExecutor

    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=800,
        execution="while",
        backend=backend,
        callbacks={"trace": dn.func.Recorder("v")},
    )
    monkeypatch.setattr(
        _RecorderWhileExecutor,
        "kernel_for",
        lambda *args: (None, "sentinel unsupported Recorder while capture"),
    )
    with pytest.raises(dn.func.FunctionalizationError, match="sentinel"):
        runner()
    assert not graphs
    assert runner.execution_report() is None


@pytest.mark.parametrize("execution", ["scan", "while"])
@pytest.mark.parametrize(
    "inference,backend,explicit,expected",
    [
        (True, "inductor", None, True),
        (True, "inductor", False, False),
        (True, "eager", None, None),
        (False, "inductor", None, None),
    ],
)
def test_convenience_cpp_policy_preserves_explicit_and_noninductor_choices(
    monkeypatch,
    execution,
    inference,
    backend,
    explicit,
    expected,
):
    if execution == "while" and not inference:
        pytest.skip("while execution is deliberately inference-only")
    captured = {}
    original = FunctionalPopulation.compile_rollout_chunk

    def compile_chunk(functional, steps, *, execution, **options):
        captured.update(options)
        return original(functional, steps, execution=execution, backend="eager")

    monkeypatch.setattr(FunctionalPopulation, "compile_rollout_chunk", compile_chunk)
    options = None if explicit is None else {"cpp_wrapper": explicit}
    original_options = None if options is None else dict(options)
    dn.func.compile(
        _model(),
        dt=DT,
        steps=4,
        execution=execution,
        inference=inference,
        backend=backend,
        options=options,
        callbacks={"trace": dn.func.Recorder("v")},
    )
    assert captured["backend"] == backend
    assert captured.get("options", {}).get("cpp_wrapper") is expected
    assert options == original_options
