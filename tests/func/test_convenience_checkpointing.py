"""Compiled scan checkpointing reuses the guarded, full-gradient scheduler."""

import inspect

import pytest
import torch
from test_compiled_chunk import DT, GNABAR, _assert_tree_close, _model
from test_convenience_runner import _close_result, _loss, _reference
from test_scan_hh_validation import _case, _clone, _FrameDiagnostics, _operands

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population import FunctionalPopulation

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


@pytest.mark.parametrize("factory", [dn.func.compile, dn.func.CompiledRunner])
def test_checkpoint_every_is_an_explicit_optional_runner_argument(factory):
    parameter = inspect.signature(factory).parameters["checkpoint_every"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("factory", [dn.func.compile, dn.func.CompiledRunner])
@pytest.mark.parametrize("interval", [True, False, 0, -1, 2.5, "4"])
def test_bad_checkpoint_interval_fails_before_functionalization(
    monkeypatch, factory, interval
):
    import dendra.func._convenience as convenience

    def must_not_functionalize(*args, **kwargs):
        pytest.fail("invalid checkpointing must fail before functionalization")

    monkeypatch.setattr(convenience, "make_functional", must_not_functionalize)
    with pytest.raises(ValueError, match="checkpoint_every must be a positive integer"):
        factory(object(), dt=DT, steps=8, inference=False, checkpoint_every=interval)


@pytest.mark.parametrize("factory", [dn.func.compile, dn.func.CompiledRunner])
@pytest.mark.parametrize(
    ("execution", "inference"), [("scan", True), ("while", True), ("while", False)]
)
def test_checkpointing_requires_training_scan_before_functionalization(
    monkeypatch, factory, execution, inference
):
    import dendra.func._convenience as convenience

    def must_not_functionalize(*args, **kwargs):
        pytest.fail("unsupported checkpointing must fail before functionalization")

    monkeypatch.setattr(convenience, "make_functional", must_not_functionalize)
    with pytest.raises(
        ValueError, match="checkpoint_every requires.*scan.*inference=False"
    ):
        factory(
            object(),
            dt=DT,
            steps=8,
            execution=execution,
            inference=inference,
            checkpoint_every=3,
        )


@pytest.mark.parametrize("interval", [None, 3])
def test_checkpoint_policy_is_not_forwarded_to_torch_compile(monkeypatch, interval):
    captured = {}
    original = FunctionalPopulation.compile_rollout_chunk

    def compile_chunk(functional, steps, **keywords):
        captured.update(keywords)
        return original(functional, steps, **keywords)

    monkeypatch.setattr(FunctionalPopulation, "compile_rollout_chunk", compile_chunk)
    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=8,
        inference=False,
        checkpoint_every=interval,
        backend="eager",
    )
    assert runner._checkpoint_every == interval
    assert "checkpoint_every" not in captured
    assert runner.kernel.steps == 8


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "public-fallback"])
def test_both_schedulers_receive_checkpoint_policy_and_replay(monkeypatch, owned):
    import dendra.func._convenience as convenience
    import dendra.func._runners as runners

    schedules, checkpoints = [], []
    original_schedule = convenience._run_source_validated if owned else convenience.run
    original_checkpoint = runners.checkpoint

    def schedule(*args, **keywords):
        schedules.append(keywords["checkpoint_every"])
        return original_schedule(*args, **keywords)

    def checkpoint(*args, **keywords):
        checkpoints.append(keywords["use_reentrant"])
        return original_checkpoint(*args, **keywords)

    monkeypatch.setattr(
        convenience, "_run_source_validated" if owned else "run", schedule
    )
    monkeypatch.setattr(runners, "checkpoint", checkpoint)
    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=8,
        inference=False,
        trainable=(GNABAR,),
        checkpoint_every=3,
        backend="eager",
    )
    for _ in range(2):
        final, _ = runner()
        gradient = torch.autograd.grad(
            final["integrator"]["v"].square().mean(), runner.parameters[GNABAR]
        )[0]
        assert torch.isfinite(gradient).all()
    assert schedules == [3, 3]
    assert checkpoints == [False] * 6  # Two independent 3 + 3 + 2 schedules.
    assert runner.execution_report().strategy == "compiled_scan"


@pytest.mark.slow
@pytest.mark.parametrize("interval", [1, 3, 10], ids=["one", "tail", "over-horizon"])
@pytest.mark.parametrize("recording", ["soma", "full"])
def test_checkpoint_sizes_preserve_recordings_state_and_all_eight_gradients(
    interval, recording
):
    case = _case("sc", torch.float64, 2)
    from test_scan_hh_validation import DT as HH_DT

    configs = {
        "trace": (
            dn.func.Recorder("v", node_indices=[0])
            if recording == "soma"
            else dn.func.Recorder(("v", "hh.m", "hh.h", "hh.n", "t"))
        ),
        "diagnostics": _FrameDiagnostics(),
    }
    # AOT reuses captured forward/backward graphs across checkpoint spans.
    # Raw eager scan repartitions its joint graph on each span; that backend's
    # API and scheduling coverage remains in the other tests in this module.
    runner = dn.func.compile(
        case.model,
        dt=HH_DT,
        steps=8,
        inference=False,
        checkpoint_every=interval,
        callbacks=configs,
        backend="aot_eager",
    )

    def evaluate(*, checkpointed):
        parameters, constants, state, inputs, targets = _operands(case, 8)
        original = _clone((parameters, state, inputs))
        if checkpointed:
            final, auxiliary = runner(state, parameters=parameters, inputs=inputs)
        else:
            prepared = case.functional.prepare(parameters, constants)
            final, auxiliary = dn.func.run(
                case.functional.bind(parameters, prepared),
                state,
                steps=8,
                inputs=inputs,
                callbacks=case.functional.make_callbacks(configs),
            )
        trace = dict(auxiliary["callbacks"]["trace"])
        diagnostics = auxiliary["callbacks"]["diagnostics"]
        assert trace["v"].shape[0] == 9
        assert diagnostics["frames"].item() == 9
        loss = (
            trace["v"].square().mean() / 1000
            + diagnostics["energy"] / 90
            + sum(final["mechanisms"]["hh"][gate].square().mean() for gate in "mhn")
            / 10
        )
        gradients = dict(
            zip(
                targets, torch.autograd.grad(loss, tuple(targets.values())), strict=True
            )
        )
        assert len(gradients) == 8
        assert all(torch.isfinite(value).all() for value in gradients.values())
        _assert_tree_close((parameters, state, inputs), original)
        return _clone((final, trace, diagnostics, gradients))

    expected = evaluate(checkpointed=False)
    for _ in range(2):
        actual = evaluate(checkpointed=True)
        _assert_tree_close(actual, expected, rtol=2e-8, atol=2e-10)


def test_checkpointed_continuation_keeps_callback_carry_and_full_horizon_gradients():
    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=4,
        inference=False,
        trainable=(GNABAR,),
        checkpoint_every=3,
        callbacks={"trace": dn.func.Recorder(("v", "hh.m", "hh.h", "hh.n", "t"))},
        backend="eager",
    )
    expected = _reference(runner, runner.initial_state)
    expected_gradient = torch.autograd.grad(_loss(expected), runner.parameters[GNABAR])[
        0
    ]
    middle, first = runner()
    final, second = runner(middle, callback_state=first["callbacks"].state)
    prefix, suffix = first["callbacks"]["trace"], second["callbacks"]["trace"]
    trace = {name: torch.cat((prefix[name], suffix[name]), dim=0) for name in prefix}
    actual = (final, {"callbacks": {"trace": trace}})
    _close_result(actual, expected)
    gradient = torch.autograd.grad(_loss(actual), runner.parameters[GNABAR])[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-8, atol=2e-10)
    assert trace["v"].shape[0] == 9


@pytest.mark.parametrize("source", ["parameter", "constant", "population"])
def test_checkpoint_replay_retains_source_and_prepared_freshness_guards(
    monkeypatch, source
):
    import dendra.func._runners as runners

    refreshes = []
    original_refresh = runners._refresh_checkpoint_execution

    def refresh(execution, state):
        refreshes.append(None)
        return original_refresh(execution, state)

    monkeypatch.setattr(runners, "_refresh_checkpoint_execution", refresh)
    model = _model()
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        inference=False,
        trainable=(GNABAR,),
        checkpoint_every=3,
        backend="aot_eager",
    )
    final, _ = runner()
    forward_refreshes = len(refreshes)
    assert forward_refreshes == 2  # Forward spans of three and one accepted steps.
    loss = final["integrator"]["v"].square().mean()
    with torch.no_grad():
        if source == "parameter":
            runner.parameters[GNABAR].add_(0.01)
            match = "parameters changed"
        elif source == "constant":
            runner.constants["diam"].add_(0.01)
            match = "constants changed"
        else:
            model._flag_rebuild = True
            match = "pending structural changes"
    with pytest.raises(dn.func.FunctionalizationError, match=match):
        torch.autograd.grad(loss, runner.parameters[GNABAR])
    # AOT's saved-tensor backward actually requests checkpoint recomputation.
    # The raw eager scan may retain its own AD context without requesting it.
    assert len(refreshes) > forward_refreshes


def test_training_checkpoint_policy_under_no_grad_uses_same_spans_without_replay(
    monkeypatch,
):
    def must_not_checkpoint(*args, **keywords):
        pytest.fail("checkpoint must not execute when gradients are disabled")

    monkeypatch.setattr("dendra.func._runners.checkpoint", must_not_checkpoint)
    runner = dn.func.compile(
        _model(),
        dt=DT,
        steps=8,
        inference=False,
        checkpoint_every=3,
        callbacks={"trace": dn.func.Recorder(("v", "hh.m", "hh.h", "hh.n", "t"))},
        backend="eager",
    )
    with torch.no_grad():
        actual = runner()
        expected = _reference(runner, runner.initial_state)
    _close_result(actual, expected)
