"""Invocation-local source checks do not exempt later public boundaries."""

from collections.abc import Mapping
from types import MethodType

import pytest
import torch
from test_compiled_chunk import DT, GNABAR, _model

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._compiled import CompiledPopulationChunk
from dendra.func._population import FunctionalPopulation

pytestmark = [
    pytest.mark.cpu,
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _runner(*, execution="scan", inference=True, model=None):
    return dn.func.compile(
        _model() if model is None else model,
        dt=DT,
        steps=4,
        execution=execution,
        inference=inference,
        trainable=() if inference else (GNABAR,),
        callbacks={"trace": dn.func.Recorder(["v", "hh.m", "hh.h", "hh.n", "t"])},
        backend="eager",
    )


def _count_source(monkeypatch, functional):
    calls = []
    original = functional._validate_source

    def validate():
        calls.append(None)
        return original()

    monkeypatch.setattr(functional, "_validate_source", validate)
    return calls


@pytest.mark.parametrize(
    ("execution", "inference"), [("scan", True), ("while", True), ("scan", False)]
)
def test_one_complete_source_check_per_cold_and_warm_invocation(
    monkeypatch, execution, inference
):
    runner = _runner(execution=execution, inference=inference)
    calls = _count_source(monkeypatch, runner.functional)
    for index in range(3):
        result = runner()
        assert len(calls) == index + 1
        assert result[1]["callbacks"]["trace"]["v"].shape[0] == 5
        if not inference:
            gradient = torch.autograd.grad(
                result[0]["integrator"]["v"].square().mean(),
                runner.parameters[GNABAR],
            )[0]
            assert torch.isfinite(gradient).all()
            assert len(calls) == index + 1


def test_refresh_and_explicit_parameters_each_validate_once(monkeypatch):
    runner = _runner()
    runner()
    calls = _count_source(monkeypatch, runner.functional)
    with torch.no_grad():
        runner.constants["diam"].mul_(1.1)
    runner()
    assert len(calls) == 1
    runner(parameters=dict(runner.parameters))
    assert len(calls) == 2


def test_independent_prepare_bind_run_and_eager_step_keep_source_checks(monkeypatch):
    runner = _runner()
    runner()
    functional = runner.functional
    calls = _count_source(monkeypatch, functional)
    prepared = functional.prepare(runner.parameters, runner.constants)
    assert len(calls) == 1
    bound = runner.kernel.bind(runner.parameters, prepared)
    assert len(calls) == 2
    dn.func.run(bound, runner.initial_state, steps=4)
    assert len(calls) == 3
    functional.step(runner.parameters, prepared, runner.initial_state)
    assert len(calls) == 4


@pytest.mark.parametrize("edit", ["pending_structure", "current_method"])
def test_source_edits_are_rejected_after_successful_warm_calls(edit):
    model = _model()
    runner = _runner(model=model)
    runner()
    runner()
    if edit == "pending_structure":
        model._flag_rebuild = True
        match = "pending structural changes"
    else:
        mechanism = model.mech.mechanisms["hh"]
        original = mechanism.i_affine_terms

        def replacement(self, v):
            del self
            return original(v)

        mechanism.i_affine_terms = MethodType(replacement, mechanism)
        match = "structure changed"
    with pytest.raises(dn.func.FunctionalizationError, match=match):
        runner()


def test_external_parameter_mapping_is_materialized_before_source_validation():
    model = _model()
    runner = _runner(model=model)
    runner()

    class MutatingMapping(Mapping):
        def __iter__(self):
            model._flag_rebuild = True
            return iter(runner.parameters)

        def __getitem__(self, key):
            return runner.parameters[key]

        def __len__(self):
            return len(runner.parameters)

    with pytest.raises(
        dn.func.FunctionalizationError, match="pending structural changes"
    ):
        runner(parameters=MutatingMapping())


@pytest.mark.parametrize("boundary", ["prepare", "bind"])
def test_replaced_public_boundaries_retain_checks_after_custom_code(
    monkeypatch, boundary
):
    model = _model()
    runner = _runner(model=model)
    if boundary == "prepare":
        original = FunctionalPopulation.prepare

        def prepare(functional, parameters, constants):
            prepared = original(functional, parameters, constants)
            model._flag_rebuild = True
            return prepared

        monkeypatch.setattr(FunctionalPopulation, "prepare", prepare)
    else:
        original = CompiledPopulationChunk.bind

        def bind(kernel, parameters, prepared):
            bound = original(kernel, parameters, prepared)
            model._flag_rebuild = True
            return bound

        monkeypatch.setattr(CompiledPopulationChunk, "bind", bind)
    with pytest.raises(
        dn.func.FunctionalizationError, match="pending structural changes"
    ):
        runner()


def test_nested_invocations_check_each_source_and_do_not_leak_on_exception(monkeypatch):
    outer = _runner()
    inner_model = _model()
    inner = _runner(model=inner_model)
    inner_calls = _count_source(monkeypatch, inner.functional)
    original = outer.functional._validate_source

    def nested_validate():
        inner()
        return original()

    monkeypatch.setattr(outer.functional, "_validate_source", nested_validate)
    outer()
    assert len(inner_calls) == 1
    with pytest.raises((KeyError, ValueError, TypeError)):
        outer(state={})
    assert len(inner_calls) == 2
    inner_model._flag_rebuild = True
    with pytest.raises(
        dn.func.FunctionalizationError, match="pending structural changes"
    ):
        inner()
    assert len(inner_calls) == 3


def test_original_prepare_rebound_to_foreign_functional_is_not_bypassed(monkeypatch):
    runner = _runner()
    other = _runner()
    monkeypatch.setattr(
        runner.functional,
        "prepare",
        MethodType(FunctionalPopulation.prepare, other.functional),
    )
    with pytest.raises(dn.func.FunctionalizationError, match="different functional"):
        runner()
    assert runner.execution_report() is None


def test_original_bind_rebound_to_foreign_kernel_is_not_bypassed(monkeypatch):
    runner = _runner()
    other = _runner()
    monkeypatch.setattr(
        runner.kernel,
        "bind",
        MethodType(CompiledPopulationChunk.bind, other.kernel),
    )
    with pytest.raises(dn.func.FunctionalizationError, match="different functional"):
        runner()
    assert runner.execution_report() is None


def test_replaced_scheduler_keeps_source_check_after_custom_code(monkeypatch):
    import dendra.func._convenience as convenience

    model = _model()
    runner = _runner(model=model)
    original = convenience.run

    def schedule(*args, **kwargs):
        model._flag_rebuild = True
        return original(*args, **kwargs)

    monkeypatch.setattr(convenience, "run", schedule)
    with pytest.raises(
        dn.func.FunctionalizationError, match="pending structural changes"
    ):
        runner()
    assert runner.execution_report() is None


def test_foreign_and_modified_prepared_values_remain_invalid():
    runner = _runner()
    other = _runner()
    runner()
    original = runner._prepared
    with torch.inference_mode():
        runner._prepared = other.functional.prepare(other.parameters, other.constants)
    with pytest.raises(dn.func.FunctionalizationError, match="different functional"):
        runner()
    runner._prepared = original
    with torch.no_grad():
        original.values["integrator"]["diag_base"].add_(0.0)
    with pytest.raises(
        dn.func.FunctionalizationError, match="prepared workspaces changed"
    ):
        runner()
