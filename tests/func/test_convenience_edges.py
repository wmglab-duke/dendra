"""Preparation and compiler-policy edges of the host-side convenience API."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from test_compiled_chunk import DT, GNABAR, _assert_tree_close, _model

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


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_constructor_in_inference_mode_retains_versioned_snapshots(execution):
    model = _model()
    with torch.inference_mode():
        runner = dn.func.compile(
            model,
            dt=DT,
            steps=4,
            execution=execution,
            callbacks={"trace": dn.func.Recorder("v")},
            backend="eager",
        )
    leaves = (
        *runner.parameters.values(),
        *runner.constants.values(),
        *torch.utils._pytree.tree_leaves(runner.initial_state),
    )
    for value in leaves:
        assert not torch.is_inference(value)
        assert value._version >= 0
    result = runner()
    assert torch.isfinite(result[0]["integrator"]["v"]).all()
    assert result[1]["callbacks"]["trace"]["v"].shape[0] == 5


def test_training_runner_constructed_in_inference_mode_has_independent_backward_graphs():
    model = _model()
    with torch.inference_mode():
        runner = dn.func.compile(
            model,
            dt=DT,
            steps=4,
            inference=False,
            trainable=(GNABAR,),
            backend="eager",
        )
    gradients = []
    for _ in range(2):
        final, _ = runner()
        gradients.append(
            torch.autograd.grad(
                final["integrator"]["v"].square().mean(),
                runner.parameters[GNABAR],
            )[0]
        )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    torch.testing.assert_close(gradients[0], gradients[1])


def test_inference_cache_refreshes_after_population_parameter_edit():
    runner = dn.func.compile(_model(), dt=DT, steps=4, backend="eager")
    before = runner()
    original_prepared = runner._prepared
    with torch.no_grad():
        runner.parameters["cm_param"].add_(0.25)
    actual = runner()
    assert runner._prepared is not original_prepared
    with torch.no_grad():
        fresh = runner.functional.prepare(runner.parameters, runner.constants)
        expected = dn.func.run(
            runner.functional.bind(runner.parameters, fresh),
            runner.initial_state,
            steps=4,
        )
    _assert_tree_close(actual[0], expected[0], rtol=2.0e-10, atol=2.0e-11)
    assert not torch.equal(before[0]["integrator"]["v"], actual[0]["integrator"]["v"])


@pytest.mark.parametrize("kind", ["missing", "unexpected", "shape", "non_tensor"])
def test_explicit_parameter_overrides_retain_schema_validation(kind):
    runner = dn.func.compile(_model(), dt=DT, steps=4, backend="eager")
    parameters = dict(runner.parameters)
    if kind == "missing":
        parameters.pop(GNABAR)
        exception, match = KeyError, "parameter tree mismatch"
    elif kind == "unexpected":
        parameters["unknown.parameter"] = torch.tensor(0.0, dtype=torch.float64)
        exception, match = KeyError, "parameter tree mismatch"
    elif kind == "shape":
        parameters[GNABAR] = parameters[GNABAR].unsqueeze(0)
        exception, match = ValueError, "shape"
    else:
        parameters[GNABAR] = 0.0
        exception, match = TypeError, "Tensor"
    with pytest.raises(exception, match=match):
        runner(parameters=parameters)
    assert runner._prepared is None
    assert runner.execution_report() is None


def test_corrupted_preparation_is_not_silently_refreshed(monkeypatch):
    runner = dn.func.compile(_model(), dt=DT, steps=4, backend="eager")
    runner()
    original_prepared = runner._prepared
    # Use a solver-owned allocation, not dt, which may alias its constant.
    with torch.no_grad():
        original_prepared.values["integrator"]["diag_base"].add_(0.0)

    def must_not_prepare(*args, **kwargs):
        pytest.fail("workspace corruption must not be hidden by fresh preparation")

    monkeypatch.setattr(FunctionalPopulation, "prepare", must_not_prepare)
    with pytest.raises(
        dn.func.FunctionalizationError, match="prepared workspaces changed"
    ):
        runner()
    assert runner._prepared is original_prepared


@pytest.mark.parametrize(
    "options",
    [
        {"cpp-wrapper": False},
        {"disable_cpp_codegen": True},
        {"disable-cpp-codegen": True},
        {"fx_wrapper": True},
        {"fx-wrapper": True},
        {"use_aoti": True},
        {"use-aoti": True},
    ],
)
def test_native_wrapper_preference_preserves_explicit_codegen_options(
    monkeypatch, options
):
    captured = {}
    original = FunctionalPopulation.compile_rollout_chunk

    def compile_chunk(functional, steps, *, execution, **keywords):
        captured.update(keywords)
        # Inspect policy without invoking Inductor or interpreting its options.
        return original(functional, steps, execution=execution, backend="eager")

    monkeypatch.setattr(FunctionalPopulation, "compile_rollout_chunk", compile_chunk)
    unchanged = deepcopy(options)
    dn.func.compile(_model(), dt=DT, steps=4, options=options)
    assert captured["backend"] == "inductor"
    assert captured["options"] == unchanged
    assert options == unchanged


@pytest.mark.parametrize(
    "keywords",
    [
        {"mode": "reduce-overhead"},
        {"disable": True},
    ],
)
def test_native_wrapper_preference_preserves_mode_and_disabled_compilation(
    monkeypatch, keywords
):
    captured = {}
    original = FunctionalPopulation.compile_rollout_chunk

    def compile_chunk(functional, steps, *, execution, **options):
        captured.update(options)
        return original(functional, steps, execution=execution, backend="eager")

    monkeypatch.setattr(FunctionalPopulation, "compile_rollout_chunk", compile_chunk)
    dn.func.compile(_model(), dt=DT, steps=4, **keywords)
    assert captured == {"backend": "inductor", **keywords}


@pytest.mark.parametrize("transform", ["forward_ad", "vmap", "grad"])
def test_active_transforms_are_refused_before_preparation(monkeypatch, transform):
    runner = dn.func.compile(_model(), dt=DT, steps=4, backend="eager")

    def must_not_prepare(*args, **kwargs):
        pytest.fail("a transform must be rejected before preparation")

    monkeypatch.setattr(runner, "_prepare", must_not_prepare)
    with pytest.raises(dn.func.FunctionalizationError, match="host-side.*lower-level"):
        if transform == "forward_ad":
            with torch.autograd.forward_ad.dual_level():
                runner()
        elif transform == "vmap":
            torch.func.vmap(lambda offset: runner()[0]["integrator"]["v"] + offset)(
                torch.zeros(2, dtype=torch.float64)
            )
        else:
            torch.func.grad(
                lambda offset: runner()[0]["integrator"]["v"].sum() + offset
            )(torch.tensor(0.0, dtype=torch.float64))
    assert runner._prepared is None
    assert runner.execution_report() is None
