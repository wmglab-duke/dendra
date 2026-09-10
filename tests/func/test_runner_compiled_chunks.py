"""Host scheduling of fixed-length compiled functional transitions."""

import gc
import weakref
from functools import partial

import pytest
import torch
from test_runners import (
    DT,
    GNABAR,
    _assert_tree_close,
    _clone_mapping,
    _differentiable_case,
    _drives,
    _loss,
    _model,
)

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._compiled import CompiledPopulationChunk

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]

RUNNERS = ("run", "longrun", "longrun_checkpointed")


def _run(runner, functional, step, state, inputs, *, steps, chunklength=5, **kwargs):
    if runner == "run":
        return dn.func.run(
            functional,
            step,
            state,
            inputs,
            tstop=steps * DT,
            **kwargs,
        )
    return getattr(dn.func, runner)(
        functional,
        step,
        state,
        steps * DT,
        chunklength,
        inputs,
        **kwargs,
    )


def _evaluate(functional, tensors, ve, intra, runner, *, geometry_only=False):
    parameters, constants, state, ve, intra, targets = _differentiable_case(
        tensors, ve, intra, geometry_only=geometry_only
    )
    prepared = functional.prepare(parameters, constants)
    inputs = dn.func.RolloutInput(ve, intra)
    if runner == "eager_rollout":
        final, auxiliary = functional.rollout(parameters, prepared, state, inputs)
    else:
        chunk = functional.compile_rollout_chunk(3, backend="aot_eager")
        final, auxiliary = _run(
            runner,
            functional,
            partial(chunk, parameters, prepared),
            state,
            inputs,
            steps=ve.shape[0],
            chunklength=5,
        )
    loss = _loss(final, auxiliary)
    gradients = dict(
        zip(targets, torch.autograd.grad(loss, tuple(targets.values())), strict=True)
    )
    return final, auxiliary, loss, gradients


@pytest.mark.parametrize("runner", RUNNERS)
def test_multistep_runners_preserve_full_trajectory_gradients_with_tails(runner):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 11)
    expected = _evaluate(functional, tensors, ve, intra, "eager_rollout")

    # C=3, host chunklength=5, and T=11 require both two-step and one-step
    # tails. Gradients must cross every kernel and checkpoint boundary.
    with torch_compiler_warning_context():
        actual = _evaluate(functional, tensors, ve, intra, runner)

    for actual_tree, expected_tree in zip(actual[:3], expected[:3], strict=True):
        _assert_tree_close(actual_tree, expected_tree, rtol=2.0e-10, atol=2.0e-11)
    assert actual[3].keys() == expected[3].keys()
    for name, gradient in actual[3].items():
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name
        torch.testing.assert_close(
            gradient,
            expected[3][name],
            rtol=2.0e-8,
            atol=2.0e-10,
            msg=lambda message: f"gradient {name}: {message}",
        )


def test_multistep_checkpoint_preserves_geometry_only_preparation_dependency():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)
    expected = _evaluate(
        functional, tensors, ve, intra, "eager_rollout", geometry_only=True
    )
    with torch_compiler_warning_context():
        actual = _evaluate(
            functional,
            tensors,
            ve,
            intra,
            "longrun_checkpointed",
            geometry_only=True,
        )
    _assert_tree_close(actual[:3], expected[:3], rtol=2.0e-10, atol=2.0e-11)
    assert torch.count_nonzero(actual[3]["diam"]) > 0
    torch.testing.assert_close(
        actual[3]["diam"], expected[3]["diam"], rtol=2.0e-8, atol=2.0e-10
    )


@pytest.mark.parametrize("runner", RUNNERS)
def test_multistep_runners_preserve_direct_forward_ad_across_tails(runner):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)

    def unexpected_backend(_graph, _inputs):
        raise AssertionError("direct forward AD must use the transform-safe kernel")

    chunk = functional.compile_rollout_chunk(3, backend=unexpected_backend)
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        parameters = _clone_mapping(tensors.parameters)
        constants = _clone_mapping(tensors.constants)
        for values, key in ((parameters, GNABAR), (constants, "diam")):
            primal = values[key]
            values[key] = torch.autograd.forward_ad.make_dual(
                primal, torch.full_like(primal, 0.125)
            )
        prepared = functional.prepare(parameters, constants)
        inputs = dn.func.RolloutInput(ve, intra)
        expected, _ = functional.rollout(parameters, prepared, tensors.state, inputs)
        actual, _ = _run(
            runner,
            functional,
            partial(chunk, parameters, prepared),
            tensors.state,
            inputs,
            steps=7,
            chunklength=5,
        )
        actual_voltage = torch.autograd.forward_ad.unpack_dual(
            actual["integrator"]["v"]
        )
        expected_voltage = torch.autograd.forward_ad.unpack_dual(
            expected["integrator"]["v"]
        )
        torch.testing.assert_close(
            actual_voltage.primal, expected_voltage.primal, rtol=2.0e-10, atol=2.0e-11
        )
        assert actual_voltage.tangent is not None
        assert torch.count_nonzero(actual_voltage.tangent) > 0
        torch.testing.assert_close(
            actual_voltage.tangent, expected_voltage.tangent, rtol=2.0e-8, atol=2.0e-10
        )


@pytest.mark.parametrize(
    ("runner", "kernel_steps", "steps", "chunklength", "expected_sizes"),
    [
        ("run", 3, 11, 5, [3, 3, 3, 2]),
        ("longrun", 3, 11, 5, [3, 2, 3, 2, 1]),
        ("longrun_checkpointed", 3, 11, 5, [3, 2, 3, 2, 1]),
        ("longrun", 8, 4, 3, [3, 1]),
    ],
)
def test_multistep_runner_reuses_compiled_tails_and_exact_input_slices(
    monkeypatch, runner, kernel_steps, steps, chunklength, expected_sizes
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, steps)
    chunk = functional.compile_rollout_chunk(kernel_steps, backend="eager")
    binding = partial(chunk, tensors.parameters, prepared)
    calls = []
    original = CompiledPopulationChunk._execute_operands

    def record(self, operands, *, prewarm_structured):
        calls.append(
            (self, operands.ve.detach().clone(), operands.intra.detach().clone())
        )
        return original(self, operands, prewarm_structured=prewarm_structured)

    monkeypatch.setattr(CompiledPopulationChunk, "_execute_operands", record)
    with torch.no_grad(), torch_compiler_warning_context():
        for _ in range(2):
            _run(
                runner,
                functional,
                binding,
                tensors.state,
                dn.func.RolloutInput(ve, intra),
                steps=steps,
                chunklength=chunklength,
            )

    assert [kernel.steps for kernel, _, _ in calls] == expected_sizes * 2
    width = len(expected_sizes)
    for first, second in zip(calls[:width], calls[width:], strict=True):
        assert first[0] is second[0]
        assert first[0]._compile_options == chunk._compile_options
    for length in set(expected_sizes):
        kernels = [kernel for kernel, _, _ in calls if kernel.steps == length]
        assert all(kernel is kernels[0] for kernel in kernels)
        if length == kernel_steps:
            assert kernels[0] is chunk
    for start in (0, width):
        torch.testing.assert_close(
            torch.cat([item[1] for item in calls[start : start + width]]), ve
        )
        torch.testing.assert_close(
            torch.cat([item[2] for item in calls[start : start + width]]), intra
        )


@pytest.mark.parametrize("runner", RUNNERS)
def test_multistep_callbacks_record_every_step_and_resume_without_duplicates(runner):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, 7)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(["v", "hh.m", "t"])}
    )
    expected, expected_aux = dn.func.run(
        functional,
        partial(functional.step, tensors.parameters, prepared),
        tensors.state,
        dn.func.RolloutInput(ve, intra),
        callbacks=callbacks,
    )
    chunk = functional.compile_rollout_chunk(3, backend="aot_eager")
    binding = partial(chunk, tensors.parameters, prepared)
    with torch_compiler_warning_context():
        first, first_aux = _run(
            runner,
            functional,
            binding,
            tensors.state,
            dn.func.RolloutInput(ve[:2], intra[:2]),
            steps=2,
            callbacks=callbacks,
        )
        final, final_aux = _run(
            runner,
            functional,
            binding,
            first,
            dn.func.RolloutInput(ve[2:], intra[2:]),
            steps=5,
            callbacks=callbacks,
            callback_state=first_aux["callbacks"].state,
        )

    _assert_tree_close(final, expected, rtol=2.0e-10, atol=2.0e-11)
    for name in ("v", "hh.m", "t"):
        trace = torch.cat(
            (
                first_aux["callbacks"]["trace"][name],
                final_aux["callbacks"]["trace"][name],
            )
        )
        assert trace.shape[0] == 8
        torch.testing.assert_close(
            trace, expected_aux["callbacks"]["trace"][name], rtol=2.0e-10, atol=2.0e-11
        )


def test_multistep_checkpoint_callback_loss_preserves_gradients():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "hh.m"])})
    results = []
    for compiled in (False, True):
        parameters, constants, state, local_ve, local_intra, targets = (
            _differentiable_case(tensors, ve, intra)
        )
        prepared = functional.prepare(parameters, constants)
        kernel = (
            functional.compile_rollout_chunk(3, backend="aot_eager")
            if compiled
            else functional.step
        )
        with torch_compiler_warning_context():
            final, auxiliary = _run(
                "longrun_checkpointed" if compiled else "run",
                functional,
                partial(kernel, parameters, prepared),
                state,
                dn.func.RolloutInput(local_ve, local_intra),
                steps=7,
                callbacks=callbacks,
            )
            trace = auxiliary["callbacks"]["trace"]
            loss = trace["v"].square().mean() + trace["hh.m"].square().mean()
            gradients = torch.autograd.grad(loss, tuple(targets.values()))
        results.append((final, loss, gradients))

    _assert_tree_close(results[1][:2], results[0][:2], rtol=2.0e-10, atol=2.0e-11)
    for actual, expected in zip(results[1][2], results[0][2], strict=True):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("steps", [0, 4])
def test_multistep_runners_reject_stale_preparation_before_execution(runner, steps):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    prepared = functional.prepare(parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(3, backend="eager")
    with torch.no_grad():
        parameters[GNABAR].add_(1.0)
    with pytest.raises(
        dn.func.FunctionalizationError, match=r"parameters changed after prepare\(\)"
    ):
        _run(
            runner,
            functional,
            partial(chunk, parameters, prepared),
            tensors.state,
            None,
            steps=steps,
        )


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("steps", [0, 4])
def test_multistep_runners_reject_mismatched_prepared_dt_before_execution(
    runner, steps
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(3, backend="eager")
    with pytest.raises(
        dn.func.FunctionalizationError, match="clock advance does not match runner dt"
    ):
        _run(
            runner,
            functional,
            partial(chunk, tensors.parameters, prepared),
            tensors.state,
            None,
            steps=steps,
            dt=2 * DT,
        )


@pytest.mark.parametrize("runner", RUNNERS)
def test_multistep_runners_reject_another_functional_plan(runner):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    other, other_tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = other.prepare(other_tensors.parameters, other_tensors.constants)
    chunk = other.compile_rollout_chunk(3, backend="eager")
    with pytest.raises(
        dn.func.FunctionalizationError, match="different FunctionalPopulation plan"
    ):
        _run(
            runner,
            functional,
            partial(chunk, other_tensors.parameters, prepared),
            tensors.state,
            None,
            steps=0,
        )


@pytest.mark.parametrize(
    "adapter", ["partial_subclass", "keywords", "bound_call", "outer_compile"]
)
def test_multistep_runners_reject_unsupported_visible_adapters(adapter):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(3, backend="eager")
    if adapter == "partial_subclass":

        class AdaptedPartial(partial):
            pass

        binding = AdaptedPartial(chunk, tensors.parameters, prepared)
    elif adapter == "keywords":
        binding = partial(chunk, parameters=tensors.parameters, prepared=prepared)
    elif adapter == "bound_call":
        binding = partial(chunk.__call__, tensors.parameters, prepared)
    else:
        binding = partial(
            torch.compile(chunk, backend="eager"), tensors.parameters, prepared
        )
    with pytest.raises(dn.func.FunctionalizationError, match="partial"):
        dn.func.run(functional, binding, tensors.state, tstop=0.0)


def test_multistep_checkpoint_revalidates_preparation_during_backward():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    parameters[GNABAR].requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(3, backend="aot_eager")
    with torch_compiler_warning_context():
        final, auxiliary = dn.func.longrun_checkpointed(
            functional,
            partial(chunk, parameters, prepared),
            tensors.state,
            7 * DT,
            5,
        )
    loss = _loss(final, auxiliary)
    with torch.no_grad():
        parameters[GNABAR].add_(1.0)
    with pytest.raises(
        dn.func.FunctionalizationError, match=r"parameters changed after prepare\(\)"
    ):
        torch.autograd.grad(loss, parameters[GNABAR])


def test_multistep_kernel_cache_releases_preparation_graph_after_backward():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    chunk = functional.compile_rollout_chunk(3, backend="eager")

    def run_iteration():
        parameters = _clone_mapping(tensors.parameters)
        parameters[GNABAR].requires_grad_()
        constants = _clone_mapping(tensors.constants)
        constants["diam"].requires_grad_()
        prepared = functional.prepare(parameters, constants)
        references = {
            "prepared": weakref.ref(prepared),
            "parameter": weakref.ref(parameters[GNABAR]),
            "geometry": weakref.ref(constants["diam"]),
            "workspace": weakref.ref(prepared.values["integrator"]["diag_base"]),
        }
        with torch_compiler_warning_context():
            final, auxiliary = dn.func.longrun_checkpointed(
                functional,
                partial(chunk, parameters, prepared),
                tensors.state,
                11 * DT,
                5,
            )
            _loss(final, auxiliary).backward()
        return references

    references = run_iteration()
    # Keep the original kernel and its two tail specializations alive while
    # checking that none of them holds this iteration's preparation graph.
    assert set(chunk._runner_chunks) == {1, 2}
    gc.collect()
    assert {name: reference() for name, reference in references.items()} == {
        "prepared": None,
        "parameter": None,
        "geometry": None,
        "workspace": None,
    }


def test_multistep_cached_tails_accept_fresh_preparation_after_optimizer_update():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)
    inputs = dn.func.RolloutInput(ve, intra)
    parameters = _clone_mapping(tensors.parameters)
    parameters[GNABAR].requires_grad_()
    constants = _clone_mapping(tensors.constants)
    constants["diam"].requires_grad_()
    targets = (parameters[GNABAR], constants["diam"])
    optimizer = torch.optim.SGD(targets, lr=1.0e-5)
    chunk = functional.compile_rollout_chunk(3, backend="aot_eager")
    tail = None
    voltages = []

    for _ in range(2):
        optimizer.zero_grad()
        expected_parameters = _clone_mapping(parameters)
        expected_parameters[GNABAR].requires_grad_()
        expected_constants = _clone_mapping(constants)
        expected_constants["diam"].requires_grad_()
        expected_prepared = functional.prepare(expected_parameters, expected_constants)
        expected, expected_aux = functional.rollout(
            expected_parameters, expected_prepared, tensors.state, inputs
        )
        expected_gradients = torch.autograd.grad(
            _loss(expected, expected_aux),
            (expected_parameters[GNABAR], expected_constants["diam"]),
        )

        # Updating either optimized leaf invalidates the previous preparation;
        # the retained compiled tails must consume this fresh explicit tree.
        prepared = functional.prepare(parameters, constants)
        with torch_compiler_warning_context():
            actual, actual_aux = dn.func.longrun_checkpointed(
                functional,
                partial(chunk, parameters, prepared),
                tensors.state,
                7 * DT,
                5,
                inputs,
            )
            _loss(actual, actual_aux).backward()
        _assert_tree_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)
        _assert_tree_close(actual_aux, expected_aux, rtol=2.0e-10, atol=2.0e-11)
        for target, expected_gradient in zip(targets, expected_gradients, strict=True):
            torch.testing.assert_close(
                target.grad, expected_gradient, rtol=2.0e-8, atol=2.0e-10
            )
        if tail is None:
            tail = chunk._runner_chunks[2]
        else:
            assert chunk._runner_chunks[2] is tail
        voltages.append(actual["integrator"]["v"].detach().clone())
        optimizer.step()

    assert not torch.equal(voltages[0], voltages[1])
