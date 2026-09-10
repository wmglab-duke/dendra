"""Acceptance coverage for callbacks fused into fixed compiled runner chunks."""

import gc
import weakref
from dataclasses import dataclass
from functools import partial

import pytest
import torch
from test_runner_compiled_chunks import RUNNERS, _run
from test_runners import (
    DT,
    GNABAR,
    _assert_tree_close,
    _differentiable_case,
    _drives,
    _model,
)
from torch.nn import functional as F

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@dataclass(frozen=True)
class _TraceMSE(dn.func.FunctionalCallback):
    """Pure trace fitting reducer; its MSE nodes identify callback work in FX."""

    target: torch.Tensor

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "error": F.mse_loss(voltage, self.target[0]),
            "frames": voltage.new_ones((), dtype=torch.long),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        reference = torch.index_select(
            self.target, 0, carry["frames"].reshape(1)
        ).squeeze(0)
        return {
            "error": carry["error"] + F.mse_loss(state["integrator"]["v"], reference),
            "frames": carry["frames"] + 1,
        }, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry["error"] / carry["frames"]


class _PostStepVoltage(dn.func.FunctionalCallback):
    """Custom emitter whose sample schema appears only after a transition."""

    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"], None

    def update(self, carry, state, auxiliary):
        del carry
        voltage = state["integrator"]["v"]
        return voltage, voltage + auxiliary["v"]

    def finalize(self, carry, emissions):
        if emissions is None:
            return carry.new_empty((0, *carry.shape))
        return emissions


def _callbacks(functional, model, steps, *, offset=0.0, states=("v", "hh.m", "t")):
    target = torch.linspace(
        -64.0 + offset,
        -52.0 + offset,
        (steps + 1) * model.v.numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps + 1, *model.shape)
    return functional.make_callbacks(
        {"mse": _TraceMSE(target), "trace": dn.func.Recorder(states)}
    )


def _clone(tree):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), tree)


def _evaluate(
    functional, tensors, ve, intra, callbacks, runner, kernel=None, *, offset=0.0
):
    parameters, constants, state, local_ve, local_intra, targets = _differentiable_case(
        tensors, ve, intra
    )
    with torch.no_grad():
        parameters[GNABAR].add_(offset)
    prepared = functional.prepare(parameters, constants)
    final, auxiliary = _run(
        runner,
        functional,
        partial(functional.step if kernel is None else kernel, parameters, prepared),
        state,
        dn.func.RolloutInput(local_ve, local_intra),
        steps=ve.shape[0],
        chunklength=5,
        callbacks=callbacks,
    )
    result = auxiliary["callbacks"]
    # Combine reducer and recorded-trace losses so both paths must preserve AD.
    loss = result["mse"] + 0.1 * result["trace"]["v"].square().mean()
    gradients = dict(
        zip(targets, torch.autograd.grad(loss, tuple(targets.values())), strict=True)
    )
    return final, result, loss, gradients


def _assert_result_close(actual, expected):
    _assert_tree_close(actual[:3], expected[:3], rtol=2.0e-10, atol=2.0e-11)
    for name, gradient in actual[3].items():
        reference = expected[3][name]
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(reference) > 0, name
        # Normalize to test small physical derivatives with relative accuracy.
        scale = reference.abs().amax()
        torch.testing.assert_close(
            gradient / scale,
            reference / scale,
            rtol=2.0e-8,
            atol=2.0e-10,
            msg=lambda message: f"gradient {name}: {message}",
        )


@pytest.mark.parametrize("runner", RUNNERS)
def test_fused_callback_chunks_preserve_recordings_reducers_and_all_source_gradients(
    runner,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 11)
    callbacks = _callbacks(functional, model, 11)
    expected = _evaluate(functional, tensors, ve, intra, callbacks, "run")
    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")

    # T=11, C=3 and H=5 exercise independent kernel and checkpoint tails.
    with torch_compiler_warning_context():
        actual = _evaluate(functional, tensors, ve, intra, callbacks, runner, kernel)

    _assert_result_close(actual, expected)
    assert actual[1]["trace"]["v"].shape[0] == 12
    assert actual[1].state["mse"]["frames"].item() == 12


@pytest.mark.parametrize("runner", RUNNERS)
def test_compiler_graphs_contain_every_callback_update_in_multistep_chunks(runner):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 11)
    callbacks = _callbacks(functional, model, 11)
    widths = []

    def backend(graph, _example_inputs):
        widths.append(
            sum(
                node.op == "call_function" and node.target is F.mse_loss
                for node in graph.graph.nodes
            )
        )
        return graph.forward

    kernel = functional.compile_rollout_chunk(3, backend=backend)
    with torch_compiler_warning_context():
        actual = _evaluate(functional, tensors, ve, intra, callbacks, runner, kernel)
    assert actual[1].state["mse"]["frames"].item() == 12
    # Callback MSE nodes occur once per transition in each compiled graph.
    # A one-step fallback or callbacks executed by the host cannot pass this.
    assert set(widths) == ({2, 3} if runner == "run" else {1, 2, 3})


@pytest.mark.parametrize("runner", RUNNERS)
def test_fused_callbacks_resume_through_zero_step_segments_without_duplicate_samples(
    runner,
):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 11)
    callbacks = _callbacks(functional, model, 11)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    expected, expected_aux = _run(
        "run",
        functional,
        partial(functional.step, tensors.parameters, prepared),
        tensors.state,
        dn.func.RolloutInput(ve, intra),
        steps=11,
        callbacks=callbacks,
    )
    graph_count = 0

    def backend(graph, _example_inputs):
        nonlocal graph_count
        graph_count += 1
        return graph.forward

    kernel = functional.compile_rollout_chunk(3, backend=backend)
    binding = partial(kernel, tensors.parameters, prepared)
    state, callback_state = tensors.state, None
    samples = {name: [] for name in ("v", "hh.m", "t")}
    start = 0
    with torch.no_grad(), torch_compiler_warning_context():
        for steps in (0, 2, 0, 9, 0):
            before_count = graph_count
            before_carry = None if callback_state is None else _clone(callback_state)
            state, auxiliary = _run(
                runner,
                functional,
                binding,
                state,
                dn.func.RolloutInput(
                    ve[start : start + steps], intra[start : start + steps]
                ),
                steps=steps,
                chunklength=5,
                callbacks=callbacks,
                callback_state=callback_state,
            )
            if steps == 0:
                assert graph_count == before_count
            if before_carry is not None:
                _assert_tree_close(callback_state, before_carry)
            callback_state = auxiliary["callbacks"].state
            for name in samples:
                samples[name].append(auxiliary["callbacks"]["trace"][name])
            start += steps

    _assert_tree_close(state, expected, rtol=2.0e-10, atol=2.0e-11)
    _assert_tree_close(
        callback_state, expected_aux["callbacks"].state, rtol=2.0e-10, atol=2.0e-11
    )
    for name, parts in samples.items():
        assert [part.shape[0] for part in parts] == [1, 2, 0, 9, 0]
        torch.testing.assert_close(
            torch.cat(parts),
            expected_aux["callbacks"]["trace"][name],
            rtol=2.0e-10,
            atol=2.0e-11,
        )
    torch.testing.assert_close(
        auxiliary["callbacks"]["mse"], expected_aux["callbacks"]["mse"]
    )


def test_checkpoint_replay_preserves_fused_callback_inputs_and_outputs():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 11)
    callbacks = _callbacks(functional, model, 11)
    parameters, constants, state, ve, intra, targets = _differentiable_case(
        tensors, ve, intra
    )
    prepared = functional.prepare(parameters, constants)
    initial, initial_aux = _run(
        "run",
        functional,
        partial(functional.step, parameters, prepared),
        state,
        dn.func.RolloutInput(ve[:2], intra[:2]),
        steps=2,
        callbacks=callbacks,
    )
    callback_state = initial_aux["callbacks"].state
    source = (initial, callback_state)
    source_before = _clone(source)
    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")
    with torch_compiler_warning_context():
        final, auxiliary = _run(
            "longrun_checkpointed",
            functional,
            partial(kernel, parameters, prepared),
            initial,
            dn.func.RolloutInput(ve[2:], intra[2:]),
            steps=9,
            chunklength=5,
            callbacks=callbacks,
            callback_state=callback_state,
        )
        result_before = _clone((final, auxiliary["callbacks"]))
        loss = (
            auxiliary["callbacks"]["mse"]
            + auxiliary["callbacks"]["trace"]["v"].square().mean()
        )
        first = torch.autograd.grad(loss, tuple(targets.values()), retain_graph=True)
        second = torch.autograd.grad(loss, tuple(targets.values()))

    _assert_tree_close(source, source_before)
    _assert_tree_close((final, auxiliary["callbacks"]), result_before)
    _assert_tree_close(first, second)
    assert auxiliary["callbacks"].state["mse"]["frames"].item() == 12


def test_fused_kernels_reuse_graphs_across_fresh_training_graphs_and_callback_plans():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 11)
    plans = (
        _callbacks(functional, model, 11, states=("v",)),
        _callbacks(functional, model, 11, offset=7.0, states=("v", "t")),
    )
    graph_count = 0

    def backend(graph, _example_inputs):
        nonlocal graph_count
        graph_count += 1
        return graph.forward

    kernel = functional.compile_rollout_chunk(3, backend=backend)
    warm_count = None
    with torch_compiler_warning_context():
        for iteration in range(3):
            losses = []
            for callbacks in plans:
                offset = iteration * 0.001
                expected = _evaluate(
                    functional, tensors, ve, intra, callbacks, "run", offset=offset
                )
                actual = _evaluate(
                    functional,
                    tensors,
                    ve,
                    intra,
                    callbacks,
                    "longrun_checkpointed",
                    kernel,
                    offset=offset,
                )
                _assert_result_close(actual, expected)
                losses.append(actual[2].detach())
            assert not torch.equal(*losses)
            if warm_count is None:
                warm_count = graph_count
                assert warm_count > 0
            else:
                assert graph_count == warm_count


def test_fused_chunk_cache_releases_callback_collection_and_training_graph():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)
    kernel = functional.compile_rollout_chunk(3, backend="eager")

    def iteration():
        callbacks = _callbacks(functional, model, 7)
        parameters, constants, state, local_ve, local_intra, targets = (
            _differentiable_case(tensors, ve, intra)
        )
        prepared = functional.prepare(parameters, constants)
        references = {
            "callbacks": weakref.ref(callbacks),
            "prepared": weakref.ref(prepared),
            "parameter": weakref.ref(parameters[GNABAR]),
            "geometry": weakref.ref(constants["diam"]),
            "workspace": weakref.ref(prepared.values["integrator"]["diag_base"]),
        }
        with torch_compiler_warning_context():
            _state, auxiliary = _run(
                "longrun_checkpointed",
                functional,
                partial(kernel, parameters, prepared),
                state,
                dn.func.RolloutInput(local_ve, local_intra),
                steps=7,
                chunklength=5,
                callbacks=callbacks,
            )
            torch.autograd.grad(auxiliary["callbacks"]["mse"], tuple(targets.values()))
        return references

    references = iteration()
    # Keep the user-owned kernel alive: cached callback specializations must not
    # keep callback collections or an optimizer iteration's preparation alive.
    gc.collect()
    assert {name: reference() for name, reference in references.items()} == {
        name: None for name in references
    }


def test_fused_callbacks_discover_post_step_emission_schema_and_resume_it():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)
    options = {"threshold": -100.0, "node_check": [0, -1], "t_end_check": 5 * DT}
    callbacks = functional.make_callbacks(
        {"raster": dn.func.Raster(**options), "count": dn.func.APCount(**options)}
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    binding = partial(
        functional.compile_rollout_chunk(3, backend="eager"),
        tensors.parameters,
        prepared,
    )
    with torch.no_grad(), torch_compiler_warning_context():
        expected, expected_aux = _run(
            "run",
            functional,
            partial(functional.step, tensors.parameters, prepared),
            tensors.state,
            dn.func.RolloutInput(ve, intra),
            steps=7,
            callbacks=callbacks,
        )
        initial, initial_aux = _run(
            "run",
            functional,
            binding,
            tensors.state,
            None,
            steps=0,
            callbacks=callbacks,
        )
        assert initial_aux["callbacks"]["raster"].shape == (0, model.shape[0], 2)
        first, first_aux = _run(
            "longrun_checkpointed",
            functional,
            binding,
            initial,
            dn.func.RolloutInput(ve[:2], intra[:2]),
            steps=2,
            chunklength=5,
            callbacks=callbacks,
            callback_state=initial_aux["callbacks"].state,
        )
        final, final_aux = _run(
            "longrun_checkpointed",
            functional,
            binding,
            first,
            dn.func.RolloutInput(ve[2:], intra[2:]),
            steps=5,
            chunklength=5,
            callbacks=callbacks,
            callback_state=first_aux["callbacks"].state,
        )

    _assert_tree_close(final, expected, rtol=2.0e-10, atol=2.0e-11)
    actual_raster = torch.cat(
        (first_aux["callbacks"]["raster"], final_aux["callbacks"]["raster"])
    )
    assert actual_raster.shape[0] == 7
    assert actual_raster[0].all()
    torch.testing.assert_close(actual_raster, expected_aux["callbacks"]["raster"])
    torch.testing.assert_close(
        final_aux["callbacks"]["count"], actual_raster.sum(dim=0)
    )
    _assert_tree_close(final_aux["callbacks"].state, expected_aux["callbacks"].state)


def test_custom_post_step_emitter_preserves_resumed_trace_gradients():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)
    callbacks = functional.make_callbacks({"post": _PostStepVoltage()})
    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")

    def evaluate(compiled):
        parameters, constants, state, local_ve, local_intra, targets = (
            _differentiable_case(tensors, ve, intra)
        )
        prepared = functional.prepare(parameters, constants)
        binding = partial(kernel if compiled else functional.step, parameters, prepared)
        samples = []
        callback_state = None
        start = 0
        for steps in (0, 2, 0, 5) if compiled else (7,):
            state, auxiliary = _run(
                "longrun_checkpointed" if compiled else "run",
                functional,
                binding,
                state,
                dn.func.RolloutInput(
                    local_ve[start : start + steps], local_intra[start : start + steps]
                ),
                steps=steps,
                chunklength=5,
                callbacks=callbacks,
                callback_state=callback_state,
            )
            callback_state = auxiliary["callbacks"].state
            emitted = auxiliary["callbacks"]["post"]
            assert emitted.shape[0] == steps
            samples.append(emitted)
            start += steps
        trace = torch.cat(samples)
        loss = trace.square().mean()
        gradients = torch.autograd.grad(loss, tuple(targets.values()))
        return state, callback_state, trace, loss, gradients

    expected = evaluate(False)
    with torch_compiler_warning_context():
        actual = evaluate(True)
    _assert_tree_close(actual[:4], expected[:4], rtol=2.0e-10, atol=2.0e-11)
    for gradient, reference in zip(actual[4], expected[4], strict=True):
        scale = reference.abs().amax()
        assert scale > 0
        torch.testing.assert_close(
            gradient / scale, reference / scale, rtol=2.0e-8, atol=2.0e-10
        )


@pytest.mark.parametrize("resume", [False, True])
def test_fused_callbacks_reject_emission_shape_changes_from_initial_boundary(resume):
    class WrongShape(dn.func.FunctionalCallback):
        def initialize(self, state, auxiliary):
            del auxiliary
            return (), state["integrator"]["v"][..., 0]

        def update(self, carry, state, auxiliary):
            del auxiliary
            return carry, state["integrator"]["v"]

        def finalize(self, carry, emissions):
            del carry
            return emissions

    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks({"invalid": WrongShape()})
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    binding = partial(
        functional.compile_rollout_chunk(3, backend="eager"),
        tensors.parameters,
        prepared,
    )
    callback_state = None
    if resume:
        _state, auxiliary = _run(
            "run",
            functional,
            binding,
            tensors.state,
            None,
            steps=0,
            callbacks=callbacks,
        )
        callback_state = auxiliary["callbacks"].state
    # Full-graph tracing wraps a deliberately raised FunctionalizationError
    # in Unsupported and retains the callback diagnostic in its context.
    with (
        torch_compiler_warning_context(),
        pytest.raises(
            torch._dynamo.exc.Unsupported,
            match="functional callback.*changed shape",
        ),
    ):
        _run(
            "longrun_checkpointed",
            functional,
            binding,
            tensors.state,
            None,
            steps=3,
            chunklength=5,
            callbacks=callbacks,
            callback_state=callback_state,
        )


@pytest.mark.parametrize("runner", RUNNERS)
def test_fused_callbacks_validate_resume_discovered_emission_shape(runner):
    class AuxiliaryEnergy(dn.func.FunctionalCallback):
        def initialize(self, state, auxiliary):
            del auxiliary
            return state["integrator"]["v"].new_zeros(()), None

        def update(self, carry, state, auxiliary):
            del state
            return carry, auxiliary["v"]

        def finalize(self, carry, emissions):
            return carry if emissions is None else emissions.square().sum()

    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks({"energy": AuxiliaryEnergy()})
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def narrow_auxiliary(state, inputs):
        next_state, auxiliary = functional.step(
            tensors.parameters, prepared, state, inputs
        )
        return next_state, {**auxiliary, "v": auxiliary["v"][..., :1]}

    # A pure adapter establishes a post-step schema from real auxiliary data.
    # The plan has no bound emission schema because initialize emits nothing.
    first, first_auxiliary = _run(
        "run",
        functional,
        narrow_auxiliary,
        tensors.state,
        None,
        steps=1,
        callbacks=callbacks,
    )
    callback_state = first_auxiliary["callbacks"].state
    assert callbacks._plans[0].emission_schema is None
    assert callback_state.emission_schemas[0].leaves[0].shape == (
        *tensors.state["integrator"]["v"].shape[:-1],
        1,
    )

    # An owned compiled transition exposes full-width voltage auxiliary.
    # Its output must be checked against the earlier segment's discovered
    # schema without changing callback configuration or mutating any tensor.
    kernel = functional.compile_rollout_chunk(3, backend="eager")
    with (
        torch_compiler_warning_context(),
        pytest.raises(
            dn.func.FunctionalizationError,
            match="emission changed a leaf's shape",
        ),
    ):
        _run(
            runner,
            functional,
            partial(kernel, tensors.parameters, prepared),
            first,
            None,
            steps=3,
            callbacks=callbacks,
            callback_state=callback_state,
        )
