"""Boundary regimes for joint compiled simulation and callback chunks."""

from functools import partial

import pytest
import torch
from test_runner_compiled_callbacks import _assert_result_close, _callbacks, _evaluate
from test_runner_compiled_chunks import _run
from test_runner_stimulation_clock import STEPS as PULSE_STEPS
from test_runner_stimulation_clock import _case as _pulse_case
from test_runners import DT, GNABAR, _clone_mapping, _clone_state, _drives, _model
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


def test_single_step_chunks_compile_callback_updates_and_preserve_gradients():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 5)
    callbacks = _callbacks(functional, model, 5)
    expected = _evaluate(functional, tensors, ve, intra, callbacks, "run")
    widths = []

    def backend(graph, _example_inputs):
        widths.append(
            sum(
                node.op == "call_function" and node.target is F.mse_loss
                for node in graph.graph.nodes
            )
        )
        return graph.forward

    kernel = functional.compile_rollout_chunk(1, backend=backend)
    with torch_compiler_warning_context():
        actual = _evaluate(
            functional, tensors, ve, intra, callbacks, "longrun_checkpointed", kernel
        )

    _assert_result_close(actual, expected)
    assert widths and set(widths) == {1}
    assert actual[1].state["mse"]["frames"].item() == 6


def test_callback_chunk_dual_fallback_preserves_forward_ad_through_tails():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 7)
    callbacks = _callbacks(functional, model, 7)

    def unexpected_backend(_graph, _example_inputs):
        raise AssertionError("direct forward AD must execute the transform-safe kernel")

    kernel = functional.compile_rollout_chunk(3, backend=unexpected_backend)
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        parameters = _clone_mapping(tensors.parameters)
        constants = _clone_mapping(tensors.constants)
        state = _clone_state(tensors.state)
        for values, name in (
            (parameters, GNABAR),
            (constants, "diam"),
            (state["integrator"], "v"),
        ):
            primal = values[name]
            values[name] = torch.autograd.forward_ad.make_dual(
                primal, torch.full_like(primal, 0.125)
            )
        inputs = dn.func.RolloutInput(
            torch.autograd.forward_ad.make_dual(ve, torch.full_like(ve, 0.125)),
            torch.autograd.forward_ad.make_dual(intra, torch.full_like(intra, 1.0e-11)),
        )
        prepared = functional.prepare(parameters, constants)
        expected, expected_aux = _run(
            "run",
            functional,
            partial(functional.step, parameters, prepared),
            state,
            inputs,
            steps=7,
            callbacks=callbacks,
        )
        actual, actual_aux = _run(
            "longrun_checkpointed",
            functional,
            partial(kernel, parameters, prepared),
            state,
            inputs,
            steps=7,
            chunklength=5,
            callbacks=callbacks,
        )
        expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(
            (expected, expected_aux["callbacks"])
        )
        actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(
            (actual, actual_aux["callbacks"])
        )
        assert actual_spec == expected_spec
        for leaf, reference in zip(actual_leaves, expected_leaves, strict=True):
            primal, tangent = torch.autograd.forward_ad.unpack_dual(leaf)
            expected_primal, expected_tangent = torch.autograd.forward_ad.unpack_dual(
                reference
            )
            torch.testing.assert_close(
                primal, expected_primal, rtol=2.0e-10, atol=2.0e-11
            )
            if expected_tangent is None:
                assert tangent is None
            else:
                assert tangent is not None
                torch.testing.assert_close(
                    tangent, expected_tangent, rtol=2.0e-8, atol=2.0e-10
                )
        loss_tangent = torch.autograd.forward_ad.unpack_dual(
            actual_aux["callbacks"]["mse"]
        ).tangent
        assert loss_tangent is not None
        assert torch.count_nonzero(loss_tangent) > 0


@pytest.mark.parametrize("stimulus_kind", ["intra", "extra"])
def test_joint_chunks_preserve_narrow_bound_pulse_trace_and_amplitude_gradient(
    stimulus_kind,
):
    functional, tensors, parameters, prepared, amplitude = _pulse_case(stimulus_kind)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    expected, expected_aux = _run(
        "run",
        functional,
        partial(functional.step, parameters, prepared),
        tensors.state,
        None,
        steps=PULSE_STEPS,
        callbacks=callbacks,
    )
    expected_trace = expected_aux["callbacks"]["trace"]
    expected_gradient = torch.autograd.grad(
        expected_trace["v"].square().mean(), amplitude
    )[0]
    assert torch.isfinite(expected_gradient).all()
    assert expected_gradient.abs().item() > 0.0

    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")
    with torch_compiler_warning_context():
        actual, actual_aux = _run(
            "longrun_checkpointed",
            functional,
            partial(kernel, parameters, prepared),
            tensors.state,
            None,
            steps=PULSE_STEPS,
            chunklength=5,
            callbacks=callbacks,
        )
        actual_trace = actual_aux["callbacks"]["trace"]
        actual_gradient = torch.autograd.grad(
            actual_trace["v"].square().mean(), amplitude
        )[0]

    assert actual_trace["v"].shape[0] == PULSE_STEPS + 1
    torch.testing.assert_close(actual["integrator"]["v"], expected["integrator"]["v"])
    torch.testing.assert_close(actual_trace["v"], expected_trace["v"])
    torch.testing.assert_close(
        actual_trace["t"], expected_trace["t"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(actual_gradient, expected_gradient)
