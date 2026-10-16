"""Real CPU code-generation checks for the convenience boundary and native DHS."""

import pytest
import torch
from test_recorder_while_loop import DT, _case, _clone, _close

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE, reason="native DHS requires dendra-solvers"
    ),
    pytest.mark.skipif(
        torch.__version__.split("+")[0] not in ("2.14.0", "2.14.1"),
        reason="functional scan is audited for PyTorch 2.14.0 and 2.14.1",
    ),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _reference(runner, state, *, parameters=None, inputs=None):
    selected = runner.parameters if parameters is None else parameters
    prepared = runner.functional.prepare(selected, runner.constants)
    callbacks = runner.functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("v", "hh.m", "hh.h", "hh.n", "t")),
        }
    )
    return dn.func.run(
        runner.functional.bind(selected, prepared),
        state,
        steps=8,
        callbacks=callbacks,
        inputs=inputs,
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_cpp_inference_runner_matches_full_native_dhs_traces(execution):
    case = _case(stimulus_kind="intra")
    source_v = case.model.v.detach().clone()
    runner = dn.func.compile(
        case.model,
        dt=DT,
        steps=8,
        execution=execution,
        callbacks={"trace": dn.func.Recorder(("v", "hh.m", "hh.h", "hh.n", "t"))},
    )
    assert runner.kernel._compile_options["options"]["cpp_wrapper"] is True
    with torch.no_grad():
        reference = _reference(runner, _clone(runner.initial_state))
    for _ in range(2):
        final, auxiliary = runner()
        _close(final, reference[0])
        _close(dict(auxiliary["callbacks"]), dict(reference[1]["callbacks"]))
    torch.testing.assert_close(case.model.v, source_v, rtol=0, atol=0)
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )


def test_inductor_scan_native_dhs_gradients_and_repeated_backward():
    case = _case(stimulus_kind="intra")
    runner = dn.func.compile(
        case.model,
        dt=DT,
        steps=8,
        inference=False,
        trainable=("gnabar", "amp"),
        callbacks={"trace": dn.func.Recorder(("v", "hh.m", "hh.h", "hh.n", "t"))},
    )
    trainable_names = [
        name for name, value in runner.parameters.items() if value.requires_grad
    ]
    assert len(trainable_names) == 2

    def independent():
        parameters = {
            name: value.detach().clone() for name, value in runner.parameters.items()
        }
        for name in trainable_names:
            parameters[name].requires_grad_()
        state = _clone(runner.initial_state)
        state["integrator"]["v"].requires_grad_()
        ve = torch.linspace(
            -0.1, 0.1, 8 * case.model.v.numel(), dtype=case.model.dtype()
        )
        ve = ve.reshape(8, *case.model.shape).requires_grad_()
        inputs = dn.func.RolloutInput(ve=ve)
        targets = [parameters[name] for name in trainable_names] + [
            state["integrator"]["v"],
            ve,
        ]
        return parameters, state, inputs, targets

    def loss(result):
        final, auxiliary = result
        return (
            final["integrator"]["v"].square().mean()
            + 0.01 * auxiliary["callbacks"]["trace"]["v"].square().mean()
        )

    parameters, state, inputs, targets = independent()
    reference = _reference(runner, state, parameters=parameters, inputs=inputs)
    reference_gradients = torch.autograd.grad(loss(reference), targets)
    for _ in range(2):
        parameters, state, inputs, targets = independent()
        actual = runner(state, parameters=parameters, inputs=inputs)
        _close(actual[0], reference[0])
        _close(dict(actual[1]["callbacks"]), dict(reference[1]["callbacks"]))
        for gradient, expected in zip(
            torch.autograd.grad(loss(actual), targets), reference_gradients, strict=True
        ):
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0
            torch.testing.assert_close(gradient, expected, rtol=1e-7, atol=1e-8)

    # The default trainable leaves also support independent successive graphs.
    for _ in range(2):
        actual = runner()
        gradients = torch.autograd.grad(
            loss(actual), [runner.parameters[name] for name in trainable_names]
        )
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
