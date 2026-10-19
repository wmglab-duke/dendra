"""Normal public scan schedules preserve full HH/DHS traces and all eight AD targets.

Every compiled invocation remains a public, guarded boundary. These tests do
not wrap runners in another torch.compile, detach resumed state, compile an
unrolled horizon, or use retain_graph=True to inhibit donation implicitly.
"""

from types import SimpleNamespace

import pytest
import torch
from test_scan_hh_validation import (
    DT,
    _assert_close,
    _case,
    _clone,
    _FrameDiagnostics,
    _operands,
)

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.test_hh_rate_configuration import GATES

pytestmark = pytest.mark.cpu
_STEPS = 8
_TARGETS = (
    "initial_voltage",
    "gnabar",
    "gkbar",
    "gl",
    "ena",
    "ek",
    "el",
    "stimulus",
)


@pytest.fixture
def workflow_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _evaluate_workflow(case, route):
    parameters, constants, state, inputs, targets = _operands(case, _STEPS)
    assert tuple(targets) == _TARGETS
    for name in _TARGETS[1:-1]:
        assert (
            parameters[
                f"integrator.mech.mechanisms.{case.mechanism_name}.{name}_param"
            ].ndim
            == 0
        )
    original = _clone((parameters, constants, state, inputs))
    versions = tuple(
        leaf._version
        for leaf in torch.utils._pytree.tree_leaves(
            (parameters, constants, state, inputs)
        )
        if torch.is_tensor(leaf)
    )
    with dn.ctx(MEMBRANE_ASSEMBLY="off" if case.legacy else "auto"):
        prepared = case.functional.prepare(parameters, constants)
        if route == "eager":
            final, auxiliary = dn.func.run(
                case.functional.bind(parameters, prepared),
                state,
                steps=_STEPS,
                inputs=inputs,
                callbacks=case.callbacks,
            )
        elif route in ("resumed", "compiled_checkpointed"):
            runner = dn.func.compile(
                case.model,
                dt=DT,
                steps=4 if route == "resumed" else _STEPS,
                inference=False,
                checkpoint_every=3 if route == "compiled_checkpointed" else None,
                callbacks={
                    "trace": dn.func.Recorder(
                        ["v", *(f"{case.mechanism_name}.{gate}" for gate in GATES), "t"]
                    ),
                    "diagnostics": _FrameDiagnostics(),
                },
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                options={"cpp_wrapper": True, "compile_threads": 1},
            )
            if route == "compiled_checkpointed":
                final, auxiliary = runner(state, parameters=parameters, inputs=inputs)
            else:
                middle, first = runner(
                    state,
                    parameters=parameters,
                    inputs=dn.func.RolloutInput(intra=inputs.intra[:4]),
                )
                final, auxiliary = runner(
                    middle,
                    parameters=parameters,
                    inputs=dn.func.RolloutInput(intra=inputs.intra[4:]),
                    callback_state=first["callbacks"].state,
                )
                prefix, suffix = (
                    first["callbacks"]["trace"],
                    auxiliary["callbacks"]["trace"],
                )
                recorded = {
                    name: torch.cat((prefix[name], suffix[name]), dim=0)
                    for name in prefix
                }
            assert runner.execution_report().strategy == "compiled_scan"
        else:
            kernel = case.functional.compile_rollout_chunk(
                8 if route in ("whole", "host_spans", "checkpointed") else 4,
                execution="scan",
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                options={"cpp_wrapper": True, "compile_threads": 1},
            )
            options = (
                {"checkpoint_every": 3}
                if route == "checkpointed"
                else {"host_span_steps": 3}
                if route == "host_spans"
                else {}
            )
            final, auxiliary = dn.func.run(
                kernel.bind(parameters, prepared),
                state,
                steps=_STEPS,
                inputs=inputs,
                callbacks=case.callbacks,
                **options,
            )
            assert kernel.execution_report().strategy == "compiled_scan"
        if route != "resumed":
            recorded = auxiliary["callbacks"]["trace"]
        diagnostics = auxiliary["callbacks"]["diagnostics"]
        trace = {"v": recorded["v"], "t": recorded["t"]}
        trace.update(
            {f"hh.{gate}": recorded[f"{case.mechanism_name}.{gate}"] for gate in GATES}
        )
        assert trace["v"].shape == (_STEPS + 1, *case.model.shape)
        for gate in GATES:
            assert trace[f"hh.{gate}"].shape == (
                _STEPS + 1,
                *state["mechanisms"][case.mechanism_name][gate].shape,
            )
        assert diagnostics["frames"].item() == _STEPS + 1
        loss = (
            trace["v"].square().mean() / 1000
            + sum(trace[f"hh.{gate}"].square().mean() for gate in GATES) / 10
            + diagnostics["energy"] / diagnostics["frames"] / 10
        )
        # Primary gradient checks must not retain the AOT graph: retained-graph
        # donation policy would otherwise mask the defect this suite targets.
        gradients = dict(
            zip(
                targets, torch.autograd.grad(loss, tuple(targets.values())), strict=True
            )
        )
    current_leaves = [
        leaf
        for leaf in torch.utils._pytree.tree_leaves(
            (parameters, constants, state, inputs)
        )
        if torch.is_tensor(leaf)
    ]
    for current, saved, version in zip(
        current_leaves,
        [
            leaf
            for leaf in torch.utils._pytree.tree_leaves(original)
            if torch.is_tensor(leaf)
        ],
        versions,
        strict=True,
    ):
        torch.testing.assert_close(current, saved, rtol=0, atol=0)
        assert current._version == version
    final_values = {
        "v": final["integrator"]["v"],
        "t": final["clock"]["t"],
        "duration_remainder": final["control"]["duration_remainder"],
        **{
            f"hh.{gate}": final["mechanisms"][case.mechanism_name][gate]
            for gate in GATES
        },
    }
    assert not final_values["duration_remainder"].requires_grad
    return SimpleNamespace(
        trace=_clone(trace),
        diagnostics=_clone(diagnostics),
        loss=loss.detach(),
        gradients=_clone(gradients),
        final=_clone(final_values),
    )


# Each scheduling branch runs; the whole-rollout baseline and public compiled
# checkpointing also run at both thread counts.
@pytest.mark.parametrize(
    ("route", "workflow_threads"),
    [
        pytest.param("whole", 1, id="t1-whole"),
        pytest.param("whole", 4, id="t4-whole"),
        pytest.param("chunks", 4, id="t4-chunks"),
        pytest.param("host_spans", 1, id="t1-host_spans"),
        pytest.param("resumed", 4, id="t4-resumed"),
        pytest.param("checkpointed", 1, id="t1-checkpointed"),
        pytest.param("compiled_checkpointed", 1, id="t1-compiled_checkpointed"),
        pytest.param("compiled_checkpointed", 4, id="t4-compiled_checkpointed"),
    ],
    indirect=["workflow_threads"],
)
@pytest.mark.slow
def test_public_cpp_scan_schedules_preserve_all_eight_gradients(
    workflow_threads, route
):
    assert torch.get_num_threads() == workflow_threads
    with torch_compiler_warning_context():
        expected = _evaluate_workflow(
            _case("dhs", torch.float64, 2, legacy=True), "eager"
        )
        actual = _evaluate_workflow(_case("dhs", torch.float64, 2), route)
    assert tuple(actual.gradients) == _TARGETS
    _assert_close(actual, expected, torch.float64)
    torch.testing.assert_close(
        actual.diagnostics, expected.diagnostics, rtol=2e-10, atol=2e-9
    )
    torch.testing.assert_close(actual.final, expected.final, rtol=2e-10, atol=2e-9)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-10, atol=2e-9)
