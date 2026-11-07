"""Independent functional executors own their Dynamo specialization budgets."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas

DT = 0.025
STEPS = 4
pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


def _model(index):
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float32):
        model = dn.SingleCompartment(
            N=1,
            C=2,
            v_init=-62.0 + 0.1 * index,
            device="cpu",
            dtype=torch.float32,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(pas, g=0.0002, e=-71.0)
        model.requires_grad_(False)
        model.eval()
        model.initialize()
    return model


@pytest.mark.parametrize(
    "execution,record",
    [
        ("unrolled", False),
        ("default", False),
        ("scan", True),
        ("unrolled", True),
        ("default", True),
    ],
    ids=["unrolled", "default", "callback_scan", "callback_unrolled", "callback_while"],
)
def test_independent_functional_plans_do_not_share_recompilation_budget(
    execution, record
):
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    plans = []
    # Exceed the actual configured budget without changing compiler policy or
    # resetting any cache. Retain every plan and revisit them afterwards.
    count = max(10, int(torch._dynamo.config.recompile_limit) + 2)
    with torch.no_grad(), torch_compiler_warning_context():
        for index in range(count):
            model = _model(index)
            functional, tensors = dn.func.make_functional(model, dt=DT)
            prepared = functional.prepare(tensors.parameters, tensors.constants)
            callbacks = (
                functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
                if record
                else None
            )
            expected, expected_auxiliary = functional.rollout(
                tensors.parameters,
                prepared,
                tensors.state,
                steps=STEPS,
                callbacks=callbacks,
            )
            kernel = functional.compile_rollout_chunk(
                STEPS, execution=execution, backend=backend, fullgraph=True
            )
            bound = kernel.bind(tensors.parameters, prepared)

            def check(
                bound=bound,
                tensors=tensors,
                callbacks=callbacks,
                expected=expected,
                expected_auxiliary=expected_auxiliary,
                kernel=kernel,
            ):
                actual, auxiliary = dn.func.run(
                    bound, tensors.state, steps=STEPS, callbacks=callbacks
                )
                torch.testing.assert_close(
                    actual["integrator"]["v"],
                    expected["integrator"]["v"],
                    rtol=0,
                    atol=0,
                )
                assert actual["integrator"]["v"].dtype == torch.float32
                if record:
                    torch.testing.assert_close(
                        auxiliary["callbacks"]["trace"]["v"],
                        expected_auxiliary["callbacks"]["trace"]["v"],
                        rtol=0,
                        atol=0,
                    )
                if record and execution == "default":
                    assert kernel.execution_report().strategy == "compiled_while_loop"

            check()
            # Keep the source alive too: source lifetime is independent of the
            # compiled execution-plan cache being exercised here.
            plans.append((model, check))
        compiled_count = len(graphs)
        assert compiled_count >= count
        for _model_source, check in plans:
            check()
        assert len(graphs) == compiled_count
