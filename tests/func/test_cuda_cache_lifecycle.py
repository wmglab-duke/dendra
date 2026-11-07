"""Population lifecycle operations preserve existing real CUDA executables."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._scan_inductor import _ScanInductorBackend
from tests.func.test_cuda_population import (
    DT,
    STEPS,
    _clone,
    _close,
    _close_gradients,
    _functional_arguments,
    _functional_trajectory,
    _gradients,
    _loss,
    _model,
    _record_fields,
    _recorded_states,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


@pytest.mark.parametrize("training", [False, True], ids=["forward", "gradient"])
def test_cuda_population_lifecycle_preserves_compiled_functional_scan(
    monkeypatch, training
):
    model = _model("tree", dtype=torch.float32)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    # The lowered plan owns explicit inputs. Keep them independent of the
    # imperative source while that source is initialized again below.
    tensors = tensors._replace(
        parameters=_clone(tensors.parameters),
        constants=_clone(tensors.constants),
        state=_clone(tensors.state),
    )
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(_record_fields(tensors.state))}
    )
    kernel = functional.compile_rollout_chunk(
        STEPS, execution="scan", backend="inductor", fullgraph=True
    )

    def evaluate(compiled):
        with torch.set_grad_enabled(training):
            parameters, constants, state, ve, intra, targets = _functional_arguments(
                "tree", model, tensors
            )
            if compiled:
                prepared = functional.prepare(parameters, constants)
                final, auxiliary = dn.func.run(
                    kernel.bind(parameters, prepared),
                    state,
                    inputs=dn.func.RolloutInput(ve=ve, intra=intra),
                    steps=STEPS,
                    callbacks=callbacks,
                )
                recordings = dict(auxiliary["callbacks"]["trace"])
                loss = recordings["v"].square().mean()
            else:
                states = _functional_trajectory(
                    functional, parameters, constants, state, ve, intra
                )
                final, recordings, loss = (
                    states[-1],
                    _recorded_states(states),
                    _loss(states),
                )
            gradients = _gradients(loss, targets) if training else {}
        torch.cuda.synchronize()
        return final, recordings, gradients

    expected = evaluate(False)
    compilations = []
    original_backend_call = _ScanInductorBackend.__call__

    def counted_backend(backend, graph, inputs, **options):
        compilations.append(graph)
        # Execute the production Inductor backend, including its scan-local
        # decompositions and real CUDA forward/backward code generation.
        return original_backend_call(backend, graph, inputs, **options)

    monkeypatch.setattr(_ScanInductorBackend, "__call__", counted_backend)

    def check():
        with torch_compiler_warning_context():
            actual = evaluate(True)
        _close(actual[:2], expected[:2], dtype=torch.float32)
        if training:
            assert len(actual[2]) == 9
            _close_gradients(actual[2], expected[2], dtype=torch.float32)
        for group in ("integrator", "mechanisms"):
            for value in torch.utils._pytree.tree_leaves(actual[0][group]):
                assert value.dtype == torch.float32
                assert value.device.type == "cuda"
        assert kernel.execution_report().strategy == "compiled_scan"

    check()
    compiled_count = len(compilations)
    assert compiled_count > 0

    # Constructing a different model used to reset Dynamo's process-wide cache
    # and force this already-compiled functional graph through Inductor again.
    unrelated = _model("sc", dtype=torch.float32)
    check()
    assert len(compilations) == compiled_count

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cuda", DTYPE=torch.float32):
        model.initialize()
        unrelated.initialize()
        model.run(tstop=0, dt=DT)
        unrelated.run(tstop=0, dt=DT)
    check()
    assert len(compilations) == compiled_count
