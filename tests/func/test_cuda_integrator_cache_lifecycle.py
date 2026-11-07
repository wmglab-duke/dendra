"""Real Inductor integrators reuse code while rebuilding CUDA autograd inputs."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_cuda_population import _close, _close_gradients
from tests.func.test_scalar_tree import _morphology

DT = 0.025
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


def _model(kind):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cuda", DTYPE=torch.float32):
        if kind == "sc":
            model = dn.SingleCompartment(
                N=2,
                C=3,
                device="cuda",
                dtype=torch.float32,
                v_init=-62.0,
                integrator=dn.bwd_euler_sc(imem=False),
            )
        else:
            model = dn.Tree.from_morphology(
                _morphology(),
                N=2,
                device="cuda",
                dtype=torch.float32,
                v_init=-62.0,
                integrator=dn.dhs(threads=32, imem=False),
            )
        model.insert(pas, g=0.0002, e=-71.0)
        model.train()
        model.requires_grad_(False)
        model.cm_scale_param.rho.requires_grad_(True)
        model.initialize()
    return model


@pytest.mark.parametrize("kind", ["sc", "tree"])
def test_cuda_reinitialize_preserves_imperative_inductor_and_fresh_backward(
    monkeypatch, kind
):
    model, reference = _model(kind), _model(kind)
    compilations = []
    backend_type = torch._TorchCompileInductorWrapper
    original_backend_call = backend_type.__call__

    def counted_backend(backend, graph, inputs, **options):
        compilations.append(graph)
        return original_backend_call(backend, graph, inputs, **options)

    # Keep BACKEND="inductor" and its normal wrapper/options. Instrument only
    # entry into the production compiler, never replace it with graph.forward.
    monkeypatch.setattr(backend_type, "__call__", counted_backend)
    kernels = None
    compiled_count = None
    previous_voltage = None
    previous_workspaces = None

    def run(owner, *, jit):
        with (
            dn.ctx(
                JIT=jit,
                BACKEND="inductor",
                FULLGRAPH=True,
                REQUIRE_GRAD=1,
                DEVICE="cuda",
                DTYPE=torch.float32,
            ),
            torch_compiler_warning_context(),
        ):
            owner.initialize()
            owner.run(tstop=2 * DT, dt=DT)
            gradient = torch.autograd.grad(owner.v.sum(), owner.cm_scale_param.rho)[0]
        torch.cuda.synchronize()
        return owner.v.detach().clone(), gradient.detach().clone()

    for scale in (1.0, 1.5):
        model.cm_scale_param.set(scale)
        reference.cm_scale_param.set(scale)
        expected, expected_gradient = run(reference, jit=0)
        actual, gradient = run(model, jit=1)
        _close(actual, expected, dtype=torch.float32)
        _close_gradients(
            {"capacitance": gradient},
            {"capacitance": expected_gradient},
            dtype=torch.float32,
        )
        assert actual.dtype == gradient.dtype == torch.float32
        assert actual.device.type == gradient.device.type == "cuda"
        workspaces = tuple(
            getattr(model.integrator, name)
            for name, _role in model.integrator._prepared_workspace_schema()
        )
        if kernels is None:
            kernels = dict(model.integrator._compiled_kernels)
            compiled_count = len(compilations)
            assert kernels and compiled_count > 0
        else:
            assert model.integrator._compiled_kernels == kernels
            assert len(compilations) == compiled_count
            assert not torch.equal(actual, previous_voltage)
            assert all(
                current is not previous
                for current, previous in zip(
                    workspaces, previous_workspaces, strict=True
                )
            )
        previous_voltage = actual
        previous_workspaces = workspaces
