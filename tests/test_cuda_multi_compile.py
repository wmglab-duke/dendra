"""Fullgraph parity for the packed heterogeneous multi-tree GPU step."""

from __future__ import annotations

import pytest
import torch

from tests.test_tree_multi_reference_matrix import (
    LinearMechanism,
    _make_multi,
    _sample_inputs,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


class _CompiledLinearMechanism(LinearMechanism):
    def advance(self, voltage, dt, temp):
        pass


def _multi_cuda_fullgraph_outputs(*, threads, jit, repeats):
    model, mechanism, integrator = _make_multi(batch=2, threads=threads)
    integrator.mech = _CompiledLinearMechanism(
        mechanism.conductance.detach().clone(),
        mechanism.reversal.detach().clone(),
    )
    model.cuda()
    integrator.cuda()
    model.jit = bool(jit)
    model.backend = "inductor"
    model.fullgraph = True
    model.dynamic = False
    model.compile_mode = None
    model.compile_options = None
    dt = 0.043
    integrator._initialize(model, dt)
    voltage, intra, ve = (value.cuda() for value in _sample_inputs(model))

    outputs = []
    with torch.no_grad():
        for _ in range(repeats):
            value, _ = integrator._kernel(
                "_step",
                voltage,
                dt,
                model.celsius,
                ve,
                intra,
            )
            outputs.append(value.cpu())
    return outputs, bool(integrator._compiled_kernels)


@pytest.mark.parametrize("threads", [1, 32])
def test_multi_tree_cuda_fullgraph_matches_eager_and_repeats(threads):
    (eager,), eager_compiled = _multi_cuda_fullgraph_outputs(
        threads=threads, jit=False, repeats=1
    )
    (first, second), compiled = _multi_cuda_fullgraph_outputs(
        threads=threads, jit=True, repeats=2
    )

    assert eager_compiled is False
    assert compiled is True
    assert torch.equal(first, second)
    torch.testing.assert_close(first, eager, rtol=3.0e-11, atol=3.0e-11)
