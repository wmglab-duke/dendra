"""CUDA-context regional support survives casting and functional lowering."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms._support import SupportKind
from tests.func.test_regional_hh import (
    DT,
    GNABAR_NAME,
    STEPS,
    _assert_every_leaf_close,
    _drives,
    _imperative_step,
    _initialize_imperative,
    _model,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_default_preserves_regional_support_through_cast_and_functional_steps(
    dtype,
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cuda", DTYPE=dtype):
        source = _model((slice(None), [1, 3]))
        reference = _model((slice(None), [1, 3]))
        for model in (source, reference):
            # This calls Mechanism._apply under the CUDA construction context.
            # A failed structural comparison would permanently invalidate key.
            model.to(device="cuda", dtype=dtype)
            model.initialize()
            model.train()
            assert model.mech.hh.support_spec.kind is SupportKind.SHARED_COLUMNS
            assert model.mech.hh._support_key_values_valid

        functional, tensors = dn.func.make_functional(source, dt=DT)
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        ve, intra = _drives(source)
        timestep = _initialize_imperative(reference)
        state = tensors.state
        rtol, atol = (5e-4, 3e-4) if dtype == torch.float32 else (3e-8, 3e-9)
        for index in range(STEPS):
            state, _ = functional.step(
                tensors.parameters,
                prepared,
                state,
                dn.func.StepInput(ve=ve[index], intra=intra[index]),
            )
            _imperative_step(reference, timestep, ve[index], intra[index])
            _assert_every_leaf_close(
                state,
                functional.extract(reference).state,
                rtol=rtol,
                atol=atol,
            )

        actual = torch.autograd.grad(
            state["integrator"]["v"].square().mean(),
            tensors.parameters[GNABAR_NAME],
        )[0]
        expected = torch.autograd.grad(
            reference.v.square().mean(), dict(reference.named_parameters())[GNABAR_NAME]
        )[0]
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(expected)
        relative_error = torch.linalg.vector_norm((actual - expected).double()) / (
            torch.linalg.vector_norm(expected.double())
        )
        assert relative_error <= (2e-3 if dtype == torch.float32 else 2e-7)
