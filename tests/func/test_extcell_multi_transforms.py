"""Transform contracts for the disconnected finite-extracellular forest."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.integrators.tree_bt import DENDRA_SOLVERS_AVAILABLE
from tests.func.test_extcell_multi_population import DT, _drives, _model

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="packed ExtCell CPU execution requires dendra-solvers",
    ),
]


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _response_case(batch_calls):
    model = _model(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    bath, intra = _drives(model, steps=1)
    scale_name = "populations.tree.rhoa_scale_param.rho"
    shell_name = "populations.tree.xg"

    def response(raw_rhoa_scale, shell_conductance, intracellular_drive):
        parameters = {**tensors.parameters, scale_name: raw_rhoa_scale}
        constants = {**tensors.constants, shell_name: shell_conductance}
        state, _ = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=bath[0], intra=intracellular_drive),
        )
        return state["integrator"]["v"], state["integrator"]["vc"]

    arguments = (
        tensors.parameters[scale_name],
        tensors.constants[shell_name],
        intra[0],
    )
    return model, response, arguments


@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_extcell_multi_jvp_vjp_duality_and_grad(cpu_threads, batch_calls):
    """Both voltage outputs compose through prepared physics and the block solve."""
    _, response, arguments = _response_case(batch_calls)
    directions = (
        torch.full_like(arguments[0], 0.2),
        arguments[1] * 0.15,
        torch.full_like(arguments[2], 2.0e-10),
    )
    values, tangents = torch.func.jvp(response, arguments, directions)
    reverse_values, pullback = torch.func.vjp(response, *arguments)
    for actual, expected in zip(values, reverse_values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    cotangents = tuple(
        torch.linspace(0.5, 1.5, value.numel(), dtype=value.dtype).reshape_as(value)
        for value in values
    )
    reverse_gradients = pullback(cotangents)
    forward_contraction = sum(
        (tangent * cotangent).sum()
        for tangent, cotangent in zip(tangents, cotangents, strict=True)
    )
    reverse_contraction = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(reverse_gradients, directions, strict=True)
    )
    assert torch.isfinite(forward_contraction)
    assert torch.count_nonzero(forward_contraction)
    torch.testing.assert_close(
        forward_contraction, reverse_contraction, rtol=2.0e-9, atol=2.0e-11
    )

    def loss(*inputs):
        return sum(
            (value * cotangent).sum()
            for value, cotangent in zip(response(*inputs), cotangents, strict=True)
        )

    actual_gradients = torch.func.grad(loss, argnums=(0, 1, 2))(*arguments)
    for actual, expected in zip(actual_gradients, reverse_gradients, strict=True):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-11)


@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_extcell_multi_hidden_vmap_lanes_and_empty_lanes(cpu_threads, batch_calls):
    """Mapped physics stays independent of core replicas and explicit batches."""
    model, response, arguments = _response_case(batch_calls)
    scales = arguments[0] + arguments[0].new_tensor((-0.1, 0.0, 0.15))
    shells = torch.stack((arguments[1] * 0.9, arguments[1], arguments[1] * 1.2))
    drives = torch.stack((arguments[2] - 1.0e-10, arguments[2], arguments[2] + 2.0e-10))
    actual = torch.vmap(response)(scales, shells, drives)
    individual = tuple(
        response(scale, shell, drive)
        for scale, shell, drive in zip(scales, shells, drives, strict=True)
    )
    expected = tuple(
        torch.stack([values[index] for values in individual]) for index in range(2)
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=2.0e-10, atol=2.0e-11)

    empty = torch.vmap(response)(scales[:0], shells[:0], drives[:0])
    assert empty[0].shape == (0, *model.shape)
    assert empty[1].shape == (0, *model.shape, 3)
