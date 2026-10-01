"""Independent root VJPs compose without retaining simulator graphs."""

import math

import pytest
import torch

from dendra.models.analysis.threshold_transition_jets import (
    gaussian_expected_hard_from_root_parameter_jets,
)


def test_signed_two_root_jets_match_analytic_parameter_and_center_slopes():
    dtype = torch.float64
    p = torch.tensor([0.2, -0.3], dtype=dtype, requires_grad=True)
    center = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    roots = torch.tensor([0.4, 1.1], dtype=dtype)
    jumps = torch.tensor([2.0, -1.0], dtype=dtype)
    dr_dp = torch.tensor([[0.3, -0.1], [-0.2, 0.4]], dtype=dtype)
    sigma = 0.25
    result = gaussian_expected_hard_from_root_parameter_jets(
        center,
        roots,
        jumps,
        1.0,
        sigma,
        parameters=(p,),
        root_parameter_gradients=((dr_dp[0],), (dr_dp[1],)),
    )
    grad_p, grad_center = torch.autograd.grad(result, (p, center))
    standardized = (center.detach() - roots) / sigma
    normal_pdf = torch.exp(-0.5 * standardized.square()) / math.sqrt(2 * math.pi)
    weighted = -jumps * normal_pdf / sigma
    assert torch.allclose(grad_p, weighted @ dr_dp, rtol=1e-12, atol=1e-12)
    assert torch.allclose(grad_center, (-weighted).sum(), rtol=1e-12, atol=1e-12)


def test_no_transition_has_zero_but_connected_training_gradient():
    p = torch.tensor([0.2, 0.3], dtype=torch.float64, requires_grad=True)
    result = gaussian_expected_hard_from_root_parameter_jets(
        0.5,
        torch.empty(0, dtype=p.dtype),
        torch.empty(0, dtype=p.dtype),
        0.75,
        0.1,
        parameters=(p,),
        root_parameter_gradients=(),
    )
    assert result.item() == pytest.approx(0.75)
    (gradient,) = torch.autograd.grad(result, p)
    assert torch.equal(gradient, torch.zeros_like(p))


def test_rejects_missing_or_mismatched_root_vjp():
    p = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    roots = torch.tensor([0.4], dtype=torch.float64)
    jumps = torch.ones_like(roots)
    kwargs = dict(parameters=(p,))
    with pytest.raises(ValueError, match="one parameter-gradient row"):
        gaussian_expected_hard_from_root_parameter_jets(
            0.5,
            roots,
            jumps,
            0.0,
            0.1,
            root_parameter_gradients=(),
            **kwargs,
        )
    with pytest.raises(ValueError, match="finite and match"):
        gaussian_expected_hard_from_root_parameter_jets(
            0.5,
            roots,
            jumps,
            0.0,
            0.1,
            root_parameter_gradients=((torch.tensor([float("nan")], dtype=p.dtype),),),
            **kwargs,
        )
